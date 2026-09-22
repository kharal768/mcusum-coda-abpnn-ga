"""Classifiers for attributing signals to shifted ilr coordinates.

Networks (numpy): one sigmoid hidden layer, one sigmoid output node per ilr coordinate
(target 1 if that coordinate is shifted), sum-of-squared-errors loss.
  BP        : mini-batch gradient descent, fixed learning rate, random initial weights
  ABPNN     : BP with momentum and an adaptive (bold-driver) learning rate
  BP-GA     : BP started from GA-optimized initial weights
  ABPNN-GA  : ABPNN started from GA-optimized initial weights
All networks stop early on validation loss and keep the best-validation weights.
"""
import numpy as np

GA_SETTINGS = dict(pop=40, generations=40, pc=0.8, pm=0.1, bmin=-3.0, bmax=3.0,
                   c=1.0, fitness_subsample=3000, elite=2)
BP_SETTINGS = dict(lr=0.5, momentum=0.0, adaptive=False, batch=256, max_epochs=300, patience=25)
ABP_SETTINGS = dict(lr=0.5, momentum=0.9, adaptive=True, batch=256, max_epochs=300, patience=25,
                    lr_up=1.05, lr_down=0.5)


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


class Net:
    def __init__(self, n_in, n_hid, n_out):
        self.shape = (n_in, n_hid, n_out)
        self.n_par = n_in * n_hid + n_hid + n_hid * n_out + n_out

    def unpack(self, w):
        a, b, c = self.shape; i = 0
        W1 = w[i:i + a * b].reshape(a, b); i += a * b
        b1 = w[i:i + b]; i += b
        W2 = w[i:i + b * c].reshape(b, c); i += b * c
        b2 = w[i:i + c]
        return W1, b1, W2, b2

    def forward(self, w, X):
        W1, b1, W2, b2 = self.unpack(w)
        H = sig(X @ W1 + b1)
        return H, sig(H @ W2 + b2)

    def sse(self, w, X, Y):
        return 0.5 * np.sum((self.forward(w, X)[1] - Y) ** 2) / len(X)

    def grad(self, w, X, Y):
        W1, b1, W2, b2 = self.unpack(w)
        H, O = self.forward(w, X)
        dO = (O - Y) * O * (1 - O) / len(X)
        dH = (dO @ W2.T) * H * (1 - H)
        return np.concatenate([(X.T @ dH).ravel(), dH.sum(0), (H.T @ dO).ravel(), dO.sum(0)])


def ga_init(net, X, Y, rng, s=GA_SETTINGS):
    """Real-coded GA over the full weight vector; returns the best individual."""
    n = min(len(X), s["fitness_subsample"]); sub = rng.choice(len(X), n, replace=False)
    Xs, Ys = X[sub], Y[sub]
    P = rng.uniform(-1, 1, (s["pop"], net.n_par))
    fit = lambda w: 1.0 / (s["c"] * 2 * net.sse(w, Xs, Ys) * n)   # 1 / (c * sum of squared errors)
    F = np.array([fit(w) for w in P])
    for g in range(s["generations"]):
        order = np.argsort(-F); elite = P[order[:s["elite"]]].copy()
        prob = F / F.sum()                                           # roulette-wheel selection
        parents = P[rng.choice(len(P), size=len(P), p=prob)]
        kids = parents.copy()
        for i in range(0, len(P) - 1, 2):                            # arithmetic crossover
            if rng.random() < s["pc"]:
                a = rng.random()
                kids[i] = a * parents[i] + (1 - a) * parents[i + 1]
                kids[i + 1] = (1 - a) * parents[i] + a * parents[i + 1]
        mask = rng.random(kids.shape) < s["pm"]                      # non-uniform mutation
        r = rng.random(kids.shape); fg = r * (1 - g / s["generations"])
        up = kids + (s["bmax"] - kids) * fg
        down = kids - (kids - s["bmin"]) * fg
        kids = np.where(mask, np.where(rng.random(kids.shape) > 0.5, up, down), kids)
        kids[:s["elite"]] = elite
        P = kids; F = np.array([fit(w) for w in P])
    return P[np.argmax(F)]


def train_bp(net, w, X, Y, Xv, Yv, rng, s):
    vel = np.zeros_like(w); lr = s["lr"]
    best, best_w, wait, prev = np.inf, w.copy(), 0, np.inf
    epochs = 0
    for ep in range(s["max_epochs"]):
        epochs = ep + 1
        perm = rng.permutation(len(X))
        for i in range(0, len(X), s["batch"]):
            b = perm[i:i + s["batch"]]
            vel = s["momentum"] * vel - lr * net.grad(w, X[b], Y[b])
            w = w + vel
        v = net.sse(w, Xv, Yv)
        if s["adaptive"]:
            lr = lr * (s["lr_up"] if v < prev else s["lr_down"]); prev = v
        if v < best - 1e-7:
            best, best_w, wait = v, w.copy(), 0
        else:
            wait += 1
            if wait >= s["patience"]:
                break
    return best_w, epochs


class NetClassifier:
    """Multi-label sigmoid network decoded to one shift pattern per signal."""

    def __init__(self, patterns, n_hid, variant, seed):
        self.P = np.array(patterns); self.n_hid = n_hid; self.variant = variant
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y, Xv, yv):
        self.m, self.s = X.mean(0), X.std(0) + 1e-9
        Xn, Xvn = (X - self.m) / self.s, (Xv - self.m) / self.s
        Y, Yv = self.P[y].astype(float), self.P[yv].astype(float)
        self.net = Net(X.shape[1], self.n_hid, self.P.shape[1])
        if self.variant.endswith("GA"):
            w0 = ga_init(self.net, Xn, Y, self.rng)
        else:
            w0 = self.rng.uniform(-1, 1, self.net.n_par)
        cfg = ABP_SETTINGS if self.variant.startswith("ABPNN") else BP_SETTINGS
        self.w, self.epochs = train_bp(self.net, w0, Xn, Y, Xvn, Yv, self.rng, cfg)
        return self

    def predict(self, X):
        O = self.net.forward(self.w, (X - self.m) / self.s)[1]
        B = (O > 0.5).astype(int)
        none = B.sum(1) == 0
        B[none, np.argmax(O[none], 1)] = 1
        # map binary vector to pattern index; nearest pattern (Hamming) if needed
        dist = np.abs(B[:, None, :] - self.P[None, :, :]).sum(2)
        return np.argmin(dist, 1)
