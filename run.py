"""Full experiment: calibrate charts, simulate signals, train classifiers, evaluate.

Usage: python run.py [--runs 5] [--quick]
Writes results/results.json and results/h_calibration.json.
"""
import argparse, json, os, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from simulate import CASES, DELTAS, patterns, calibrate_h, build_dataset
from models import NetClassifier, GA_SETTINGS, BP_SETTINGS, ABP_SETTINGS

ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=int, default=5)
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
N_TRAIN, N_VAL, N_TEST = (100, 50, 200) if args.quick else (500, 150, 1000)
HIDDEN_GRID = [8, 16, 24]
SVM_MAX_TRAIN = 10000
FEATURES = {"obs": ["obs"], "u": ["u"], "u+runmean": ["u", "runmean"]}
os.makedirs("results", exist_ok=True)


def feats(D, key):
    return np.hstack([D[k] for k in FEATURES[key]])


def confusions(y, yhat, delta, n_cls):
    out = {}
    for dl in DELTAS:
        m = delta == dl
        C = np.zeros((n_cls, n_cls), int)
        np.add.at(C, (y[m], yhat[m]), 1)
        out[str(dl)] = C.tolist()
    return out


def evaluate(model, Xtr, ytr, Xv, yv, Xte, D_te, n_cls, is_net):
    t0 = time.perf_counter()
    if is_net:
        model.fit(Xtr, ytr, Xv, yv)
    else:
        model.fit(Xtr, ytr)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter(); yhat = model.predict(Xte); t_pred = time.perf_counter() - t0
    val_acc = float(np.mean(model.predict(Xv) == yv))
    return dict(conf=confusions(D_te["y"], yhat, D_te["delta"], n_cls), val_acc=val_acc,
                fit_seconds=t_fit, predict_seconds_per_1000=1000 * t_pred / len(Xte),
                epochs=getattr(model, "epochs", None))


# 1. control limits (reused if already computed)
hcal = json.load(open("results/h_calibration.json")) if os.path.exists("results/h_calibration.json") else {}
for case in [c for c in CASES if c not in hcal]:
    d = len(CASES[case]["mu0"])
    h, a, se = calibrate_h(d, n=4000 if args.quick else 20000)
    hcal[case] = dict(dim=d, h=h, arl0=a, arl0_se=se)
    print("h", case, hcal[case], flush=True)
json.dump(hcal, open("results/h_calibration.json", "w"), indent=1)

prev = json.load(open("results/results.json")) if os.path.exists("results/results.json") else None
results = dict(settings=dict(N_TRAIN=N_TRAIN, N_VAL=N_VAL, N_TEST=N_TEST, deltas=DELTAS,
                             GA=GA_SETTINGS, BP=BP_SETTINGS, ABPNN=ABP_SETTINGS,
                             hidden_grid=HIDDEN_GRID, svm_max_train=SVM_MAX_TRAIN),
               h=hcal, cases={})
if prev is not None:
    results["cases"] = prev["cases"]

for case in CASES:
    d = len(CASES[case]["mu0"]); pats = patterns(d); n_cls = len(pats); h = hcal[case]["h"]
    R = results["cases"].setdefault(case, dict(patterns=[list(p) for p in pats], runs=[]))
    choice = R.get("selection", {})
    for run in range(len(R["runs"]), args.runs):
        base = 1000 * (run + 1) + (0 if case == "bronze" else 500)
        rec = dict(run=run, models={}, t2={}, feature_ablation={})
        data = {}
        for chart in ["mcusum", "t2"]:
            data[chart] = [build_dataset(case, chart, h, n, base + 10 * j + (5 if chart == "t2" else 0))
                           for j, n in enumerate([N_TRAIN, N_VAL, N_TEST])]
        tr, va, te = data["mcusum"]
        rec["arl1"] = {str(dl): [float(te["rl"][(te["delta"] == dl) & (te["y"] == c)].mean())
                                 for c in range(n_cls)] for dl in DELTAS}
        # model selection on the validation set of run 0
        if run == 0:
            best = None
            for nh in HIDDEN_GRID:
                for fk in FEATURES:
                    m = NetClassifier(pats, nh, "ABPNN-GA", seed=base + nh)
                    m.fit(feats(tr, fk), tr["y"], feats(va, fk), va["y"])
                    acc = float(np.mean(m.predict(feats(va, fk)) == va["y"]))
                    print(case, "select", nh, fk, round(acc, 4), flush=True)
                    if best is None or acc > best[0]:
                        best = (acc, nh, fk)
            choice = dict(val_acc=best[0], n_hidden=best[1], features=best[2])
            R["selection"] = choice
        nh, fk = choice["n_hidden"], choice["features"]
        Xtr, Xv, Xte = feats(tr, fk), feats(va, fk), feats(te, fk)
        models = {
            "BP": (NetClassifier(pats, nh, "BP", base + 1), True),
            "ABPNN": (NetClassifier(pats, nh, "ABPNN", base + 2), True),
            "BP-GA": (NetClassifier(pats, nh, "BP-GA", base + 3), True),
            "ABPNN-GA": (NetClassifier(pats, nh, "ABPNN-GA", base + 4), True),
            "MLP-Adam": (make_pipeline(StandardScaler(), MLPClassifier((nh,), max_iter=500, early_stopping=True,
                                                                         random_state=base + 5)), False),
            "RandomForest": (RandomForestClassifier(300, random_state=base + 6), False),
            "GradBoost": (HistGradientBoostingClassifier(random_state=base + 7), False),
            "SVM-RBF": (make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")), False),
        }
        for name, (m, is_net) in models.items():
            Xt_, yt_ = Xtr, tr["y"]
            if name == "SVM-RBF" and len(Xtr) > SVM_MAX_TRAIN:
                sub = np.random.default_rng(base + 8).choice(len(Xtr), SVM_MAX_TRAIN, replace=False)
                Xt_, yt_ = Xtr[sub], tr["y"][sub]
            rec["models"][name] = evaluate(m, Xt_, yt_, Xv, va["y"], Xte, te, n_cls, is_net)
            print(case, run, name, round(rec["models"][name]["val_acc"], 4),
                  round(rec["models"][name]["fit_seconds"], 1), flush=True)
        # feature ablation for ABPNN-GA
        for fk2 in FEATURES:
            if fk2 == fk:
                rec["feature_ablation"][fk2] = rec["models"]["ABPNN-GA"]; continue
            m = NetClassifier(pats, nh, "ABPNN-GA", base + 9)
            rec["feature_ablation"][fk2] = evaluate(m, feats(tr, fk2), tr["y"], feats(va, fk2), va["y"],
                                                    feats(te, fk2), te, n_cls, True)
        # T2 chart signals (only the current observation is available)
        t_tr, t_va, t_te = data["t2"]
        for name, m, is_net in [("ABPNN-GA", NetClassifier(pats, nh, "ABPNN-GA", base + 12), True),
                                ("RandomForest", RandomForestClassifier(300, random_state=base + 13), False)]:
            rec["t2"][name] = evaluate(m, t_tr["obs"], t_tr["y"], t_va["obs"], t_va["y"], t_te["obs"], t_te,
                                       n_cls, is_net)
        rec["t2_arl1"] = {str(dl): [float(t_te["rl"][(t_te["delta"] == dl) & (t_te["y"] == c)].mean())
                                    for c in range(n_cls)] for dl in DELTAS}
        R["runs"].append(rec)
        json.dump(results, open("results/results.json", "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
        print("saved", case, run, flush=True)
print("DONE", flush=True)
