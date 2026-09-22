"""Generate the LaTeX tables and figure used in the manuscript from results/summary.json."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = json.load(open("results/summary.json")); R = json.load(open("results/results.json"))
H = json.load(open("results/h_calibration.json"))
D = [str(x) for x in R["settings"]["deltas"]]
f = lambda ms, n=3: ("%%.%df" % n) % ms[0] + r" (%s)" % (("%%.%df" % n) % ms[1])
T = {}
# Table: diagnostic performance of ABPNN-GA by delta
rows = []
for d in D:
    cells = []
    for case in ["bronze", "lithium"]:
        b = S[case]["ABPNN-GA"]["by_delta"][d]
        cells += [f(b[k]) for k in ["acc", "prec", "rec", "f1"]]
    rows.append(r"%s & %s \\" % (d, " & ".join(cells)))
T["bydelta"] = "\n".join(rows)
# Table: classifier comparison
names = ["BP", "ABPNN", "BP-GA", "ABPNN-GA", "MLP-Adam", "RandomForest", "GradBoost", "SVM-RBF"]
labels = {"BP": "BP", "ABPNN": "ABPNN", "BP-GA": "BP-GA", "ABPNN-GA": "ABPNN-GA", "MLP-Adam": "MLP (Adam)",
          "RandomForest": "Random forest", "GradBoost": "Gradient boosting", "SVM-RBF": "SVM (RBF)"}
rows = []
for n in names:
    c = []
    for case in ["bronze", "lithium"]:
        o = S[case][n]; c += [f(o["overall"]["acc"]), f(o["overall"]["f1"]), "%.1f" % o["fit"][0]]
    rows.append(r"%s & %s \\" % (labels[n], " & ".join(c)))
T["models"] = "\n".join(rows)
# Table: feature ablation
rows = []
for fk, lab in [("obs", r"$\bar{\mathbf{y}}_t^{*}-\bm{\mu}_0^{*}$ (current observation)"), ("u", r"$\mathbf{u}_t$ (cumulative vector)"),
                ("u+runmean", r"$\mathbf{u}_t$ and run mean $\bar{\mathbf{e}}_t$")]:
    c = []
    for case in ["bronze", "lithium"]:
        c += [f(S[case]["feat_" + fk]["acc"]), f(S[case]["feat_" + fk]["f1"])]
    rows.append(r"%s & %s \\" % (lab, " & ".join(c)))
T["features"] = "\n".join(rows)
# Table: MCUSUM vs T2
rows = []
for d in D:
    c = []
    for case in ["bronze", "lithium"]:
        c += [f(S[case]["ABPNN-GA"]["by_delta"][d]["acc"]), f(S[case]["t2_ABPNN-GA"][d]["acc"]),
              "%.1f" % np.mean(S[case]["arl1"][d]), "%.1f" % np.mean(S[case]["t2_arl1"][d])]
    rows.append(r"%s & %s \\" % (d, " & ".join(c)))
T["t2"] = "\n".join(rows)
json.dump(T, open("results/latex_tables.json", "w"), indent=1)
# Figure
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
x = [float(d) for d in D]
for a, case, title in zip(ax, ["bronze", "lithium"], ["Three-part composition", "Four-part composition"]):
    for key, lab, st in [("ABPNN-GA", r"MCUSUM-CoDa, input $\mathbf{u}_t$" + (" and run mean" if S[case]["selection"]["features"] != "u" else ""), "o-"),
                         ("feat_obs", "MCUSUM-CoDa, current observation only", "s--"), ("t2", r"$T^2$-CoDa", "^:")]:
        if key == "ABPNN-GA":
            y = [S[case]["ABPNN-GA"]["by_delta"][d]["acc"][0] for d in D]
        elif key == "t2":
            y = [S[case]["t2_ABPNN-GA"][d]["acc"][0] for d in D]
        else:
            runs = R["cases"][case]["runs"]
            y = [np.mean([np.trace(np.array(r["feature_ablation"]["obs"]["conf"][d])) / np.sum(r["feature_ablation"]["obs"]["conf"][d]) for r in runs]) for d in D]
        a.plot(x, y, st, label=lab)
    a.set_title(title); a.set_xlabel(r"Shift size $\delta$ (in standard deviations)"); a.grid(alpha=.3)
ax[0].set_ylabel("Diagnostic accuracy (ABPNN-GA)"); ax[0].legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig("results/accuracy_vs_delta.pdf"); fig.savefig("results/accuracy_vs_delta.png", dpi=150)
print(json.dumps(T, indent=1)[:3000])
