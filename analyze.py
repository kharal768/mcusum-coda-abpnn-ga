"""Summarize results/results.json into tables (mean and SD over independent runs)."""
import json, numpy as np
r = json.load(open("results/results.json"))
D = [str(x) for x in r["settings"]["deltas"]]

def metrics(C):
    C = np.array(C, float); rec = np.diag(C) / C.sum(1); pr = np.diag(C) / np.maximum(C.sum(0), 1)
    f1 = np.where(pr + rec > 0, 2 * pr * rec / np.maximum(pr + rec, 1e-12), 0)
    return dict(acc=np.trace(C) / C.sum(), prec=pr.mean(), rec=rec.mean(), f1=f1.mean(), rec_c=rec, prec_c=pr, f1_c=f1)

def pooled(conf):
    return np.sum([np.array(conf[d]) for d in D], 0)

def summ(case, getter):
    runs = r["cases"][case]["runs"]
    return [getter(run) for run in runs]

out = {}
for case in r["cases"]:
    runs = r["cases"][case]["runs"]; o = out[case] = {}
    o["selection"] = r["cases"][case]["selection"]
    for name in runs[0]["models"]:
        per_d = {d: [metrics(run["models"][name]["conf"][d]) for run in runs] for d in D}
        allm = [metrics(pooled(run["models"][name]["conf"])) for run in runs]
        o[name] = dict(
            overall={k: (np.mean([m[k] for m in allm]), np.std([m[k] for m in allm], ddof=1)) for k in ["acc", "prec", "rec", "f1"]},
            by_delta={d: {k: (np.mean([m[k] for m in per_d[d]]), np.std([m[k] for m in per_d[d]], ddof=1)) for k in ["acc", "prec", "rec", "f1"]} for d in D},
            fit=(np.mean([run["models"][name]["fit_seconds"] for run in runs]), np.std([run["models"][name]["fit_seconds"] for run in runs], ddof=1)),
            pred=np.mean([run["models"][name]["predict_seconds_per_1000"] for run in runs]))
    for fk in runs[0]["feature_ablation"]:
        allm = [metrics(pooled(run["feature_ablation"][fk]["conf"])) for run in runs]
        o["feat_" + fk] = {k: (np.mean([m[k] for m in allm]), np.std([m[k] for m in allm], ddof=1)) for k in ["acc", "f1"]}
    for name in runs[0]["t2"]:
        per_d = {d: [metrics(run["t2"][name]["conf"][d]) for run in runs] for d in D}
        o["t2_" + name] = {d: {k: (np.mean([m[k] for m in per_d[d]]), np.std([m[k] for m in per_d[d]], ddof=1)) for k in ["acc", "f1"]} for d in D}
    # per-class for ABPNN-GA pooled over runs at each delta
    o["perclass"] = {d: {k: np.mean([metrics(run["models"]["ABPNN-GA"]["conf"][d])[k] for run in runs], 0).tolist() for k in ["prec_c", "rec_c", "f1_c"]} for d in D}
    o["arl1"] = {d: np.mean([run["arl1"][d] for run in runs], 0).tolist() for d in D}
    o["t2_arl1"] = {d: np.mean([run["t2_arl1"][d] for run in runs], 0).tolist() for d in D}
    o["conf_run0"] = runs[0]["models"]["ABPNN-GA"]["conf"]
json.dump(out, open("results/summary.json", "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else x)
if __name__ == "__main__":
    for case, o in out.items():
        print("=====", case, o["selection"])
        for name in r["cases"][case]["runs"][0]["models"]:
            ov = o[name]["overall"]
            print("%-12s acc %.3f±%.3f f1 %.3f±%.3f fit %.1f±%.1fs pred %.3fs/1000" % (name, *ov["acc"], *ov["f1"], *o[name]["fit"], o[name]["pred"]))
        print("features", {k: tuple(round(x, 3) for x in o[k]["acc"]) for k in o if k.startswith("feat_")})
        for d in D:
            a = o["ABPNN-GA"]["by_delta"][d]; t = o["t2_ABPNN-GA"][d]; rf = o["RandomForest"]["by_delta"][d]
            print(d, "ABPNN-GA acc %.3f±%.3f f1 %.3f | T2 acc %.3f±%.3f | RF %.3f" % (*a["acc"], a["f1"][0], *t["acc"], rf["acc"][0]), "ARL1", np.round(o["arl1"][d], 1), "T2 ARL1", np.round(o["t2_arl1"][d], 1))
