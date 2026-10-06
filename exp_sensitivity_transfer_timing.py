import sys, json, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
import data, feats, braids
G = {"Pooled spectra": "pool_", "Start and end snapshots": "ends_", "Sorted eigenvalue series": "sorted_", "Tracked geometry": "tracked_",
     "Braid exponent sums": "summ_", "Burau matrix": "burau_"}
X = lambda Df, p: Df[[c for c in Df if c.startswith(p)]].fillna(0).values
rf = lambda: RandomForestClassifier(300, random_state=0, n_jobs=4)
def one(label, seed, regime, users):
    rng = np.random.default_rng(seed); snaps, info = data.sample(label, rng, regime, users=users, deg=10)
    row, w = feats.all_features(snaps); row.update(cls=label, exact=w == info["planted"], edges=info["edges"]); return row
def table(labels, n, regime, users, base):
    jobs = [(lb, base + 1000 * k + j) for k, lb in enumerate(labels) for j in range(n)]
    return pd.DataFrame(Parallel(n_jobs=4)(delayed(one)(lb, s, regime, users) for lb, s in jobs))
which = sys.argv[1]
if which == "sweep":
    users = int(sys.argv[2]); Df = table(list(data.LABELS), 50, "mixed", users, 6 * 10**6 + users)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    row = dict(edges=int(Df.edges.mean()), word_exact=float(Df.exact.mean()))
    for nm, p in G.items(): row[nm] = float(cross_val_score(rf(), X(Df, p), Df.cls, cv=cv).mean())
    print(row, flush=True); json.dump(row, open(f"sweep_{users}.json", "w"))
elif which == "shift":
    res = {}
    for task, labels in (("history", list(data.LABELS)), ("order", list(data.ORDERS))):
        n = 100 if task == "history" else 50
        Dc, Dt = table(labels, n, "calm", 20, 7 * 10**6), table(labels, n, "turbulent", 20, 8 * 10**6)
        for nm, p in G.items():
            res[f"{task}|{nm}"] = [float((rf().fit(X(a, p), a.cls).predict(X(b, p)) == b.cls).mean()) for a, b in ((Dc, Dt), (Dt, Dc))]
            print(task, nm, np.round(res[f"{task}|{nm}"], 3), flush=True)
    json.dump(res, open("shift.json", "w"), indent=1)
elif which == "time":
    rng = np.random.default_rng(0); snaps, info = data.sample("Broadcast-driven", rng, "mixed", users=300, deg=20, D=6); n = len(snaps[0])
    for m in (10**5, 10**6, 10**7):
        lab = rng.integers(0, n, max(1000, m // 20)); src = rng.integers(0, len(lab), m); dst = rng.integers(0, len(lab), m)
        t0 = time.perf_counter()
        for _ in range(5):
            C = np.bincount(lab[src] * n + lab[dst], minlength=n * n).reshape(n, n).astype(float); P = C / C.sum(1, keepdims=True)
        print(f"edges {m:>9d}: aggregate {1000*(time.perf_counter()-t0)/5:.1f} ms")
    t0 = time.perf_counter(); L = braids.track(snaps, substeps=10); t1 = time.perf_counter(); Lu = feats.strands(snaps); w = braids.word(Lu); feats.burau_features(w)
    print(f"communities {n}, snapshots {len(snaps)}: tracking {1000*(t1-t0)/len(snaps):.1f} ms per snapshot")
