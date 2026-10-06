import sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
import data, feats
G = {"Tracked geometry": "tracked_", "Exponent sums": "summ_", "Burau matrix": "burau_"}
X = lambda Df, p: Df[[c for c in Df if c.startswith(p)]].fillna(0).values
rf = lambda: RandomForestClassifier(300, random_state=0, n_jobs=4)
def one(label, seed, regime, users):
    rng = np.random.default_rng(seed); snaps, info = data.sample(label, rng, regime, users=users, deg=10)
    row, w = feats.all_features(snaps); row.update(cls=label, edges=info["edges"]); return row
def table(labels, n, regime, users, base):
    jobs = [(lb, base + 1000 * k + j) for k, lb in enumerate(labels) for j in range(n)]
    return pd.DataFrame(Parallel(n_jobs=4)(delayed(one)(lb, s, regime, users) for lb, s in jobs))
users = int(sys.argv[1]); out = {}
for task, labels, n in (("history", list(data.LABELS), 100), ("order", list(data.ORDERS), 50)):
    Dm, Dc, Dt = table(labels, n, "mixed", users, 11 * 10**6 + users), table(labels, n, "calm", users, 12 * 10**6 + users), table(labels, n, "turbulent", users, 13 * 10**6 + users)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    for nm, p in G.items():
        a = cross_val_score(rf(), X(Dm, p), Dm.cls, cv=cv)
        tr = [float((rf().fit(X(x, p), x.cls).predict(X(y, p)) == y.cls).mean()) for x, y in ((Dc, Dt), (Dt, Dc))]
        out[f"{task}|{nm}"] = [float(a.mean()), float(a.std())] + tr
        print(f"edges {int(Dm.edges.mean()):6d} {task:8s} {nm:18s} CV {100*a.mean():5.1f}±{100*a.std():.1f}  C->T {100*tr[0]:5.1f}  T->C {100*tr[1]:5.1f}", flush=True)
json.dump(out, open(f"dens_{users}.json", "w"), indent=1)
