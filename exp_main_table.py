import sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
import data, feats
USERS, DEG = 20, 10                                   # about 3,400 interactions per snapshot
def one(label, seed):
    rng = np.random.default_rng(seed); snaps, info = data.sample(label, rng, "mixed", users=USERS, deg=DEG)
    row, w = feats.all_features(snaps); row.update(cls=label, exact=w == info["planted"], edges=info["edges"], ncomm=info["n"]); return row
G = {"Pooled spectra": "pool_", "Start and end snapshots": "ends_", "Sorted eigenvalue series": "sorted_", "Tracked geometry": "tracked_",
     "Braid exponent sums": "summ_", "Burau matrix": "burau_"}
X = lambda Df, p: Df[[c for c in Df if c.startswith(p)]].fillna(0).values
task = sys.argv[1]; labels = list(data.LABELS) if task == "history" else list(data.ORDERS); n = int(sys.argv[2])
jobs = [(lb, 9 * 10**6 + 10**5 * (task == "order") + 1000 * k + j) for k, lb in enumerate(labels) for j in range(n)]
Df = pd.DataFrame(Parallel(n_jobs=4)(delayed(one)(lb, s) for lb, s in jobs)); Df.to_csv(f"sparse_{task}.csv", index=False)
print(f"{task}: {len(Df)} networks, {len(labels)} classes, mean interactions/snapshot {Df.edges.mean():.0f}, exact word {100*Df.exact.mean():.1f}%")
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0); res = {}
for nm, p in G.items():
    a = cross_val_score(RandomForestClassifier(300, random_state=0, n_jobs=4), X(Df, p), Df.cls, cv=cv); res[nm] = [a.mean(), a.std()]
a = cross_val_score(KNeighborsClassifier(1), X(Df, "burau_"), Df.cls, cv=cv); res["Burau matrix, 1-NN"] = [a.mean(), a.std()]
for k, v in res.items(): print(f"  {k:28s} {100*v[0]:5.1f} ± {100*v[1]:.1f}")
json.dump(res, open(f"sparse_{task}.json", "w"), indent=1)
