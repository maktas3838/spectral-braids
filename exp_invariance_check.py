"""Direct check of the invariance remark: calm and turbulent networks of the same class
give the same braid word (dense snapshots, so sampling noise plays no role)."""
import warnings, numpy as np, data, feats, braids
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
def one(lb, seed, reg):
    rng = np.random.default_rng(seed); snaps, info = data.sample(lb, rng, reg, users=300, deg=20)
    return reg, braids.word(feats.strands(snaps)) == info["planted"], len(snaps)
if __name__ == "__main__":
    jobs = [(lb, 16 * 10**6 + 1000 * k + j, reg) for reg in ("calm", "turbulent") for k, lb in enumerate(data.LABELS) for j in range(40)]
    R = Parallel(n_jobs=4)(delayed(one)(*j) for j in jobs)
    for reg in ("calm", "turbulent"):
        x = [r for r in R if r[0] == reg]; print(reg, "networks", len(x), "exact word", sum(r[1] for r in x), "snapshots", min(r[2] for r in x), "-", max(r[2] for r in x))
