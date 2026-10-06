import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, braids, data, feats
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#8a4fd3"]; INK = "#222222"; OUT = "figures_out/"
plt.rcParams.update({"font.size": 8.5, "mathtext.fontset": "cm"})
def clean(a):
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
rng = np.random.default_rng(11)
snaps, info = data.sample("Mixed, broadcast first", rng, "mixed", users=60, deg=10, D=4)
S = np.array([feats.top(M)[:4] for M in snaps])              # sorted separately in every snapshot (baseline)
Lu = feats.strands(snaps)[::10]                               # tracked strands at the snapshot times (ours)
t = np.arange(len(snaps)); print("edges/snapshot", info["edges"], "word", braids.fmt(braids.word(feats.strands(snaps))), "sorted shape", S.shape)
fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.85), sharex=True, sharey=True)
for a, Z, ttl in ((ax[0], S, "(a) sorted"), (ax[1], Lu, "(b) tracked (ours)")):
    for k in range(4): a.plot(t, Z[:, k].real, color=COL[k], lw=1.3)
    a.set_title(ttl, loc="left", fontsize=8, color=INK); a.set_xlabel("time (snapshots)"); clean(a); a.set_ylim(0.0, 0.52)
ax[0].set_ylabel("Re $\\lambda$")
fig.tight_layout(w_pad=0.6); fig.savefig(OUT + "fig_sorting.pdf"); fig.savefig(OUT + "fig_sorting.png", dpi=200)
