import json, glob, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, braids, data, feats
from scipy.ndimage import uniform_filter1d
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#8a4fd3", "#d1a000", "#c2408a"]; INK, MUTED = "#222", "#777"; OUT = "figures_out/"
plt.rcParams.update({"font.size": 8.5})
def tex(w): return " ".join(r"$\sigma_%d$" % g if g > 0 else r"$\sigma_%d^{-1}$" % -g for g in w)
def clean(a):
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
def diagram(b, Lu, lw=2.0):
    n = Lu.shape[1]; t = np.arange(len(Lu)) / 10
    rot = Lu * np.exp(-1j * braids.PHI); U = uniform_filter1d(rot.real, 31, axis=0, mode="nearest"); V = uniform_filter1d(rot.imag, 31, axis=0, mode="nearest")
    for i in range(n): b.plot(t, U[:, i], color=COL[i], lw=lw, solid_capstyle="round")
    for i in range(n):
        for j in range(i):
            s = np.sign(U[:, i] - U[:, j])
            for x in np.where(s[1:] != s[:-1])[0]:
                f = i if V[x, i] < V[x, j] else j; sl = slice(max(x - 22, 0), x + 24)
                b.plot(t[sl], U[sl, f], color="white", lw=lw + 4, solid_capstyle="butt"); b.plot(t[sl], U[sl, f], color=COL[f], lw=lw)
    b.set_yticks([]); b.set_xlabel("time (snapshots)"); clean(b)
def get(label, seed, D):
    rng = np.random.default_rng(seed); snaps, info = data.sample(label, rng, "mixed", users=400, deg=25, D=D)
    Lu = feats.strands(snaps); return snaps, info, Lu, braids.word(Lu)

def pipeline():
    snaps, info, Lu, w = get("Mixed, broadcast first", 11, 4)
    fig, ax = plt.subplots(1, 4, figsize=(12, 2.75), gridspec_kw=dict(width_ratios=[0.8, 1.05, 1.25, 0.8]))
    a = ax[0]; im = a.imshow(snaps[len(snaps) // 2], cmap="Blues", vmin=0, vmax=0.6); a.set_xticks([]); a.set_yticks([])
    a.set_xlabel("to community"); a.set_ylabel("from community"); a.set_title("(a) one snapshot: $P_t$", loc="left", fontsize=8.5, color=INK)
    a = ax[1]
    for i in range(Lu.shape[1]):
        a.plot(Lu[:, i].real, Lu[:, i].imag, color=COL[i], lw=1.1); a.plot(Lu[0, i].real, Lu[0, i].imag, "o", color=COL[i], ms=5, mec="white")
        a.plot(Lu[-1, i].real, Lu[-1, i].imag, "s", color=COL[i], ms=4.5, mec="white")
    a.set_xlabel(r"Re $\lambda$  (echo-chamber level)"); a.set_ylabel(r"Im $\lambda$  (broadcast level)"); clean(a)
    a.set_title("(b) eigenvalue paths (circle: start, square: end)", loc="left", fontsize=8.5, color=INK)
    diagram(ax[2], Lu); ax[2].set_title("(c) braid:  " + tex(w), loc="left", fontsize=8.5, color=INK)
    a = ax[3]; B = feats.burau(w, Lu.shape[1], 0.5).real; a.imshow(B, cmap="RdBu_r", vmin=-2.2, vmax=2.2); a.set_xticks([]); a.set_yticks([])
    for i in range(B.shape[0]):
        for j in range(B.shape[1]): a.text(j, i, f"{B[i, j]:.2g}", ha="center", va="center", fontsize=7.5, color=INK)
    a.set_title("(d) Burau matrix ($t=0.5$)", loc="left", fontsize=8.5, color=INK)
    fig.tight_layout(w_pad=0.6); fig.savefig(OUT + "fig_pipeline.pdf"); fig.savefig(OUT + "fig_pipeline.png", dpi=150); print("pipeline word", braids.fmt(w), info["n"])

def classes():
    fig, ax = plt.subplots(1, 4, figsize=(12, 2.1))
    for k, (lb, a) in enumerate(zip(data.LABELS, ax)):
        snaps, info, Lu, w = get(lb, 30 + k, 5); diagram(a, Lu, lw=1.8)
        a.set_title(f"{lb}\n{tex(w)}", loc="left", fontsize=8.5, color=INK)
    ax[0].set_ylabel("strands, ordered by Re $\\lambda$")
    fig.tight_layout(w_pad=0.6); fig.savefig(OUT + "fig_classes.pdf"); fig.savefig(OUT + "fig_classes.png", dpi=150)

def sweep():
    R = sorted([json.load(open(f)) for f in glob.glob("results/sweep_*.json")], key=lambda r: r["edges"]); x = [r["edges"] for r in R]
    fig, a = plt.subplots(figsize=(3.5, 2.65))
    for nm, c, ls, lab in (("Burau matrix", COL[0], "-", "Ours: Burau matrix"), ("Tracked geometry", COL[2], "-", "Ours: tracked geometry")):
        a.plot(x, [100 * r[nm] for r in R], ls, marker="o", ms=3.5, color=c, lw=1.6, label=lab)
    a.plot(x, [100 * r["word_exact"] for r in R], "--", marker="s", ms=3, color=COL[0], lw=1, label="Ours: exact word")
    for nm, c, mk, lab in (("Sorted eigenvalue series", COL[4], "^", "Sorted series"), ("Start and end snapshots", "#8a8a8a", "v", "Start and end"),
                           ("Pooled spectra", COL[1], "D", "Pooled spectra")):
        a.plot(x, [100 * r[nm] for r in R], "-", marker=mk, ms=3.2, color=c, lw=1.1, label=lab)
    a.axhline(25, color=INK, lw=0.6, ls=":"); a.set_xscale("log"); a.set_ylim(0, 104)
    a.set_xlabel("edges per snapshot"); a.set_ylabel("accuracy (%)"); a.set_xticks([2000, 5000, 10000, 20000]); a.set_xticklabels(["2k", "5k", "10k", "20k"]); a.minorticks_off()
    a.legend(frameon=False, fontsize=6.3, ncol=2, loc="lower center", bbox_to_anchor=(0.45, 1.0), columnspacing=1.0); clean(a)
    fig.tight_layout(); fig.savefig(OUT + "fig_sweep.pdf"); fig.savefig(OUT + "fig_sweep.png", dpi=150)
pipeline(); classes(); sweep()
