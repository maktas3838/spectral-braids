import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, braids, forums as fo
BLUE, ORANGE, INK = "#2a78d6", "#eb6834", "#222222"; OUT = "figures_out/"
plt.rcParams.update({"font.size": 8.5, "mathtext.fontset": "cm"})
def tex(w): return " ".join(r"$\sigma_%d$" % g if g > 0 else r"$\sigma_%d^{-1}$" % -g for g in w)
u = np.linspace(0, 1, 121)
# closed: A goes once around B and returns
th = np.pi - 2 * np.pi * u
closed = (np.stack([0.27 + 0.11 * np.cos(th), np.full_like(u, 0.27)], 1), np.stack([0.17 + 0.085 * np.sin(th), np.full_like(u, 0.17)], 1))
# open: A passes over B and ends somewhere else; B drifts
opened = (np.stack([0.16 + 0.27 * u, 0.27 - 0.03 * u], 1), np.stack([0.17 + 0.10 * np.sin(np.pi * u) + 0.05 * u, 0.17 - 0.04 * u], 1))
fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
for a, (X, Y), name in ((ax[0], closed, "(a) closed: "), (ax[1], opened, "(b) open: ")):
    L = braids.track([fo.flow_matrix(x, y, np.array([3, 3])) for x, y in zip(X, Y)], substeps=4)
    up = np.where(L[0].imag > 0.05)[0]; up = up[np.argsort(L[0, up].real)]; Lu = L[:, up]; w = braids.word(Lu)
    for i, c in ((0, BLUE), (1, ORANGE)):
        z = Lu[:, i]; a.plot(z.real, z.imag, color=c, lw=1.8)
        if np.abs(np.diff(z)).sum() > 0.1:
            k = len(z) // 4; a.annotate("", xy=(z[k + 8].real, z[k + 8].imag), xytext=(z[k - 8].real, z[k - 8].imag), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.5))
        a.plot(z[-1].real, z[-1].imag, "s", color=c, ms=7.5, mec="white", zorder=4); a.plot(z[0].real, z[0].imag, "o", color="white", ms=4, mec=c, mew=1.3, zorder=5)
    a.set_title(name + tex(w), loc="left", fontsize=8, color=INK); a.set_xlabel(r"Re $\lambda$"); a.set_xlim(0.04, 0.50); a.set_ylim(0.03, 0.31)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    print(name, braids.fmt(w), "start", np.round(Lu[0], 3), "end", np.round(Lu[-1], 3))
ax[0].set_ylabel(r"Im $\lambda$")
fig.tight_layout(w_pad=0.6); fig.savefig(OUT + "fig_open.pdf"); fig.savefig(OUT + "fig_open.png", dpi=200)
