import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, braids, forums as fo
BLUE, ORANGE, LB, LO, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#9cc0ec", "#f3b79c", "#222222", "#777777", "#d9d9d9"
OUT = "figures_out/"; plt.rcParams.update({"font.size": 8.5, "mathtext.fontset": "cm"})
def clean(a):
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
def tex(w): return " ".join(r"$\sigma_%d$" % g if g > 0 else r"$\sigma_%d^{-1}$" % -g for g in w)
# closed path: group A goes once around group B and returns to its starting state
u = np.linspace(0, 1, 121); x0, y0, Rx, Ry = 0.27, 0.17, 0.11, 0.085
th = np.pi - 2 * np.pi * u                                   # start left of B, go over the top (clockwise)
X = np.stack([x0 + Rx * np.cos(th), np.full_like(u, x0)], 1); Y = np.stack([y0 + Ry * np.sin(th), np.full_like(u, y0)], 1)
L = braids.track([fo.flow_matrix(x, y, np.array([3, 3])) for x, y in zip(X, Y)], substeps=4)
up = np.where(L[0].imag > 0.05)[0]; up = up[np.argsort(L[0, up].real)]
lo = np.array([int(np.argmin(np.abs(L[0] - np.conj(L[0, i])))) for i in up]); re = np.array([i for i in range(L.shape[1]) if i not in up and i not in lo])
wu, wl, wf = braids.word(L[:, up]), braids.word(L[:, lo]), braids.word(L[:, np.concatenate([up, lo])])
print("upper:", braids.fmt(wu), "| lower:", braids.fmt(wl), "| all complex strands:", braids.fmt(wf), "exponent sum", int(sum(np.sign(wf))), "| real eigenvalues:", np.round(L[0, re].real, 3))

fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.5), gridspec_kw=dict(width_ratios=[1.55, 1, 1]))
a = ax[0]; a.axhline(0, color=GRID, lw=1, zorder=0)
for idx, cols, ls, lw in ((up, (BLUE, ORANGE), "-", 1.9), (lo, (LB, LO), "-", 1.9)):
    for i, c in zip(idx, cols):
        z = L[:, i]; a.plot(z.real, z.imag, color=c, lw=lw, ls=ls)
        if np.ptp(z.real) > 0.05:
            k = len(z) // 4; a.annotate("", xy=(z[k + 8].real, z[k + 8].imag), xytext=(z[k - 8].real, z[k - 8].imag), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6))
        a.plot(z[0].real, z[0].imag, "o", color=c, ms=6, mec="white", zorder=5)
a.plot(L[0, re].real, np.zeros(len(re)), "o", color=INK, ms=4, zorder=5)
for idx, cols, dy, va in ((up, (BLUE, ORANGE), 0.028, "bottom"), (lo, ("#5f93cf", "#e08a63"), -0.028, "top")):
    zA, zB = L[0, idx[0]], L[0, idx[1]]
    a.text(zA.real - 0.022, zA.imag, r"$A$" if dy > 0 else r"$\bar A$", color=cols[0], ha="right", va="center", fontsize=9)
    a.text(zB.real, zB.imag + dy, r"$B$" if dy > 0 else r"$\bar B$", color=cols[1], ha="center", va=va, fontsize=9)
a.text(0.27, 0.285, "clockwise", color=BLUE, ha="center", fontsize=7.5); a.text(0.27, -0.305, "counter-clockwise", color="#5f93cf", ha="center", fontsize=7.5)
a.text(0.47, 0.17, r"Im $\lambda>0$:" + "\nused", color=INK, va="center", fontsize=7.5); a.text(0.47, -0.17, "mirror\nimages", color=MUTED, va="center", fontsize=7.5)
a.text(0.93, 0.03, "real", color=INK, ha="center", fontsize=7.5)
a.set_xlim(0.05, 1.05); a.set_ylim(-0.36, 0.36); a.set_xlabel(r"Re $\lambda$"); a.set_ylabel(r"Im $\lambda$"); clean(a)
a.set_title("(a) all eigenvalues of a real matrix", loc="left", fontsize=8, color=INK)
for b, idx, cols, w, ttl in ((ax[1], up, (BLUE, ORANGE), wu, "(b) upper half-plane: "), (ax[2], lo, (LB, LO), wl, "(c) mirror images: ")):
    Lu = L[:, idx]; t = np.linspace(0, 1, len(Lu)); rot = Lu * np.exp(-1j * braids.PHI); U, V = rot.real, rot.imag
    for i, c in enumerate(cols): b.plot(t, U[:, i], color=c, lw=2.4, solid_capstyle="round")
    sg = np.sign(U[:, 0] - U[:, 1])
    for x in np.where(sg[1:] != sg[:-1])[0]:
        f = 0 if V[x, 0] < V[x, 1] else 1; sl = slice(max(x - 30, 0), x + 32)
        b.plot(t[sl], U[sl, f], color="white", lw=9, solid_capstyle="butt", zorder=3); b.plot(t[sl], U[sl, f], color=cols[f], lw=2.4, zorder=4)
    b.set_xticks([]); b.set_yticks([]); b.set_xlabel("time"); clean(b); b.set_title(ttl + tex(w), loc="left", fontsize=8, color=INK)
fig.tight_layout(w_pad=0.8); fig.savefig(OUT + "fig_uhp.pdf"); fig.savefig(OUT + "fig_uhp.png", dpi=170)
