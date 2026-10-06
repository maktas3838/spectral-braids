import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
import braids, forums as fo
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#222222", "#777777", "#e2e2e2"
OUT = "figures_out/"
plt.rcParams.update({"font.size": 8.5, "mathtext.fontset": "cm"})
def clean(a):
    for sp in ("top", "right"): a.spines[sp].set_visible(False)

# ---------------------------------------------------------------- anatomy of one crossing
def crossing():
    fig, ax = plt.subplots(1, 4, figsize=(7.16, 2.05), gridspec_kw=dict(width_ratios=[1.45, 0.75, 1.45, 0.75]))
    for col, s in ((0, 1), (2, -1)):
        u = np.linspace(0, 1, 61); x1, x2, y0, A = 0.16, 0.36, 0.17, 0.085; m, rho = (x1 + x2) / 2, (x2 - x1) / 2
        X = np.stack([m - rho * np.cos(np.pi * u), m + rho * np.cos(np.pi * u)], 1)
        Y = np.stack([y0 + s * A * np.sin(np.pi * u), y0 - s * A * np.sin(np.pi * u)], 1)
        mats = [fo.flow_matrix(x, y, np.array([3, 3])) for x, y in zip(X, Y)]
        L = braids.track(mats, substeps=4); up = np.where(L[0].imag > 0.05)[0]; up = up[np.argsort(L[0, up].real)]; Lu = L[:, up]
        w = braids.word(Lu); tex = r"$\sigma_1^{-1}$" if w == [-1] else r"$\sigma_1$"; assert len(w) == 1
        a = ax[col]
        for i, c in ((0, BLUE), (1, ORANGE)):
            z = Lu[:, i]; a.plot(z.real, z.imag, color=c, lw=1.8)
            k = len(z) // 2; a.annotate("", xy=(z[k + 6].real, z[k + 6].imag), xytext=(z[k - 6].real, z[k - 6].imag), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6))
            a.plot(z[0].real, z[0].imag, "o", color=c, ms=6, mec="white", zorder=5); a.plot(z[-1].real, z[-1].imag, "s", color=c, ms=5.5, mec="white", zorder=5)
        zA, zB = Lu[len(Lu) // 2, 0], Lu[len(Lu) // 2, 1]
        a.text(zA.real, zA.imag - s * 0.022, "group A", color=BLUE, ha="center", va="top" if s > 0 else "bottom", fontsize=7.5)
        a.text(zB.real, zB.imag + s * 0.022, "group B", color=ORANGE, ha="center", va="bottom" if s > 0 else "top", fontsize=7.5)
        a.set_xlim(Lu.real.min() - 0.05, Lu.real.max() + 0.05); a.set_ylim(0.0, 0.36)
        a.set_xlabel(r"Re $\lambda$  (echo-chamber level)"); a.set_ylabel(r"Im $\lambda$  (broadcast level)"); clean(a)
        a.set_title(("(a) " if s > 0 else "(c) ") + ("A broadcasts more than B" if s > 0 else "A broadcasts less than B"), loc="left", fontsize=8, color=INK)
        # braid diagram
        b = ax[col + 1]; t = np.linspace(0, 1, len(Lu)); rot = Lu * np.exp(-1j * braids.PHI); U, V = rot.real, rot.imag
        for i, c in ((0, BLUE), (1, ORANGE)): b.plot(t, U[:, i], color=c, lw=2.4, solid_capstyle="round")
        x = int(np.where(np.sign(U[1:, 0] - U[1:, 1]) != np.sign(U[:-1, 0] - U[:-1, 1]))[0][0]); f = 0 if V[x, 0] < V[x, 1] else 1
        sl = slice(x - 30, x + 32); b.plot(t[sl], U[sl, f], color="white", lw=10, solid_capstyle="butt", zorder=3); b.plot(t[sl], U[sl, f], color=(BLUE, ORANGE)[f], lw=2.4, zorder=4)
        b.set_xticks([]); b.set_yticks([]); b.set_xlabel("time"); b.set_ylabel("order by Re $\\lambda$", labelpad=2); clean(b)
        b.set_title(("(b) " if s > 0 else "(d) ") + tex, loc="left", fontsize=8, color=INK)
        print("s =", s, "word", braids.fmt(w), "front strand:", "A" if f == 0 else "B")
    fig.tight_layout(w_pad=0.5); fig.savefig(OUT + "fig_crossing.pdf"); fig.savefig(OUT + "fig_crossing.png", dpi=170)

# ---------------------------------------------------------------- worked example of one group
def graph(a, r, f, g, title):
    ang = np.deg2rad([90, 210, 330]); pos = [(np.cos(t), np.sin(t)) for t in ang]
    for i in range(3):
        for j, p, rad in (((i + 1) % 3, f, 0.28), ((i - 1) % 3, g, 0.28)):
            a.add_patch(FancyArrowPatch(pos[i], pos[j], connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=5 + 14 * p,
                                        lw=0.4 + 5.5 * p, color=BLUE, alpha=0.85, shrinkA=9, shrinkB=9))
    for i, (x, y) in enumerate(pos):
        a.add_patch(Circle((x, y), 0.20, facecolor="white", edgecolor=BLUE, lw=1.4, zorder=3)); a.text(x, y, "123"[i], ha="center", va="center", fontsize=8, zorder=4)
    a.set_xlim(-1.75, 1.75); a.set_ylim(-1.2, 1.45); a.set_aspect("equal"); a.axis("off")
    a.set_title(title, loc="left", fontsize=8, color=INK)
    M = np.array([[r, f, g], [g, r, f], [f, g, r]])
    rows = r" \\ ".join(" & ".join(f"{v:.3g}" for v in row) for row in M)
    a.text(0, -1.25, "\n".join("  ".join(f"{v:5.3f}".rstrip("0").rstrip(".").ljust(5) for v in row) for row in M), ha="center", va="top", fontsize=7.5, family="monospace", color=INK)
    return np.linalg.eigvals(M)

def example():
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.35), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    r, x = 0.40, 0.05
    e1 = graph(ax[0], r, 0.50, 0.05, "(a) broadcast: $r=0.40$, $f=0.50$, $g=0.05$")
    e2 = graph(ax[1], r, 0.275, 0.275, "(b) conversation: $r=0.40$, $f=g=0.275$")
    a = ax[2]; a.axhline(0, color=GRID, lw=0.9, zorder=0)
    for e, c, mk, lab in ((e1, BLUE, "o", "broadcast"), (e2, ORANGE, "s", "conversation")):
        a.plot(e.real, e.imag, mk, color=c, ms=6.5, mec="white", label=lab, zorder=4 if mk == "o" else 3)
    l1 = e1[np.argmax(e1.imag)]
    a.annotate("", xy=(l1.real, 0.03), xytext=(l1.real, l1.imag - 0.03), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1)); 
    a.annotate("", xy=(l1.real, -0.03), xytext=(l1.real, -l1.imag + 0.03), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1))
    a.text(l1.real + 0.035, l1.imag, r"$\lambda_1=0.125+0.390\,i$", color=BLUE, va="center", fontsize=7.5)
    a.text(l1.real + 0.035, -l1.imag, r"$\lambda_2=0.125-0.390\,i$", color=BLUE, va="center", fontsize=7.5)
    a.text(l1.real + 0.035, 0.07, r"$\lambda_1=\lambda_2=0.125$", color=ORANGE, va="center", fontsize=7.5)
    a.text(0.95, 0.07, r"$\lambda_0=0.95$", color=INK, ha="center", fontsize=7.5)
    a.set_xlim(-0.1, 1.38); a.set_ylim(-0.55, 0.55); a.set_xticks([0, 0.5, 1.0]); a.set_xlabel(r"Re $\lambda$"); a.set_ylabel(r"Im $\lambda$"); clean(a)
    a.legend(frameon=False, fontsize=7, loc="lower right", handletextpad=0.2)
    a.set_title("(c) eigenvalues of the two blocks", loc="left", fontsize=8, color=INK)
    fig.tight_layout(w_pad=0.6); fig.savefig(OUT + "fig_example.pdf"); fig.savefig(OUT + "fig_example.png", dpi=170)
    print("broadcast eig", np.round(np.sort_complex(e1), 3), "| conversation eig", np.round(np.sort_complex(e2), 3))
crossing(); example()
