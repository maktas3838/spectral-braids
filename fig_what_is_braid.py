"""Figure 1: what is a braid?"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
C = ["#2a78d6", "#eb6834", "#1baf7a"]; INK, MUTED = "#222222", "#777777"
plt.rcParams.update({"font.size": 8.5})

def positions(s):
    """3 points; s in [0,1]: first half p0,p1 swap counter-clockwise, second half p1(now at x=0) and p2 swap clockwise."""
    P = np.array([-1.0, 0.0, 1.0], dtype=complex)
    if s <= 0.5:
        a = np.pi * (s / 0.5)                      # 0 -> pi, counter-clockwise
        m = -0.5
        P[0] = m + (-0.5) * np.exp(1j * a)
        P[1] = m + (0.5) * np.exp(1j * a)
    else:
        P[0], P[1] = 0.0, -1.0                     # after first swap
        a = -np.pi * ((s - 0.5) / 0.5)             # 0 -> -pi, clockwise
        m = 0.5
        P[0] = m + (-0.5) * np.exp(1j * a)
        P[2] = m + (0.5) * np.exp(1j * a)
    return P

S = np.linspace(0, 1, 400)
traj = np.array([positions(s) for s in S])
fig = plt.figure(figsize=(10.5, 2.9))
# (a) motion in the plane
ax = fig.add_subplot(1, 3, 1)
for i in range(3):
    ax.plot(traj[:, i].real, traj[:, i].imag, color=C[i], lw=1.6)
    ax.plot(traj[0, i].real, traj[0, i].imag, "o", color=C[i], ms=9, mec="white", mew=1.5)
    ax.text(traj[0, i].real, traj[0, i].imag - 0.32, f"point {i+1}", ha="center", color=C[i], fontsize=8)
for (i, k) in ((0, 100), (1, 100), (0, 300), (2, 300)):
    ax.add_patch(FancyArrowPatch((traj[k, i].real, traj[k, i].imag), (traj[k + 8, i].real, traj[k + 8, i].imag),
                                 arrowstyle="-|>", mutation_scale=11, color=C[i], lw=0))
ax.text(-0.5, 0.62, "step 1: points 1,2\nswap counter-clockwise", ha="center", fontsize=7.5, color=INK)
ax.text(0.5, -0.78, "step 2: points 1,3\nswap clockwise", ha="center", fontsize=7.5, color=INK)
ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.05, 1.0); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("(a) points moving in the plane", loc="left", fontsize=9, color=INK)
# (b) strands in plane x time (time vertical)
ax = fig.add_subplot(1, 3, 2, projection="3d")
for i in range(3):
    ax.plot(traj[:, i].real, traj[:, i].imag, S, color=C[i], lw=2.2)
for s0 in (0.0, 0.5, 1.0):
    P = positions(s0)
    xx = np.array([-1.6, 1.6, 1.6, -1.6, -1.6]); yy = np.array([-0.8, -0.8, 0.8, 0.8, -0.8])
    ax.plot(xx, yy, np.full(5, s0), color="#c8c8c8", lw=0.7)
    for i in range(3):
        ax.scatter([P[i].real], [P[i].imag], [s0], color=C[i], s=16, depthshade=False)
ax.text(1.75, -0.8, 0.0, "t = 0", fontsize=7, color=MUTED); ax.text(1.75, -0.8, 1.0, "t = 1", fontsize=7, color=MUTED)
ax.set_xlim(-1.6, 1.6); ax.set_ylim(-0.8, 0.8); ax.set_zlim(0, 1)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_edgecolor("white"); axis.line.set_color((1, 1, 1, 0))
ax.grid(False)
ax.view_init(elev=14, azim=-80); ax.set_box_aspect((1.5, 0.75, 1.5))
ax.set_title("(b) paths over time = strands", loc="left", fontsize=9, color=INK)
# (c) braid diagram + generators
ax = fig.add_subplot(1, 3, 3)
def crossing(ax, x0, i, sign, cols):
    # strands at heights i (bottom) and i+1 swap between x0 and x0+1; sign +1: strand moving up is over
    xs = np.linspace(x0, x0 + 1, 50); up = i + (xs - x0); dn = i + 1 - (xs - x0)
    over, under = (up, dn) if sign > 0 else (dn, up)
    co, cu = (cols[0], cols[1]) if sign > 0 else (cols[1], cols[0])
    gap = np.abs(xs - (x0 + 0.5)) < 0.13
    ax.plot(xs, over, color=co, lw=2.2, zorder=3)
    u = under.copy(); u[gap] = np.nan; ax.plot(xs, u, color=cu, lw=2.2, zorder=2)
# braid of panel (a): strand order bottom->top = position x: p1(-1), p2(0), p3(1)
# step 1: positions 1,2 swap  -> sigma_1 ; step 2: positions 2,3 swap -> sigma_2^{-1}
ax.plot([0, 0.4], [0, 0], color=C[0], lw=2.2); ax.plot([0, 0.4], [1, 1], color=C[1], lw=2.2); ax.plot([0, 1.4], [2, 2], color=C[2], lw=2.2)
crossing(ax, 0.4, 0, +1, (C[0], C[1]))
ax.plot([1.4, 1.8], [0, 0], color=C[1], lw=2.2); ax.plot([1.4, 1.8], [1, 1], color=C[0], lw=2.2)
ax.plot([1.4, 1.8], [2, 2], color=C[2], lw=2.2)
crossing(ax, 1.8, 1, -1, (C[0], C[2]))
ax.plot([1.8, 3.2], [0, 0], color=C[1], lw=2.2); ax.plot([2.8, 3.2], [1, 1], color=C[2], lw=2.2); ax.plot([2.8, 3.2], [2, 2], color=C[0], lw=2.2)
ax.text(0.9, -0.55, r"$\sigma_1$", ha="center", fontsize=11); ax.text(2.3, -0.55, r"$\sigma_2^{-1}$", ha="center", fontsize=11)
ax.annotate("", xy=(3.2, -0.95), xytext=(0, -0.95), arrowprops=dict(arrowstyle="-|>", color=MUTED))
ax.text(1.6, -1.25, "time", ha="center", color=MUTED, fontsize=8)
ax.text(3.35, 1.0, r"braid $=\sigma_1\,\sigma_2^{-1}$", fontsize=9.5, va="center", color=INK)
ax.set_xlim(-0.2, 5.2); ax.set_ylim(-1.4, 2.4); ax.axis("off")
ax.set_title("(c) the braid diagram and its word", loc="left", fontsize=9, color=INK)
fig.tight_layout(w_pad=0.5)
fig.savefig("figures_out/fig_what_is_braid.pdf"); fig.savefig("figures_out/fig_what_is_braid.png", dpi=150)
