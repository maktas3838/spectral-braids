"""Spectral braid computation for a sequence of snapshot operators.

Convention: strands are projected on a generic line; a crossing of positions
i, i+1 is sigma_i when the swap is counter-clockwise (the strand moving right
passes in front = has the smaller coordinate orthogonal to the line) and
sigma_i^{-1} when clockwise.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

PHI = 0.37


def track(mats, substeps=20):
    """Eigenvalue strands along the piecewise-linear path through mats."""
    lam = np.linalg.eigvals(mats[0])
    L = [lam]
    for a, b in zip(mats[:-1], mats[1:]):
        for s in np.linspace(0, 1, substeps + 1)[1:]:
            new = np.linalg.eigvals((1 - s) * a + s * b)
            _, c = linear_sum_assignment(np.abs(L[-1][:, None] - new[None]))
            L.append(new[c])
    return np.array(L)


def word(L, phi=PHI):
    rot = L * np.exp(-1j * phi)
    U, V = rot.real, rot.imag
    order = list(np.argsort(U[0]))
    w = []
    for j in range(1, len(L)):
        changed = True
        while changed:
            changed = False
            for p in range(len(order) - 1):
                a, b = order[p], order[p + 1]
                if U[j, a] > U[j, b]:                    # a (left) moves right past b
                    va, vb = V[j - 1:j + 1, a].mean(), V[j - 1:j + 1, b].mean()
                    w.append((p + 1) * (1 if va < vb else -1))   # in front = smaller v
                    order[p], order[p + 1] = b, a
                    changed = True
    return reduce(w)


def reduce(w):
    out = []
    for g in w:
        if out and out[-1] == -g:
            out.pop()
        else:
            out.append(g)
    return out


def upper_half(L):
    """Braid of the strands in the upper half-plane (conjugation-equivariant).
    Returns (word, number of strands, ok) ; ok=False if a strand crossed R."""
    up = np.where(L[0].imag > 1e-9)[0]
    ok = bool(np.all(L[:, up].imag > 1e-9))
    return word(L[:, up]), len(up), ok


def permutation(w, n):
    p = list(range(n))
    for g in w:
        i = abs(g) - 1
        p[i], p[i + 1] = p[i + 1], p[i]
    return p


def braid_features(L):
    wf = word(L)
    wu, m, ok = upper_half(L)
    perm = permutation(wu, m)
    return dict(up_expsum=sum(np.sign(wu)), up_len=len(wu),
                up_swapped=float(sum(p != i for i, p in enumerate(perm)) > 0),
                full_expsum=sum(np.sign(wf)), full_len=len(wf)), wu, ok


def fmt(w):
    return " ".join(("s%d" % g) if g > 0 else ("S%d" % -g) for g in w) or "trivial"
