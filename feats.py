import numpy as np, braids
NMAX = 6
TS = (0.5, np.exp(1j))                                    # two evaluation points of the Burau variable t

def burau(w, n, t):
    """Unreduced Burau matrix of a braid word in B_n: sigma_i -> I (+) [[1-t, t],[1, 0]] (+) I."""
    B = np.eye(n, dtype=complex)
    for g in w:
        i = abs(g) - 1; G = np.eye(n, dtype=complex); G[i:i + 2, i:i + 2] = [[1 - t, t], [1, 0]]
        B = B @ (G if g > 0 else np.linalg.inv(G))
    return B

def burau_features(w):
    """Braid on m <= NMAX strands is placed in B_NMAX by the standard inclusion (extra strands idle)."""
    f = {}
    for k, t in enumerate(TS):
        B = burau(w, NMAX, t)
        for i in range(NMAX):
            for j in range(NMAX):
                f[f"burau_{k}r{i}{j}"] = B[i, j].real
                if np.iscomplex(t): f[f"burau_{k}i{i}{j}"] = B[i, j].imag
    return f

def strands(snaps, thr=0.05):
    L = braids.track(snaps, substeps=10)
    up = np.where(L[0].imag > thr)[0][:NMAX + 2]; up = up[np.argsort(L[0, up].real)][:NMAX]
    return L[:, up]

def area(z): return 0.5 * np.sum(z.real * np.roll(z.imag, -1) - np.roll(z.real, -1) * z.imag)
def series(Z, pre):
    f = {}
    for k in range(NMAX):
        if k >= Z.shape[1]:
            f.update({f"{pre}{k}_{s}": 0.0 for s in ("rm", "im", "rs", "is", "area", "path", "ric")}); continue
        z = Z[:, k]
        f.update({f"{pre}{k}_rm": z.real.mean(), f"{pre}{k}_im": z.imag.mean(), f"{pre}{k}_rs": z.real.std(), f"{pre}{k}_is": z.imag.std(),
                  f"{pre}{k}_area": area(z), f"{pre}{k}_path": np.abs(np.diff(z)).sum(), f"{pre}{k}_ric": np.nan_to_num(np.corrcoef(z.real, z.imag)[0, 1])})
    return f
def top(M, thr=0.05):
    ev = np.linalg.eigvals(M); up = np.sort_complex(ev[ev.imag > thr])[:NMAX]
    return np.pad(up, (0, NMAX - len(up)))

def all_features(snaps):
    Lu = strands(snaps); w = braids.word(Lu); m = Lu.shape[1]
    row = dict(word=braids.fmt(w), nstrands=m)
    row.update(burau_features(w))
    for g in range(1, NMAX):                              # simple hand-made braid summary, for comparison
        row[f"summ_e{g}"] = sum(np.sign(x) for x in w if abs(x) == g); row[f"summ_n{g}"] = sum(abs(x) == g for x in w)
    S = np.array([top(M) for M in snaps]); al = S[S != 0]
    for nm, q in (("re", al.real), ("im", al.imag)):
        for pc in (5, 25, 50, 75, 95): row[f"pool_{nm}{pc}"] = np.percentile(q, pc)
    H, _, _ = np.histogram2d(al.real, al.imag, bins=6, range=((0, 0.52), (0.05, 0.3)), density=True)
    row.update({f"pool_h{i}": v for i, v in enumerate(H.ravel())})
    a, b = S[:3].mean(0), S[-3:].mean(0)
    for k in range(NMAX): row.update({f"ends_a{k}r": a[k].real, f"ends_a{k}i": a[k].imag, f"ends_b{k}r": b[k].real, f"ends_b{k}i": b[k].imag})
    row.update(series(S, "sorted_")); row.update(series(Lu[::10], "tracked_"))
    return row, w
