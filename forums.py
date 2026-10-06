"""Open-braid benchmark, online-community version: ECHO-CHAMBER HISTORIES.

D forums; forum d has k_d subgroups (3 or 4, drawn at random per network) of unequal size.
A user's interactions: a share r stays in the own subgroup (echo-chamber level), the rest
goes to the next / previous subgroup of the forum with net one-way share c (broadcast level),
and a share XC goes to other forums. Forum d contributes the eigenvalue
    z_d = r + (1 - XC - r) cos(2 pi / k_d)  +  i c sin(2 pi / k_d)
(k=3: 1.5 r - (1-XC)/2 + i 0.866 c;  k=4: r + i c).  States are specified in the eigenvalue
plane (x = echo-chamber coordinate, y = broadcast coordinate) and converted to (r, c)."""
import numpy as np

XC, Y0 = 0.05, 0.17
REGIMES = {"calm":      dict(A=(0.070, 0.075), bursty=False, jit=0.000, drift=0.0000, asym=0.0, dur=(16, 20), gap=(4, 6)),
           "turbulent": dict(A=(0.050, 0.090), bursty=True,  jit=0.006, drift=0.0008, asym=0.3, dur=(10, 26), gap=(1, 9))}
REGIMES["mixed"] = REGIMES["turbulent"]

def slots(D): return np.linspace(0.06, 0.46, D)

def to_rc(x, y, k):
    R, ck, sk = 1 - XC, np.cos(2 * np.pi / k), np.sin(2 * np.pi / k)
    return (x - R * ck) / (1 - ck), y / sk

def flow_matrix(x, y, ks):
    off = np.concatenate([[0], np.cumsum(ks)]); n = off[-1]; W = np.zeros((n, n))
    for d, k in enumerate(ks):
        r, c = to_rc(x[d], y[d], k); rest = 1 - XC - r; f, g = (rest + c) / 2, (rest - c) / 2
        assert r > 0.02 and g > 0.005, (r, c, k)
        others = [e for e in range(len(ks)) if e != d]
        for a in range(k):
            i = off[d] + a
            for e in others: W[i, off[e]:off[e + 1]] = XC / len(others) / ks[e]
            W[i, i] += r; W[i, off[d] + (a + 1) % k] += f; W[i, off[d] + (a - 1) % k] += g
    return W

def warp(n, rng, bursty):
    if not bursty: return np.linspace(0, 1, n + 1)[1:]
    u = np.cumsum(rng.gamma(0.8, 1.0, n)); return u / u[-1]

def states(events, D, rng, regime="mixed"):
    g = REGIMES[regime]; s0 = slots(D); slot = s0 + rng.uniform(-g["jit"], g["jit"], D)
    at = list(range(D)); X = [slot.copy()]; Y = [np.full(D, Y0)]
    def hold(n):
        for _ in range(n): X.append(X[-1].copy()); Y.append(Y[-1].copy())
    hold(int(rng.integers(*g["gap"]) + 1))
    for p, s in events:
        a, b = at[p], at[p + 1]; m, rho = (slot[p] + slot[p + 1]) / 2, (slot[p + 1] - slot[p]) / 2
        A = rng.uniform(*g["A"]); ka, kb = 1 + rng.uniform(-g["asym"], g["asym"], 2); bx, by = X[-1].copy(), Y[-1].copy()
        for u in warp(int(rng.integers(g["dur"][0], g["dur"][1] + 1)), rng, g["bursty"]):
            xx, yy = bx.copy(), by.copy()
            xx[a] = m - rho * np.cos(np.pi * u); yy[a] = Y0 + s * A * ka * np.sin(np.pi * u)
            xx[b] = m + rho * np.cos(np.pi * u); yy[b] = Y0 - s * A * kb * np.sin(np.pi * u)
            X.append(xx); Y.append(yy)
        at[p], at[p + 1] = b, a; hold(int(rng.integers(*g["gap"]) + 1))
    X, Y = np.array(X), np.array(Y)
    if g["drift"]: Y = Y + np.cumsum(rng.normal(0, g["drift"], Y.shape), 0)
    return X, Y

def network(events, D, rng, regime="mixed", users=None, deg=None, ks=None):
    ks = np.array(ks) if ks is not None else rng.choice([3, 4], D)
    users = users or int(rng.integers(150, 501)); deg = deg or int(rng.integers(10, 31))
    size = np.maximum((users * rng.uniform(0.5, 1.5, ks.sum())).astype(int), 2)      # unequal subgroup sizes
    X, Y = states(events, D, rng, regime); snaps = []
    for t in range(len(X)):
        W = flow_matrix(X[t], Y[t], ks)
        C = np.vstack([rng.multinomial(size[i] * deg, W[i]) for i in range(len(W))])
        snaps.append(C / C.sum(1, keepdims=True))
    return snaps, dict(ks=ks, edges=int(size.sum() * deg), users=int(size.sum()))
