"""Dynamic-network classification dataset (online communities, no forum labels given).

Each sample = one dynamic directed network with its OWN communities (12-24, in random order)
and ONE label saying how its strongest echo chamber formed. Complex eigenvalues can only come
from loops of >=3 communities, so the generator contains D in {4,5,6} hidden loops of 3 or 4
communities; the loops are not given to any method, the community order is shuffled, and a
random background mixing (EPS) breaks the exact block structure."""
import numpy as np, forums as fo

LABELS = {"Broadcast-driven": lambda n: [1] * n, "Conversation-driven": lambda n: [-1] * n,
          "Mixed, broadcast first": lambda n: [(-1) ** p for p in range(n)],
          "Mixed, conversation first": lambda n: [-(-1) ** p for p in range(n)]}
EPS = 0.03
# second task: the same four events (two pairs of neighbours each swap twice, once per style), in 8 different orders
ORDERS = {}
for a, b in ((0, 1), (1, 0)):
    for x in (1, -1):
        for y in (1, -1):
            ORDERS[f"order{len(ORDERS) + 1}"] = [(a, x), (b, y), (a, -x), (b, -y)]

def sample(label, rng, regime="mixed", users=None, deg=None, D=None):
    D = D or int(rng.integers(4, 7)); ks = rng.choice([3, 4], D); n = int(ks.sum())
    events = ORDERS[label] if label in ORDERS else [(p, s) for p, s in enumerate(LABELS[label](D - 1))]
    users = users or int(rng.integers(150, 501)); deg = deg or int(rng.integers(10, 31))
    size = np.maximum((users * rng.uniform(0.5, 1.5, n)).astype(int), 2)
    Q = rng.dirichlet(np.ones(n), n); perm = rng.permutation(n)
    while True:                                           # redraw the rare trajectory that leaves the feasible region
        X, Y = fo.states(events, D, rng, regime)
        try: Ws = [fo.flow_matrix(X[t], Y[t], ks) for t in range(len(X))]; break
        except AssertionError: pass
    snaps = []
    for t in range(len(X)):
        W = (1 - EPS) * Ws[t] + EPS * Q
        C = np.vstack([rng.multinomial(size[i] * deg, W[i]) for i in range(n)])
        C = C[np.ix_(perm, perm)]                         # communities appear in random order
        snaps.append(C / C.sum(1, keepdims=True))
    return snaps, dict(D=D, n=n, edges=int(size.sum() * deg), users=int(size.sum()), planted=[-(p + 1) * s for p, s in events])
