# The key identity behind the spectral action classicality theorem: for a finite quantum group Q with
# C(Q) = sum of M_(n_beta), e the counit projection and Pi = Delta(e), the partial trace over the first leg
# of L_Pi R_Pi equals sum_beta n_beta^(-2) L_(z_beta). Checked for Q = dual of G, i.e. C(Q) = C[G], G nonabelian.
import numpy as np, itertools
out = open("beckham-output.txt", "w")
def say(s):
    print(s, flush=True); out.write(s + "\n"); out.flush()
def perm_group(gens, n):
    G = {tuple(range(n))}; frontier = list(G)
    while frontier:
        nf = []
        for g in frontier:
            for h in gens:
                k = tuple(g[h[i]] for i in range(n))
                if k not in G: G.add(k); nf.append(k)
        frontier = nf
    return sorted(G)
def quaternion():   # Q_8 as permutations of itself, via its regular action
    one = np.eye(2); i_ = np.array([[1j, 0], [0, -1j]]); j_ = np.array([[0, 1], [-1, 0]]); k_ = i_ @ j_
    els = [s * m for s in (1, -1) for m in (one, i_, j_, k_)]
    idx = lambda M: [t for t in range(8) if np.allclose(els[t], M)][0]
    return [tuple(idx(els[a] @ els[b]) for b in range(8)) for a in range(8)]
groups = {"S_3": perm_group([(1, 0, 2), (1, 2, 0)], 3), "Q_8": perm_group(quaternion()[2:4], 8),
          "S_4": perm_group([(1, 0, 2, 3), (1, 2, 3, 0)], 4), "D_5": perm_group([(1, 2, 3, 4, 0), (0, 4, 3, 2, 1)], 5)}
for name, G in groups.items():
    n = len(G); ix = {g: t for t, g in enumerate(G)}
    mul = lambda a, b: tuple(a[b[i]] for i in range(len(a)))
    Lg = [np.zeros((n, n)) for _ in G]; Rg = [np.zeros((n, n)) for _ in G]
    for a in G:
        for b in G:
            Lg[ix[a]][ix[mul(a, b)], ix[b]] = 1; Rg[ix[a]][ix[mul(b, a)], ix[b]] = 1
    # Pi = Delta(e) = (1/|G|) sum_g g (x) g, acting by left and right multiplication on L^2(Q) (x) L^2(Q)
    LPi = sum(np.kron(Lg[t], Lg[t]) for t in range(n)) / n
    RPi = sum(np.kron(Rg[t], Rg[t]) for t in range(n)) / n
    M = (LPi @ RPi).reshape(n, n, n, n)
    X = np.einsum('aiaj->ij', M)       # partial trace over the first leg
    ev = np.round(np.linalg.eigvalsh((X + X.T) / 2), 8)
    vals, mults = np.unique(ev, return_counts=True)
    # predicted: eigenvalue 1/n_beta^2 with multiplicity n_beta^2 for each irreducible representation
    say("%s, |G| = %d: eigenvalues of E(L_Pi R_Pi) %s with multiplicities %s" % (name, n, [round(float(v), 6) for v in vals], [int(m) for m in mults]))
    nbeta = [round(1 / np.sqrt(v)) for v in vals]
    say("   reading n_beta = 1/sqrt(eigenvalue): %s, multiplicity / n_beta^2 = number of irreducibles of that dimension: %s" % (nbeta, [int(m) // (nb * nb) for m, nb in zip(mults, nbeta)]))
    X_commutes = all(np.allclose(X @ Lg[t], Lg[t] @ X) and np.allclose(X @ Rg[t], Rg[t] @ X) for t in range(n))
    say("   E(L_Pi R_Pi) commutes with left and right multiplications, i.e. is a central multiplier: %s" % X_commutes)
    d = n; values = sorted(set(d + 2 / nb ** 2 for nb in nbeta))
    say("   spectral action values d + 2/n_beta^2 on the blocks: %s, against d + 2 = %s" % ([round(v, 4) for v in values], d + 2))
say("done")
