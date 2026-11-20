# Explicit witnesses for the spectral action test of section 8, for Q = dual of a finite group G,
# C(Q) = C[G], with the regular coaction U(g x h) = g x gh and the counit block projection e.
# (1) coefficientwise reality K U (xi x 1) = U (J xi x 1) holds, while K U = U K fails for G nonabelian;
# (2) two-point external triple on C^3 = span(0,L,R), D = B + C: the fluctuated operator (B+C) x (1-L_e)
#     has nullity d+2, while the transformed operator B x (1-P) + C x (1-R) has partial kernel trace
#     d + 2 E(PR), with eigenvalues d + 2/n^2 on the blocks of size n;
# (3) zero momentum sector of the flat 4-torus: t gamma^1 x (P - R) has partial kernel trace 4((d-2) + 2 E(PR)).
import numpy as np
out = open("zlatan-output.txt", "w")
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
def quaternion():
    one = np.eye(2); i_ = np.array([[1j, 0], [0, -1j]]); j_ = np.array([[0, 1], [-1, 0]]); k_ = i_ @ j_
    els = [s * m for s in (1, -1) for m in (one, i_, j_, k_)]
    idx = lambda M: [t for t in range(8) if np.allclose(els[t], M)][0]
    return [tuple(idx(els[a] @ els[b]) for b in range(8)) for a in range(8)]
groups = {"S_3": perm_group([(1, 0, 2), (1, 2, 0)], 3), "Q_8": perm_group(quaternion()[2:4], 8)}
# external two-point triple: basis 0, L, R with bimodule labels (1,1), (2,1), (1,2)
e0, eL, eR = np.eye(3)
Bm = np.outer(e0, eL) + np.outer(eL, e0); Cm = np.outer(e0, eR) + np.outer(eR, e0)
Dm = Bm + Cm; gam = np.diag([1., -1., -1.]); Jperm = np.array([[1., 0, 0], [0, 0, 1], [0, 1, 0]])   # J = Jperm o conjugation
p = np.diag([0., 1., 0.])
left = lambda a1, a2: np.diag([a1, a2, a1]); right = lambda b1, b2: np.diag([b1, b1, b2])
fo = max(np.abs((Dm @ left(*a) - left(*a) @ Dm) @ right(*b) - right(*b) @ (Dm @ left(*a) - left(*a) @ Dm)).max()
         for a in [(1, 0), (0, 1)] for b in [(1, 0), (0, 1)])
say("two-point triple: first order defect %.1e, JD = DJ %s, J gamma = gamma J %s, D gamma = -gamma D %s, B = (1-2p)[D,p] %s" % (
    fo, np.allclose(Jperm @ Dm, Dm @ Jperm), np.allclose(Jperm @ gam, gam @ Jperm), np.allclose(Dm @ gam, -gam @ Dm),
    np.allclose(Bm, (np.eye(3) - 2 * p) @ (Dm @ p - p @ Dm))))
def kernel_proj(X, tol=1e-9):
    ev, V = np.linalg.eigh((X + X.conj().T) / 2)
    K = V[:, np.abs(ev) < tol]
    return K @ K.conj().T
for name, G in groups.items():
    n = len(G); ix = {g: t for t, g in enumerate(G)}
    mul = lambda a, b: tuple(a[b[i]] for i in range(len(a)))
    inv = lambda a: tuple(sorted(range(len(a)), key=lambda i: a[i]))
    Lg = [np.zeros((n, n)) for _ in G]; Rg = [np.zeros((n, n)) for _ in G]
    for a in G:
        for b in G:
            Lg[ix[a]][ix[mul(a, b)], ix[b]] = 1; Rg[ix[a]][ix[mul(b, a)], ix[b]] = 1
    # (1) reality conditions for U(g x h) = g x gh, with j(g) = g^{-1} on both legs
    Ureg = np.zeros((n * n, n * n)); Kp = np.zeros((n * n, n * n)); jp = np.zeros((n, n))
    for a in G:
        jp[ix[inv(a)], ix[a]] = 1
        for b in G:
            Ureg[ix[a] * n + ix[mul(a, b)], ix[a] * n + ix[b]] = 1
            Kp[ix[inv(a)] * n + ix[inv(b)], ix[a] * n + ix[b]] = 1
    one = ix[tuple(range(len(G[0])))]
    coeff = all(np.allclose(Kp @ Ureg[:, ix[a] * n + one], Ureg @ np.kron(jp[:, ix[a]], np.eye(n)[one])) for a in G)
    say("%s: coefficientwise reality K U (xi x 1) = U (J xi x 1): %s;  full commutation K U = U K: %s" % (name, coeff, np.allclose(Kp @ Ureg, Ureg @ Kp)))
    Le = sum(Lg) / n
    LPi = sum(np.kron(Lg[t], Lg[t]) for t in range(n)) / n
    RPi = sum(np.kron(Rg[t], Rg[t]) for t in range(n)) / n
    say("   U (L_e x 1) U^* = L_Pi: %s;  K L_Pi K^{-1} = R_Pi: %s" % (np.allclose(Ureg @ np.kron(Le, np.eye(n)) @ Ureg.T, LPi), np.allclose(Kp @ LPi @ Kp.T, RPi)))
    # (2) two-point witness
    DA = np.kron(Bm + Cm, np.eye(n) - Le)
    null0 = int(round(np.trace(kernel_proj(DA)).real))
    Dt = np.kron(Bm, np.eye(n * n) - LPi) + np.kron(Cm, np.eye(n * n) - RPi)
    Kt = kernel_proj(Dt).reshape(3, n, n, 3, n, n)
    E = np.einsum('xaixaj->ij', Kt)
    vals, mults = np.unique(np.round(np.linalg.eigvalsh((E + E.conj().T) / 2), 6), return_counts=True)
    say("   two-point: nullity of the fluctuated operator %d (d+2 = %d); partial kernel trace of the transformed one: eigenvalues %s, multiplicities %s" % (
        null0, n + 2, [float(v) for v in vals], [int(m) for m in mults]))
    # (3) torus, zero momentum sector: t gamma^1 x (P - R), spinor factor C^4
    g1 = np.kron(np.array([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]]))
    Kz = kernel_proj(np.kron(g1, 0.3 * (LPi - RPi))).reshape(4, n, n, 4, n, n)
    Ez = np.einsum('xaixaj->ij', Kz)
    vals, mults = np.unique(np.round(np.linalg.eigvalsh((Ez + Ez.conj().T) / 2), 6), return_counts=True)
    say("   torus zero modes: partial kernel trace eigenvalues %s, multiplicities %s, against 4d = %d before transformation" % (
        [float(v) for v in vals], [int(m) for m in mults], 4 * n))
    # (4) heat cutoff exp(-s D^2) at s = 1: the transformed partial trace minus the original trace, on each block,
    #     against the exact value 2 (1 - e^{-s})^2 (1/n^2 - 1)
    for s in (0.4, 1.0):
        ev, V = np.linalg.eigh((Dt + Dt.T) / 2)
        Ht = (V * np.exp(-s * ev ** 2)) @ V.conj().T
        Et = np.einsum('xaixaj->ij', Ht.reshape(3, n, n, 3, n, n))
        S0 = np.sum(np.exp(-s * np.linalg.eigvalsh(DA) ** 2))
        diff = np.round(np.linalg.eigvalsh((Et + Et.conj().T) / 2) - S0, 8)
        a = np.exp(-s)
        say("   heat cutoff s = %.1f: transformed minus original, eigenvalues %s, predicted %s" % (
            s, sorted(set(float(v) + 0.0 for v in diff)), sorted(set(round(float(2 * (1 - a) ** 2 * (1 / k ** 2 - 1)), 8) for k in (1, 2)))))
    # (5) left and right multiplications: L(C(Q)) cap R(C(Q)) is the center, of dimension the number of blocks
    Lspan = np.array([M.flatten() for M in Lg]).T; Rspan = np.array([M.flatten() for M in Rg]).T
    inter = Lspan.shape[1] + Rspan.shape[1] - np.linalg.matrix_rank(np.hstack([Lspan, Rspan]))
    say("   dim of L(C(Q)) cap R(C(Q)) = %d (number of conjugacy classes, i.e. dimension of the center)" % inter)
say("done")
out.close()
