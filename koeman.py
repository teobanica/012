# Fourier expansion of the Weyl models: with V_ab(Y) = W_a Y W_b^* and t_ab(x) = tr(W_a^* x W_b x^*), check that
#   P^x_rs = Proj(W_r x W_s^*) = (1/m) sum_{a,b} d(a,r) conj(d(b,s)) t_ab(x) V_ab ,   m = |Gamma| = N^2,
# and that V_ab V_cd = sigma(a,c) conj(sigma(b,d)) V_{a+c,b+d}, for H = Z_3, Z_4 and Z_2 x Z_2.
import itertools, numpy as np
out = open("koeman-output.txt", "w")
def say(s):
    print(s, flush=True); out.write(s + "\n"); out.flush()
def weyl_cyclic(n):
    w = np.exp(2j * np.pi / n)
    labels = [(i, j) for i in range(n) for j in range(n)]
    W = {}
    for (i, j) in labels:
        M = np.zeros((n, n), complex)
        for k in range(n):
            M[k, (k + j) % n] = w ** (k * i)
        W[((i,), (j,))] = M
    return W
def group_data(ns):
    # Gamma = prod_i (Z_ni x Z_ni), labels ((i_1..i_r),(j_1..j_r)), Weyl operators as tensor products
    parts = [weyl_cyclic(n) for n in ns]
    labels = []; W = {}
    for combo in itertools.product(*[list(p.keys()) for p in parts]):
        lab = (tuple(c[0][0] for c in combo), tuple(c[1][0] for c in combo))
        M = np.array([[1.0 + 0j]])
        for c, p in zip(combo, parts):
            M = np.kron(M, p[c])
        labels.append(lab); W[lab] = M
    add = lambda g, h: (tuple((a + b) % n for a, b, n in zip(g[0], h[0], ns)), tuple((a + b) % n for a, b, n in zip(g[1], h[1], ns)))
    return labels, W, add
def check(ns, name, rng):
    labels, W, add = group_data(ns)
    N = W[labels[0]].shape[0]; m = len(labels)
    tr = lambda X: np.trace(X) / N
    # read off sigma and d from the operators: W_g W_h = sigma(g,h) W_{g+h}, W_r W_a W_r^* = d(a,r) W_a
    sig = {(g, h): tr(W[add(g, h)].conj().T @ W[g] @ W[h]) for g in labels for h in labels}
    d = {(a, r): tr(W[a].conj().T @ W[r] @ W[a] @ W[r].conj().T) for a in labels for r in labels}
    V = {(a, b): np.kron(W[a], W[b].conj()) for a in labels for b in labels}      # row-major vec(W_a Y W_b^*)
    mult = max(np.abs(V[a, b] @ V[c, e] - sig[a, c] * np.conj(sig[b, e]) * V[add(a, c), add(b, e)]).max()
               for a, b, c, e in [tuple(labels[i] for i in rng.integers(0, m, 4)) for _ in range(200)])
    A = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)); x, _ = np.linalg.qr(A)
    t = {(a, b): tr(W[a].conj().T @ x @ W[b] @ x.conj().T) for a in labels for b in labels}
    worst = 0
    for _ in range(12):
        r, s = labels[rng.integers(m)], labels[rng.integers(m)]
        v = (W[r] @ x @ W[s].conj().T).flatten(); v = v / np.linalg.norm(v)
        P = np.outer(v, v.conj())
        Q = sum(d[a, r] * np.conj(d[b, s]) * t[a, b] * V[a, b] for a in labels for b in labels) / m
        worst = max(worst, np.abs(P - Q).max())
    say("%s: N = %d, |Gamma| = %d: V_ab V_cd rule defect %.1e, Fourier expansion of P^x_rs defect %.1e (12 random entries)" % (name, N, m, mult, worst))
rng = np.random.default_rng(11)
check([3], "H = Z_3", rng); check([4], "H = Z_4", rng); check([2, 2], "H = Z_2 x Z_2", rng)
say("done")
out.close()
