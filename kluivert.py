# The weak case N=2: the Weyl basis of M_2(C) is the Pauli basis, d takes the values +1,-1, and gl_2 is the
# twist of the color Lie algebra C[Z_2^2] with bracket [e_g,e_h] = (1-d(g,h)) e_(g+h). Also the Pauli models.
import numpy as np
out = open("kluivert-output.txt", "w")
def say(s):
    print(s, flush=True); out.write(s + "\n"); out.flush()
N = 2; w = -1.0
G = [(i, j) for i in range(N) for j in range(N)]
add = lambda g, h: ((g[0] + h[0]) % N, (g[1] + h[1]) % N)
def W(g):
    i, j = g; M = np.zeros((N, N), dtype=complex)
    for k in range(N): M[k, (k + j) % N] = w ** (k * i)
    return M
for g in G: say("W%s = %s" % (str(g), W(g).real.astype(int).tolist()))
sigma = lambda g, h: w ** (h[0] * g[1])
d = lambda g, h: sigma(h, g) / sigma(g, h)
say("W_g W_h = sigma(g,h) W_(g+h): %s" % all(np.allclose(W(g) @ W(h), sigma(g, h) * W(add(g, h))) for g in G for h in G))
say("values of d: %s" % sorted(set(int(round(d(g, h))) for g in G for h in G)))
say("[W_g,W_h] = sigma(g,h)(1-d(g,h)) W_(g+h): %s" % all(np.allclose(W(g) @ W(h) - W(h) @ W(g), sigma(g, h) * (1 - d(g, h)) * W(add(g, h))) for g in G for h in G))
nz = [g for g in G if g != (0, 0)]
say("untwisted bracket [e_g,e_h] = 2 e_(g+h) for distinct nonzero g,h: %s" % all(np.isclose(1 - d(g, h), 2) for g in nz for h in nz if g != h))
say("untwisted bracket symmetric on nonzero labels, [e_h,e_g] = [e_g,e_h]: %s" % all(np.isclose(-d(h, g) * (1 - d(g, h)), 1 - d(h, g)) for g in nz for h in nz))
br = lambda g, h: (1 - d(g, h), add(g, h))
def nested(a, b, c):
    c1, bc = br(b, c); c2, _ = br(a, bc); return c1 * c2
jac = max(abs(d(c, a) * nested(a, b, c) + d(a, b) * nested(b, c, a) + d(b, c) * nested(c, a, b)) for a in G for b in G for c in G)
say("eps-Jacobi identity, max defect over all 64 triples: %.2e" % jac)
say("e_0 central: %s" % all(np.isclose(1 - d((0, 0), h), 0) for h in G))
say("ordered pairs of nonzero labels interacting: %d of 9" % sum(1 for g in nz for h in nz if not np.isclose(1 - d(g, h), 0)))
# the Pauli models P^x_rs = Proj(W_r x W_s^*), x in SU_2
rng = np.random.default_rng(1)
def proj(v):
    v = v.reshape(-1); v = v / np.linalg.norm(v); return np.outer(v, v.conj())
def model(x):
    return {(r, s): proj(W(r) @ x @ W(s).conj().T) for r in G for s in G}
for t in range(3):
    a, b = rng.normal(size=2) + 1j * rng.normal(size=2); n = np.sqrt(abs(a) ** 2 + abs(b) ** 2); a, b = a / n, b / n
    x = np.array([[a, -b.conjugate()], [b, a.conjugate()]])
    P = model(x)
    magic = all(np.allclose(sum(P[(r, s)] for s in G), np.eye(4)) and np.allclose(sum(P[(s, r)] for s in G), np.eye(4)) for r in G)
    comm = max(np.abs(P[(G[0], G[1])] @ P[(G[1], G[2])] - P[(G[1], G[2])] @ P[(G[0], G[1])]).max() for _ in [0])
    say("random x in SU_2, test %d: P^x magic %s, noncommutativity %.3f" % (t + 1, magic, comm))
say("done")
