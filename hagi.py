# The Lie algebra gl_3 in the Weyl basis W_(i,j) = sum_k w^(ki) e_(k,k+j), indexed by Z_3^2, as the twist of
# the color Lie algebra C[Z_3^2] with bracket [e_g,e_h] = (1-d(g,h)) e_(g+h), and the gauge field structure constants.
import numpy as np, itertools
out = open("hagi-output.txt", "w")
def say(s):
    print(s, flush=True); out.write(s + "\n"); out.flush()
N = 3; w = np.exp(2j * np.pi / N)
G = [(i, j) for i in range(N) for j in range(N)]
add = lambda g, h: ((g[0] + h[0]) % N, (g[1] + h[1]) % N)
def W(g):
    i, j = g; M = np.zeros((N, N), dtype=complex)
    for k in range(N): M[k, (k + j) % N] = w ** (k * i)
    return M
sigma = lambda g, h: w ** (h[0] * g[1])                 # sigma((i,j),(i',j')) = w^(i'j)
d = lambda g, h: sigma(h, g) / sigma(g, h)              # so that sigma(g,h)d(g,h) = sigma(h,g)
ok = all(np.allclose(W(g) @ W(h), sigma(g, h) * W(add(g, h))) for g in G for h in G)
say("W_g W_h = sigma(g,h) W_(g+h): %s" % ok)
det = lambda g, h: (g[0] * h[1] - g[1] * h[0]) % N
say("d(g,h) = w^det(g,h) with det(g,h) = g_0 h_1 - g_1 h_0: %s" % all(np.isclose(d(g, h), w ** det(g, h)) for g in G for h in G))
say("W_g orthonormal for <a,b> = tr(ab^*): %s" % all(np.isclose(np.trace(W(g) @ W(h).conj().T) / N, 1.0 * (g == h)) for g in G for h in G))
# 1. commutators of gl_3 in the Weyl basis
ok = all(np.allclose(W(g) @ W(h) - W(h) @ W(g), sigma(g, h) * (1 - d(g, h)) * W(add(g, h))) for g in G for h in G)
say("[W_g,W_h] = sigma(g,h)(1-d(g,h)) W_(g+h): %s" % ok)
# 2. the color Lie algebra on C[Z_3^2]
br = lambda g, h: (1 - d(g, h), add(g, h))
eps = d
say("d is a commutation factor, d(g,h)d(h,g) = 1: %s" % all(np.isclose(d(g, h) * d(h, g), 1) for g in G for h in G))
say("eps-antisymmetry [e_h,e_g] = -d(h,g)[e_g,e_h]: %s" % all(np.isclose(br(h, g)[0], -d(h, g) * br(g, h)[0]) for g in G for h in G))
def nested(a, b, c):   # coefficient of e_(a+b+c) in [e_a,[e_b,e_c]]
    c1, bc = br(b, c); c2, _ = br(a, bc); return c1 * c2
jac = max(abs(eps(c, a) * nested(a, b, c) + eps(a, b) * nested(b, c, a) + eps(b, c) * nested(c, a, b)) for a in G for b in G for c in G)
say("eps-Jacobi identity, max defect over all triples: %.2e" % jac)
say("e_0 is central: %s" % all(np.isclose(br((0, 0), h)[0], 0) for h in G))
# 3. structure constants of the gauge field, and the trigonometric normalization
nz = sum(1 for g in G for h in G if g != (0, 0) and h != (0, 0) and not np.isclose(1 - d(g, h), 0))
say("ordered pairs of nonzero labels with nonzero bracket: %d of 64" % nz)
bad = []
for k in range(N):
    c = lambda g: w ** (k * g[0] * g[1])
    vals = set()
    for g in G:
        for h in G:
            J = lambda x: c(x) * W(x)
            C = J(g) @ J(h) - J(h) @ J(g); T = J(add(g, h))
            coef = np.trace(C @ T.conj().T) / N
            vals.add((round(float(coef.real), 4) + 0.0, round(float(coef.imag), 4) + 0.0))
            if k == 2 and not np.isclose(coef, 2j * np.sin(2 * np.pi * det(g, h) / N)): bad.append((g, h))
    say("normalization J_g = w^(%d g_0 g_1) W_g: structure constants in %s" % (k, sorted(vals)))
say("with k = 2: [J_g,J_h] = 2i sin(2 pi det(g,h)/3) J_(g+h) for all g,h: %s" % (len(bad) == 0))
say("done")
