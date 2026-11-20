# The unoriented area tensor Re(b) on the 9 points of F_3^2: reduction to the quasigroup map
# L(v_a,v_c) = v_{-a-c}, classical symmetry group of L, failure of the Weyl models to intertwine L,
# and the smallest dimensions of the irreducible representations of PU_3 (no action of G_3 on <= 8 points).
import itertools, numpy as np
out = open("gullit-output.txt", "w")
def say(s):
    print(s, flush=True); out.write(s + "\n"); out.flush()
w = np.exp(2j * np.pi / 3)
pts = [(i, j) for i in range(3) for j in range(3)]
ix = {p: k for k, p in enumerate(pts)}
det = lambda g, h: (g[0] * h[1] - g[1] * h[0]) % 3
sub = lambda g, h: ((g[0] - h[0]) % 3, (g[1] - h[1]) % 3)
neg = lambda g: ((-g[0]) % 3, (-g[1]) % 3)
add = lambda g, h: ((g[0] + h[0]) % 3, (g[1] + h[1]) % 3)
# tensors T[a,c,r] = coefficient of v_r in T(v_a,v_c)
B = np.array([[[w ** (-det(sub(pts[a], pts[r]), sub(pts[c], pts[r]))) / 9 for r in range(9)] for c in range(9)] for a in range(9)])
ReB = B.real
one = np.ones((9, 9, 9))
C = np.array([[[1.0 if det(sub(pts[a], pts[r]), sub(pts[c], pts[r])) == 0 else 0.0 for r in range(9)] for c in range(9)] for a in range(9)])
say("Re(b) = -1/18 (v_a,v_c -> 1) + C/6: %s" % np.allclose(ReB, -one / 18 + C / 6))
L = np.zeros((9, 9, 9)); Id1 = np.zeros((9, 9, 9)); Id2 = np.zeros((9, 9, 9)); M = np.zeros((9, 9, 9)); U = np.zeros((9, 9, 9))
for a in range(9):
    for c in range(9):
        L[a, c, ix[neg(add(pts[a], pts[c]))]] = 1
        Id1[a, c, a] = 1; Id2[a, c, c] = 1
        if a == c:
            M[a, c, a] = 1; U[a, c, :] = 1
say("C = v_a + v_c + v_(-a-c) + delta_ac (1 - 3 v_a): %s" % np.allclose(C, Id1 + Id2 + L + U - 3 * M))
# the classical permutations preserving L, by brute force over the 9! permutations
cnt = 0
for perm in itertools.permutations(range(9)):
    if all(perm[ix[neg(add(pts[a], pts[c]))]] == ix[neg(add(pts[perm[a]], pts[perm[c]]))] for a in range(9) for c in range(a, 9)):
        cnt += 1
say("permutations of the 9 points preserving L: %d (expect |F_3^2 x| GL_2(3)| = 9*48 = 432)" % cnt)
# Weyl models: they intertwine b, but not L, so they do not factor through the stabilizer of Re(b)
Zc = np.diag([w ** k for k in range(3)]); S = np.roll(np.eye(3), -1, axis=0)
Wl = {g: np.linalg.matrix_power(Zc, g[0]) @ np.linalg.matrix_power(S, g[1]) for g in pts}
def model(x):
    P = {}
    for r in range(9):
        for s in range(9):
            v = (Wl[pts[r]] @ x @ Wl[pts[s]].conj().T).flatten() / np.sqrt(3)
            P[r, s] = np.outer(v, v.conj())
    return P
def defect(P, T):
    # max over r,i,j of || sum_{a,c} T[a,c,r] P[a,i] P[c,j] - sum_k P[r,k] T[i,j,k] ||
    m = 0
    for r, i, j in itertools.product(range(9), repeat=3):
        lhs = sum(T[a, c, r] * P[a, i] @ P[c, j] for a in range(9) for c in range(9) if T[a, c, r] != 0)
        rhs = sum(P[r, k] * T[i, j, k] for k in range(9) if T[i, j, k] != 0)
        m = max(m, np.linalg.norm(lhs - rhs))
    return m
rng = np.random.default_rng(7)
A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)); x, _ = np.linalg.qr(A)
P = model(x)
say("random Weyl model: defect for b %.2e, for L %.3f, for Re(b) %.3f" % (defect(P, B), defect(P, L), defect(P, ReB)))
# the quasigroup argument: in the relation sum_{x*y=k} p_xi p_yj = p_(k,i*j), the products PQ, PR, RQ coincide
# when the relation holds; here we only record that for a commutative (permutation) model the relation holds
perm = [ix[((pts[k][0] + 1) % 3, (2 * pts[k][1] + 2) % 3)] for k in range(9)]   # the affine map (v1,v2) -> (v1+1,2v2+2), of determinant -1
Pp = {(r, s): np.array([[1.0 if perm[s] == r else 0.0]]) for r in range(9) for s in range(9)}
say("orientation reversing affine permutation (v1,v2) -> (v1+1,2v2+2): defect for L %.1e, for Re(b) %.1e, for b %.3f" % (defect(Pp, L), defect(Pp, ReB), defect(Pp, B)))
# smallest dimensions of irreducible representations of PU_3: SU_3 highest weights (a,b) with a = b mod 3
dims = sorted(set((a + 1) * (b + 1) * (a + b + 2) // 2 for a in range(12) for b in range(12) if (a - b) % 3 == 0))
say("dimensions of the irreducible representations of PU_3, smallest ones: %s" % dims[:8])
say("so a coaction of G_3 on k <= 8 points, whose magic representation is 1 + (k-1), must be trivial")
say("done")
out.close()
