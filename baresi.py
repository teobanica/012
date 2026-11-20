# The toy real spectral triple with S_N^+ symmetry from section 6, at N=4: H = L (x) L (x) C^2, L = l^2(X), |X| = 4,
# D = (P(x)1 + 1(x)P)(x)sigma_x, gamma = 1(x)1(x)sigma_z, J = flip (x) sigma_x composed with complex conjugation.
# Checks the axioms, and the symmetry under the Pauli magic model of S_4^+ acting via u(x)u.
import numpy as np
out = open("baresi-output.txt", "w")
def say(s):
    print(s, flush=True); out.write(s + "\n"); out.flush()
N = 4; I = np.eye; kron = np.kron
P = np.ones((N, N)) / N
sx = np.array([[0, 1], [1, 0]]); sz = np.diag([1, -1])
D = kron(kron(P, I(N)) + kron(I(N), P), sx)
gam = kron(I(N * N), sz)
F = np.zeros((N * N, N * N))
for i in range(N):
    for j in range(N): F[i * N + j, j * N + i] = 1
Jlin = kron(F, sx)                      # J(v) = Jlin conj(v)
def Jop(A): return Jlin @ A.conj() @ Jlin.T   # J A J^{-1}
def alg(f): return kron(kron(np.diag(f), I(N)), I(2))
rng = np.random.default_rng(0)
fs = [rng.normal(size=N) + 1j * rng.normal(size=N) for _ in range(3)]
say("J^2 = 1: %s, JD = DJ: %s, J gamma = -gamma J: %s" % (np.allclose(Jlin @ Jlin.conj(), I(2 * N * N)), np.allclose(Jop(D), D), np.allclose(Jop(gam), -gam)))
say("gamma D = -D gamma: %s, gamma commutes with the algebra: %s" % (np.allclose(gam @ D, -D @ gam), all(np.allclose(gam @ alg(f), alg(f) @ gam) for f in fs)))
say("order zero condition: %s" % all(np.allclose(alg(f) @ Jop(alg(g)), Jop(alg(g)) @ alg(f)) for f in fs for g in fs))
say("first order condition: %s" % all(np.allclose((D @ alg(f) - alg(f) @ D) @ Jop(alg(g)), Jop(alg(g)) @ (D @ alg(f) - alg(f) @ D)) for f in fs for g in fs))
# Pauli magic model of S_4^+: u_rs = Proj(W_r x W_s^*), W the Pauli/Weyl basis of M_2, x in SU_2
W = [np.array(m, complex) for m in ([[1, 0], [0, 1]], [[0, 1], [1, 0]], [[1, 0], [0, -1]], [[0, 1], [-1, 0]])]
a, b = (rng.normal(size=2) + 1j * rng.normal(size=2)); n = np.sqrt(abs(a) ** 2 + abs(b) ** 2); a, b = a / n, b / n
x = np.array([[a, -b.conjugate()], [b, a.conjugate()]])
def proj(M):
    v = M.reshape(-1); v = v / np.linalg.norm(v); return np.outer(v, v.conj())
u = [[proj(W[r] @ x @ W[s].conj().T) for s in range(N)] for r in range(N)]
K = 4
say("magic: %s, noncommutative coefficients: %s" % (all(np.allclose(sum(u[r][s] for s in range(N)), I(K)) and np.allclose(sum(u[s][r] for s in range(N)), I(K)) for r in range(N)), not np.allclose(u[0][1] @ u[1][2], u[1][2] @ u[0][1])))
def E(i, j):
    M = np.zeros((N, N)); M[i, j] = 1; return M
# U = u_13 u_23 on L (x) L (x) K, then tensored with C^2 in the middle
U13 = sum(kron(kron(E(i, j), I(N)), u[i][j]) for i in range(N) for j in range(N))
U23 = sum(kron(kron(I(N), E(i, j)), u[i][j]) for i in range(N) for j in range(N))
Uf = U13 @ U23
say("U unitary: %s" % np.allclose(Uf @ Uf.conj().T, I(N * N * K)))
def lift(A):   # operator on L (x) L (x) C^2 -> operator on L (x) L (x) C^2 (x) K, C^2 moved last before K
    return kron(A, I(K))
def liftU(V):  # operator on L (x) L (x) K -> on L (x) L (x) C^2 (x) K
    V4 = V.reshape(N * N, K, N * N, K)
    out_ = np.zeros((N * N, 2, K, N * N, 2, K), complex)
    for c in range(2): out_[:, c, :, :, c, :] = V4
    return out_.reshape(N * N * 2 * K, N * N * 2 * K)
UU = liftU(Uf)
say("U commutes with D (x) 1: %s, with gamma (x) 1: %s" % (np.allclose(UU @ lift(D), lift(D) @ UU), np.allclose(UU @ lift(gam), lift(gam) @ UU)))
ok = True
for j in range(N):
    lhs = UU @ lift(alg(np.eye(N)[j])) @ UU.conj().T
    rhs = sum(kron(kron(kron(np.diag(np.eye(N)[i]), I(N)), I(2)), u[i][j]) for i in range(N))
    ok = ok and np.allclose(lhs, rhs)
say("U implements the coaction delta_j -> sum_i delta_i (x) u_ij on the first leg: %s" % ok)
# real structure compatibility: flipping the 2 legs and taking adjoints of the coefficients gives back U
U4 = Uf.reshape(N, N, K, N, N, K)
Ut = np.transpose(U4, (1, 0, 2, 4, 3, 5))                     # flip of the 2 legs, in and out
Ut = np.conj(np.transpose(Ut, (0, 1, 5, 3, 4, 2)))            # adjoint of each coefficient
say("compatibility with J: flip of the legs plus adjoint of the coefficients fixes U: %s" % np.allclose(Ut.reshape(N * N * K, N * N * K), Uf))
say("done")
