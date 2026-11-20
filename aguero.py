# The quark bound: if T on color x generations commutes with 1 x Y_u Y_u^* and 1 x Y_d Y_d^*, with simple spectra,
# then T lies in M_3(C) x C^c, with c the number of connected components of the support graph of the CKM matrix,
# so at most 3c mutually orthogonal nonzero projections T_f can occur. We compute the dimension of this commutant
# for a connected (generic) mixing matrix, a block mixing matrix (c = 2) and a monomial one (c = 3).
import numpy as np
out = open("aguero-output.txt", "w")
def say(s):
    print(s, flush=True); out.write(s + "\n"); out.flush()
rng = np.random.default_rng(5)
def haar(n):
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)); q, r = np.linalg.qr(A); return q * (np.diag(r) / abs(np.diag(r)))
def commutant_dim(ops, n):
    rows = [np.kron(O, np.eye(n)) - np.kron(np.eye(n), O.T) for O in ops]
    return n * n - np.linalg.matrix_rank(np.vstack(rows), tol=1e-9)
Uu = haar(3); Hu = Uu @ np.diag([0.2, 1.0, 3.0]) @ Uu.conj().T
for name, V in [("connected CKM", haar(3)), ("block CKM, c = 2", np.block([[haar(2), np.zeros((2, 1))], [np.zeros((1, 2)), np.eye(1)]])), ("monomial CKM, c = 3", np.eye(3)[[1, 2, 0]])]:
    Hd = Uu @ V @ np.diag([0.5, 1.5, 2.5]) @ V.conj().T @ Uu.conj().T
    ops = [np.kron(np.eye(3), Hu), np.kron(np.eye(3), Hd)]
    dim = commutant_dim(ops, 9)
    say("%s: commutant of 1 x Y_uY_u^*, 1 x Y_dY_d^* in M_3 x M_3 has dimension %d = 9c, so at most %d orthogonal projections" % (name, dim, 3 * dim // 9))
say("done")
out.close()
