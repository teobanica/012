# Mixing matrices: (1) E_D E_{UDU^*} = E_{C1} iff |U_ij|^2 = 1/n, tested on random unitaries, Fourier matrices
# and the identity; with the intersection version, the identity gives a commuting square over D itself;
# (2) the tribimaximal matrix with U_31 = +1/sqrt 6, from an earlier version, is not unitary, while the corrected one is;
# (3) the Hadamard distance delta(U) = sum |U_ij|^4 - 1 for some reference matrices, and the TM2 value of
#     sin^2 theta_12 = 1/(3 cos^2 theta_13).
import numpy as np
out = open("overmars-output.txt", "w")
def say(s):
    print(s, flush=True); out.write(s + "\n"); out.flush()
def ED(X):                     # conditional expectation onto the diagonal algebra
    return np.diag(np.diag(X))
def EU(U, X):                  # conditional expectation onto U D U^*
    return U @ ED(U.conj().T @ X @ U) @ U.conj().T
def EC(X):
    n = X.shape[0]; return np.trace(X) / n * np.eye(n)
def test(U, label, trials=20, rng=np.random.default_rng(5)):
    n = U.shape[0]; dev = 0
    for _ in range(trials):
        X = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        dev = max(dev, np.abs(ED(EU(U, X)) - EC(X)).max())
    flat = np.allclose(np.abs(U) ** 2, 1 / n)
    say("%s: E_D E_UDU* = E_C1 %s, flat moduli %s" % (label, dev < 1e-10, flat))
w = np.exp(2j * np.pi / 3)
F3 = np.array([[w ** (i * j) for j in range(3)] for i in range(3)]) / np.sqrt(3)
F4 = np.array([[1j ** (i * j) for j in range(4)] for i in range(4)]) / 2
rng = np.random.default_rng(1)
A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)); Ur, _ = np.linalg.qr(A)
test(F3, "F_3/sqrt(3)"); test(F4, "F_4/2"); test(Ur, "random unitary"); test(np.eye(3), "identity")
X = np.random.default_rng(2).normal(size=(3, 3))
say("identity, intersection version: E_D E_D = E_D, so the square over D cap D = D commutes: %s" % np.allclose(ED(ED(X)), ED(X)))
s6, s3, s2 = 1 / np.sqrt(6), 1 / np.sqrt(3), 1 / np.sqrt(2)
printed = np.array([[np.sqrt(2 / 3), s3, 0], [-s6, s3, s2], [s6, s3, -s2]])
fixed = np.array([[np.sqrt(2 / 3), s3, 0], [-s6, s3, s2], [-s6, s3, -s2]])
say("U_TB with U_31 = +1/sqrt 6, as in an earlier version: unitary %s, scalar product of rows 1 and 3 = %.4f" % (np.allclose(printed @ printed.T, np.eye(3)), printed[0] @ printed[2]))
say("corrected U_TB (U_31 = -1/sqrt 6): unitary %s, middle column flat %s" % (np.allclose(fixed @ fixed.T, np.eye(3)), np.allclose(fixed[:, 1] ** 2, 1 / 3)))
delta = lambda U: (np.abs(U) ** 4).sum() - 1
say("Hadamard distance delta(U) = sum |U_ij|^4 - 1: identity %.4f, U_TB %.4f, F_3/sqrt 3 %.4f, the 3 other MUB transitions %s" % (
    delta(np.eye(3)), delta(fixed), delta(F3), [round(float(delta(M)), 4) for M in (np.diag([1, 1, w]) @ F3, np.diag([1, 1, w * w]) @ F3, F3.conj().T @ np.diag([1, 1, w]) @ F3)]))
def hsdefect(U):   # || E_D E_UDU* - E_C1 ||^2 as a linear map on M_n, Hilbert-Schmidt norm
    n = U.shape[0]; basis = [np.zeros((n, n), complex) for _ in range(n * n)]
    for k in range(n * n): basis[k][k // n, k % n] = 1
    M = np.array([(ED(EU(U, X)) - EC(X)).flatten() for X in basis]).T
    return float(np.sum(np.abs(M) ** 2))
for U, lab in ((np.eye(3), "identity"), (fixed, "U_TB"), (F3, "F_3/sqrt 3"), (Ur, "random unitary")):
    say("%s: delta(U) = %.6f, ||E_D E_UDU* - E_C1||_HS^2 = %.6f" % (lab, delta(U), hsdefect(U)))
for s13 in (0.0, 0.022):
    say("TM2: |U_e2|^2 = 1/3 with sin^2 theta_13 = %.3f gives sin^2 theta_12 = %.4f" % (s13, 1 / (3 * (1 - s13))))
say("done")
out.close()
