# The algebra B_F = C + C + M_2(C) + M_3(C) acting on one generation of fermions, H_f + conj(H_f) = C^32,
# the trace Tr_H induced on B_F, and the operator m m^* for this trace, compared with the canonical trace.
import numpy as np
out = open("maradona-output.txt", "w")
def say(s):
    print(s, flush=True); out.write(s + "\n"); out.flush()

# basis of H_f + conj(H_f): particles first, then antiparticles
labels = ["nuR", "eR", "nuL", "eL"] + [q + c for q in ["uR", "dR", "uL", "dL"] for c in "123"]
labels += ["a" + x for x in labels]
idx = {l: i for i, l in enumerate(labels)}
D = len(labels)
say("dimension of one generation: %d" % D)

def rep(lam, lamb, q, m):
    # lam acts on nuR,uR and on the antileptons, lamb = second copy of C acts on eR,dR,
    # q in M_2 on the left doublets, m in M_3 on the color of the antiquarks
    P = np.zeros((D, D), dtype=complex)
    for l in ["nuR"] + ["uR" + c for c in "123"] + ["anuR", "aeR", "anuL", "aeL"]:
        P[idx[l], idx[l]] = lam
    for l in ["eR"] + ["dR" + c for c in "123"]:
        P[idx[l], idx[l]] = lamb
    for (a, b) in [("nuL", "eL")] + [("uL" + c, "dL" + c) for c in "123"]:
        ia, ib = idx[a], idx[b]
        P[np.ix_([ia, ib], [ia, ib])] = q
    for f in ["auR", "adR", "auL", "adL"]:
        ii = [idx[f + c] for c in "123"]
        P[np.ix_(ii, ii)] = m
    return P

Z2, Z3 = np.zeros((2, 2)), np.zeros((3, 3))
E2 = np.zeros((2, 2)); E2[0, 0] = 1
E3 = np.zeros((3, 3)); E3[0, 0] = 1
w = {"C (lambda)": np.trace(rep(1, 0, Z2, Z3)).real, "C (conj lambda)": np.trace(rep(0, 1, Z2, Z3)).real,
     "M_2 (min proj)": np.trace(rep(0, 0, E2, Z3)).real, "M_3 (min proj)": np.trace(rep(0, 0, Z2, E3)).real}
for k in w: say("Tr_H of %s = %d" % (k, w[k]))
say("check: total = %d" % (w["C (lambda)"] + w["C (conj lambda)"] + 2 * w["M_2 (min proj)"] + 3 * w["M_3 (min proj)"]))

# the abstract algebra B = C + C + M_2 + M_3, basis of matrix units, with a trace given by the weights t_i
ns = [1, 1, 2, 3]
basis = [(b, a, c) for b, n in enumerate(ns) for a in range(n) for c in range(n)]
def mmstar(t):
    # orthonormal basis f = e / sqrt(t_block), m(f_x f_y) expanded in f, then m m^* = M M^dagger
    K = len(basis); M = np.zeros((K, K * K))
    for x, (b1, a1, c1) in enumerate(basis):
        for y, (b2, a2, c2) in enumerate(basis):
            if b1 == b2 and c1 == a2:
                z = basis.index((b1, a1, c2))
                M[z, x * K + y] = np.sqrt(t[b1]) / (np.sqrt(t[b1]) * np.sqrt(t[b1]))
    return M @ M.T
names = ["C (lambda)", "C (conj lambda)", "M_2", "M_3"]
tot = 32.0
tphys = [w["C (lambda)"] / tot, w["C (conj lambda)"] / tot, w["M_2 (min proj)"] / tot, w["M_3 (min proj)"] / tot]
tcan = [n / 15.0 for n in ns]
for nm, t in [("physical trace Tr_H/32", tphys), ("canonical trace", tcan)]:
    say(nm + ": minimal projection weights " + str([round(float(x), 4) for x in t]))
    MM = mmstar(t)
    for b, n in enumerate(ns):
        ii = [i for i, B in enumerate(basis) if B[0] == b]
        blk = MM[np.ix_(ii, ii)]
        say("   block %s: m m^* = %.4f * id (scalar: %s), n_i/t_i = %.4f" % (names[b], blk[0, 0], np.allclose(blk, blk[0, 0] * np.eye(len(ii))), ns[b] / t[b]))
# the Heisenberg group of order 27, acting by clock and shift on the color index of the antiquarks, and by the
# conjugate representation on the quarks: decomposition of one generation into irreducible representations, via the central character
w = np.exp(2j * np.pi / 3)
Xs = np.roll(np.eye(3), 1, axis=0); Zc = np.diag([1, w, w * w])
def heis(A):   # action of A in U_3 on H_f + conj(H_f)
    M = np.eye(D, dtype=complex)
    for q in ["uR", "dR", "uL", "dL"]:
        ii = [idx[q + c] for c in "123"]; M[np.ix_(ii, ii)] = A.conj()
        jj = [idx["a" + q + c] for c in "123"]; M[np.ix_(jj, jj)] = A
    return M
cen = heis(w * np.eye(3))
ev = np.round(np.diag(cen), 6)
n1 = int(np.sum(np.isclose(ev, 1))); nw = int(np.sum(np.isclose(ev, w))); nw2 = int(np.sum(np.isclose(ev, w * w)))
say("Heisenberg group on one generation: central character 1 on %d dims, w on %d dims, w^2 on %d dims" % (n1, nw, nw2))
say("hence, for one generation: %d copies of trivial-center representations (the leptons), %d copies of 3 (antiquarks) and %d copies of 3bar (quarks)" % (n1, nw // 3, nw2 // 3))
say("check of the Heisenberg relation ZX = w XZ for clock and shift: %s" % np.allclose(Zc @ Xs, w * Xs @ Zc))
say("done")
