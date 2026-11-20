# A 2-sided spatial implementer for the Standard Model coactions, tested with the Haar space of C[S_3].
# H_F: one generation (32 states, as in maradona.py) times 3 generations, with Yukawa matrices having distinct masses
# and generation mixing, and a Majorana term. Coefficient algebra M = C[S_3] acting on H_Q = l^2(S_3), j(a) = a^*.
# Weak phase: a unitary z in M. Color: v = (F x 1) diag(g_1,g_2,g_3) (F^* x 1), a unitary in M_3(M) with noncommuting
# entries and sum_i v_ij v_ik^* = delta_jk, giving beta(b) = v (b x 1) v^*. The implementer is
#   W = L_{h_s} V_R on particles of species s, W = R_{h_s^*} V_L on antiparticles, V_L = (L_{v_ij}), V_R = (R_{v_ij^*}),
# with h_nu = h_u = 1, h_e = h_d = z^*. We check: W unitary, WK = KW, W implements the coaction on the 15 matrix units
# of B_F, W commutes with D_F and gamma_F, and the partial trace (Tr_H x id)(W (X x 1) W^*) = Tr(X) 1.
import itertools, numpy as np
out = open("vanbasten-output.txt", "w")
def say(s):
    print(s, flush=True); out.write(s + "\n"); out.flush()
rng = np.random.default_rng(21)
# the group S_3 and its regular representations
G = sorted(itertools.permutations(range(3))); n = len(G); gi = {g: i for i, g in enumerate(G)}
mul = lambda a, b: tuple(a[b[i]] for i in range(3)); inv = lambda a: tuple(sorted(range(3), key=lambda i: a[i]))
def L(a):   # a = dict g -> coefficient, left multiplication on l^2(G)
    M = np.zeros((n, n), complex)
    for g, c in a.items():
        for h in G: M[gi[mul(g, h)], gi[h]] += c
    return M
def R(a):
    M = np.zeros((n, n), complex)
    for g, c in a.items():
        for h in G: M[gi[mul(h, g)], gi[h]] += c
    return M
star = lambda a: {inv(g): np.conj(c) for g, c in a.items()}
prod = lambda a, b: {k: v for k, v in _prod(a, b).items()}
def _prod(a, b):
    c = {}
    for g, x in a.items():
        for h, y in b.items():
            k = mul(g, h); c[k] = c.get(k, 0) + x * y
    return c
jmat = np.zeros((n, n))
for g in G: jmat[gi[inv(g)], gi[g]] = 1          # j = jmat o complex conjugation
s1, s2 = (1, 0, 2), (0, 2, 1)
z = {s1: 1.0}                                    # weak phase, a unitary of C[S_3]
gs = [{(0, 1, 2): 1.0}, {s1: 1.0}, {mul(s1, s2): 1.0}]
w3 = np.exp(2j * np.pi / 3); F = np.array([[w3 ** (i * k) for k in range(3)] for i in range(3)]) / np.sqrt(3)
v = [[{} for _ in range(3)] for _ in range(3)]
for i in range(3):
    for j in range(3):
        acc = {}
        for k in range(3):
            for g, c in gs[k].items(): acc[g] = acc.get(g, 0) + F[i, k] * c * np.conj(F[j, k])
        v[i][j] = acc
orth = max(np.abs(sum(L(_prod(v[i][j], star(v[i][k]))) for i in range(3)) - (j == k) * np.eye(n)).max() for j in range(3) for k in range(3))
comm = np.abs(L(v[0][0]) @ L(v[0][1]) - L(v[0][1]) @ L(v[0][0])).max()
say("color unitary: sum_i v_ij v_ik^* = delta_jk up to %.1e; entries noncommuting (commutator %.3f)" % (orth, comm))
# one generation, as in maradona.py, times 3 generations
labels = ["nuR", "eR", "nuL", "eL"] + [q + c for q in ["uR", "dR", "uL", "dL"] for c in "123"]
labels += ["a" + x for x in labels]
ix = {l: i for i, l in enumerate(labels)}; d1 = len(labels); dF = 3 * d1
def emb(M1, Mg=np.eye(3)): return np.kron(M1, Mg)        # internal (32) x generations (3)
def unit(a, b):
    E = np.zeros((d1, d1), complex); E[ix[a], ix[b]] = 1; return E
def rep(lam, lamb, q, m):
    P = np.zeros((d1, d1), complex)
    for l in ["nuR"] + ["uR" + c for c in "123"] + ["anuR", "aeR", "anuL", "aeL"]: P[ix[l], ix[l]] = lam
    for l in ["eR"] + ["dR" + c for c in "123"]: P[ix[l], ix[l]] = lamb
    for (a, b) in [("nuL", "eL")] + [("uL" + c, "dL" + c) for c in "123"]:
        ii = [ix[a], ix[b]]; P[np.ix_(ii, ii)] = q
    for f in ["auR", "adR", "auL", "adL"]:
        ii = [ix[f + c] for c in "123"]; P[np.ix_(ii, ii)] = m
    return emb(P)
def yuk(masses):
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)); U1, _ = np.linalg.qr(A)
    B = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)); U2, _ = np.linalg.qr(B)
    return U1 @ np.diag(masses) @ U2
Y = {"nu": yuk([0.1, 0.2, 0.3]), "e": yuk([0.5, 1.1, 1.7]), "u": yuk([0.4, 1.3, 2.9]), "d": yuk([0.6, 0.9, 2.2])}
YR = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)); YR = YR + YR.T          # symmetric Majorana matrix
Dp = np.zeros((dF, dF), complex)
def put(Mat, a, b, Yg):      # Mat[a-block, b-block] = Yg on generations, and the adjoint
    Mat += np.kron(unit(a, b), Yg) + np.kron(unit(b, a), Yg.conj().T)
for sp, (r, l) in {"nu": ("nuR", "nuL"), "e": ("eR", "eL")}.items(): put(Dp, l, r, Y[sp])
for c in "123":
    put(Dp, "uL" + c, "uR" + c, Y["u"]); put(Dp, "dL" + c, "dR" + c, Y["d"])
# antiparticle part: conjugate of the particle part, then the Majorana term between nuR and its antiparticle
half = d1 // 2
Pp = np.zeros((d1, d1)); Pp[:half, :half] = np.eye(half)
S = Dp.copy()
Dfull = S.copy()
for a in range(half):
    for b in range(half):
        Dfull[np.ix_(range(3 * (a + half), 3 * (a + half) + 3), range(3 * (b + half), 3 * (b + half) + 3))] = \
            S[np.ix_(range(3 * a, 3 * a + 3), range(3 * b, 3 * b + 3))].conj()
Dfull += np.kron(unit("anuR", "nuR"), YR) + np.kron(unit("nuR", "anuR"), YR.conj().T)
gam1 = np.zeros(d1)
for l in labels:
    base = l[1:] if l.startswith("a") else l
    left = base.endswith("L") or (len(base) > 1 and base[-2] == "L")
    gam1[ix[l]] = (1 if left else -1) * (1 if not l.startswith("a") else -1)
gam = emb(np.diag(gam1))
Jperm1 = np.zeros((d1, d1))
for l in labels[:half]: Jperm1[ix["a" + l], ix[l]] = 1; Jperm1[ix[l], ix["a" + l]] = 1
Jperm = emb(Jperm1)
say("finite geometry: D_F self-adjoint %s, J D_F = D_F J %s, J gamma = -gamma J %s, D_F gamma = -gamma D_F %s" % (
    np.allclose(Dfull, Dfull.conj().T), np.allclose(Jperm @ Dfull.conj(), Dfull @ Jperm), np.allclose(Jperm @ gam, -gam @ Jperm), np.allclose(Dfull @ gam, -gam @ Dfull)))
# the implementer W on H_F x H_Q
I6 = np.eye(n)
def block(states, op):        # op acting on the H_Q leg, for the given internal states, all generations
    P = np.zeros((d1, d1)); 
    for st in states: P[ix[st], ix[st]] = 1
    return np.kron(emb(P), op)
species = {"nu": ["nuR", "nuL"], "e": ["eR", "eL"]}
h = {"nu": {(0, 1, 2): 1.0}, "u": {(0, 1, 2): 1.0}, "e": star(z), "d": star(z)}
W = np.zeros((dF * n, dF * n), complex)
for sp, sts in species.items():
    W += block(sts, L(h[sp])) + block(["a" + x for x in sts], R(star(h[sp])))
VR = [[R(star(v[i][j])) for j in range(3)] for i in range(3)]
VL = [[L(v[i][j]) for j in range(3)] for i in range(3)]
for sp, qs in {"u": ["uR", "uL"], "d": ["dR", "dL"]}.items():
    for part, Vm, hop in ((False, VR, L(h[sp])), (True, VL, R(star(h[sp])))):
        for q in qs:
            for i in range(3):
                for j in range(3):
                    a = ("a" if part else "") + q + "123"[i]; b = ("a" if part else "") + q + "123"[j]
                    W += np.kron(emb(unit(a, b)), hop @ Vm[i][j])
N = dF * n
say("W unitary: %s" % np.allclose(W @ W.conj().T, np.eye(N)))
Kmat = np.kron(Jperm, jmat)
say("W K = K W: %s" % np.allclose(Kmat @ W.conj() @ Kmat.T, W))
# the coaction on the 15 matrix units
Z2, Z3 = np.zeros((2, 2)), np.zeros((3, 3))
def e2(a, b): E = np.zeros((2, 2)); E[a, b] = 1; return E
def e3(a, b): E = np.zeros((3, 3)); E[a, b] = 1; return E
checks = []
checks.append((rep(1, 0, Z2, Z3), np.kron(rep(1, 0, Z2, Z3), I6)))
checks.append((rep(0, 1, Z2, Z3), np.kron(rep(0, 1, Z2, Z3), I6)))
for a in range(2):
    for b in range(2):
        img = np.kron(rep(0, 0, e2(a, b), Z3), I6 if a == b else (L(z) if (a, b) == (0, 1) else L(star(z))))
        checks.append((rep(0, 0, e2(a, b), Z3), img))
for i in range(3):
    for j in range(3):
        img = sum(np.kron(rep(0, 0, Z2, e3(k, l)), L(_prod(v[k][i], star(v[l][j])))) for k in range(3) for l in range(3))
        checks.append((rep(0, 0, Z2, e3(i, j)), img))
dev = max(np.abs(W @ np.kron(Pa, I6) @ W.conj().T - img).max() for Pa, img in checks)
say("W implements the coaction on all %d matrix units of B_F: max deviation %.1e" % (len(checks), dev))
say("W commutes with D_F x 1: %s, with gamma_F x 1: %s" % (np.allclose(W @ np.kron(Dfull, I6), np.kron(Dfull, I6) @ W), np.allclose(W @ np.kron(gam, I6), np.kron(gam, I6) @ W)))
pt = 0
for _ in range(5):
    X = rng.normal(size=(dF, dF)) + 1j * rng.normal(size=(dF, dF))
    T = (W @ np.kron(X, I6) @ W.conj().T).reshape(dF, n, dF, n)
    E = np.einsum('aiaj->ij', T)
    pt = max(pt, np.abs(E - np.trace(X) * I6).max())
say("partial trace (Tr_H x id)(W (X x 1) W^*) = Tr(X) 1 for random X: max deviation %.1e" % pt)
say("done")
out.close()
