# The Hesse configuration as an incidence graph: 9 points of F_3^2, 12 affine lines. With I: C^12 -> C^9 the incidence
# matrix and m_l the pointwise product on line functions, check T(v_a x v_c) = I m_l(I^* v_a x I^* v_c)
# = v_a + v_c + v_{-a-c} + delta_ac 1 on all 81 ordered pairs (integer arithmetic), the degrees 4 and 3, the fact that
# the indicators m_l(I^* v_a x I^* v_c), a != c, span the line functions, and the 432 permutations preserving the lines.
import itertools
out = open("haaland-output.txt", "w")
def say(s):
    print(s, flush=True); out.write(s + "\n"); out.flush()
pts = [(i, j) for i in range(3) for j in range(3)]; ix = {p: k for k, p in enumerate(pts)}
lines = set()
for a in pts:
    for dvec in [(0, 1), (1, 0), (1, 1), (1, 2)]:
        lines.add(frozenset(((a[0] + t * dvec[0]) % 3, (a[1] + t * dvec[1]) % 3) for t in range(3)))
lines = sorted(lines, key=lambda l: sorted(l)); say("number of lines: %d" % len(lines))
I = [[1 if pts[a] in l else 0 for l in lines] for a in range(9)]
say("degrees: points on %s lines, lines through %s points" % (sorted(set(sum(r) for r in I)), sorted(set(sum(I[a][l] for a in range(9)) for l in range(12)))))
ok = True; span = []
for a in range(9):
    for c in range(9):
        la = [I[a][l] for l in range(12)]; lc = [I[c][l] for l in range(12)]
        prodl = [x * y for x, y in zip(la, lc)]
        T = [sum(I[r][l] * prodl[l] for l in range(12)) for r in range(9)]
        neg = ix[((-pts[a][0] - pts[c][0]) % 3, (-pts[a][1] - pts[c][1]) % 3)]
        exp = [(r == a) + (r == c) + (r == neg) + (a == c) for r in range(9)]
        ok &= (T == exp)
        if a != c: span.append(tuple(prodl))
say("T(v_a x v_c) = v_a + v_c + v_(-a-c) + delta_ac 1 on all 81 ordered pairs: %s" % ok)
say("the line indicators obtained for a != c: %d distinct, all of the 12 lines: %s" % (len(set(span)), len(set(span)) == 12))
cnt = sum(1 for p in itertools.permutations(range(9)) if all(frozenset(pts[p[ix[q]]] for q in l) in set(lines) for l in lines))
say("permutations of the 9 points mapping lines to lines: %d (expect 432)" % cnt)
say("done")
out.close()
