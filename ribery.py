# Central heat rates on S_N^+, with exact integer arithmetic. With d_k = D_k(N) and s_k = D_k'(N), we have
#   d_0 = 1, d_1 = N-1, d_{k+1} = (N-2) d_k - d_{k-1},   s_0 = 0, s_1 = 1, s_{k+1} = d_k + (N-2) s_k - s_{k-1}.
# (1) endpoint rates c_k = 2(N-1) s_k/d_k: c_1 = 2, c_k = k(k+1) at N = 4, c_2 = 2(N-1)(2N-3)/(N^2-3N+1),
#     and the variance of the first character has zero derivative at t = 0;
# (2) the inequalities |D_k'(x)| <= D_k'(N) and D_k(x) <= D_k(N) on a grid of [0,N], with D_k(x), D_k'(x) computed by
#     the same recurrence (an illustration, the proof being the Chebyshev argument);
# (3) the sphere threshold A* = 2 eta sqrt(N(N-4)) / ((N-1) c_1), eta = arccosh((N-2)/2), and the logarithms of
#     the summands d_k^2 exp(-A c_k) for the endpoint rates, computed exactly, at A = 0.9 A*, A*, 1.1 A*.
import math
from fractions import Fraction
out = open("ribery-output.txt", "w")
def say(s):
    print(s, flush=True); out.write(s + "\n"); out.flush()
def exact(N, K):
    d, s = [1, N - 1], [0, 1]
    while len(d) <= K:
        d.append((N - 2) * d[-1] - d[-2]); s.append(d[-2] + (N - 2) * s[-1] - s[-2])
    return d, s
def at(x, K):
    d, s = [1.0, x - 1.0], [0.0, 1.0]
    while len(d) <= K:
        d.append((x - 2) * d[-1] - d[-2]); s.append(d[-2] + (x - 2) * s[-1] - s[-2])
    return d, s
d4, s4 = exact(4, 8)
say("N = 4: 6 s_k/d_k for k = 0..7: %s (expect k(k+1))" % [str(Fraction(6 * s4[k], d4[k])) for k in range(8)])
for N in (5, 6, 9, 12):
    K = 60
    d, s = exact(N, K)
    c = [Fraction(2 * (N - 1) * s[k], d[k]) for k in range(K + 1)]
    dd = N - 1
    dV = -dd * c[1] - d[2] * c[2] + 2 * dd * dd * c[1]
    f2 = Fraction(2 * (N - 1) * (2 * N - 3), N * N - 3 * N + 1)
    okd = okv = True
    for t in range(4001):
        x = N * t / 4000; dx, sx = at(x, 30)
        okd &= all(abs(sx[k]) <= s[k] * (1 + 1e-12) + 1e-9 for k in range(31))
        okv &= all(dx[k] <= d[k] * (1 + 1e-12) + 1e-9 for k in range(31))
    slope = 2 * (N - 1) / math.sqrt(N * (N - 4))
    say("N = %2d: c_1 = %s, c_2 = %s = formula %s, V'(0) = %s, |D_k'| <= D_k'(N) and D_k <= D_k(N) on [0,N], k <= 30: %s %s" % (
        N, c[1], c[2], f2, dV, okd, okv))
    say("        c_k/k for k = 10, 20, 40, 60: %.8f %.8f %.8f %.8f, limit 2(N-1)/sqrt(N(N-4)) = %.8f" % (
        float(c[10]) / 10, float(c[20]) / 20, float(c[40]) / 40, float(c[60]) / 60, slope))
    eta = math.acosh((N - 2) / 2)
    Astar = 2 * eta * math.sqrt(N * (N - 4)) / ((N - 1) * 2)
    row = lambda A: " ".join("%.4f" % (2 * math.log(d[k]) - A * float(c[k])) for k in (20, 40, 60))
    say("        A* = %.4f; log of d_k^2 exp(-A c_k) at k = 20, 40, 60:  A = 0.9 A*: %s;  A = A*: %s;  A = 1.1 A*: %s" % (
        Astar, row(0.9 * Astar), row(Astar), row(1.1 * Astar)))
    say("        predicted limit at A = A*: log(exp(eta sqrt(N/(N-4)))/(N-4)) = %.4f" % (eta * math.sqrt(N / (N - 4)) - math.log(N - 4)))
say("done")
out.close()
