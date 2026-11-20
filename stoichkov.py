# Positivity test for the Casimir heat weights on S_N^+: a central state with phi(chi_k) = d_k exp(-t k(k+1))
# would give the first character chi = chi_1 the variance V(t) = 1 + d e^{-2t} + (d^2-d-1) e^{-6t} - d^2 e^{-4t},
# with d = N-1, which must be >= 0. We check V'(0) = -2N(N-4) and find negative values of V(t).
# Also: large N behavior of the formal partition function Z_N(g,A) = sum_k d_k^{2-2g} e^{-A k(k+1)} at A = 1.
import math
out = open("stoichkov-output.txt", "w")
def say(s):
    print(s, flush=True); out.write(s + "\n"); out.flush()
def V(N, t):
    d = N - 1
    return 1 + d * math.exp(-2 * t) + (d * d - d - 1) * math.exp(-6 * t) - d * d * math.exp(-4 * t)
for N in range(4, 13):
    d = N - 1
    dV0 = -2 * d - 6 * (d * d - d - 1) + 4 * d * d
    h = 1e-6
    num = (V(N, h) - V(N, 0)) / h
    tmin = min((V(N, k / 1000), k / 1000) for k in range(1, 2001))
    say("N = %2d: V(0) = %g, V'(0) = %d = -2N(N-4) = %d (numerical %.3f), min of V on (0,2]: %.4f at t = %.3f" % (
        N, V(N, 0), dV0, -2 * N * (N - 4), num, tmin[0], tmin[1]))
def dims(N, K):
    ds = [1, N - 1]
    while len(ds) < K: ds.append((N - 2) * ds[-1] - ds[-2])
    return ds
say("formal partition functions at A = 1, as N grows:")
for N in (4, 5, 9, 20, 50, 100):
    ds = dims(N, 40)
    Z = {g: sum(float(dk) ** (2 - 2 * g) * math.exp(-k * (k + 1)) for k, dk in enumerate(ds)) for g in (0, 1, 2, 3)}
    say("N = %3d: genus 0: %.6g, genus 1: %.6f, genus 2: %.6f, genus 3: %.6f" % (N, Z[0], Z[1], Z[2], Z[3]))
say("done")
out.close()
