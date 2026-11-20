# Dimensions d_k of the irreducible representations of S_N^+, N>=4, via d_{k+1}=(N-2)d_k-d_{k-1},
# and the 2-dimensional Yang-Mills quantities Z(Sigma_g,A) = sum_k d_k^(2-2g) exp(-A k(k+1)).
import math
out = open("rijkaard-output.txt", "w")
def say(s):
    print(s, flush=True); out.write(s + "\n"); out.flush()
def dims(N, K):
    d = [1, N - 1]
    while len(d) < K: d.append((N - 2) * d[-1] - d[-2])
    return d
for N in range(4, 10):
    say(f"N={N}: d_0,...,d_6 = {dims(N, 7)}")
F = [0, 1]
for i in range(80): F.append(F[-1] + F[-2])
L = [2, 1]
for i in range(80): L.append(L[-1] + L[-2])
say(f"N=9: d_k = Fibonacci F_(4k+2) for k<15: {dims(9, 15) == [F[4*k+2] for k in range(15)]}")
say(f"N=5: d_k = Lucas L_(2k+1) for k<15: {dims(5, 15) == [L[2*k+1] for k in range(15)]}")
say("")
for g in (2, 3):
    for N in (4, 5, 6, 7, 8, 9):
        K = 10**6 if N == 4 else 300
        if N == 4:
            val = sum((2*k+1.0)**(2-2*g) for k in range(K))
        else:
            val = sum(math.exp((2-2*g)*math.log(x)) for x in dims(N, K))
        line = f"g={g}, N={N}: sum_k d_k^(2-2g) = {val:.12f}"
        if N == 4:
            exact = math.pi**2/8 if g == 2 else math.pi**4/96
            line += f"   [SO_3 value (1-2^(2-2g))zeta(2g-2) = {exact:.12f}]"
        say(line)
    say("")
for N in (4, 5, 9):
    for A in (0.1, 1.0):
        d = dims(N, 300)
        for g in (0, 1):
            val = sum(math.exp((2-2*g)*math.log(x) - A*k*(k+1)) for k, x in enumerate(d))
            say(f"N={N}, A={A}, g={g}: Z = {val:.10g}")
out.close()
