# Independent check of the area tensor on 9 points (Z_3^2), and of its classical symmetry group
import itertools, numpy as np, builtins
_out=open('seedorf-output.txt','w')
def print(*a,**k):
    builtins.print(*a,**k,flush=True); _out.write(' '.join(str(x) for x in a)+'\n'); _out.flush()
w = np.exp(2j*np.pi/3)
pts = [(i,j) for i in range(3) for j in range(3)]
idx = {p:k for k,p in enumerate(pts)}
det = lambda g,h: (g[0]*h[1]-g[1]*h[0]) % 3
add = lambda g,h: ((g[0]+h[0])%3,(g[1]+h[1])%3)
sub = lambda g,h: ((g[0]-h[0])%3,(g[1]-h[1])%3)
d = lambda g,h: w**det(g,h)
chi = np.array([[d(g,r) for r in pts] for g in pts])          # chi_g(r)
# b on character basis: b(chi_g,chi_h) = d(g,h) chi_{g+h}; build tensor in point basis
Cinv = np.linalg.inv(chi.T)                                   # delta_a = sum_g Cinv[g,a] chi_g ... solve below
# express delta_a in chi basis: chi.T[r,g] coefficients => delta = chi.T^{-1}
B = np.zeros((9,9,9),complex)  # B[a,c,r] = b(delta_a,delta_c)(r)
coef = np.linalg.inv(chi.T)    # coef[g,a]: delta_a = sum_g coef[g,a] chi_g
for a in range(9):
  for c in range(9):
    v = np.zeros(9,complex)
    for gi,g in enumerate(pts):
      for hi,h in enumerate(pts):
        v += coef[gi,a]*coef[hi,c]*d(g,h)*chi[idx[add(g,h)]]
    B[a,c] = v
F = np.array([[[ (1/9)*d(sub(pts[a],pts[r]),sub(pts[c],pts[r]))**-1 for r in range(9)] for c in range(9)] for a in range(9)])
print("point formula matches:", np.allclose(B,F))
# associativity
mult = lambda x,y: np.einsum('a,c,acr->r',x,y,B)
E = np.eye(9)
ok = all(np.allclose(mult(mult(E[a],E[b]),E[c]), mult(E[a],mult(E[b],E[c]))) for a in range(9) for b in range(9) for c in range(9))
print("associative:", ok, " unit = constant 1:", np.allclose(mult(np.ones(9),E[4]),E[4]))
# commutative? (should be noncommutative: b is m^op-type twisted product)
print("noncommutative:", not np.allclose(B, B.transpose(1,0,2)))
# classical permutations of the 9 points preserving b
cnt = 0
for perm in itertools.permutations(range(9)):
  P = np.array(perm)
  if np.allclose(B[np.ix_(P,P,P)] , B): cnt += 1
print("permutations of 9 points preserving b:", cnt, "(expect 216 = 9*24)")
# cubic invariants: S_N^+ vs SU_3 adjoint via NC(k) and k! counts
from math import comb
NC = {0:1,1:1,2:2,3:5}
print("Fix(r1^3) S_9^+:", sum((-1)**(3-j)*comb(3,j)*NC[j] for j in range(4)),
      " SU_3 adjoint:", sum((-1)**(3-j)*comb(3,j)*[1,1,2,6][j] for j in range(4)))
# orientation: permutations sending b to its complex conjugate (orientation reversing), and the total
cntbar = 0; cntre = 0
Re = B.real
for perm in itertools.permutations(range(9)):
  P = np.array(perm); Bp = B[np.ix_(P,P,P)]
  if np.allclose(Bp, B.conj()): cntbar += 1
  if np.allclose(Bp.real, Re): cntre += 1
print("permutations sending b to conj(b):", cntbar, " total preserving b up to conjugation:", 216+cntbar, "(Aut of Delta(27) has order 432)")
print("permutations preserving Re(b):", cntre)
