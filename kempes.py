# Weyl ray matrix models pi_x(p_rs) = Proj(W_r x W_s^*) on M_3 (Hilbert-Schmidt): magic, and is the area tensor b an intertwiner?
import numpy as np, itertools
rng=np.random.default_rng(1)
n=3; w=np.exp(2j*np.pi/3)
Z=np.diag([w**k for k in range(n)]); S=np.roll(np.eye(n),-1,axis=0)   # (S f)(z)=f(z+1)
pts=[(i,j) for i in range(3) for j in range(3)]
W={g: np.linalg.matrix_power(Z,g[0])@np.linalg.matrix_power(S,g[1]) for g in pts}
det=lambda g,h:(g[0]*h[1]-g[1]*h[0])%3
sub=lambda g,h:((g[0]-h[0])%3,(g[1]-h[1])%3)
def B(conj=False):
    T=np.zeros((9,9,9),complex)
    for a,c,r in itertools.product(range(9),repeat=3):
        T[a,c,r]=(1/9)*w**(-det(sub(pts[a],pts[r]),sub(pts[c],pts[r])))
    return T.conj() if conj else T
def model(x):
    P={}
    for r in range(9):
        for s in range(9):
            v=(W[pts[r]]@x@W[pts[s]].conj().T).flatten()/np.sqrt(n)
            P[r,s]=np.outer(v,v.conj())
    return P
def magic(P):
    return all(np.allclose(sum(P[r,s] for s in range(9)),np.eye(9)) and np.allclose(sum(P[s,r] for s in range(9)),np.eye(9)) for r in range(9))
def intertwines(P,T):
    # sum_{a,c} T[a,c,r] P[a,i] P[c,j] == sum_k P[r,k] T[i,j,k]
    for r,i,j in itertools.product(range(9),repeat=3):
        L=sum(T[a,c,r]*P[a,i]@P[c,j] for a in range(9) for c in range(9))
        R=sum(P[r,k]*T[i,j,k] for k in range(9))
        if not np.allclose(L,R,atol=1e-9): return False
    return True
out=open("kempes-output.txt","w")
def say(s): print(s,flush=True); out.write(s+"\n")
for trial in range(2):
    A=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)); x,_=np.linalg.qr(A)
    P=model(x)
    say(f"random unitary {trial+1}: magic {magic(P)}; b intertwiner {intertwines(P,B())}; conj(b) intertwiner {intertwines(P,B(True))}")
    # noncommutativity of the model
    nc=max(np.linalg.norm(P[a,b]@P[c,d]-P[c,d]@P[a,b]) for a,b,c,d in itertools.product(range(9),repeat=4) if (a+b+c+d)%7==0)
    say(f"   model noncommutative: {nc>1e-6}")
out.close()
