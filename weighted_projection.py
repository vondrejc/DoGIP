from fenics import *
import numpy as np
from scipy.sparse import csr_matrix, linalg

# PARAMETERS
dim=2 # dimension of the problem
N=3 # no. of elements
pol_order=1 # polynomial order of FEM approximation

# creating MESH, defining MATERIAL and SOURCE
if dim==2: 
    mesh = UnitSquareMesh(N, N)
    A = Expression("1+10*16*x[0]*(1-x[0])*x[1]*(1-x[1])", degree=4)
    f = Expression("x[0]*x[0]*x[1]", degree=3)
elif dim==3:
    mesh = UnitCubeMesh(N, N, N)
    A = Expression("1+100*x[0]*(1-x[0])*x[1]*x[2]", degree=4)
    f = Expression("(1-x[0])*x[1]*x[2]", degree=3)

## standard approach with FEniCS #############################################
V = FunctionSpace(mesh, "CG", pol_order) # original FEM space
u = TrialFunction(V)
v = TestFunction(V)
u_fenics = Function(V)
solve(A*u*v*dx==A*f*v*dx, u_fenics)

## DoGIP - double-grid integration with interpolation-projection #############
W = FunctionSpace(mesh, "CG", 2*pol_order) # double-grid space
w = TestFunction(W)
Adiag = assemble(A*w*dx).get_local() # diagonal matrix of material coefficients
b = assemble(A*f*v*dx) # vector of right-hand side

# assembling interpolation-projection matrix B
B = np.zeros([V.dim(), W.dim()])
for ii in range(V.dim()):
    bfun = Function(V)
    bfun.vector()[ii] = 1.
    bfund = interpolate(bfun, W)
    B[ii] = bfund.vector().get_local()

np.putmask(B, np.abs(B)<1e-14, 0) # removing the values that are close to zero
B = csr_matrix(B) # changing to the compressed sparse row (CSR) format
B.eliminate_zeros()
B=B.T

## linear solver on double grid, standard
def Afun(x):
    return B.T.dot(Adiag*B.dot(x))

Alinoper = linalg.LinearOperator((V.dim(), V.dim()), matvec=Afun, dtype=np.float)
x, info = linalg.cg(Alinoper, b.get_local(), x0=np.zeros(V.dim()),
                    tol=1e-8, maxiter=1e3, callback=None)

print('difference FEniCS vs DoGIP: {}'.format(np.linalg.norm(u_fenics.vector().get_local()-x)))
print('END')
