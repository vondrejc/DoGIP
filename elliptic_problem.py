from fenics import *
import numpy as np
from scipy.sparse import csr_matrix, linalg

# PARAMETERS
dim=2 # dimension of the problem
N=3 # no. of elements
pol_order=1 # polynomial order of FEM approximation

# creating MESH, defining MATERIAL and SOURCE
if dim==2:
    mesh=UnitSquareMesh(N,N)
    A=Expression("1+10*16*x[0]*(1-x[0])*x[1]*(1-x[1])", degree=4)
    f=Expression("80*x[0]*(0.5-x[0])*(1.-x[0])*x[1]*(1.-x[1])", degree=5)
elif dim==3:
    mesh = UnitCubeMesh(N, N, N)
    A=Expression("1+10*16*x[0]*(1-x[0])*(1-x[1])*x[2]", degree=4)
    f=Expression("80*x[0]*(0.5-x[0])*(1.-x[0])*x[1]*(1.-x[1])", degree=5)

## standard approach with FEniCS #############################################
V=FunctionSpace(mesh, "CG", pol_order) # original FEM space
bc=DirichletBC(V, Constant(0.0), lambda x, on_boundary: on_boundary)
u=TrialFunction(V)
v=TestFunction(V)
u_fenics=Function(V) # the vector for storing the solution
solve(A*inner(grad(u), grad(v))*dx==f*v*dx, u_fenics, bc) # solution by FEniCS

## DoGIP - double-grid integration with interpolation-projection #############
W=FunctionSpace(mesh, "DG", 2*(pol_order-1)) # double-grid space
Wvector=VectorFunctionSpace(mesh, "DG", 2*(pol_order-1)) # vector variant of double-grid space
w=TestFunction(W)
Adiag=assemble(A*w*dx).get_local() # diagonal matrix of material coefficients
Adiag_full=np.einsum('i,jk->ijk', Adiag, np.eye(dim)) # block-diagonal mat. for non-isotropic mat. 
b=assemble(f*v*dx).get_local() # vector of right-hand side
b[bc.get_boundary_values().keys()]=0 # application of homogeneous Dirichlet boundary conditions

# assembling interpolation-projection matrix B
B=np.zeros([V.dim(), Wvector.dim()])
for ii in range(V.dim()):
    bfun=Function(V)
    bfun.vector()[ii]=1.
    bfund=project(grad(bfun), Wvector)
    B[ii]=bfund.vector().get_local()

np.putmask(B, np.abs(B)<1e-14, 0) # removing the values that are close to zero
B=csr_matrix(B) # changing to the compressed sparse row (CSR) format
B.eliminate_zeros()
B=B.T

# solution to DoGIP problem
def Afun(x):
    xd=B.dot(x).reshape((-1, dim))
    Axd=np.einsum('...jk,...j', Adiag_full, xd)
    Afunx=B.T.dot(Axd.ravel())
    Afunx[bc.get_boundary_values().keys()]=0 # application of Dirichlet BC
    return Afunx

Alinoper=linalg.LinearOperator((b.size, b.size), matvec=Afun, dtype=np.float) # system matrix
x, info=linalg.cg(Alinoper, b, x0=np.zeros_like(b), tol=1e-8, maxiter=1e2) # conjugate gradients

print('difference FEniCS vs DoGIP: {}'.format(np.linalg.norm(u_fenics.vector().get_local()-x)))
print('END')
