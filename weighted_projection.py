from fenics import (UnitSquareMesh, UnitCubeMesh, FunctionSpace, Function, TestFunction,
                    TrialFunction, Expression, assemble, dx, cells, assemble_local, inner, grad,
                    EigenMatrix, solve)
import numpy as np
from scipy.sparse import csr_matrix, linalg
from dogip import get_B

# PARAMETERS
dim=2 # dimension of the problem
N=3 # no. of elements
pol_order=1 # polynomial order of FEM approximation

# creating MESH, defining MATERIAL and SOURCE
if dim==2:
    mesh = UnitSquareMesh(N, N)
    m = Expression("1+10*16*x[0]*(1-x[0])*x[1]*(1-x[1])", degree=4) # material coefficients
    f = Expression("x[0]*x[0]*x[1]", degree=3)
elif dim==3:
    mesh = UnitCubeMesh(N, N, N)
    m = Expression("1+100*x[0]*(1-x[0])*x[1]*x[2]", degree=4) # material coefficients
    f = Expression("(1-x[0])*x[1]*x[2]", degree=3)

mesh.coordinates()[:] += 0.1*np.random.random(mesh.coordinates().shape)  # mesh perturbation

## standard approach with FEniCS #############################################
V = FunctionSpace(mesh, "CG", pol_order) # original FEM space
u, v = TrialFunction(V), TestFunction(V)
u_fenics = Function(V)
solve(m*u*v*dx==m*f*v*dx, u_fenics)

## DoGIP - double-grid integration with interpolation-projection #############
W = FunctionSpace(mesh, "CG", 2*pol_order) # double-grid space
w = TestFunction(W)
A_dogip = assemble(m*w*dx).get_local() # diagonal matrix of material coefficients
b = assemble(m*f*v*dx) # vector of right-hand side

# assembling interpolation-projection matrix B
B=get_B(V, W, dim, pol_order, problem=0)

## linear solver on double grid, standard
Afun = lambda x: B.T.dot(A_dogip*B.dot(x))

Alinoper = linalg.LinearOperator((V.dim(), V.dim()), matvec=Afun, dtype=np.float)
x, info = linalg.cg(Alinoper, b.get_local(), x0=np.zeros(V.dim()),
                    tol=1e-10, maxiter=1e3, callback=None)

print('difference FEniCS vs DoGIP: {}'.format(np.linalg.norm(u_fenics.vector().get_local()-x)))
print('END')
