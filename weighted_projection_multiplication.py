"""
Implementation of system matrix-vector multiplication in standard FEM and in DoGIP
"""
from __future__ import print_function, division
from fenics import (UnitSquareMesh, UnitCubeMesh, FunctionSpace, Function, TestFunction,
                    TrialFunction, Expression, assemble, dx, cells, inner, grad,
                    EigenMatrix)
import numpy as np
from dogip import get_Bhat, get_A_T

# PARAMETERS
dim=2 # dimension of the problem
N=2 # no. of elements
pol_order=2 # polynomial order of FEM approximation

print('dim={0}, pol_order={1}, N={2}'.format(dim, pol_order, N))

# creating MESH and defining MATERIAL
if dim==2:
    mesh=UnitSquareMesh(N,N)
    m=Expression("1+10*16*x[0]*(1-x[0])*x[1]*(1-x[1])", degree=4)  # material coefficients
elif dim==3:
    mesh = UnitCubeMesh(N, N, N)
    m=Expression("1+10*16*x[0]*(1-x[0])*(1-x[1])*x[2]", degree=1) # material coefficients

mesh.coordinates()[:] += 0.1*np.random.random(mesh.coordinates().shape) # mesh perturbation

V=FunctionSpace(mesh, "CG", pol_order) # original FEM space
W=FunctionSpace(mesh, "CG", 2*pol_order) # double-grid space

print('assembling local matrices for DoGIP...')
B0 = get_Bhat(dim, pol_order, problem=0) # projection between V on W on a reference element
AT_dogip = get_A_T(m, V, W, problem=0)

print('multiplication...')
ur = Function(V) # creating random vector
ur_vec = 5*np.random.random(V.dim())
ur.vector().set_local(ur_vec)

dofmapV = V.dofmap()
def system_multiplication_DoGIP(AT_dogip, B, u_vec):
    # mutliplication with DoGIP decomposition
    Au = np.zeros_like(u_vec)
    for ii, cell in enumerate(cells(mesh)):
        ind = dofmapV.cell_dofs(ii) # local to global map
        Au[ind] += B.T.dot(AT_dogip[ii]*B.dot(u_vec[ind]))
    return Au

u, v = TrialFunction(V), TestFunction(V)
Asp = assemble(m*u*v*dx, tensor=EigenMatrix()) # standard sparse matrix
Asp = Asp.sparray()

Au_DoGIP = system_multiplication_DoGIP(AT_dogip, B0, ur_vec)
Auex = Asp.dot(ur_vec) # standard multiplication with sparse matrix

print('norm of difference between standard FEM and DoGIP =')
print(np.linalg.norm(Auex-Au_DoGIP))
print('END')
