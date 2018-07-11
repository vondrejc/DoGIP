"""
Calculating computational and memory requirements for DoGIP
"""
from __future__ import division, print_function
from dolfin import (UnitSquareMesh, UnitCubeMesh, Expression, FunctionSpace,
                    TrialFunction, TestFunction, assemble, dx, inner, grad, EigenMatrix)
import numpy as np
from scipy.io import savemat, loadmat
import itertools
from dogip.simplicialFE import get_Bhat, get_A_T_empty

calculate = -1 # 0: table; 1: store matrices; -1: test

dims = [2,3]
problems = [0,1] # 0: L2-projection, 1: scalar elliptic problem
threshold = 1e-10

## OPERATORS
nnz = lambda B, threshold=threshold: np.sum(np.abs(B) > threshold)
mem_sparse = lambda B: 2*nnz(B)+B.shape[0]
ones= lambda B: np.sum(np.abs(np.abs(B) - 1) < threshold)

for dim, problem in itertools.product(dims, problems):
    if calculate in [0,1]:
        if dim==2:
            Ne_max=3*5*8*10 # maximal number of elements in each spatial direction
            p_list=[1, 2, 3, 4, 5, 6, 8] # polynomial orders
        elif dim==3:
            Ne_max=3*8*4 # maximal number of elements in each spatial direction
            p_list=[1, 2, 3, 4] # polynomial orders

    elif calculate in [-1]:
        print('testing...')
        if dim==2:
            Ne_max=3*5*8
            p_list=[1,2,3]
        elif dim==3:
            Ne_max=3*4*2
            p_list=[1,2,3]

    print('\n== dim = {0}, grad = {1}, Ne_max = {2} =='.format(dim, problem, Ne_max))
    ss=''
    for p in p_list:
        if calculate in [-1,0,1]:
            Ne=int(Ne_max/p)
        else:
            Ne=int(Ne_max)
        print('-- p = {0}, N = {1} --------'.format(p, Ne))
        filen_data='data/t%d_d%d_Nm%d_p%.2d_Ne%.3d' % (problem, dim, Ne_max, p, Ne)

        if dim==2:
            mesh=UnitSquareMesh(Ne, Ne)
            Amat=Expression("1+10*exp(x[0])*exp(x[1])", degree=2)
        elif dim==3:
            mesh=UnitCubeMesh(Ne, Ne, Ne)
            Amat=Expression("1+100*exp(x[0])*exp(x[1])*x[2]*x[1]", degree=3)

        mesh.coordinates()[:] += 0.1*np.random.random(mesh.coordinates().shape) # mesh perturbation

        V=FunctionSpace(mesh, 'CG', p)
        print('V.dim = {0}'.format(V.dim()))

        if calculate in [1,-1]:
            ut, vt=TrialFunction(V), TestFunction(V)
            print('generating matrices A, Ad, A_T')
            if problem==0:
                AG=assemble(Amat*ut*vt*dx, tensor=EigenMatrix())
                A_T=get_A_T_empty(V, problem=problem, full=False)
            else:
                AG=assemble(inner(Amat*grad(ut), grad(vt))*dx, tensor=EigenMatrix())
                A_T=get_A_T_empty(V, problem=problem, full=False)

            print('generating matrices B, B0...')
            B0=get_Bhat(dim=dim, pol_order=p, problem=problem)

            A=AG.sparray()
            if calculate == 1:
                print('saving the matrices...')
                savemat(filen_data, dict(A=A, B0=B0, A_T=A_T), do_compression=True)

        else:
            data=loadmat(filen_data)
            A = data['A']
            A_T = data['A_T']
            B0 = data['B0']

        mem_eff=mesh.num_cells()*A_T[0].size/mem_sparse(A)
        com_eff=(2*(nnz(B0)-ones(B0))*mesh.num_cells()+mesh.num_cells()*A_T[0].size)/nnz(A)

        mem_Aloc=V.element().space_dimension()**2
        print('[Ne, p, mem_sparse(A), mem_Aloc, nnz(A_T[0]), nnz(B0), mem_eff, com_eff]')
        for val in [Ne, p, mem_sparse(A), mem_Aloc,
                    mesh.num_cells()*A_T[0].size, A_T[0].size, nnz(B0),
                    mem_eff, com_eff]:
            if isinstance(val, int):
                ss+=' {:,} &'.format(val)
            elif isinstance(val, float):
                ss+=' %.2f &' % val
        ss=ss[:-1]
        ss+=' \\\\ \n'

        print(ss)

    if calculate in [0]:
        with open('data/stat_d%d_p%d.txt' % (dim, problem), 'w') as tf:
            tf.write(ss)

print('END')
