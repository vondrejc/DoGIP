from fenics import (UnitSquareMesh, UnitCubeMesh, FunctionSpace, Function, TestFunction,
                    TrialFunction, Expression, assemble, dx, solve, cells, EigenMatrix)
import numpy as np
from scipy.sparse import linalg
from dogip.simplicialFE import get_B, get_Bhat, get_A_T
import itertools
import unittest


class Test_weighted_projection(unittest.TestCase):

    def test_DoGIP_vs_FEniCS(self):
        print('\n== testing DoGIP vs. FEniCS for problem of weighted projection ====')

        for dim, pol_order in itertools.product([2,3],[1,2]):
            print('dim={}; pol_order={}'.format(dim,pol_order))
            N=2
            # creating MESH, defining MATERIAL and SOURCE
            if dim==2:
                mesh=UnitSquareMesh(N, N)
                m=Expression("1+10*16*x[0]*(1-x[0])*x[1]*(1-x[1])", degree=3) # material coefficients
                f=Expression("x[0]*x[0]*x[1]", degree=2)
            elif dim==3:
                mesh=UnitCubeMesh(N, N, N)
                m=Expression("1+100*x[0]*(1-x[0])*x[1]*x[2]", degree=2) # material coefficients
                f=Expression("(1-x[0])*x[1]*x[2]", degree=2)

            mesh.coordinates()[:]+=0.1*np.random.random(mesh.coordinates().shape) # mesh perturbation

            ## standard approach with FEniCS #############################################
            V=FunctionSpace(mesh, "CG", pol_order) # original FEM space
            u, v=TrialFunction(V), TestFunction(V)
            u_fenics=Function(V)
            solve(m*u*v*dx==m*f*v*dx, u_fenics)

            ## DoGIP - double-grid integration with interpolation-projection #############
            W=FunctionSpace(mesh, "CG", 2*pol_order) # double-grid space
            w=TestFunction(W)
            A_dogip=assemble(m*w*dx).get_local() # diagonal matrix of material coefficients
            b=assemble(m*f*v*dx) # vector of right-hand side

            # assembling interpolation-projection matrix B
            B=get_B(V, W, problem=0)

            # # linear solver on double grid, standard
            Afun=lambda x: B.T.dot(A_dogip*B.dot(x))

            Alinoper=linalg.LinearOperator((V.dim(), V.dim()), matvec=Afun, dtype=np.float)
            x, info=linalg.cg(Alinoper, b.get_local(), x0=np.zeros(V.dim()),
                                tol=1e-10, maxiter=1e3, callback=None)

            # testing the difference between DoGIP and FEniCS
            self.assertAlmostEqual(0, np.linalg.norm(u_fenics.vector().get_local()-x))
            print('...ok')

    def test_multiplication(self):
        print('\n== testing multiplication of system matrix for problem of weighted projection ====')
        for dim, pol_order in itertools.product([2,3],[1,2]):
            N=2 # no. of elements

            print('dim={0}, pol_order={1}, N={2}'.format(dim, pol_order, N))

            # creating MESH and defining MATERIAL
            if dim==2:
                mesh=UnitSquareMesh(N, N)
                m=Expression("1+10*16*x[0]*(1-x[0])*x[1]*(1-x[1])", degree=2) # material coefficients
            elif dim==3:
                mesh=UnitCubeMesh(N, N, N)
                m=Expression("1+10*16*x[0]*(1-x[0])*(1-x[1])*x[2]", degree=2) # material coefficients

            mesh.coordinates()[:]+=0.1*np.random.random(mesh.coordinates().shape) # mesh perturbation

            V=FunctionSpace(mesh, "CG", pol_order) # original FEM space
            W=FunctionSpace(mesh, "CG", 2*pol_order) # double-grid space

            print('assembling local matrices for DoGIP...')
            Bhat=get_Bhat(dim, pol_order, problem=0) # projection between V on W on a reference element
            AT_dogip=get_A_T(m, V, W, problem=0)

            dofmapV=V.dofmap()

            def system_multiplication_DoGIP(AT_dogip, Bhat, u_vec):
                # mutliplication with DoGIP decomposition
                Au=np.zeros_like(u_vec)
                for ii, cell in enumerate(cells(mesh)):
                    ind=dofmapV.cell_dofs(ii) # local to global map
                    Au[ind]+=Bhat.T.dot(AT_dogip[ii]*Bhat.dot(u_vec[ind]))
                return Au

            print('assembling FEM sparse matrix')
            u, v=TrialFunction(V), TestFunction(V)
            Asp=assemble(m*u*v*dx, tensor=EigenMatrix()) #
            Asp=Asp.sparray()

            print('multiplication...')
            ur=Function(V) # creating random vector
            ur_vec=5*np.random.random(V.dim())
            ur.vector().set_local(ur_vec)

            Au_DoGIP=system_multiplication_DoGIP(AT_dogip, Bhat, ur_vec) # DoGIP multiplication
            Auex=Asp.dot(ur_vec) # FEM multiplication with sparse matrix

            # testing the difference between DoGIP and FEniCS
            self.assertAlmostEqual(0, np.linalg.norm(Auex-Au_DoGIP))
            print('...ok')


if __name__ == "__main__":
    unittest.main()
