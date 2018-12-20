from dolfin import (UnitSquareMesh, UnitCubeMesh, FunctionSpace, Function, TestFunction,
                    TrialFunction, Expression, assemble, dx, inner, grad, DirichletBC, Constant,
                    solve, VectorFunctionSpace, cells, EigenMatrix)
import numpy as np
from scipy.sparse import linalg
from dogip.simplicialFE import get_B, get_Bhat, get_A_T
import itertools
import unittest


class Test_elliptic_problem(unittest.TestCase):

    def test_DoGIP_vs_FEniCS(self):
        print('\n== testing DoGIP vs. FEniCS for problem of weighted projection ====')
        for dim, pol_order in itertools.product([2,3], [1,2]):
            print('dim={}; pol_order={}'.format(dim,pol_order))
            N=2 # no. of elements

            # creating MESH, defining MATERIAL and SOURCE
            if dim==2:
                mesh=UnitSquareMesh(N,N)
                m=Expression("1+10*16*x[0]*(1-x[0])*x[1]*(1-x[1])", degree=4)
                f=Expression("80*x[0]*(0.5-x[0])*(1.-x[0])*x[1]*(1.-x[1])", degree=5)
            elif dim==3:
                mesh = UnitCubeMesh(N, N, N)
                m=Expression("1+10*16*x[0]*(1-x[0])*(1-x[1])*x[2]", degree=4)
                f=Expression("80*x[0]*(0.5-x[0])*(1.-x[0])*x[1]*(1.-x[1])", degree=5)

            mesh.coordinates()[:] += 0.1*np.random.random(mesh.coordinates().shape) # mesh perturbation

            ## standard approach with FEniCS #############################################
            V=FunctionSpace(mesh, "CG", pol_order) # original FEM space
            bc=DirichletBC(V, Constant(0.0), lambda x, on_boundary: on_boundary)
            u, v=TrialFunction(V), TestFunction(V)
            u_fenics=Function(V) # the vector for storing the solution
            solve(m*inner(grad(u), grad(v))*dx==f*v*dx, u_fenics, bc) # solution by FEniCS

            ## DoGIP - double-grid integration with interpolation-projection #############
            W=FunctionSpace(mesh, "DG", 2*(pol_order-1)) # double-grid space
            Wvector=VectorFunctionSpace(mesh, "DG", 2*(pol_order-1)) # vector variant of double-grid space
            w=TestFunction(W)
            A_dogip=assemble(m*w*dx).get_local() # diagonal matrix of material coefficients
            A_dogip_full=np.einsum('i,jk->ijk', A_dogip, np.eye(dim)) # block-diagonal mat. for non-isotropic mat.
            bv=assemble(f*v*dx)
            bc.apply(bv)
            b=bv.get_local() # vector of right-hand side

            # assembling global interpolation-projection matrix B
            B=get_B(V, Wvector, problem=1)

            # solution to DoGIP problem
            def Afun(x):
                Axd=np.einsum('...jk,...j', A_dogip_full, B.dot(x).reshape((-1, dim)))
                Afunx=B.T.dot(Axd.ravel())
                Afunx[list(bc.get_boundary_values())]=0 # application of Dirichlet BC
                return Afunx

            Alinoper=linalg.LinearOperator((b.size, b.size), matvec=Afun, dtype=np.float) # system matrix
            x, info=linalg.cg(Alinoper, b, x0=np.zeros_like(b), tol=1e-8, maxiter=1e2) # conjugate gradients

            # testing the difference between DoGIP and FEniCS
            self.assertAlmostEqual(0, np.linalg.norm(u_fenics.vector().get_local()-x))
            print('...ok')

    def test_multiplication(self):
        print('== testing multiplication of system matrix for problem of weighted projection ====')

        for dim, pol_order in itertools.product([2,3],[1,2]):
            N=2 # no. of elements
            print('dim={0}, pol_order={1}, N={2}'.format(dim, pol_order, N))

            # creating MESH and defining MATERIAL
            if dim==2:
                mesh=UnitSquareMesh(N, N)
                m=Expression("1+10*16*x[0]*(1-x[0])*x[1]*(1-x[1])", degree=4) # material coefficients
            elif dim==3:
                mesh=UnitCubeMesh(N, N, N)
                m=Expression("1+10*16*x[0]*(1-x[0])*(1-x[1])*x[2]", degree=1) # material coefficients

            mesh.coordinates()[:]+=0.1*np.random.random(mesh.coordinates().shape) # mesh perturbation

            V=FunctionSpace(mesh, "CG", pol_order) # original FEM space
            W=FunctionSpace(mesh, "DG", 2*(pol_order-1)) # double-grid space

            print('assembling local matrices for DoGIP...')
            Bhat=get_Bhat(dim, pol_order, problem=1) # interpolation between V on W on a reference element
            AT_dogip=get_A_T(m, V, W, problem=1)

            dofmapV=V.dofmap()

            def system_multiplication_DoGIP(AT_dogip, Bhat, u_vec):
                # matrix-vector mutliplication in DoGIP
                Au=np.zeros_like(u_vec)
                for ii, cell in enumerate(cells(mesh)):
                    ind=dofmapV.cell_dofs(ii) # local to global map
                    Bu=Bhat.dot(u_vec[ind])
                    ABu=np.einsum('rsj,sj->rj', AT_dogip[ii], Bu)
                    Au[ind]+=np.einsum('rjl,rj->l', Bhat, ABu)
                return Au

            print('assembling system matrix for FEM')
            u, v=TrialFunction(V), TestFunction(V)
            Asp=assemble(m*inner(grad(u), grad(v))*dx, tensor=EigenMatrix())
            Asp=Asp.sparray() # sparse FEM matrix

            print('multiplication...')
            ur=Function(V) # creating random vector
            ur_vec=10*np.random.random(V.dim())
            ur.vector().set_local(ur_vec)

            Au_DoGIP=system_multiplication_DoGIP(AT_dogip, Bhat, ur_vec) # DoGIP multiplication
            Auex=Asp.dot(ur_vec) # FEM multiplication with sparse matrix

            # testing the difference between DoGIP and FEniCS
            self.assertAlmostEqual(0, np.linalg.norm(Auex-Au_DoGIP))
            print('...ok')


if __name__ == "__main__":
    unittest.main()
