import numpy as np
import scipy.sparse
from reference import get_reference_element, Mapping
from fenics import (TrialFunction, TestFunction, assemble, inner, grad, FunctionSpace, Function,
                    EigenMatrix, dx, Mesh, MeshEditor, Point, Cell, project, interpolate,
                    assemble_local, cells, triangle, tetrahedron, FiniteElement)


def get_Bhat(dim=2, pol_order=1, problem=1):
    """
    Calculate interpolation matrix between V and W on a reference element

    Parameters
    ----------
    dim : int
        topological dimension of the problem
    por_order : int
        polynomial order of finite element space
    problem : {int, string}
        parameter defining problem: 0 - weighted projection, 1 - elliptic problem

    Returns
    -------
    B : ndarray
        interpolation matrix between original and double-grid finite element basis
    """

    mesh=get_reference_element(dim)
    cell=Cell(mesh,0)
    V=FunctionSpace(mesh, "CG", pol_order) # original FEM space
    if problem in [1, 'elliptic']:
        W=FunctionSpace(mesh, "DG", 2*(pol_order-1)) # double-grid space
    elif problem in [0, 'projection']:
        W=FunctionSpace(mesh, "CG", 2*pol_order) # double-grid space

    elementV = V.element()
    dofcoors=W.element().tabulate_dof_coordinates(cell)
    if problem in [1, 'elliptic']:
        B = np.zeros([W.dim(),V.dim(),dim])
        eval_basis = lambda *args: elementV.evaluate_basis_derivatives_all(*args)
        der = (1,)
        postprocess = lambda B: np.einsum('jkr->rjk',B)
        val_shape = (V.dim(), dim)
    elif problem in [0, 'projection']:
        B = np.zeros([W.dim(),V.dim()])
        eval_basis = lambda *args: elementV.evaluate_basis_all(*args)
        der = ()
        postprocess = lambda B: B
        val_shape = (V.dim())

    for jj, dofcoor in enumerate(dofcoors):
        val = np.zeros(val_shape, dtype=float)
        args = der + (val, dofcoor, cell.get_vertex_coordinates(), cell.orientation())
        eval_basis(*args)
        B[jj] = val

    return postprocess(B)

def get_B(V, W, problem=1, sparse=True, threshold=1e-14):
    """
    Projection/interpolation operator between spaces V and W on the whole computational domain.

    Parameters
    ----------
    V : FunctionSpace
        original finite element space
    W : FunctionSpace
        double-grid finite element space
    problem : {int, string}
        parameter defining problem: 0 - weighted projection, 1 - elliptic problem
    sparse : boolean
        determining the format of output matrix
    threshold : float
        parameter under which the values are considered to be zero

    Returns
    -------
    B : ndarray
        interpolation matrix between original and double-grid finite element basis
    """
    if problem in [1, 'elliptic']: # elliptic problem
        operator = grad
        try:
            IP = Project_multiple(W)
        except:
            IP = project
    elif problem in [0, 'projection']: # weighted projection
        operator = lambda x: x
        IP = interpolate

    if sparse: # sparse version
        col = np.array([])
        row = np.array([])
        data = np.array([])
        for ii in range(V.dim()):
            bfun = Function(V)
            bfun.vector()[ii] = 1.
            bfund = IP(operator(bfun), W)
            vals = bfund.vector().get_local()
            indc = np.where(np.abs(vals) > threshold)[0]
            col = np.hstack([col, indc])
            row = np.hstack([row, ii*np.ones(indc.size)])
            data = np.hstack([data, vals[indc]])

        col = np.array(col)
        row = np.array(row)
        data = np.array(data)

        B = scipy.sparse.csr_matrix((data, (row, col)), shape=(V.dim(), W.dim()))

    else: # dense version
        B = np.zeros([V.dim(), W.dim()])
        for ii in range(V.dim()):
            bfun = Function(V)
            bfun.vector()[ii] = 1.
            bfund = IP(operator(bfun), W)
            B[ii] = bfund.vector().get_local()()
        np.putmask(B, np.abs(B)<threshold, 0)
        B = scipy.sparse.csr_matrix(B)
        B.eliminate_zeros()

    return B.T

def get_A_T(m, V, W, problem=1):
    """
    The element-wise (local) system matrices of DoGIP
    Projection/interpolation operator between spaces V and W on the whole computational domain.

    Parameters
    ----------
    m : Expression
        defines material coefficients as a scalar valued function
    V : FunctionSpace
        original finite element space
    W : FunctionSpace
        double-grid finite element space
    problem : {int, string}
        parameter defining problem: 0 - weighted projection, 1 - elliptic problem

    Returns
    -------
    AT_dogip : ndarray
        element-wise (local) system matrices of DoGIP
    """
    mesh = V.mesh()
    dim = mesh.geometry().dim()

    w = TestFunction(W)
    if problem in [0, 'projection']:
        AT_dogip = np.empty([mesh.num_cells(), W.element().space_dimension()])
        for ii, cell in enumerate(cells(V.mesh())):
            AT_dogip[ii] = assemble_local(m*w*dx, cell)
    elif problem in [1, 'elliptic']:
        # assumes (without generality) that material coefficients are isotropic
        AT_dogip = np.empty([mesh.num_cells(), dim, dim, W.element().space_dimension()])
        for ii, cell in enumerate(cells(V.mesh())):
            refmap = Mapping(nodes=cell.get_vertex_coordinates().reshape(dim+1,dim))
            MT = np.einsum('ij,k',np.eye(dim), assemble_local(m*w*dx, cell))
            AT_dogip[ii] = np.einsum('rp,sq,pqj->rsj', refmap.Ai, refmap.Ai, MT)
    return AT_dogip

class Project_multiple():
    # projection for multiple right-hand sides - used in get_B only for speed-up
    def __init__(self, V):
        u=TrialFunction(V)
        v=TestFunction(V)
        A=assemble(inner(u, v)*dx, tensor=EigenMatrix())
        self.V=V
        self.v=v

        try:
            import sksparse
            from sksparse.cholmod import cholesky # sparse Cholesky decomposition
            print('...projection with cholesky of version {}'.format(sksparse.__version__))
            self.solve=cholesky(A.sparray().tocsc())
        except:
            print('...projection with LU decomposition')
            invA=scipy.sparse.linalg.splu(A.sparray().tocsc()) # sparse LU decomposition
            self.solve=invA.solve

    def __call__(self, f, W=None):
        # f = Function(V)
        fvec = assemble(inner(f,self.v)*dx)
        Pf = Function(self.V)
        Pf.vector()[:] = self.solve(fvec.get_local())
        return Pf
