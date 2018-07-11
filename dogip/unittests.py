import unittest
import numpy as np
import itertools
from dogip.simplex import get_dof_coordinates, get_reference_element
from dogip.dim_spaces import dimP
from dolfin import FunctionSpace, Cell


class Test_FEMsimplex(unittest.TestCase):

    def test_simplex(self):
        print('\n== Testing DOFs ====')

        for dim, p in itertools.product([2,3],range(1,6)):
            print('dim={}; p={}'.format(dim,p))
            mesh=get_reference_element(dim)
            W=FunctionSpace(mesh, "DG", p)
            cell=Cell(mesh,0)
            dofs=W.element().tabulate_dof_coordinates(cell)
            dofs2=get_dof_coordinates(dim, pol_order=p)
            dofs=dofs[np.lexsort(dofs.T)]
            dofs2=dofs2[np.lexsort(dofs2.T)]
            self.assertAlmostEqual(0,np.linalg.norm(dofs-dofs2), delta=1e-14)
            self.assertAlmostEqual(W.element().space_dimension(),dimP(dim,p), delta=1e-13)
        print('...ok')

if __name__ == "__main__":
    unittest.main()
