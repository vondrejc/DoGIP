import numpy as np
from fenics import (Mesh, MeshEditor)


def get_reference_element(dim=2):
    # get mesh as a reference element
    if dim==2:
        nodes=np.array([[0., 0.],
                        [1., 0.],
                        [0., 1.]])
        cell='triangle'
    elif dim==3:
        nodes=np.array([[0., 0., 0.],
                        [1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]])
        cell='tetrahedron'

#     cells=np.arange([[0, 1, 2]], dtype=np.uintp) # connection of nodes (defines element)
    cells=np.atleast_2d(np.arange(dim+1, dtype=np.uintp)) # connection of nodes (defines element)
#     cells=np.array([[2,0,1]], dtype=np.uintp)
    print '-- Fenics assembling ---'
    # this piece of code creates a mesh containing one element only
    mesh=Mesh()
    editor=MeshEditor()
    editor.open(mesh, cell, dim, dim)
    editor.init_vertices(dim+1)
    editor.init_cells(1)

    [editor.add_vertex(i, n) for i, n in enumerate(nodes)]
    [editor.add_cell(i, n) for i, n in enumerate(cells)]
    editor.close()
    return mesh

class Mapping():
    def __init__(self, nodes):
        # nodes of simplex of shape=(dim+1,dim)
        self.nodes=nodes
        self.b=nodes[0]
        self.A=np.vstack([nodes[ii]-nodes[0] for ii in range(1,nodes.shape[0])]).T
        self.Ai=np.linalg.inv(self.A)

    def __call__(self, hx):
        return self.A.dot(hx)+self.b

    def jacobian(self):
        return self.A

    def det_jacobian(self):
        return np.linalg.det(self.A)
