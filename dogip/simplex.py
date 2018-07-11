import numpy as np
from dolfin import (Mesh, MeshEditor, Point)
import itertools


def get_reference_coordinates(dim):
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
    return nodes, cell

def get_reference_element(dim=2):
    nodes, cell = get_reference_coordinates(dim)
    cells=np.atleast_2d(np.arange(dim+1, dtype=np.uintp)) # connection of nodes (defines element)

    # this piece of code creates a mesh containing one element only
    mesh=Mesh()
    editor=MeshEditor()
    editor.open(mesh, cell, dim, dim)
    editor.init_vertices(dim+1)
    editor.init_cells(1)
    for i, n in enumerate(nodes):
        p=Point(n)
        editor.add_vertex(i, p)
    for i, n in enumerate(cells):
        editor.add_cell(i, n)
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

def get_dof_coordinates(dim, pol_order):
    nodes, _=get_reference_coordinates(dim=dim)
    if pol_order==0:
        lams = np.array([1.])/(dim+1)
    else:
        lams = np.linspace(0, 1, pol_order+1)

    dofs=[]
    for lam in itertools.product(lams, repeat=dim+1):
        lam=np.array(lam)
        if np.abs(lam.sum()-1)<1e-13:
            dofs.append(lam.dot(nodes))
    return np.array(dofs)

if __name__=='__main__':
    dim=2
    mesh=get_reference_element(dim)
    print(mesh.coordinates())
    import matplotlib.pyplot as pl
    from dolfin import plot
    plot(mesh)
    pl.show()
    print('END')
