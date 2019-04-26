"""
Dimensions of FEM spaces - redundant???
"""

import numpy as np
from scipy.sparse import csr_matrix
fact = np.math.factorial


class Dimensions(): # super class
    def __init__(self, dim=2, Ne=5, p=1):
        self.d = dim
        self.N = Ne
        self.p = p

    def nel(self):
        if self.d == 2:
            return 2*self.N**self.d
        elif self.d == 3:
            return int(6*self.N**self.d)

    def nvertex(self):
        return int((self.Ne+1)**self.d)


class DimCG(Dimensions):
    def dim_element(self):
        d, p = self.d, self.p
        return int(fact(d+p)/fact(p)/fact(d))

    def dim(self):
        d, N, p = self.d, self.N, self.p
        return int((N*p+1)**d)


class DimDG(Dimensions):

    def dim(self):
        d, N, p = self.d, self.N, self.p
        return int(N**d*fact(d+p)/fact(p))


def remove_zeros(As, threshold=1e-14):
    As2 = csr_matrix(As, copy=True)
    np.putmask(As2.data, np.abs(As2.data)<threshold, 0)
    As2.eliminate_zeros()
    return As2


def dimP(d, p): # polynomial dimension
    return int(fact(d+p)/fact(d)/fact(p))

def dimQ(d, p): # polynomial dimension
    return int((p+1)**d)

if __name__=='__main__':
    d=2
    p=2
    print(dimP(d,p))
    print(dimQ(d,p))
