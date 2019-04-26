from ffc.fiatinterface import create_quadrature


def get_npoints(dim, deg, method=1):
    if method in [0, 'Witherden']: # Witherden and Vincent 2015
        if dim in [2]:
            nips=[1,1,3,6,6,7,12,15,16,19,25,28,33,37,42,49,55,60,67,73,79]
        elif dim in [3]:
            nips=[1,1,4,8,14,14,24,35,46,59,81]
        nip=nips[deg]
    elif method in [1, 'Shunn']: # Shunn and Ham 2012, Williams, Shunn, Jameson 2014
        if dim==2:
            nip = int(deg * ( deg + 1 )/2) # guess without literature
        elif dim==3:
            nip = int(deg * ( deg + 1 )*( deg + 2 )/6) # Shunn 2012
    return nip

class QuadSimplex(): # integration on simplexes using FEniCS
    def __init__(self, dim, degree):
        self.dim=dim
        self.degree=degree

    def get_integration(self):
        if self.dim == 2:
            cell='triangle'
        elif self.dim==3:
            cell='tetrahedron'
        ip, iw=create_quadrature(cell, self.degree)
        return ip, iw

    def get_npoints(self):
        return self._get_npoints(self.dim, self.degree)

    @staticmethod
    def _get_npoints(dim, deg):
        if dim == 2:
            cell='triangle'
        elif dim==3:
            cell='tetrahedron'
        ip, iw=create_quadrature(cell, deg)
        return ip.shape[0]


if __name__=='__main__':
    import itertools
    from dogip.dim_spaces import dimP

    for dim, deg in itertools.product([2,3], range(1,18)):
        quad=QuadSimplex(dim,deg)
        ip, iw=quad.get_integration()
        print('min(iw)={}'.format(iw.min()))
        nip=quad.get_npoints()
        nip2=get_npoints(dim,deg)
        dP=dimP(dim, deg)
        print('dim={}, deg={}, dimP={}, nip={}, nip={}'.format(dim, deg, dP, nip, nip2))

    quad=QuadSimplex(2,5)
    ip, iw=quad.get_integration()

    import matplotlib.pyplot as pl
    pl.figure()
    for ip0 in ip:
        pl.plot(ip0[0], ip0[1], 'x')
    pl.xlim([0,1])
    pl.ylim([0,1])
    pl.show()


