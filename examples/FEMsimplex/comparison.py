"""
This script generates the figures with the number of integration points
with respect to the dimensions of polynomial spaces.
"""
from __future__ import division, print_function

from dogip.dim_spaces import dimP
from dogip.quadrature import get_npoints, QuadSimplex

import matplotlib as mpl
import matplotlib.pyplot as pl
import figures_par
parf=figures_par.set_pars(mpl)


dims = [2,3]
ps=range(1,11) # polynomial orders

for dim in dims:

    plot=pl.semilogy
    pl.figure()
    plot(ps, [dimP(dim,p) for p in ps], 'k--', label='$\mathrm{dim}(\mathcal{V})$')
    plot(ps, [dimP(dim, 2*p) for p in ps], 'k:', label='$\mathrm{dim}(\mathcal{W})$')
    if dim==2:
        plot(ps, [get_npoints(dim, 2*p, method=0) for p in ps], 'bs-', label='$\mathrm{nip}(2k)$ by W\&V')
        plot([p for p in ps if 3*p<=20], [get_npoints(dim, 3*p, method=0) for p in ps if 3*p<=20], 'bD-', label='$\mathrm{nip}(3k)$ by W\&V')
    elif dim==3:
        plot([p for p in ps if p<=5], [get_npoints(dim, 2*p, method=0) for p in ps if p<=5], 'bs-', label='$\mathrm{nip}(2k)$ by W\&V')
        plot([p for p in ps if 3*p<=10], [get_npoints(dim, 3*p, method=0) for p in ps if 3*p<=10], 'bD-', label='$\mathrm{nip}(3k)$ by W\&V')
#     plot(ps, [get_npoints(dim, 2*p, method=1) for p in ps], '-s', label='$\mathrm{nip}(2p)$ by Shunn')
#     plot(ps, [get_npoints(dim, 3*p, method=1) for p in ps], '-D', label='$\mathrm{nip}(3p)$ by Shunn')
    plot(ps, [QuadSimplex._get_npoints(dim, 2*p) for p in ps], 'r+-.', label='$\mathrm{nip}(2k)$ by FEniCS')
    plot(ps, [QuadSimplex._get_npoints(dim, 3*p) for p in ps], 'rx-.', label='$\mathrm{nip}(3k)$ by FEniCS')
    pl.xlabel('Polynomial order $k$')
    pl.ylabel('Dimensions')
    pl.legend(loc='best')
    filen='figures/nip_d{}.pdf'.format(dim)
    pl.savefig(filen, dpi=parf['dpi'], pad_inches=parf['pad_inches'], bbox_inches='tight')
    pl.close()

print('END')