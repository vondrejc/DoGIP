
- [Introduction](#introduction)
- [Examples](#examples)
- [References](#references)

# DoGIP

## Introduction

This repository is focused on double-grid integration with interpolation-projection (DoGIP),
which is a novel discretisation approach.
At the moment the focus is concentrated only on FEM on simplexes.
The further development will be published along with the new papers.

The code is optimised for [Python](https://www.python.org) (version 3.6) and
depends on the following numerical libraries:
- [NumPy](http://www.numpy.org) (version 1.15.4) and [SciPy](https://www.scipy.org) (version 1.1.0) for scientific computing as well as on the
- [Scikit-sparse](https://pypi.org/project/scikit-sparse/) (version 0.4.4) for Cholesky decomposition, and
- [FEniCS](https://fenicsproject.org/) (version 2018.1.0), which is an open-source computational platform for the solution of partial differential equations using finite element method.

## License
This repository is distributed under an open MIT license.
If you find the code and approach interesting, you are kindly asked to cite the papers
in [References](#references).

## Structure of the code
At the moment the code contains the examples presented in the paper in [References](#references),
which is focused on DoGIP approach for FEM on simplexes.
The structure of the paper is following:

- `dogip` package contains all necessary functions for DoGIP approach.
Particularly, it contains the computation of the dimensions of function spaces,
coordinates and mappings to a reference element, and material coefficients
with interpolation matrices for DoGIP.

- `examples` package contains the numerial examples.
    - `FEMsimplex.dogip_dimensions.py` computes the dimensions of the DoGIP and
    standard FEM systems, which is incorporated in the referenced paper.
    - `FEMsimplex.elliptic_problem.py` compares the implemenation for DoGIP and standard FEM
    for the scalar elliptic problem in 2D and 3D.
    - `FEMsimplex.weighted_projection.py` compares the implemenation for DoGIP and standard FEM
    for the problem of the weighted projection in 2D and 3D.

All files in `examples` can be run as a python script,
i.e. the file `name_of_file.py` can be run using the following shell command

```
python3 name_of_file.py
```

## References

1. Jaroslav Vondřejc: *Double-grid quadrature with interpolation-projection (DoGIP) as a novel discretisation approach: An application to FEM on simplices.* 2017. [arXiv:1710.09913](http://arxiv.org/abs/1710.09913)
