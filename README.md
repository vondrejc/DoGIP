
- [Introduction](#introduction)
- [Examples](#examples)
- [References](#references)

# DoGIP

## Introduction

This repository is focused on double-grid integration with interpolation-projection (DoGIP), which is a novel discretisation approach. 

The code is written in [Python](https://www.python.org) (version 2.7.12) and depends on the following numerical libraries [NumPy](http://www.numpy.org) (version 1.14.1) and [SciPy](https://www.scipy.org) (version 1.0.0) for scientific computing as well as on the [FEniCS](https://fenicsproject.org/) (version 2017.2.0), which is an open-source computational platform for the solution of partial differential equations using finite element method.

This repository is distributed under an MIT license. If you find the code and approach interesting, you are kindly asked to cite the papers in [References](#references).

## Examples

At the moment the code contains two examples:

- weighted_projection.py - the DoGIP approach for weighted projection in 2D and 3D using simplicial elements of order p
- elliptic_problem.py - the DoGIP approach for scalar elliptic problem (diffusion, stationary heat transfer, etc.) in 2D and 3D using simplicial elements of order p

## References

- J. Vondřejc: *Double-grid quadrature with interpolation-projection (DoGIP) as a novel discretisation approach: An application to FEM on simplices.* 2017.
