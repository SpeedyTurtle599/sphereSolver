This project aims to simulate 3D fluid flow around a sphere in NVIDIA CUDA C++. In addition to that base premise, it's got some ambitious additions:
- Turbulent flow, modelled by k-epsilon equations (soon to be replaced by SST k-omega)
- Viscous boundary layer
- Log wall law for transition from viscous to inviscid flow
- SIMPLEC algorithm for pressure-velocity coupling

As of now, the full set of features are implemented but not yet functional. The residuals initialise to 0 and cause an early convergence to the simulation. I hope to resolve this issue by the end of 2024.

The files here are under the creative commons license "Attribution-NonCommercial-ShareAlike" CC BY-NC-SA, unless indicated otherwise.
