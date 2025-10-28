This project aims to simulate 3D fluid flow around a sphere in NVIDIA CUDA C++. In addition to that base premise, it's got some additions:
- Turbulent flow, modelled by k-epsilon equations (soon to be replaced by SST k-omega)
- Viscous boundary layer
- Log wall law for transition from viscous to inviscid flow
- SIMPLEC algorithm for pressure-velocity coupling
