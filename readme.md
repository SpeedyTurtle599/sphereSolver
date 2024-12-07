This project aims to simulate 3D fluid flow around a sphere in NVIDIA CUDA C++. In addition to that base premise, it's got some complicated additions:
- Turbulent flow, modelled by k-epsilon equations
- Viscous boundary layer
- Log wall law for transition from viscous to inviscid flow
- SIMPLEC algorithm for pressure-velocity coupling

As of now, the full set of features are implemented but not yet functional. The velocity field does not properly propagate forward in time, leaving me a very messy todo.md. I'm currently working on debugging the SIMPLEC algorithm and checking boundary conditions for the pressure field. I also need to check the code for consistency with constant values usage -- I want to ensure that there are no hardcoded values wherever possible.