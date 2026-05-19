# femsolver

A Python finite element solver for structural analysis. Architecture inspired by OpenSees.

## Phase 1 capabilities

- **Elements**: Truss2D, Truss3D, BeamColumn2D, BeamColumn3D (Euler-Bernoulli), Quad4 (plane stress / plane strain)
- **Materials**: linear elastic isotropic
- **Loads**: nodal point loads, element distributed loads, gravity
- **Constraints**: single-point (BC) by penalty/transformation, equal-DOF / rigid-link multi-point
- **Analysis**: linear static
- **Solver**: SciPy sparse direct (LU)
- **I/O**: JSON model files, VTK output for ParaView

## Install

```bash
pip install -e .[dev,viz]
```

## Quick start

```python
from femsolver import Model
from femsolver.materials import ElasticIsotropic
from femsolver.elements import Truss2D
from femsolver.analysis import LinearStaticAnalysis

m = Model(ndm=2, ndf=2)
m.add_node(1, 0.0, 0.0)
m.add_node(2, 1.0, 0.0)
m.add_node(3, 1.0, 1.0)

mat = ElasticIsotropic(tag=1, E=2.0e11, nu=0.3, rho=7850.0)
m.add_material(mat)

m.add_element(Truss2D(tag=1, nodes=(1, 2), material=mat, area=1e-4))
m.add_element(Truss2D(tag=2, nodes=(2, 3), material=mat, area=1e-4))
m.add_element(Truss2D(tag=3, nodes=(1, 3), material=mat, area=1e-4))

m.fix(1, [1, 1])           # pin
m.fix(2, [0, 1])           # roller
m.add_nodal_load(3, [1.0e4, 0.0])

analysis = LinearStaticAnalysis(m)
analysis.run()

print(m.node(3).disp)
```

## Run tests

```bash
pytest
```

## Roadmap

- **Phase 2** — nonlinear (Newton-Raphson, arc-length), dynamics (Newmark, HHT), modal, plasticity
- **Phase 3** — shells (MITC, DKT), solid elements (hex, tet), contact, large deformation, parallel solvers
