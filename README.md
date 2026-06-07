# femsolver

A Python finite-element solver for structural engineering — from
linear analysis through full performance-based earthquake engineering
(PBE), member-level design checks to four major codes, and bridge
engineering.

[![tests](https://img.shields.io/badge/tests-1198_passing-brightgreen)](#run-tests)
[![python](https://img.shields.io/badge/python-3.11+-blue)](#install)
[![codes](https://img.shields.io/badge/design_codes-ACI%20%7C%20AISC%20%7C%20IS-yellow)](#design-codes)

## What's inside

| Layer | Coverage |
|-------|----------|
| **Elements** | Truss/Beam (corotational, force-based, fiber, hinged), Quad4, MITC4/9, DKMQ4, Tri3, DKT3, Hex8, Tet4, ZeroLength, isolators |
| **Materials** | Elastic, J₂ / Drucker-Prager plasticity, uniaxial library (Bilinear, IMK, BRB, Takeda, Pivot, Mander, Kent-Park, Menegotto-Pinto, …) |
| **Sections** | Elastic 2D/3D, Fiber 2D/3D, Layered shell + ply failure, hinges, Wall fiber (T/L/U/I) |
| **Analysis** | Linear/nonlinear static, transient (Newmark/HHT/Gen-α/CD), eigen, buckling, response spectrum, modal pushover, IDA |
| **Constraints + solvers** | Rigid diaphragm, equal-DOF, rigid link, MPC; direct sparse + iterative |
| **PBE pipeline** | IDA → collapse → fragility → CMS-based record selection → SSI → FEMA P-58 component damage |
| **Design codes** | ACI 318, AISC 360, IS 456, IS 800, ASCE 7/41, IS 1893 / IS 13920, AISC 341 detailing |
| **Bridges** | Influence lines + HL-93 / IRC, PT tendons + losses, CEB-FIP creep / shrinkage, composite sections |
| **Connections** | Krawinkler panel zone, AISC 358 RBS, Richard-Abbott PR, bolts / welds |
| **V&V** | 11 closed-form benchmarks (NAFEMS-style), CSV export, pass/fail report |

## Install

```bash
pip install -e .[dev,viz]
```

## Quick start

```python
from femsolver import (
    ElasticIsotropic, LinearStaticAnalysis, Model, Truss2D,
)

m = Model(ndm=2, ndf=2)
m.add_node(1, 0.0, 0.0)
m.add_node(2, 1.0, 0.0)
m.add_node(3, 1.0, 1.0)

mat = ElasticIsotropic(tag=1, E=2.0e11, nu=0.3, rho=7850.0)
m.add_material(mat)
for tag, (i, j) in enumerate([(1, 2), (2, 3), (1, 3)], start=1):
    m.add_element(Truss2D(tag, (i, j), mat, area=1.0e-4))

m.fix(1, [1, 1]); m.fix(2, [0, 1])
m.add_nodal_load(3, [1.0e4, 0.0])

LinearStaticAnalysis(m).run()
print("Node 3 displacement:", m.node(3).disp)
```

## Capstone examples

```bash
python examples/43_pbe_full_workflow.py        # IDA → P-58 loss curve
python examples/45_coupled_wall_pushover.py    # coupled shear walls
python examples/47_indian_codes_frame_design.py  # IS-codes frame
python examples/48_connection_capstone.py      # panel zone + RBS + PR
python examples/49_psc_girder_bridge.py        # PSC bridge with HL-93
python examples/53_full_design_report.py       # HTML/CSV design report
```

50+ self-contained scripts in `examples/` covering every analysis
type, material model, design code, and capstone workflow.

## Design codes

| Code | Module | Coverage |
|------|--------|----------|
| **ACI 318-19** | `femsolver.design.concrete` | Beam flexure / shear, column P-M |
| **AISC 360-22** | `femsolver.design.steel` | Tension / compression / flexure / shear / combined |
| **IS 456:2000** | `femsolver.design.is456` | Beam flexure, shear (Table 19), column P-M |
| **IS 800:2007** | `femsolver.design.is800` | Tension, Perry-Robertson compression, flexure + LTB, shear, combined |
| **ASCE 7-22** | `femsolver.analysis.load_combinations` | LRFD combinations, drift check |
| **ASCE 41-17** | `femsolver.performance.capacity_design`, `femsolver.sections.response.wall_shear` | Coefficient method, wall cracked-section factors |
| **IS 1893:2016** | `femsolver.design.is1893` | Design spectrum, base shear, vertical distribution |
| **AISC 341 / ACI 18** | `femsolver.design.seismic` | SCWB, capacity shear, confinement |
| **IS 13920:2016** | `femsolver.design.is13920` | SCWB, capacity shear, confinement |
| **AISC 358** | `femsolver.design.connections.rbs` | Reduced beam section |
| **AASHTO LRFD** | `femsolver.bridges` | HL-93 envelope, PT losses |
| **CEB-FIP MC 2010** | `femsolver.bridges.creep_shrinkage` | Creep + shrinkage |

## Run tests

```bash
pytest -q
# 1198 passed
```

Run the V&V benchmark suite:

```bash
python examples/46_vnv_report.py --csv vnv_report.csv
```

## Build the docs

```bash
pip install sphinx myst-parser sphinx-rtd-theme sphinx-autodoc-typehints
cd docs
make html         # docs/build/html/index.html
make theory       # docs/theory/theory.pdf (LaTeX)
```

## Architecture

```
src/femsolver/
├── core/              # Model, Node, DOF numbering
├── materials/         # continuum + uniaxial library
├── sections/          # elastic, fiber, shell, wall
├── elements/          # truss, beam, plane, shell, solid, zerolength
├── analysis/          # static / transient / eigen / buckling / PBE
├── constraints/       # MPC, rigid link, rigid diaphragm
├── design/            # ACI, AISC, IS, seismic detailing, connections
├── bridges/           # influence lines, PT, creep/shrinkage, composite
├── benchmarks/        # V&V suite
└── io/                # VTK, JSON
```

## Roadmap

See `docs/source/theory/overview.md` and the recommendations in
`docs/source/guide/`. Upcoming themes under consideration:

* **Theme C** — thermal / fire engineering
* **Theme F** — foundation extensions (mat / raft, pile groups, liquefaction)
* **Theme B** — large strain + contact
* **Bridges Phase 2** — staged construction with time-dependent effects, cable elements
* **Theme K** — solver performance (PARDISO / MUMPS, substructuring)

## License

See `LICENSE`.
