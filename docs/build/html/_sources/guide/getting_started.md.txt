# Getting started

This page walks you from a fresh Python install to running your first
analysis with **femsolver**.

## Install

femsolver is pure Python with NumPy + SciPy as the only hard
dependencies:

```bash
pip install -e .[dev,viz]
```

This will install the package in editable mode plus the optional
visualisation and dev-tool extras.

To verify the install:

```bash
pytest -q tests/test_truss.py
```

## A first model — pin-roller truss

```python
from femsolver import (
    ElasticIsotropic, LinearStaticAnalysis, Model, Truss2D,
)

# 3-bar truss, 1 m base, 1 m height
m = Model(ndm=2, ndf=2)
m.add_node(1, 0.0, 0.0)
m.add_node(2, 1.0, 0.0)
m.add_node(3, 1.0, 1.0)

mat = ElasticIsotropic(tag=1, E=2.0e11, nu=0.3, rho=7850.0)
m.add_material(mat)

for tag, (i, j) in enumerate([(1, 2), (2, 3), (1, 3)], start=1):
    m.add_element(Truss2D(tag, (i, j), mat, area=1.0e-4))

m.fix(1, [1, 1])          # pin at node 1
m.fix(2, [0, 1])          # roller at node 2 (uy = 0)
m.add_nodal_load(3, [1.0e4, 0.0])

LinearStaticAnalysis(m).run()
print("Node 3 displacement:", m.node(3).disp)
```

Run it with `python first_truss.py` and you should see a 2-vector of
displacements at node 3.

## What's next

Once this works, try:

* {doc}`modeling_primer` — the modelling vocabulary (nodes, materials,
  sections, elements, constraints).
* {doc}`analysis_types` — linear/nonlinear static, transient, eigen,
  buckling.
* The {doc}`/tutorials/index` — fifty worked examples, from a 2D truss
  through full PBE workflows on bridges.

## Where to look in the source

| You want to … | Open … |
|---------------|--------|
| add a new element type | `src/femsolver/elements/` (mirror an existing class) |
| add a uniaxial constitutive law | `src/femsolver/materials/uniaxial/` |
| add a new analysis driver | `src/femsolver/analysis/` |
| add a design-code check | `src/femsolver/design/` |
| see worked examples | `examples/` (50+ scripts) |
