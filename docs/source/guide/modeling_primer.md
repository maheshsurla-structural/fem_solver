# Modelling primer

femsolver follows the **OpenSees** vocabulary: a `Model` holds nodes,
elements, materials, and constraints, plus the analysis drivers that
march the model along an equilibrium path.

## The five core objects

```
Material        Section          Element
-----------     ----------       -----------
ElasticIso  →   FiberSection2D → BeamColumn2D   (mat → section → element)
J2Plasticity →  ElasticSection → Hex8
UniaxialBilin → (uses material directly)         → Truss2D, ZeroLength
```

* **Material** — constitutive law (sigma vs epsilon). Examples:
  `ElasticIsotropic`, `J2Plasticity3D`, `UniaxialBilinear`,
  `ConcreteMander`, `UniaxialIMK`.
* **Section** — cross-section response: maps generalised strains
  (axial, curvature) to generalised stresses (N, M). Built from
  materials. Examples: `ElasticSection2D`, `FiberSection2D`,
  `LayeredShellSection`, `wall_section_2d`.
* **Element** — integrates section response over a finite length /
  volume. Examples: `BeamColumn2D`, `ShellMITC4`, `Hex8`,
  `ZeroLengthElement`.
* **Node** — point in space carrying DOFs (translations + rotations).
* **Constraint** — DOF coupling: `RigidLink`, `RigidDiaphragm`,
  `EqualDOF`, `MPConstraint`.

## Building the model

```python
from femsolver import Model, BeamColumn2D, ElasticIsotropic

m = Model(ndm=2, ndf=3)        # 2D frame: 3 DOFs per node (u, v, theta_z)
m.add_node(1, 0.0, 0.0)
m.add_node(2, 3.0, 0.0)

steel = ElasticIsotropic(1, E=2.0e11, nu=0.3, rho=7850.0)
m.add_material(steel)

m.add_element(BeamColumn2D(1, (1, 2), steel, area=0.01, Iz=8.33e-7))

m.fix(1, [1, 1, 1])               # fully fixed
m.add_nodal_load(2, [0.0, -1.0e3, 0.0])
```

## ndm / ndf conventions

* `ndm` is the spatial dimension (2 or 3).
* `ndf` is the number of DOFs per node:
  - `ndm=2, ndf=2` — plane truss (u, v).
  - `ndm=2, ndf=3` — plane frame (u, v, θz).
  - `ndm=3, ndf=3` — 3D truss / solid (ux, uy, uz).
  - `ndm=3, ndf=6` — 3D frame / shell (ux, uy, uz, θx, θy, θz).

## Boundary conditions

`m.fix(node_tag, mask)` where `mask` is a list of 0/1 per DOF.
`1` = restrained, `0` = free. Examples:

```python
m.fix(1, [1, 1, 1])          # 2D pin (no translation, no rotation)
m.fix(2, [0, 1, 0])          # 2D roller in y
m.fix(3, [1, 1, 1, 0, 0, 0]) # 3D pin (translations only)
```

## Loads

* **Nodal**: `m.add_nodal_load(node, force_vector)`.
* **Element distributed** (beams): the element itself accepts a
  distributed-load attribute.
* **Body forces / gravity**: see
  {mod}`femsolver.analysis.loads`.

## Running an analysis

```python
from femsolver import LinearStaticAnalysis
LinearStaticAnalysis(m).run()

# Read results
print(m.node(2).disp)
print(m.node(2).reaction)
```

For nonlinear / transient analyses, swap in
`NonlinearStaticAnalysis` / `NonlinearTransientAnalysis` and pass the
appropriate integrator and convergence test.
