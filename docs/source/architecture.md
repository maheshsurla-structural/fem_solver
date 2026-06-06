# femsolver architecture & naming guide

**Status:** planning document — *no files have been moved.* This is the
agreed target organization and naming convention for femsolver, to be
executed later in small, test-validated increments. The public
`import femsolver` API will be preserved exactly throughout any
migration.

**Last updated:** 2026-06-05

---

## 1. Organizing principle

A commercial structural-FE product is organized as a **layered model
tree**, in the order an engineer builds and runs a model:

```
  kernel  →  modeling  →  loads / hazard  →  analysis  →  design  →  results
```

* **Kernel** — the domain container, numbering, equation solving.
* **Modeling** — what the structure *is*: materials, sections, elements.
* **Loads / hazard** — what acts on it: patterns, combinations,
  prestress, thermal, seismic & wind demand.
* **Analysis** — how it responds: the solver drivers.
* **Design** — code checks on the response.
* **Results** — outputs: plots, files, reports, deliverables.

Specialist domains (bridges, geotechnical, thermal, performance/PBE,
reliability) sit alongside as self-contained packages. Reference data
(section/material/bolt catalogues) and the V&V benchmark suite are
support packages.

The guiding rule: **a newcomer should be able to guess which package a
feature lives in from its engineering role**, and **no package should
mix unrelated domains**.

---

## 2. Current package map

| Package | Files | Responsibility | Assessment |
|---|---:|---|---|
| `core` | 3 | `Model`, `Node`, container | ✅ clean |
| `numerics` | 3 | quadrature, DOF numbering | ✅ clean |
| `constraints` | 7 | MP constraints (rigid link/diaphragm, equalDOF) | ✅ clean |
| `materials` | 14 | constitutive (root: elastic + multiaxial + damage + thermal + time) | 🟡 flat, mixed |
| `materials/uniaxial` | 14 | 1-D fiber materials | ✅ clean |
| `materials/timber` | 5 | NDS/EC5/IS 883 timber | ✅ clean |
| `sections` | 14 | section ABC, elastic, fiber, shell, wall | ✅ exemplary |
| `sections/{catalogue,geometry,hinges,parametric}` | 14 | factories + helpers | ✅ clean |
| `elements` | 25 | element library (flat) | 🟡 flat |
| `analysis` | **40** | solver kernel **+** drivers **+** dynamics **+** geotech **+** thermal **+** seismic/PBE **+** loads | ❌ overloaded |
| `design` | 14 | code design (root: ec/is/punching/psc/reports loose) | 🟡 loose root |
| `design/{concrete,steel,seismic,timber,connections}` | 31 | per-material design | ✅ clean |
| `seismic` | 8 | seismic **hazard** (GMPE, PSHA, deagg, site response) | 🟡 name overlaps design/seismic |
| `wind` | 7 | wind loads (ASCE 7 / IS 875 / EC1) | ✅ clean |
| `bridges` | 11 | influence lines, tendons, cable, staged, PSC | ✅ clean |
| `reliability` | 5 | FORM / SORM / Monte-Carlo | ✅ clean |
| `catalogs` | 5 | raw data tables (bolts, material grades, EC/IS sections) | 🟡 name clash with `sections/catalogue` |
| `io` | 6 | VTK, JSON deck, diagrams, extract, HDF5 | 🟡 overlaps postproc/deliverables |
| `postproc` | 3 | matplotlib + PyVista plotting | 🟡 (see above) |
| `deliverables` | 5 | calc-sheet, DXF, BOM, QA | 🟡 (see above) |
| `mesh` | 4 | structured mesh generators, quality metrics | ✅ clean |
| `benchmarks` | 8 | V&V suite (NAFEMS etc.) | ✅ clean |

### The six issues to fix

1. **`analysis/` mixes six domains** (kernel / drivers / dynamics /
   geotech / thermal / seismic-PBE / loads).
2. **Seismic scattered** across `seismic/` (hazard), `analysis/`
   (response-spectrum, IDA, pushover, P-58), `design/seismic/`.
3. **Three output packages** (`io`, `postproc`, `deliverables`) with
   overlapping roles.
4. **`catalogs/` vs `sections/catalogue/`** — near-identical names
   (raw data vs section factories).
5. **`materials/` root flat** — multiaxial / damage / hyperelastic /
   thermal / time not sub-grouped.
6. **Naming nits** (see §5).

---

## 3. Target architecture

```
femsolver/
  # ---- kernel ----
  core/            Model, Node
  numerics/        quadrature, dof_numbering
  constraints/     MP constraints + handlers (move constraint_handler here)

  # ---- modeling ----
  materials/
    uniaxial/      1-D fiber materials                 (as-is)
    multiaxial/    j2_plasticity, finite_j2, drucker_prager,
                   mohr_coulomb, cam_clay, hyperelastic, orthotropic
    concrete/      concrete_damage, concrete_damage_plasticity, concrete_time
    timber/        (as-is)
    elastic.py base.py thermal.py                      (stay at root)
  sections/        (unchanged — already exemplary)
  elements/        line/ surface/ solid/ link/ thermal/   (optional sub-grouping)

  # ---- loads / hazard ----
  loads/           patterns + combinations (from analysis/loads.py),
                   initial_stress, thermal_strain, time_dependent,
                   moving (from bridges/influence + moving_load)
  hazard/
    seismic/       gmpe, bssa14, psha, deaggregation, site_response,
                   equivalent_linear, risk_targeted, response_spectrum
    wind/          (from top-level wind/)

  # ---- analysis (solver only) ----
  analysis/
    kernel/        assembler, parallel_assembler, solvers, eigen,
                   substructure, algorithm, convergence, integrator,
                   transient_integrator
    drivers/       linear_static, nonlinear_static, buckling, transient,
                   nonlinear_transient, modal_pushover, envelope
    dynamics/      damping  (+ response_spectrum if not in hazard)

  # ---- specialist domains ----
  geotech/         ssi, soil_springs, winkler, pile_group,
                   liquefaction, dynamic_gazetas
  thermal/         heat_conduction, thermal_strain (shared w/ loads), fire
  performance/     ida, ida_collapse, fragility, record_scaling, cms,
                   p58, capacity_design, drift_check
  bridges/         (as-is)
  reliability/     (as-is)

  # ---- design ----
  design/
    concrete/ steel/ timber/ seismic/ connections/     (as-is)
    psc.py punching.py two_way_slab.py diaphragm.py reports.py
    codes/         ec2, ec3, ec8, is456, is800, is1893, is13920  (group code files)

  # ---- support ----
  data/            (= renamed catalogs/) bolts, materials, sections_ec, sections_is
  results/         (= io/ + postproc/ + deliverables/)
    files:   vtk_io, json_io, ida_hdf5, extract
    plots:   plot (→ plot_2d), plot_3d
    reports: diagrams, calc_sheet, dxf, bom, qa
  mesh/            (as-is)
  benchmarks/      (as-is)
```

---

## 4. Module migration map

The actionable core. Each row = one move; the public API entry in
`femsolver/__init__.py` is updated to re-import from the new path so
the top-level API is unchanged.

### analysis/ → split

| Current | Target |
|---|---|
| `analysis/assembler.py` | `analysis/kernel/assembler.py` |
| `analysis/parallel_assembler.py` | `analysis/kernel/parallel_assembler.py` |
| `analysis/solvers.py` | `analysis/kernel/solvers.py` |
| `analysis/eigen.py` | `analysis/kernel/eigen.py` |
| `analysis/substructure.py` | `analysis/kernel/substructure.py` |
| `analysis/algorithm.py` | `analysis/kernel/algorithm.py` |
| `analysis/convergence.py` | `analysis/kernel/convergence.py` |
| `analysis/integrator.py` | `analysis/kernel/static_integrator.py` *(rename)* |
| `analysis/transient_integrator.py` | `analysis/kernel/transient_integrator.py` |
| `analysis/constraint_handler.py` | `constraints/handlers.py` *(rename)* |
| `analysis/linear_static.py` | `analysis/drivers/linear_static.py` |
| `analysis/nonlinear_static.py` | `analysis/drivers/nonlinear_static.py` |
| `analysis/buckling.py` | `analysis/drivers/buckling.py` |
| `analysis/transient.py` | `analysis/drivers/transient.py` |
| `analysis/nonlinear_transient.py` | `analysis/drivers/nonlinear_transient.py` |
| `analysis/modal_pushover.py` | `analysis/drivers/modal_pushover.py` |
| `analysis/envelope.py` | `analysis/drivers/envelope.py` |
| `analysis/damping.py` | `analysis/dynamics/damping.py` |
| `analysis/response_spectrum.py` | `hazard/seismic/response_spectrum.py` |
| `analysis/loads.py` | `loads/combinations.py` *(rename; LoadPattern + ASCE-7 combos)* |
| `analysis/initial_stress.py` | `loads/initial_stress.py` |
| `analysis/thermal_strain.py` | `loads/thermal_strain.py` (+ re-export in `thermal/`) |
| `analysis/time_dependent.py` | `loads/time_dependent.py` |
| `analysis/heat_conduction.py` | `thermal/heat_conduction.py` |
| `analysis/fire.py` | `thermal/fire.py` |
| `analysis/ssi.py` | `geotech/ssi.py` |
| `analysis/soil_springs.py` | `geotech/soil_springs.py` |
| `analysis/winkler.py` | `geotech/winkler.py` |
| `analysis/pile_group.py` | `geotech/pile_group.py` |
| `analysis/liquefaction.py` | `geotech/liquefaction.py` |
| `analysis/dynamic_gazetas.py` | `geotech/dynamic_gazetas.py` |
| `analysis/capacity_design.py` | `performance/capacity_design.py` |
| `analysis/ida.py` | `performance/ida.py` |
| `analysis/ida_collapse.py` | `performance/ida_collapse.py` |
| `analysis/fragility.py` | `performance/fragility.py` |
| `analysis/record_scaling.py` | `performance/record_scaling.py` |
| `analysis/cms.py` | `performance/cms.py` |
| `analysis/p58.py` | `performance/p58.py` |
| `analysis/drift_check.py` | `performance/drift_check.py` |

### materials/ → sub-group

| Current | Target |
|---|---|
| `materials/j2_plasticity.py` | `materials/multiaxial/j2.py` *(rename)* |
| `materials/finite_j2.py` | `materials/multiaxial/j2_finite_strain.py` *(rename)* |
| `materials/drucker_prager.py` | `materials/multiaxial/drucker_prager.py` |
| `materials/mohr_coulomb.py` | `materials/multiaxial/mohr_coulomb.py` |
| `materials/cam_clay.py` | `materials/multiaxial/cam_clay.py` |
| `materials/hyperelastic.py` | `materials/multiaxial/hyperelastic.py` |
| `materials/orthotropic.py` | `materials/multiaxial/orthotropic.py` |
| `materials/concrete_damage.py` | `materials/concrete/isotropic_damage.py` *(rename)* |
| `materials/concrete_damage_plasticity.py` | `materials/concrete/plastic_damage.py` *(rename)* |
| `materials/concrete_time.py` | `materials/concrete/time_dependent.py` *(rename)* |
| `materials/{elastic,base,thermal}.py` | stay at `materials/` root |

### top-level → regroup

| Current | Target |
|---|---|
| `seismic/*` (hazard) | `hazard/seismic/*` |
| `wind/*` | `hazard/wind/*` |
| `catalogs/*` | `data/*` *(rename package)* |
| `io/* + postproc/* + deliverables/*` | `results/{files,plots,reports}/*` |
| `design/{ec2,ec3,ec8,is456,is800,is1893,is13920}.py` | `design/codes/*` |
| `bridges/influence.py`, `bridges/moving_load.py` | `loads/moving.py` (+ re-export in bridges) |

---

## 5. Naming conventions

**Modules** — `snake_case.py`, named for the *thing*, not the phase.
**Classes** — `CamelCase`, dimension suffix where relevant (`Truss2D`,
`Hex8`, `J2Plasticity3D`). **Functions** — `snake_case`, verb-first for
actions (`apply_thermal_load`), noun for queries (`creep_compliance`).
**Result dataclasses** — `<Thing>Result` (`StagedConstructionResult`).

### Specific renames (disambiguation)

| Current | Rename to | Why |
|---|---|---|
| `analysis/integrator.py` | `static_integrator.py` | it holds *static* integrators (LoadControl, ArcLength); paired with `transient_integrator.py` |
| `analysis/loads.py` | `loads/combinations.py` | it's load *combinations* (ASCE 7) + patterns, not element loads |
| `materials/finite_j2.py` | `j2_finite_strain.py` | clearer than the abbreviation |
| `materials/concrete_damage.py` | `concrete/isotropic_damage.py` | distinguish from plastic-damage |
| `materials/concrete_damage_plasticity.py` | `concrete/plastic_damage.py` | Lubliner-Lee-Fenves |
| `materials/concrete_time.py` | `concrete/time_dependent.py` | strength/E gain curves |
| `analysis/constraint_handler.py` | `constraints/handlers.py` | belongs with the constraints it serves |
| `catalogs/` (package) | `data/` | removes clash with `sections/catalogue/` |

No public class/function is renamed — only modules/packages move, so
`from femsolver import X` keeps working.

---

## 6. Migration strategy (when executed)

**Invariant:** the top-level `femsolver/__init__.py` public API
(~200 names) stays byte-for-byte identical; only the *paths it imports
from* change. Deep imports (`from femsolver.analysis.ssi import …`) in
internal modules and tests are updated in lockstep.

**Increment order** (each = one commit; full 2365-test suite must pass
before the next):

1. `analysis/` → `kernel/` + `drivers/` + `dynamics/` sub-packages
   (largest churn, highest payoff; add `__init__` re-exports so
   `femsolver.analysis.X` still resolves during transition).
2. Carve out `geotech/` and `thermal/` from `analysis/`.
3. Create `loads/` (combinations, initial_stress, thermal_strain,
   time_dependent, moving).
4. Create `hazard/` (seismic + wind) and `performance/` (PBE).
5. Consolidate `io/ + postproc/ + deliverables/` → `results/`.
6. `materials/` sub-grouping + `catalogs/` → `data/` + module renames.
7. `design/codes/` grouping.

**Risk controls:** keep transitional re-export shims at old paths until
the suite is green, update tests per increment, and run
`python -m pytest -q` after every move. Examples (`examples/*.py`) and
docs are updated last.

**Not in scope of the move:** behaviour, formulas, test assertions —
this is pure relocation + renaming. Any logic change is a separate task.
