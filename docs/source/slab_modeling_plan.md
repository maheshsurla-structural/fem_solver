# Slab / surface (shell) modeling — commercial-parity roadmap

*Status: **IN PROGRESS**. Successor context to the desktop commercial roadmap.
Analysis + engine are largely done; this plan is almost entirely a **desktop
GUI + data-model** effort. Work happens on branch `feat/slab-modeling`
(worktree off `main`).*

---

## 1. Where we are (the double-check)

The claim "our tool doesn't support the modelling of slabs" is **true at the
GUI / project level and false at the engine level**. The two layers have
diverged: the solver can already do slabs; the desktop model has no concept of
a surface at all.

### 1a. The engine already models slabs — and it is validated

Surface / continuum elements (all exported from `femsolver.__init__`, all with
green tests):

| Element | File | DOF/node | Role |
|---|---|---|---|
| `ShellMITC4` | `elements/shell.py` | 6 | **workhorse** general shell (SAP/ETABS/MIDAS idiom; thin+thick, no shear locking) |
| `ShellMITC9` | `elements/shell_mitc9.py` | 6 | 9-node higher-order shell |
| `ShellDKMQ4` | `elements/shell_dkmq4.py` | 6 | Discrete-Kirchhoff-Mindlin quad (thin plate) |
| `ShellTri3` | `elements/shell_tri.py` | 6 | 3-node shell (mesh transitions) |
| `ShellDKT3` | `elements/shell_dkt3.py` | 6 | Discrete-Kirchhoff triangle |
| `MembraneQ4Drilling` | `elements/membrane_drilling.py` | 3 | in-plane membrane w/ drilling DOF |
| `Quad4`, `Quad8` | `elements/plane.py` | 2 | plane stress/strain |
| `Hex8`, `Hex20`, `Tet4`, `Hex8TL` | `elements/solid.py`, `hex8_TL.py` | 3 | solids (thick slabs / detailed regions) |

Shell **sections** (`femsolver/shell_sections/`): `ElasticShellSection`
(single isotropic layer), `LayeredShellSection` / `ShellLayer` (laminate /
sandwich), `CLTSection` (cross-laminated timber), `ReinforcedConcreteShellSection`
(layered concrete + smeared rebar, `shell_moment_curvature`), and composite
`ply_failure` criteria. `ShellMITC4` accepts either `(material, thickness)` or
`section=` (the layered/RC path).

Mesh generators (`femsolver/mesh/generators.py`): `rectangle_quad4`,
`disk_quad4`, `ring_quad4`, `shell_curved_cylinder`, plus `StructuredMesh`,
quality metrics, and stress recovery.

Slab **design** already exists (`femsolver/design/`, "Theme W"):
- `two_way_slab.py` — `ddm_panel`, `ddm_minimum_thickness` (ACI Direct Design Method).
- `diaphragm.py` — `classify_diaphragm`, `rigid_transfer`, `flexible_transfer`.
- `punching.py` / `punching_reinforcement.py` — ACI 318 / EC2 / IS 456 punching capacity, demand and rebar.

Core supports it: nodes carry the model-global `ndf`, so in a **3-D model
(`ndm=3, ndf=6`)** shells and `BeamColumn3D` share the same 6 DOF/node and
assemble together (`Model.element_dof_map` takes the first `dofs_per_node`
equations per node). Rigid/semi-rigid diaphragms are expressible today via
`Model.equal_dof` / MP-constraints. `tests/test_slab_diaphragm.py`,
`test_shell_*.py`, `test_rc_layered_shell.py`, `test_quad4.py` — **82 tests
pass**.

### 1b. The desktop app models nothing but lines

`desktop/project.py` is the source of truth for what the GUI can build, and it
has **no surface entity**:

- Geometry dataclasses: `Node` and `Member` (a 2-node line: `n1`, `n2`,
  `section`, `material`, `kind` ∈ beamcolumn/cable). Nothing with 3+ nodes.
- `Section` describes a *line* cross-section only (`A`, `Iz`, `Iy`, `J`,
  `gsd_spec`). There is no thickness / shell-section property type.
- `Project.build_model()` compiles **only** `BeamColumn2D/3D` and `Truss2D/3D`.
  Not one shell element is ever instantiated from a project.
- Loads: `Load` (nodal) and `MemberLoad` (line UDL). No area/pressure load.
- Drawing/editing: `_set_mode("draw_node" | "draw_member")`, `add_member`,
  generators produce frames only. No draw-area / mesh tool.
- Rendering: `model_geometry.py` *already anticipates* surfaces —
  `_member_segments` closes the boundary loop "for elements with 3+ nodes
  (quad/shell)" and `diagram_meshes` builds face `PolyData` — but nothing ever
  creates such an element, and there are no shell **contour** results.

**Conclusion:** the work is ~90% GUI + data-model plumbing on top of a mature,
tested engine, plus one genuine engine addition (surface/pressure equivalent
loads on shells). This is a far smaller lift than "add slab FE from scratch."

---

## 2. The gap, mapped to the commercial benchmark

Benchmark = SAP2000 / ETABS / MIDAS Gen area-object workflow.

| Capability | Engine | Desktop GUI | Gap |
|---|---|---|---|
| Shell/plate finite elements | ✅ full family | ❌ | GUI only |
| Layered / RC / CLT shell sections | ✅ | ❌ | GUI only |
| 6-DOF beam+shell coexistence | ✅ (3-D) | ❌ (default 2-D, lines only) | data model + 3-D gate |
| Draw area object (quad/tri/polygon) | n/a | ❌ | new |
| Auto-meshing of area objects | ✅ generators | ❌ | wire + UI |
| Uniform area / pressure loads | ❌ (no shell `f_eq`) | ❌ | **engine + GUI** |
| Self-weight of shells | ✅ mass matrix | ❌ | wire |
| Filled/thickness rendering, local axes | ⚠️ partial (faces) | ❌ | GUI |
| Result contours (U, σ, M, V) | ✅ `recover()` resultants | ❌ | GUI |
| Section/design cuts (integrate over a line) | ❌ | ❌ | engine + GUI |
| Rigid/semi-rigid diaphragm | ✅ `equal_dof` | ❌ | GUI |
| Two-way slab / punching / diaphragm design | ✅ | ❌ | wire results → design |

---

## 3. Epics

Ordered roughly by dependency. ★ = high user-visible value.

### S0 — Surface data model ★★ (foundation)
Add to `desktop/project.py`:
- `Area` dataclass: `id`, `nodes: list[int]` (3 or 4 corner nodes; polygon in a
  later phase), `shell_section: int`, `material: int`, `kind`
  ("shell" | "plate-thin" | "plate-thick" | "membrane"), optional
  `local_axis` override, and a `mesh: MeshSpec | None` (see S3) so the drawn
  **area object** is distinct from its analysis mesh (SAP/ETABS idiom).
- `ShellSection` (a.k.a. "Thickness") dataclass: `id`, `name`, `thickness`,
  `material`, membrane/bending **modifiers** (f11,f22,f12,m11,m22,m12,v13,v23,
  mass, weight — the SAP stiffness-modifier set), and an optional `layers`
  list / `rc_spec` for the layered & RC paths.
- `Project` gains `areas` and `shell_sections` lists + accessor helpers, and
  persists them in the project JSON (bump the schema/migration like prior work).

### S1 — build_model wiring ★★
In `Project.build_model` (and `build_buckling_model`, staged, etc.): after the
member loop, emit a shell element per `Area` (`ShellMITC4` default; `ShellTri3`
for 3-node; DKMQ4 when kind="plate-thin"), resolving `ShellSection` →
`ElasticShellSection` / `LayeredShellSection` / `ReinforcedConcreteShellSection`.
Gate on `ndm==3` (see Decision D1). Apply stiffness modifiers.

### S2 — Draw & edit areas ★
- New view modes `draw_area_quad` / `draw_area_tri` / `draw_area_poly` mirroring
  `draw_member` (`main_window._set_mode`, `model_view` pick callbacks): click N
  existing/ground-plane nodes → create an `Area`.
- "Draw rectangular area" quick tool (2-corner drag on a work plane).
- Modeless property editing via the existing `PickDialog` pattern
  (`desktop/pick.py`).

### S3 — Automatic meshing ★
- `MeshSpec` on the area (n×m divisions, or target size) → subdivide via
  `mesh.generators.rectangle_quad4` at solve time, generating interior nodes /
  sub-elements and **merging** coincident boundary nodes so meshed areas stay
  compatible with adjacent frames and areas (edge-node matching).
- Mesh preview overlay + "mesh now" vs "auto-mesh on run" toggle.

### S4 — Shell section library GUI ★
- A "Thickness / Shell Sections" manager (parallel to the Section manager):
  create isotropic-thickness, layered, and RC-layered sections; set modifiers;
  live preview. Reuse Section-Designer materials.

### S5 — Area loads ★★ (**the one real engine addition**)
- Engine: add `f_eq_global()` (equivalent nodal loads) to the shell elements
  for (a) uniform out-of-plane **pressure**, (b) self-weight/body force from the
  mass matrix × g, (c) projected loads. Today shells have `M_global` +
  `recover` but no `f_eq` — pressure loads cannot be applied.
- GUI: `AreaLoad` dataclass (uniform pressure, gravity, optionally
  one-/two-way "load transfer to edges" like ETABS shell-uniform-to-frame) +
  a dialog; wire into `apply_loads`.

### S6 — Rendering ★
- Filled shaded faces for areas (extend `model_geometry` face `PolyData`;
  currently only lines are drawn for the model), thickness extrusion option,
  per-area **local axes** triads, and area selection/highlight in `model_view`.

### S7 — Result contours ★★
- Post-process `element.recover()` → per-node smoothed contours for:
  displacement, in-plane stress (σ11/σ22/σ12, von Mises) top/bottom fibre,
  bending moments (M11/M22/M12), transverse shear (V13/V23), with
  nodal-averaging vs unaveraged toggle and a legend. Reuse the existing
  diagram/units chrome. Add **section cuts** (integrate resultants along a
  drawn line) in a later sub-phase.

### S8 — Diaphragm constraints GUI
- Assign rigid / semi-rigid diaphragm to a set of nodes/areas at a level →
  `equal_dof` / master-node MP-constraints; classify via
  `design.diaphragm.classify_diaphragm`.

### S9 — Slab design integration
- Tie shell results to the existing design engine: `two_way_slab.ddm_panel`,
  `punching`/`punching_reinforcement` at columns, Wood–Armer design moments →
  required reinforcement contours, `diaphragm` transfer checks. Surface in a
  Design tab alongside the beam/column design already present.

### S10 — Robustness & parity polish
- Mesh compatibility between coarse/fine areas (edge constraints / mid-side
  tying), area copy/replicate, offsets/insertion point, openings, curved-area
  auto-mesh (`disk_quad4`/`ring_quad4`), model-check rules for areas
  (`model_checks`), and DXF/table import-export of areas.

---

## 4. Recommended sequencing

**Phase A — "a slab you can draw, load, solve, and see" (MVP):**
S0 → S1 → S4 (isotropic thickness only) → S2 (rectangular area) → S5 (pressure +
self-weight) → S6 → S7 (displacement + basic stress contour). This alone gives
a defensible "we model slabs" story and exercises the full pipeline end-to-end.

**Phase B — usable for real floors:** S3 (auto-mesh + node merge), S2 (polygon &
node-pick areas), S7 (moments M11/M22/M12, averaging), S8 (diaphragms).

**Phase C — commercial parity:** S9 (slab/punching/diaphragm design),
layered/RC/CLT sections in S4, S7 section cuts, S10 robustness.

---

## 5. Key decisions / open questions

- **D1 — 3-D only.** Slabs need `ndf=6`; the desktop default is `ndm=2, ndf=3`.
  Recommend: areas are a **3-D feature** — creating one requires (or offers to
  convert to) a 3-D project. A 2-D project keeps behaving exactly as today.
- **D2 — Default element.** `ShellMITC4` (thin+thick, general) as the default,
  exposing DKMQ4 as "plate-thin" and `MembraneQ4Drilling` as "membrane", to
  mirror SAP's Shell-thin/Shell-thick/Membrane/Plate menu.
- **D3 — Area object vs analysis mesh.** Follow SAP/ETABS: the user draws one
  *area object*; meshing is a property resolved to elements at solve time
  (keeps the model editable and the tree clean).
- **D4 — Pressure `f_eq`.** The only engine code to write. Scope it to the shell
  family used (start with `ShellMITC4`/`ShellTri3`), consistent-load formulation,
  with tests against closed-form plate reactions.
- **D5 — Storage/units.** Areas/thicknesses stored in SI (like the rest of the
  model); shell result quantities need new unit categories (stress, moment/length,
  force/length) in the units layer.

---

## 6. Tracker

| ID | Item | Layer | Status |
|---|---|---|---|
| S0 | `Area` + `ShellSection` data model | GUI | ✅ 9af7a54 (8 tests) |
| S1 | build_model emits shells | GUI | ✅ (10 tests) |
| S2 | Draw/edit areas (rect → poly) | GUI | ☐ |
| S3 | Auto-mesh + node merge | GUI | ☐ |
| S4 | Shell-section (thickness) manager | GUI | ☐ |
| S5a | Shell `f_eq` pressure/body loads | **Engine** | ☐ |
| S5b | `AreaLoad` + dialog + apply_loads | GUI | ☐ |
| S6 | Filled faces / local axes / selection | GUI | ☐ |
| S7 | Result contours (U/σ/M/V) + cuts | GUI | ☐ |
| S8 | Diaphragm assignment | GUI | ☐ |
| S9 | Slab/punching/diaphragm design | GUI+wire | ☐ |
| S10 | Meshing robustness & parity polish | both | ☐ |
