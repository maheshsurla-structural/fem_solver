# Slab / surface (shell) modeling — commercial-parity roadmap

*Status: **CORE ROADMAP COMPLETE (S0–S10) + PUSHED to `origin/main`.** The
desktop app now models slabs end-to-end, matching the SAP2000/ETABS/MIDAS
area-object workflow. Last published commit: **`7f28b58`** (2026-09-18);
`origin/main` == local `main`. ~150 slab-specific tests green; the only failing
tests in the suite are 4 pre-existing order-8 quadrature failures unrelated to
slabs.*

---

## 0. Resume here (session hand-off)

**What ships today (all on `origin/main`, GUI is 3-D-only for areas):**
- **Model:** Draw ▸ Area (click-to-draw: click corners, click the first again to
  close — tri/quad/**polygon**) + Area… dialog; Home ▸ Thickness (shell-section
  manager); auto-mesh (quad n×m; polygon = centroid-fan tris) with coincident-
  node merge; click-select / highlight / delete / inspect areas.
- **Loads:** Loads ▸ Area — gravity, normal pressure, or **self-weight (ρ·t·g)**.
- **Constraints:** Home ▸ Constraints ▸ Diaphragm (rigid floor).
- **Results:** Results ▸ Deflection; Shell F/M (M11/M22/M12, N, Vmax, Wood-Armer
  design moments — GP→node-extrapolated peaks); Slab rebar (required As);
  Punching (ACI); Section cut (design strip M/V); Export slab (CSV).
- **Engine additions:** shell `f_eq` (surface/pressure loads, `elements/shell*.py`)
  and `femsolver.design.wood_armer`. Everything else is desktop GUI + data model.

**How to work:** isolated git worktree `.claude/worktrees/slab-modeling` on
branch `feat/slab-modeling` (== `main`). Tests:
`PYTHONPATH=src QT_QPA_PLATFORM=offscreen <repo>/.venv-gui/Scripts/python -m
pytest tests/ -q` (run from the worktree/repo root). Cadence: work on the
branch → merge each item to `main` (FF) → push. **Gotcha:** patch modal
`QMessageBox` in headless tests or they hang.

**Remaining backlog (optional; pick one to resume):**
1. **Beam-to-meshed-edge compatibility** — auto-subdivide beams lying along a
   meshed slab edge so they share the edge mesh nodes (a real correctness fix).
   *Higher risk:* touches `build_model` member emission + member-load/results
   wiring (member.id is the element tag; splitting needs a member→sub-tags map).
   **Detailed scope + epic breakdown (BE0–BE7): see §3 "BE — Beam-to-slab-edge
   compatibility".** Foundation slice **BE0–BE2 prototyped** (geometric splitter
   + shared-node merge, behind a toggle); BE3 results aggregation is the next
   real work.
2. **Non-convex polygon meshing** — the current centroid-fan assumes convex
   polygons; an L-shape needs ear-clipping / constrained triangulation.
3. **DXF export** of the slab geometry/mesh (reuse `femsolver.results.dxf`).
4. **Diaphragm rigid/flexible classification** per ASCE 7 via
   `design.diaphragm.classify_diaphragm` (needs a lateral-analysis deflection +
   drift extraction).
5. **AreaDialog for polygons** — the add-area *dialog* still has 4 fixed corner
   slots (polygons are created via the draw tool); a variable node-list input
   would let polygons be typed/edited too.

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

### BE — Beam-to-slab-edge compatibility (detailed scope)

**Problem.** `build_model` emits one `BeamColumn` per `Member`
([project.py] `build_model`), then meshes each `Area` into an `n1×n2` grid,
merging coincident nodes by `_coord_key`. A beam along a slab edge connects only
the slab's two *corner* nodes; the slab's *interior edge* nodes (n−1 per meshed
edge) never appear on the beam, so the slab edge can deflect relative to the
beam — no compatibility, no load transfer at those points.

**What we can reuse.**
- The slab mesher's coordinate-keyed node registry auto-merges anything placed
  at a slab node's coordinate — so a beam node created there is *automatically*
  stitched to the slab.
- The engine's constraints package (`femsolver.constraints`): `EqualDOF`,
  `RigidLink`, `RigidOffset`, `MPConstraint`, and the ready-made
  `beam_shell_offset_coupling(model, beam_node, shell_node)` for the *eccentric*
  (offset slab / composite T-beam) case, via `model.add_mp_constraint`.
- Precedent: `area_element_tag`/`_area_element_tags` (object→sub-tags map) and
  `build_buckling_model` (already subdivides members).

**Two approaches.**
- **A — split the edge beam at the slab edge nodes (shared-node).** Insert a
  beam node at each slab edge-node coordinate on the beam and emit consecutive
  `BeamColumn` sub-elements; the merge registry makes beam+slab share the nodes.
  Exact physics, works with the current linear solver, no constraint math. Cost:
  new sub-element tags ripple into the ~15 files that read forces by member id;
  needs build re-ordering + load/hinge distribution.
- **B — tie slab nodes to the beam with MPCs (keep beam whole).** Preserves beam
  identity, but `RigidOffset`/`RigidLink` tie rigidly to *one* master node —
  correct only at a beam *joint*, not mid-span (over-stiffens, ignores bending).
  A mid-span interpolation MPC does not exist yet.

**Decision.** A is the foundation (you need a beam node at each slab-edge-node
plan position regardless). Centroidal beams → shared nodes via merge, zero
constraints. Eccentric/composite slabs → split for plan-coincident nodes, then
`beam_shell_offset_coupling` per node carries the vertical offset (v2 layer).

| Epic | Work | Risk |
|---|---|---|
| **BE0** | Two-pass build: pre-compute + register all slab edge-node coordinates *before* the member loop, so members can split against them. | Med |
| **BE1** | Detection — slab edge nodes collinear with and interior to a member (tolerance point-on-segment, ordered). v1: beam endpoints == two adjacent area corners (beam *is* the edge). | Med |
| **BE2** | Splitter — emit `k` `BeamColumn` sub-elements through those nodes; deterministic `member_element_tag(id, j)` stride; inherit section/material/kind. | Med |
| **BE3** | Results/design aggregation — `member_element_tags(mb)` map + shim so the ~15 consumers see one member (concatenate diagrams, envelope for design, i/j end forces from first/last sub-element). **Largest surface.** | **High** |
| **BE4** | Load distribution — member UDL/line loads to sub-elements by span fraction; nodal loads unaffected. | Med |
| **BE5** | Hinges — map member-end hinges to correct sub-element ends; v1 may forbid splitting a hinged member (clear message). | Low |
| **BE6** *(v2)* | Eccentric/composite — tie slab nodes to beam nodes with `beam_shell_offset_coupling`; spike first. | Med |
| **BE7** | Tests — patch test (SS beam under slab UDL vs. fine reference), slab-on-beam deflection convergence, and a **regression guard that pure-frame results are unchanged** when no slab touches the beam. | Med |

**Biggest risk: BE3.** Splitting changes a beam's element identity; 15 desktop
files read forces by member id (results/design/diagrams/end-forces). Gate the
whole feature behind an "auto-connect beams to slab mesh" toggle (default on).
BE0–BE2 (centroidal, in-plane compatibility) is a self-contained first slice
that already delivers correct physics; BE3+BE4 are the bulk; BE6 is a clean
follow-on on the existing constraint primitive.

---

## 4. Recommended sequencing

**Phase A — "a slab you can draw, load, solve, and see" (MVP) — ✅ DONE.**
S0 → S1 → S4 (isotropic thickness) → S2 (area dialog) → S5 (pressure +
gravity) → S6 (filled faces) → S7 (displacement contour). Merged to `main`.

**Phase B — usable for real floors — ✅ mostly done.** S3 (auto-mesh + node
merge) ✅, S7 rich (moments M11/M22/M12, membrane N, shear — nodal-averaged) ✅.
**Remaining: S8 (rigid diaphragms) ← next**, then the S2/S6 UX refinements
(polygon & click-to-draw areas, area selection + local-axis triads).

**Phase C — commercial parity:** S9 (slab/punching/diaphragm design wiring —
tie shell results to the existing `design.two_way_slab` / `punching` / `diaphragm`
engine), layered/RC/CLT sections in S4, S7 refinements (GP→node extrapolation
for sharp support peaks, **section cuts**), S10 robustness (beam-to-meshed-edge
compatibility, area copy/replicate, openings, curved-area mesh, model-checks,
DXF/table import-export of areas).

### Immediate next items (in order)
1. **S8 — Rigid diaphragms.** ✅ done — `Diaphragm` data model + `build_model`
   wiring (auto master node at centroid, out-of-plane DOFs pinned,
   `RigidDiaphragm` MP-constraint) + "Add diaphragm…" GUI (Home ▸ Constraints,
   seeded from the selected joints).
2. **S9 — Slab design wiring.** Feed shell design moments (Wood–Armer from
   M11/M22/M12) to `design.two_way_slab` / `punching_reinforcement`; a Design
   tab panel + required-reinforcement contour.
3. **S7 refinement.** GP→node extrapolation (sharper clamped-edge peaks) +
   section cuts (integrate resultants along a drawn line).

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
| S2 | Draw/edit areas (`AreaDialog` + click-to-draw mode) | GUI | ✅ (7+5 tests; viewport draw_area quad/tri); polygon areas later |
| S3 | Auto-mesh + coincident-node merge | GUI | ✅ (7 tests, converges to analytical) |
| S4 | Shell-section (thickness) manager | GUI | ✅ (6 tests) |
| S5a | Shell `f_eq` pressure/body loads | **Engine** | ✅ (6 tests) |
| S5b | `AreaLoad` + dialog + apply_loads (gravity/pressure/**self-weight** ρ·t·g) | GUI | ✅ (12 tests) |
| S6 | Filled faces + local-axis triads + **area selection** | GUI | ✅ (`areas_mesh`, `area_local_axes`, click-select/highlight/delete/props via `nearest_item`+`area_faces_mesh`; 6 tests) |
| S7 | Result contours (U/σ/M/V) + cuts | GUI | ✅ displacement + M/N/V + Wood-Armer + GP→node extrapolation + **section cuts** (`section_cut`, design-strip M/V validated vs wL²/8·B; 5 tests) — COMPLETE |
| S8 | Diaphragm assignment (rigid) | GUI | ✅ (9 tests; RigidDiaphragm, auto master, rigid-floor solve) |
| S9 | Slab/punching/diaphragm design | GUI+wire | ✅ Wood-Armer moments + required-As contour + **punching check** (ACI, demand from node reaction; `slab_punching.py`, 29 tests). Diaphragm classification later |
| S10 | Meshing robustness & parity polish | both | ◑ area selection ✅; slab-results CSV export ✅; **polygon areas ✅** (centroid-fan tris, click-to-draw close-on-first); beam-edge/DXF later |
| BE0–BE2 | Beam-to-slab-edge: two-pass build + detection + splitter (shared-node) | GUI | ◑ prototyped (splitter emits sub-elements through slab edge nodes, merged via `_coord_key`; behind `auto_connect_beams` toggle) |
| BE3 | Member↔element results/design aggregation | GUI | ◑ prototyped — `Project.member_element_tags(mb)` + `decode_member_id(tag)`; design (`design_all`/`design_envelope`) resolves sub-elements to their member; viewport selection (`nearest_item`, window/polygon) reports the member id, and highlight/framing cover all sub-elements. Force diagrams already worked (iterate all elements). Consumers found to be **2-D-only** (temp-gradient, construction-stages) can't see splits. 13 tests |
| BE4 | Member line-load distribution over a split member | GUI | ◑ prototyped — `_apply_member_load` applies the UDL *intensity* (N/m) to every sub-element via `member_element_tags`; each element integrates over its own length, so the total load is exact (no span-fraction weighting needed — proven by test). Factor-scaled; unsplit path unchanged. 4 tests |
| BE5 | Hinges (and cables) vs splitting | GUI | ◑ prototyped — single `_splittable(mb)` predicate (used by build, the tag map, and load distribution) refuses to split a **hinged** or **cable** member; a lumped hinge assumes one element end-to-end. **Finding:** lumped fiber hinges are 2-D-only (nonlinear.py) and splitting is 3-D-only, and `build_nonlinear_model` never splits — so the two are orthogonal today; this guard just future-proofs 3-D hinges. 2 tests |
| BE7 | Patch + convergence validation | GUI | ◑ prototyped — auto-split ≡ hand-built pre-subdivision (identical solved displacements to 1e-12); slab-on-beam midspan deflection converges under mesh refinement (2% at 8→16 edge divisions). 2 tests |
| BE6 | Composite/eccentric T-beam (`beam_shell_offset_coupling`) | both | ☐ (split for plan-coincident nodes, then tie each slab node to its beam node with the vertical offset — the one item needing the engine's constraint package; **needs a modeling-workflow decision: how the user specifies a composite beam / its offset**) |
