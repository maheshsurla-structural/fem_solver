# Wall (shear-wall / pier) modeling — commercial-parity roadmap

*Status: **IN PROGRESS — W0 + W1a + W2 complete** (branch `feat/wall-modeling`).
This roadmap takes the desktop app from "a wall is just a vertical `Area`"
to an ETABS-style wall workflow: a labeled **wall / pier / spandrel** object, a
**story** context to draw and stack it in, automatic **pier force integration**,
and a **wall design** check wired to the reinforcement the engine already knows
how to size. As with the slab work, the split is almost entirely
**GUI + data-model**, not engine — the solver mechanics already exist.*

---

## 0. Resume here (session hand-off)

**W0 + W1a shipped** on branch `feat/wall-modeling` (worktree
`.claude/worktrees/wall-modeling`). W0 (`da4f925`): `Area.role` (slab|wall|shell)
+ optional `pier`/`spandrel` labels, JSON round-trip + legacy migration,
`Project` wall/pier accessors, `tests/test_desktop_wall_model.py`. W1a
(`5e9a721`): Draw ▸ Wall extrudes a selected base line upward into vertical wall
panels (`desktop/walls.py`, `editing.WallDialog`, `MainWindow.draw_wall`),
`tests/test_desktop_wall_draw.py`. W2 (`9d8ada7`): pier force integration —
`desktop/piers.py` + Results ▸ Wall ▸ Pier forces (`pier_forces_dialog.py`),
`tests/test_desktop_pier_forces.py`. **Next: W3** (wall design — wire pier P/V/M
to the concrete-design machinery), or **W1b** (elevation draw plane). Note: the
GUI venv lives in the **main** repo, so run pytest with
`PYTHONPATH=src QT_QPA_PLATFORM=offscreen
/c/Mahesh/fem_solver/.venv-gui/Scripts/python -m pytest ...`. Tests:
`PYTHONPATH=src QT_QPA_PLATFORM=offscreen <repo>/.venv-gui/Scripts/python -m
pytest tests/ -q` (from the worktree root). Cadence (same as slab stream): work
on the branch → merge each epic to `main` (FF) → push. **Gotcha:** patch modal
`QMessageBox` in headless tests or they hang.

**The one-line thesis:** the wall *physics* is done; what's missing is the
*object model, the drawing context, and the design read-out* that let a
structural engineer treat a wall as a wall instead of as an anonymous vertical
plate. W0–W3 alone (label + draw + pier forces + design) turn this into a
credible wall tool with almost no engine work. W4 (stories/grids) is the larger
building-modeling investment.

---

## 1. Where we are (the double-check)

The claim "our tool can't model walls like ETABS" is **true at the GUI /
project level and mostly false at the engine level** — the same divergence the
slab stream found.

### 1a. The engine already models walls — and it is validated

| Capability | File | Notes |
|---|---|---|
| Continuum shell wall | `elements/shell.py` (`ShellMITC4` + family) | membrane + bending + drilling; the ETABS "shell/membrane wall" idiom. **Membrane forces N11/N22/N12 already contoured in the GUI.** |
| Macro fiber wall (2-D) | `sections/response/wall.py` (`wall_section_2d`) | confined boundary elements + unconfined web + smeared vertical rebar → drives a vertical `BeamColumn2D`. PERFORM-3D / OpenSees `Wall` idiom. |
| 3-D wall cross-sections | `sections/response/wall.py` | `t_wall_section_3d`, `l_wall_section_3d`, `i_wall_section_3d`, `u_wall_section_3d`. |
| Squat-wall shear flexibility | `sections/response/wall_shear.py` | closed-form flexure+shear stiffness, ACI 318-19 cracked factors, ASCE 41-17 nonlinear factors, shear-spring helper. |
| Coupled walls + coupling beams | `elements/coupling_beam.py` | rigid face offsets + coupling-beam element; example `examples/45_coupled_wall_pushover.py`. |
| Fiber P-M-M interaction (for pier design) | `design/concrete/biaxial.py`, `column.py` | reusable for a pier boundary/section check. |
| Boundary-element confinement, capacity shear, SCWB | `design/seismic/confinement.py`, `capacity_shear.py`, `scwb.py`; `design/is13920.py`; `design/ec8.py` | detailing checks — **present but not wired to any GUI wall**. |
| ~30 wall tests | `tests/test_walls.py` | all green. |

### 1b. The desktop app has no wall — only a generic vertical plate

- `desktop/project.py` `Area` (line 218) is a generic 3/4-corner shell. A wall
  is simply an `Area` that happens to be vertical — no role, no label, no
  height/story, no pier grouping.
- Drawing is on the **ground plane** (`main_window.py` `draw_node` help text) —
  so placing a vertical wall means hand-placing elevated nodes, then forming an
  area from them. Clunky vs. ETABS "draw wall on elevation / by line".
- Results expose membrane N11/N22 + Wood-Armer **slab** design moments
  (`model_geometry.py:138`), and a manual **section cut / design strip**
  integrator (`section_cut_dialog.py`, `model_geometry.py:634`) — the closest
  thing to a pier force, but transient and slab-oriented.
- **No** Story/Level object, **no** grid system, **no** pier/spandrel label,
  **no** wall design read-out. ("storey" appears only in the frame generator and
  a lateral-load helper — `desktop/generators.py`, `editing.py`.)

---

## 2. The gap, mapped to the ETABS benchmark

| ETABS concept | Here today | Epic |
|---|---|---|
| **Wall object** distinct from slab, carrying a role | ❌ generic `Area` | **W0** |
| **Pier / Spandrel label** + persistent section grouping | ⚠️ manual section cut only | **W0 / W2** |
| **Draw wall** in elevation or by line + height | ⚠️ ground-plane node placement | **W1** |
| **Automatic pier force integration** (P, M, V per pier per story) | ⚠️ one-off section cut | **W2** |
| **Wall design** — ACI 318 §18.10 boundary elements, P-M pier check, distributed/shear reinf; IS 13920 / EC8 detailing | ❌ engine only, unwired | **W3** |
| **Story / Level system** + elevation views + grid | ❌ none | **W4** |
| **Openings** (doors / windows) with opening-aware meshing | ⚠️ polygon areas, not opening-aware | **W5** |
| Membrane N11/N22/N12 + in-plane shear contours | ✅ present | — |
| Wall self-weight, in-plane & pressure loads | ✅ present | — |
| Rigid diaphragm coupling | ✅ present | — |
| Coupled-wall / coupling-beam macro model | ✅ engine (no GUI) | **W6** *(v2)* |

---

## 3. Epics

Star = priority / leverage. W0–W3 are the "credible wall tool" core and are
mostly glue over existing engine capability. W4 is the big modeling investment.
W5/W6 are v2.

### W0 — Wall / pier data model ★★ (foundation)
- Add a lightweight role tag to `Area` (e.g. `role: str = "slab"` with
  `"wall" | "slab" | "shell"`, or a dedicated `Wall`/`Pier` grouping object that
  references a set of area ids). **Decision — see §5.**
- `pier: str | None` and `spandrel: str | None` label fields (ETABS lets one
  wall carry a pier *and* a spandrel label across stories).
- Persist in the JSON round-trip; migrate old files (absent → `"slab"`).
- No engine change. Small, unlocks everything below.

### W1 — Draw & edit walls ★★
- **W1a — DONE** (`5e9a721`): **Draw ▸ Wall** extrudes a selected base line
  (2+ nodes, or members whose ends form the base) upward by a height into
  vertical quad wall panels (`role="wall"` + optional pier label). Pure builder
  `walls.build_wall_line` (one panel per base segment, shared top nodes, mesh =
  base × height), `editing.WallDialog`, `MainWindow.draw_wall` (selection-driven,
  undo/redo-aware), `wall` icon. Reuses the S3 quad mesher + area select/delete/
  properties path. `tests/test_desktop_wall_draw.py`.
- **W1b — TODO**: a true **draw-plane / work-plane** selector (XY ground,
  **XZ / YZ elevation**, or a picked 3-point plane) so base nodes can be placed
  *off* the ground plane by clicking — today the base line comes from selection.
  Touches `model_view._world_on_ground` (VTK ray→plane; currently hard-wired to
  z = 0). Higher-risk viewport work; deferred so W2/W3 can proceed.

### W2 — Pier force integration ★★ (the most-used ETABS wall output) — **DONE** (`9d8ada7`)
- `desktop/piers.py` `pier_forces()` integrates the meshed shell membrane
  stresses into **P (axial, +tension), V (in-plane shear), M (about the pier
  centroid)** at each mesh-row elevation. Each wall element's local membrane
  resultant is rebuilt as a global tensor (`model_geometry._area_frame`) and the
  traction on a horizontal cut (normal +Z) is summed element-wise across the
  cut width. Validated by free-body equilibrium (base cut = total applied load
  above).
- `pier_forces_dialog.PierForcesDialog` — Results ▸ Wall ▸ Pier forces: per-pier
  P/V/M table + shear/moment elevation diagram (display units) + CSV export.
- `MainWindow.show_pier_forces` + `act_pier_forces` in a new Results "Wall"
  ribbon group. `tests/test_desktop_pier_forces.py`.
- Cut elevations default to mid-height of each mesh row (a profile without a
  Story object — that's W4). Engine unchanged.

### W3 — Wall design ★★
- Wire pier P-M-V to the existing concrete-design machinery:
  - **Boundary-element / section P-M check** via `design/concrete/biaxial.py`
    (+ `column.py`) — does the pier section pass its demand?
  - **In-plane shear** via `design/concrete/shear.py` + `wall_shear.py` cracked
    factors.
  - **Detailing** — boundary-element confinement (`design/seismic/confinement.py`)
    and IS 13920 / EC8 special-wall provisions.
- **Engine gap to fill (small):** an ACI 318-19 **§18.10 special structural
  wall** module (boundary-element trigger `c ≥ ℓ_w/(600·δ_u/h_w)`, distributed
  web reinforcement ρ_ℓ/ρ_t, min boundary length). Everything it needs
  (fiber P-M, confinement, shear) already exists — this assembles them.
- Results ▸ Design ▸ Wall: pass/fail + required boundary reinf + web ρ, per pier.

### W4 — Story / Level system ★ (building-modeling backbone)
- A `Story` object (elevation, height, master/similar-to) + a story-aware model
  tree and an **elevation view** (2-D per-plane view).
- Grid system (named X/Y grid lines) to snap drawing.
- "Similar stories" replicate walls/columns up the building.
- Larger effort; touches drawing, the viewport, the tree, and load application.
  Can land after W0–W3 since walls already work story-agnostically.

### W5 — Openings ★ *(v2)*
- Door/window openings in a wall panel with **opening-aware meshing** (extends
  the polygon-area mesher; needs ear-clipping for the non-convex remainder — the
  same gap noted in the slab backlog).

### W6 — Coupled walls / coupling beams in the GUI *(v2)*
- Surface the engine's `coupling_beam.add_coupling_beam_2d` as a GUI action
  (pick two piers + a floor line → coupling beam with rigid face offsets).
- Macro fiber-wall option (`wall_section_2d`) as an alternative to the meshed
  shell wall, for nonlinear/pushover users.

---

## 4. Recommended sequencing

1. **W0** — wall/pier label (½ day; pure data model).
2. **W1** — elevation draw-plane + Draw ▸ Wall (the workflow unlock).
3. **W2** — pier force integration (the headline ETABS output).
4. **W3** — wall design check (turns forces into a code result).
5. *Ship & reassess.* W0–W3 = a credible wall design tool.
6. **W4** — stories/grids (big, optional, do when building-scale models matter).
7. **W5 / W6** — openings, coupled walls (v2).

### Immediate next item
**W0.** Add the `role` + `pier`/`spandrel` fields to `Area`, the JSON migration,
and a couple of round-trip tests. Nothing renders differently yet — it just
gives every later epic something to hang on.

---

## 5. Key decisions / open questions

1. **Wall = tagged `Area`, or a new `Wall` object?** Recommend **tagged `Area`**
   (`role="wall"` + `pier`/`spandrel` labels) to reuse the entire area
   draw/mesh/load/results pipeline — mirrors how ETABS treats a wall as an area
   object with a pier assignment. A separate `Wall` class would duplicate that
   machinery. (Piers spanning several areas are then a *label group*, not a new
   type.)
2. **Meshed shell wall vs. macro fiber wall.** Default to the **meshed shell**
   (matches the slab pipeline and ETABS' shell/membrane wall). Offer the macro
   fiber wall (W6) only for nonlinear users who want a single-element wall.
3. **Pier cut geometry.** v1: horizontal cut at each story level, integrating
   the membrane resultants of the elements the pier owns. Curved/stepped walls
   deferred.
4. **Design code scope for W3.** Start with one code end-to-end (ACI 318-19
   §18.10) then add IS 13920 / EC8 detailing, reusing the existing seismic
   modules.
5. **Does W4 (stories) block anything?** No — walls work story-agnostically
   after W1–W3. Stories are an ergonomics/scale upgrade, not a correctness gate.

---

## 6. Tracker

| Epic | Title | Status |
|---|---|---|
| W0 | Wall / pier data model | ☑ done (`da4f925`) |
| W1a | Draw Wall by base line + height | ☑ done (`5e9a721`) |
| W1b | Elevation / XZ-YZ draw plane | ☐ todo |
| W2 | Pier force integration | ☑ done (`9d8ada7`) |
| W3 | Wall design check | ☐ proposed |
| W4 | Story / Level system + grids | ☐ proposed |
| W5 | Openings (opening-aware mesh) | ☐ proposed (v2) |
| W6 | Coupled walls / coupling beams GUI | ☐ proposed (v2) |

*Engine additions required across the whole stream: only (a) the W2 pier-cut
integrator and (b) the W3 ACI 318 §18.10 assembler. Everything else is desktop
GUI + data model over capability that already ships and is tested.*
