# Wall (shear-wall / pier) modeling — commercial-parity roadmap

*Status: **ROADMAP COMPLETE — W0–W6 + W1b + W3-polish** (W0–W6 on `origin/main`;
W3-polish on branch `feat/wall-detailing`). The core "credible wall tool"
(W0–W3) is done: label → draw → pier forces → design; W4 (a/b/c) adds the
Story/Grid model + manager, viewport overlay + snapping, and similar-story
replication; W1b adds the XZ/YZ elevation draw plane; W5 adds wall openings; W6
adds coupling beams; **W3-polish** adds the ACI §18.10.6.2 drift-based boundary
trigger and IS 13920 / EC8 detailing (multi-code selector). Only minor
refinements remain (non-rectangular openings, a macro fiber-wall option, a
3-point work plane). This roadmap takes the desktop app from "a wall is just a
vertical `Area`"
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
`tests/test_desktop_pier_forces.py`. W3 (`29a3594` engine + `f69ba85` desktop):
ACI 318-19 §18.10 wall design — `femsolver/design/walls.py` + Results ▸ Wall ▸
Wall design (`wall_design_dialog.py`). **Next: W4** (story/level system + grids
— the big building-modeling investment) or **W1b** (elevation draw plane), or
polish W3 (IS 13920/EC8 detailing, displacement-based boundary trigger). Note:
the GUI venv lives in the **main** repo, so run pytest with
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
- **W1b — DONE** (`6d0dcd6`): a **work-plane** selector (XY ground, **XZ / YZ
  elevation**) at a fixed-axis offset — `ModelView.set_work_plane` rebuilds the
  pickable plane there and the draw-node pick keeps the click's full 3-D
  position (pinned to the offset) instead of forcing z = 0, snapping the two
  in-plane axes to the spacing/named grid + story levels. A Plane XY/XZ/YZ +
  offset selector in the Draw ribbon. `tests/test_desktop_work_plane.py`. *(A
  picked 3-point arbitrary plane is a possible future refinement.)*

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

### W3 — Wall design ★★ — **DONE** (`29a3594` engine + `f69ba85` desktop)
- **Engine (the stream's one real add):** `femsolver/design/walls.py` — a
  self-contained **ACI 318-19 §18.10** module: `wall_shear_strength` (§18.10.4
  Vn/αc/cap/two-curtains), `boundary_element_check` (§18.10.6.3 stress trigger),
  `wall_min_web_reinforcement` (§18.10.2), `wall_pm_capacity` (§22.2/§21.2
  strip-integrated rectangular-wall P-M with distributed web + boundary bars,
  φ from net tensile strain), assembled by `design_wall_pier` → governing DCR +
  verdict + notes. Empirical √f'c uses the MPa calibration for SI consistency.
  `tests/test_wall_design.py` (13, vs hand calcs).
- **Desktop:** Results ▸ Wall ▸ Wall design (`wall_design_dialog.py`) runs it per
  pier against the W2 demand (P negated to compression-positive), per cut, with
  f'c/fy/ρl/ρt/boundary inputs and a governing-DCR PASS/FAIL summary +
  boundary/detailing flags. `piers.pier_geometry` supplies (ℓw, t, hw).
  `MainWindow.show_wall_design` + `act_wall_design`. `tests/
  test_desktop_wall_design.py` (5).
- **W3-polish — DONE** (`aff2adc`): the displacement-based boundary trigger
  (`c ≥ ℓw/(600·δu/hw)`, §18.10.6.2) with the real ℓbe = max(c−0.1ℓw, c/2), plus
  a **multi-code detailing** layer (`WALL_CODES` = ACI 318-19 / IS 13920 / EC8;
  `wall_detailing`) — boundary/web minimum-reinforcement ratios (IS 13920 §10.4.4
  0.8%, EC8 §5.4.3.4.2 0.5%) and code-referenced notes. `design_wall_pier(code=,
  drift=)`; the dialog gains a code selector + design-drift input.

### W4 — Story / Level system ★ (building-modeling backbone)
- **W4a — DONE** (`939548f`): `project.Story` (named level at an elevation +
  height + 'similar-to' master) and `project.GridLine` (named X/Y reference
  line); `Project` accessors (`stories_sorted`, `story_elevations`, `story_at`,
  `grid_lines_on`); JSON round-trip + migration; `story_grid_dialog.
  StoryGridDialog` (Home ▸ Levels ▸ Stories & grid) with editable tables + a
  Generate story-stack helper; tie-in: the pier-forces table now carries a
  **Story** column (`story_at` each cut). `tests/test_desktop_stories.py`.
- **W4b — DONE** (`aae1ca7`): grid/story-plane **rendering** in the viewport
  (`model_geometry.grid_story_mesh` → `ModelView.set_story_grid`, a `storygrid`
  actor) + **snap-to-grid** while drawing (`snap_targets`/`snap_to_grid`; the
  draw-node path snaps to the nearest named grid line) + a View ▸ Display ▸ Grid
  toggle. `tests/test_desktop_story_grid_view.py`. *Still TODO:* a dedicated 2-D
  **elevation view** (the orientation Front/Side views already give elevations;
  a true single-plane editing view pairs with W1b).
- **W4c — DONE** (`8818731`): "similar stories" replicate a source story's
  walls/columns/beams up to target stories (`story_replicate.replicate_story` —
  elevation-band selection, translated copy, coincident-node merge, pier labels
  run up the building) via `StoryReplicateDialog` + Home ▸ Levels ▸ Replicate.
  `tests/test_desktop_story_replicate.py`.
- Walls already work story-agnostically, so these are ergonomics/scale upgrades,
  not correctness gates.

### W5 — Openings ★ — **DONE** (`4e59f54`)
- Rectangular door/window openings in a wall panel (`Area.openings`, parametric
  `(u0,v0,u1,v1)`). **Opening-aware meshing without ear-clipping**: the quad
  mesher drops the grid cells whose centre falls in an opening (the ETABS
  approach) and builds only the nodes a kept cell uses, so openings leave no
  orphaned/singular nodes; `area_quad_cells`/`area_quad_needed_nodes` are shared
  by the mesher, `_area_element_tags` and `_area_edge_coords` so they never
  disagree. `wall_opening_dialog.WallOpeningDialog` (physical X/Z/W/H → fraction)
  + Draw ▸ Openings on the selected wall. `tests/test_desktop_wall_openings.py`.
  *(Non-rectangular openings / arbitrary polygon holes would still want a
  constrained triangulator — deferred.)*

### W6 — Coupled walls / coupling beams in the GUI — **DONE** (`2bb1ea1`)
- `coupling.add_coupling_beam` + `CouplingBeamDialog` + Draw ▸ Coupling: a beam
  `Member` between two shell-wall piers' facing inner edges at an elevation
  (snapped to each wall's nearest mesh row), tied into the shell via the build's
  coincident-node merge. **Note:** the engine's `coupling_beam.add_coupling_beam_2d`
  is a *2-D macro* helper (centroid nodes, rigid face offsets) that doesn't fit
  the desktop's 3-D meshed shell walls, so W6 is a desktop-native coupling beam
  for shell walls rather than a wrapper over it. `tests/test_desktop_coupling.py`.
- *Deferred:* a macro fiber-wall option (`wall_section_2d`) for nonlinear/
  pushover users — a separate modeling idiom from the shell-wall GUI.

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
| W1b | Elevation / XZ-YZ draw plane | ☑ done (`6d0dcd6`) |
| W2 | Pier force integration | ☑ done (`9d8ada7`) |
| W3 | Wall design check (ACI 318 §18.10) | ☑ done (`29a3594`+`f69ba85`) |
| W4a | Story / Grid data model + manager | ☑ done (`939548f`) |
| W4b | Grid render + snap-to-grid | ☑ done (`aae1ca7`) |
| W4c | Similar-story replication | ☑ done (`8818731`) |
| W5 | Openings (opening-aware mesh) | ☑ done (`4e59f54`) |
| W6 | Coupled walls / coupling beams GUI | ☑ done (`2bb1ea1`) |

*Engine additions required across the whole stream: only (a) the W2 pier-cut
integrator and (b) the W3 ACI 318 §18.10 assembler. Everything else is desktop
GUI + data model over capability that already ships and is tested.*
