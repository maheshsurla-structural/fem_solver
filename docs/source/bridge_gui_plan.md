# Bridge-analysis GUI plan — wiring the analysis engine into the desktop

**Created:** 2026-09-14
**Scope:** surface the bridge (and remaining) analysis capabilities in the
**desktop GUI** (PySide6). The engine work is done (see
`bridge_analysis_plan.md`); this plan is about the *front end*.

## Approach

Follow the proven Modal / Response-Spectrum / Buckling pattern already in the
Analysis-cases home:

    config dialog  →  MainWindow.run_*  →  Analysis-cases row + dispatch  →  results view

Most bridge analyses run on the **existing frame model** (a girder line, or a
grillage of beams), so they need *definition + dialog + results*, not a new
area/shell modeler. Persisted bridge definitions (lanes, stages, cables) are
added to `project.py` only when an analysis needs to remember them between
runs; the first increments keep config transient (like Modal/RS/Buckling).

## Tracker

- [x] **G1 — Moving Load / influence lines** ✅ DONE 2026-09-14. `moving_load_dialog.py`
  (lane = multi-select girder nodes, ordered by X; vehicle = HL-93 full/truck/
  tandem, IRC A/70R; response = moment/shear at a member end, or nodal
  displacement/reaction), `moving_load_results_dialog.py` (influence-line plot +
  vehicle envelope, matplotlib, non-modal), `MainWindow.run_moving_load`
  (`InfluenceLineEngine` + `aashto_hl93_envelope` / `moving_load_envelope`).
  Analysis-cases "Moving Load" row now **live** (`_PLANNED` empty — every row
  runnable). 2-D girder line. `tests/test_desktop_moving_load.py` (6).
- [x] **G2 — Temperature-gradient load** ✅ DONE 2026-09-14. Wired as an
  **analysis-case launcher** (consistent with Modal/RS/etc.), not a persisted
  load type. `temperature_gradient_dialog.py` (AASHTO zone 1–4 or linear
  top→bottom, α, member multi-select), `temperature_gradient_results_dialog.py`
  (section self-stress + temperature diagram, non-modal),
  `MainWindow.run_temperature_gradient` — derives rectangular-equivalent
  depth/width per member from A/Iz, `equivalent_thermal_actions` +
  `apply_beam_thermal_actions` per member, solves, renders the deflected shape,
  reports self-stress + max deflection + continuity moment. Analysis-cases
  "Temperature Gradient" row live. 2-D only. `tests/test_desktop_temperature_gradient.py`
  (6).
- [x] **G3 — Construction stages + camber** ✅ DONE 2026-09-14. Persisted
  `project.Stage` (id, name, add_members) + `Project.stages` + serialization.
  `stage_dialog.py` `StageManagerDialog` (ordered stages, add/delete/↑↓, per-
  stage member multi-select with exclusivity, **auto-sequence L→R**),
  `construction_stage_results_dialog.py` (camber diagram: per-stage deflection
  history + final deflection + build-high camber),
  `MainWindow.run_construction_stages` (self-weight per stage →
  `IncrementalStagedAnalysis` → `staged_camber`; unassigned members prepended
  to stage 1; renders final deflected shape; guards zero-density + unstable
  build order). Analysis-cases "Construction Stages" row live. 2-D only.
  `tests/test_desktop_construction_stages.py` (7).
- [x] **G4 — Influence surfaces / multi-lane** ✅ DONE 2026-09-14.
  `influence_surface_dialog.py` (deck-node multi-select, response = vertical
  displacement/reaction at a node, vehicle, multiple-presence toggle),
  `influence_surface_results_dialog.py` (plan contour of the influence surface +
  governing multi-lane envelope), `MainWindow.run_influence_surface`
  (`DeckSurface` load_dof=uz, `InfluenceLineEngine.influence_surface`,
  `Vehicle2D.from_axle_train`, `generate_design_lanes` from the deck transverse
  extent, `multi_lane_envelope` with AASHTO multiple presence). **Requires a
  3-D deck/grillage** (gated). Analysis-cases "Influence Surface" row live.
  `tests/test_desktop_influence_surface.py` (4).
- [x] **G5 — Vehicle–bridge dynamics** ✅ DONE 2026-09-14.
  `vehicle_dynamics_dialog.py` (lane reused from G1; analysis kind = moving
  force / sprung-mass VBI via QStackedWidget; vehicle preset or sprung-mass
  mass/bounce-freq/suspension-damping; speed km/h; bridge ζ; response node),
  `vehicle_dynamics_results_dialog.py` (deflection dynamic-vs-static time-history
  + DAF, plus contact-force panel for VBI), `MainWindow.run_vehicle_dynamics`
  (Rayleigh damping from modes 1&3; `MovingForceAnalysis` / `VBIAnalysis`).
  Analysis-cases "Vehicle Dynamics" row live. 2-D, mass from density.
  `tests/test_desktop_vehicle_dynamics.py` (6).
- [x] **G6 — Cable-stayed tuning (ULF)** ✅ DONE 2026-09-14.
  `cable_tuning_dialog.py` (stay-member multi-select + target-node multi-select),
  `cable_tuning_results_dialog.py` (tuned-tension table + before/after deck
  profile), `MainWindow.run_cable_tuning` (`Cable` from designated members,
  `Displacement` targets = 0, `unknown_load_factors`, `apply_cable_tensions`
  for the after-profile; renders tuned deflected shape). 2-D; dead load required.
  Stays are designated among existing members (a pin-ended cable/truss member
  type is a future refinement). Analysis-cases "Cable Tuning" row live.
  `tests/test_desktop_cable_tuning.py` (4).

**GUI plan COMPLETE (G1–G6, 2026-09-14).** Analysis-cases home has 11 live
rows — every analysis is reachable from the desktop.

## Track A — polish / coverage (harden what G1–G6 built)

- [x] **A1 — Pin-ended cable / truss member type** ✅ DONE 2026-09-14.
  `project.Member.kind == "cable"` compiles to `Truss2D` / `Truss3D` (axial
  only) in `build_model` + `build_buckling_model` (single element, no
  sub-division); `MemberDialog` gains a Beam/Cable "Type" selector; kind
  serializes (old projects → beam). Works inside the ndf=3 beam model (truss
  maps to ux/uy, leaves rz to the beams). `tests/test_desktop_cable_member.py`
  (7). Now the cable-tuning stays are real pin-ended cables.
- [ ] **A2 — Units-layer retrofit** on the new results dialogs (Temp Gradient,
  Stages, Vehicle Dynamics, Influence Surface, Cable Tuning) — Moving Load
  already uses the project display units; the rest use fixed SI-derived labels.
- [ ] **A3 — 3-D coverage** for the cases gated to 2-D (Buckling, Moving Load,
  Temp Gradient, Stages, Vehicle Dynamics) — needs `BeamColumn3D.
  K_geometric_global`, 3-D moving-load beam-force ILs, and 3-D vertical-DOF
  handling in the runners.

## Notes
- Every GUI change is flagged to the user (their standing request).
- Pre-existing: `test_desktop_member_load.py::test_main_window_wires_line_loads`
  fails on main from the merged model-navigation tree refactor (spawned a task);
  unrelated to this plan.
