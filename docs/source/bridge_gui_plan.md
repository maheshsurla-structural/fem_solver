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
- [ ] **G4 — Multi-lane placement** — design-lane definition + multi-presence
  on top of G1 (influence surfaces need a grillage/deck; the deck-surface
  picker is the extra piece).
- [ ] **G5 — Vehicle–bridge dynamics** — moving-force / sprung-mass VBI
  (`MovingForceAnalysis` / `VBIAnalysis`): vehicle + speed + lane → DAF +
  time-history / contact-force plots.
- [ ] **G6 — Cable-stayed tuning (ULF)** — cable + target-condition editor →
  `unknown_load_factors` → tuned tensions + before/after deck profile. Needs
  truss/cable members + target editor in the GUI.

## Notes
- Every GUI change is flagged to the user (their standing request).
- Pre-existing: `test_desktop_member_load.py::test_main_window_wires_line_loads`
  fails on main from the merged model-navigation tree refactor (spawned a task);
  unrelated to this plan.
