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
- [ ] **G2 — Temperature-gradient load** — a load type in the Loads UI (AASHTO
  zone / section profile) → `equivalent_thermal_actions` → self-stress +
  deflection results. Needs a project load-type + a results view.
- [ ] **G3 — Construction stages + camber** — a stage manager (assign members
  to stages, per-stage loads) on `IncrementalStagedAnalysis` → camber diagram
  (`staged_camber`). Needs a persisted stage list in `project.py`.
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
