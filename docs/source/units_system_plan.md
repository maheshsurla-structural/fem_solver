# Dynamic unit system plan — MIDAS-style force × length

**Created:** 2026-09-14
**Scope:** give the desktop GUI a **live, switchable unit system** — the
feature every commercial FEM tool (MIDAS, SAP2000, CSiBridge, STAAD) has and
this one does not. Today `Project.force_unit` / `length_unit` are *label
strings only*: they are printed next to fields but nothing converts, and no UI
ever changes them. This plan makes units real.

## Current state (audit 2026-09-14)

- `Project.force_unit="N"`, `length_unit="m"` (`project.py`) — used purely as
  label text at ~30 call sites (`editing.py`, `member_load_dialog.py`,
  `pushover_dialog.py`, `nl_report.py`, `main_window` status bar, …).
- **No conversion anywhere.** A value typed into a `[kN]`-labelled field is
  stored as-is; the engine always works in SI base (N, m, Pa).
- The status-bar `force · length` chip is a static, non-interactive `QLabel`.
- No Preferences / Units dialog exists. `desktop/units.py` never existed as
  source (a stale `.pyc` is the only trace).
- Minor bug: dataclass default `"N"` vs `from_dict` default `"kN"`.

## Decisions (locked with the user)

- **Unit model: MIDAS-style pair.** Two choices — Force × Length — and every
  derived unit (stress, moment, distributed load, area, second moment, mass,
  density) follows automatically. Not per-quantity, not fixed presets.
- **First cut: SI family only** (force {N, kN, kgf, tonf}; length {m, cm, mm}).
  Imperial (kip, lbf, in, ft, ksi, psi) is a later step (U5) once live
  switching works end-to-end.

## Architecture

The engine and stored model stay in **SI base (N, m, Pa, rad)** — the single
source of truth. Units are a **presentation + input layer** on top:

    stored value (SI)  --to_display-->  spin box / label   (SI → chosen unit)
    spin box value     --to_si------->  stored value       (chosen unit → SI)

`UnitSystem` holds the current Force and Length and derives every quantity's
conversion factor. All widgets format via one `fmt()` and parse via one
`to_si()`, so a unit change is: mutate the `UnitSystem`, re-render.

## Tracker

- [ ] **U1 — Units core** `desktop/units.py`: `Quantity` enum (length, force,
  stress, moment, dist_load, area, inertia, mass, density, disp, rotation),
  a `UnitSystem(force, length)` with `to_si` / `to_display` / `factor` /
  `label(qty)`, SI-family catalogs, pure + fully unit-tested. No GUI.
- [x] **U2 — Display pass** ✅ 2026-09-14. `MainWindow._units()` →
  `UnitSystem.from_project`; status-bar chip (`pair_label`), cursor-coord
  readout (converts LENGTH), and diagram-max log/status line (converts
  FORCE/MOMENT) now format through it. `tests/test_desktop_statusbar.py`
  gains a non-SI (kN·mm) conversion test. Moving-load *results dialog* + its
  numeric plots deferred to U6 (whole-widget, kept mutually consistent).
  Display converts; inputs still SI.
- [x] **U3 — Input pass** ✅ 2026-09-14. New `desktop/unit_widgets.py`
  `UnitSpin` (shows display units, stores SI via `set_si`/`si_value`) +
  `labeled()`; `UnitSystem.dof_quantity()` splits a load vector's rotational
  DOFs (moments, F·L) from translational (forces). Converted the model-building
  dialogs in `editing.py` (Node, Load, LoadGen, Move, Copy, Mirror, Rotate),
  `member_load_dialog.py` (line loads, F/L), and the live **`properties.py`**
  inspector (node + load forms — the right-hand panel; it imports
  `editing._coord_spin`/`_force_spin`, so the signature change *required*
  updating it — regression caught by `test_desktop_nav.py`).
  `tests/test_desktop_units_input.py` (9) proves conversion under a non-SI
  (kN·mm) project, incl. the Properties panel end-to-end — existing dialog
  tests stay green because default N/m is the identity. **Deferred to U6:**
  `SectionDialog` A/Iz/Iy/J (area/inertia, tied to AISC catalog auto-fill),
  the generator dialogs (`FrameDialog`/grid — take no `project`, need a
  signature change), and `pushover_dialog`/`hinge_editor` (carry their own
  persist + meta-display).
- [x] **U4 — Preferences + live switch** ✅ 2026-09-14. `desktop/units_dialog.py`
  `UnitsDialog` (Force × Length combos + live "everything else follows" preview
  of derived labels). Status-bar chip is now a `_ClickableLabel` →
  `MainWindow.change_units`: opens the dialog, applies via `_apply_edit` (so
  it's **undoable + marks the model dirty**), persists to `QSettings`
  (`units/force`, `units/length`) as the **app default**, and re-renders the
  status readouts + live Properties inspector. `load_project` adopts the app
  default for new/generated/demo models (`path is None`) while opened files
  keep their saved units. Fixed the default-mismatch bug: `from_dict` now
  defaults `"N"` (was `"kN"`) to match the dataclass + SI-base baseline.
  `tests/test_desktop_units_dialog.py` (7). Full desktop suite green.
  (`UnitSystem()`'s standalone fallback stays kN·m; the *app* baseline is N·m.)
- [x] **U5 — Imperial + full catalog** ✅ 2026-09-14. Added `kip`
  (4448.2216152605 N) + `lbf` (exact international pound-force) to `FORCE_UNITS`
  and `in` (0.0254 m) + `ft` (0.3048 m) to `LENGTH_UNITS`. Stress/moment stay
  **compositional** (kip·ft, kip/in² — the latter *is* ksi, `lbf/in²` = psi),
  keeping the derive-from-the-pair design rather than special-casing friendly
  names. Combos + preview pick them up for free. Tests in
  `test_desktop_units.py` (cross-family factors) + `test_desktop_units_input.py`
  (imperial dialog round-trips: 10 ft→3.048 m, 2 kip→8896 N, kip·in moment).
  Full desktop suite green. Friendly stress aliases (ksi/psi/MPa/kPa) noted as
  possible future polish.
- [x] **U6 — Polish sweep (live GUI + pushover export)** ✅ 2026-09-14.
  Converted: `SectionDialog` + `properties._section_form` A/Iz/Iy/J
  (AREA/INERTIA; AISC catalog auto-fill uses `set_si` so SI catalog values seed
  correctly); `pushover_dialog` target/axial inputs + `_report_meta` scalars +
  all three matplotlib plots (curve, fiber geometry, deformed shape — data +
  axes; fiber *stress* deliberately stays MPa); `nl_report.py` `curve_csv` /
  `fibers_csv` geometry / `report_html` metrics (peak shear, disps, stiffness
  F/L, energy F·L, ASCE milestones) — so a report reads consistently with the
  converted plots. Tests: `test_desktop_units_input.py` (+section),
  `test_desktop_nlreport_units.py` (4). Identity under default m/N keeps all
  prior tests green.
- [x] **U6b — remainder** ✅ 2026-09-14 (branch `feat/units-u6b`).
  **Moving-load results** (`moving_load_results_dialog.py`): now takes
  `unitsys` + response `quantity` (M→MOMENT, V/reaction→FORCE, disp→LENGTH);
  converts the envelope max/min, station axis (LENGTH), and influence ordinate;
  `main_window.run_moving_load` passes them + converts its log/status lines.
  **`hinge_editor`**: mode-aware — absolute lengths convert SI↔display
  (`_seed` on load, ×`_lfac` in `data()`), relative fractions stay raw; the
  manager table (`_length_text`) converts absolute rows only.
  **`FrameDialog`**: optional `units` arg → bay/storey inputs read/store in the
  project's length unit (defaults to metres = identity); `_len_spin` removed;
  `main_window.generate_frame` passes `self._units()`.
  Tests: `test_desktop_units_u6b.py` (5). **#4 friendly stress aliases
  (ksi/psi/MPa) SKIPPED** — `Quantity.STRESS`'s label surfaces only in the
  units-dialog preview line, so an alias would be a cosmetic relabel that dents
  the compositional model; left as a documented non-goal.

## Notes

- Every GUI change is flagged to the user (their standing request).
- SI base is never rescaled — only display/parse. Save files stay SI, so old
  projects load unchanged and `force_unit`/`length_unit` become *preferences*.
