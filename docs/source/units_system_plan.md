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
- [ ] **U2 — Display pass** route the status bar, output-log diagram lines and
  results labels through `UnitSystem.fmt(value, qty)` (display converts;
  inputs still SI). Read-only, low risk.
- [ ] **U3 — Input pass** dialogs parse display→SI on commit and SI→display on
  load (`editing.py`, `member_load_dialog.py`, `pushover_dialog.py`, …). A
  shared `unit_spin(qty, unitsys)` helper replaces ad-hoc `_spin(unit=…)`.
- [ ] **U4 — Preferences + live switch** a Units dialog (Force + Length combos,
  live preview) and a **clickable** status-bar unit chip that opens it;
  persist per-project (`project.py`) and as app default (`QSettings`); on
  change, re-render every open view. This is the moment it becomes "dynamic".
- [ ] **U5 — Imperial + full catalog** kip/lbf/kgf/tonf, in/ft, ksi/psi/MPa,
  with correct cross-family factors.
- [ ] **U6 — Polish sweep** results dialogs, CSV/HTML exports (`nl_report.py`),
  matplotlib axis labels, section designer, coord readout, nav cube.

## Notes

- Every GUI change is flagged to the user (their standing request).
- SI base is never rescaled — only display/parse. Save files stay SI, so old
  projects load unchanged and `force_unit`/`length_unit` become *preferences*.
