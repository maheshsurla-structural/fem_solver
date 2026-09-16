# Analysis-cases manager — saved, named, multi-instance analysis cases

**Created:** 2026-09-15
**Branch:** `feat/analysis-cases-manager`
**Scope:** turn the **Analysis-cases home** (`desktop/analysis_cases_dialog.py`)
from a fixed list of one-shot *launchers* into a true CSiBridge / MIDAS
**Load-Cases** manager: every analysis type can have multiple **named, saved,
editable** cases, listed with full **Add / Modify / Delete / Run**, run from
their stored parameters with **no re-prompt**.

## Problem

Today the list mixes two kinds of row:

* **Nonlinear Static** — a real persisted entity (`project.NonlinearCase`), with
  a proper editor. Add / Modify / Delete work.
* **Everything else** (Modal, Response Spectrum, Buckling, Moving Load, Temp
  Gradient, Construction Stages, Vehicle Dynamics, Influence Surface, Cable
  Tuning, Load Rating, Time History) — a single fixed **launcher** row. Each
  `MainWindow.run_*` pops its setup dialog when `config is None`, runs, and
  **throws the config away**. Nothing is stored, so there is nothing to Modify or
  Delete, and you cannot keep two differently-tuned cases of one type (an "RS-X"
  and an "RS-Y", a "Moving HL-93" and a "Moving Permit").

`AnalysisCasesDialog._sync_buttons` gates Modify / Delete on `kind ==
"nonlinear"` — the visible symptom.

## Approach

Make every type a persisted **`project.AnalysisCase`** the way nonlinear cases
already are, driven by a small per-type **adapter registry**. Two facts make
this cheap:

1. **The runners already accept a `config`.** `run_modal(num_modes, lumped)`,
   `run_buckling(config=…)`, `run_moving_load(config=…)`, … all run headlessly
   from a passed config. Feeding them a *saved* config needs **no runner
   change** — only a place to store the config and a dispatch that passes it.
2. **Almost every setup dialog already returns a JSON-friendly config** — a
   dict/tuple of primitives (`(num_modes, lumped)`; `(selection, num_modes,
   subdivisions)`; `dict(lane, vehicle, response)`; …). The one exception is
   **Response Spectrum**, whose `result()` returns a *derived* `ResponseSpectrum`
   object — so its case must persist the **inputs** (source + parameters) and
   rebuild the object headlessly.

### Data model (`project.py`)

```python
@dataclass
class AnalysisCase:
    id: int
    name: str
    type: str                 # "modal" | "buckling" | "responsespectrum" | …
    params: dict = {}         # JSON-friendly saved config (inputs, not objects)
    notes: str = ""
```

Stored in a **new** `Project.analysis_cases` list, serialized alongside
`nonlinear_cases`. Nonlinear stays in its own list with its rich editor — the
two coexist; the home merges both into one visual list.

**Principle: persist inputs, not derived objects.** `params` holds exactly what
the user typed (spectrum source + code parameters, lane node ids, vehicle key,
response target). The runtime config (including any live object such as a
`ResponseSpectrum`) is rebuilt from `params` at Run time.

### Adapter registry (`desktop/case_types.py`)

One `CaseType` per analysis type, keyed by `type_id`:

| member | job |
|---|---|
| `type_label`, `icon` | the list row's Type text + icon |
| `detail(project, params)` | the list row's Details text |
| `default_params(project)` | seed a fresh Add |
| `edit(parent, project, case=None) → AnalysisCase\|None` | open the type's setup dialog **seeded** from `case.params`; return the (re)named case |
| `build_config(project, params)` | rebuild, headlessly, the runtime config the runner expects |
| `dispatch(win, config)` | call the matching `MainWindow.run_*` |

The home uses the registry generically; `main_window` dispatches a saved case by
`build_config` → `dispatch`. Each setup dialog gains (a) an optional `initial=`
to seed its widgets and (b) a shared **`CaseHeader`** (Name / Notes) so every
saved case has a name, mirroring the nonlinear dialog.

### Migration — strangler, one type at a time

The home renders: the **Linear Static** singleton launcher (unchanged), the
**nonlinear** cases (existing editor), the **saved `AnalysisCase`** rows (generic
CRUD), and — for types **not yet migrated** — their existing launcher row
(unchanged). Migrating a type = give its dialog a header + `initial`, add its
adapter, drop its launcher row, and add it to the **Add ▾** menu. The app stays
shippable at every commit.

## Per-type tracker

Param schema is the JSON-friendly `params` dict; "config" is what the runner
already consumes.

- [x] **F — Foundation** ✅ 2026-09-15. `AnalysisCase` + `Project.analysis_cases`
  + serialization + `analysis_case(id)`; `case_types.py` registry + `CaseType`
  base; `AnalysisCasesDialog` generic CRUD (Add ▾ menu, Modify/Delete/Run for
  saved cases, editable rows merged); `main_window` commits `analysis_cases` and
  dispatches `("case", id)`.
- [x] **M1 — Modal** ✅ 2026-09-15. params `{num_modes, lumped}`; config
  `(num_modes, lumped)`; dispatch `run_modal`. `ModalDialog` header + `initial`.
- [x] **M2 — Buckling** ✅ 2026-09-15. params `{selection:[kind,id], num_modes,
  subdivisions}`; config `(tuple(selection), num_modes, subdivisions)`; dispatch
  `run_buckling`. `BucklingDialog` header + `initial`.
- [x] **M3 — Moving Load** ✅ 2026-09-15. params `{lane:[ids], vehicle,
  response:[kind,id,end]}` = config (identity). `MovingLoadDialog` header +
  `initial` + `_seed`.
- [x] **M4 — Temperature Gradient** ✅ 2026-09-15. params `{source, zone, dt_top,
  dt_bot, alpha (SI), members:[ids]}` = config. Dialog header + `initial` +
  `_seed` (α shown ×10⁻⁵, stored SI).
- [x] **M5 — Load Rating** ✅ 2026-09-15. params `{lane:[ids],
  response:[comp,id,end], Rn/DC/DW/P (SI), phi, phi_c, phi_s, im, adtt,
  permit_gamma_LL}` = config. Dialog header + `initial` + `_seed` (capacity/dead
  loads SI→display, quantity from the effect so set effect first).
- [x] **M6 — Response Spectrum** (the object case) ✅ 2026-09-15. params =
  spectrum **inputs** `{source, damping, num_modes, direction, combination,
  asce7:{SDS,SD1,TL}, ec8:{ag,ground,q,type}, is1893:{zone,I,R,soil},
  custom:[[T,Sa]]}` (all source pages stored). New module-level
  `spectrum_from_params()` rebuilds the `ResponseSpectrum` headlessly; the dialog
  now delegates `build_spectrum` to it and gains `params()` + `_seed` + header.
  `build_config` = `(spectrum_from_params(params), num_modes, direction,
  combination)`. Verified across all 4 sources + custom-underdefined guard.
- [x] **M7 — Vehicle Dynamics / Influence Surface / Cable Tuning** ✅ 2026-09-15.
  dict configs of primitives; header + `initial` + `_seed` each (vehicle dynamics
  reverses its t/%/km-h→SI conversions on seed). Round-trip verified headless.
- **M8 — Time History** — **IN PROGRESS via a Time-History Function library**
  (user chose this over path/embedded/leave-as-launcher, 2026-09-16). CSiBridge
  model: named ground-motion records persisted once, cases reference one by id.
  - [x] **TH-1 — Function library** ✅ 2026-09-16. `project.TimeHistoryFunction`
    (id, name, dt, values, in_g, source; `npts`/`duration` props) +
    `Project.th_functions` + `th_function(id)` + serialization. `th_functions.py`:
    `TimeHistoryFunctionDialog` (name/dt/units + import via `load_accel_record`
    + live preview) and `TimeHistoryFunctionManagerDialog` (list/add/edit/delete,
    in-use guard vs referencing TH cases). New `function` icon; `act_th_functions`
    homed in ribbon Analysis ▸ Functions; `manage_th_functions`. 7 tests + ribbon
    invariant green.
  - [ ] **TH-2 — TH as a saved case.** Refactor `TimeHistoryDialog`: replace
    "Load record…" with a **function picker** (combo of `th_functions`; in_g moves
    to the function), add `CaseHeader` + `initial` + a config-collect path.
    `case_types.TimeHistoryType`: `edit` collects params `{function_id,
    control_node, direction, scale, zeta, density}`; `build_config` loads the
    referenced function; `dispatch` opens the (interactive) runner seeded, ready
    to Run. Drop the launcher row; add to Add ▾. Needs a `run_timehistory(config=)`
    seed path on the runner.
- [ ] **M9 — Construction Stages** — **STAYS A LAUNCHER by design.** Its config
  *is* the project-global `stages` list (already persisted, edited via
  `StageManagerDialog`); there is one stage sequence per model, so a
  multi-instance saved case would only duplicate/diverge from `project.stages`.
  Left as a launcher row (already operates on persistent data).
- [x] **P — Polish** ✅ 2026-09-15 / 16. **notes** surfaced as a row tooltip;
  **duplicate-name auto-suffix** (`_unique_name` → "Name (2)", "(3)" on Add /
  Modify); the **run log names the case** (`'<name>' (<Type>) — running…` in
  `_run_saved_case`); **Duplicate** button (`_duplicate` — clone a saved or
  nonlinear case → "… (copy)", the fast RS-X→RS-Y workflow); **reorder**
  (`_move` ↑/↓ within the case's own list, id-safe for staged `continue_from`).
  Remaining nice-to-haves (not blocking): a fuller results-history label
  threading the case name into each runner's `RunRecord` (only a log line so
  far); user-guide docs.

**End state:** 9 types are saved, multi-instance AnalysisCases (Modal, Buckling,
Moving Load, Temp Gradient, Load Rating, Response Spectrum, Vehicle Dynamics,
Influence Surface, Cable Tuning); **Time History is being migrated** via the
function-library route (TH-1 done, TH-2 next). Two remain launchers **by
design**: Linear Static (the always-available current-model run) and Construction
Stages (operates on the shared stage set).

## Decisions / open questions

* **Two lists, not one.** Keep `nonlinear_cases` separate rather than folding
  nonlinear into the generic `AnalysisCase{params}` — preserves the rich editor,
  its tests, and staged `continue_from` wiring. Revisit only if it causes drift.
* **Run replaces re-prompt.** A saved case Runs from `params` with no dialog.
  (The launcher rows that remain still re-prompt until their type is migrated.)
* **Linear Static** stays a singleton launcher (the always-available run of the
  current model), not a saved case.
