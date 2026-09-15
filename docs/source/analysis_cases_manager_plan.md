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
- [ ] **M6 — Response Spectrum** (the object case). params = spectrum **inputs**
  `{source, damping, asce7:{SDS,SD1,TL} | ec8:{…} | is1893:{…} | custom:[[T,Sa]]},
  num_modes, direction, combination}`. `build_config` rebuilds the
  `ResponseSpectrum` (factor `build_spectrum` into an input→object helper).
  Dialog: `params()` / `from_params()` + header.
- [ ] **M7 — Vehicle Dynamics / Influence Surface / Cable Tuning.** dict configs
  of primitives; header + `initial` each.
- [ ] **M8 — Time History.** audit the ground-motion dialog's config; persist the
  record reference + integration params; header + `initial`.
- [ ] **M9 — Construction Stages** (shared-entity special case). Its config is
  the project-global `stages` list, already persisted. A "stages" case stores
  only run options (reference load / camber target) and references the shared
  stage set — decide whether it is a case at all or stays a launcher.
- [ ] **P — Polish.** Notes surfaced; duplicate-name guard or auto-suffix;
  reorder; per-type default names ("RS-X"); results-history label carries the
  case name; docs.

## Decisions / open questions

* **Two lists, not one.** Keep `nonlinear_cases` separate rather than folding
  nonlinear into the generic `AnalysisCase{params}` — preserves the rich editor,
  its tests, and staged `continue_from` wiring. Revisit only if it causes drift.
* **Run replaces re-prompt.** A saved case Runs from `params` with no dialog.
  (The launcher rows that remain still re-prompt until their type is migrated.)
* **Linear Static** stays a singleton launcher (the always-available run of the
  current model), not a saved case.
