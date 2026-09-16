# Analysis cases → commercial parity — the "finest level" roadmap

**Created:** 2026-09-16
**Status:** IN PROGRESS — successor to
[`analysis_cases_manager_plan.md`](analysis_cases_manager_plan.md) (complete: 10
saved, named, multi-instance types). **Shipped to `main` (`7889505`) 2026-09-16:**
the first slices of **E2** — E2a/E2b (initial-condition data model + the
`seed_to_committed_state` engine helper) and **E2c** (Time History from a
nonlinear case's committed state); see the
[E2 sub-plan](analysis_cases_initial_conditions_plan.md). **Everything else in
this roadmap is still pending** — see §4 sequencing and the §6 tracker.
**Reference:** SAP2000 / CSiBridge *Define ▸ Load Cases* + *Load Case Data*
dialogs (the user's benchmark).
**Scope:** close the gap from "a working saved-case manager" to a
**commercial-grade analysis-case system** — one that looks, behaves, and models
the way SAP2000 / CSiBridge / MIDAS do.

---

## 1. Where we are (the foundation is strong)

The manager migration is done and shipped to `main`:

* `project.AnalysisCase{id, name, type, params, notes}` — saved, named,
  multi-instance ([`desktop/project.py:152`](../../desktop/project.py)).
* Adapter registry `desktop/case_types.py` — **10 types** (Modal, Buckling,
  Moving Load, Temp Gradient, Load Rating, Response Spectrum, Vehicle Dynamics,
  Influence Surface, Cable Tuning, Time History) with a clean
  `detail / edit / build_config / dispatch` contract, "persist inputs not
  derived objects."
* `desktop/analysis_cases_dialog.py` — Add ▾ / Modify / **Duplicate** / Delete /
  reorder (↑↓) / Run, unique-name auto-suffix, notes-on-hover.
* `project.NonlinearCase` with `continue_from` **staged pushover chaining**
  ([`desktop/nonlinear.py:462`](../../desktop/nonlinear.py) `_case_chain`).
* `project.TimeHistoryFunction` library referenced by TH cases.
* A `model_checks.Check` validation framework, incl. `_check_nonlinear_cases`
  (already validates `continue_from`).

This is genuinely good — better than most in-house solvers. What it is **not**
yet is *SAP-grade*. The gaps below are what a commercial user notices in the
first five minutes.

---

## 2. The gap, mapped to the benchmark

| SAP2000 *Load Case Data* feature | Our state today | Gap |
|---|---|---|
| **One dialog, a Type dropdown** reshapes the panel | N separate dialogs behind an **Add ▾** menu | **E1** unified editor shell |
| **Stiffness to Use** — Zero IC vs *"Stiffness at End of Nonlinear Case"* | only `NonlinearCase.continue_from` (pushover-only, replays pushes) | **E2** general initial-conditions/state chaining ★ |
| **Loads Applied** — (Load Pattern, Scale) rows | cases implicitly use "current loads / combinations" | **E3** explicit Loads-Applied spec |
| **Show Load Case Tree** — dependency DAG | none | **E4** case tree + cycle/stale detection |
| **Run Analysis** across every case, status, Run All | `RunAnalysisDialog` is **stale** — lists only Linear/Nonlinear/TH, *misses all 10 saved cases*; no persisted status | **E5** run management |
| **Set Def Name / Notes dialog / Design…** | notes field only; no auto-name; no design tag | **E6** case metadata & chrome |
| **Load Combos combine case *results*** (static + RS + moving envelope) | `LoadCombination` combines **patterns** (pre-analysis) only | **E3b** result combinations |
| Referential integrity on delete/rename | `continue_from` + TH-function guards only | **E7** integrity & validation |
| Type list: Multi-step Static, Steady State, PSD, Hyperstatic | not present | **E8** (mostly future engine work) |

★ = highest engineering value; it is exactly the user's own validation workflow
(`FH1_P2400_PRELOAD → FH1_MONO_P2400 → FH1_TH_* In ±ve`: a gravity/axial preload,
then monotonic, then a time-history sweep — every downstream case wants the
**state at the end of the preload**).

### Terminology note (a strategic decision, not a bug)

SAP's vocabulary is *inverted* from ours:

* SAP **Load Pattern** (spatial load: DEAD/LIVE/WIND) = our `project.LoadCase`
  (we even expose `Project.load_patterns()` for the engine —
  [`desktop/project.py:647`](../../desktop/project.py)).
* SAP **Load Case** (the *analysis*: type + IC + loads-applied) = our
  `NonlinearCase` / `AnalysisCase` / the Linear-Static launcher.
* SAP **Load Combination** (combines analysis *results*) = our
  `LoadCombination` (combines *patterns*).

We should **keep "Analysis Case" as the umbrella term** (it is clearer than
SAP's overloaded "Load Case"), but relabel `LoadCase` → **"Load Pattern"** in the
loads UI to kill the Load-Case/Analysis-Case ambiguity. This is a rename +
migration, tracked as **E3a**.

---

## 3. Epics

### E1 — Unified "Load Case Data" editor shell  ★ visible signature

Build one `CaseEditor` dialog matching SAP's layout:
`Name [Set Def Name] · Notes [Modify/Show] · Type ▾ · Design… ·
Stiffness-to-Use · <type body> · OK/Cancel`. The **Type ▾** swaps the body;
each existing per-type dialog's *body* becomes a swappable **page** (a
`QStackedWidget` of the current widgets), so we reuse all validated inputs and
`build_config`. **Add** then opens this one dialog defaulted to a type (SAP's
*Add New Load Case…*), and the **Add ▾** menu becomes an optional shortcut.

* Reuses the shared `CaseHeader` (Name/Notes) already on every dialog.
* Allow **changing a case's type** where params map; warn+reset where they don't.
* Files: new `desktop/case_editor.py`; refactor each `*_dialog.py` to expose its
  body as an embeddable widget; `case_types.CaseType.edit` grows a "give me your
  body widget" seam.
* Risk: **medium** (touches every type dialog). Do it as a strangler — keep the
  standalone dialogs working while the shell adopts them one page at a time.

### E2 — Initial conditions / state chaining  ★★ highest engineering value

Generalize `continue_from` into a first-class **initial condition** on *every*
case: `("zero",)` (unstressed) or `("state", nonlinear_case_id)` ("continue from
the committed displacement + geometric stiffness at the end of case N"). This is
SAP's *Stiffness to Use* radio group. Unlocks the workflows commercial users
expect and that this project needs for its own fiber-hinge validation:

* **Modal / Response Spectrum after gravity** (P-Δ / geometric-stiffness modes).
* **Pushover / Time History from a gravity+axial preload** (the `FH1_*` chain).
* **Staged static** on a preloaded structure.

Work items (engine-touching — sequence per type):

* **E2a** data model: `AnalysisCase.initial_condition` +
  `NonlinearCase.initial_condition` (subsumes `continue_from`); serialize +
  migrate old `continue_from`.
* **E2b** engine seam: extend `nonlinear._case_chain` into a general
  "run upstream case, hand its committed state (u, internal vars) + geometric
  stiffness to the downstream analysis." Persist/cache the end-state so a
  downstream case need not re-run the parent every time (see E5 status/results).
* **E2c** wire it type-by-type: Nonlinear (already), then Time History, Modal,
  Response Spectrum, Buckling (buckling-from-state is the P-Δ eigenproblem).
* **E2d** editor: the *Stiffness to Use* panel in E1, listing eligible nonlinear
  cases with the exact SAP note *"Loads from the Nonlinear Case are NOT included
  in the current case."*
* Risk: **high** — real solver work. Biggest payoff. Deserves its own sub-plan.

### E3 — Loads Applied, patterns, and result combinations

* **E3a** Rename `LoadCase` → **Load Pattern** in the UI (data class can keep its
  name or rename with a serialization shim); the loads editors and nature map
  follow. Removes the core naming confusion.
* **E3b** **Loads Applied** spec on static/nonlinear cases: an explicit list of
  `(pattern_id, scale)` rows (SAP's grid), replacing "implicitly all current
  loads." Lets you save "1.0 Dead + 0.5 Live" as a named case without inventing
  a combination. Reuse `Project.load_patterns()` — the scaling machinery already
  exists (`apply_case(model, cid, factor)`).
* **E3c** **Result combinations**: extend `LoadCombination` to combine analysis
  *case results* (envelope a static with a Response-Spectrum or Moving-Load
  result), not just patterns. This is a meaningful modeling gap — today you
  cannot build "1.2D + 1.0E(RS)" as a design combination.
* Risk: E3a low, E3b medium, E3c medium-high (needs a results store — see E5).

### E4 — Load Case Tree (SAP's *Show Load Case Tree…*)

A tree/graph view of case dependencies (initial-condition chains, modal→RS/TH
links, pattern usage). Cycle detection, **topological run order**, and **stale**
flags (a downstream case whose upstream inputs changed since its last run).
Extend `model_checks` (already DAG-aware for `continue_from`) to the general
graph. Risk: **low-medium**; high polish-per-effort.

### E5 — Run management to commercial grade

* **E5a** *(quick win)* Fix the stale `RunAnalysisDialog`: it lists only
  Linear/Nonlinear/TH and **misses all 10 saved `analysis_cases`**
  ([`desktop/run_analysis_dialog.py`](../../desktop/run_analysis_dialog.py) has
  zero `analysis_cases` references). Fold them in via the `case_types` registry.
* **E5b** **Run All** in dependency order (uses E4's topo sort); "Run/Do-not-run"
  already exists per row.
* **E5c** Persist per-case **status** (`Not Run / Running / Finished / Not
  Finished / Could Not Start`) + **last-run timestamp** on the case; show as
  columns in the manager (SAP does). New fields → serialization + migration.
* **E5d** **Results association**: link each case to its result set (extend the
  `runs`/`RunRecord` store, [`desktop/project.py:255`](../../desktop/project.py))
  keyed by case id, so re-opening shows "Finished" and jumps to results.
* Risk: E5a low; E5b–d medium.

### E6 — Case metadata & dialog chrome (parity polish)

* **E6a** **Set Def Name** — per-type auto-name generator (`MODAL1`, `RS1`, …),
  unique against existing names (`_unique_name` already exists).
* **E6b** **Notes** — a proper *Modify/Show* multi-line editor (we store notes;
  just needs the richer widget SAP uses).
* **E6c** **Design…** — tag a case with a design purpose (strength/service/…),
  feeding the auto-combination generator; ties into the material-code/design-code
  work stream.
* **E6d** Manager UX: right-click context menu (Run/Modify/Duplicate/Delete/
  Rename/Show tree), a Status + Last-run column, and a filter/search box
  (mirror the model-navigation filter). Add-Copy parity: our **Duplicate**
  already covers SAP's *Add Copy of Load Case…*.
* Risk: **low** throughout.

### E7 — Referential integrity & validation

* Guard deletion/rename of a nonlinear case referenced by another case's
  initial condition (generalize the existing `continue_from` delete guard in
  `analysis_cases_dialog._delete` + `model_checks._check_nonlinear_cases`).
* Surface case validation **in the manager** (a warning glyph per row), not only
  at Run time.
* Serialization versioning for all new fields (`initial_condition`, status,
  loads-applied); tolerant load of old projects. Risk: **low-medium**.

### E8 — New analysis types (future engine work)

SAP's type list we don't yet offer as saveable cases: **Multi-step Static**,
**Steady State**, **Power Spectral Density**, **Hyperstatic**. These are engine
capabilities first, manager rows second — out of scope for parity of the
*manager*, listed so the Type ▾ can grow into them. Risk: **high** (solver work),
**deferred**.

---

## 4. Recommended sequencing

Optimizing for "looks and feels commercial fast, then gets deep":

* **Phase 1 — parity you can see, low risk (≈ the first sprint).**
  E5a (fix stale Run dialog) · E6a/b/d (Set Def Name, Notes dialog, Status/Last-run
  columns, context menu, filter) · E4 (Load Case Tree) · E3a (rename → Load
  Pattern). Ships the SAP *look* and closes the most embarrassing gap (Run
  ignoring saved cases) without touching the solver.
* **Phase 2 — the unified editor.** E1 (`CaseEditor` shell) — big visual payoff,
  reuses existing bodies.
* **Phase 3 — the engineering leap.** E2 (initial-conditions/state chaining) with
  E5c/d (status + results store it depends on). This is the star; give it a
  dedicated sub-plan (`analysis_cases_initial_conditions_plan.md`).
* **Phase 4 — depth.** E3b/c (Loads-Applied + result combinations), E6c
  (Design…), E7 (integrity), then E8 as the engine grows.

**My recommendation:** start with **E5a + E4 + E6d** (a tight, high-visibility
first commit that makes the manager feel finished), then commit to **E2** as the
headline feature — it is the one thing that separates a "case list" from a
commercial analysis engine, and it is what your own `FH1_*` validation chain
actually needs.

---

## 5. Decisions / open questions

* **Umbrella term** — keep "Analysis Case" (recommended) or adopt SAP's "Load
  Case"? Drives E3a labels. *Needs a call.*
* **Unified dialog vs Add ▾ menu** — do both (E1 shell + keep the menu as a
  shortcut), or replace the menu entirely? *Recommend: both.*
* **State caching for E2** — persist a nonlinear case's end-state in the project
  file (fast re-runs, large files) or recompute on demand (small files, slower)?
  *Recommend: recompute, cache in-session only, revisit if slow.*
* **Result combinations (E3c)** — needs a durable per-case results store; couple
  it with E5d rather than building twice.

## 6. Tracker

* [ ] **E5a** Run dialog includes saved `analysis_cases` *(quick win)*
* [ ] **E6a** Set Def Name · **E6b** Notes editor · **E6d** columns/context-menu/filter
* [ ] **E4** Load Case Tree (DAG + cycle/stale)
* [ ] **E3a** `LoadCase` → "Load Pattern" relabel
* [ ] **E1** `CaseEditor` unified shell (strangler over the per-type bodies)
* [x] **E2a–c** Initial-conditions / state chaining — data model, seeding helper,
  **Time History from state** ✅ 2026-09-16 (`ead1d26`/`422b230`/`7889505`) *(sub-plan)*
* [ ] **E2d / E2e / E2-ui** P-Δ modal + Response Spectrum from state · buckling
  from state · adopt `InitialConditionCard` elsewhere + referential-integrity
  delete-guard *(sub-plan)*
* [ ] **E5b/c/d** Run All (topo) · persisted status/last-run · results association
* [ ] **E3b** Loads-Applied spec · **E3c** result combinations
* [ ] **E6c** Design… classification · **E7** integrity/validation/versioning
* [ ] **E8** new engine types (deferred)
