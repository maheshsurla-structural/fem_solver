# E2 — Initial conditions / state chaining (sub-plan)

**Created:** 2026-09-16
**Parent:** [`analysis_cases_commercial_parity_plan.md`](analysis_cases_commercial_parity_plan.md) ▸ **E2** (the headline feature)
**Status:** ✅ **COMPLETE 2026-09-16** — Time History, P-Δ Modal, Response
Spectrum and Buckling can all start from a nonlinear case's committed state
(E2a–E2e), the Stiffness-to-use card is in every relevant dialog, and dangling
references are prevented + detected (E2-ui). E2a–E2e merged + pushed to
`origin/main`; the E2-ui delete-guard is the final commit.
**Reference:** SAP2000 *Load Case Data* ▸ **Stiffness to Use** — *"Zero Initial
Conditions – Unstressed State"* vs *"Stiffness at End of Nonlinear Case,"* with
the note *"Loads from the Nonlinear Case are NOT included in the current case."*

---

## 1. Goal & the workflow it unlocks

Let **any** analysis case start from the committed **state + stiffness** at the
end of a saved nonlinear case, instead of always from the unstressed state.

This is precisely the user's own fiber-hinge validation chain (the `FH1_*`
cases in the benchmark screenshot):

```
FH1_P2400_PRELOAD  (Nonlinear Static: gravity + 2400-unit axial, held)
      │  ← state + geometric stiffness at end of preload
      ├── FH1_MONO_P2400       (Nonlinear Static push, from preload)   ✅ already works (continue_from)
      ├── FH1_TH_0.25 In +ve   (Time History, from preload)            ⬅ E2
      ├── FH1_TH_0.50 In +ve   …                                        ⬅ E2
      └── (P-Δ Modal / Response Spectrum on the preloaded column)       ⬅ E2
```

Commercial equivalents unlocked: **P-Δ modal**, **response spectrum on a
preloaded structure**, **time history from gravity**, **buckling from an
arbitrary committed stress state**.

## 2. Semantic model (what carries — matches SAP exactly)

* **Carried:** the committed nodal displacements, every element/material's
  committed history, and hence the **tangent / geometric stiffness** implied by
  that state.
* **NOT carried:** the upstream case's *applied loads*. Mirrors SAP's explicit
  warning. If the downstream case needs the gravity/axial preload held *during*
  its run (it does, for TH — otherwise the structure springs back), the user
  re-declares it in that case's **Loads Applied** (parent-plan **E3b**), or —
  interim, before E3b — E2 offers a **"hold source loads constant"** checkbox
  that reuses the source's held force vector. This keeps the transient in
  equilibrium at t=0.
* **Source eligibility:** the initial condition may reference only a
  **Nonlinear Static** case (`project.NonlinearCase`), exactly as SAP restricts
  it to a nonlinear case. Chains are allowed (source may itself continue from
  another) and validated for cycles.

## 3. Engine findings (grounded — what exists vs what's new)

| Component | File | State-aware today? | E2 work |
|---|---|---|---|
| **StagedAnalysis** | `src/femsolver/analysis/staged.py` | ✅ carries committed state (`keep_state`) + holds prior load (`const_force`), leaves model at final committed state | reuse as-is; factor a "run to state" helper out of the desktop layer |
| **NonlinearStaticAnalysis** | `nonlinear_static.py:111` | ✅ `keep_state` / `const_force` params | none |
| **Time History** (`NonlinearTransientAnalysis`) | `nonlinear_transient.py:199` | ✅ **"We do NOT reset Node.disp/velocity — they hold initial conditions"** (l.201) | wire the held load into `load_function`; seed velocity=0 |
| **Buckling** (`LinearBucklingAnalysis`) | `buckling.py:70` | ⚠️ forms `K + K_g` but from its **own** reference load | add an option to take the prestress from the **committed model state** instead of re-loading |
| **Modal** (`EigenAnalysis`) | `eigen.py:87` | ❌ uses `assemble_stiffness` = **elastic K only** | **new capability:** `stiffness="tangent"` → assemble `K + K_g` at the committed state |
| **Response Spectrum** | `response_spectrum.py:144` | ❌ (built on modal) | inherits the modal `stiffness="tangent"` option |

**The only genuinely new solver capability is tangent-stiffness modal (P-Δ
modal).** Everything else is plumbing on primitives that already exist —
`_assemble_tangent` / `_assemble_geometric` already live in
`static_integrator.py` (buckling imports them), so the tangent assembly is a
reuse, not a from-scratch build.

## 4. Data model (`desktop/project.py`)

```python
# on BOTH NonlinearCase and AnalysisCase:
initial_condition: tuple = ("zero",)        # or ("state", <nonlinear_case_id>)
```

* JSON-friendly (list on disk). Serialize + tolerant load (default `("zero",)`
  for old projects).
* **`continue_from` stays** for the nonlinear→nonlinear pushover case: it does
  more than seed state — it *replays* the ancestor's push so the hysteresis
  curve is continuous (`nonlinear.py:_case_chain`). `initial_condition` is the
  *general, cross-type* state seed. A nonlinear case may use either; documented
  in §9. (Unifying the two is a later cleanup, not part of E2.)

## 5. Engine changes

* **E2-eng-1 — "run to committed state" helper.** Factor out of
  `desktop/nonlinear.py:run_case` a
  `seed_to_committed_state(project, nl_case, *, materials) -> (model, F_const)`
  that builds the model, runs the source case's chain via `StagedAnalysis`
  (already leaves the model committed), and returns the model + the held
  constant-force vector. `run_case` then reuses it (no behaviour change to
  pushover). This is the shared prelude every downstream runner calls.
* **E2-eng-2 — tangent modal.** `EigenAnalysis(..., stiffness="elastic" |
  "tangent")`; when `"tangent"`, assemble `K + K_g` at the model's current state
  via the existing `_assemble_tangent`. Default `"elastic"` (no change to
  existing modal). RS inherits it through its `EigenAnalysis` construction.
* **E2-eng-3 — buckling from state.** `LinearBucklingAnalysis(..., prestress=
  "reference" | "current_state")`; `"current_state"` skips the internal
  reference-load step and differences the tangent at the (already committed)
  state. Default `"reference"` (unchanged).
* **E2-eng-4 — transient held load.** Add the held `F_const` to the transient's
  `load_function` as a constant term so equilibrium holds at t=0 (or expose an
  `initial_force`/`const_force` seam mirroring the static one).

## 6. Desktop wiring (`desktop/main_window.py`, `case_types.py`)

Each downstream `run_*` gains a **prelude**: if the case's `initial_condition`
is `("state", nl_id)`, call `seed_to_committed_state(...)` and hand the returned
seeded model (and `F_const`) to the analysis instead of a fresh
`build_model()`; else behave exactly as today.

* `run_modal` — seeded model + `stiffness="tangent"`.
* `run_response_spectrum` — seeded model + tangent modal basis.
* `run_buckling` — seeded model + `prestress="current_state"`.
* `run_timehistory_dialog` — seeded model (ICs already respected) + held load.
* `case_types.*.build_config` carries `initial_condition` through; a source that
  was deleted raises a friendly error (the TH-function-deleted pattern already
  in `_run_saved_case`).

## 7. UI — the "Stiffness to Use" panel

* A shared `InitialConditionWidget`: radio *Zero initial conditions* /
  *State at end of nonlinear case* + a combo of eligible `NonlinearCase`s +
  the SAP note label + (interim) the "hold source loads constant" checkbox.
* Lands in the per-type dialogs now; **folds into the E1 unified editor** as the
  *Stiffness to Use* group when E1 arrives. Building it as one reusable widget
  means E1 gets it for free.
* Manager `detail` string gains a *"from ‹source case›"* suffix so the
  dependency is visible in the list; the E4 **Load Case Tree** renders the full
  chain.

## 8. Validation (how we prove it — closed-form, per type)

Correctness is provable in closed form; each becomes a `tests/` case:

* **P-Δ modal** — a column under axial load `P` has fundamental frequency
  `ω(P) = ω₀·√(1 − P/P_cr)` (vibration of a beam-column). Seed the column to a
  committed state under `P` via a preload case, run tangent modal, assert the
  softened frequency matches the analytical curve; `P → P_cr` drives `ω → 0`.
* **Buckling from state** — buckling seeded from a committed state under a
  reference load must return the **same** critical multiplier as the existing
  standalone buckling under that load (consistency vs the already-Euler-validated
  path).
* **Time history from gravity** — preload statically, then run TH with **zero**
  ground motion but the preload held: displacements must **stay** at the static
  solution (no spurious transient) — the equilibrium check for E2-eng-4.
* **Round-trip** — save a case with `("state", N)`, reload the project, run
  headless: identical result (serialization).

## 9. Sequencing (strangler, shippable each step)

- [x] **E2a** ✅ 2026-09-16 (`422b230`). `initial_condition` on both case classes
  + `_coerce_ic` + serialization/migration + manager `· from ‹source›` suffix.
- [x] **E2b** ✅ 2026-09-16 (`422b230`). `seed_to_committed_state` helper +
  `run_case` refactored onto shared module-level stage factories (pushover
  behaviour-identical); `StagedAnalysis.const_force_final` exposed.
- [x] **E2c** **Time History from state** ✅ 2026-09-16. `run_time_history` gains
  `initial_case` + `hold_source_loads`: when set it seeds via
  `seed_to_committed_state` (with `density`, so the model bears mass) and, when
  held, adds `F_const` as a constant term to the excitation so the preload stays
  in equilibrium. Reusable **`analysis_ui.InitialConditionCard`** (Stiffness-to-
  use radio + source combo + hold checkbox + SAP note) built here and dropped
  into `TimeHistoryCaseDialog`; the case now carries `AnalysisCase.initial_condition`
  + `params["hold_source_loads"]`. `CaseType.build_config` threads
  `initial_condition` (base + Modal/Buckling/RS accept-and-ignore, TH consumes);
  `_run_saved_case` passes `c.initial_condition`. Runner shows the initial-state
  summary. **Validation: gravity-hold equilibrium test green** (preload + zero
  motion + hold ⇒ structure stays at rest; without hold it springs back).
- [x] **E2d** **P-Δ Modal + Response Spectrum** ✅ 2026-09-16. `EigenAnalysis`
  gains `stiffness="elastic"|"tangent"` (tangent = `K + K_g` at the committed
  state, mirroring buckling's dedicated-K_g / state-tangent split);
  `ResponseSpectrumAnalysis` threads it and skips `reset_results` so the seeded
  state survives. `run_modal` / `run_response_spectrum` seed a fiber model from
  the source case (`_seed_modal_model`, mass from material ρ) and solve on the
  tangent basis; both dialogs carry the `InitialConditionCard` (`show_hold`
  toggle added — eigen inherits stiffness+state only). Validation:
  `tests/test_pdelta_modal.py` (ω(P)=ω₀√(1−P/Pcr) within 5%, ω→0 at Pcr,
  tangent==elastic unstressed) + fiber-path softening / RS period-lengthening.
- [x] **E2e** **Buckling from state** ✅ 2026-09-16. `LinearBucklingAnalysis`
  gains `prestress="reference"|"current_state"`; "current_state" skips the
  internal static solve + reset and buckles from the committed state (λ scales
  that state's load). `run_buckling` seeds via the shared `_seed_state_model`
  (renamed from `_seed_modal_model`, `require_mass=False` for buckling);
  `BucklingType` config → 4-tuple; `BucklingDialog` carries the card. Validation:
  `tests/test_buckling_from_state.py` — from-state == reference-load buckling
  under the same load, and recovers the analytical Euler load.
- [x] **E2-ui (card)** ✅ 2026-09-16. The `InitialConditionCard` is now in every
  dialog where continuing from a nonlinear state is meaningful — Time History,
  Modal, Response Spectrum, Buckling. The other types (moving load, temp
  gradient, load rating, vehicle dynamics, influence surface, cable tuning) have
  no "from state" concept, so nothing to adopt there.
- [x] **E2-ui (delete-guard)** ✅ 2026-09-16. `analysis_cases_dialog._delete`
  refuses to delete a nonlinear case referenced by another case's `continue_from`
  or `initial_condition` (names the dependents); `model_checks` flags a dangling
  `initial_condition` on nonlinear + analysis cases.

**Sequence (all done):** E2a ✅ → E2b ✅ → E2c (Time History) ✅ → E2d
(P-Δ modal + RS) ✅ → E2e (buckling from state) ✅ + IC card in all four relevant
dialogs ✅ → E2-ui delete-guard ✅. **E2 is complete.**

## 10. Risks & decisions

* **Risk — transient equilibrium at t=0.** The dominant correctness risk; fully
  covered by the gravity-hold test (§8). If held-load wiring proves fiddly, ship
  E2c with the "hold source loads constant" checkbox and defer the full
  Loads-Applied integration to E3b.
* **Risk — tangent modal conditioning.** Near `P_cr` the tangent stiffness
  loses positive-definiteness; report a clear "structure near buckling —
  reduce preload" message rather than a raw solver failure. The benchmark
  deliberately probes `P → P_cr`.
* **Decision — keep `continue_from` separate** (pushover curve-continuation) from
  the general `initial_condition` (state seed). Lower risk; revisit unification
  after E2 lands. *Needs a nod.*
* **Decision — held loads: checkbox now, Loads-Applied later?** Recommend the
  interim checkbox in E2c so Time History is usable before E3b. *Needs a nod.*
* **Open — source = nonlinear only** (SAP parity) vs also allowing a linear
  static source. Recommend nonlinear-only for E2; linear-static IC is a trivial
  later add if wanted.
