# Construction-stage / staged-construction — commercial-parity roadmap

*Status: **IN PROGRESS — C0 + C1a + C4 + C5 + C1b + C7 DONE** (branch `feat/construction-stage-parity`;
C0/C1a `e96f7ee`, C4 `95f989e`, C5 `1bf7d4b`, C1b `4c7e811`, C7 committed next). This plan takes the staged-construction stack from
"most of the physics exists, fragmented across three drivers and barely exposed
in the GUI" to SAP2000 / CSiBridge / MIDAS Civil grade: one unified nonlinear +
time-dependent staged case that composes birth/death + per-element
creep/shrinkage + tendon stressing + geometric nonlinearity on general 3-D
frame/shell models, driven from a real GUI stage manager. Sibling to the
(complete) bridge-analysis and bridge-GUI work streams.*

**Done so far (C0 + C1a):**
- **C0** — `desktop/project.Stage` extended: `remove_members` (death),
  `duration_days`, `age_at_activation_days`, `creep` flag; JSON round-trip +
  backward-compat migration (`.get` defaults). Tests in
  `tests/test_desktop_construction_stages.py`.
- **C1a** — `IncrementalStagedAnalysis` now accepts a `StagedCreep` config
  (`f_cm`, `chi`, `RH`, `h_0`, `initial_age_days`, `final_time_days`) and applies
  a **per-element, per-stage** age-based EMM factor `1/(1+chi·phi)` (CEB-FIP
  MC2010) in place of only the scalar `stiffness_factor`. `ErectionStage` gained
  `duration_days` / `age_at_activation_days`; `IncrementalStagedResult` gained
  `creep_factors`. **Key correctness fix:** element-force accumulation now uses
  the *effective* (creep-scaled) stiffness, so a determinate structure's member
  forces stay constant under creep while deflection grows (`δ∞ = δinst·(1+φ)`).
  GUI runner (`run_construction_stages`) passes the new stage fields through and
  builds a `StagedCreep` when a stage opts in and a concrete `f_cm` is derivable
  (full material-creep UI is C5). Tests: `tests/test_staged_creep.py` (5) —
  backward-compat, scalar-equivalence, determinate `δ∞=δinst(1+φ)` + force
  invariance, load-age monotonicity, differential-age redistribution. 216
  bridge/staged/analysis-case tests green.
- **C4** — `stage_dialog.StageManagerDialog` now edits each stage's
  **duration (days)**, **age at cast (days)**, a **Compute creep** checkbox, and
  a second **Members removed (falsework / props)** list, with built/removed
  mutual exclusivity and a richer stage-list summary (`built / removed / d / φ`).
  Tests in `tests/test_desktop_construction_stages.py` (dialog edit + reselect +
  exclusivity + a full-path creep-amplifies-deflection integration test).
- **C5** — `Material` gained a structured `creep` dict (`enabled`, `f_cm`, `RH`,
  `h_0`, `chi`); the material editor grew a "Time-dependent (creep / shrinkage)"
  card with those inputs + a live `φ(50 yr, 28 d)` / deflection-multiplier
  readout (`f_cm` auto-seeded from `f'c`+8 MPa). `run_construction_stages` now
  builds `StagedCreep` from a creep-enabled material's props (χ threaded
  through), falling back to `f'c`+8 MPa then elastic. Tests in
  `test_desktop_materials.py` + a χ-affects-droop end-to-end test.
- **C1b** — `StepByStepCreepFrame` (see the C1 sub-epic below): step-by-step
  creep + shrinkage on 2-D frames, exact for axial + constant-moment bending,
  with differential creep. This is the "full creep integration" path (relaxes
  a developing restraint force, redistributes under differential creep) that
  C1a's per-increment EMM cannot do.
- **C7** — staged results beyond camber: the results dialog gained a **Stage
  forces** tab (stage selector → per-element axial/moment/E-factor table + peak
  summary). See the C7 epic below.
- **Next:** wire `StepByStepCreepFrame` into the staged birth/death driver (creep
  across stages) + Gauss-point curvature for exact varying-moment redistribution;
  C6 (tendon modelling + stressing sequence); or C2 (deck bending + 3-D in the
  nonlinear cable erection driver).

---

## 0. Resume here (session hand-off)

**Nothing built yet — this is the charter.** The double-check (§1) found the same
pattern as the slab/wall work: the **engine already carries most of the hard,
code-verified physics**, but it lives in three drivers that don't compose, and
the desktop GUI exposes only a thin 2-D self-weight-erection + camber slice.

**Recommended first slice:** **C0** (unified staged data model) → **C1a**
(per-element age-based EMM creep inside `IncrementalStagedAnalysis`) → **C4**
(GUI stage-manager parity: time / age / loads / activation groups). That trio
turns the existing birth/death driver into a genuine time-dependent staged case
and makes it drivable, before the bigger geometric-nonlinear / tendon lifts.

**How to work:** engine changes are additive (new module `analysis/staged_case.py`
or extensions to `bridges/staged_construction.py`); the only true new physics is
per-element time-stepping creep on **frame** elements (C1a/C1b) and deck bending
in the nonlinear cable driver (C2). Everything else composes existing, tested
pieces or is GUI/data-model plumbing. Tests:
`PYTHONPATH=src <repo>/.venv-gui/Scripts/python -m pytest tests/ -q`
(desktop tests need `QT_QPA_PLATFORM=offscreen`; patch modal `QMessageBox` in
headless tests or they hang). **User authorized this area but wants each change
flagged before it lands.**

---

## 1. Where we are (the double-check)

### 1a. The engine already carries the physics — and it is validated

| Capability | Where | Validated |
|---|---|---|
| Element **birth** (stress-free in deformed geometry) | `bridges/staged_construction.py` `IncrementalStagedAnalysis` | ✅ closed-form |
| Element **death** (locked-in force released onto remaining structure — falsework/prop) | same | ✅ propped→un-propped exact |
| Per-stage incremental loads + per-element force history | same | ✅ |
| Per-stage **uniform** EMM stiffness factor | same (`stiffness_factor`) | ✅ softening |
| Scalar **EMM creep redistribution** (CEB-FIP φ + ageing χ) | `StagedConstructionAnalysis` | ✅ |
| **Rigorous step-by-step creep** (Volterra superposition, relaxation) | `analysis/time_dependent.py` `StepByStepCreep` / `StepByStepCreepFE` | ✅ |
| **Age-dependent** strength/modulus/tensile gain (EN1992, ACI209) | `materials/concrete/time_dependent.py` | ✅ |
| CEB-FIP MC2010 creep + shrinkage; PT losses; steel relaxation | `bridges/creep_shrinkage.py` | ✅ |
| **Geometric-nonlinear cable erection** (corotational, Ernst sag, tension-only, staged pretension) | `bridges/nonlinear_staged.py` `NonlinearStagedErection` | ✅ taut-string/Ernst |
| Unknown-load-factor / cable-force tuning | `bridges/cable_tuning.py` | ✅ |
| **Camber / geometry control** (residual camber, stage-deflection table, birth stage) | `bridges/construction_stage.py` `staged_camber` | ✅ |
| Tendon stressing-per-stage loads; composite (wet→hardened) via element birth | same `tendon_stage_loads` | ✅ |

### 1b. But the physics is fragmented across three non-composing drivers

- **`IncrementalStagedAnalysis`** — birth/death, linear, general frame/shell.
  But creep is a *single scalar* `stiffness_factor` applied identically to every
  active element; **no per-element aging, no stress-history creep
  redistribution, no shrinkage eigenstrain, no tendon inside the driver**. It
  also rejects **MP constraints** (rigid links/diaphragms) and has no per-stage
  **support/restraint** activation (only element death).
- **`StagedConstructionAnalysis`** — scalar-EMM creep, one `K0` rescaled per
  stage. **No birth/death.**
- **`NonlinearStagedErection`** — geometric nonlinearity + Ernst sag, but
  **2-D pin-jointed cable/truss only; no deck bending (beam), no 3-D, no shells**
  (deck bending is explicitly a deferred future extension in the docstring).
- **`StepByStepCreepFE`** — rigorous time-stepping creep, but **continuum-only
  (Quad4/Hex8)** — not the beam elements bridges are actually modeled with.

### 1c. The desktop GUI exposes a thin slice

`desktop/main_window.py:run_construction_stages` + `desktop/stage_dialog.py`:
ordered stages, member→stage assignment, auto-sequence L→R, runs
`IncrementalStagedAnalysis` under **self-weight only** and reports **camber**.
**2-D only.** `project.Stage` holds only `id`, `name`, `add_members` — **no
duration, no concrete age, no per-stage loads, no death, no tendon, no creep**.
`StagedConstructionAnalysis`, `StepByStepCreep*`, `NonlinearStagedErection`, and
`tendon_stage_loads` are **not wired to the GUI at all**.

**Conclusion:** ~70–80% of the individual physics exists and is tested. The
parity gap is **integration** (one solver composing the pieces) + **exposure**
(a GUI that lets an engineer actually drive time, age, tendons, temporary
supports, and 3-D), plus two genuine new-physics items (per-element frame creep;
deck bending in the nonlinear cable driver).

---

## 2. The gap, mapped to the commercial benchmark

Benchmark = CSiBridge / SAP2000 "nonlinear staged construction" load case +
MIDAS Civil "construction stage analysis".

| Capability | Engine | GUI | Gap |
|---|---|---|---|
| Element birth / death (add/remove structure) | ✅ | ⚠️ birth only, self-weight only | unify + GUI |
| Per-stage **time** (duration) + concrete **age at activation** | ⚠️ scalar EMM only | ❌ | data model + engine + GUI |
| Per-element **age-based creep** redistribution across stages | ❌ (uniform factor) | ❌ | **engine** |
| **Shrinkage** eigenstrain in the staged frame model | ⚠️ continuum only | ❌ | **engine** + GUI |
| Aging modulus / strength per stage (EN1992/ACI209) | ✅ curves | ❌ | wire + GUI |
| **Tendon stressing** at a stage (primary + secondary + losses) | ✅ loads/losses | ❌ | wire into driver + GUI |
| Per-stage **support / restraint** activation & release | ❌ | ❌ | **engine** + GUI |
| Per-stage **loads** (not just self-weight) | ✅ (driver) | ❌ | GUI |
| **Geometric nonlinearity** (P-Δ / large-disp) in a staged frame | ⚠️ cable driver only | ❌ | **engine** + GUI |
| Cable **sag (Ernst)** + tension-only in staging | ✅ (cable driver) | ❌ | wire + GUI |
| Deck **bending** during cable erection; **3-D** erection | ❌ | ❌ | **engine** |
| Unknown-load-factor / cable-force tuning in staging | ✅ | ❌ | wire + GUI |
| MP constraints (rigid links/diaphragms) in staging | ❌ | ❌ | **engine** |
| **Camber / geometry control** (forward + backward) | ⚠️ forward camber | ❌ | engine (backward) + GUI |
| **Staged results**: per-stage U/σ/M/N, tendon force, stage force tables, stepping, envelopes | ⚠️ in result objects | ❌ (camber only) | GUI |

---

## 3. Epics

Ordered by dependency. ★ = high user-visible value. Engine-only items are
marked **[E]**; GUI/wiring **[G]**.

### C0 — Unified staged data model ★★ (foundation) **[G]** ✅ DONE
Extend `desktop/project.Stage` (and the project JSON schema/migration, as prior
work) so a stage is the commercial "stage object":
- `duration_days`, `age_at_activation_days` (concrete age when born).
- `add_members` / `remove_members` (death), and later `add_areas` / `remove_areas`.
- `add_supports` / `remove_supports` (restraint activation — needs a project
  concept of a named support the stage toggles).
- `loads`: per-stage load patterns (reuse existing `Load` / `MemberLoad`, plus a
  "stage self-weight of newly-born members" flag).
- `tendons_stressed`: tendon ids stressed in this stage (see C6).
- `creep`: whether time-dependent effects run, + the material creep spec ref.

### C1 — Unified staged-construction engine case ★★ **[E]** (the core lift)
A single driver (new `analysis/staged_case.py`, or a superset of
`IncrementalStagedAnalysis`) that composes the pieces. Break into sub-slices:

- **C1a — Per-element age-based EMM creep ★★. ✅ DONE.** Replaced the single
  scalar `stiffness_factor` with a **per-element, per-stage** effective-modulus
  factor computed from each element's own concrete age and the CEB-FIP creep
  coefficient (`bridges/creep_shrinkage.cebfip_creep_coefficient`), via the new
  `StagedCreep` config on `IncrementalStagedAnalysis`. Element-force
  accumulation uses the effective stiffness (determinate forces stay constant,
  deflection grows). See status block above.
- **C1b — Step-by-step creep + shrinkage on frame elements. ✅ DONE (first
  pass).** New `analysis/time_dependent.StepByStepCreepFrame` — the frame
  counterpart of `StepByStepCreepFE`. Each `BeamColumn2D` carries a generalised
  section state (axial force `N`, average curvature `κ=(θ₁−θ₂)/L`); the creep
  (from the element's own stress-increment history) + shrinkage increment is
  imposed each step as an eigenstrain via the temperature-action equivalent-load
  form (`apply_beam_thermal_actions` pattern), then the active set is re-solved
  (`LinearStaticAnalysis`). Per-element `element_phi` gives **differential
  creep**. Result: `CreepFrameResult` (disp / reaction / axial / moment
  histories). **Exact for axial (any indeterminacy) and constant-moment
  bending; the average-curvature imposition is approximate for
  moment-varying-along-a-member bending (mesh-refine).** Tests
  (`tests/test_creep_frame.py`, 5): cantilever tip-moment growth `(1+φ)`,
  determinate moment unchanged, restrained-shrinkage relaxation, homogeneous
  axial no-redistribution, differential-axial redistribution. **Remaining:**
  Gauss-point curvature integration for exact varying-moment bending
  redistribution (the two-span-continuity closed form), 3-D beams, and wiring
  the march into the staged birth/death driver (creep across stages).
- **C1c — Tendon stressing integrated per stage.** Fold `tendon_stage_loads`
  into the driver so a tendon can be stressed at the stage it physically is,
  with time-dependent PT losses (`prestress_long_term_loss`) accruing over
  later stages; expose primary + secondary force history.
- **C1d — Support / restraint activation & release ★.** Let a stage add/remove
  boundary conditions (temporary towers, bearings installed late), releasing the
  reaction the removed restraint carried onto the remaining structure (mirror
  the element-death release logic at the DOF level).
- **C1e — Geometric nonlinearity + MP constraints in the general driver.**
  Optional P-Δ / large-displacement (bridge to `beam_corot` / geometric
  stiffness) inside a stage's Newton solve, and lift the MP-constraint
  restriction so rigid links / diaphragms coexist with staging.

### C2 — Nonlinear staged erection: deck bending + 3-D ★ **[E]**
Extend `NonlinearStagedErection` from pin-jointed 2-D cable/truss to include a
**corotational beam** with a birth datum (deck bending during free-cantilever
erection) and a **3-D** node/DOF set. This is where cable-stayed / suspension
erection differs most from a linear run; the cable nonlinearities already exist.

### C3 — Backward / forward staged analysis + camber-control loop ★ **[E/G]**
- Forward camber already exists (`staged_camber`). Add the **backward** (initial
  shape / target-configuration) solve: given the target final profile, find the
  cast-in camber per segment so the completed structure lands on target — the
  MIDAS "backward analysis" / geometry-control loop.
- Cable-force iteration to a target profile (compose with `unknown_load_factors`).

### C4 — GUI: staged-case manager parity ★★ **[G]** ✅ DONE (first pass)
`stage_dialog.StageManagerDialog` now edits, per stage: name, **duration
(days)**, **age at cast (days)**, a **Compute creep** toggle, the members
**built**, and the members **removed** (falsework / temporary props), with
built/removed exclusivity. **Remaining for a later pass:** element **groups**
(named sets) instead of per-member picking, per-stage supports + arbitrary loads
+ tendons, and richer registration in the Analysis-cases manager
(`case_types.py`). Those depend on C1c/C1d/C6 and a project group concept.

### C5 — GUI: time-dependent material inputs ★ **[G]** ✅ DONE (first pass)
`Material` carries a structured `creep` dict (`enabled`, `f_cm`, `RH`, `h_0`,
`chi`); the material editor has a "Time-dependent (creep / shrinkage)" card
feeding it, with a live `φ(50 yr, t₀=28 d)` + long-term-deflection-multiplier
readout via `cebfip_creep_coefficient`. `run_construction_stages` sources
`StagedCreep` from a creep-enabled material (χ included). **Remaining for a later
pass:** shrinkage `ε_cs` inputs + a full strength-gain/creep-vs-time chart
(`strength_gain_curve`), EN1992/ACI209 code selection + cement class, and
per-material (not per-analysis) creep so a mixed-material model uses each
element's own `f_cm`/`h_0` (needs `StagedCreep` to accept per-tag params).

### C6 — GUI: tendon modeling + stressing sequence ★ **[G]**
A tendon object in the project (profile via `parabolic_drape_profile` /
`TendonProfile`, jacking force, friction/anchorage/long-term loss inputs) and a
per-stage "stress tendon" action, wired to C1c. Render tendon profiles; report
primary + secondary + effective force after losses.

### C7 — GUI: staged results ★★ **[G]** ✅ DONE (first pass)
`ConstructionStageResultsDialog` is now tabbed: **Camber** (as before) + **Stage
forces** — a stage selector over a per-element table (status active/born, axial
`N`, peak `|M|`, and the applied creep E-factor when non-unity), with a
peak-force stage summary, in the project's units. The runner passes the full
`IncrementalStagedResult` + model. **Remaining:** per-stage deformed-shape
stepping in the 3-D view, tendon-force history (needs C6), stage envelopes, and
CSV export.

### C8 — GUI: cable-stayed staging + tuning + 3-D ★ **[G]**
Expose `NonlinearStagedErection` (C2) and cable-force tuning (C3) in the GUI:
draw stays, set pretension per stage, run the geometric-nonlinear erection, and
report stay-tension history + the tuned initial forces for a target profile.

---

## 4. Decisions / open questions

- **D1 — One superset driver, or a new `StagedConstructionCase` that
  orchestrates the three?** Leaning toward growing `IncrementalStagedAnalysis`
  into the general linear+time-dependent case (C1a–C1d) and keeping
  `NonlinearStagedErection` as the geometric-nonlinear cable path (C2), with a
  thin GUI case that dispatches to whichever the model needs. Revisit once C1a
  lands.
- **D2 — Creep fidelity default:** per-element **AAEM/EMM** (one-step, C1a) is
  the cheap default that already matches CSI's "creep — quick" mode; the
  **step-by-step** march (C1b) is the "creep — full integration" equivalent.
  Offer both; default to AAEM.
- **D3 — 2-D vs 3-D:** the current GUI staged flow is 2-D. Gate the richer cases
  behind `ndm==3` like the slab work, or promote staged construction to 3-D
  first? (Bridges want 3-D; frames-only users are fine in 2-D.)
- **D4 — Support activation** needs a first-class "boundary condition object"
  the stage can toggle; today supports are node-fixity flags. Small data-model
  addition in C0/C1d.

## 5. Validation strategy

Every engine slice ships with a closed-form or cross-check test, matching the
existing staged tests (`test_incremental_staged.py`,
`test_bridge_construction_stage.py`, `test_bridge_nonlinear_staged.py`):
- C1a: per-element EMM factor vs. the scalar case when all ages are equal;
  long-term deflection δ∞ = δinst·(1+φ) on a determinate span.
- C1b: relaxation of a restrained shrinkage force vs. `StepByStepCreep`; creep
  redistribution of the support reaction in a two-span continuous beam made
  continuous *after* first loading (the classic staged-continuity benchmark).
- C1c: tendon secondary moment on a continuous beam; effective force after
  long-term loss vs. `prestress_long_term_loss`.
- C1d: temporary-support removal reaction release vs. one-shot final structure.
- C2: 2-D deck-bending erection vs. the linear limit; 3-D against a known frame.
- C3: backward camber → forward re-solve lands on the target profile.
```
