# Bridge analysis plan — MIDAS Civil / CSiBridge parity

**Created:** 2026-09-14
**Scope:** *analysis capabilities* for bridges, benchmarked against MIDAS
Civil and CSiBridge / SAP2000. Pre/post-processing GUI, interoperability,
reporting and packaging are tracked separately (see the desktop commercial
roadmap); this document is about what the **solver** can and cannot do for
bridge engineering.

Companion to `future_roadmap.md` (which is material/section-breadth focused).
The bridge **engine** is already deep; this plan closes the remaining
analysis gaps that MIDAS/CSI users expect.

---

## 1. Where femsolver already matches MIDAS / CSI (engine level)

| Capability | femsolver | Notes |
|---|---|---|
| Moving load / **influence lines** (2-D) | `bridges/moving_load.py`, `influence.py` | factorize-once; reaction/disp/force ILs |
| **Vehicle libraries** | AASHTO HL-93 (truck+tandem+lane), IRC Class A/AA/70R | `influence.py` presets + envelopes |
| **Cable element + form-finding** | `CableElement3D` (Ernst), force-density FDM | cable-stayed / suspension shape + pretension |
| **Staged construction** | `IncrementalStagedAnalysis` | element birth/death, per-element force history |
| **Time-dependent concrete** | creep, shrinkage, strength-gain (EN/ACI), AAEM, `StepByStepCreep` | |
| **Prestress / PT** | `Tendon` (pre/post), friction + losses, primary + secondary | `tendon.py`, `pt_tendon.py`; PSC design AASHTO/EC2 |
| **Composite section** | modular ratio *n*, effective width, fibre stresses | `composite_section.py` |
| **Seismic** | modal, RS (SRSS/CQC), THA, multi-support excitation | `analysis/` |
| **Seismic isolation (physics)** | lead-rubber + friction-pendulum, gap/hook/uplift | `elements/isolators.py`, `zero_length.py` |
| **Buckling / P-Δ** | dedicated `K_g` (2-D) + corotational geometric NL | `analysis/buckling.py` |

**None of this is reachable from the desktop GUI yet** (the GUI is
frame-only, with no bridge model). GUI is out of scope for this plan except
where an item needs data the model does not yet carry — those are flagged
**[GUI]** and raised with the user before any GUI change.

---

## 2. Gap tiers (analysis)

### Tier 1 — core parity (do first)

| # | Item | What MIDAS/CSI call it | Nature |
|---|---|---|---|
| **T1.1** | **Influence surfaces** — 2-D moving load on decks / grillages; single-vehicle 2-D placement optimisation; uniform patch (lane) load integration | Influence Surface / Moving Load on plate deck | **new analysis** |
| **T1.2** | **Multi-lane placement** — transverse lane optimisation + multi-presence / lane-reduction factors; design-lane vehicle+lane combination | Traffic Line/Lane + multi-presence | new analysis (builds on T1.1) |
| **T1.3** | **Temperature-gradient load** — AASHTO §3.12.3 / EN 1991-1-5 nonlinear vertical gradient → self-equilibrated axial+curvature eigenstrain + continuity moments | Temperature Gradient load | new load type |
| **T1.4** | **Construction-stage bridge workflows** — camber / geometry control (backward + forward staged), tendon **stressing sequence** within stages, composite (wet→hardened) staging | Construction Stage Analysis + camber | driver on existing staged core |

### Tier 2 — differentiators

| # | Item | Note |
|---|---|---|
| **T2.1** | **Vehicle–Bridge Interaction / moving-load time-history** — moving mass, dynamic amplification, EN 1991-2 HSLM high-speed-rail resonance | ✅ **DONE 2026-09-14.** T2.1a moving-force (`bridges/moving_force.py`: `VehicleAxles`, `MovingForceAnalysis` on Newmark `TransientAnalysis`, consistent Hermite distribution, dynamic vs quasi-static + DAF; validated PL³/48EI, DAF→1 slow, DAF↑ speed, train resonance; 7 tests). T2.1b **coupled sprung-mass VBI** (`bridges/vbi.py`: `SprungMassVehicle`, `VBIAnalysis` — monolithic average-accel Newmark on the coupled bridge+vehicle system, moving contact blocks `k_s NNᵀ`; returns bridge history, **contact-force history**, DAF; validated: static=WL³/48EI, DAF→1 slow, contact mean=W, matches moving-force when vehicle dynamics secondary; 8 tests). 2-D girder line; engine only. |
| **T2.2** | **Cable-stayed initial-force optimisation** — MIDAS "Unknown Load Factor" / target-shape iteration; nonlinear staged erection (geometric NL + cable sag) | ✅ **ULF DONE 2026-09-14** (`bridges/cable_tuning.py`: `Cable`, `unknown_load_factors` — factorize-once influence matrix of unit cable pretensions vs `ResponseExtractor` targets, solve `b0+Ax=t` via lstsq; `apply_cable_tensions`; `CableTuningResult`). Validated: targets met to machine precision, positive stay tensions, independent re-solve ≈ target, member-force targets, over-/under-determined lstsq. `tests/test_bridge_cable_tuning.py` (7). **Nonlinear staged cable-stayed erection (geom NL + sag) still open.** Engine only. |
| **T2.3** | **Load rating** — AASHTO LRFR, permit rating, rating factors | for existing bridges |

### Tier 3 — completeness

| # | Item | Note |
|---|---|---|
| **T3.1** | Braking / centrifugal / bearing-friction longitudinal live-load forces (AASHTO/IRC/EN) | |
| **T3.2** | Static wind on deck (buffeting/flutter explicitly out of scope) | |
| **T3.3** | AASHTO approximate live-load distribution factors (lever rule / DF equations) | exact via grillage ILs already; code-DF workflow missing |
| **T3.4** | Rail–structure interaction (UIC 774-3 / EN 1991-2 CWR longitudinal) | niche |

### Out of scope (as `future_roadmap.md`)
Aeroelastic flutter, buffeting (CFD), coupled FSI, topology optimisation.

---

## 3. Tier 1 tracker

- [x] **T1.1 Influence surfaces** — ✅ DONE 2026-09-14. `DeckSurface`,
  `InfluenceSurface` (Delaunay/LinearND interpolation, 0 outside hull,
  `integrate_patch`), `InfluenceLineEngine.influence_surface(s)` (reuses the
  factorize-once unit-load core), `Vehicle2D` (+ `from_axle_train` wheel
  split) and `moving_load_surface_envelope` (2-D placement, both directions).
  All in `bridges/moving_load.py` + `influence.py`, exported from
  `bridges/__init__`. **Validation:** IS == direct solve to machine precision
  (Betti) for displacement + reaction; vehicle placement == multi-load direct
  solve. `tests/test_bridge_influence_surface.py` (8). *Engine only — no GUI.*
- [x] **T1.2 Multi-lane placement** — ✅ DONE 2026-09-14. `multi_presence_factor`
  + `AASHTO_MULTI_PRESENCE` (1.20/1.00/0.85/0.65), `number_of_design_lanes`
  (incl. 6.0–7.2 m → 2), `DesignLane` + `generate_design_lanes`,
  `multi_lane_envelope` (per-lane optimise transversely within the band +
  longitudinally over the deck; governing `max_k m(k)·Σ top-k`), and
  `lane_load_surface_envelope` (uniform design-lane load via `integrate_patch`).
  All in `influence.py`, exported from `bridges/__init__`. **Validation:**
  governing combination matches the closed formula to machine precision;
  single-lane == restricted surface envelope × 1.20; lane contributions
  superpose to a direct multi-load solve (Betti). `tests/test_bridge_multi_lane.py`
  (9). *Engine only — no GUI.*
- [x] **T1.3 Temperature gradient** — ✅ DONE 2026-09-14.
  `bridges/thermal_gradient.py`: `TemperatureGradient` + `aashto_gradient`
  (zones 1–4, Fig 3.12.3-2 profile) + `linear_gradient`;
  `equivalent_thermal_actions` (section reduction → ΔT_N, curvature κ,
  equivalent linear gradient, **self-equilibrated σ(y)**); `SectionThermalActions`;
  `apply_beam_thermal_actions` (imposes ε₀ + κ as equivalent beam nodal loads →
  free camber on determinate spans, continuity/secondary moments on continuous).
  Exported from `bridges/__init__`. **Validation:** linear profile → 0 self-stress;
  nonlinear self-stress integrates to N=M=0; simple-span camber = κL²/8 (up for
  top-hotter); fixed-fixed continuity moment = EIκ; restrained axial = EAε₀.
  `tests/test_bridge_thermal_gradient.py` (9). *Engine only — no GUI.*
  *EN 1991-1-5 Annex B table builder deferred (arbitrary profiles already
  supported via `TemperatureGradient`).*
- [x] **T1.4 Construction-stage workflows** — ✅ DONE 2026-09-14.
  `bridges/construction_stage.py` on the `IncrementalStagedAnalysis` core:
  `staged_camber` (→ `StagedCamber`: final deflection, **final camber**
  = −deflection, per-stage **stage_deflection** history, node birth stage) for
  geometry control; `tendon_stage_loads` (a `Tendon`'s equivalent nodal loads
  as one stage's `loads`, model left untouched) for **tendon stressing
  sequence**; `merge_stage_loads`; and **composite (wet→hardened) staging** via
  element birth (deck born at hardening → wet load on bare girder, later load on
  composite). Exported from `bridges/__init__`. **Validation:** single-stage
  camber == direct solve; tendon-at-stage == complete-structure (machine
  precision); composite wet == bare `PL³/48EI_g`, sdl == composite
  `PL³/48E(I_g+I_d)`. `tests/test_bridge_construction_stage.py` (6). *Engine
  only — no GUI.*

**Tier 1 COMPLETE (2026-09-14).**

**Test convention:** each item validated to hand-calc or closed-form where
one exists, else to a direct solve; headless; matches the repo's existing
bridge-test style (`tests/test_moving_load.py`, `test_bridge_*`).

---

## 4. Suggested sequencing

T1.1 → T1.2 (moving-load story for decks) → T1.3 (thermal gradient) →
T1.4 (staged workflows). Then T2.1 (VBI) and T2.2 (cable-stayed ULF) as the
headline differentiators. GUI wiring for bridges (a "bridge modeler") is a
separate, later companion — flagged per item, raised before any GUI change.
