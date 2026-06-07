# femsolver future roadmap

**Last updated:** 2026-06-03
**Current test count:** 2365 passing, 1 skipped, 0 failures
**Recently completed:** Section Designer Phase 2 (II.10 – II.16); Timber D.1.1–D.1.5 (NDS/EC5/IS 883 + CLT); **Bridge analysis core (Phase B.1–B.4): general moving-load/influence-line engine, 3-D cable + form-finding, incremental staged construction with element birth + death, high-level `Tendon` + equivalent-load `apply_to` (beam-columns)**

---

## Prestressing capability (Phase B.4, 2026-06-03)

"Define a tendon, apply the prestress" — `femsolver.bridges.Tendon`:
define type (pre/post-tension) + profile (per-node eccentricity) +
jacking force; `tendon.apply_to(model)` computes losses → effective
`P(x)` → equivalent nodal loads (axial `P`, primary moment `P·e`,
curvature balancing load `8Pa/L²`) on the host beam elements, and
registers as a `LoadPattern`. `tendon_secondary_moment` splits total =
primary + secondary (parasitic). 17 tests: load balancing, straight-
tendon `P·e` (exact), friction `P(x)`, determinate=0 / continuous≠0
secondary moment.

| Element type | Prestress today |
|---|---|
| **Beam-column (2-D + 3-D)** | ✅ equivalent-load `Tendon.apply_to` (B.4 + B.5, biaxial `e_y`/`e_z`) + pre-strained bonded fiber (`PrestressedUniaxial`, section level) |
| **Solid (Hex8 / Tet4) + plane (Quad4)** | ✅ directional initial-stress eigenstrain `apply_initial_stress` + `prestress_initial_stress` (B.6, `f = -∫Bᵀσ₀`) |
| **Plate / shell (ShellMITC4)** | ✅ membrane initial-stress `apply_initial_stress` (B.7, `N₀=σ₀·t`, `f=-∫Bmᵀ N₀ dA`) |

**Prestress is now a "yes" on every element type** — beam-columns
(2-D + 3-D), solids/plane, and shells. Validated by patch tests to
machine precision (free strain `ε=-D⁻¹σ₀`, restrained reactions,
shell↔Quad4 equivalence, balanced-load camber, `P·e` moments).

**Secondary (hyperstatic / parasitic) effects** are fully captured:
`tendon_secondary_forces` returns the secondary support reactions from
a prestress-only solve (zero for determinate, parasitic for
indeterminate — the self-equilibrated equivalent load makes this
exact); `tendon_secondary_moment` / `tendon_secondary_shear` split
total = primary (`P·e`, `P·e'`) + secondary. Cross-validated: pier
secondary moment from `R_end·L` == `M_total − P·e`.

**PSC bridge limit-state design (Phase B.8)** — `femsolver.design.psc`
consumes the primary + secondary moments in the code checks:
extreme-fibre stress engine (`f = P/A ∓ P·e/S ± M/S`, `M = M_ext +
M_sec`); **AASHTO LRFD §5.9.2.3** transfer + service stress limits
(Class U/T, `√f'c` taken in ksi); **EN 1992 §5.10/§7.2** transfer +
characteristic + quasi-permanent + §7.3.1 decompression; and the ULS
`psc_factored_moment` that folds the secondary moment into the strength
demand at load factor 1.0. 22 tests (hand-calc fibre stresses, both
codes' limits, secondary-consumption, end-to-end tendon→secondary→
limit-state chain). Note: ULS section *capacity* (φMn / M_Rd) is
available via the Section Designer's prestressed moment-curvature /
P-M-M. **A one-call ULS bonded-flexure wrapper now exists (Phase
B.10):** `aashto_flexure_capacity` (approximate `f_ps` method,
rectangular + flanged, `β1`, φ interpolation from net tensile strain),
`ec2_flexure_capacity` (simplified rectangular block, `f_pd =
f_p01k/γ_s`), and `psc_flexure_check(M_u, capacity)` → DCR, where
`M_u` (from `psc_factored_moment`) already folds in the secondary
moment. Validated to hand calc (rectangular bonded beam M_n = 1832
kN·m, φ=1.0 tension-controlled). 9 tests.

**Time-dependent concrete (Phase B.11).** `materials/concrete_time.py`
gives the **strength-gain curve** f_cm(t)/f_ck(t)/f_ctm(t)/E_cm(t) per
EN 1992-1-1 §3.1.2 (`β_cc(t)`) and ACI 209R-92 (`t/(a+bt)`), with
`strength_gain_curve(...)` returning plot arrays. Creep φ(t,t₀) and
shrinkage ε_cs(t) already exist (CEB-FIP MC 2010). `analysis/
time_dependent.py` adds the **structural effects**: `age_adjusted_
modulus` (AAEM, Trost-Bazant), `creep_deflection` (δ·(1+φ)),
`shrinkage_axial_force` (restrained member), `differential_shrinkage_
curvature`, and `apply_shrinkage_load(model, eps_sh)` which imposes
shrinkage as an eigenstrain on a continuum FE model (restraint
reactions when held, free shortening when not) by reusing the
thermal-strain machinery. Staged creep redistribution remains in
`IncrementalStagedAnalysis` (per-stage EMM). 18 tests, validated vs
EN 1992 table values (E_cm(C40)=35 GPa) and hand calcs. *Open: a full
multi-step age-adjusted creep-relaxation time integrator at the
structure level (Dirichlet-series / step-by-step) — the per-stage EMM
covers the common case.*

**Step-by-step creep/relaxation integrator (Phase B.12).**
`StepByStepCreep` (in `analysis/time_dependent.py`) solves the uniaxial
ageing-viscoelastic **superposition (Volterra) integral** incrementally
on a time grid: `strain_history(stress)` (exact creep under any stress
history) and `relaxation_history(strain)` (true stress relaxation, no
ageing-coefficient approximation), with optional shrinkage.
`restraint_force_relaxation` gives the relaxing restraint force of a
held strain/shrinkage. Validated: constant stress → ε=σ(1+φ)/E to
machine precision; stress↔strain round-trip exact; CEB-FIP relaxation
implies the textbook **ageing coefficient χ(∞,28)=0.84** (Trost/Bažant
~0.80); creep recovery on unload. 6 tests. This is the rigorous engine
for time-stepping creep; *the remaining step is wiring it per-element
into an FE time-march (per-IP creep state) for full structural
redistribution — the section/member level and per-stage EMM cover the
common cases today.*

**Per-element FE creep time-march (Phase B.13).** `StepByStepCreepFE`
marches a continuum (Quad4/Hex8) model through a time grid: each step
each element's creep-strain increment (from its stored stress history
via the superposition integral) plus the shrinkage increment are
imposed as an **eigenstrain** (reusing `apply_initial_stress`), the
increment is solved, and stress/displacement/reactions accumulate. A
**determinate** structure just creeps (deflection → δ(1+φ), exact); an
**indeterminate** one **redistributes** its internal forces with time.
Validated: determinate axial creep grows exactly ×(1+φ) with stress
unchanged; a both-ends-restrained bar under shrinkage relaxes its axial
stress to the same R/E ratio (≈0.33) the B.12 uniaxial relaxation gives
— end-to-end structural creep redistribution. 4 tests. *(Beam/frame
fiber-level creep march is the natural extension; the continuum march
+ section/member closed forms + per-stage EMM cover the field today.)*

**General PSC section design is turnkey (Phase B.9).** An arbitrary
polygon outline + interior holes (`custom_polygon_section(outline=...,
holes=[...])`) carries rebar + tendons and runs ULS biaxial P-M-M and
SLS stress checks. Two conveniences closed the seams:
`ReinforcementLayout.from_perimeter(section, n_bars, bar_area, cover,
include_holes, n_bars_per_hole)` auto-distributes bars around any
outline (and each hole) at a clear cover via shapely buffer +
arc-length placement; `PscSection.from_section(section, axis)` pulls
gross A, I, and the exact extreme-fibre distances (asymmetric / voided
sections handled). 9 tests; bars verified inside the material at cover,
asymmetric fibres correct. Net: define any voided shape -> auto-rebar
-> ULS + SLS design.

---

## 0. Analysis-core audit (2026-06-03)

A capability review separated **building** analysis (essentially
complete: full element library, rigid diaphragms, P-Δ, modal/RS/THA,
pushover, capacity design) from **bridge** analysis, which had three
real gaps for a commercial product:

| Bridge gap | Status |
|---|---|
| **General moving load / influence lines on arbitrary models** | ✅ **closed — Phase B.1** (`femsolver.bridges.moving_load`): unit-load traversal → influence lines for reaction/displacement/internal-force on *any* model (continuous girders, grillages, frames); plugs into AASHTO/IRC vehicle convolution + HL-93 envelope. Factorize-once fast path; MP-constraint fallback. 22 tests, validated vs closed-form + direct solves to machine precision. |
| **3-D cable element + form-finding (cable-stayed / suspension)** | ✅ **closed — Phase B.2** (`CableElement3D` Ernst-sag + `force_density_form_find`, Schek 1974 FDM): finds the equilibrium shape + pretensions of any 3-D cable net under dead load via a linear solve. 15 tests, validated to machine precision against the cable-beam analogy and to ~1% against the closed-form catenary. |
| **Staged construction with element death** | ✅ **closed — Phase B.3** (`IncrementalStagedAnalysis`): true incremental active-set analysis with element **birth** (stress-free in the deformed geometry) and **death** (the dying element's locked-in force `K_e u_e` is released onto the remaining structure). Per-element force history. 12 tests: 1-stage == one-shot and falsework-removal == no-prop solution, both to machine precision. *(Per-element step-by-step creep with age tracking remains a future refinement; per-stage scalar EMM factor is supported.)* |

**All three bridge analysis-core gaps from the audit are now closed.**

Remaining analysis-core items (lower priority): element-export
packaging tidy-up; tension/compression-only member wrapper;
documented nonlinear-buckling workflow; beam-force diagrams for 3-D
beams (moving-load internal-force ILs are currently 2-D only).

This document is the strategic plan for everything after the General
Section Designer was closed out. Phase C (vendor V&V with MIDAS / CSI /
Abaqus benchmarks) was originally next but has been **deferred at
user request** — to be picked up at a later session once we have more
solver capability to verify against.

---

## 1. Where we are

| Layer | Status | Notes |
|---|---|---|
| Phase A — solver claims audit | ✓ | 18 Beta caveats identified |
| Phase B — Theme HH caveat consolidation | ✓ | 6 of 8 Beta items closed; 2 minor deferred |
| Theme II — Section Designer unification | ✓ | 18 scattered dataclasses → 1 canonical `Section` |
| Section Designer Phase 2 (II.10 – II.16) | ✓ | RC + PSC general section designer; biaxial P-M-M in ACI/EC2/IS 456, M-φ, cracked I, stress field, prestress |
| **Phase C** — vendor V&V (MIDAS / CSI / Abaqus) | **deferred (user)** | Pick up later when more capability is built |
| Phase D — new material classes | pending | Timber, CFS, masonry, aluminum, glass |
| Phase E — specialist analyses | pending | Tunnels, tanks, waves, slope, membranes |
| Tier 5 — Desktop GUI | gated | Behind D + E + V&V |

## 2. Strategic compass (founding promise, restated)

The user's founding strategic statement still holds:

> *"Solver completeness is the prerequisite. No desktop GUI until
> there is nothing more to add to the solver. The unique selling
> point is one software that handles ALL structural engineering
> problems."*

So the **immediate priority is BREADTH of coverage** — every gap in the
"what structural problems can this software solve?" map is a higher
priority than:
- Polishing what's already covered
- Additional design-code coverage in already-supported markets
- Performance optimization

V&V (Phase C) was the next natural step but is now deferred. The
question becomes: **what's the highest-value coverage gain per
session right now?**

---

## 3. Recommended next phases (re-ordered)

### Tier I — biggest "missing problem class" wins (do these first)

These each unlock a whole class of structural engineering problems
that femsolver cannot currently model. Highest strategic value.

| Phase | Closes | Effort | Why |
|---|---|---|---|
| **D.1 — Timber + NDS/EC5/IS 883** | mass timber CLT, glulam, sawn lumber buildings | 8-12 sessions | Mass timber is the fastest-growing structural material globally; zero coverage today; unblocks an entire building type |
| **D.2 — Cold-formed steel + AISI S100** | steel rack systems, light commercial, residential trusses | 8-10 sessions | Huge unaddressed market; CFS sections + cold-work strain hardening + effective width |
| **D.3 — Masonry + TMS 402 / IS 1905 / EC6** | URM + RM buildings, heritage retrofit | 6-8 sessions | Common in low-rise + retrofit work |
| **E.1 — Tunnel staged excavation** | TBM tunnels, NATM | 6-8 sessions | Niche but no commercial RC/excavation FE handles this AND general structures together |
| **E.2 — Storage tank sloshing (API 650)** | cylindrical tanks (oil, water, LNG) | 5-7 sessions | API 650 has specific seismic sloshing requirements |
| **E.3 — Wave loading (Morison)** | offshore platforms, jetties, hydraulic structures | 4-6 sessions | Regular + irregular waves; pile-water interaction |

### Tier II — important but smaller scope

| Phase | Closes | Effort | Why |
|---|---|---|---|
| **F.1 — Steel Section Designer Phase 2** | biaxial M-M-N for any steel polygon; LTB for custom shapes; fiber-section nonlinear steel | 6-8 sessions | Mirror of RC Section Designer Phase 2; brings steel to feature parity |
| **F.2 — Composite Section Designer** | encased composite columns, concrete-filled tubes (AISC 360 Ch. I) | 5-7 sessions | Common in high-rise construction |
| **D.4 — Aluminum + ADM** | architectural facades, light structures | 4-6 sessions | Specialty market |
| **D.5 — Glass + ASTM E1300 / EN 13474** | structural glass façades, glass beams | 4-6 sessions | Specialty / luxury |
| **E.4 — Slope stability driver** | wraps MC soil into LE / FE limit equilibrium | 4-5 sessions | Has MC constitutive, just needs the driver |
| **E.5 — Membrane form-finding** | tensile fabric structures, cable nets | 5-7 sessions | Specialty roof structures |
| **II.17 — Slender column P-M-φ surface** | column 2nd-order analysis (AISC moment magnifier, ACI direct analysis) | 1-2 sessions | Deferred from Section Designer Phase 2 |

### Tier III — deferred (will do when ready)

| Phase | Closes | Effort | Why deferred |
|---|---|---|---|
| **C.1-C.6 — Vendor V&V** | MIDAS, CSI, Abaqus benchmark validation | 12-20 sessions | User wants to defer; will pick up once more capability is in place to validate against |

### Tier IV — out of scope (explicitly chosen NOT to do)

- Aeroelastic flutter (wind-tunnel / full CFD required)
- Buffeting analysis (CFD coupling)
- Coupled fluid-structure interaction
- Topology optimization (different software class)
- Detailed CAD modeling (Revit / Tekla replacement)

---

## 4. Recommended path (my suggested ordering)

If I had to choose **the single highest-value next phase**:

### My recommendation: **Phase D.1 — Timber (mass timber CLT + NDS + EC5)**

**Why timber first:**

1. **Largest growing structural market** — mass timber construction is
   the fastest-growing building structure type in North America and
   Europe; commercial software still has uneven CLT support. Being
   "the one that does timber properly" is a real positioning advantage.

2. **Zero coverage today** — femsolver currently has NO timber
   constitutive, NO CLT panel section, NO timber connection design.
   It's a complete gap, not a partial one.

3. **Theme II makes it cheap** — adding a `CLTGeometry` and a
   `TimberMaterial` plugs into the unified `Section` architecture
   we just shipped. Most of the work is constitutive + design code,
   not architecture.

4. **Mass timber design code is mature** — NDS-2024 (US), EC5 (EU),
   IS 883 / Canadian O86 all have well-defined cross-laminated timber
   provisions. We can implement against published standards.

5. **Compatible with existing analysis pipelines** — once timber
   materials exist, all the linear/nonlinear/transient/seismic
   capabilities just work on timber structures.

**Estimated structure (8-12 sub-phases):**

| Sub-phase | Deliverable |
|---|---|
| D.1.1 | Timber material constitutive (orthotropic with fibre direction); NDS reference values for sawn lumber, glulam, CLT |
| D.1.2 | CLT panel section in unified `Section` framework (layered laminae with alternating fibre direction) |
| D.1.3 | Timber design checks: NDS Ch. 3-5 (tension, compression, flexure, shear, combined H) |
| D.1.4 | EC5 §6 (tension, compression, flexure, combined) |
| D.1.5 | CLT-specific design (Wood Handbook / APA): two-way bending, shear-flow, vibration |
| D.1.6 | Timber connections: dowels (NDS Ch. 11), bolts, screws, nailed (NDS / EC5) |
| D.1.7 | Diaphragm + shear-wall design for wood-frame buildings (SDPWS) |
| D.1.8 | Fire design for CLT (charring rate, residual section, EC5 §4.2 / NDS Ch. 16) |
| D.1.9 | Capstone: 3-story mass-timber building demo |

### Or alternative: **Phase F.1 — Steel Section Designer Phase 2**

If you'd rather mirror the RC Section Designer Phase 2 we just
finished, this gives the same capabilities for steel:

- Biaxial M-M-N for any steel polygon (not just AISC catalogue shapes)
- LTB for arbitrary cross-section (not just doubly-symmetric W shapes)
- Fiber-section nonlinear steel analysis (already partial via fiber sections; needs better integration)
- Plastic interaction (full Mp surfaces for biaxial loading)

This is more "deepen what we have" rather than "add new problem
class" — but it ensures steel has parity with concrete after our
recent Section Designer push.

Estimated: 6-8 sub-phases.

### Or alternative: **Phase E.1 — Tunnel staged excavation**

Niche but unique. Most commercial general-purpose FE doesn't do
tunnels well; tunnel-specific FE (PLAXIS, Midas GTS) doesn't do
buildings well. femsolver could be the one that does both.

Estimated: 6-8 sub-phases.

---

## 5. Three-question check for the user

Before committing, three strategic questions:

1. **Material breadth vs. analysis depth?** Timber (D.1) and CFS
   (D.2) add new MATERIAL classes; tunnel staging (E.1) adds a new
   ANALYSIS class. Which gap matters more for your positioning?

2. **Mirror the RC Section Designer success?** F.1 (Steel Section
   Designer Phase 2) is the natural parallel to what we just did
   for RC. It's "more of the same" — high quality, but less
   strategic differentiation.

3. **MIDAS V&V deferral — for how long?** The deferral is sensible
   if the next phases will GENERATE more capability to verify. But
   if D + E will take 30+ sessions, you may want a single
   "credibility check" V&V mid-way (e.g., after D.1 timber to verify
   we got the orthotropic constitutive right against a known
   benchmark).

---

## 6. What happens after Tier I + II are done

Once both Tier I (biggest "missing problem class" wins) and Tier II
(steel/composite/aluminum/glass section designers) are done:

| Layer | Status |
|---|---|
| Solver coverage | "femsolver handles every commonly-encountered structural problem class" |
| Material breadth | RC, PSC, structural steel, mass timber, CFS, masonry, aluminum, glass, FRP composites |
| Analysis types | static, dynamic, transient, seismic, wind, fire, foundation, geotechnical, tunnel, tank, wave, slope, fragility, P-58 |
| Design codes | ACI, AISC, AISC 341/358, ASCE 7, EC0-EC8, IS 456/800/875/1893/13920, NDS, AISI S100, TMS 402, ADM |
| Vendor V&V | Phase C ready to run with credibility |
| Tier 5 GUI | Now meaningful — covers all the cases users will throw at it |

That's the picture you described at the founding conversation:
**one software that handles ALL structural engineering problems**,
validated against vendor benchmarks. At that point, the GUI is the
last gate, not the first.

---

## 7. What I need from you

To unblock the next session, pick one of:

1. **"Start D.1 (Timber + NDS/EC5)"** — recommended highest-value
2. **"Start D.2 (CFS + AISI S100)"** — large market, niche but valuable
3. **"Start E.1 (Tunnel staged excavation)"** — unique positioning, smaller breadth gain
4. **"Start F.1 (Steel Section Designer Phase 2)"** — parallel to RC, less new ground
5. **"Start II.17 (Slender column P-M-φ surface)"** — quick 1-2 session add-on
6. **Something else** — e.g., "Phase G (composite section)" or "Phase H (3D continuum SSI)"

Or if you'd rather discuss the strategic ordering first before
committing, just say so and I'll wait.
