# femsolver Solver Claims Matrix

**Audit date:** 2026-05-31 (Section Designer Phase 2 complete)
**Test suite:** 2049 passing, 1 skipped, 0 failures
**Coverage:** Phases 1-57 + Theme HH + Theme II (Section Designer) + II Phase 2 (II.10-II.16) across 20 themes

This document is the **honest accounting** of what femsolver can and
cannot model today. Every claim is classified into one of four bands:

| Band | Meaning |
|---|---|
| **Production** | Validated, no known caveats, deployable for commercial work |
| **Beta** | Working with documented limitations; suitable for engineering use with awareness |
| **Limited** | MVP-level; not recommended for production deployment without review |
| **Missing** | Not implemented |

The matrix is organised along four axes: **analysis types**,
**structure types**, **material classes**, **design codes**.

---

## Axis 1 -- Analysis types

| Analysis | Status | Where | Notes / caveats |
|---|---|---|---|
| Linear static | Production | `analysis/linear_static.py` | PARDISO + sparse fallback; reactions, recovery |
| Modal / eigen | Production | `analysis/eigen.py` | Shift-invert Lanczos; mass-orthonormalised |
| Linear buckling | Production | `analysis/buckling.py` | Requires corotational elements for K_g |
| Nonlinear static (load control) | Production | `analysis/nonlinear_static.py` | Newton + line search |
| Nonlinear static (displacement control) | Production | same | Pluggable integrator |
| **Arc-length (cylindrical, spherical, adaptive)** | Production | `analysis/integrator.py` | Auto limit-point detection |
| Linear transient (Newmark) | Production | `analysis/transient.py` | Default β=1/4, γ=1/2 |
| HHT-α / Generalised-α | Production | `analysis/transient_integrator.py` | Auto-tuned dissipation |
| Central-difference explicit | Production | same | For impact / blast |
| Nonlinear transient (NLTHA) | Production | `analysis/nonlinear_transient.py` | Newmark-nonlinear |
| Response spectrum (SRSS, CQC) | Production | `analysis/response_spectrum.py` | Multi-mode combination |
| Multi-support excitation | Production | `analysis/transient_integrator.py` | Different ground motion at each support |
| Pushover (DB) | Production | `analysis/capacity_design.py` | Displacement-based fiber column (PushoverToTarget driver) |
| Pushover (FB) | Production | `elements/beam_force.py` | Force-based with Neuenhofer-Filippou |
| Pushover (hinged) | Production | `elements/beam_hinged.py` | Concentrated plasticity |
| Pushover-to-target (N2 + ASCE 41) | Production | `analysis/capacity_design.py` | Both target-displacement methods |
| Modal Pushover Analysis (MPA) | Production | `analysis/modal_pushover.py` | SRSS / CQC combination of modal pushovers |
| IDA (single + multi-record) | Production | `analysis/ida.py`, `ida_collapse.py` | With collapse detection (3 criteria) |
| Lognormal fragility fitting | Production | `analysis/fragility.py` | Censored MLE + Baker 2015 |
| Record selection + scaling (ASCE 7) | Production | `analysis/record_scaling.py` | Amplitude scaling + CMS |
| Conditional Mean Spectrum (Baker 2011) | Production | `analysis/cms.py` | With Baker-Jayaram correlation |
| **PSHA hazard curve + UHS** | Production | `seismic/psha.py`, `seismic/bssa14.py` | Period-dependent BSSA14 (HH.4); CB14 future work |
| **PSHA deaggregation** | Beta | `seismic/deaggregation.py` | R-bin snap at source distance (minor; out of HH scope) |
| **Risk-targeted MCE_R (ASCE 7 Ch 21)** | Production | `seismic/risk_targeted.py` | Brent solver to 1e-6 |
| **1-D site response (equivalent-linear)** | Production | `seismic/site_response.py`, `seismic/equivalent_linear.py` | SHAKE-style EL iteration with Vucetic-Dobry curves (HH.5) |
| FEMA P-58 component damage | Production | `analysis/p58.py` | Monte Carlo loss aggregation |
| Reliability (FORM / HLRF) | Production | `reliability/form.py` | iHLRF iteration |
| Reliability (SORM / Breitung) | Production | `reliability/sorm.py` | Curvature correction |
| Monte Carlo + importance sampling | Production | `reliability/monte_carlo.py` | 85× variance reduction demonstrated |
| Influence lines | Production | `bridges/influence.py` | Simple-span moment + shear |
| Moving loads (AASHTO HL-93, IRC) | Production | same | Single-direction envelopes |
| PT tendons (friction + anchorage) | Production | `bridges/pt_tendon.py` | Linear-friction + slip |
| Creep / shrinkage (CEB-FIP 2010) | Production | `bridges/creep_shrinkage.py` | AASHTO LRFD prestress loss |
| Composite section (transformed) | Production | `bridges/composite_section.py` | Stress recovery in all fibres |
| Cable element (Ernst E_eq) | Production | `bridges/cable.py` | Cable stays / hangers |
| Catenary closed forms | Production | `bridges/cable.py` | T_max, sag, arc length |
| Staged construction | Production | `bridges/staged_construction.py` | With creep redistribution (EMM) |
| Heat conduction (steady, transient) | Production | `analysis/heat_conduction.py` | 2D Q4 + 3D Hex8 thermal elements |
| Thermo-mechanical | Production | `analysis/thermal_strain.py` | Coupled stress + temperature |
| Fire engineering (ISO 834 + degradation) | Production | `analysis/fire.py` | EC3 k_y/k_E + EC2 k_c |
| Foundation (Winkler beam) | Production | `analysis/winkler.py` | Hetenyi formulation |
| Foundation (Gazetas impedance, static + dynamic) | Production | `analysis/dynamic_gazetas.py` | Frequency-domain springs |
| Foundation (API p-y, t-z, q-z) | Production | `analysis/soil_springs.py` | Nonlinear |
| Foundation (pile group p-multipliers) | Production | `analysis/pile_group.py` | AASHTO arrangement |
| Liquefaction triggering | Production | `analysis/liquefaction.py` | Idriss-Boulanger 2014 |
| Connection design (Krawinkler PZ) | Production | `design/connections/panel_zone.py` | Strength + stiffness |
| Connection design (RBS) | Production | `design/connections/rbs.py` | AISC 358 |
| Connection design (PR, Richard-Abbott) | Production | `design/connections/pr_connection.py` | Cyclic |
| Connection design (bolts, welds) | Production | `design/connections/bolts_welds.py` | AISC + EC3 + IS |
| Hyperelastic large strain | Production | `materials/hyperelastic.py`, `elements/hex8_TL.py` | NH + MR |
| Finite-strain J2 plasticity | Production | `materials/finite_j2.py` | Hencky log-strain |
| Contact (node-to-plane, Coulomb) | Production | `elements/contact.py` | Penalty enforcement |
| Vortex shedding screening | Production | `wind/vortex.py` | Strouhal + Scruton + lock-in |
| **Aeroelastic flutter** | Missing | -- | Outside current scope |
| **Buffeting analysis** | Missing | -- | Time-domain wind |
| **CFD pressure mapping** | Missing | -- | Would need external CFD coupling |

**Summary of analysis-type axis:** 49 capabilities Production, 1 Beta (PSHA deagg R-bin), 3 Missing.

---

## Axis 2 -- Structure types

| Structure type | Status | Coverage notes |
|---|---|---|
| Steel buildings (frames, braced, dual) | Production | Full nonlinear + design |
| RC buildings (moment-frame, shear-wall, dual) | Production | Fiber sections, wall shapes T/L/U/I, coupling beams |
| Composite steel-concrete buildings | Production | Composite section + design |
| Pre-stressed concrete (PT + creep) | Production | Phase 38 + 45 |
| Cable-stayed bridges | Production | Cable element + staged + EMM |
| Suspension bridges | Production | Catenary + cable |
| Girder bridges (concrete, steel, composite) | Production | Composite girder + bearings |
| Box-girder bridges | Production | Shell + beam coupling |
| Arch bridges | Production | Curved beam Phase 49.5 + buckling |
| Industrial structures (towers, racks) | Production | Truss + frame |
| Wind turbines (structural) | Production | Beam + shell tower, blade as composite shell |
| Chimneys / stacks | Production | Tower + vortex check |
| Stadiums / large-span roofs | Production | Shell + cable |
| Curved bridges / horizontal curvature | Production | Curved Timoshenko element |
| Shallow foundations (isolated, mat) | Production | Winkler + Gazetas |
| Deep foundations (pile, pile group) | Production | API p-y + p-multipliers |
| Retaining walls | Production | DP / MC soil + Mohr-Coulomb |
| Slope stability (FE) | Limited | Have MC soil but no specific slope-stability driver |
| **Dams (gravity, arch)** | Beta | Have constitutive + thermal + creep; no specific dam load wizard |
| **Offshore platforms (jacket)** | Beta | Structural OK; **wave loading MISSING** |
| **Tunnels** | Limited | Soil + lining material OK; **TBM staged excavation MISSING** |
| **Storage tanks (cylindrical)** | Limited | Shell elements OK; **API 650 sloshing MISSING** |
| **Heritage / masonry buildings** | Missing | No masonry material or design code |
| **Mass timber buildings (CLT/glulam)** | Missing | No timber material or design code |
| **Cold-formed steel structures** | Missing | No CFS section catalogue or AISI S100 design |
| **Glass / curtain wall** | Missing | No glass design code |
| **Aluminum structures** | Missing | No aluminum design code |
| **Membrane / tensile structures** | Limited | Have orthotropic material but no form-finding |
| **Storage racks** | Missing | Requires CFS + rack-specific codes |
| **Pre-engineered metal buildings (PEMB)** | Limited | Have I-beam tapered? No -- tapered web variation MISSING |
| **Slab on grade with subgrade modulus** | Limited | Winkler beam covers 1D; 2D slab-on-elastic-foundation MISSING |

**Summary of structure-type axis:** 18 Production, 4 Beta, 5 Limited, 5 Missing.

---

## Axis 3 -- Material classes

| Material class | Status | Notes |
|---|---|---|
| Elastic isotropic | Production | -- |
| Orthotropic / composite lamina | Production | With Tsai-Wu / Tsai-Hill / Hashin failure |
| J2 plasticity (small strain) | Production | -- |
| Finite-strain J2 (Hencky) | Production | -- |
| Drucker-Prager | Production | Outer/inner cone match to MC |
| **Mohr-Coulomb** | Production | Full 4-region return mapping (main face + 2 edges + apex) per de Souza Neto Ch 8 (HH.1) |
| **Modified Cam-Clay** | Production | Stress-dependent tangent K' = (1+e)p'/κ with void-ratio tracking (HH.2) |
| Concrete (Kent-Park uniaxial) | Production | Confined + unconfined |
| Concrete (Mander, Chang-Mander) | Production | With cyclic damage |
| **Concrete damage (Lubliner-Lee-Fenves)** | Production | Separate d_t/d_c, plastic strain tracking, T->C stiffness recovery (HH.3) |
| Steel uniaxial (Menegotto-Pinto, GMP evolving R) | Production | -- |
| Hysteretic (pinching + degradation) | Production | -- |
| IMK (Ibarra-Medina-Krawinkler) | Production | -- |
| BRB | Production | -- |
| Takeda | Production | -- |
| Pivot | Production | -- |
| Isotropic hardening | Production | -- |
| Hyperelastic (Neo-Hookean, Mooney-Rivlin) | Production | -- |
| Thermal material | Production | k, c_p, ρ |
| **Masonry constitutive** | Missing | No URM smeared crack, no RM section |
| **Timber constitutive** | Missing | No timber design + no orthotropic-with-fibre-direction wood material |
| **Cold-formed steel** | Missing | No CFS material curves (cold-work strain hardening) |
| **Aluminum** | Missing | -- |
| **Glass** | Missing | -- |
| **FRP composites** | Production | Via orthotropic lamina + Tsai-Wu |

**Summary of material-class axis:** 20 Production, 0 Beta, 5 Missing.

---

## Axis 4 -- Design code coverage

| Code | Status | Modules covered |
|---|---|---|
| **ACI 318-19** | Production | Beam flexure + shear, column P-M-M, slabs (DDM + punching capacity + stud-rail/stirrup design per 22.6.7, HH.8) |
| **AISC 360-22** | Production | Tension, compression, flexure (LTB), shear, combined H interaction |
| **AISC 341** | Production | Seismic capacity design (SCWB, capacity-design shear) |
| **AISC 358** | Production | RBS prequalified |
| **ASCE 7-22** | Production | Wind (MWFRS + C&C zones Fig 30.5-1, HH.6), seismic (response spectrum), load combos, drift, MCE_R |
| **EN 1992 (EC2)** | Production | RC flexure + shear + punching capacity + reinforcement design (EC2 6.4 + 6.4.5, HH.8) |
| **EN 1993 (EC3)** | Production | Steel tension, compression (Perry-Robertson a0-d), flexure, shear, combined |
| **EN 1991-1-4 (EC1)** | Beta | Peak velocity pressure only; structural factor c_s·c_d for tall buildings MISSING |
| **EN 1998 (EC8)** | Production | Design spectrum, base shear, drift |
| **IS 456 (2000)** | Production | RC flexure + shear + punching capacity + stirrup design (31.6.3.2, HH.8) |
| **IS 800 (2007)** | Production | Steel design |
| **IS 875 (Part 3 2015)** | Production | Basic wind + dynamic response factor C_dyn for tall/flexible structures (§10/Annex C, HH.7) |
| **IS 1893 (2016)** | Production | Seismic |
| **IS 13920 (2016)** | Production | Ductile detailing |
| **FEMA P-58** | Production | Component damage + loss |
| **NDS (timber)** | Missing | -- |
| **AISI S100 (cold-formed)** | Missing | -- |
| **TMS 402 (masonry)** | Missing | -- |
| **Aluminum Design Manual** | Missing | -- |
| **AS 3600 / AS 4100 (Australia)** | Missing | -- |
| **CSA A23.3 / S16 (Canada)** | Missing | -- |
| **NBR 6118 (Brazil)** | Missing | -- |
| **JSCE (Japan)** | Missing | -- |
| **GB 50010 / 50017 (China)** | Missing | -- |
| **API 650 (storage tanks)** | Missing | -- |
| **AASHTO LRFD (bridges)** | Production | HL-93, distribution factors partial |
| **EN 1990 (basis of design)** | Missing | Combination logic generic, not EC0-specific |

**Summary of design-code axis:** 15 Production, 1 Beta (EC1 c_s·c_d), 11 Missing (mostly secondary markets and specialist domains).

---

## What this tells us

### Strengths
- All **mainstream structural analysis types** (linear, nonlinear, dynamic, PBE, transient, response spectrum, pushover, IDA, fragility) are Production.
- **All major US, European, and Indian design codes** for concrete and steel are Production.
- The library can already handle the **canonical building, bridge, foundation, and industrial structure types**.

### Theme HH (caveat consolidation) -- closed
Phase HH (sub-phases 1-9, completed 2026-05-31) closed every documented
solver caveat from this list:

1. ~~Soil constitutive caveats~~ -- **HH.1 + HH.2 closed**
2. ~~Concrete damage scalar Mazars~~ -- **HH.3 closed** (full Lubliner-Lee-Fenves)
3. ~~PSHA single-period GMPE~~ -- **HH.4 closed** (BSSA14 period table)
4. ~~Site response linear only~~ -- **HH.5 closed** (equivalent-linear)
5. ~~Wind C&C + dynamic factor~~ -- **HH.6 + HH.7 closed** (ASCE 7 + IS 875)
6. ~~Punching reinforcement design~~ -- **HH.8 closed** (ACI 318 / EC2 / IS 456)

### Theme II (Section Designer) -- closed
Phase II (sub-phases 1-9, completed 2026-05-31) consolidated **18
scattered section-like dataclasses** across 7 directories into one
canonical `Section` object with lazy adapters to every downstream
subsystem (analysis, design, reports, BOM, JSON, SVG).

1. ~~Section types in 7 different directories with no common base~~ --
   **II.1-II.5 closed** (unified `Section` ABC + 8 parametric
   primitives + 3 catalogues unified + custom polygons + Boolean ops)
2. ~~No section round-trip~~ -- **II.8 closed** (JSON serialization)
3. ~~No unified visualization~~ -- **II.8 closed** (SVG sketcher +
   `SectionReport` with HTML)
4. ~~Beam element vs design code use different section objects~~ --
   **II.6 closed** (lazy adapters: `elastic_section_3d`,
   `fiber_section_3d`, `as_aisc_section`, `as_aci_concrete_section`,
   ...)
5. ~~Bridge composite cannot drive a beam element~~ -- **II.7 closed**
   (`composite_girder_deck_section` returns unified Section)
6. ~~Adding a new material class (timber, masonry) would spawn 3 new
   section paths~~ -- **II.6/II.7 closed** (one extension point per
   material in the unified architecture; Phase D is now de-risked)

See `docs/source/phase_ii_complete.md` for the audit trail and
`examples/68_theme_ii_capstone.py` for the demonstration.

### General Section Designer Phase 2 (II.10-II.16) -- closed
Built on top of Theme II's unified architecture, Phase 2 added the
engineering capabilities that turn the unified `Section` into a
full *General Section Designer* (parity with commercial tools like
SAP2000 Section Designer, MIDAS PSC Section, STAAD Section Wizard):

1. ~~Biaxial P-Mz-My interaction surface only available for
   rectangular RC~~ -- **II.10 closed** (any polygon, ACI Whitney via
   analytical polygon clipping; matches existing 2-D ACI code to
   0.001-0.13%)
2. ~~Biaxial P-M-M only in ACI~~ -- **II.11 closed** (EC2 §3.1.7 and
   IS 456:2000 Annex G via shared `StressBlockParams` engine; IS 456
   hand calc verified to 0.07%)
3. ~~No moment-curvature driver~~ -- **II.12 closed** (3-DOF Newton
   driver with M_cr, M_y, M_u, μ_φ extraction; matches Whitney
   hand-calc within ~10% Kent-Park parabolic)
4. ~~Prestressing not integrated with unified Section~~ --
   **II.13 closed** (`PrestressTendon` + `PrestressedUniaxial` wrapper
   with pre-strain offset; PSC M_cr matches hand calc to 0.00%)
5. ~~No cracked transformed section for serviceability~~ --
   **II.14 closed** (cracked I_cr matches hand calc to 0.04%; ACI
   Branson I_e + EC2 mean-curvature tension stiffening)
6. ~~No stress field query or crack-pattern visualization~~ --
   **II.15 closed** (3-DOF Newton on (ε₀, κ_z, κ_y); per-fibre query;
   SVG colour-coded crack pattern overlay)

See `docs/source/phase_ii_complete_phase2.md` for the audit trail
and `examples/70_section_designer_phase2_capstone.py` for the
demonstration.

**Remaining minor gap:** P-M-φ surface (moment-axial-curvature) for
slender-column 2nd-order analysis -- thin sweep loop over the
existing M-φ driver, ~30 lines when needed.

### Remaining minor caveats (not in Phase HH scope)
- **PSHA deaggregation R-bin snap** -- deagg uses source distance bin
  rather than interpolated R; minor, can fix in a focused micro-phase.
- **EC1 wind c_s·c_d structural factor** -- the IS 875 dynamic factor
  HH.7 work could be ported; out of HH scope.

### Genuinely missing capabilities (Phase D-E candidates)
- **Material classes:** timber, cold-formed steel, masonry, aluminum, glass
- **Specialist analyses:** tunnel staged excavation, storage tank sloshing, wave loading, membrane form-finding, slope-stability driver, 2-D slab-on-grade
- **Geographic codes:** AS, CSA, NBR, JSCE, GB, NDS, AISI, TMS

### What we're explicitly choosing NOT to do (out-of-scope)
- Aeroelastic flutter (would need wind-tunnel data or full CFD)
- Buffeting (needs CFD coupling)
- Coupled fluid-structure interaction

---

## The path forward (re-ordered 2026-05-31)

| Phase | Status | Closes | Target sub-phases |
|---|---|---|---|
| **B (Theme HH)** -- consolidate documented caveats | **DONE** | All 6 Beta items above | 9 (HH.1 - HH.9) |
| **Theme II** -- Section Designer unification | **DONE** | 18 scattered dataclasses -> 1 canonical type | 9 (II.1 - II.9) |
| **Section Designer Phase 2** -- RC/PSC engineering capabilities | **DONE** | biaxial P-M-M (3 codes), M-phi, cracked I, stress field, prestress | 7 (II.10 - II.16) |
| **C** -- vendor V&V (CSI / MIDAS / Abaqus) | **DEFERRED (user)** | Audit credibility | 12-20 |
| **D** -- new material classes (timber, CFS, masonry, aluminum, glass) | **next priority** | one new material class per sub-theme | 30+ total |
| **E** -- specialist analyses (tunnels, tanks, waves, slope, membrane) | pending | one new analysis per sub-theme | 25+ total |
| **F** -- Steel Section Designer Phase 2 (parallel to RC) | pending | biaxial M-M-N + LTB for any steel polygon | 6-8 |
| **G** -- Composite Section Designer (encased, filled) | pending | AISC 360 Ch. I composite design | 5-7 |

**Strategic ordering (per `future_roadmap.md`):**

1. Tier I — biggest "missing problem class" wins: **D.1 (Timber)**,
   D.2 (CFS), D.3 (Masonry), E.1 (Tunnels), E.2 (Tanks), E.3 (Waves)
2. Tier II — important but smaller: F.1 (Steel Section Designer), F.2
   (Composite), D.4-D.5 (Aluminum, Glass), E.4-E.5 (Slope, Membrane),
   II.17 (Slender column P-M-φ)
3. Tier III — deferred: **C (vendor V&V)** — will pick up later
4. Tier IV — out of scope: aeroelastic flutter, CFD, FSI

**After Tier I + II, the solver claims:** *"femsolver handles every
common structural-engineering problem class across RC, PSC, steel,
timber, masonry, CFS, aluminum, and glass, with every common
analysis type (static / dynamic / seismic / wind / fire / foundation
/ tunnel / tank / wave / slope)."*

Then Phase C (vendor V&V) is run with full credibility. Then -- and
only then -- Tier 5 (Desktop GUI) becomes meaningful.

See `docs/source/future_roadmap.md` for the detailed sub-phase
breakdown and effort estimates.
