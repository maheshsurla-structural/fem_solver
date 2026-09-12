# Fiber Hinge (Nonlinear Fiber Beam‑Column) — Master Implementation Plan

**Status:** Living document — the single source of truth for the Fiber Hinge work stream.
**Owner rotation:** Multiple Claude sessions across multiple accounts. **The git repo is the
only shared state** (memory files do NOT cross accounts). Read this file first, do one work
item, update the [Status Tracker](#12-status-tracker), commit, and stop.

**Reference models (decoded in §2):**
- Midas Civil MCT — `…/Material Testing/Axial Load Test/Material Testing_Axial Load.mct`
- CSiBridge `$br` — `…/Material Testing/Axial Load Test/CSI/FH1_Material_Test Axial.$br`

Both model the **same physical specimen**: a Caltrans 84″ (7 ft) circular RC bridge column,
one cantilever frame element, with a **fiber plastic hinge**, validated under an axial‑load
protocol (P = 2400 kip preload) and then moment / cyclic lateral displacement.

---

## 0. How to use this document (session protocol)

This work will be done across many sessions on different Claude accounts. To stay consistent:

1. **Read this whole file before writing any code.** It freezes the architecture, public API
   names, sign conventions, units, and validation targets so independent sessions converge.
2. **Pick the lowest-numbered unchecked item** in the [Status Tracker](#12-status-tracker)
   whose dependencies are all checked. One phase ≈ one session.
3. **Code to the contracts in §5.** Do not rename classes/methods or change signatures that
   another phase depends on. If a contract must change, edit §5 in the *same* commit and note
   it in §13 (Change log).
4. **Every phase ships:** implementation + unit test(s) under `tests/` + one numbered example
   under `examples/` + a docstring in the literate house style (see §11).
5. **Update the Status Tracker** (check the box, add the commit hash + date) and append a line
   to §13. Commit with a message tagged `feat(fiber-hinge Pn): …`.
6. **Do not rebuild what §3 says already exists.** Reuse it.
7. If you discover the reference-model decode in §2 is wrong, fix §2 and say so in §13 — §2 is
   the spec the numbers are validated against.

Conventions (units, signs, naming, tests, examples) are in §11 and are **binding**.

---

## 1. Goal & scope

### 1.1 Goal
Give femsolver a first‑class **nonlinear fiber beam‑column / fiber‑hinge capability** that
reproduces the modeling workflow and results of Midas Civil's *Inelastic Fiber Hinge* and
CSI's *Fiber P‑M2‑M3 Hinge*, verified against both on the attached Caltrans column benchmark.

### 1.2 In scope
- Circular / Caltrans‑circle fiber section builder (confined core + unconfined cover + rebar ring).
- Mander confinement pre‑processor (geometry + hoops → confined `f'cc, ε_cc, ε_cu`; cover spalling).
- Steel model parity with the Caltrans/Park idealized curve used by both tools.
- A user‑facing **fiber‑hinge element/assignment** (finite‑length hinge in an otherwise‑elastic
  member) *and* the fully‑distributed fiber element (already present) — both map to real
  Midas/CSI options.
- 3‑D fiber beam‑column path for genuine **P‑M2‑M3**.
- Staged / sequential analysis (apply and hold axial P, then push/cycle laterally).
- Load protocols: monotonic ramp, stepped cyclic amplitudes, arbitrary time function.
- Recorders: per‑fiber σ‑ε, section N‑M / M‑φ, hinge force‑deformation, monitored DOF.
- Validation datasets + regression tests against Midas & CSI numbers.

### 1.3 Out of scope (for now — list here so sessions don't gold‑plate)
- Biaxial concrete constitutive coupling and nonlinear shear in the fiber (uniaxial fibers only;
  shear/torsion stay uncoupled — consistent with both reference tools' fiber hinges).
- Bond‑slip / anchorage‑slip springs, buckling of longitudinal bars, low‑cycle fatigue.
- Full MCT/`$br` round‑trip import of an entire bridge (we parse only what the benchmark needs).
- Dynamic (true time‑history with mass/damping); we do quasi‑static staged pushover/cyclic. The
  transient solver already exists and can be wired later.

---

## 2. Reference models decoded (the benchmark spec)

> Native units: **Midas = kip, in**; **CSI = kip, ft**. This plan states everything in
> **kip, inch** (the natural unit for the section). Conversions shown where they matter.
> All decoded values below were read directly from the two attached files.

### 2.1 Specimen (identical in both tools)
| Quantity | Value | Source |
|---|---|---|
| Shape | Solid circular column | Midas `SR`, CSI Caltrans Circle |
| Diameter `D` | **84 in** (7 ft) | Midas SECTION D=84; CSI Height=Width=7 ft |
| Clear cover (to hoop) | **2.0 in** (0.1667 ft) | CSI `CoverRing1=0.1667` |
| Confined core dia. | ≈ **79 in** | CSI `CnfDiam=6.583 ft` |
| Element length `L` | **49 in** (4.0833 ft) | Midas node 2 @ x=49; CSI joint 28 @ 4.0833 ft |
| Boundary | Base fixed (all 6), tip free/loaded | Midas CONSTRAINT 1111111; CSI joint 27 all‑fixed |
| Gross area `Ag` | ≈ 5542 in² | π·84²/4 |

### 2.2 Materials (expected / overstrength values — a "material test")
**Concrete — expected f'c = 5 ksi** (1.25 × nominal 4 ksi), Mander model:
| Param | Value | Notes |
|---|---|---|
| `f'c` (expected) | 5.0 ksi | Midas "5 ksi Expected"; CSI `Fc=720 ksf` = 5 ksi |
| `E_c` | 3605 ksi | Midas 3.605e3; CSI `E1=519119 ksf` = 3605 ksi |
| `ε_c0` (peak strain) | 0.002219 | Midas `Eco`; CSI `SFc=0.002219` |
| `ε_cu` core (cap) | 0.005 | CSI `SCap=0.005`; Midas confined `Ecu` |
| Unconfined `ε_cu` | 0.02219 (spalling ~0.005) | Midas unconfined line |
| Model / cyclic | Mander envelope, **Takeda** unload (CSI) | CSI `SSCurveOpt=Mander, SSHysType=Takeda` |

Midas provides the confined branch **pre‑computed** via its MANDER card (has `f'cc`, `ε_cc`,
confinement steel `#8 @ 6″`, `ρ_s`, etc.); CSI computes it internally from the Caltrans
confinement rebar. **Our builder must derive `f'cc, ε_cc, ε_cu` from geometry+hoops (Phase P2).**

**Reinforcing steel — A615 Gr60, expected:**
| Param | Value | Notes |
|---|---|---|
| `f_y` (expected) | 68 ksi | Midas 68; CSI `EffFy=9792 ksf` |
| `f_u` (expected) | 95 ksi | Midas 95; CSI `EffFu=13680 ksf` |
| `E_s` | 29000 ksi | both |
| `ε_sh` (hardening onset) | 0.0075 | Midas; CSI `SHard=0.0075` |
| `ε_su` (ultimate) | 0.09 (Midas) / 0.06 (CSI `SCap`) | ⚠ **discrepancy — see §10 O1** |
| Model | Caltrans/Park idealized (elastic–plastic–parabolic hardening), kinematic cyclic | Midas `PM`; CSI `Simple` + Kinematic |

### 2.3 Reinforcement
- **Longitudinal:** **56 × #14** bars (2.25 in² each) in one ring near the perimeter.
  CSI: 28 bundles × "2‑Bar Radial" = 56; Midas `NumLBar=56`.
- **Transverse:** **#8 circular hoop @ 6″** (0.5 ft) spacing. CSI `CnfType=Hoop`, Midas `#8, s=6`.

### 2.4 Fiber discretization (differs by tool — expose both, check convergence)
| Tool | Scheme | Counts |
|---|---|---|
| CSI | Cylindrical | 24 circumferential × 8 radial, `GridAngle=22.5°`, rebar not lumped |
| Midas | Cartesian grid + explicit rebar fibers | NY=15 × NZ=15 |

### 2.5 The fiber hinge
| Aspect | Midas | CSI | femsolver mapping |
|---|---|---|---|
| Type | Inelastic hinge, `FIBER`, **DIST**ributed | **Fiber P‑M2‑M3** | see §4 G4 / §5 |
| Definition | `FIBER-DIVISION` from section | `DefType="From Section"` | build fibers from the section |
| Location | on element (distributed) | `RelDist=0.5` (mid) | hinge at relative location |
| Hinge length | element `R=1`, NY/NZ integ. | `LengthType=Relative, SSRelLen=1` | relative hinge length |
| Save fiber resp | `bMONITOR=YES` | `SaveFibResp=Yes` | recorder (G8) |

**Interpretation:** CSI inserts a **finite‑length fiber hinge** at mid‑member inside an elastic
frame; Midas spreads fiber behavior over the element (distributed). Because the benchmark
element is short and fully governed by the fiber response, both reduce to essentially the same
answer — but we will support both idioms (§4 G4).

### 2.6 Load protocol & solution
| Stage | Midas | CSI |
|---|---|---|
| 1 — Axial | `P_RAMP` time fn ramps `PX_UNIT` (Fx=−1) | `FH1_P2400_PRELOAD` = `PX_TEST`×2400, **Full Load**, monitor U1 |
| 2 — Moment/lateral | `M_AFTER_P`, cyclic `Time History` fn on `MZ_UNIT` | `FH1_MONO_P2400` **Displ‑control** (monitor U2) + cyclic `FH1_TH_0.25…1.00 ±` |
| P magnitude | via scale | **2400 kip compression** (≈ 0.087 `f'c·Ag`) |
| Geometry | — | `GeoNonLin=None` (small‑disp) |
| Solver | time‑history (quasi‑static) | **Iterative Events + Newton‑Raphson**, tol 1e‑4, up to 40 it |

The "**Axial Load Test**" folder isolates **Stage 1**: validate the fiber section's axial force
vs. imposed axial strain/shortening (Σ σ·A) — pure material/section validation before moment is
added. Companion folders (moment, cyclic) reuse the same section.

### 2.7 femsolver equivalents of the benchmark (target model)
```
Model(ndm=2, ndf=3)                 # 2-D for axial + Mz; ndm=3 later for P-M2-M3
nodes 1@(0,0) fixed, 2@(49,0)
section = circular Caltrans fiber section (P2/P1)   # 5 ksi conc, 56#14, #8@6 hoop
element = ForceBeamColumn2DCorotational(1,(1,2),..., section=section)   # or FiberHinge (G4)
Stage 1: NonlinearStaticAnalysis(load_control) apply Fx=-2400 at node 2, hold
Stage 2: NonlinearStaticAnalysis(displacement_control on node2 U-lateral) monotonic + cyclic
```

---

## 3. What already EXISTS in femsolver — reuse, do NOT rebuild

Verified present (with file references). Sessions must build on these.

| Capability | Where | Notes |
|---|---|---|
| **Fiber section 2‑D** (`N, Mz`, P‑M tangent coupling, state, `clone`) | `sections/response/fiber.py` → `FiberSection2D`, `Fiber` | only `rectangular()` factory today |
| **Fiber section 3‑D** (`N, Mz, My, T`, uncoupled GJ) | same → `FiberSection3D` | `rectangular()` only |
| **Force‑based beam‑column 2‑D** (OpenSees `forceBeamColumn`, Neuenhofer‑Filippou) | `elements/beam_force.py` → `ForceBeamColumn2DCorotational` | consumes fiber section at Gauss‑Lobatto IPs |
| Displacement‑based beam 2‑D/3‑D + corotational | `elements/beam.py`, `beam_corot.py`, `beam_corot_3d.py` | `BeamColumn2D/3D`, `…Corotational` |
| **Concentrated hinge beam** (rotational spring + static condensation) | `elements/beam_hinged.py` → `HingedBeamColumn2D` | |
| **Zero‑length** per‑DOF uniaxial springs ("discrete plastic hinges") | `elements/zero_length.py` → `ZeroLengthElement` | |
| **Mander concrete** (Popovics + Chang‑Mander cyclic degr.) | `materials/uniaxial/concrete.py` → `ConcreteMander` | takes `fpc, eps_c0, Ec` **pre‑computed** |
| Kent‑Park, Parabola‑Rect (EC2), Trilinear, Tension‑stiffening | same file | |
| **Menegotto‑Pinto steel** (Giuffré evolving R) | `materials/uniaxial/menegotto_pinto.py` → `UniaxialMenegottoPinto` | cyclic |
| Bilinear / Isotropic / Takeda / Pivot / IMK / BRB uniaxials | `materials/uniaxial/*` | |
| **Nonlinear static** — load / **displacement** / **arc‑length** control, Newton, conv. tests | `analysis/nonlinear_static.py` → `NonlinearStaticAnalysis` | `track=(node,dof)` |
| Nonlinear transient, modal pushover | `analysis/nonlinear_transient.py`, `modal_pushover.py` | |
| Gauss‑Lobatto quadrature | `numerics/quadrature.py` → `gauss_lobatto_1d` | |
| Exact M‑φ, fibre P‑M slice, P‑M‑M surface (section designer) | `core.exact_mphi`, `section_pm_slice`; `examples/69_biaxial_pmm_surface.py` | design‑side, complementary |
| **Existing fiber examples** (reuse idiom) | `examples/09,11,16,19,21,44` | 44 = CSI fiber hysteresis; 11 = corot fiber column + axial |

**Public API idiom** (frozen — build the benchmark exactly this way):
```python
from femsolver import (Model, FiberSection2D, Fiber, ForceBeamColumn2DCorotational,
                       ConcreteMander, UniaxialMenegottoPinto, NonlinearStaticAnalysis)
m = Model(ndm=2, ndf=3)
m.add_node(1, 0.0, 0.0); m.add_node(2, 49.0, 0.0); m.fix(1, [1,1,1])
m.add_element(ForceBeamColumn2DCorotational(1, (1,2), mat, section=section))
m.add_nodal_load(2, [-2400.0, 0.0, 0.0])
res = NonlinearStaticAnalysis(m, num_steps=n, dlambda=1/n, track=(2,1), tol=1e-6).run()
# res["lambdas"], res["tracked"]
```

---

## 4. Gap analysis — what to BUILD

| ID | Gap | Why needed | Maps to phase |
|---|---|---|---|
| **G1** | Circular fiber builder (rings × wedges; core/cover split; rebar‑ring placement) | benchmark is circular; only `rectangular()` exists | P1 |
| **G2** | Mander **confinement pre‑processor** (geometry+hoops → `f'cc, ε_cc, ε_cu`; cover spalling) | both tools auto‑derive confined props; `ConcreteMander` needs them pre‑computed | P2 |
| **G3** | **Caltrans/Park steel** curve (elastic–plastic plateau–parabolic hardening to `f_u,ε_su`) | matches Midas `PM` / CSI `Simple`; MP is a superset but not the exact backbone | P3 |
| **G4** | **Fiber‑hinge element + assignment** (finite‑length fiber hinge at rel‑location in elastic member) | CSI "Fiber P‑M2‑M3 at RelDist"; Midas lumped/dist option | P5 |
| **G5** | **3‑D force‑based** beam‑column (`ForceBeamColumn3D`) | genuine P‑M2‑M3 (only 2‑D force‑based exists) | P7 |
| **G6** | **Staged / sequential** analysis (hold P from stage 1, continue into stage 2) | preload‑then‑push workflow; keep committed state across cases | P6 |
| **G7** | **Load protocols** (monotonic ramp, stepped cyclic ±0.25…1.0, time‑function driver) | reproduce both tools' protocols | P6 |
| **G8** | **Recorders** (per‑fiber σ‑ε, section N‑M / M‑φ, hinge F‑D, monitored DOF → CSV) | `SaveFibResp`; needed to compare to golden data | P4 |
| **G9** | **MCT / `$br` mini‑parser** for the benchmark (section, materials, hinge, protocol) | reproducible import; cross‑validation harness | P9 (optional) |
| **G10** | **Validation datasets + regression tests** vs Midas & CSI | prove parity; lock it in | P8, P10 |

---

## 5. Target architecture & public API (contracts — do not diverge)

New/extended modules. **Names below are binding**; if you must change one, update this section
and §13 in the same commit.

### 5.1 Section builders — extend `sections/response/fiber.py` (or a new `fiber_build.py`)
```python
# G1 — circular fiber discretization
FiberSection2D.circular(
    diameter, n_rings, n_wedges, material, *, centroid_y=0.0) -> FiberSection2D
FiberSection3D.circular(
    diameter, n_rings, n_wedges, material, *, GJ, ...) -> FiberSection3D

# G1+G2 — high-level RC circular column section (the benchmark one-liner)
def rc_circular_column_section(
    *, diameter, cover,                     # geometry (to hoop c/c or bar face — DOCUMENT which)
    core_concrete, cover_concrete,          # UniaxialMaterial (confined / unconfined)
    n_bars, bar_area, steel,                # longitudinal ring
    n_rings=8, n_wedges=24,                 # mesh (CSI defaults)
    threeD=False, GJ=None,
    lump_rebar=False,
) -> FiberSection2D | FiberSection3D
```
Sign/coordinate convention **must** match existing fiber code: `y` from centroid, positive +y;
`eps_f = eps_a - y*kappa_z (+ z*kappa_y)`; `N=Σσ·A`, `Mz=-Σy·σ·A`, `My=+Σz·σ·A`.

### 5.2 Mander confinement — new `materials/uniaxial/mander_confine.py`
```python
@dataclass
class ConfinedCircular:               # G2 result
    fcc: float; eps_cc: float; eps_cu: float; ke: float; fl: float

def mander_confined_circular(
    *, fco, eps_co, Ec,               # unconfined concrete
    D_core, hoop_area, hoop_spacing, fyh,   # confinement
    rho_long=0.0, hoop_type="spiral"|"hoop",
) -> ConfinedCircular
# Then: ConcreteMander(fpc=res.fcc, eps_c0=res.eps_cc, Ec=Ec, ...); ε_cu from res.eps_cu
```
Also a `mander_unconfined(...)` helper (spalling `ε_cu ≈ 0.005`). Formulas: Mander/Priestley
(1988) energy‑balance; cite in docstring. **Validate the intermediate `f'cc` against the Midas
MANDER card values in §2.2 before trusting the section.**

### 5.3 Caltrans/Park steel — add to `materials/uniaxial/` (e.g. `reinforcing_steel.py`)
```python
class ReinforcingSteelParkCaltrans(UniaxialMaterial):   # G3
    def __init__(self, *, E, fy, fu, eps_sh, eps_su, cyclic="kinematic"): ...
```
Backbone: linear to `fy`, yield plateau to `eps_sh`, parabolic hardening to `(eps_su, fu)`;
kinematic cyclic. (`UniaxialMenegottoPinto` remains available for smooth Bauschinger studies.)

### 5.4 Fiber hinge — new `elements/beam_fiber_hinge.py` (G4)
```python
class FiberHingeBeamColumn2D(Element):
    """Elastic member with a finite-length fiber hinge at a relative location.
    Mirrors CSI 'Fiber P-M2-M3' (RelDist, relative hinge length) and Midas lumped hinge.
    Implementation: 'beam with hinges' (Scott–Fenves modified Gauss–Radau) OR an
    elastic segment + short force-based fiber segment. Pick ONE and document in the
    docstring; keep the external 6-DOF interface identical to BeamColumn2D."""
    def __init__(self, tag, nodes, elastic_mat, section, *,
                 rel_loc=0.5, rel_hinge_len=..., n_ip=...): ...
```
> **Decision D1 (see §10):** distributed `ForceBeamColumn` already covers the physics; G4 adds
> the *hinge idiom*. Recommended: implement as **beam‑with‑hinges** reusing the force‑based
> state determination at the hinge integration points. Do P1–P3 first; both idioms share them.

### 5.5 Staged analysis & protocols — `analysis/staged.py` + `analysis/protocols.py` (G6/G7)
```python
# G6 — run case B continuing from case A's committed state (no unload)
class StagedAnalysis:
    def add_stage(self, name, analysis_factory) -> None
    def run(self) -> dict            # committed state persists across stages

# G7 — displacement protocols returning a target-history array
def monotonic(target, n_steps) -> np.ndarray
def stepped_cyclic(amplitudes=(0.25,0.5,0.75,1.0), cycles=1, pts_per_cycle=...) -> np.ndarray
def from_time_function(times, values, dt) -> np.ndarray
```
Confirm whether `NonlinearStaticAnalysis` already continues from a model's committed state (it
operates on live `Node`/element state). If yes, `StagedAnalysis` is a thin orchestrator.

### 5.6 Recorders — `results/recorders.py` (G8)
```python
class FiberRecorder:     # per-fiber (y,z,σ,ε) at chosen IP/section, per step -> CSV
class SectionRecorder:   # (N, Mz[, My], eps_a, kappa) per step -> CSV
class NodeRecorder:      # monitored DOF (disp, reaction) per step -> CSV
```

### 5.7 Importers — `io/midas_mct.py`, `io/csi_b.py` (G9, optional)
Parse **only** the benchmark subset: section geometry, materials, fiber division, hinge
assignment, load cases/protocol. Return a `Model` + a stage list. Keep it narrow.

---

## 6. Phased roadmap (each phase ≈ one session)

Ordered by dependency. Acceptance = tests green + example runs + numbers within tolerance (§7).

| Phase | Deliverable | Depends on | Acceptance |
|---|---|---|---|
| **P0** | This plan reviewed; confirm §2 decode & §10 decisions | — | Plan committed; O1/D1 answered or defaulted |
| **P1** | **G1** circular fiber builders (`.circular`, `rc_circular_column_section`) | P0 | `Ag, Iz, centroid` match analytic circle; mesh‑refinement test |
| **P2** | **G2** Mander confinement pre‑processor | P1 | `f'cc, ε_cc` match Midas MANDER card §2.2 within 2% |
| **P3** | **G3** Caltrans/Park steel material | P0 | backbone hits `(fy,ε_sh)`,(fu,ε_su); monotonic + cyclic tests |
| **P4** | **G8** recorders + **M‑φ** of the benchmark section (monotonic) | P1‑P3 | M‑φ vs Midas/CSI section M‑φ within tol |
| **P5** | **Axial‑load benchmark** (Stage 1): build model, apply P, N‑vs‑axial‑strain | P1‑P4 | **matches Midas & CSI axial test §7.1** |
| **P6** | **G6/G7** staged analysis + protocols; monotonic pushover (P then lateral) | P5 | pushover base‑shear/tip‑disp vs both tools §7.2 |
| **P7** | **G5** 3‑D force‑based element + `FiberSection3D.circular`; P‑M2‑M3 | P1‑P3 | biaxial pushover sanity; matches 2‑D in uniaxial limit |
| **P8** | **Cyclic** benchmark (stepped ±0.25…1.0) + energy dissipation | P5‑P7 | hysteresis loops & energy vs both tools §7.3 |
| **P9** | **G4** fiber‑hinge element idiom (CSI RelDist / Midas lumped) | P5 | reproduces P5/P6 within tol via the hinge idiom |
| **P10** | **G9** MCT/`$br` mini‑importer + end‑to‑end regression harness | P5‑P9 | import both files → run → auto‑compare to golden CSVs |

> Sessions may split a phase (e.g. P1a builders, P1b factory) — keep the tracker granular.

---

## 7. Validation strategy & golden numbers

**Store golden data as CSV** under `tests/data/fiber_hinge/` and reference from tests. The user
has already compared Midas vs CSI; those exports become the golden targets — **paste values in
§7.x as they arrive** (placeholders `TBD` until then).

Recommended tolerances (relative, unless noted): elastic stiffness ±2%; yield force/moment ±5%;
peak/ultimate ±10%; cyclic energy per loop ±15%. Justify any looser tolerance in the test.

### 7.1 Axial‑load test (Stage 1 — primary for this folder)
- Curve: **axial force N vs axial displacement** (and vs axial strain `N` vs `ε_a`).
- Checks: initial `EA = E_c·A_transformed`; onset of concrete nonlinearity near `ε_c0`; peak `N`;
  post‑peak softening; the P=2400 kip working point strain.
- Golden: Midas `axial` export = `TBD`; CSI `FH1_P2400_PRELOAD` = `TBD`.

### 7.2 Monotonic M‑φ and pushover
- Section M‑φ at N=2400 kip; column base‑shear vs tip‑displacement.
- Golden: `TBD`.

### 7.3 Cyclic
- Stepped ±0.25/0.5/0.75/1.0 in; loop shapes, peak force per amplitude, dissipated energy.
- Golden: `TBD`.

### 7.4 Cross‑checks that need no external data (do these regardless)
- Circle `Ag`, `Iz` vs closed form; mesh refinement monotone‑convergent.
- Mander `f'cc` vs Midas card (§2.2).
- 3‑D reduces to 2‑D under uniaxial bending.
- Force‑based one‑element result invariant to `n_ip` for elastic section.

---

## 8. Risks & mitigations
- **Confinement formula mismatch** → validate `f'cc` against the Midas card *before* the section
  (P2 gate).
- **Steel `ε_su` discrepancy (0.09 vs 0.06)** → §10 O1; run both, report sensitivity.
- **State‑determination non‑convergence at softening** (`beam_force.py` warns on singular
  flexibility) → hardening tangents, smaller steps, or switch to displacement control past peak.
- **Unit slips (kip‑in vs kip‑ft)** → all benchmark code in **kip, in**; assert `Ag≈5542 in²`.
- **Cross‑account drift** → this doc + Status Tracker are authoritative; no private assumptions.

---

## 9. File map (where new code goes)
```
src/femsolver/
  sections/response/fiber.py            # G1 .circular factories (extend)
  sections/response/fiber_build.py      # G1 rc_circular_column_section (or in fiber.py)
  materials/uniaxial/mander_confine.py  # G2
  materials/uniaxial/reinforcing_steel.py # G3
  elements/beam_fiber_hinge.py          # G4
  elements/beam_force_3d.py             # G5 ForceBeamColumn3D
  analysis/staged.py                    # G6
  analysis/protocols.py                 # G7
  results/recorders.py                  # G8
  io/midas_mct.py, io/csi_b.py          # G9 (optional)
tests/
  test_fiber_circular.py, test_mander_confine.py, test_reinforcing_steel.py,
  test_fiber_hinge_axial.py, test_fiber_hinge_pushover.py, test_fiber_hinge_cyclic.py,
  data/fiber_hinge/*.csv                # golden Midas/CSI exports
examples/
  8x_fiber_hinge_axial_test.py, 8x_fiber_hinge_pushover.py, 8x_fiber_hinge_cyclic.py
```
(Next free example number: check `ls examples/` — currently ~79; use the next integers.)

---

## 10. Open decisions (resolve in P0; default in **bold** if no answer)
- **O1 — Steel `ε_su`:** Midas 0.09 vs CSI 0.06. **Default: use 0.09 (Midas) for the primary
  run; add a CSI‑matched 0.06 variant in the cyclic test.** Confirm with user which governs.
- **D1 — Hinge idiom vs distributed element:** ✅ **RESOLVED (2026‑09‑12, user):** ship the
  distributed `ForceBeamColumn` path first (P5/P6); add the finite‑length fiber‑hinge element
  (P9) after.
- **O2 — Cover reference:** does `cover` mean to hoop c/c, hoop face, or bar centroid? **Default:
  clear cover to the *hoop outer face* (2.0 in), core dia = D − 2·(cover) ≈ matches CnfDiam 79 in.**
  Verify against CSI `CnfDiam=6.583 ft`.
- **O3 — Units for the shipped example:** **Default kip, in.** (Solver is consistent‑unit agnostic.)
- **O4 — Golden data source of record:** ✅ **RESOLVED (2026‑09‑12, user):** user will export
  **both** Midas + CSI (force‑displacement / M‑φ / axial‑load); embed as CSV golden targets and
  report our result against each independently. Until the exports land, use §7.4 cross‑checks.

---

## 11. Conventions (binding)
- **Units:** consistent‑unit system; benchmark code in **kip, inch**. No hidden conversions.
- **Signs:** compression‑negative stress in concrete materials (as in `ConcreteMander`); fiber
  kinematics `eps_f = eps_a − y·κ_z + z·κ_y`; `N=ΣσA`, `Mz=−Σy σA`, `My=+Σz σA` (match §3).
- **Naming:** classes `CamelCase`, functions `snake_case`; keep existing names in §5.
- **Docstrings:** literate house style — physics + formulation + OpenSees/Midas/CSI equivalence,
  as in `beam_force.py` / `fiber.py` / `concrete.py`. Explain *why*, not just *what*.
- **Tests:** `tests/test_*.py`, pytest; run with `PYTHONPATH=src`. Golden CSVs in
  `tests/data/fiber_hinge/`. (Note: 4 pre‑existing order‑8 quadrature failures on
  Py3.14/numpy 2.5.1 are unrelated — don't chase them.)
- **Examples:** numbered `examples/NN_name.py` with a `main()` and a `Run::` block; print a
  clear pass/interpretation summary like the existing ones.
- **Public API:** export new user‑facing classes from `femsolver/__init__.py` and add to
  `tests/data/public_api.txt` (there's a public‑API test — keep it in sync).

---

## 12. Status Tracker
Legend: ☐ todo ◐ in progress ☑ done. Update the row, add `commit` + `date` + your account tag.

| Phase | Item | State | Commit / date / by |
|---|---|---|---|
| P0 | Plan + decode reviewed; D1 & O4 resolved (distributed‑first; user exports both) | ☑ | 2026‑09‑12 — plan authored; O1/O2/O3 defaulted |
| P1 | G1 circular fiber builders | ☐ | |
| P2 | G2 Mander confinement pre‑processor | ☐ | |
| P3 | G3 Caltrans/Park steel | ☐ | |
| P4 | G8 recorders + benchmark M‑φ | ☐ | |
| P5 | Axial‑load benchmark (Stage 1) | ☐ | |
| P6 | G6/G7 staged + protocols; monotonic pushover | ☐ | |
| P7 | G5 3‑D force‑based + circular 3‑D; P‑M2‑M3 | ☐ | |
| P8 | Cyclic benchmark + energy | ☐ | |
| P9 | G4 fiber‑hinge element idiom | ☐ | |
| P10 | G9 importers + regression harness | ☐ | |

---

## 13. Change log (append one line per session; newest last)
- 2026‑09‑12 — Initial plan authored. Decoded both attached benchmark files (Caltrans 84″
  circular column, axial‑load fiber‑hinge test); completed gap analysis against the existing
  fiber/force‑beam/material/solver stack; froze architecture, API contracts, phased roadmap,
  and validation strategy. Golden numbers (§7) pending user's Midas↔CSI comparison export.
- 2026‑09‑12 — P0 closed. User resolved D1 (distributed `ForceBeamColumn` first, finite‑length
  fiber hinge in P9) and O4 (user will export both Midas + CSI as independent golden targets).
  Next unblocked item: **P1 — circular fiber builders (G1)**.
