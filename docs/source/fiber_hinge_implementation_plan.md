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
confinement rebar. **Our `mander_confined_circular` (P2, ✅) derives these from geometry+hoops.**

**Decoded Midas confined targets** (MCT line 304 — core dia 79″, #8 hoop @ 6″, `f_yh` = 68 ksi,
`f'c` = 5 ksi), used as the P2 golden gate:
| Quantity | Midas card | Ours | Δ |
|---|---|---|---|
| `f'cc` | 6.35571 ksi | 6.371 | 0.24% |
| `ε_cc` | 0.00522731 | 0.00526 | 0.64% |
| `k_e` | 0.962451 | 0.96245 | exact |
| `f_l` | 0.218155 ksi | 0.21816 | exact |
| `ρ_s` | 0.00666667 | 0.006667 | exact |
(Δ on `f'cc/ε_cc` is a textbook‑vs‑Midas `k_e` convention: ours includes the standard
`/(1−ρ_cc)` Acc term. `ε_cu` ≈ 0.013 experimental; Midas card ≈ 0.0123.)

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
| **Park reinforcing steel** (`E, f_y, f_su, eps_sh, eps_su`, monotonic) | `materials/uniaxial/reinforcing.py` → `UniaxialReinforcingSteel` | **This is P3's model** — already built (monotonic); cyclic via Menegotto‑Pinto |
| **Section → fiber compiler** (polygon grid + rebar) | `sections/section.py` → `Section.fiber_section_2d/3d`, `_discretize_polygon_to_fibers` | the canonical "compile a defined section to a `FiberSection`" — unify P1 into this (§15 U2) |
| **Exact section integrator** (M‑φ, P‑M slice, width bands) | ⚠ `section_gui_core.py` (root GUI file) → `exact_mphi`, `section_pm_slice`, `_width_bands` | **trapped in the GUI layer** — must be lifted into the engine (§15 U1) |
| **Design M‑φ + biaxial P‑M‑M** (AASHTO/EC2/IS456) | `design/concrete/moment_curvature.py`, `design/concrete/biaxial.py` | a 3rd M‑φ path — consolidate behind one API (§15 U3) |
| **Section Designer (GUI)** — draw/define, section‑level M‑φ / P‑M‑M | `section_gui_core.py` (Streamlit core), `desktop/section_designer.py` | already ~80% on the femsolver engine; RC core/cover + composite fiber assembly live in the GUI file (lift to engine, §15 U2) |

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

### 5.1 Section builders — ✅ SHIPPED in P1 (`sections/response/fiber.py` + `fiber_build.py`)
Final as-built API (in `femsolver.sections.response.fiber` / `.fiber_build`, re-exported at
`femsolver.*`). Signatures below are the **binding contract**:
```python
# G1 — low-level annular meshing (public helper; angle phi=0 -> +z, phi=pi/2 -> +y)
circular_sector_fibers(r_in, r_out, n_rings, n_wedges, material,
    *, centroid_y=0.0, centroid_z=0.0, theta_start=0.0, theta_end=2*pi) -> list[Fiber]

# G1 — circular fiber sections (classmethods)
FiberSection2D.circular(diameter, n_rings, n_wedges, material, *, centroid_y=0.0)
FiberSection3D.circular(diameter, n_rings, n_wedges, material, *, GJ,
    centroid_y=0.0, centroid_z=0.0)

# G1 — high-level RC circular column (the benchmark one-liner)
def rc_circular_column_section(
    *, diameter, cover,                       # cover = to core boundary; core dia = D - 2*cover (O2)
    core_concrete, cover_concrete,            # UniaxialMaterial (confined / unconfined)
    n_bars, bar_area, steel,                  # longitudinal ring
    bar_circle_diameter=None,                 # default = core diameter
    n_core_rings=8, n_cover_rings=2, n_wedges=24,   # mesh (CSI-like)
    subtract_rebar_from_core=True,            # keep total area = gross exactly
    three_d=False, GJ=None,
    centroid_y=0.0, centroid_z=0.0,
) -> FiberSection2D | FiberSection3D
```
> **Contract changes vs the original draft (recorded per §0 rule):** split `n_rings` into
> `n_core_rings`/`n_cover_rings` (a 2-in cover on an 84-in column is thinner than one full-radius
> ring, so core and cover are meshed as separate regions); added `bar_circle_diameter`,
> `subtract_rebar_from_core`; renamed `threeD`→`three_d`; dropped `lump_rebar` (bars are always
> discrete for now). `circular_sector_fibers` is exposed as a public helper.

Sign/coordinate convention matches existing fiber code: `y` from centroid, positive +y;
`eps_f = eps_a - y*kappa_z (+ z*kappa_y)`; `N=Σσ·A`, `Mz=-Σy·σ·A`, `My=+Σz·σ·A`.
Verified: area exact to 1e-10; `Iz` error 8.6%→0.09% over meshes 2×8→24×72 (monotone);
3-D `Iz==Iy`, `Iyz≈0`; RC total area == gross exactly; transformed `EA` matches by hand.

### 5.2 Mander confinement — ⚠ **core calc already exists** (lifted to `sections/analysis.py` in U1)
> `femsolver.sections.analysis.mander_confinement(m: dict)` computes confined `fcc, eps_cc,
> eps_cu, ke, fl` for circular **and** rectangular cores (energy‑balance or experimental
> `eps_cu`), and `concrete_uniaxial_from(..., conc_model="Mander")` already wires it into a
> confined `ConcreteMander` law. **P2 remaining = (a) verify `fcc/eps_cc/eps_cu` against the Midas
> MANDER card (§2.2) within tol; (b) add a clean engine wrapper for the fiber‑hinge stream
> (below), reusing the SAME `mander_confinement` core (U4 — one calc for tool + hinge).** Do NOT
> write a second confinement calculator.

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

### 5.3 Caltrans/Park steel — ✅ **already exists** (`materials/uniaxial/reinforcing.py`)
```python
class UniaxialReinforcingSteel(UniaxialMaterial):   # G3 backbone — SHIPPED
    def __init__(self, E, f_y, f_su, eps_sh, eps_su): ...   # Park hardening, MONOTONIC
```
The monotonic Park backbone (linear → `f_y` → plateau to `eps_sh` → parabolic hardening to
`(eps_su, f_su)`) is done. **P3 ✅ SHIPPED (2026‑09‑12):** cyclic added as
`ReinforcingSteelKinematic` (`materials/uniaxial/reinforcing.py`) — a kinematic‑hardening return
map over the *same* Park backbone (chosen over Menegotto‑Pinto so the backbone shape stays exactly
the Caltrans/Park one both tools use). Monotonic reproduces the backbone to 1e‑13; reversals unload
at `E` and show the kinematic Bauschinger shift.

### 5.4 Fiber hinge — ✅ SHIPPED in P9 (`elements/beam_fiber_hinge.py`, G4)
> **As‑built:** `FiberHingeBeamColumn2D(tag, nodes, material, *, section, lp, lp_j=None)` —
> beam‑with‑hinges. Rather than the Scott‑Fenves modified Gauss‑Radau quadrature, it uses the
> equivalent, unambiguous **flexibility decomposition**: exact elastic member flexibility plus a
> localized plastic correction at each end hinge,
> `F = F_el + Σ_h lp_h·b(x_h)ᵀ(f_fiber,h − f_el)b(x_h)`, `v = F_el·q + Σ_h lp_h·b(x_h)ᵀ(e_fiber,h −
> f_el·s_h)`. Hinges at the member ends; reuses `ForceBeamColumn2DCorotational._section_strain_for_force`
> and the corotational wrapping. `f_el` is probed at a small **compressive** strain (compression‑only
> concrete reports zero tangent at exactly ε=0). Elastic response is exact for any `lp`.
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

### 5.5 Staged analysis & protocols — ✅ SHIPPED in P6 (`analysis/staged.py` + `analysis/protocols.py`, G6/G7)
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
> **As‑built note (P6):** `NonlinearStaticAnalysis` did **not** continue for free — `run()` calls
> `reset_results()` (zeroes node disp) and its residual scales a single pattern (`λ·F_ref`). So P6
> added two small, backward‑compatible primitives: `StaticIntegrator.set_constant_force` (residual →
> `F_const + λ·F_ref − f_int`) and `NonlinearStaticAnalysis(keep_state=…, const_force=…)`.
> `StagedAnalysis` is then the orchestrator: keep state between stages, fold each converged stage's
> applied load into `F_const`, clear the pattern for the next stage.

### 5.6 Recorders — ✅ SHIPPED in P4 (`results/recorders.py`, G8)
```python
class FiberRecorder:     # per-fiber (y,z,σ,ε) at chosen fibers, per step -> CSV
class SectionRecorder:   # (N, Mz[, My], eps_a, kappa) per step -> CSV
class NodeRecorder:      # monitored DOF (disp, reaction) per step -> CSV
```
Exported from `femsolver.results`. Plain step data-loggers: call `record(...)`
per committed step, `to_csv(path)` at the end. `SectionRecorder`/`FiberRecorder`
take the section's generalized strain `e` and re-read it (non-committing);
`NodeRecorder` snapshots a node's `disp`/`reaction`. P4 also added the
**fibre-consistent** M-φ driver `fiber_section_moment_curvature`
(`sections/response/fiber_mphi.py`) that these log.

### 5.7 Importers — `io/midas_mct.py`, `io/csi_b.py` (G9, optional)
Parse **only** the benchmark subset: section geometry, materials, fiber division, hinge
assignment, load cases/protocol. Return a `Model` + a stage list. Keep it narrow.

---

## 6. Phased roadmap (each phase ≈ one session)

Ordered by dependency. Acceptance = tests green + example runs + numbers within tolerance (§7).

| Phase | Deliverable | Depends on | Acceptance |
|---|---|---|---|
| **P0** | This plan reviewed; confirm §2 decode & §10 decisions | — | Plan committed; O1/D1 answered or defaulted |
| **P1** | **G1** circular fiber builders (`.circular`, `rc_circular_column_section`) — ✅ shipped; **reconcile into the one compiler** in §15 U2 (route circular meshing through `Section.fiber_section_2d`) | P0 | `Ag, Iz, centroid` match analytic circle; mesh‑refinement test |
| **P2** | **G2** Mander confinement pre‑processor — ✅ **done**: core `mander_confinement` (lifted U1) + typed `mander_confined_circular`/`ConfinedCircular` wrapper (`sections/analysis.py`); verified vs Midas card | P1 | ✅ `f'cc` 0.24%, `ε_cc` 0.64%, `ke`/`fl` exact vs Midas §2.2; `test_mander_confinement.py` (8) |
| **P3** | **G3** Caltrans/Park steel — ⚠ **mostly done**: `UniaxialReinforcingSteel` already provides the monotonic Park backbone. P3 = verify vs §2.2 + add cyclic (reuse `UniaxialMenegottoPinto`, or a kinematic-hysteresis wrapper on the Park backbone) | P0 | backbone hits `(fy,ε_sh)`,(fu,ε_su); monotonic ✓ (exists) + cyclic test |
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
- **Our result (P5, 2026‑09‑12, no external data yet — §7.4 cross‑checks):** the
  two‑node force‑based fiber column holds the 2400 kip preload with base reaction
  = 2400.0 kip; tip shortening 0.00509 in (strain 1.04e‑4, i.e. `< ε_c0`, elastic
  working point); transformed `EA` = 2.308e7 kip vs `Ec·Ac + Es·As` = 2.318e7
  (0.41%). Section axial capacity (imposed strain, `N = Σσ·A`): peak 41,291 kip at
  ε ≈ 0.0047 (near the confined `ε_cc` = 0.00526), softening to 32,772 kip at
  ε = 0.02 as the cover spalls and the confined core degrades. Paste Midas/CSI
  numbers here to lock the regression.

### 7.2 Monotonic M‑φ and pushover
- Section M‑φ at N=2400 kip; column base‑shear vs tip‑displacement.
- Golden: `TBD`.
- **Our result (P6, 2026‑09‑12, no external data yet):** staged pushover (hold
  P=2400 kip, displacement‑control lateral tip). Axial reaction held at 2400.0
  kip throughout. Base shear rises with a clear yield knee — 3370 kip @ 0.02 in,
  5649 @ 0.04, 7447 @ 0.09, 8039 @ 0.15 in — and `V·L` at the peak (32,825
  kip‑ft) matches the P4 section M‑φ peak (~32,900 kip‑ft) to 0.2%, confirming
  the element pushover and the section M‑φ are the same response. Paste
  Midas/CSI base‑shear/tip‑disp here to lock the regression.

### 7.3 Cyclic
- Stepped ±0.25/0.5/0.75/1.0 in; loop shapes, peak force per amplitude, dissipated energy.
- Golden: `TBD`.
- **Our result (P8, 2026‑09‑12, no external data yet):** hold P=2400 kip, cycle the tip through
  ±(0.25,0.5,0.75,1.0)×0.12 in (peaks 0.03…0.12 in), 4 displacement‑based fiber elements,
  `ReinforcingSteelKinematic` bars. Axial held at 2400.0 kip; symmetric hysteresis with peak base
  shear ≈ 7,968 kip (matches the P6 monotonic ~8,000 kip). Peak +V/−V and dissipated energy per
  loop: 0.030 in → ±4,58x, 8 kip‑in; 0.060 → ±7,1xx, 126; 0.090 → ±7,8xx, 555; 0.120 → ±7,9xx,
  1,093; total ≈ 1,781 kip‑in. Paste Midas/CSI loop force + per‑loop energy here to lock the
  regression. (Force‑based element flexibility is singular at cyclic reversals — plan §8 — so the
  cyclic run uses a displacement‑based mesh; it converges to the force‑based capacity at ~4 elements.)

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
  **✅ Convergence‑robustness pass (2026, engine §8):** the force‑based / fiber‑hinge inner
  state‑determination leaves an irreducible residual *noise floor* in the assembled unbalance, so an
  absolute `NormUnbalance` tolerance set below it made the outer Newton chatter forever while `||du||`
  had already collapsed to machine zero. Fixed two scale‑fragile absolute tolerances (both fixes
  strictly *additive* — they only accept convergence the old tests would miss): (1) `NormUnbalance`
  now also converges on **displacement stagnation** (`||du|| ≤ stag_du·||du_ref||` with a residual
  guard `||R|| ≤ stag_res·||R_ref||` so a stuck/singular iteration is never mistaken for
  convergence); (2) the elements' inner NF loop gained a **relative** deformation tolerance
  (`state_det_rel_tol`) beside the absolute `state_det_tol`. Effect: the benchmark force‑based
  pushover extends from ~0.15 in to ~0.36 in tip drift, and the fiber‑hinge element to ~0.60 in,
  before the *remaining* limit — the force‑based element's genuinely **singular flexibility** at deep
  softening (a true formulation limit; use a smaller step or the displacement‑based / fiber‑hinge
  path there). `test_convergence_robustness.py` (5).
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
| P1 | G1 circular fiber builders | ☑ | 2026‑09‑12 — `fiber.circular`/`circular_sector_fibers`, `fiber_build.rc_circular_column_section`; 12 tests, example 80; area exact, Iz→0.09% |
| P2 | G2 Mander confinement pre‑processor | ☑ | done (row was stale) — `mander_confinement` (engine) + typed `mander_confined_circular`/`ConfinedCircular` in `sections/analysis.py`; verified vs Midas card (fcc 0.24%, ke/fl exact); `test_mander_confinement.py` present |
| P3 | G3 Caltrans/Park steel | ☑ | 2026‑09‑12 — verified the monotonic Park backbone (`UniaxialReinforcingSteel`) hits §2.2 (eps_sh,f_y)=(0.0075,68) & (eps_su,f_su)=(0.09,95); added cyclic `ReinforcingSteelKinematic` (kinematic hardening over the same backbone; monotonic reproduces it to 1e‑13, elastic unload = E, Bauschinger shift). `test_uniaxial_materials.py` +7; example 81; full suite 2470 pass |
| P4 | G8 recorders + benchmark M‑φ | ☑ | 2026‑09‑12 — added `results/recorders.py` (`SectionRecorder`/`FiberRecorder`/`NodeRecorder`, CSV) + fibre‑consistent `fiber_section_moment_curvature` (`sections/response/fiber_mphi.py`); benchmark section M‑φ at P=2400 kip (N held to 1e‑11, M→32.9k kip‑ft). `test_fiber_mphi_recorders.py` (8), example 82. Golden Midas/CSI M‑φ (§7.2) pending user export; §7.4 cross‑checks pass. Full suite 2478 pass |
| P5 | Axial‑load benchmark (Stage 1) | ☑ | 2026‑09‑12 — two‑node `ForceBeamColumn2DCorotational` + Caltrans fiber section; load‑control preload holds 2400 kip (base reaction exact), EA fiber↔hand 0.9959, working‑point strain 1.04e‑4 (<ε_c0); section axial capacity (imposed strain, N=Σσ·A) peaks 41,291 kip @ ε≈0.0047 then softens to 32,772 @ 0.02. `test_fiber_hinge_axial.py` (3), example 83. Golden Midas/CSI axial (§7.1) pending export |
| P6 | G6/G7 staged + protocols; monotonic pushover | ☑ | 2026‑09‑12 — `analysis/staged.py` (`StagedAnalysis`, continuation + constant‑load hold via new `StaticIntegrator.set_constant_force` + `NonlinearStaticAnalysis(keep_state, const_force)`) + `analysis/protocols.py` (`monotonic`/`stepped_cyclic`/`from_time_function`). Benchmark staged pushover holds P=2400 kip exactly through the lateral push; base‑shear·L peak 32,825 kip‑ft matches the P4 M‑φ peak (0.2%). Guarded ConcreteMander softening‑tail overflow. `test_fiber_hinge_pushover.py` (6), example 84. Full suite 2487 pass |
| P7 | G5 3‑D force‑based + circular 3‑D; P‑M2‑M3 | ☑ | 2026‑09‑12 — `elements/beam_force_3d.py` `ForceBeamColumn3D` (small‑disp, 6‑DOF basic system, NF state determination). Elastic K == displacement‑based to 2e‑16; n_ip‑invariant (≥3); reduces to 2‑D under uniaxial bending to 1e‑7; biaxial 45° push gives Mz=−My, resultant = uniaxial capacity. `FiberSection3D.circular` from P1. `test_force_beam_3d.py` (4), example 85. Full suite 2490 pass |
| P8 | Cyclic benchmark + energy | ☑ | 2026‑09‑12 — staged cyclic (hold P=2400, `stepped_cyclic` protocol via a new `DisplacementControl` per‑step increment schedule); `ReinforcingSteelKinematic` steel. Axial held; symmetric hysteresis, peak shear 7968 kip (≈ P6 monotonic 8000 w/ 4 disp‑based elements); loop energy grows 8→126→555→1093 kip‑in. Force‑based element is cyclically singular at reversals (§8) → disp‑based mesh used. `test_fiber_hinge_cyclic.py` (2), example 86. Full suite 2493 pass |
| P9 | G4 fiber‑hinge element idiom | ☑ | 2026‑09‑12 — `elements/beam_fiber_hinge.py` `FiberHingeBeamColumn2D` (beam‑with‑hinges: exact elastic interior + fiber hinge length `lp` at each end, `F = F_el + Σ lp·bᵀ(f_fiber−f_el)b`; reuses force‑based section inversion). Elastic K == distributed to 8e‑16 for any lp; axial reproduces distributed to 0.27% (P5); pushover yields, peak V brackets distributed (lp 4/8/16 → 7138/6567/5967 vs 6851) (P6). `test_fiber_hinge_element.py` (6), example 87 |
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
- 2026‑09‑12 — **P1 shipped (G1).** Added `circular_sector_fibers`, `FiberSection2D/3D.circular`,
  and `rc_circular_column_section` (+ exports, `public_api.txt`). Tests `test_fiber_circular.py`
  (12) + example `80_fiber_circular_section.py`. Contract adjusted (see §5.1 note). Area exact,
  `Iz` converges to 0.09%, transformed `EA` verified. Next: **P2 — Mander confinement (G2)**.
- 2026‑09‑12 — Added **§14 GUI & Productization roadmap** and **§15 Unified Section Analysis
  architecture** (per user). Audited the desktop GUI (linear‑only today) and the Section
  Designer (`section_gui_core.py`); found P3's steel already exists (`UniaxialReinforcingSteel`),
  a canonical section→fiber compiler already exists (`Section.fiber_section_2d/3d`), and the
  exact integrator is trapped in the GUI file — updated §3, §6 (P1/P3) accordingly.
- 2026‑09‑12 — **U1 shipped.** Lifted the section‑analysis core (`exact_mphi`, `section_pm_slice`,
  `_width_bands`, the `concrete/steel_uniaxial_from` factories, and `mander_confinement`) from the
  GUI file `section_gui_core.py` into the engine `femsolver/sections/analysis.py`; GUI re‑exports
  them (desktop + Streamlit unchanged). Verified before/after byte‑identical + shim identity;
  `test_section_analysis_exact.py` (7); full suite 2454 pass (only the 4 pre‑existing quadrature
  failures). This also front‑loads P2's Mander calc (now in the engine). Next: **P2 — verify
  confinement vs Midas card + fiber‑hinge wrapper**.
- 2026‑09‑12 — **P2 shipped (G2).** Verified `mander_confinement` vs the Midas card (§2.2):
  `f'cc` 0.24%, `ε_cc` 0.64%, `k_e`/`f_l`/`ρ_s` exact. Added typed `mander_confined_circular` +
  `ConfinedCircular` (`.to_material`) in `sections/analysis.py`, unit‑agnostic, reusing the same
  calc (U4). `test_mander_confinement.py` (8); example 80 upgraded to the real confined core
  (placeholder removed). Next: **P4** (recorders + section M‑φ on the unified core) or **U3**
  (one M‑φ/P‑M‑M API), then **P5** (axial benchmark).
- 2026‑09‑12 — **U3 shipped.** Added the unified section‑analysis API to
  `femsolver/sections/analysis.py`: backend‑selector tokens `C_EXACT/C_FIBRE/C_NOMINAL/C_DESIGN`
  (+ `MPHI_BACKENDS`/`PM_BACKENDS`), result types `MomentCurvatureResult`/`PMInteractionResult`,
  and dispatchers `moment_curvature_analysis(case, P, backend=…)` (wraps `exact_mphi` / `mphi_data`)
  and `pm_interaction(case, backend=…)` (wraps `section_pm_slice` for the fibre envelope and the
  code stress‑block `pmm_slice` for nominal/design). Lifted `pmm_slice` out of the GUI
  `section_gui_core.py` into the engine (verbatim; GUI now re‑exports it — same U1 shim pattern), so
  all four M‑φ / P‑M capacity models live in one engine core the Section Designer tool, the tests,
  and the fiber‑hinge stream share. `test_section_analysis_unified.py` (10); full suite 2463 pass
  (only the 4 pre‑existing order‑8 quadrature failures). Next: **P4** (recorders + benchmark M‑φ on
  the unified core), then **P5** (axial benchmark).
- 2026‑09‑12 — **P3 shipped (G3).** Verified the existing monotonic Park backbone
  (`UniaxialReinforcingSteel`) reproduces the §2.2 Caltrans A615 Gr60 landmarks exactly —
  (eps_sh, f_y) = (0.0075, 68 ksi) and (eps_su, f_su) = (0.09, 95 ksi). Added the **cyclic**
  `ReinforcingSteelKinematic` (in `materials/uniaxial/reinforcing.py`): the same Park backbone
  wrapped in a kinematic‑hardening return map — a monotonic push reproduces the backbone to 1e‑13,
  reversals unload elastically (slope E) and re‑yield early in compression (kinematic / Bauschinger
  shift), matching Midas "Park PM" / CSI "Simple" + Kinematic. Because ∫H dp = α(p+dλ)−α(p) exactly,
  no numerical integration of the hardening modulus is needed; it reduces to `UniaxialBilinear` for
  a constant H. O1 (eps_su 0.09 Midas vs 0.06 CSI) covered by a test on both variants; both build
  and hit (eps_su, 95). Exported from `femsolver.materials.uniaxial` (same placement as the
  monotonic class; top‑level public surface unchanged). `test_uniaxial_materials.py` +7 tests;
  example `81_reinforcing_steel_cyclic.py`; full suite 2470 pass (only the 4 pre‑existing quadrature
  failures). Next: **P4** (recorders + benchmark section M‑φ on the unified core), then **P5**.
- 2026‑09‑12 — **P4 shipped (G8 + benchmark M‑φ).** Added `femsolver/results/recorders.py` with
  `SectionRecorder` (N, Mz[, My], eps_a, kappa per step), `FiberRecorder` (per‑fiber y,z,eps,sigma,
  long format) and `NodeRecorder` (monitored disp/reaction DOFs) — plain step data‑loggers →CSV via
  stdlib `csv`, exported from `femsolver.results` (results names are not on the top‑level surface, so
  `public_api.txt` unchanged for them). Added the **fibre‑consistent** M‑φ driver
  `fiber_section_moment_curvature` (`sections/response/fiber_mphi.py`, exported top‑level +
  `public_api.txt`): drives a stateful `FiberSection2D` through prescribed curvatures at constant
  axial force, Newton‑solving eps_a on the section axial residual (tangent `EA = ks[0,0]`) and
  committing each step — so it integrates the section's own fibers/laws (confined Mander core, Park
  steel), the same response the fiber hinge will integrate (§15 principle). Verified elastic
  M=E·Iz·κ exactly; benchmark Caltrans section at P=2400 kip holds N to 1e‑11 and reaches
  M≈32.9k kip‑ft, M(κ) monotonic to peak; kinematic‑steel path‑dependence confirmed. Example
  `82_fiber_section_mphi.py` runs the benchmark M‑φ and writes both recorders to CSV.
  `test_fiber_mphi_recorders.py` (8); full suite 2478 pass (only the 4 pre‑existing quadrature
  failures). Golden Midas/CSI section M‑φ to be pasted into §7.2 on export. Next: **P5**
  (axial‑load benchmark, Stage 1) — or **P6/P7** per the tracker.
- 2026‑09‑12 — **P5 shipped (axial‑load benchmark, Stage 1).** Built the benchmark model exactly
  per §2.7: two nodes (base fixed, tip axial‑only), one `ForceBeamColumn2DCorotational` carrying the
  Caltrans fiber section (P1‑P2 + Park steel P3). Load‑control to P=2400 kip holds the preload (base
  reaction 2400.0 kip), tip shortens 0.00509 in (strain 1.04e‑4 `< ε_c0`), transformed `EA` matches
  `Ec·Ac+Es·As` to 0.41%. The force‑based element flexibility goes singular at full crushing (its
  documented limit), so the axial *capacity* curve is taken at the section level (imposed strain,
  `N=Σσ·A`, the pure material validation §7.1 describes): peak 41,291 kip near `ε_cc`, softening to
  32,772 kip at ε=0.02 (cover spalls, confined core degrades); the preload strain carries 2400 kip
  on the bare section (element↔section consistency). Logged with `SectionRecorder` (G8 reuse).
  `test_fiber_hinge_axial.py` (3); example `83_fiber_hinge_axial_test.py`; results recorded in §7.1.
  Full suite 2481 pass (only the 4 pre‑existing quadrature failures). Next: **P6** (staged analysis
  + protocols; monotonic pushover) or **P7** (3‑D force‑based + circular 3‑D; P‑M2‑M3).
- 2026‑09‑12 — **P6 shipped (G6 + G7 + monotonic pushover).** `analysis/protocols.py` (G7):
  `monotonic`, `stepped_cyclic` (±0.25…1.0‑style growing cycles), `from_time_function` — pure
  target‑history generators. `analysis/staged.py` (G6): `StagedAnalysis` runs sequential stages on
  one model, continuing from committed state and holding earlier stages' loads constant. This needed
  two small, backward‑compatible core primitives: `StaticIntegrator.set_constant_force` (residual
  becomes `F_const + λ·F_ref − f_int`) and `NonlinearStaticAnalysis(keep_state=…, const_force=…)`
  (skip `reset_results`, pass the baseline). Verified the benchmark staged pushover: stage 1 applies
  P=2400 kip (load control), stage 2 holds it (axial reaction stays 2400.0 kip) and pushes the tip
  laterally (displacement control) — base shear yields, and `V·L` peak (32,825 kip‑ft) matches the
  P4 section M‑φ peak to 0.2% (element pushover ≡ section M‑φ). Also guarded a `ConcreteMander`
  softening‑tail `OverflowError` reachable only by a diverging force‑based state‑determination probe
  (returns the finite asymptotic σ→0; normal results unchanged). All new top‑level exports
  (`StagedAnalysis`, `monotonic`, `stepped_cyclic`, `from_time_function`) added to `public_api.txt`.
  `test_fiber_hinge_pushover.py` (6); example `84_fiber_hinge_pushover.py`; results in §7.2. Full
  suite 2487 pass (only the 4 pre‑existing quadrature failures). Next: **P7** (3‑D force‑based +
  circular 3‑D; P‑M2‑M3) or **P8** (cyclic benchmark + energy).
- 2026‑09‑12 — **P7 shipped (G5).** Added `ForceBeamColumn3D` (`elements/beam_force_3d.py`): a
  small‑displacement force‑based 3‑D beam‑column (subclasses `BeamColumn3D`), giving genuine
  **P‑M2‑M3** with one element per member. 6‑DOF basic system
  `v=[u, θz1, θz2, θy1, θy2, φ]` / `q=[N, Mz1, Mz2, My1, My2, T]`; constant basic‑to‑local matrix
  `a` (`v=a·u_local`, `f_local=aᵀq`), 4×6 force interpolation `b(x)` (bending linear, N/T constant),
  and the Neuenhofer‑Filippou state‑determination loop (same structure as the 2‑D element, 4 section
  resultants). Small‑disp chosen to match the benchmark's `GeoNonLin=None`. Verified: elastic K
  equals the displacement‑based closed form to 2e‑16 and is n_ip‑invariant (≥3 IPs); under uniaxial
  bending it reproduces the 2‑D force‑based response to ~1e‑7; a 45° biaxial push develops
  `Mz = −My` with resultant equal to the uniaxial capacity (symmetric circular section).
  `FiberSection3D.circular` was already shipped in P1. `test_force_beam_3d.py` (4); example
  `85_fiber_hinge_pmm_3d.py`; `ForceBeamColumn3D` exported top‑level + `public_api.txt`. Full suite
  2490 pass (only the 4 pre‑existing quadrature failures). Next: **P8** (cyclic benchmark + energy)
  or **P9** (finite‑length fiber‑hinge element idiom).
- 2026‑09‑12 — **P8 shipped (cyclic benchmark + energy).** Extended `DisplacementControl` to accept
  a **per‑step increment schedule** (scalar `du_step` still works; a sequence traces an arbitrary
  reversed‑cyclic path in one analysis — feed `np.diff(stepped_cyclic(...))`). Staged cyclic run:
  hold P=2400 kip (StagedAnalysis), then cycle the tip through ±(0.25,0.5,0.75,1.0)×0.12 in with the
  `ReinforcingSteelKinematic` steel (P3). Axial held at 2400.0 kip; symmetric hysteresis, peak base
  shear ≈ 7,968 kip (matches the P6 monotonic ~8,000), loop energy grows 8→126→555→1,093 kip‑in
  (total ~1,781). **Element choice:** the force‑based element's flexibility goes singular at cyclic
  reversals (documented §8), so P8 uses displacement‑based fiber elements — a 4‑element mesh is
  cyclically robust and converges to the same capacity as the one‑element force‑based monotonic push
  (mesh study: 1 el 10,862 kip → 2 el 8,940 → 4 el 8,001). `test_fiber_hinge_cyclic.py` (2); example
  `86_fiber_hinge_cyclic.py`; results in §7.3. Full suite 2493 pass (only the 4 pre‑existing
  quadrature failures). Next: **P9** (finite‑length fiber‑hinge element idiom) or **P10** (importers
  + regression harness).
- 2026‑09‑12 — **P9 shipped (G4).** Added `FiberHingeBeamColumn2D` (`elements/beam_fiber_hinge.py`):
  the finite‑length fiber‑hinge idiom (CSI "Fiber P‑M2‑M3" over a relative hinge length / Midas
  lumped inelastic hinge) — an elastic member with a fiber plastic hinge of length `lp` at each end.
  Per D1, implemented as beam‑with‑hinges reusing the force‑based section state determination, via a
  clean flexibility decomposition (exact elastic interior + localized plastic correction:
  `F = F_el + Σ lp·bᵀ(f_fiber−f_el)b`) rather than a from‑memory Gauss‑Radau reconstruction — so the
  elastic response is exact for any `lp` and plasticity localizes over `lp`. Found + fixed a real
  gotcha: `f_el` must be probed at a small **compressive** strain because `ConcreteMander` reports a
  zero tangent at exactly ε=0 (a zero‑strain probe dropped all concrete stiffness, making the member
  ~5× too flexible). Verified: elastic K equals the distributed element to 8e‑16 for lp∈{4,8,15};
  axial reproduces the distributed element to 0.27% (P5); the pushover yields and its peak base shear
  brackets the distributed value across lp (4/8/16 in → 7138/6567/5967 kip vs 6851) (P6).
  `FiberHingeBeamColumn2D` exported top‑level + `public_api.txt`. `test_fiber_hinge_element.py` (6);
  example `87_fiber_hinge_element.py`. Next: **P10** (MCT/`$br` mini‑importer + regression harness,
  needs the golden exports).
- 2026‑09‑12 — **GUI‑1 shipped (§14).** New session (justforjme account) resumed after the other
  account committed P1–P9 + U1/U3. Started the GUI roadmap: added the desktop **inelastic‑material
  editor with a live engine‑backed σ‑ε preview**. `desktop/materials.py` bridges a `project.Material`
  to the femsolver uniaxial library (`uniaxial_law`, `stress_strain_curve`) — one source, no
  re‑implemented constitutive maths (§15). `desktop/material_editor.py` = `MaterialDialog`
  (kind selector + per‑kind param fields + matplotlib σ‑ε canvas) and `MaterialManagerDialog`
  (list/add/edit/delete, delete‑in‑use guard). `Material.params` added; "Materials…" wired into the
  Edit menu. `test_desktop_materials.py` (11, first headless‑offscreen desktop test). Also corrected
  the stale **P2**/**U4** tracker rows (their code shipped earlier). Next GUI: **GUI‑2** (fiber‑mesh
  panel + preview, needs U2) or **GUI‑5** (threaded solver + progress dock).
- 2026‑09‑12 — **GUI‑6 fiber contour + step slider shipped.** Engine: `BeamColumn2DCorotational`
  stores per‑IP `_e_sections` (converged section deformations; additive). `run_pushover(capture_fibers=True)`
  snapshots the monitored base section's per‑fiber (y,z,σ,ε) each step (from a section clone, so live
  state is untouched). `PushoverDialog` now has a tabbed right pane — pushover curve | **fiber‑stress
  contour** — with a **step slider** and a **stress/strain toggle**. The contour reads exactly like
  CSI's SaveFibResp: stress mode shows the rebar yield pattern (bottom bars +500 MPa tension, top bars
  compression), strain mode shows the plane‑section gradient with the extreme concrete crushing (69
  fibers past ε_cu, 8 bars yielded at the demo drift). Tests: `test_pushover_capture_fibers`,
  `test_dialog_fiber_contour`. Remaining GUI‑6: deformed‑shape step animation (3‑D view) + hinge‑state
  coloring. Next: **GUI‑3** (hinge UI) / **GUI‑4** (case manager), then GUI‑6 anim/hinge‑state, GUI‑7.
- 2026‑09‑12 — **GUI‑5 UI shipped (nonlinear‑run subsystem complete).** Engine enabler: added an
  optional `step_callback` to `NonlinearStaticAnalysis` (called after each committed step; returning
  `False` cancels — backward‑compatible, default None). `run_pushover` gained `on_step`/`should_cancel`.
  `desktop/pushover_dialog.py`: `PushoverWorker` (QThread — streams per‑step progress, cooperative
  cancel) + `PushoverDialog` (control node/DOF/target/steps/axial inputs, progress bar, convergence
  log, Cancel, live base‑shear vs displacement curve). "Nonlinear pushover…" wired into the Analysis
  menu (guarded: needs a GSD/fiber section). Headless‑verified: worker streams + cancels, dialog
  builds, main‑window wiring. `test_desktop_nonlinear.py` (7), `test_desktop_pushover_ui.py` (4). The
  app can now run a fiber pushover without freezing and watch the curve build. Next: **GUI‑6** (step
  slider / fiber‑stress contour / hinge state) or **GUI‑3/4** (hinge + case‑manager UIs).
- 2026‑09‑12 — **GUI‑5 background shipped (nonlinear‑run subsystem, compute half).** `desktop/nonlinear.py`:
  `fiber_section_from_spec` compiles a GSD `Spec` → inelastic `FiberSection2D` (circular via the U2
  `polar_cells` mesh; concrete + rebar from the shared `section_gui_core` factories); `build_nonlinear_model`
  turns GSD‑section members into displacement‑based fiber `BeamColumn2DCorotational` (robust for pushover —
  the force‑based element goes singular near softening, §8/P8); `run_pushover` runs displacement control with
  an optional constant‑held axial preload (`StagedAnalysis`). Verified: pushover of a D=0.6 m / L=3 m fiber
  column yields (early K 1.2e7 → late 1.2e6) and axial preload raises capacity (339→358 kN — P‑M signature).
  `test_desktop_nonlinear.py` (5). Next: the GUI‑5 UI (threaded run + progress/convergence dock + curve).
- 2026‑09‑12 — **U2 done (polar‑mesh unification).** Extracted the annular‑sector polar mesh into
  engine helpers `polar_divisions` + `polar_cells` (`sections/response/fiber.py`); `circular_sector_fibers`
  now builds on them, and the Section Designer GUI (`section_gui_core.section_fibers` /
  `section_fiber_mesh`), which carried a duplicate copy of the formula, now imports the same helpers —
  the annular‑sector maths lives in exactly one place (§15 "no forked section engine"). Provably
  identical: `polar_cells(0,42,8,24)` area = πR² exact, `polar_divisions(1400)`=(15,94) matches the GUI
  heuristic, GUI circular section area = πR². 101 affected tests pass; helpers exported from
  `sections.response` (`public_api` unchanged). Unblocks **GUI‑2** (fiber‑mesh preview). Next: **GUI‑2**
  or **GUI‑5**.
- 2026‑09‑12 — **GUI‑6 complete (deformed‑shape animation + hinge‑state coloring).** Finished the last
  GUI‑6 items. Engine: `run_pushover(capture_shape=True)` snapshots, per step, the whole model's nodal
  deformation (`{node: (dx, dy)}`) plus a per‑fiber‑member **hinge‑state** value = peak |fiber strain|
  over the base section (`_peak_abs_strain`, from `_e_sections` — no live‑state perturbation); returns
  `shape_frames` + `damage_frames`. Both capture flags coexist and stay step‑aligned. `PushoverDialog`
  gained a third right‑pane tab, **Deformed shape**: light‑gray undeformed reference + the deformed
  members colored by peak fiber strain (YlOrRd, "peak fiber strain" colorbar) at an adjustable
  displacement scale, node markers on top. The step slider is now **shared** across the fiber‑stress and
  deformed‑shape tabs, with a new **▶ Play/Pause** button (a `QTimer` auto‑advances and loops the steps)
  — the genuine "step animation". Verified headless (offscreen) render: D=0.6 m/L=3 m column at 0.05 m
  drift shows the base member dark‑red at ε≈5.6e‑3 (hinge formed). Tests: `test_pushover_capture_shape`,
  `test_pushover_capture_shape_and_fibers_together`, `test_pushover_no_capture_omits_frames`,
  `test_dialog_deformed_shape`, `test_dialog_animation_advances_and_loops` (18 desktop NL/pushover tests
  pass). Next: **GUI‑3** (hinge assignment UI) / **GUI‑4** (nonlinear case manager), then **GUI‑7**.
- 2026‑09‑12 — **GUI‑3 shipped (hinge property + assignment + model‑view marking).** Wires the P9
  `FiberHingeBeamColumn2D` into the desktop front‑to‑back. Data model (`desktop/project.py`):
  `Hinge` (id, name, `lp`, `lp_j`, `relative`) + `Member.hinge` + `Project.hinges`;
  `member_length` + `resolve_hinge_lengths` (ratio×L when relative else absolute, clamped so
  lp_i+lp_j < L — the element requires ≤ L); loaded in `from_dict` (old projects: `hinge`/`hinges`
  default to none). `build_nonlinear_model` (`desktop/nonlinear.py`): a member carrying a hinge
  compiles to `FiberHingeBeamColumn2D` (lumped fiber hinge each end), else the distributed
  `BeamColumn2DCorotational`. UI (`desktop/hinge_editor.py`): `HingeDialog`
  (relative/absolute × symmetric/asymmetric, live hint), `HingeManagerDialog` (list/add/edit/delete
  with a delete‑in‑use guard), `HingeAssignmentDialog` (per‑fiber‑member hinge combo + "set all to");
  `MemberDialog` gained a Hinge combo (preserves the assignment on edit); "Hinges…" / "Assign
  hinges…" wired into the Edit menu; `ModelView.mark_hinges` overlays a magenta sphere inside each
  hinged member end (called after `set_model`). GUI‑6 capture generalized (`_section_def`) to read the
  force‑based element's `_e_committed`, so hinge members keep the fiber‑stress contour + damage
  coloring. The force‑based hinge element is more drift‑sensitive than the distributed element (§8) —
  it converges at a looser tol, so the pushover dialog now exposes **Convergence tol** + **Max
  iterations** (default 1e‑6 / 60). Verified: a hinged column pushes to 0.02 m at tol 1e‑5, the base
  bars yield (+500 MPa), damage grows; a portal frame shows magenta base hinges. Tests:
  `test_desktop_hinges.py` (13; 42 desktop tests pass). Next: **GUI‑4** (nonlinear case manager) or
  **GUI‑7** (report).
- 2026‑09‑12 — **GUI‑4 shipped (nonlinear case manager).** Nonlinear analysis cases are now persistent
  project objects instead of ad‑hoc dialog inputs. Data model (`desktop/project.py`): `NonlinearCase`
  (control node/DOF, target, `protocol` monotonic\|cyclic, cyclic `amplitudes`/`cycles`/`pts_per_cycle`,
  held `axial` preload, `continue_from`, `tol`/`max_iter`) + `Project.nonlinear_cases` +
  `nonlinear_case()` lookup + serialization (old projects default to none). Runner
  (`desktop/nonlinear.py`): `run_case(project, case, …)` dispatches monotonic (P7 `monotonic`, scalar
  `DisplacementControl` increment) or cyclic (P7 `stepped_cyclic` → `np.diff` signed increment schedule
  → **signed** disp/shear tracing the hysteresis); applies a held axial preload and chains
  `continue_from` through `StagedAnalysis` (P6), accumulating each stage's final load factor so the
  reported base shear is the **total** across stages; `_case_chain` walks the continue‑from links,
  `case_total_steps` sizes the progress bar. The GUI‑6 per‑step capture was refactored into a shared
  `_Capturer` class so `run_pushover` and `run_case` don't duplicate it (`run_pushover` behaviour
  unchanged; both now return a `protocol` key). UI (`desktop/nonlinear_cases.py`): `NonlinearCaseDialog`
  (protocol toggles the monotonic‑steps vs cyclic‑amplitudes fields; continue‑from combo of other cases)
  + `NonlinearCaseManagerDialog` (list/add/edit/delete with a continued‑from delete guard); "Nonlinear
  cases…" wired into the Analysis menu. The `PushoverDialog` gained a **Case** selector: choosing a
  saved case populates and locks the manual inputs and runs it via `run_case`; the curve tab renders a
  cyclic run as a hysteresis (signed axes + origin cross‑hairs, "Cyclic hysteresis" title). Verified: a
  saved cyclic case (±0.5/±1.0 × 0.02 m) traces symmetric ±210 kN loops; a two‑case continue‑from chain
  produces a continuous curve to 0.06 m. Tests: `test_desktop_nlcases.py` (11; 53 desktop tests pass).
  Note (§8): the force‑based/cyclic paths are more drift‑sensitive than a monotonic distributed push, so
  cases carry their own tol/max_iter (cyclic converges at ~1e‑5). Next: **GUI‑7** (report) or the infra
  (**GUI‑I1** step‑indexed results, **GUI‑I2** NL‑def persistence — largely covered by the case model).
- 2026‑09‑12 — **GUI‑7 shipped (nonlinear report + export).** The nonlinear post‑processing is now
  exportable. New pure module `desktop/nl_report.py` (Qt‑free, unit‑tested): `curve_csv` (step /
  displacement / base shear — signed, so a cyclic run round‑trips the hysteresis), `fibers_csv` (a
  captured step's per‑fiber y,z,σ,ε), `run_summary` (peak base shear + displacement at peak, max drift,
  initial stiffness, dissipated energy for cyclic runs, peak fiber strain), `member_peak_strain` (each
  fiber member's peak |strain| over the run — the hinge‑state summary), and `report_html` (a
  self‑contained, printable HTML report: run‑metadata table + results table + per‑member hinge‑state
  table + an embedded base64 PNG of the curve). `PushoverDialog` gained an **Export** row below the step
  controls — Curve CSV / Fibers CSV / Image (PNG·PDF·SVG of whichever tab is shown) / Report… — each
  enabled once a run has results and wired through `QFileDialog`, plus dialog helpers `_result`,
  `_report_meta`, `_curve_png_b64`. Verified: an RC‑column pushover exports a curve CSV, a fibers CSV, and
  a full HTML report with the embedded response curve. Tests: `test_desktop_nlreport.py` (10; 63 desktop
  tests pass). **The GUI fiber‑hinge workflow is now complete end‑to‑end** (GUI‑1‑2‑3‑4‑5‑6‑7): define
  inelastic materials → build a fiber section → define & assign hinges → define nonlinear cases → run
  threaded → post‑process (curve / fiber contour / deformed shape + hinge state) → export + report. Next:
  infra polish (**GUI‑I1** step‑indexed results model; **GUI‑I2** is largely covered — materials, hinges,
  and nonlinear cases all serialize), a convergence‑robustness pass on the force‑based element (§8), or
  **U5**/**P10** once the Midas/CSI exports land.
- 2026 — **Engine convergence‑robustness pass (§8).** Diagnosed the force‑based/fiber‑hinge outer
  Newton chatter: with an absolute `NormUnbalance` tol below the elements' state‑determination
  residual noise floor, `||R||` oscillated at the floor (~1e‑9 kip on the benchmark) while `||du||`
  had collapsed to ~1e‑16 (fully converged). Fix (two additive tolerance changes, so no
  previously‑converging step can now fail): (1) `NormUnbalance` also converges on displacement
  stagnation — `||du|| ≤ stag_du·||du_ref||` guarded by `||R|| ≤ stag_res·||R_ref||` (a stuck /
  near‑singular iteration with large residual is still rejected); (2) the force‑based elements
  (`beam_force`, `beam_force_3d`, `beam_fiber_hinge`) gained a relative inner tolerance
  `state_det_rel_tol` beside the absolute `state_det_tol`. Benchmark force‑based pushover now
  converges to ~0.36 in tip drift (was ~0.15 at tight tol) and the fiber‑hinge element to ~0.60 in;
  the remaining force‑based failure at deeper drift is the genuine **singular‑flexibility** limit of
  the flexibility formulation (smaller step or displacement‑based/fiber‑hinge path there).
  `test_convergence_robustness.py` (5); `test_newton_raphson.py` still green (stagnation guard leaves
  the `max_iter` too‑small case raising). Full suite unchanged except the 4 pre‑existing order‑8
  quadrature failures.
- 2026 — Added **§16 Commercial‑completeness roadmap** (gaps to a commercial fiber‑hinge tool,
  excluding the CSI/Midas comparison): core capability (C1 confined core/cover, C2 3‑D nonlinear GUI,
  C3 dynamic time‑history, C4 adaptive step‑cutting, C5 acceptance criteria), GUI/workflow (main‑view
  scrubbing, results persistence, run comparison, confinement input, model checks), and product infra
  (performance, V&V manual, packaging).
- 2026 — **C1 shipped (confined core / unconfined cover).** `fiber_section_from_spec`
  (`desktop/nonlinear.py`) now builds a **confined Mander core + unconfined cover** instead of one
  concrete law. `section_gui_core.Spec` gained hoop fields `conf_Asp`/`conf_s`/`conf_fyh`/
  `conf_hooptype` (all default 0 ⇒ unconfined, so existing runs are byte‑for‑byte unchanged);
  `_confinement_kw` derives the `conf_*` dict (core dia / `bc,dc` from geometry − 2·cover, `rho_cc`
  from the bars) and the shared `concrete_uniaxial_from`+`mander_confinement` raise the core peak to
  `f'cc` at `eps_cc`. Fibers are split by radius (circular: core ring `[0,R−cover]` + cover ring) or
  by the cover‑eroded core polygon (rect); the cover shell keeps `f'c`. Verified: core f'cc ≈ 41 MPa
  vs cover 35 MPa (fc=35), unconfined specs stay single‑law, rectangular split works, confined column
  solves a pushover. `test_desktop_nonlinear.py` +4 (15 total); 466 section + 74 desktop tests pass.
  The hoop **input UI** + core/cover preview is the separate item **G‑S4**.
- 2026 — **C1 refined into a true unification + G‑S4 input bridge.** Found the Section Designer
  already sources confinement from the section's tie **Link** group (`_auto_section_confinement`) and
  builds a two‑zone confined‑core/unconfined‑cover fibre section (`_confined_fiber_section`) for its
  confined M‑φ. Re‑did C1 to **reuse that exact path** instead of the parallel `Spec.conf_*` fields
  the first draft added (now removed): lifted `_parse_legs`/`_auto_section_confinement`/
  `_section_confinement` into the Qt‑free `section_gui_core` (shims in `section_designer`), and
  `fiber_section_from_spec` now delegates to `_confined_fiber_section` with `conf` from
  `_section_confinement` (tie group or manual override), threading an optional `materials` map
  (through `build_nonlinear_model`/`run_pushover`/`run_case`) for the hoop `f_yh`. So the pushover
  fibre section == the confined M‑φ section (one confinement source, no parallel path — §15), and the
  confinement **input** the M‑φ used now drives the pushover too (G‑S4 input bridge).
  `test_desktop_nonlinear.py` (16, incl. the tie‑group auto path); 466 section + 75 desktop tests
  pass. Remaining for G‑S4: a distinct core‑vs‑cover fibre preview in the Section Designer.
- 2026 — **G‑S1 shipped + G‑S2 (session‑level).** Main‑view step scrubbing: `ModelView.show_nl_step` renders a run step on the main 3‑D view — the deformed shape at an adjustable scale over a grey ghost, each member coloured by its peak fiber strain (hinge state) — from `NonlinearResults.step(k)` (GUI‑I1). The main window gained an **Analysis‑steps** dock (step slider + scale spin, hidden until a run) and `set_nl_results`, which the pushover dialog calls on close so results **survive the dialog** and stay scrubbable (G‑S2 session‑level; disk persistence + run history remain). `test_desktop_pushover_ui.py` +2 (scrub every step; dock hidden with no results); 77 desktop tests pass.
- 2026 — **C4 shipped (adaptive step‑cutting).** `NonlinearStaticAnalysis(substep=True)` halves the
  increment and retries on `NotConvergedError`, subdividing to cover the nominal step then growing
  back (≤ `max_substep_halvings`). Integrators advertise `supports_substep` + honour `set_step_scale`
  (LoadControl, scalar DisplacementControl; a cyclic schedule opts out → no‑op). Fixed a latent bug
  en route: a failed Newton solve leaves the drifted trial `disp` on the nodes (revert_step only
  rolls back the integrator/elements), so the substep loop snapshots/restores node `disp` per
  sub‑attempt — without it a retried sub‑step over‑shot. Opt‑in (default off ⇒ the "raise on
  non‑convergence" contract is unchanged, e.g. the `max_iter`‑too‑small test still raises); the GUI
  `run_pushover`/`run_case` push stages enable it. Only rescues `NotConvergedError`, not the
  force‑based singular‑flexibility `RuntimeError` (a genuine formulation limit). `test_adaptive_substep.py`
  (5): a too‑large step that aborts without substep completes with it and reaches the fine‑reference
  tip/base‑shear; still raises when even the smallest sub‑step can't converge. Full suite 2561 pass
   (only the 4 pre‑existing order‑8 quadrature failures).
- 2026 — **C3 (engine/compute) shipped: nonlinear dynamic time‑history.** `desktop/nonlinear.run_time_history` runs the fiber model under rigid‑base ground acceleration: mass from `density` (rho·A·L, new `build_nonlinear_model(density=…)` giving the element materials `rho`), Rayleigh damping to `zeta` at the first two modal frequencies (`EigenAnalysis` → `RayleighDamping.from_modes`), base excitation `−M·ι·ü_g(t)` via `ground_motion_force`, and `NonlinearTransientAnalysis` (Newmark+Newton). Returns the monitored DOF's disp/vel/accel history + peak drift; SI force scale means a looser default `tol` (in force units) than the static default. Verified: runs to completion on a synthetic accelerogram, finite response, scales with excitation, sublinear (yielding) at high amplitude. `test_desktop_timehistory.py` (3). Remaining: the GUI `time_history` case type (record import + damping/direction inputs + response‑history plot).
- 2026‑09‑12 — **C5 shipped: ASCE 41 hinge acceptance criteria.** New engine module
  `femsolver/performance/acceptance.py` — per‑material fibre‑strain IO/LS/CP limits
  (`FiberStrainLimits`; concrete/steel defaults taken from the **benchmark CSI acceptance table**,
  §2), `classify_strain`, and `section_state` (worst fibre → governing level). The pushover / case
  runs capture per‑step per‑member acceptance level (`accept_frames`) and the first drift reaching
  IO/LS/CP (`accept_milestones`); `NonlinearResults` exposes `member_state` + `accept_milestones`.
  The main‑view step scrubbing gained a **discrete IO/LS/CP colour mode** (Analysis‑steps dock
  toggle + governing level in the step label), the pushover dialog logs the milestones, and the HTML
  report shows an ASCE 41 hinge‑state column + acceptance‑milestone table (backward‑compatible: falls
  back to the peak‑strain yielded/elastic flag when no acceptance capture). Demo column reached IO at
  d≈42 mm, LS at d≈74 mm. `test_acceptance.py` (6) + report/results C5 tests. Next per §16.5: **C2**
  (3‑D nonlinear in the app) or close the remaining ◐ (C3 GUI case type, G‑S2 disk, G‑S4 preview).

---

## 14. GUI & Productization roadmap

**Where the GUI is today (audited 2026‑09‑12):** the desktop app (`desktop/`, PySide6 +
PyVista/VTK) is **linear‑only**. It builds models, has a strong Section Designer with
*section‑level* fiber analysis (M‑φ, P‑M‑M), load cases / combos, and results as deformed shape
+ N/V/M diagrams via `LinearStaticAnalysis` (`desktop/main_window.py:290`). There is **no**
hinge, nonlinear‑solver, displacement‑control, staged‑case, pushover/cyclic, or hysteresis UI
anywhere. Exposing fiber hinges to users = building the nonlinear‑frame workflow front‑to‑back.

Each row: **Background** (engine/infra) + **GUI**. Status ✅ exists · ⚠ partial · ❌ missing.

| Stage (user workflow) | Background (engine/infra) | GUI | Status |
|---|---|---|---|
| **1. Define inelastic materials** (σ‑ε preview; confined/unconfined; Park steel) | Mander confinement calc (P2 ❌); `ConcreteMander`/`UniaxialReinforcingSteel`/Menegotto‑Pinto ✅; material def‑schema serialization ⚠ | Inelastic‑material editor with live σ‑ε plot; hoop inputs → f′cc/ε_cc/ε_cu ❌ | mostly ❌ |
| **2. Build the fiber section** (mesh + fiber preview) | one compiler `Section.fiber_section_2d/3d` ✅ (extend for circular + core/cover, §15 U2) | fiber‑mesh panel (rings/wedges), fiber preview colored by material, per‑region material assign ⚠ (SD exists) | ⚠ |
| **3. Define & assign the hinge** (from section; RelDist, length, save‑fiber‑resp) | fiber‑hinge element + assignment data model (P9 ❌) | hinge property dialog + member‑assignment tool + show hinges in model view ❌ | ❌ |
| **4. Nonlinear load cases & protocols** (control mode, monitor DOF, staged, cyclic, NL params) | staged continuation (P6 ✅); protocol generators (P7 ✅); `NonlinearStaticAnalysis` ✅ | nonlinear case type in the case manager: control/monitor/target/continue‑from/cyclic table/NL‑params ✅ (GUI‑4: `NonlinearCase` + manager + `run_case`) | ✅ |
| **5. Run the solve** (no UI freeze; progress + convergence; cancel) | **run on a worker thread**, stream step/convergence callbacks, cancellation, capture per‑step state ❌ (solve is a blocking call) | analysis‑run dialog + progress/convergence dock + non‑convergence diagnostics ❌ | ❌ (biggest infra gap) |
| **6. Post‑process** (hysteresis/pushover; step slider/animation; fiber contour; hinge state) | recorders: monitored DOF, section N‑M/M‑φ, **per‑fiber σ‑ε**, hinge F‑D (P8 ❌); **step‑indexed results model** ⚠ (`node.disp` is one state) | X‑Y plot panel; step slider/animation; fiber σ‑ε contour on the section; hinge‑state color map; energy ❌ | ❌ (what makes it feel commercial) |
| **7. Report & export** | nonlinear result export (curves, fiber states) ✅ (GUI‑7 `nl_report`); results→CSV ✅ | export buttons on plots ✅; hinge/analysis report ✅ (HTML) | ✅ |

**Cross‑cutting background infrastructure (prerequisites):**
- **Threaded solve + progress/cancel** (stage 5) — prerequisite for *all* nonlinear UI.
- **Step‑indexed results model** — every view (deformed, diagrams, section) reads "state at step k";
  today state is a single snapshot.
- **Project persistence** — inelastic materials, fiber sections, hinges, nonlinear cases must
  save/load in the `.` project format (`desktop/project.py`).
- **Performance** — fiber sections are heavy (fibers × IPs × steps); vectorize/cache; "results on
  disk" for long cyclic runs.
- **Validation & model checks** — units, section/material sanity, surface the Midas/CSI golden
  comparison (P10) in‑app.

**Suggested sequencing** (interleaves with engine phases):
1. **Now (rides on P1–P3):** GUI‑1, GUI‑2 — inelastic‑material editor + σ‑ε preview; fiber‑mesh/preview in the Section Designer.
2. **Then:** GUI‑5 (threaded solver + progress dock — reusable everywhere) and GUI‑4 (nonlinear case manager).
3. **Then:** GUI‑6 (recorders P8 + hysteresis/step‑slider/fiber‑contour). *This is the commercial differentiator.*
4. **Last:** GUI‑3 (hinge‑assignment UI, needs P9) and GUI‑7 (reporting).

> Fastest path to a demoable "fiber hinge in the GUI": the **distributed `ForceBeamColumn`**
> (chosen D1 default) driven by GUI‑1‑2‑5‑6 — no hinge‑assignment UI (GUI‑3) needed until CSI‑style
> hinge parity later.

### 14.1 GUI status tracker
| Item | Deliverable | Depends on | State | Notes |
|---|---|---|---|---|
| GUI‑1 | Inelastic‑material editor + σ‑ε preview | P2, P3 | ☑ | 2026‑09‑12 — `desktop/materials.py` (engine‑backed `uniaxial_law`/`stress_strain_curve`) + `desktop/material_editor.py` (`MaterialDialog` w/ live matplotlib σ‑ε preview + `MaterialManagerDialog`); `Material.params`; wired "Materials…" into main_window; `test_desktop_materials.py` (11, headless offscreen). Kinds: elastic / Kent‑Park / Mander concrete / Park + cyclic steel |
| GUI‑2 | Fiber‑mesh panel + fiber preview in Section Designer | P1, §15 U2 | ☑ | pre‑existing in `desktop/section_designer.py` (Fibres tab, mesh overlay + centroids toggle, mesh‑density combo, fibres‑CSV export via `core.section_fibers`/`section_fiber_mesh`) — now running on the **U2‑unified `polar_cells`** mesher. Confirmed 2026‑09‑12 |
| GUI‑3 | Hinge property + assignment UI | P9 | ☑ | 2026‑09‑12 — hinge **property** + **assignment** + **model‑view marking**, wiring the P9 `FiberHingeBeamColumn2D` into the desktop. Data model: `project.Hinge` (lp / lp_j / relative) + `Member.hinge` + `Project.hinges` + `resolve_hinge_lengths` (ratio×L or absolute, clamped lp_i+lp_j<L) + serialization. `build_nonlinear_model` compiles a hinged member to `FiberHingeBeamColumn2D` (lumped fiber hinge each end), else the distributed corotational element. `desktop/hinge_editor.py`: `HingeDialog` (relative/absolute, symmetric/asymmetric), `HingeManagerDialog` (list/add/edit/delete + in‑use guard), `HingeAssignmentDialog` (per‑member combo + "set all"); `MemberDialog` gained a Hinge combo; "Hinges…"/"Assign hinges…" in the Edit menu; `ModelView.mark_hinges` draws a magenta marker inside each hinged member end. GUI‑6 capture generalized to read the force‑based element's `_e_committed`, so hinge members keep their fiber contour + damage coloring. The force‑based hinge element needs a looser tol than the distributed default (§8), so the pushover dialog now exposes **Convergence tol** + **Max iterations**. Tests: `test_desktop_hinges.py` (13). |
| GUI‑4 | Nonlinear case manager (control/monitor/staged/cyclic/NL‑params) | P6, P7 | ☑ | 2026‑09‑12 — nonlinear analysis **cases as persistent project objects** + manager + runner. Data model (`project.py`): `NonlinearCase` (control node/DOF, target, protocol monotonic\|cyclic, cyclic amplitudes/cycles/pts, held axial preload, `continue_from`, tol/max_iter) + `Project.nonlinear_cases` + `nonlinear_case()` + serialization. Runner (`nonlinear.py`): `run_case` dispatches monotonic (P7 `monotonic`) or cyclic (P7 `stepped_cyclic` → signed `DisplacementControl` schedule → signed disp/shear tracing the hysteresis), holds an axial preload and chains `continue_from` via `StagedAnalysis` (P6) with correct total‑base‑shear accounting across stages; `_case_chain`/`case_total_steps` helpers; capture refactored into a shared `_Capturer` (reused by `run_pushover`). UI (`nonlinear_cases.py`): `NonlinearCaseDialog` (protocol‑dependent fields toggle) + `NonlinearCaseManagerDialog` (list/add/edit/delete + continued‑from guard); "Nonlinear cases…" in the Analysis menu; the pushover dialog gained a **Case** selector that populates+locks the manual inputs and runs the saved case (cyclic → hysteresis curve). Tests: `test_desktop_nlcases.py` (11; 53 desktop tests pass). |
| GUI‑5 | Threaded solver + progress/convergence dock + cancel | (infra) | ☑ | 2026‑09‑12 — **compute** (`desktop/nonlinear.py`: `fiber_section_from_spec`/`build_nonlinear_model`/`run_pushover`) + **threaded UI**. Engine enabler: optional `step_callback` on `NonlinearStaticAnalysis` (per‑step hook; return False = cancel). `desktop/pushover_dialog.py` = `PushoverWorker` (QThread, streams progress, cooperative cancel) + `PushoverDialog` (inputs, progress bar, convergence log, Cancel, live base‑shear/disp curve); "Nonlinear pushover…" wired into the Analysis menu. Tests: `test_desktop_nonlinear.py` (7, incl. step_callback + cancel), `test_desktop_pushover_ui.py` (4, headless — wiring/worker/cancel/dialog). |
| GUI‑6 | NL post‑processing (hysteresis, step slider/anim, fiber contour, hinge state) | P8, GUI‑5 | ☑ | 2026‑09‑12 — **fiber‑stress/strain contour + step slider** then **deformed‑shape animation + hinge‑state coloring** shipped. Engine: `BeamColumn2DCorotational` stores per‑IP `_e_sections` (converged section deformations). `run_pushover(capture_fibers=True)` snapshots the base section's per‑fiber (y,z,σ,ε) each step (from a clone — live state untouched); `capture_shape=True` snapshots every node's (dx,dy) + per‑member peak \|fiber strain\| (`shape_frames`/`damage_frames`). `PushoverDialog` right pane = **curve \| fiber stress \| deformed shape** tabs sharing one **step slider** + a **▶ Play/Pause** `QTimer` animation; fiber tab has a stress/strain toggle (steel‑yield pattern / plane‑section gradient + concrete crush), deformed‑shape tab draws the deformed frame colored by peak fiber strain (hinge state) with an adjustable displacement scale over a gray undeformed reference. Tests: `test_pushover_capture_fibers`, `test_pushover_capture_shape`, `test_pushover_capture_shape_and_fibers_together`, `test_pushover_no_capture_omits_frames`; `test_dialog_fiber_contour`, `test_dialog_deformed_shape`, `test_dialog_animation_advances_and_loops`. |
| GUI‑7 | Nonlinear report + export | GUI‑6 | ☑ | 2026‑09‑12 — export + report from the pushover dialog. Pure `desktop/nl_report.py`: `curve_csv` (step/disp/base‑shear, signed so cyclic round‑trips), `fibers_csv` (a captured step's y,z,σ,ε), `run_summary` (peak shear + disp‑at‑peak, max drift, initial stiffness, cyclic dissipated energy, peak fiber strain), `member_peak_strain` (per‑member hinge state), `report_html` (self‑contained printable report: run metadata + metrics + hinge‑state table + embedded base64 curve PNG). `PushoverDialog` gained an **Export** row — Curve CSV / Fibers CSV / Image (PNG·PDF·SVG of the shown tab) / Report… — enabled once a run has results, wired through `QFileDialog`. Tests: `test_desktop_nlreport.py` (10; 63 desktop tests pass). |
| GUI‑I1 | Step‑indexed results model | (infra) | ☑ | 2026 — `desktop/nl_results.py`: pure, Qt‑free `NonlinearResults` (+ `StepState`) consolidating a run's parallel per‑step arrays (curve, monitored‑section fibers, nodal deformation, per‑member peak strain) into one **"state at step k"** model — `from_run`/`to_dict` round‑trip the run dict (export/report unchanged), `step(k)`, `has_fibers`/`has_shape`, summary helpers. `PushoverDialog` refactored onto it (the three parallel frame lists removed; fiber/shape tabs + export read `_results.step(k)`). Shared, testable base for future views (e.g. main‑view step scrubbing — follow‑up). `test_desktop_nl_results.py` (7); 70 desktop tests pass |
| GUI‑I2 | Project persistence for NL defs | (infra) | ☐ | |

---

## 15. Unified Section Analysis architecture (de‑duplication)

**Goal (user):** one section‑analysis capability, not several parallel ones. What you analyze in
the **Advanced Section Analysis** tool must be *exactly* what runs inside a **fiber hinge** — same
section definition, same material laws, same fiber discretization, same code.

### 15.1 The duplication today (concrete)
1. **Exact integrator trapped in the GUI.** `exact_mphi`, `section_pm_slice`, `_width_bands` live
   in `section_gui_core.py` (a *GUI* file). The desktop app reaches **up** into it
   (`desktop/section_designer.py` → `core.exact_mphi`). The engine, tests, and the fiber‑hinge
   stream cannot reuse it.
2. **Two+ fiber‑generation paths.** `Section.fiber_section_2d/3d` (polygon grid) +
   `_discretize_polygon_to_fibers`; the RC **core/cover** split and composite multi‑material
   assembly done separately inside `section_gui_core.py`; and P1's circular ring/wedge builder.
3. **Three M‑φ implementations.** `exact_mphi` (GUI), `design/concrete/moment_curvature`, and the
   fiber‑section path.
4. **Good news — already shared:** section geometry factories, `ReinforcementLayout`,
   `femsolver.materials.uniaxial` (incl. `UniaxialReinforcingSteel`), `FiberSection2D`, biaxial
   P‑M‑M — the Section Designer is already ~80% on the engine.

### 15.2 Target architecture (one pipeline, two views)
```
        ┌─────────────────────────── ONE definition ───────────────────────────┐
        │  femsolver.sections.Section  (polygon geometry + ReinforcementLayout   │
        │  + material regions: core/cover/…  + tendons)                          │
        └───────────────────────────────┬───────────────────────────────────────┘
                                         │  ONE compiler
                                         │  Section.fiber_section_2d/3d(...)
                                         │  (polygon grid | circular ring‑wedge | core/cover split)
                                         ▼
                               FiberSection2D / FiberSection3D
                        (element‑facing; discrete; stateful commit/revert/clone)
                    ┌────────────────────┴───────────────────────┐
                    │                                             │
         ONE analysis core (engine)                     the SAME object feeds
   femsolver.sections.analysis:                         the nonlinear element
     • exact_mphi / section_pm_slice  (lifted from GUI)         │
     • biaxial P‑M‑M (design/concrete)                          ▼
     • fiber‑path M‑φ / P‑M                            ForceBeamColumn / FiberHinge
     • nominal / design capacity                       (P5/P6/P9)  → staged NL analysis
   backends behind ONE result API; tokens C_EXACT/C_FIBRE/C_NOMINAL/C_DESIGN
                    │
      ┌─────────────┴──────────────┐
      ▼                            ▼
  Advanced Section Analysis UI   (headless / CLI / tests)
  (desktop + Streamlit call the ENGINE, never each other's GUI modules)
```

### 15.3 Consolidation tasks
| ID | Task | Touches | Notes |
|---|---|---|---|
| **U1** | **Lift the exact integrator into the engine.** Move `exact_mphi`, `section_pm_slice`, `_width_bands` from `section_gui_core.py` → `femsolver/sections/analysis.py` (new). Leave thin re‑export shims in the GUI file; point `desktop/section_designer.py` at the engine. Add engine tests. | `section_gui_core.py`, `desktop/section_designer.py`, new `sections/analysis.py` | highest‑value; unblocks reuse + testing |
| **U2** | **One section→fiber compiler.** Extend `Section.fiber_section_2d/3d` to (a) mesh circular geometry via P1's `circular_sector_fibers` (ring/wedge, not bbox grid), (b) support **core/cover material regions** (fold in the GUI's RC assembly), (c) optional rebar‑area subtraction. Make `rc_circular_column_section` a **thin wrapper** that builds a `Section` + calls the compiler. | `sections/section.py`, `sections/response/fiber_build.py`, `section_gui_core.py` | removes the parallel fiber paths (incl. P1's) |
| **U3** | **One M‑φ / P‑M‑M API.** Wrap `exact` (U1), `fiber`, `design.moment_curvature`, and biaxial P‑M‑M behind a single results type + backend selector (the existing C_EXACT/C_FIBRE/C_NOMINAL/C_DESIGN tokens). | `sections/analysis.py`, `design/concrete/*` | one result object the GUI + hinge both consume |
| **U4** | **One material library everywhere** (already `femsolver.materials.uniaxial`). Ensure the Mander confinement calc (P2) produces the confined law used by **both** the section tool and the hinge. | `materials/uniaxial/*` | mostly done; P2 closes the gap |
| **U5** | **One section object across tool + hinge (UI).** In the app, "Analyze section" and "Assign as fiber hinge" are two actions on the *same* `Section`. No separate hinge‑section definition. | `desktop/section_designer.py`, hinge UI (GUI‑3) | delivers "what you analyze is what you run" |

### 15.4 Impact on the engine phases
- **P1** — keep the shipped code; **re‑wire** the circular mesher through `Section.fiber_section_2d`
  (U2) so there is one public fiber‑build path.
- **P3** — `UniaxialReinforcingSteel` already covers the monotonic Park backbone; P3 shrinks to
  verify + cyclic.
- **U1–U3** slot in **before/around P4** (the section M‑φ phase) so P4 is built on the unified core,
  not a soon‑to‑be‑replaced one.

### 15.5 Unification status tracker
| Item | Deliverable | Depends on | State | Notes |
|---|---|---|---|---|
| U1 | Lift exact integrator → `sections/analysis.py` (+ shims, tests) | — | ☑ | 2026‑09‑12 — moved `exact_mphi`/`section_pm_slice`/`_width_bands` + material factories + `mander_confinement` to `femsolver/sections/analysis.py`; `section_gui_core` re‑exports (desktop+Streamlit unchanged); before/after byte‑identical; `test_section_analysis_exact.py` (7); full suite 2454 pass |
| U2 | One section→fiber compiler (circular + core/cover); P1 rewired | P1 | ☑ | 2026‑09‑12 — unified the **polar circular‑mesh math** into the engine: `polar_divisions` + `polar_cells` in `sections/response/fiber.py`; `circular_sector_fibers` now builds on them, and the GUI `section_fibers`/`section_fiber_mesh` (which had a duplicate annular‑sector formula) now import the same helpers — one implementation. Verified identical (`polar_divisions(1400)`=(15,94); GUI circular cells area=πR² exact); 101 affected tests pass. Deferred nicety: `Section.fiber_section_2d` auto‑polar routing + core/cover (rc_circular_column_section already builds circular RC for the hinge stream). |
| U3 | One M‑φ / P‑M‑M API + result type | U1 | ☑ | 2026‑09‑12 — added backend tokens `C_EXACT/C_FIBRE/C_NOMINAL/C_DESIGN` + `MomentCurvatureResult`/`PMInteractionResult` + `moment_curvature_analysis(backend=…)`/`pm_interaction(backend=…)` in `sections/analysis.py`; lifted `pmm_slice` from the GUI (verbatim + re-export shim, U1 pattern); `test_section_analysis_unified.py` (10); full suite 2463 pass (only the 4 pre-existing quadrature failures) |
| U4 | Confinement law shared tool↔hinge | P2 | ☑ | one calc (`mander_confinement` + `mander_confined_circular`) shared by the Section Designer, the fiber hinge (P5), and now the GUI‑1 material bridge |
| U5 | Same `Section` for analysis + hinge (UI) | U2, GUI‑3 | ☐ | |

**Principle:** *one section definition → one compiler → one analysis core → two views (design tool
and fiber hinge).* Every session adds to this pipeline; nobody forks a parallel section engine.

---

## 16. Commercial‑completeness roadmap (post‑benchmark)

Gaps that remain to make the fiber‑hinge capability **commercial‑grade**, *excluding* the
CSI/Midas golden comparison (that is §7 + P10 + U5, blocked on the user's exports). Grounded in the
current tree, not aspirational. Legend: ☐ todo · ◐ partial · ☑ done.

### 16.1 Core capability (limits what users can model)
| ID | Item | State | Notes |
|---|---|---|---|
| **C1** | **Confined core / unconfined cover split** in the analysis fiber section | ☑ | 2026 — the nonlinear pushover now builds a **confined Mander core + unconfined cover** through the **same** path as the Section Designer's confined M‑φ (unified, per §15): `desktop/nonlinear.fiber_section_from_spec` delegates to `section_gui_core._confined_fiber_section`, with the confinement `conf` dict from the shared `_section_confinement` (tie **Link** group + geometry, or a manual override). Confinement helpers (`_parse_legs`/`_auto_section_confinement`/`_section_confinement`) lifted from the GUI into the Qt‑free core (shims left in `section_designer`). Unconfined sections keep the single‑law path (unchanged). `test_desktop_nonlinear.py` +5. *(An earlier draft added parallel `Spec.conf_*` fields; removed in favour of the one tie‑group source.)* **Confinement *input UI* + core/cover preview = G‑S4.** |
| **C2** | **3‑D nonlinear in the app** (P‑M2‑M3) | ☐ | `ForceBeamColumn3D` (P7) exists; `desktop/nonlinear.build_nonlinear_model` raises for `ndm≠2`. Needs 3‑D fiber section + element wiring + 3‑D control/monitor UI. |
| **C3** | **Dynamic time‑history** (true transient: mass + damping) for fiber elements | ◐ | **Engine/compute done** — `desktop/nonlinear.run_time_history(project, accel, dt, …)` runs a nonlinear dynamic base‑excitation analysis on the fiber model: mass from `density` (rho·A·L, via `build_nonlinear_model(density=…)`), Rayleigh damping calibrated to `zeta` at the first two modal frequencies (`EigenAnalysis`+`RayleighDamping.from_modes`), base motion via `ground_motion_force` (`−M·ι·ü_g`), integrated with `NonlinearTransientAnalysis` (Newmark+Newton). Returns the monitored DOF's disp/vel/accel history + peak. `test_desktop_timehistory.py` (3). **Remaining:** a GUI `time_history` case type (ground‑motion record import, `direction`/`zeta`/`density`, response‑history plot). |
| **C4** | **Adaptive step‑cutting / substepping** on non‑convergence | ☑ | 2026 — `NonlinearStaticAnalysis(substep=True)` halves the increment and retries on `NotConvergedError`, subdividing to cover the nominal step then growing back (up to `max_substep_halvings`). Integrators advertise `supports_substep` + honour `set_step_scale` (LoadControl, scalar DisplacementControl; a cyclic schedule opts out). Snapshots/restores node `disp` on a failed sub‑attempt (the drifted trial isn't otherwise rolled back). Opt‑in (default off ⇒ existing "raise on non‑convergence" contract unchanged); the GUI push runs enable it. `test_adaptive_substep.py` (5): a step that fails with a small max_iter now completes and reaches the fine‑reference target/base‑shear; still raises when hopeless. Note: only rescues `NotConvergedError`, not the force‑based singular‑flexibility `RuntimeError` (a true limit). |
| **C5** | **Hinge acceptance criteria** (ASCE 41 IO/LS/CP) | ☑ | 2026 — engine `femsolver/performance/acceptance.py`: per‑material fibre‑strain limits (`FiberStrainLimits`, concrete/steel defaults **from the benchmark CSI acceptance table**: concrete comp 0.003/0.006/0.015 tension‑ignored, steel tension 0.01/0.02/0.05 comp 0.005/0.01/0.02), `classify_strain`, `section_state` (worst fibre → level). `run_pushover`/`run_case` capture per‑step per‑member acceptance level (`accept_frames`) + first‑reach milestones (`accept_milestones` — drift at IO/LS/CP). Surfaced: `NonlinearResults.member_state`/`accept_milestones`; main‑view step scrubbing gains a **discrete IO/LS/CP colour mode** (Analysis‑steps dock toggle + governing state in the step label); pushover dialog logs the milestones; the HTML report shows an ASCE 41 hinge‑state column + acceptance table (falls back to yielded/elastic when no capture). `test_acceptance.py` (6) + report/results C5 tests. |

### 16.2 GUI / workflow
| ID | Item | State | Notes |
|---|---|---|---|
| **G‑S1** | **Main model‑view step scrubbing** | ☑ | 2026 — `ModelView.show_nl_step(model, node_disp, member_damage, scale)` renders a run step on the main 3‑D view (deformed shape over a ghost, members coloured by peak fiber strain / hinge state). The main window gained an **Analysis‑steps** dock (step slider + displacement‑scale) that scrubs the whole model through the run, driven by `NonlinearResults.step(k)` (GUI‑I1). |
| **G‑S2** | **Nonlinear results persistence + re‑open + run history** | ◐ | **Session‑level done** — the pushover/case dialog hands its `NonlinearResults` back to the main window (`set_nl_results`), so results survive the dialog close and stay scrubbable (re‑view). **Remaining:** persist to the `.` project file (at least the curve) across save/load, and a multi‑run history. |
| **G‑S3** | **Run comparison / envelopes** | ☐ | Overlay pushovers, build cyclic backbone/envelope across runs. |
| **G‑S4** | **Confinement input bridge + core/cover preview** | ◐ | **Input bridge done** — the tie **Link** group (Rebars tab) + Confinement tab + manual override that already drove the confined M‑φ now also drive the pushover (C1 unification), so "what you input is what you run". **Remaining:** a distinct **core‑vs‑cover fibre preview** in the Section Designer fibres view. |
| **G‑S5** | **In‑app model checks / diagnostics** | ◐ | Convergence dock exists; add units/section/material sanity + non‑convergence guidance surfaced pre‑ and post‑run. |

### 16.3 Cross‑cutting product infra
| ID | Item | State | Notes |
|---|---|---|---|
| **X1** | **Performance** (fibers × IPs × steps × elements) | ☐ | Vectorize/cache; "results on disk" for long cyclic runs; benchmark. |
| **X2** | **Verification manual** (non‑CSI/Midas) | ☐ | Package the internal cross‑checks + analytical/closed‑form + published experiments (Caltrans column has public test data) into a V&V doc. |
| **X3** | **Packaging / docs / tutorials / in‑app examples** | ☐ | Installer, user manual, examples gallery. |

### 16.4 Explicitly deferred (§1.3 — named, not scheduled)
Biaxial concrete constitutive coupling & nonlinear shear in the fiber; bond‑slip / anchorage‑slip;
longitudinal‑bar buckling; low‑cycle fatigue. Consistent with what the reference fiber hinges omit.

### 16.5 Suggested sequence
**C1** (confined core/cover — accuracy, engine already supports it) → **G‑S2 + G‑S1** (results
persistence + main‑view scrubbing — makes the tool feel finished) → **C4** (adaptive stepping —
"it just works") → **C3** (dynamic time‑history) → **C2** + **C5** (3‑D nonlinear + acceptance).

### 16.6 Status tracker
| ID | State | Commit / date |
|---|---|---|
| C1 confined core/cover | ☑ | 2026 — unified onto section_gui_core._confined_fiber_section (one tie-group source) |
| C2 3‑D nonlinear GUI | ☐ | |
| C3 dynamic time‑history | ◐ | 2026 — engine run_time_history done; GUI case type remains |
| C4 adaptive step‑cutting | ☑ | 2026 — NonlinearStaticAnalysis(substep=True); GUI runs enable it |
| C5 acceptance criteria | ☑ | 2026 — performance/acceptance.py (ASCE 41 IO/LS/CP fibre limits); accept_frames+milestones; main-view state colouring; report acceptance table |
| G‑S1 main‑view scrubbing | ☑ | 2026 — ModelView.show_nl_step + Analysis-steps dock |
| G‑S2 results persistence | ◐ | session-level (held on main window); disk + history remain |
| G‑S3 run comparison | ☐ | |
| G‑S4 confinement input + preview | ◐ | input bridge done (tie group drives pushover); core/cover preview remains |
| G‑S5 model checks | ◐ | |
| X1 performance | ☐ | |
| X2 verification manual | ☐ | |
| X3 packaging/docs | ☐ | |
