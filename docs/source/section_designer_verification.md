# Section Designer verification set (femsolver vs Midas GSD)

**Status:** release-gating V&V for the General Section Designer
(biaxial P-M-M + moment-curvature).
**Codes:** AASHTO LRFD BDS (10th ed., 2024) and Eurocode 2
(EN 1992-1-1).
**Generator:** [`examples/78_section_designer_verification.py`](../../examples/78_section_designer_verification.py)
driving [`femsolver.benchmarks.section_designer`](../../src/femsolver/benchmarks/section_designer.py).

This document pins down five standardized cross-sections **exactly** so
the identical model can be built in Midas GSD and compared in parallel.
Every dimension, material property, rebar/tendon coordinate, analysis
setting, sign convention, and unit is stated here; nothing is left to
interpretation.

## Workflow

1. Build sections **S1-S5** below in Midas GSD using the exact
   coordinates and material properties given.
2. Run `python examples/78_section_designer_verification.py`. It writes
   `examples/section_designer_verification/`:
   `section_designer_verification.csv` (results with a **blank
   `midas_gsd` column**), the Markdown table, one geometry SVG per
   section, and P-M / M-phi plots.
3. Read each quantity off the GSD model, fill the `midas_gsd` column of
   the CSV, and re-run with `--gsd <filled.csv>`. The driver prints the
   percentage difference and pass/fail against each quantity's
   tolerance.

## Global conventions (must match in GSD)

| Item | Convention |
|---|---|
| Local frame | Origin at the **geometric centroid**; `y` vertical (depth / strong axis), `z` horizontal (width). |
| `M_z` | Bending about the local **z**-axis (strong-axis bending for a taller-than-wide section). |
| `M_y` | Bending about the local **y**-axis. |
| Axial sign | **Compression positive** (AASHTO / ACI convention). |
| Rebar / tendon coords | `(z, y)` from the **centroid**, in metres internally / mm in this sheet. |
| Steel modulus | `E_s = 200 GPa` (rebar), `E_p = 195 GPa` (strand). |
| Display units | force **kN**, moment **kN·m**, length **mm**, area **mm²**, `I` **mm⁴**, stress **MPa**, curvature **1/m**. |

All interaction quantities are taken on the **θ = 0** slice (bending
about the centroidal z-axis, compression on the +y face).

## Design-code settings

**AASHTO LRFD BDS Section 5 (10th ed., 2024)** — normal-strength
concrete:

- Rectangular stress block: intensity `α₁·f'c` with `α₁ = 0.85`
  (Art. 5.6.2.2, for `f'c ≤ 69 MPa`), depth `β₁·c` with `β₁` per
  5.6.2.2 (= the ACI 318 value: 0.85 for `f'c ≤ 28 MPa`, reducing 0.05
  per 7 MPa to a floor of 0.65). Extreme-fibre `ε_cu = 0.003`.
- Resistance factors `φ` (Art. 5.5.4.2): **0.75** compression-
  controlled (tied *and* spiral), **0.90** tension-controlled RC,
  **1.00** tension-controlled PC; linear transition on the net tensile
  strain `ε_t` between the compression-controlled limit (`ε_y = f_y/E_s`)
  and the tension-controlled limit `0.005`.
- Pure-axial cap (5.6.4.4): `P_n,max = 0.80·P_o` tied, `0.85·P_o` spiral.

Because the normal-strength AASHTO stress block equals the ACI Whitney
block, the **nominal** surface is identical to ACI; only the
`φ`-reduced **design** surface differs.

**Eurocode 2 (EN 1992-1-1 §3.1.7)** — design values, partial factors
built into the stress block:

- `f_cd = α_cc·f_ck/γ_c = 0.85·f_ck/1.5`, block depth `λ·c` with
  `λ = 0.80` (`f_ck ≤ 50 MPa`), `ε_cu3 = 0.0035`.
- `f_yd = f_yk/γ_s = f_yk/1.15`. No separate `φ` (already design-level),
  so nominal ≡ design in the tables.

`f'c` (AASHTO) and `f_ck` (EC2) are both the cylinder strength and use
the **same** numeric value for a given section.

## Moment-curvature constitutive (S3, S5)

Moment-curvature is constitutive-driven, not code-specific. Enter these
**exact** material laws in GSD:

- **Concrete — Kent-Park** (unconfined): peak stress `f'c` at
  `ε_c0 = 0.002`; linear post-peak to `0.4·f'c` (= `f_cu`) at
  `ε_cu = 0.0035`; zero tension except for the cracking check.
- **Cracking**: modulus of rupture `f_r = 0.62·√(f'c [MPa]) MPa`
  (AASHTO 5.4.2.6). `M_cr` from the uncracked transformed section
  (including prestress for S5).
- **Rebar — bilinear**: `E_s = 200 GPa`, yield `f_y`, strain-hardening
  ratio `b = 0.01`.
- **Strand (S5) — bilinear**: `E_p = 195 GPa`, `f_py = 1675 MPa`,
  `b = 0.005`, effective prestress after losses `f_pe = 1100 MPa`
  applied as an initial strain `ε_pe = f_pe/E_p`.
- Sweep `κ = 0 … 0.06 /m` in 60 steps at `P = 0`. Ductility
  `μ_φ = κ_u/κ_y`; `κ_u` at peak `M` (or concrete crushing at
  `ε_cu`, whichever first).

---

## S1 — Rectangular tied RC column, 400 × 600 mm

- **Geometry:** solid rectangle, width `b = 400 mm` (z), depth
  `h = 600 mm` (y). Centroid at origin → `z ∈ [-200, +200]`,
  `y ∈ [-300, +300]`.
- **Concrete:** `f'c = f_ck = 30 MPa`. **Rebar:** `f_y = 500 MPa`.
- **Reinforcement:** 8 × 25 mm bars (`A = 490.9 mm²` each), clear cover
  50 mm to bar centroid. Tied column.

| Bar | z [mm] | y [mm] | A [mm²] |
|--:|--:|--:|--:|
| 1 | −150 | −250 | 490.9 |
| 2 | 0 | −250 | 490.9 |
| 3 | +150 | −250 | 490.9 |
| 4 | −150 | +250 | 490.9 |
| 5 | 0 | +250 | 490.9 |
| 6 | +150 | +250 | 490.9 |
| 7 | −150 | 0 | 490.9 |
| 8 | +150 | 0 | 490.9 |

## S2 — Circular spiral RC column, 600 mm diameter

- **Geometry:** solid circle `D = 600 mm`, centroid at origin. femsolver
  approximates the circle by a **120-gon** (area 282 743 mm², within
  0.03 % of `πD²/4`); GSD uses the exact circle, so allow the 1.5 %
  section-property tolerance.
- **Concrete:** `f'c = f_ck = 30 MPa`. **Rebar:** `f_y = 500 MPa`.
- **Reinforcement:** 8 × 25 mm bars on a 500 mm-diameter circle
  (radius 250 mm, i.e. 50 mm cover), at 45° spacing starting on the +z
  axis. **Spiral** transverse steel (φ cap 0.85).

| Bar | z [mm] | y [mm] | A [mm²] |
|--:|--:|--:|--:|
| 1 | +250.0 | 0 | 490.9 |
| 2 | +176.8 | −176.8 | 490.9 |
| 3 | 0 | −250.0 | 490.9 |
| 4 | −176.8 | −176.8 | 490.9 |
| 5 | −250.0 | 0 | 490.9 |
| 6 | −176.8 | +176.8 | 490.9 |
| 7 | 0 | +250.0 | 490.9 |
| 8 | +176.8 | +176.8 | 490.9 |

## S3 — Doubly-reinforced RC beam, 300 × 600 mm (moment-curvature)

- **Geometry:** rectangle `b = 300 mm` (z) × `h = 600 mm` (y), centroid
  at origin.
- **Concrete:** `f'c = 30 MPa`. **Rebar:** `f_y = 500 MPa`.
- **Reinforcement:** bottom 3 × 25 mm (`A = 490.9 mm²`), top 2 × 20 mm
  (`A = 314.2 mm²`); clear cover 40 mm.

| Bar | z [mm] | y [mm] | A [mm²] | size |
|--:|--:|--:|--:|:--|
| 1 | −110 | −260 | 490.9 | 25 mm |
| 2 | 0 | −260 | 490.9 | 25 mm |
| 3 | +110 | −260 | 490.9 | 25 mm |
| 4 | −110 | +260 | 314.2 | 20 mm |
| 5 | +110 | +260 | 314.2 | 20 mm |

Analysis: moment-curvature at `P = 0` (see constitutive block above).

## S4 — L-shaped RC pier, 800 × 800 mm, 400 mm legs (asymmetric biaxial)

- **Geometry:** an L formed by an 800 × 800 mm square with the
  top-right 400 × 400 mm quadrant removed, then **recentred to its own
  centroid**. Vertex list in the centroidal frame (mm):

  `(-333.3, -333.3) → (466.7, -333.3) → (466.7, 66.7) → (66.7, 66.7) → (66.7, 466.7) → (-333.3, 466.7)` → close.

  Gross area 480 000 mm²; centroid 333.3 mm above the bottom fibre
  (asymmetric).
- **Concrete:** `f'c = f_ck = 30 MPa`. **Rebar:** `f_y = 500 MPa`.
- **Reinforcement:** 12 × 25 mm bars (`A = 490.9 mm²`) at these exact
  coordinates (50 mm cover; place them explicitly in GSD):

| Bar | z [mm] | y [mm] | | Bar | z [mm] | y [mm] |
|--:|--:|--:|--|--:|--:|--:|
| 1 | −283.3 | −283.3 | | 7 | +31.3 | +31.3 |
| 2 | −283.3 | −51.8 | | 8 | +259.0 | +16.7 |
| 3 | −283.3 | +179.8 | | 9 | +416.7 | −57.2 |
| 4 | −283.3 | +411.3 | | 10 | +411.3 | −283.3 |
| 5 | −57.2 | +416.7 | | 11 | +179.8 | −283.3 |
| 6 | +16.7 | +259.0 | | 12 | −51.8 | −283.3 |

Interaction reported on the θ = 0 slice (compression on the +y face).

## S5 — Prestressed concrete girder, 400 × 900 mm

- **Geometry:** rectangle `b = 400 mm` (z) × `h = 900 mm` (y), centroid
  at origin (`y ∈ [-450, +450]`).
- **Concrete:** `f'c = 40 MPa`. **Mild steel:** `f_y = 500 MPa`.
- **Mild reinforcement:** 2 × 16 mm top bars (`A = 201.1 mm²`), cover
  50 mm, at `y = +400 mm`, `z = ±160 mm`.
- **Prestress:** 6 × 0.6″ strands (`A_p = 140 mm²` each), Grade 270
  (`f_pu = 1860 MPa`; bilinear `f_py = 1675 MPa`, `E_p = 195 GPa`,
  `b = 0.005`), effective prestress after losses `f_pe = 1100 MPa`, all
  at `y = −380 mm`:

| Tendon | z [mm] | y [mm] | A_p [mm²] | f_pe [MPa] |
|--:|--:|--:|--:|--:|
| 1 | −150 | −380 | 140 | 1100 |
| 2 | −90 | −380 | 140 | 1100 |
| 3 | −30 | −380 | 140 | 1100 |
| 4 | +30 | −380 | 140 | 1100 |
| 5 | +90 | −380 | 140 | 1100 |
| 6 | +150 | −380 | 140 | 1100 |

Reported: axial landmarks (`P_o` including tendon `f_pu`, pure tension)
in both codes, plus moment-curvature (`M_cr` including the prestress
benefit, `M_u`). The flexural interaction landmarks are omitted for S5
because the tension face carries tendons rather than mild bars.

---

## Computed reference values (femsolver)

Snapshot below; regenerate any time with example 78. The **Midas GSD**
and **Diff %** columns are filled from the parallel models.

### S1 — Rect column 400×600 (8-25M)

| Code | Quantity | Units | femsolver | Midas GSD | Diff % | Tol % |
|---|---|---|--:|--:|--:|--:|
| - | Gross area A_g | mm² | 240 000 |  |  | 1.0 |
| - | I_zz | mm⁴ | 7.200e9 |  |  | 1.0 |
| - | I_yy | mm⁴ | 3.200e9 |  |  | 1.0 |
| - | Centroid above bottom fibre | mm | 300 |  |  | 1.0 |
| AASHTO LRFD 2024 | P_o squash (nominal) | kN | 7983 |  |  | 1.0 |
| AASHTO LRFD 2024 | P_n,max (0.80·P_o) | kN | 6387 |  |  | 1.0 |
| AASHTO LRFD 2024 | M_n at P=0 (nominal) | kN·m | 498.5 |  |  | 2.0 |
| AASHTO LRFD 2024 | φ·M_n at P=0 (design) | kN·m | 448.7 |  |  | 2.0 |
| AASHTO LRFD 2024 | Balanced P_b | kN | 2510 |  |  | 2.0 |
| AASHTO LRFD 2024 | Balanced M_b (nominal) | kN·m | 804.9 |  |  | 2.0 |
| Eurocode 2 | P_o squash | kN | 5721 |  |  | 1.0 |
| Eurocode 2 | M_n at P=0 | kN·m | 431.8 |  |  | 2.0 |
| Eurocode 2 | Balanced P_b | kN | 1900 |  |  | 2.0 |
| Eurocode 2 | Balanced M_b | kN·m | 617.1 |  |  | 2.0 |

### S2 — Circular column D600 (8-25M spiral)

| Code | Quantity | Units | femsolver | Midas GSD | Diff % | Tol % |
|---|---|---|--:|--:|--:|--:|
| - | Gross area A_g | mm² | 282 743 |  |  | 1.5 |
| - | I_zz = I_yy | mm⁴ | 6.362e9 |  |  | 1.5 |
| AASHTO LRFD 2024 | P_o squash (nominal) | kN | 9073 |  |  | 1.0 |
| AASHTO LRFD 2024 | P_n,max (0.85·P_o) | kN | 7712 |  |  | 1.0 |
| AASHTO LRFD 2024 | M_n at P=0 (nominal) | kN·m | 431.2 |  |  | 2.0 |
| AASHTO LRFD 2024 | φ·M_n at P=0 (design) | kN·m | 388.1 |  |  | 2.0 |
| AASHTO LRFD 2024 | Balanced P_b | kN | 2800 |  |  | 2.0 |
| Eurocode 2 | P_o squash | kN | 6447 |  |  | 1.0 |
| Eurocode 2 | M_n at P=0 | kN·m | 368.1 |  |  | 2.0 |

### S3 — RC beam 300×600 (moment-curvature)

| Code | Quantity | Units | femsolver | Midas GSD | Diff % | Tol % |
|---|---|---|--:|--:|--:|--:|
| - | Gross area A_g | mm² | 180 000 |  |  | 1.0 |
| - | I_zz | mm⁴ | 5.400e9 |  |  | 1.0 |
| M-φ | Cracking moment M_cr | kN·m | 61.1 |  |  | 3.0 |
| M-φ | First-yield moment M_y | kN·m | 373.7 |  |  | 3.0 |
| M-φ | Ultimate moment M_u | kN·m | 404.4 |  |  | 3.0 |
| M-φ | Yield curvature κ_y | 1/m | 0.0070 |  |  | 3.0 |
| M-φ | Curvature ductility μ_φ | - | 4.86 |  |  | 3.0 |

### S4 — L-pier 800×800 t400 (12-25M)

| Code | Quantity | Units | femsolver | Midas GSD | Diff % | Tol % |
|---|---|---|--:|--:|--:|--:|
| - | Gross area A_g | mm² | 480 000 |  |  | 1.0 |
| - | I_zz = I_yy | mm⁴ | 2.347e10 |  |  | 1.0 |
| - | Centroid above bottom fibre | mm | 333.3 |  |  | 1.0 |
| AASHTO LRFD 2024 | P_o squash (nominal) | kN | 15 040 |  |  | 1.0 |
| AASHTO LRFD 2024 | P_n,max (0.80·P_o) | kN | 12 030 |  |  | 1.0 |
| AASHTO LRFD 2024 | M_n at P=0 (nominal) | kN·m | 1086 |  |  | 2.0 |
| AASHTO LRFD 2024 | φ·M_n at P=0 (design) | kN·m | 977.8 |  |  | 2.0 |
| Eurocode 2 | P_o squash | kN | 10 620 |  |  | 1.0 |
| Eurocode 2 | M_n at P=0 | kN·m | 911.7 |  |  | 2.0 |

### S5 — PSC girder 400×900 (6×0.6″ strand)

| Code | Quantity | Units | femsolver | Midas GSD | Diff % | Tol % |
|---|---|---|--:|--:|--:|--:|
| - | Gross area A_g | mm² | 360 000 |  |  | 1.0 |
| - | I_zz | mm⁴ | 2.430e10 |  |  | 1.0 |
| AASHTO LRFD 2024 | P_o squash (incl. tendon f_pu) | kN | 13 820 |  |  | 1.0 |
| AASHTO LRFD 2024 | P pure tension | kN | −1617 |  |  | 2.0 |
| Eurocode 2 | P_o squash | kN | 9723 |  |  | 1.0 |
| M-φ | Cracking moment M_cr | kN·m | 701.5 |  |  | 3.0 |
| M-φ | Ultimate moment M_u | kN·m | 1111 |  |  | 3.0 |

## Expected differences (read before flagging a mismatch)

- **Circle faceting (S2):** femsolver uses a 120-gon; GSD uses an exact
  circle. Section properties differ ~0.03 %, moment capacities well
  under the 1.5 % / 2 % tolerances.
- **EC2 axial cap:** femsolver reports the uncapped squash `P_o` for
  EC2 (no `0.80·P_o` cap — EC2 governs pure axial by minimum
  eccentricity instead). Compare `P_o`, not a capped value, for EC2.
- **Strand `f_pu` (S5):** the engine reads the strand ultimate off the
  bilinear curve at 2 % strain (≈ 1686 MPa, not the nominal 1860 MPa).
  Use the **same bilinear strand curve** in GSD so `P_o` matches.
- **Balanced point:** defined here as extreme-tension steel strain
  `ε_t = ε_y` (nominal `f_y` for AASHTO, `f_yd = f_yk/1.15` for EC2).
  If GSD reports the balanced point at a slightly different strain
  definition, expect a small offset — compare against the same
  definition where possible.
- **Nominal vs design:** AASHTO rows split nominal and `φ`-reduced; EC2
  rows are already design-level (factors in the stress block). Make
  sure GSD is set to the matching output.
