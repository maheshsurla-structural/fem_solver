"""Standardized Section-Designer verification benchmarks.

A curated, **fully-specified** set of cross-sections whose Section
Designer results (section properties, biaxial P-M-M interaction
landmarks, and moment-curvature milestones) are intended to be
reproduced independently in a commercial general-section designer
(e.g. **Midas Gen / Midas GSD**) and compared side by side before a
release.

Design codes
------------
Each interaction section is evaluated in **two** codes, matching the
codes the parallel GSD models will be built in:

* **AASHTO LRFD BDS (10th ed., 2024)** -- nominal and phi-reduced
  (Art. 5.5.4.2: phi = 0.75 compression-controlled tied/spiral, 0.90
  tension-controlled RC, 1.00 tension-controlled PC). For
  normal-strength concrete the *nominal* stress block is identical to
  the ACI 318 Whitney block.
* **Eurocode 2** (EN 1992-1-1) -- design values with partial factors
  gamma_c = 1.5, gamma_s = 1.15 built into the stress block.

Moment-curvature is constitutive-driven (not code-specific): the exact
concrete (Kent-Park) and steel (bilinear) laws are pinned down in the
spec sheet so the same curves can be entered in GSD.

Conventions (must be matched in GSD)
------------------------------------
* **Frame**: section-local, origin at the **geometric centroid**;
  ``y`` vertical (depth / strong axis), ``z`` horizontal (width).
* **Moments**: ``M_z`` is bending about the local z-axis (strong-axis
  bending for a taller-than-wide section); ``M_y`` about the local
  y-axis. Reported about the centroid.
* **Axial sign**: compression positive (AASHTO/ACI convention).
* **Rebar / tendon coordinates**: measured from the centroid, in the
  same (z, y) frame.

Display units (Theme: kN, mm, MPa)
----------------------------------
Force ``kN``; moment ``kN.m``; length ``mm``; area ``mm^2``; second
moment ``mm^4``; stress ``MPa``; curvature ``1/m``; ductility is
dimensionless. The engine works in strict SI internally; conversion to
display units happens only in :class:`VerificationItem`.

The reference ("Midas GSD") column is intentionally **blank** -- fill
it in from the parallel model and re-run :func:`compare_items` (or the
``examples/78_section_designer_verification.py`` driver) to get the
percentage differences and pass/fail against the documented tolerance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from femsolver.design.concrete import (
    biaxial_pmm_point,
    biaxial_pmm_point_aashto,
    biaxial_pmm_point_ec2,
    biaxial_pmm_surface_aashto,
    biaxial_pmm_surface_ec2,
    moment_curvature,
)
from femsolver.design.concrete.biaxial import BiaxialPMMPoint
from femsolver.materials.uniaxial import ConcreteKentPark, UniaxialBilinear
from femsolver.sections import (
    PrestressTendon,
    RebarBar,
    ReinforcementLayout,
    Section,
    TendonLayout,
    circular_section,
    custom_polygon_section,
    rc_rectangular_section,
)
from femsolver.design.concrete import ConcreteMaterial


# ============================================================ display units

N_TO_KN = 1.0e-3
NM_TO_KNM = 1.0e-3
M_TO_MM = 1.0e3
M2_TO_MM2 = 1.0e6
M4_TO_MM4 = 1.0e12
PA_TO_MPA = 1.0e-6

#: Steel elastic modulus used throughout (Pa). Matches the engine
#: default (ACI 20.2.2.2 / EC2 3.2.7): 200 GPa.
E_S = 200.0e9


# ============================================================ result item

@dataclass
class VerificationItem:
    """One scalar quantity to compare against the parallel GSD model.

    ``computed`` is in the display units named by ``units``. ``gsd`` is
    left ``None`` until the user fills it in from the Midas GSD run;
    :func:`compare_items` then populates ``diff_pct`` / ``passed``.
    """

    section_id: str
    section_name: str
    code: str                    # "-", "AASHTO LRFD 2024", "Eurocode 2"
    quantity: str                # human label, e.g. "P_o (squash)"
    units: str
    computed: float
    tol_pct: float = 2.0
    gsd: Optional[float] = None
    note: str = ""

    # populated by compare_items()
    diff_pct: Optional[float] = None
    passed: Optional[bool] = None


def compare_items(items: list[VerificationItem]) -> list[VerificationItem]:
    """Fill ``diff_pct`` / ``passed`` for every item whose ``gsd`` is set.

    Percentage difference is ``100 * (computed - gsd) / |gsd|`` (or the
    absolute computed value when ``gsd == 0``). ``passed`` is ``True``
    when ``|diff_pct| <= tol_pct``.
    """
    for it in items:
        if it.gsd is None:
            it.diff_pct = None
            it.passed = None
            continue
        if it.gsd == 0.0:
            it.diff_pct = abs(it.computed) * 100.0
        else:
            it.diff_pct = 100.0 * (it.computed - it.gsd) / abs(it.gsd)
        it.passed = abs(it.diff_pct) <= it.tol_pct
    return items


# ============================================================ landmark solvers

def _solve_c(
    point_fn: Callable[[float], BiaxialPMMPoint],
    key: Callable[[BiaxialPMMPoint], float],
    target: float,
    *,
    c_lo: float,
    c_hi: float,
    iters: int = 80,
) -> BiaxialPMMPoint:
    """Bisection on neutral-axis depth ``c`` so that ``key(point) ==
    target`` along the uniaxial (theta = 0) slice.

    ``key`` must be monotone increasing in ``c`` over ``[c_lo, c_hi]``
    (true for both ``P_n`` and the extreme-tension strain ``-epsilon_t``
    used here). Returns the converged interaction point.
    """
    p_lo = point_fn(c_lo)
    p_hi = point_fn(c_hi)
    f_lo = key(p_lo) - target
    f_hi = key(p_hi) - target
    if f_lo > 0:          # target below the achievable range -> clamp low
        return p_lo
    if f_hi < 0:          # target above the achievable range -> clamp high
        return p_hi
    p_mid = p_lo
    for _ in range(iters):
        c_mid = 0.5 * (c_lo + c_hi)
        p_mid = point_fn(c_mid)
        f_mid = key(p_mid) - target
        if abs(f_mid) < 1e-9 or (c_hi - c_lo) < 1e-9:
            break
        if f_mid < 0:
            c_lo = c_mid
        else:
            c_hi = c_mid
    return p_mid


@dataclass
class PMMLandmarks:
    """Key points on a uniaxial (strong-axis) interaction diagram."""

    P_o: float                   # squash load (N, +compression)
    P_n_max: float               # capped pure axial (N); NaN if not applicable
    M0_nominal: float            # pure-flexure nominal moment at P=0 (N.m)
    M0_phi: float                # phi-reduced pure-flexure moment (N.m)
    phi0: float                  # phi at the P=0 point
    P_bal: float                 # balanced axial load (N)
    M_bal_nominal: float         # balanced nominal moment (N.m)
    M_bal_phi: float             # phi-reduced balanced moment (N.m)


def strong_axis_landmarks(
    point_fn: Callable[[float], BiaxialPMMPoint],
    *,
    P_o: float,
    P_n_max: float,
    depth_y: float,
    eps_y: float,
) -> PMMLandmarks:
    """Extract squash / pure-flexure / balanced landmarks from a
    uniaxial interaction slice, given a ``point_fn(c) -> BiaxialPMMPoint``
    at ``theta = 0``.

    ``depth_y`` is the section depth in the y direction (m); ``eps_y``
    is the reinforcement yield strain (for the balanced point defined
    by extreme-tension strain == eps_y).
    """
    c_lo = 0.02 * depth_y
    c_hi = 3.0 * depth_y

    # Pure flexure: P_n = 0
    p0 = _solve_c(point_fn, lambda p: p.P_n, 0.0, c_lo=c_lo, c_hi=c_hi)
    # Balanced: extreme tension steel strain == eps_y  (key = -epsilon_t,
    # increasing in c because higher c -> less tension)
    pb = _solve_c(point_fn, lambda p: -p.epsilon_t, -eps_y,
                  c_lo=c_lo, c_hi=c_hi)
    return PMMLandmarks(
        P_o=P_o, P_n_max=P_n_max,
        M0_nominal=p0.M_nz, M0_phi=p0.phi_M_nz, phi0=p0.phi,
        P_bal=pb.P_n, M_bal_nominal=pb.M_nz, M_bal_phi=pb.phi_M_nz,
    )


# ============================================================ the sections

@dataclass
class SectionCase:
    """A fully-specified verification section plus its metadata."""

    section_id: str
    name: str
    section: Section
    f_c_prime: float             # concrete cylinder strength (Pa)
    f_y: float                   # reinforcement yield (Pa)
    kind: str                    # "column" | "beam" | "pier" | "psc"
    spiral: bool = False
    prestressed: bool = False
    note: str = ""


def _bar(area_mm2: float) -> float:
    """mm^2 -> m^2 bar area."""
    return area_mm2 * 1e-6


def rebar_area_from_dia(dia_mm: float) -> float:
    """Area (m^2) of a round bar of diameter ``dia_mm`` (mm)."""
    return math.pi / 4.0 * (dia_mm * 1e-3) ** 2


def _recenter_outline(outline: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Translate a polygon outline so its centroid is at the origin,
    matching the 'reference at centroid' convention GSD uses."""
    from shapely.geometry import Polygon

    c = Polygon(outline).centroid
    return [(z - c.x, y - c.y) for (z, y) in outline]


# ---- S1: rectangular RC column -------------------------------------

def build_rect_column() -> SectionCase:
    """S1 -- Rectangular tied RC column, 400 x 600 mm.

    * Concrete f'c = f_ck = 30 MPa.
    * Reinforcement f_y = 500 MPa, E_s = 200 GPa.
    * 8 x 25 mm bars (A = 490.9 mm^2 each): 3 top, 3 bottom, 2 side at
      mid-height. Clear cover to bar centroid = 50 mm.
    """
    fc, fy = 30e6, 500e6
    A25 = rebar_area_from_dia(25.0)
    rl = ReinforcementLayout.from_rectangular_layers(
        b=0.4, h=0.6,
        bottom_bars=[(A25, "25M")] * 3,
        top_bars=[(A25, "25M")] * 3,
        side_bars=[(A25, "25M", 0.0), (A25, "25M", 0.0)],
        bottom_cover=0.05, top_cover=0.05,
    )
    # place the two side bars on the +/- z faces at mid-height
    edge_z = 0.4 / 2.0 - 0.05
    side = [b for b in rl.bars if b.y == 0.0]
    if len(side) == 2:
        side[0].z, side[1].z = -edge_z, +edge_z
    sec = rc_rectangular_section(
        b=0.4, h=0.6, concrete=ConcreteMaterial(fc_prime=fc, fy=fy),
        reinforcement=rl, name="S1 Rect column 400x600 (8-25M)",
    )
    return SectionCase(
        section_id="S1", name=sec.name, section=sec,
        f_c_prime=fc, f_y=fy, kind="column", spiral=False,
        note="Tied column; 8-25M; cover 50 mm to bar centroid.",
    )


# ---- S2: circular spiral RC column ---------------------------------

def build_circular_column() -> SectionCase:
    """S2 -- Circular spiral RC column, 600 mm diameter.

    * Concrete f'c = f_ck = 30 MPa; f_y = 500 MPa.
    * 8 x 25 mm bars on a 500 mm-diameter bolt circle (cover 50 mm to
      bar centroid).
    * Circle approximated by a 120-gon; area within 0.03% of the exact
      circle (documented in the spec sheet).
    """
    fc, fy = 30e6, 500e6
    A25 = rebar_area_from_dia(25.0)
    sec = circular_section(
        D=0.6, material=ConcreteMaterial(fc_prime=fc, fy=fy),
        n_sides=120, name="S2 Circular column D600 (8-25M spiral)",
    )
    sec.reinforcement = ReinforcementLayout.from_perimeter(
        sec, n_bars=8, bar_area=A25, cover=0.05, designation="25M",
    )
    return SectionCase(
        section_id="S2", name=sec.name, section=sec,
        f_c_prime=fc, f_y=fy, kind="column", spiral=True,
        note="Spiral column; 8-25M on 500 mm circle; 120-gon geometry.",
    )


# ---- S3: rectangular RC beam (moment-curvature) --------------------

def build_rc_beam() -> SectionCase:
    """S3 -- Doubly-reinforced RC beam, 300 x 600 mm (for M-phi).

    * Concrete f'c = 30 MPa; f_y = 500 MPa.
    * Bottom 3 x 25 mm (A = 490.9 mm^2 each), top 2 x 20 mm
      (A = 314.2 mm^2 each). Cover 40 mm.
    """
    fc, fy = 30e6, 500e6
    A25 = rebar_area_from_dia(25.0)
    A20 = rebar_area_from_dia(20.0)
    rl = ReinforcementLayout.from_rectangular_layers(
        b=0.3, h=0.6,
        bottom_bars=[(A25, "25M")] * 3,
        top_bars=[(A20, "20M")] * 2,
        bottom_cover=0.04, top_cover=0.04,
    )
    sec = rc_rectangular_section(
        b=0.3, h=0.6, concrete=ConcreteMaterial(fc_prime=fc, fy=fy),
        reinforcement=rl, name="S3 RC beam 300x600 (3-25M / 2-20M)",
    )
    return SectionCase(
        section_id="S3", name=sec.name, section=sec,
        f_c_prime=fc, f_y=fy, kind="beam",
        note="Doubly reinforced; M-phi at P=0.",
    )


# ---- S4: L-shaped RC pier (asymmetric biaxial) ---------------------

def build_L_pier() -> SectionCase:
    """S4 -- L-shaped RC pier, 800 x 800 mm with 400 mm legs.

    Outer 800 x 800 mm square with the top-right 400 x 400 mm quadrant
    removed -> an L. Recentred to its own centroid. Reinforced with 12
    x 25 mm bars auto-distributed around the perimeter at 50 mm cover.

    * Concrete f'c = 30 MPa; f_y = 500 MPa.
    """
    fc, fy = 30e6, 500e6
    A25 = rebar_area_from_dia(25.0)
    # L outline (z, y), corner at origin, 800 leg, 400 thickness
    raw = [
        (0.0, 0.0), (0.8, 0.0), (0.8, 0.4),
        (0.4, 0.4), (0.4, 0.8), (0.0, 0.8),
    ]
    outline = _recenter_outline(raw)
    sec = custom_polygon_section(
        outline=outline, material=ConcreteMaterial(fc_prime=fc, fy=fy),
        name="S4 L-pier 800x800 t400 (12-25M)",
    )
    sec.reinforcement = ReinforcementLayout.from_perimeter(
        sec, n_bars=12, bar_area=A25, cover=0.05, designation="25M",
    )
    return SectionCase(
        section_id="S4", name=sec.name, section=sec,
        f_c_prime=fc, f_y=fy, kind="pier",
        note="Asymmetric L; 12-25M perimeter; biaxial (theta=0 and 90).",
    )


# ---- S5: prestressed concrete girder -------------------------------

def build_psc_girder() -> SectionCase:
    """S5 -- Rectangular PSC girder, 400 x 900 mm.

    * Concrete f'c = 40 MPa; mild steel f_y = 500 MPa.
    * 2 x 16 mm top bars (A = 201.1 mm^2 each), cover 50 mm.
    * 6 x 0.6" strands (A_p = 140 mm^2 each), Grade 270
      (f_pu = 1860 MPa), effective prestress after losses
      f_pe = 1100 MPa, at y = -380 mm (near the soffit).
    """
    fc, fy = 40e6, 500e6
    A16 = rebar_area_from_dia(16.0)
    rl = ReinforcementLayout.from_rectangular_layers(
        b=0.4, h=0.9,
        top_bars=[(A16, "16M")] * 2,
        top_cover=0.05,
    )
    sec = rc_rectangular_section(
        b=0.4, h=0.9, concrete=ConcreteMaterial(fc_prime=fc, fy=fy),
        reinforcement=rl, name="S5 PSC girder 400x900 (6x0.6in strand)",
    )
    strand_mat = UniaxialBilinear(E=195e9, sigma_y=1675e6, b=0.005)
    A_strand = 140e-6
    z_positions = [-0.15, -0.09, -0.03, 0.03, 0.09, 0.15]
    sec.prestress = TendonLayout(tendons=[
        PrestressTendon(z=z, y=-0.38, area=A_strand, material=strand_mat,
                        f_pe=1100e6, designation="0.6in Gr270")
        for z in z_positions
    ])
    return SectionCase(
        section_id="S5", name=sec.name, section=sec,
        f_c_prime=fc, f_y=fy, kind="psc", prestressed=True,
        note="6 bonded strands at y=-380 mm; f_pe=1100 MPa; M-phi + PMM.",
    )


ALL_SECTION_BUILDERS: list[Callable[[], SectionCase]] = [
    build_rect_column,
    build_circular_column,
    build_rc_beam,
    build_L_pier,
    build_psc_girder,
]


# ============================================================ item assembly

# Default expected-agreement tolerances (percent) by quantity class.
TOL_SECTION_PROP = 1.0
TOL_AXIAL = 1.0
TOL_NOMINAL_M = 2.0
TOL_PHI_M = 2.0
TOL_MPHI = 3.0
TOL_CIRCLE = 1.5     # slightly looser: polygon-facetting vs exact circle


def _section_property_items(case: SectionCase) -> list[VerificationItem]:
    g = case.section.geometry
    cz, cy = g.centroid
    miny = g.polygon.bounds[1]
    tol = TOL_CIRCLE if case.kind == "column" and case.spiral else TOL_SECTION_PROP
    return [
        VerificationItem(case.section_id, case.name, "-",
                         "Gross area A_g", "mm^2",
                         g.area * M2_TO_MM2, tol_pct=tol),
        VerificationItem(case.section_id, case.name, "-",
                         "I_zz (about centroidal z)", "mm^4",
                         g.I_zz * M4_TO_MM4, tol_pct=tol),
        VerificationItem(case.section_id, case.name, "-",
                         "I_yy (about centroidal y)", "mm^4",
                         g.I_yy * M4_TO_MM4, tol_pct=tol),
        VerificationItem(case.section_id, case.name, "-",
                         "Centroid above bottom fibre", "mm",
                         (cy - miny) * M_TO_MM, tol_pct=tol,
                         note="Reveals asymmetry (h/2 for symmetric)."),
    ]


def _pmm_items_for_code(
    case: SectionCase, code: str,
) -> list[VerificationItem]:
    """Build the P-M-M landmark items for one section in one code."""
    sec = case.section
    depth_y = sec.geometry.polygon.bounds[3] - sec.geometry.polygon.bounds[1]

    if code == "AASHTO LRFD 2024":
        surf = biaxial_pmm_surface_aashto(
            sec, f_c_prime=case.f_c_prime, f_y=case.f_y,
            spiral=case.spiral, prestressed=case.prestressed,
            n_angles=4, n_depths=8,
        )

        def point_fn(c: float) -> BiaxialPMMPoint:
            return biaxial_pmm_point_aashto(
                sec, 0.0, c, f_c_prime=case.f_c_prime, f_y=case.f_y,
                spiral=case.spiral, prestressed=case.prestressed,
            )
        P_n_max = surf.P_n_max
        applic_pnmax = True
    elif code == "Eurocode 2":
        surf = biaxial_pmm_surface_ec2(
            sec, f_ck=case.f_c_prime, f_yk=case.f_y,
            n_angles=4, n_depths=8,
        )

        def point_fn(c: float) -> BiaxialPMMPoint:
            return biaxial_pmm_point_ec2(
                sec, 0.0, c, f_ck=case.f_c_prime, f_yk=case.f_y,
            )
        P_n_max = float("nan")
        applic_pnmax = False   # EC2 uses min-eccentricity, not a 0.80 cap
    else:
        raise ValueError(f"unknown code {code!r}")

    # Balanced point uses the code-consistent yield strain: nominal
    # f_y for AASHTO, design f_yd = f_yk/1.15 for EC2.
    eps_y = (case.f_y / 1.15 if code == "Eurocode 2" else case.f_y) / E_S

    lm = strong_axis_landmarks(
        point_fn, P_o=surf.P_o, P_n_max=P_n_max,
        depth_y=depth_y, eps_y=eps_y,
    )

    sid, nm = case.section_id, case.name
    items = [
        VerificationItem(sid, nm, code, "P_o squash (nominal)", "kN",
                         lm.P_o * N_TO_KN, tol_pct=TOL_AXIAL),
    ]
    if applic_pnmax:
        cap = "0.85" if case.spiral else "0.80"
        items.append(VerificationItem(
            sid, nm, code, f"P_n,max ({cap}*P_o)", "kN",
            lm.P_n_max * N_TO_KN, tol_pct=TOL_AXIAL,
            note="AASHTO 5.6.4.4 axial cap."))
    items += [
        VerificationItem(sid, nm, code, "M_n at P=0 (nominal)", "kN.m",
                         lm.M0_nominal * NM_TO_KNM, tol_pct=TOL_NOMINAL_M),
        VerificationItem(sid, nm, code, "phi*M_n at P=0 (design)", "kN.m",
                         lm.M0_phi * NM_TO_KNM, tol_pct=TOL_PHI_M,
                         note=f"phi={lm.phi0:.3f} at P=0."),
        VerificationItem(sid, nm, code, "Balanced P_b (eps_t=eps_y)", "kN",
                         lm.P_bal * N_TO_KN, tol_pct=TOL_NOMINAL_M),
        VerificationItem(sid, nm, code, "Balanced M_b (nominal)", "kN.m",
                         lm.M_bal_nominal * NM_TO_KNM, tol_pct=TOL_NOMINAL_M),
    ]
    return items


def _psc_axial_items(case: SectionCase) -> list[VerificationItem]:
    """Axial-capacity landmarks for a prestressed section (P_o and pure
    tension), reported in both codes. The flexural-interaction
    landmarks are omitted for PSC because the extreme-tension *rebar*
    strain (which drives phi and the balanced point) is not meaningful
    when the tension face carries tendons rather than mild bars --
    those are verified through moment-curvature instead."""
    sec = case.section
    out: list[VerificationItem] = []
    surf_a = biaxial_pmm_surface_aashto(
        sec, f_c_prime=case.f_c_prime, f_y=case.f_y,
        prestressed=True, n_angles=4, n_depths=6,
    )
    surf_e = biaxial_pmm_surface_ec2(
        sec, f_ck=case.f_c_prime, f_yk=case.f_y, n_angles=4, n_depths=6,
    )
    for code, surf in (("AASHTO LRFD 2024", surf_a), ("Eurocode 2", surf_e)):
        out.append(VerificationItem(
            case.section_id, case.name, code,
            "P_o squash (incl. tendon f_pu)", "kN",
            surf.P_o * N_TO_KN, tol_pct=TOL_AXIAL))
        out.append(VerificationItem(
            case.section_id, case.name, code,
            "P pure tension (rebar+tendon)", "kN",
            surf.P_pure_tension * N_TO_KN, tol_pct=TOL_NOMINAL_M))
    return out


def _moment_curvature_items(case: SectionCase) -> list[VerificationItem]:
    """M-phi milestones (constitutive-driven; see spec sheet for the
    exact concrete / steel laws)."""
    concrete_uni = ConcreteKentPark(
        fpc=case.f_c_prime, eps_c0=0.002,
        fpcu=0.4 * case.f_c_prime, eps_cu=0.0035,
    )
    steel_uni = UniaxialBilinear(E=E_S, sigma_y=case.f_y, b=0.01)
    f_rupture = 0.62 * math.sqrt(case.f_c_prime / 1e6) * 1e6   # ~ACI/AASHTO f_r
    res = moment_curvature(
        case.section, P_target=0.0,
        concrete_uniaxial=concrete_uni, steel_uniaxial=steel_uni,
        kappa_max=0.06, n_steps=60, f_y=case.f_y, E_s=E_S,
        f_rupture=f_rupture,
    )
    sid, nm = case.section_id, case.name
    code = "M-phi (Kent-Park + bilinear)"
    items = [
        VerificationItem(sid, nm, code, "Cracking moment M_cr", "kN.m",
                         (res.M_cr or 0.0) * NM_TO_KNM, tol_pct=TOL_MPHI,
                         note=f"f_r={f_rupture/1e6:.2f} MPa."),
        VerificationItem(sid, nm, code, "Ultimate moment M_u", "kN.m",
                         (res.M_u or 0.0) * NM_TO_KNM, tol_pct=TOL_MPHI),
    ]
    if res.M_y is not None:
        items.append(VerificationItem(
            sid, nm, code, "First-yield moment M_y", "kN.m",
            res.M_y * NM_TO_KNM, tol_pct=TOL_MPHI))
    if res.kappa_y is not None:
        items.append(VerificationItem(
            sid, nm, code, "Yield curvature kappa_y", "1/m",
            res.kappa_y, tol_pct=TOL_MPHI))
    if res.mu_phi is not None:
        items.append(VerificationItem(
            sid, nm, code, "Curvature ductility mu_phi", "-",
            res.mu_phi, tol_pct=TOL_MPHI))
    return items


def items_for_case(case: SectionCase) -> list[VerificationItem]:
    """All verification items for one section case."""
    items = _section_property_items(case)
    if case.kind in ("column", "pier"):
        items += _pmm_items_for_code(case, "AASHTO LRFD 2024")
        items += _pmm_items_for_code(case, "Eurocode 2")
    if case.kind == "psc":
        items += _psc_axial_items(case)
    if case.kind in ("beam", "psc"):
        items += _moment_curvature_items(case)
    return items


def all_verification_items() -> list[VerificationItem]:
    """Build every verification item across all standardized sections."""
    items: list[VerificationItem] = []
    for build in ALL_SECTION_BUILDERS:
        items += items_for_case(build())
    return items


# ============================================================ reporters

import csv as _csv

CSV_COLUMNS = [
    "section_id", "section", "code", "quantity", "units",
    "femsolver", "midas_gsd", "diff_pct", "tol_pct", "passed", "note",
]


def _num(x: float) -> str:
    """Readable formatting: fixed for moderate magnitudes, scientific
    for very large / very small."""
    if x == 0 or (1e-3 <= abs(x) < 1e6):
        return f"{x:.4g}"
    return f"{x:.4e}"


def export_csv(items: list[VerificationItem], path: str) -> None:
    """Write the verification table to CSV with a blank ``midas_gsd``
    column for the user to fill from the parallel model."""
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = _csv.writer(fp)
        w.writerow(CSV_COLUMNS)
        for it in items:
            w.writerow([
                it.section_id, it.section_name, it.code, it.quantity,
                it.units, f"{it.computed:.6g}",
                "" if it.gsd is None else f"{it.gsd:.6g}",
                "" if it.diff_pct is None else f"{it.diff_pct:.3f}",
                f"{it.tol_pct:.2f}",
                "" if it.passed is None else ("yes" if it.passed else "no"),
                it.note,
            ])


def load_gsd_csv(
    items: list[VerificationItem], path: str,
) -> list[VerificationItem]:
    """Read a filled CSV back, copy the ``midas_gsd`` values into the
    matching items (keyed on section_id + code + quantity), and
    re-run :func:`compare_items`. Rows with a blank ``midas_gsd`` are
    left pending."""
    idx = {
        (it.section_id, it.code, it.quantity): it for it in items
    }
    with open(path, newline="", encoding="utf-8") as fp:
        for row in _csv.DictReader(fp):
            raw = (row.get("midas_gsd") or "").strip()
            if not raw:
                continue
            it = idx.get(
                (row["section_id"], row["code"], row["quantity"])
            )
            if it is not None:
                try:
                    it.gsd = float(raw)
                except ValueError:
                    pass
    return compare_items(items)


def to_markdown(items: list[VerificationItem]) -> str:
    """Render the items as GitHub-flavoured Markdown tables, one per
    section, with a blank Midas GSD column."""
    lines: list[str] = []
    cur: Optional[str] = None
    for it in items:
        if it.section_id != cur:
            cur = it.section_id
            lines.append("")
            lines.append(f"### {it.section_id} — {it.section_name}")
            lines.append("")
            lines.append(
                "| Code | Quantity | Units | femsolver | Midas GSD "
                "| Diff % | Tol % |"
            )
            lines.append("|---|---|---|--:|--:|--:|--:|")
        gsd = "" if it.gsd is None else _num(it.gsd)
        diff = "" if it.diff_pct is None else f"{it.diff_pct:+.2f}"
        lines.append(
            f"| {it.code} | {it.quantity} | {it.units} "
            f"| {_num(it.computed)} | {gsd} | {diff} | {it.tol_pct:.1f} |"
        )
    return "\n".join(lines) + "\n"


def format_console(items: list[VerificationItem]) -> str:
    """Compact fixed-width console table grouped by section."""
    lines: list[str] = []
    cur: Optional[str] = None
    for it in items:
        if it.section_id != cur:
            cur = it.section_id
            lines.append("")
            lines.append(f"== {it.section_id}: {it.section_name} "
                         + "=" * max(0, 60 - len(it.section_name)))
            lines.append(f"  {'Code':<18}{'Quantity':<34}"
                         f"{'femsolver':>14}  {'GSD':>12}  {'Diff%':>7}")
        gsd = "" if it.gsd is None else _num(it.gsd)
        diff = "" if it.diff_pct is None else f"{it.diff_pct:+.2f}"
        code = (it.code[:16] + "..") if len(it.code) > 18 else it.code
        q = f"{it.quantity} [{it.units}]"
        lines.append(f"  {code:<18}{q[:34]:<34}"
                     f"{_num(it.computed):>14}  {gsd:>12}  {diff:>7}")
    return "\n".join(lines) + "\n"
