"""Rectangular beam shear strength per ACI 318-19 Ch. 22.5 + 9.6.3 + 9.7.6.2.

Implements the standard one-way shear design for non-prestressed
rectangular concrete members:

V_n = V_c + V_s              (ACI 22.5.1)
V_u <= φ V_n,   φ = 0.75     (Table 21.2.1)

Concrete contribution (ACI 22.5.5.1 -- simplified form):

    V_c = 0.17 · λ · √f_c'[MPa] · b_w · d        (in N when b, d in m)

This is the common simplified form used when minimum shear
reinforcement is provided (A_v >= A_v,min) -- ACI 22.5.5.1 case (a).
More elaborate ρ_w-dependent forms in Table 22.5.5.1 are documented as
future extensions.

Stirrup contribution (ACI 22.5.10.5.3 -- vertical stirrups):

    V_s = A_v · f_yt · d / s

where A_v = total area of stirrup legs at a section and s = stirrup
spacing along the beam axis.

Maximum stirrup spacing (ACI 9.7.6.2.2):

* s_max = min(d/2, 600 mm)           if V_s <= 0.33 √f_c'[MPa] b_w d  (in N)
* s_max = min(d/4, 300 mm)           if V_s >  0.33 √f_c'[MPa] b_w d

Minimum shear reinforcement (ACI 9.6.3.3, SI form):

    (A_v / s)_min = max(0.062 √f_c'[MPa] · b_w / f_yt[MPa],
                          0.35 · b_w / f_yt[MPa])

required where V_u > 0.5 φ V_c per ACI 9.6.3.1.

Inputs are SI: lengths in m, stresses in Pa, forces in N. The
empirical coefficients are applied with internal MPa / mm
conversions where required.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from femsolver.design.concrete.section import (
    ConcreteSection,
    PhiFactors,
)


PHI_SHEAR = 0.75       # ACI 318-19 Table 21.2.1
LAMBDA_NW = 1.0        # ACI 19.2.4.2: normal-weight concrete
S_MAX_LOW_HARD = 0.600   # 600 mm max spacing limit
S_MAX_HIGH_HARD = 0.300  # 300 mm max spacing limit


# ============================================================ result

@dataclass
class ShearCheck:
    """Result of a beam shear-strength evaluation.

    Attributes
    ----------
    V_n : float
        Nominal shear capacity = V_c + V_s (N).
    phi : float
        Strength-reduction factor (= 0.75).
    phi_V_n : float
        Design shear capacity (N).
    V_c : float
        Concrete contribution per 22.5.5.1 (N).
    V_s : float
        Stirrup contribution per 22.5.10.5.3 (N).
    s_max : float
        Maximum allowable stirrup spacing along beam axis (m), per
        9.7.6.2.2.
    s_actual : float
        Stirrup spacing from the rebar layout (m).
    Av_min_per_s : float
        Minimum required ``A_v / s`` per 9.6.3.3 (m²/m).
    Av_per_s_actual : float
        Provided ``A_v / s`` (m²/m).
    spacing_ok : bool
        Whether ``s_actual <= s_max``.
    min_reinforcement_ok : bool
        Whether the provided A_v/s satisfies the minimum.
    notes : str
        Warnings about spacing limits, min reinforcement, demand vs
        capacity, etc.
    """

    V_n: float
    phi: float
    phi_V_n: float
    V_c: float
    V_s: float
    s_max: float
    s_actual: float
    Av_min_per_s: float
    Av_per_s_actual: float
    spacing_ok: bool
    min_reinforcement_ok: bool
    notes: str = ""


@dataclass
class ShearDesign:
    """Result of forward shear design: given V_u, output required
    stirrup spacing.

    Attributes
    ----------
    V_u : float
        Factored shear demand (N).
    V_c : float
        Concrete contribution (N).
    V_s_required : float
        Required stirrup contribution = V_u/φ - V_c (N). Can be
        negative if V_c alone is sufficient.
    Av : float
        Total stirrup leg area = ``rebar.Av`` (m²).
    s_required : float
        Required spacing along beam axis (m). ``inf`` if no stirrups
        needed beyond minimum.
    s_max : float
        Maximum allowable spacing per 9.7.6.2.2 (m).
    s_min_reinforcement : float
        Spacing that gives ``A_v / s = (A_v/s)_min`` (m).
    s_recommended : float
        ``min(s_required, s_max, s_min_reinforcement)`` -- the spacing
        you should actually use.
    notes : str
        Design-decision notes.
    """

    V_u: float
    V_c: float
    V_s_required: float
    Av: float
    s_required: float
    s_max: float
    s_min_reinforcement: float
    s_recommended: float
    notes: str = ""


# ============================================================ helpers

def _Vc_simplified(fc_prime: float, b: float, d: float) -> float:
    """V_c per ACI 22.5.5.1 simplified form:

        V_c = 0.17 · λ · √f_c'[MPa] · b · d           (N, with b·d in m²)

    Returns V_c in **newtons**.
    """
    fc_MPa = fc_prime / 1.0e6
    # 0.17 √(f_c'[MPa]) is in MPa. Multiply by b[mm]·d[mm] gives N.
    b_mm = b * 1000.0
    d_mm = d * 1000.0
    return 0.17 * LAMBDA_NW * math.sqrt(fc_MPa) * b_mm * d_mm


def _Vs_threshold_for_max_spacing(fc_prime: float, b: float, d: float) -> float:
    """Threshold V_s = 0.33 √f_c'[MPa] b d (N) above which the stricter
    maximum-spacing limit applies (9.7.6.2.2).
    """
    fc_MPa = fc_prime / 1.0e6
    b_mm = b * 1000.0
    d_mm = d * 1000.0
    return 0.33 * math.sqrt(fc_MPa) * b_mm * d_mm


def _s_max(d: float, V_s: float, V_s_threshold: float) -> float:
    """Maximum stirrup spacing per 9.7.6.2.2."""
    if V_s <= V_s_threshold:
        return min(d / 2.0, S_MAX_LOW_HARD)
    return min(d / 4.0, S_MAX_HIGH_HARD)


def _Av_min_per_s(fc_prime: float, fyt: float, b: float) -> float:
    """Minimum ``A_v / s`` per ACI 9.6.3.3 (SI form):

        (A_v / s)_min = max(0.062 √f_c'[MPa] b / f_yt[MPa],
                             0.35 b / f_yt[MPa])

    Result in m²/m (consistent with our SI conventions).
    """
    fc_MPa = fc_prime / 1.0e6
    fyt_MPa = fyt / 1.0e6
    b_mm = b * 1000.0
    # mm²/mm = mm; convert to m²/m by *1e-3
    a1 = 0.062 * math.sqrt(fc_MPa) * b_mm / fyt_MPa     # mm
    a2 = 0.35 * b_mm / fyt_MPa                           # mm
    return max(a1, a2) * 1.0e-3                          # m²/m


# ============================================================ check entry

def beam_shear_strength(section: ConcreteSection, *,
                          V_u: float | None = None) -> ShearCheck:
    """Compute nominal and design shear strength of a rectangular
    concrete beam per ACI 318-19 Ch. 22.5 + 9.6.3 + 9.7.6.2.

    Parameters
    ----------
    section : ConcreteSection
        Section with rebar layout (uses ``rebar.Av`` and
        ``rebar.stirrup_spacing``).
    V_u : float, optional
        Factored shear demand at the section (N), used only to flag
        capacity-vs-demand status in ``notes``. The capacity itself
        is computed independently.
    """
    fc = section.material.fc_prime
    fyt = section.material.fy
    b = section.b
    d = section.d
    Av = section.rebar.Av
    s = section.rebar.stirrup_spacing

    V_c = _Vc_simplified(fc, b, d)
    V_s = Av * fyt * d / s if s > 0.0 else 0.0
    V_n = V_c + V_s
    phi_V_n = PHI_SHEAR * V_n

    V_s_thr = _Vs_threshold_for_max_spacing(fc, b, d)
    s_max = _s_max(d, V_s, V_s_thr)
    Av_min_per_s = _Av_min_per_s(fc, fyt, b)
    Av_per_s_actual = Av / s if s > 0.0 else 0.0
    spacing_ok = s <= s_max + 1.0e-9
    min_reinforcement_ok = Av_per_s_actual >= Av_min_per_s - 1.0e-12

    notes: list[str] = []
    if not spacing_ok:
        notes.append(
            f"stirrup spacing s = {s*1000:.0f} mm exceeds s_max = "
            f"{s_max*1000:.0f} mm per ACI 9.7.6.2.2"
        )
    if not min_reinforcement_ok:
        notes.append(
            f"A_v/s = {Av_per_s_actual*1e6:.0f} mm²/m < required min "
            f"{Av_min_per_s*1e6:.0f} mm²/m per ACI 9.6.3.3"
        )
    if V_u is not None:
        if V_u > phi_V_n + 1.0e-6:
            notes.append(
                f"V_u = {V_u/1e3:.1f} kN > φV_n = {phi_V_n/1e3:.1f} kN "
                f"(undercapacity)"
            )
        # 9.6.3.1: stirrups required if V_u > 0.5 φ V_c
        if V_u > 0.5 * PHI_SHEAR * V_c and Av <= 0.0:
            notes.append(
                "V_u > 0.5 φV_c but no transverse reinforcement provided "
                "(ACI 9.6.3.1)"
            )

    return ShearCheck(
        V_n=V_n, phi=PHI_SHEAR, phi_V_n=phi_V_n,
        V_c=V_c, V_s=V_s,
        s_max=s_max, s_actual=s,
        Av_min_per_s=Av_min_per_s,
        Av_per_s_actual=Av_per_s_actual,
        spacing_ok=spacing_ok,
        min_reinforcement_ok=min_reinforcement_ok,
        notes="; ".join(notes),
    )


# ============================================================ design entry

def design_stirrup_spacing(section: ConcreteSection, V_u: float) -> ShearDesign:
    """Given a factored shear demand ``V_u``, return the required
    stirrup spacing per ACI 318-19 to satisfy ``V_u <= φ V_n``.

    The stirrup area is read from ``section.rebar.Av`` -- the user
    supplies a stirrup designation and the design output is the
    required spacing.

    Parameters
    ----------
    section : ConcreteSection
    V_u : float
        Factored shear demand at the section (N).

    Returns
    -------
    ShearDesign
        Includes ``s_recommended`` -- the spacing you should actually
        use (the most restrictive of strength, max spacing, and
        min-reinforcement requirements).
    """
    fc = section.material.fc_prime
    fyt = section.material.fy
    b = section.b
    d = section.d
    Av = section.rebar.Av

    V_c = _Vc_simplified(fc, b, d)
    V_s_req = V_u / PHI_SHEAR - V_c

    # Spacing from strength requirement
    if V_s_req <= 0.0:
        s_required = float("inf")    # V_c alone is sufficient
        V_s_req_clipped = 0.0
    else:
        s_required = Av * fyt * d / V_s_req
        V_s_req_clipped = V_s_req

    # Max spacing based on V_s level
    V_s_thr = _Vs_threshold_for_max_spacing(fc, b, d)
    s_max_strict = _s_max(d, V_s_req_clipped, V_s_thr)

    # Minimum-reinforcement spacing: s <= Av / (Av_min_per_s)
    Av_min_per_s = _Av_min_per_s(fc, fyt, b)
    if Av_min_per_s > 0.0:
        s_min_reinf = Av / Av_min_per_s
    else:
        s_min_reinf = float("inf")

    s_recommended = min(s_required, s_max_strict, s_min_reinf)

    notes: list[str] = []
    if V_s_req <= 0.0:
        notes.append(
            "V_c alone covers demand; provide minimum stirrups per 9.6.3"
        )
    if s_recommended == s_max_strict and s_max_strict < s_required:
        notes.append(f"max-spacing limit governs: s_max = {s_max_strict*1000:.0f} mm")
    if s_recommended == s_min_reinf and s_min_reinf < s_required:
        notes.append(
            f"min-reinforcement governs: s = {s_min_reinf*1000:.0f} mm"
        )
    if V_s_req > 0.66 * math.sqrt(fc / 1e6) * b * 1000 * d * 1000:
        # ACI 22.5.1.2: V_s should not exceed 0.66 √f_c' b d -- if it does,
        # the section is too small
        notes.append(
            "required V_s exceeds 0.66 √f_c' b·d (ACI 22.5.1.2); enlarge "
            "section -- stirrups alone cannot resist demand"
        )

    return ShearDesign(
        V_u=V_u, V_c=V_c, V_s_required=V_s_req, Av=Av,
        s_required=s_required, s_max=s_max_strict,
        s_min_reinforcement=s_min_reinf,
        s_recommended=s_recommended,
        notes="; ".join(notes),
    )
