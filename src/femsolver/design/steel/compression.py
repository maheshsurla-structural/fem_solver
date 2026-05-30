"""Compression strength of doubly-symmetric I-shapes per AISC 360-22
Chapter E.

Implements the flexural-buckling limit state (E3) for non-slender
W-shapes:

    F_e = π² E / (KL/r)²                          (E3-4) elastic buckling

    If KL/r <= 4.71 √(E/F_y)  (equivalently F_y/F_e <= 2.25):

        F_cr = (0.658)^(F_y/F_e) · F_y           (E3-2) inelastic

    Otherwise:

        F_cr = 0.877 · F_e                        (E3-3) elastic

    P_n = F_cr · A_g                              (E3-1)
    φ_c = 0.90  (LRFD)

Slenderness checks (B4.1a, Table B4.1a):

* **Flange** (Case 1 -- unstiffened element under uniform compression):
  ``b/t = (b_f / 2) / t_f <= 0.56 √(E / F_y)``
* **Web** (Case 5 -- stiffened element under uniform compression):
  ``h/t_w = (d - 2 k_des) / t_w <= 1.49 √(E / F_y)``

Sections that exceed either limit are **slender** and Ch. E7 applies
a reduced-effective-area treatment that this implementation does not
yet cover -- the warning in :attr:`CompressionCheck.notes` flags this.

The recommended slenderness ratio ``KL/r <= 200`` (E2) is checked
and flagged but not enforced (it is a "should" not a "shall").
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from femsolver.design.steel.sections import SteelMaterial, SteelSection


PHI_COMPRESSION = 0.90        # AISC 360-22 LRFD


# ============================================================ result

@dataclass
class CompressionCheck:
    """Result of an AISC 360-22 Ch. E compression-strength evaluation.

    Attributes
    ----------
    P_n : float
        Nominal compressive strength (N).
    phi : float
        Strength-reduction factor (= 0.90 for LRFD).
    phi_P_n : float
        Design compressive strength (N).
    F_cr : float
        Critical (buckling) stress (Pa).
    F_e : float
        Elastic (Euler) buckling stress (Pa).
    KL_over_r : float
        Governing slenderness ratio.
    governing_axis : str
        ``"x"`` or ``"y"`` -- the axis whose slenderness governs.
    buckling_regime : str
        ``"inelastic"`` (E3-2) or ``"elastic"`` (E3-3).
    slenderness_ok : bool
        ``True`` if KL/r <= 200 (E2 recommendation).
    section_nonslender : bool
        ``True`` if both flange and web meet the non-slender limits
        of Table B4.1a -- a precondition for using E3 instead of E7.
    notes : str
        Warnings if E2 limit exceeded or section is slender.
    """

    P_n: float
    phi: float
    phi_P_n: float
    F_cr: float
    F_e: float
    KL_over_r: float
    governing_axis: str
    buckling_regime: str
    slenderness_ok: bool
    section_nonslender: bool
    notes: str = ""


# ============================================================ helpers

def _classify_section_slenderness(section: SteelSection,
                                     material: SteelMaterial) -> tuple[bool, dict]:
    """Return (is_nonslender, details) per Table B4.1a Cases 1 and 5
    for an I-shape under uniform compression.
    """
    E = material.E
    Fy = material.Fy
    # Flange (Case 1): b = bf/2 (half flange), t = tf
    lambda_flange = (section.bf / 2.0) / section.tf
    lambda_flange_lim = 0.56 * math.sqrt(E / Fy)
    flange_ok = lambda_flange <= lambda_flange_lim
    # Web (Case 5): h = d - 2*kdes (clear distance between flange fillets)
    h_clear = section.d - 2.0 * section.k_des
    lambda_web = h_clear / section.tw
    lambda_web_lim = 1.49 * math.sqrt(E / Fy)
    web_ok = lambda_web <= lambda_web_lim
    details = {
        "lambda_flange": lambda_flange,
        "lambda_flange_lim": lambda_flange_lim,
        "lambda_web": lambda_web,
        "lambda_web_lim": lambda_web_lim,
        "flange_ok": flange_ok,
        "web_ok": web_ok,
    }
    return (flange_ok and web_ok, details)


# ============================================================ check entry

def compression_strength(
    section: SteelSection,
    material: SteelMaterial,
    *,
    L: float,
    K_x: float = 1.0,
    K_y: float = 1.0,
    L_x: float | None = None,
    L_y: float | None = None,
) -> CompressionCheck:
    """Compute the nominal and design compressive strength of a
    doubly-symmetric I-shape per AISC 360-22 Ch. E (flexural buckling
    limit state, E3).

    Parameters
    ----------
    section : SteelSection
    material : SteelMaterial
    L : float
        Default unbraced length about both axes (m). Used when
        ``L_x`` / ``L_y`` are not provided.
    K_x, K_y : float, default 1.0
        Effective length factors about the strong (x) and weak (y)
        axes. Typical values: 1.0 for braced frames (pin-pin),
        0.65–0.8 for fixed conditions, >1.0 for sway frames (use
        a stability analysis or AISC Appendix 7 to estimate K).
    L_x, L_y : float, optional
        Per-axis unbraced lengths (m). Default to ``L`` if omitted.

    Returns
    -------
    CompressionCheck
    """
    if L <= 0.0 and (L_x is None or L_x <= 0.0):
        raise ValueError("L must be positive")

    Lx = L_x if L_x is not None else L
    Ly = L_y if L_y is not None else L
    if Lx <= 0.0 or Ly <= 0.0:
        raise ValueError(f"L_x and L_y must be positive, got {Lx}, {Ly}")
    if K_x <= 0.0 or K_y <= 0.0:
        raise ValueError(f"K_x and K_y must be positive, got {K_x}, {K_y}")

    E = material.E
    Fy = material.Fy

    # Per-axis slenderness ratios
    slr_x = K_x * Lx / section.rx
    slr_y = K_y * Ly / section.ry
    if slr_x >= slr_y:
        slr = slr_x
        governing_axis = "x"
    else:
        slr = slr_y
        governing_axis = "y"

    # Elastic (Euler) buckling stress, Eq. E3-4
    F_e = math.pi ** 2 * E / (slr ** 2) if slr > 0.0 else float("inf")

    # Inelastic vs elastic regime, Eqs. E3-2/E3-3
    transition = 4.71 * math.sqrt(E / Fy)
    if slr <= transition:
        # Inelastic: F_cr = (0.658^(Fy/Fe)) · Fy
        F_cr = (0.658 ** (Fy / F_e)) * Fy
        buckling_regime = "inelastic"
    else:
        # Elastic: F_cr = 0.877 · F_e
        F_cr = 0.877 * F_e
        buckling_regime = "elastic"

    P_n = F_cr * section.A
    phi = PHI_COMPRESSION
    phi_P_n = phi * P_n

    slenderness_ok = (slr <= 200.0)
    nonslender, details = _classify_section_slenderness(section, material)

    notes: list[str] = []
    if not slenderness_ok:
        notes.append(
            f"KL/r = {slr:.0f} > 200 (AISC E2 recommended limit)"
        )
    if not nonslender:
        bad = []
        if not details["flange_ok"]:
            bad.append(
                f"flange λ = {details['lambda_flange']:.2f} > "
                f"{details['lambda_flange_lim']:.2f}"
            )
        if not details["web_ok"]:
            bad.append(
                f"web λ = {details['lambda_web']:.2f} > "
                f"{details['lambda_web_lim']:.2f}"
            )
        notes.append(
            "section is SLENDER (" + ", ".join(bad) + "); Ch. E7 reduction "
            "applies but is not implemented here -- result is "
            "unconservative for these sections"
        )

    return CompressionCheck(
        P_n=P_n, phi=phi, phi_P_n=phi_P_n,
        F_cr=F_cr, F_e=F_e,
        KL_over_r=slr, governing_axis=governing_axis,
        buckling_regime=buckling_regime,
        slenderness_ok=slenderness_ok,
        section_nonslender=nonslender,
        notes="; ".join(notes),
    )
