"""Combined-force interaction per AISC 360-22 Chapter H.

For a doubly-symmetric I-shape under axial + biaxial bending, the
governing interaction equations (H1.1 -- compression; H1.2 --
tension) are:

* **High axial demand** (``P_r / P_c >= 0.2``, Eq H1-1a):

      P_r / P_c + (8/9) · (M_rx / M_cx + M_ry / M_cy) <= 1.0

* **Low axial demand** (``P_r / P_c < 0.2``, Eq H1-1b):

      P_r / (2 P_c) + (M_rx / M_cx + M_ry / M_cy) <= 1.0

Definitions
-----------
* ``P_r`` -- required axial strength (factored, ``P_u`` for LRFD).
* ``P_c`` -- available axial strength: ``φP_n`` from Ch. E for
  compression, from Ch. D for tension.
* ``M_rx``, ``M_ry`` -- required flexural strengths about strong (x)
  and weak (y) axes.
* ``M_cx`` -- available strong-axis flexural strength from Ch. F2
  (including LTB at the user-supplied ``L_b`` and ``C_b``).
* ``M_cy`` -- available weak-axis flexural strength. For W-shapes
  bent about the weak axis there is no LTB; per F6 the nominal is
  the plastic moment ``F_y · Z_y``, capped at ``1.6 F_y S_y`` to
  prevent excessive deformation. For typical sections (``Z_y / S_y < 1.6``)
  the cap is not active.

The ``CombinedForceCheck`` returned carries the demand-capacity ratio
``DCR``, the equation used, and the individual ratios so the user
can see which term dominates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from femsolver.design.steel.compression import compression_strength
from femsolver.design.steel.flexure import flexural_strength
from femsolver.design.steel.sections import SteelMaterial, SteelSection
from femsolver.design.steel.tension_shear import tension_strength


PHI_FLEXURE_WEAK = 0.90        # AISC F6 (same as F2 for the LRFD φ_b)


# ============================================================ result

@dataclass
class CombinedForceCheck:
    """Result of an AISC 360-22 §H1 interaction-equation evaluation.

    Attributes
    ----------
    DCR : float
        Demand-capacity ratio. Per AISC §H1, the section is adequate
        for combined forces when ``DCR <= 1.0``.
    equation_used : str
        ``"H1-1a"`` (P_r/P_c >= 0.2) or ``"H1-1b"`` (P_r/P_c < 0.2).
    P_r, P_c : float
        Required and available axial strengths (both positive).
    M_rx, M_cx : float
        Required and available strong-axis flexural strengths.
    M_ry, M_cy : float
        Required and available weak-axis flexural strengths.
    P_r_over_P_c, M_rx_over_M_cx, M_ry_over_M_cy : float
        Individual demand/capacity terms (useful for diagnosing which
        action dominates).
    is_compression : bool
        ``True`` if ``P_r`` is compression, ``False`` if tension.
    notes : str
    """

    DCR: float
    equation_used: str
    P_r: float
    P_c: float
    M_rx: float
    M_cx: float
    M_ry: float
    M_cy: float
    P_r_over_P_c: float
    M_rx_over_M_cx: float
    M_ry_over_M_cy: float
    is_compression: bool
    notes: str = ""


# ============================================================ weak-axis flexure

def _weak_axis_flexural_strength(section: SteelSection,
                                    material: SteelMaterial) -> float:
    """Available weak-axis flexural strength ``φM_n_y`` per AISC F6.

    For doubly-symmetric I-shapes bent about the weak axis:

        M_n_y = min(F_y · Z_y, 1.6 · F_y · S_y)        (F6-1)

    No LTB occurs about the weak axis. Compactness slenderness limits
    for the unstiffened flange element apply per Table B4.1b Case 10;
    most rolled W-shapes are compact, and the F6 formula gives the
    plastic moment directly.

    ``φ_b = 0.90`` per AISC Table 21.2.1 (same as strong-axis F2).
    """
    Fy = material.Fy
    M_p_y = Fy * section.Zy
    M_n_cap = 1.6 * Fy * section.Sy
    M_n = min(M_p_y, M_n_cap)
    return PHI_FLEXURE_WEAK * M_n


# ============================================================ entry

def combined_force_check(
    section: SteelSection,
    material: SteelMaterial,
    *,
    P_r: float,
    M_rx: float = 0.0,
    M_ry: float = 0.0,
    L: float,
    L_b: float | None = None,
    K_x: float = 1.0,
    K_y: float = 1.0,
    L_x: float | None = None,
    L_y: float | None = None,
    C_b: float = 1.0,
    A_e: float | None = None,
) -> CombinedForceCheck:
    """Combined axial + biaxial flexure check per AISC 360-22 §H1.

    Parameters
    ----------
    section : SteelSection
    material : SteelMaterial
    P_r : float
        Required axial force (N). **Positive = compression, negative
        = tension** -- the routine selects the correct capacity
        (Ch. E or Ch. D).
    M_rx : float, default 0
        Required strong-axis moment (N·m, absolute value used).
    M_ry : float, default 0
        Required weak-axis moment (N·m, absolute value used).
    L : float
        Member length / unbraced length (m).
    L_b : float, optional
        Unbraced length for LTB about the strong axis (defaults to
        ``L``).
    K_x, K_y, L_x, L_y : float
        Effective length factors / per-axis unbraced lengths for the
        compression check (defaults: K = 1, L_x = L_y = L).
    C_b : float, default 1.0
        Moment-gradient factor for LTB.
    A_e : float, optional
        Effective net area for tension rupture check (defaults to
        ``A_g``).
    """
    if L_b is None:
        L_b = L

    is_compression = P_r >= 0.0
    P_r_abs = abs(P_r)
    M_rx_abs = abs(M_rx)
    M_ry_abs = abs(M_ry)

    # Axial capacity
    if is_compression:
        comp = compression_strength(
            section, material, L=L,
            K_x=K_x, K_y=K_y, L_x=L_x, L_y=L_y,
        )
        P_c = comp.phi_P_n
    else:
        tens = tension_strength(section, material, A_e=A_e, L=L, K=K_x)
        P_c = tens.phi_P_n

    # Strong-axis flexural capacity (with LTB)
    flex = flexural_strength(section, material, L_b=L_b, C_b=C_b)
    M_cx = flex.phi_M_n

    # Weak-axis flexural capacity (no LTB, F6)
    M_cy = _weak_axis_flexural_strength(section, material)

    # Avoid division by zero
    if P_c <= 0.0:
        raise RuntimeError(
            "axial capacity P_c <= 0; check section / material / lengths"
        )
    if M_cx <= 0.0:
        raise RuntimeError("strong-axis flexural capacity M_cx <= 0")
    if M_cy <= 0.0:
        raise RuntimeError("weak-axis flexural capacity M_cy <= 0")

    r_P = P_r_abs / P_c
    r_Mx = M_rx_abs / M_cx
    r_My = M_ry_abs / M_cy

    # Interaction equation per H1.1 / H1.2
    if r_P >= 0.2:
        DCR = r_P + (8.0 / 9.0) * (r_Mx + r_My)
        eq = "H1-1a"
    else:
        DCR = 0.5 * r_P + (r_Mx + r_My)
        eq = "H1-1b"

    notes: list[str] = []
    if DCR > 1.0:
        notes.append(
            f"DCR = {DCR:.3f} > 1.0 -- section is undersized for the "
            "combined demand"
        )
    if is_compression and not compression_strength(
        section, material, L=L, K_x=K_x, K_y=K_y, L_x=L_x, L_y=L_y,
    ).section_nonslender:
        notes.append("compression section is slender (Ch. E7 not implemented)")
    if not flex.section_compact:
        notes.append("flexural section is non-compact (Ch. F3/F4 not "
                       "implemented)")

    return CombinedForceCheck(
        DCR=DCR, equation_used=eq,
        P_r=P_r_abs, P_c=P_c,
        M_rx=M_rx_abs, M_cx=M_cx,
        M_ry=M_ry_abs, M_cy=M_cy,
        P_r_over_P_c=r_P,
        M_rx_over_M_cx=r_Mx,
        M_ry_over_M_cy=r_My,
        is_compression=is_compression,
        notes="; ".join(notes),
    )
