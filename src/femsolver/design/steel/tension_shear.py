"""Tension and shear strength of W-shapes per AISC 360-22 Ch. D + G.

Tension (Ch. D2)
----------------
Two limit states; the lower governs:

* **Tensile yielding** on the gross section (D2-1):

      P_n = F_y · A_g            φ_t = 0.90

* **Tensile rupture** on the net section (D2-2):

      P_n = F_u · A_e             φ_t = 0.75

  ``A_e = A_n · U`` where ``A_n`` is net area (gross minus bolt-hole
  area) and ``U`` is the shear-lag factor (Table D3.1). For a member
  without holes or with negligible shear lag, ``A_e = A_g``.

§D3 recommends ``KL/r <= 300`` for tension members (slenderness
limit). Not enforced -- only flagged.

Shear (Ch. G2)
--------------
For doubly-symmetric I-shapes, the nominal shear strength (G2-1):

    V_n = 0.6 · F_y · A_w · C_v1

* ``A_w = d · t_w`` (gross web area, per G2.1).
* ``C_v1`` per G2.1b is the **web shear strength coefficient**:
   - If ``h / t_w <= 2.24 √(E / F_y)``: ``C_v1 = 1.0`` AND ``φ_v = 1.00``
     (per G1 -- no reduction for compact webs of rolled I-shapes).
   - Otherwise, ``φ_v = 0.90`` and ``C_v1`` from G2-3/G2-4 with
     ``k_v = 5.34`` (unstiffened webs, per G2.1b).

The first branch (``φ_v = 1.00``) covers the vast majority of rolled
W-shapes; W4 through W36 sections from the embedded catalog all have
``h / t_w`` comfortably below the 2.24√(E/F_y) limit (~54 for A992).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from femsolver.design.steel.sections import SteelMaterial, SteelSection


PHI_TENSION_YIELD = 0.90        # AISC D2(a)
PHI_TENSION_RUPTURE = 0.75      # AISC D2(b)


# ============================================================ TENSION

@dataclass
class TensionCheck:
    """Result of an AISC 360-22 §D2 tensile-strength evaluation.

    Attributes
    ----------
    P_n_yield, P_n_rupture : float
        Per-limit-state nominal strengths (N).
    phi_P_n_yield, phi_P_n_rupture : float
        Per-limit-state design strengths.
    P_n : float
        Governing nominal strength (= smaller of the two limit
        states).
    phi : float
        Governing strength-reduction factor.
    phi_P_n : float
        Governing design strength.
    governing_limit_state : str
        ``"yielding"`` (D2-1) or ``"rupture"`` (D2-2).
    A_g : float
        Gross area used in yielding (m²).
    A_e : float
        Effective net area used in rupture (m²).
    notes : str
        Slenderness warnings if KL/r > 300.
    """

    P_n_yield: float
    P_n_rupture: float
    phi_P_n_yield: float
    phi_P_n_rupture: float
    P_n: float
    phi: float
    phi_P_n: float
    governing_limit_state: str
    A_g: float
    A_e: float
    notes: str = ""


def tension_strength(
    section: SteelSection,
    material: SteelMaterial,
    *,
    A_e: float | None = None,
    L: float | None = None,
    K: float = 1.0,
) -> TensionCheck:
    """Tensile strength per AISC 360-22 §D2.

    Parameters
    ----------
    section : SteelSection
    material : SteelMaterial
    A_e : float, optional
        Effective net area in m². If ``None`` (default), uses
        ``A_g`` (no bolt holes, no shear lag).
    L : float, optional
        Member length (m). Used only to flag the §D3 slenderness
        recommendation ``KL/r <= 300`` (informational; not enforced).
    K : float, default 1.0
        Effective length factor (default tension members K = 1).
    """
    A_g = section.A
    if A_e is None:
        A_e = A_g
    if A_e <= 0.0:
        raise ValueError(f"A_e must be positive, got {A_e}")
    Fy = material.Fy
    Fu = material.Fu

    # D2-1: tensile yielding on gross section
    P_n_y = Fy * A_g
    phi_P_n_y = PHI_TENSION_YIELD * P_n_y
    # D2-2: tensile rupture on effective net section
    P_n_r = Fu * A_e
    phi_P_n_r = PHI_TENSION_RUPTURE * P_n_r

    # Governing
    if phi_P_n_y <= phi_P_n_r:
        P_n = P_n_y
        phi = PHI_TENSION_YIELD
        phi_P_n = phi_P_n_y
        governing = "yielding"
    else:
        P_n = P_n_r
        phi = PHI_TENSION_RUPTURE
        phi_P_n = phi_P_n_r
        governing = "rupture"

    notes: list[str] = []
    if L is not None:
        # Use the smaller r (governing slenderness)
        r_min = min(section.rx, section.ry)
        slr = K * L / r_min
        if slr > 300.0:
            notes.append(
                f"KL/r = {slr:.0f} > 300 (AISC §D3 recommended limit "
                "for tension members)"
            )

    return TensionCheck(
        P_n_yield=P_n_y, P_n_rupture=P_n_r,
        phi_P_n_yield=phi_P_n_y, phi_P_n_rupture=phi_P_n_r,
        P_n=P_n, phi=phi, phi_P_n=phi_P_n,
        governing_limit_state=governing,
        A_g=A_g, A_e=A_e,
        notes="; ".join(notes),
    )


# ============================================================ SHEAR

@dataclass
class ShearCheck:
    """Result of an AISC 360-22 §G2 shear-strength evaluation.

    Attributes
    ----------
    V_n : float
        Nominal shear strength (N).
    phi : float
        Strength-reduction factor (= 1.00 for compact rolled
        I-shape webs, 0.90 otherwise).
    phi_V_n : float
        Design shear strength.
    A_w : float
        Web area = d · t_w (m²).
    h_over_tw : float
        Web slenderness ratio.
    C_v1 : float
        Web-shear strength coefficient (1.0 for compact webs).
    web_compact : bool
        ``True`` if ``h/t_w <= 2.24 √(E/F_y)`` (qualifies for the
        more favourable φ_v = 1.00).
    notes : str
    """

    V_n: float
    phi: float
    phi_V_n: float
    A_w: float
    h_over_tw: float
    C_v1: float
    web_compact: bool
    notes: str = ""


def shear_strength(
    section: SteelSection,
    material: SteelMaterial,
) -> ShearCheck:
    """Shear strength of a doubly-symmetric I-shape per AISC 360-22
    §G2.

    Assumes **unstiffened** web (``k_v = 5.34``). For stiffened webs
    or built-up members, the user can post-correct ``C_v1`` and
    re-multiply.
    """
    Fy = material.Fy
    E = material.E
    h_clear = section.d - 2.0 * section.k_des
    h_over_tw = h_clear / section.tw
    A_w = section.d * section.tw      # gross web area per G2.1

    # Compact-web boundary for the more favourable φ_v = 1.00
    threshold_phi = 2.24 * math.sqrt(E / Fy)
    web_compact = h_over_tw <= threshold_phi

    if web_compact:
        # Most rolled W-shapes land here -- no buckling reduction
        C_v1 = 1.0
        phi = 1.00
    else:
        # G2-3 / G2-4 for unstiffened webs with k_v = 5.34
        kv = 5.34
        threshold_Cv1 = 1.10 * math.sqrt(kv * E / Fy)
        if h_over_tw <= threshold_Cv1:
            C_v1 = 1.0
        else:
            C_v1 = 1.10 * math.sqrt(kv * E / Fy) / h_over_tw
        phi = 0.90

    V_n = 0.6 * Fy * A_w * C_v1
    phi_V_n = phi * V_n

    notes: list[str] = []
    if not web_compact:
        notes.append(
            f"web h/tw = {h_over_tw:.1f} > {threshold_phi:.1f} -- using "
            f"φ_v = 0.90 with C_v1 = {C_v1:.3f}"
        )

    return ShearCheck(
        V_n=V_n, phi=phi, phi_V_n=phi_V_n,
        A_w=A_w, h_over_tw=h_over_tw, C_v1=C_v1,
        web_compact=web_compact,
        notes="; ".join(notes),
    )
