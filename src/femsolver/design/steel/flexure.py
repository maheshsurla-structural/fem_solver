"""Flexural strength of doubly-symmetric compact I-shapes per AISC
360-22 Chapter F2.

Implements the major-axis bending strength of W-shapes with three
limit states ordered by increasing unbraced length ``L_b``:

* **Yielding (plastic moment, F2-1)** for ``L_b <= L_p``:

      M_n = M_p = F_y · Z_x

* **Inelastic LTB (F2-2)** for ``L_p < L_b <= L_r``:

      M_n = C_b · [M_p - (M_p - 0.7 F_y S_x) · (L_b - L_p) / (L_r - L_p)]
            ≤ M_p

* **Elastic LTB (F2-3 with F2-4)** for ``L_b > L_r``:

      M_n = F_cr · S_x ≤ M_p

      F_cr = (C_b · π² E) / (L_b / r_ts)²
             · √[1 + 0.078 · (J c / (S_x h_o)) · (L_b / r_ts)²]

Geometric helpers (computed from the section table):

* ``L_p = 1.76 · r_y · √(E / F_y)``                          (F2-5)
* ``r_ts² = √(I_y · C_w) / S_x``                              (F2-7)
* ``h_o = d - t_f``                                            (clear
  distance between flange centroids; exact form F2.4)
* ``L_r`` per (F2-6) (more involved expression below)

Inputs/outputs in SI: lengths in m, stresses in Pa, moments in N·m.
``φ_b = 0.90`` (LRFD).

C_b (moment-gradient factor)
----------------------------
Per AISC §F1, ``C_b`` modifies LTB strength to account for non-uniform
moment along the unbraced length:

    C_b = 12.5 M_max / (2.5 M_max + 3 M_A + 4 M_B + 3 M_C)

The user supplies ``C_b`` directly (default 1.0 for uniform-moment /
conservative). A helper :func:`c_b_from_moments` is provided to
compute C_b from the four moments at the quarter-, mid-, and three-
quarter points plus the maximum within the segment.

Section compactness (Table B4.1b)
---------------------------------
F2 applies only to **compact** sections (compact flange + compact web).
Limits for doubly-symmetric I-shapes:

* Flange (Case 10): ``b_f / (2 t_f) <= 0.38 √(E / F_y)``
* Web    (Case 15): ``(d - 2 k_des) / t_w <= 3.76 √(E / F_y)``

Non-compact sections require F3 (non-compact flange) or F4 (non-compact
web) reductions, which this implementation flags but does not yet apply.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from femsolver.design.steel.sections import SteelMaterial, SteelSection


PHI_FLEXURE = 0.90        # AISC 360-22 LRFD


# ============================================================ result

@dataclass
class FlexureCheck:
    """Result of an AISC 360-22 Ch. F2 flexural-strength evaluation.

    Attributes
    ----------
    M_n : float
        Nominal flexural strength (N·m).
    phi : float
        Strength-reduction factor (= 0.90 for LRFD).
    phi_M_n : float
        Design flexural strength.
    M_p : float
        Plastic moment ``F_y · Z_x`` (N·m).
    L_b, L_p, L_r : float
        Unbraced length, plastic limit, LTB limit (m).
    C_b : float
        Moment-gradient factor used in this evaluation.
    regime : str
        ``"plastic"`` / ``"inelastic-LTB"`` / ``"elastic-LTB"``.
    F_cr : float
        Critical LTB stress (Pa). Equals ``M_n / S_x`` in the elastic
        regime; computed for reporting in all regimes.
    section_compact : bool
        Flange + web both compact per Table B4.1b.
    notes : str
        Warnings (non-compact section, etc.).
    """

    M_n: float
    phi: float
    phi_M_n: float
    M_p: float
    L_b: float
    L_p: float
    L_r: float
    C_b: float
    regime: str
    F_cr: float
    section_compact: bool
    notes: str = ""


# ============================================================ C_b helper

def c_b_from_moments(M_max: float, M_a: float, M_b: float,
                       M_c: float) -> float:
    """Compute C_b per AISC Eq. F1-1 from segment moments.

    Parameters
    ----------
    M_max : float
        Absolute maximum moment within the unbraced segment.
    M_a, M_b, M_c : float
        Absolute moments at the quarter-, mid-, and three-quarter
        points of the unbraced segment.

    Returns
    -------
    float
        ``C_b = 12.5 M_max / (2.5 M_max + 3 M_a + 4 M_b + 3 M_c)``,
        clipped to a maximum of 3.0 per AISC.
    """
    denom = (2.5 * abs(M_max) + 3.0 * abs(M_a)
              + 4.0 * abs(M_b) + 3.0 * abs(M_c))
    if denom <= 0.0:
        return 1.0
    cb = 12.5 * abs(M_max) / denom
    return min(cb, 3.0)


# ============================================================ section properties

def _section_compactness(section: SteelSection,
                          material: SteelMaterial) -> tuple[bool, dict]:
    """Compactness check per Table B4.1b for doubly-symmetric I-shape
    in flexure about the strong axis.
    """
    E = material.E
    Fy = material.Fy
    lambda_flange = (section.bf / 2.0) / section.tf
    lambda_flange_p = 0.38 * math.sqrt(E / Fy)       # Case 10
    h_clear = section.d - 2.0 * section.k_des
    lambda_web = h_clear / section.tw
    lambda_web_p = 3.76 * math.sqrt(E / Fy)          # Case 15
    flange_compact = lambda_flange <= lambda_flange_p
    web_compact = lambda_web <= lambda_web_p
    return (flange_compact and web_compact, {
        "lambda_flange": lambda_flange,
        "lambda_flange_p": lambda_flange_p,
        "lambda_web": lambda_web,
        "lambda_web_p": lambda_web_p,
        "flange_compact": flange_compact,
        "web_compact": web_compact,
    })


def _r_ts(section: SteelSection) -> float:
    """Eq F2-7: r_ts² = √(I_y · C_w) / S_x → r_ts = (√(I_y · C_w) / S_x)^0.5."""
    return math.sqrt(math.sqrt(section.Iy * section.Cw) / section.Sx)


def _h_o(section: SteelSection) -> float:
    """Distance between flange centroids ≈ d - tf."""
    return section.d - section.tf


def _L_p(section: SteelSection, material: SteelMaterial) -> float:
    """Eq F2-5."""
    return 1.76 * section.ry * math.sqrt(material.E / material.Fy)


def _L_r(section: SteelSection, material: SteelMaterial) -> float:
    """Eq F2-6 for doubly-symmetric I-shape (c = 1):

        L_r = 1.95 · r_ts · (E / (0.7 F_y))
              · √[J c / (S_x h_o) + √((J c / (S_x h_o))² + 6.76 · (0.7 F_y / E)²)]
    """
    E = material.E
    Fy = material.Fy
    rts = _r_ts(section)
    ho = _h_o(section)
    c = 1.0       # doubly-symmetric I-shape
    A = section.J * c / (section.Sx * ho)
    B = 0.7 * Fy / E
    inner = math.sqrt(A * A + 6.76 * B * B)
    outer = math.sqrt(A + inner)
    return 1.95 * rts * (E / (0.7 * Fy)) * outer


# ============================================================ check entry

def flexural_strength(
    section: SteelSection,
    material: SteelMaterial,
    *,
    L_b: float,
    C_b: float = 1.0,
) -> FlexureCheck:
    """Flexural strength of a doubly-symmetric I-shape per AISC 360-22
    Ch. F2 (compact major-axis bending).

    Parameters
    ----------
    section : SteelSection
    material : SteelMaterial
    L_b : float
        Unbraced length of the compression flange (m).
    C_b : float, default 1.0
        Moment-gradient factor per Eq. F1-1. Use 1.0 (conservative,
        uniform moment) if unsure; use :func:`c_b_from_moments` to
        compute from quarter/mid/three-quarter moments.
    """
    if L_b <= 0.0:
        raise ValueError(f"L_b must be positive, got {L_b}")
    if C_b <= 0.0:
        raise ValueError(f"C_b must be positive, got {C_b}")

    E = material.E
    Fy = material.Fy

    # Plastic moment
    M_p = Fy * section.Zx

    # Length limits
    L_p = _L_p(section, material)
    L_r = _L_r(section, material)
    Sx = section.Sx
    rts = _r_ts(section)
    ho = _h_o(section)

    # Determine regime
    if L_b <= L_p:
        regime = "plastic"
        M_n = M_p
        F_cr = M_n / Sx
    elif L_b <= L_r:
        regime = "inelastic-LTB"
        # F2-2: linear interpolation between M_p and 0.7 F_y S_x
        M_n_lin = C_b * (M_p - (M_p - 0.7 * Fy * Sx)
                          * (L_b - L_p) / (L_r - L_p))
        M_n = min(M_n_lin, M_p)
        F_cr = M_n / Sx
    else:
        regime = "elastic-LTB"
        # F2-4: elastic LTB critical stress
        c = 1.0
        Lb_rts = L_b / rts
        bracket = math.sqrt(
            1.0 + 0.078 * (section.J * c / (Sx * ho)) * (Lb_rts ** 2)
        )
        F_cr = (C_b * math.pi ** 2 * E) / (Lb_rts ** 2) * bracket
        M_n_elastic = F_cr * Sx
        M_n = min(M_n_elastic, M_p)

    phi_M_n = PHI_FLEXURE * M_n

    compact, details = _section_compactness(section, material)
    notes: list[str] = []
    if not compact:
        bad = []
        if not details["flange_compact"]:
            bad.append(
                f"flange λ = {details['lambda_flange']:.2f} > λ_p = "
                f"{details['lambda_flange_p']:.2f}"
            )
        if not details["web_compact"]:
            bad.append(
                f"web λ = {details['lambda_web']:.2f} > λ_p = "
                f"{details['lambda_web_p']:.2f}"
            )
        notes.append(
            "section is NON-COMPACT (" + ", ".join(bad) + "); F3/F4 "
            "reductions apply but are not implemented here -- result "
            "is unconservative for these sections"
        )

    return FlexureCheck(
        M_n=M_n,
        phi=PHI_FLEXURE,
        phi_M_n=phi_M_n,
        M_p=M_p,
        L_b=L_b, L_p=L_p, L_r=L_r,
        C_b=C_b,
        regime=regime,
        F_cr=F_cr,
        section_compact=compact,
        notes="; ".join(notes),
    )
