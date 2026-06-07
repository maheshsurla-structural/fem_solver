"""NDS-2024 C-factor framework.

The adjusted allowable strength is ``F' = F · C1 · C2 · ...`` where
the C-factors capture load-duration, moisture, temperature, size,
stability, and other modifying effects. NDS §2.3 lists all factors;
this module implements the ones used in member design.

Factors implemented
-------------------
* :func:`C_D_load_duration` -- §2.3.2 Table 2.3.2
* :func:`C_M_wet_service` -- §2.3.3 (multiplier values from each table)
* :func:`C_F_size_factor` -- §4.3.6 Table 4A footnotes (dimension lumber)
* :func:`C_r_repetitive_member` -- §4.3.9 (joists in repetitive use)
* :func:`C_L_lateral_stability` -- §3.3.3 (LTB of bending members)
* :func:`C_P_column_stability` -- §3.7 (column buckling)

Factors NOT implemented (but slots exist in :class:`NDSFactors`):
* C_t (temperature, §2.3.4) -- typically 1.0 for normal conditions
* C_fu (flat use, §4.3.8) -- only when bending about weak axis
* C_i (incising, §4.3.10) -- for pressure-treated lumber
* C_b (bearing area, §3.10) -- for bearing perpendicular checks

Users may pass these explicitly via :class:`NDSFactors`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ============================================================ NDSFactors

@dataclass
class NDSFactors:
    """All NDS C-factors applicable to a design check.

    Defaults to 1.0 (no modification). Pass only the factors that
    deviate from unity.
    """
    C_D: float = 1.0   # Load duration (§2.3.2)
    C_M: float = 1.0   # Wet service (§2.3.3)
    C_t: float = 1.0   # Temperature (§2.3.4)
    C_F: float = 1.0   # Size (§4.3.6)
    C_L: float = 1.0   # Lateral stability for bending (§3.3.3)
    C_P: float = 1.0   # Column stability (§3.7)
    C_r: float = 1.0   # Repetitive member (§4.3.9)
    C_fu: float = 1.0  # Flat use (§4.3.8)
    C_i: float = 1.0   # Incising (§4.3.10)
    C_b: float = 1.0   # Bearing area (§3.10)

    def __post_init__(self) -> None:
        for name in ("C_D", "C_M", "C_t", "C_F", "C_L", "C_P",
                      "C_r", "C_fu", "C_i", "C_b"):
            v = getattr(self, name)
            if v <= 0:
                raise ValueError(f"{name} must be positive, got {v}")


# ============================================================ C_D load duration

_C_D_TABLE = {
    "permanent": 0.9,     # > 10 years, dead load
    "normal": 1.0,        # 10 years, occupancy live load
    "two_months": 1.15,   # snow load
    "seven_days": 1.25,   # construction load
    "ten_minutes": 1.6,   # wind / seismic (NDS Table 2.3.2)
    "impact": 2.0,        # impact
}


def C_D_load_duration(duration: str) -> float:
    """Load-duration factor per NDS Table 2.3.2.

    Parameters
    ----------
    duration : str
        One of: ``"permanent"`` (0.9), ``"normal"`` (1.0),
        ``"two_months"`` (snow, 1.15), ``"seven_days"`` (construction,
        1.25), ``"ten_minutes"`` (wind/seismic, 1.6), ``"impact"`` (2.0).
    """
    if duration not in _C_D_TABLE:
        raise ValueError(
            f"unknown duration {duration!r}; valid: {sorted(_C_D_TABLE)}"
        )
    return _C_D_TABLE[duration]


# ============================================================ C_M wet service

def C_M_wet_service(stress_type: str, *, wet: bool = True) -> float:
    """Wet-service factor per NDS Table 4A footnotes.

    Applied when moisture content in service exceeds 19% for sawn
    lumber. For dry service (the default reference condition),
    ``C_M = 1.0``.

    Parameters
    ----------
    stress_type : str
        Stress type the factor applies to. One of:
        ``"F_b"`` (bending, 0.85), ``"F_t"`` (tension, 1.0),
        ``"F_c"`` (compression parallel, 0.8),
        ``"F_v"`` (shear, 0.97), ``"F_c_perp"`` (bearing, 0.67),
        ``"E"`` (modulus, 0.9).
    wet : bool
        ``True`` if service moisture content > 19%; default ``True``
        since the function is typically called to apply the factor.
    """
    if not wet:
        return 1.0
    table = {
        "F_b": 0.85,
        "F_t": 1.0,
        "F_c": 0.8,
        "F_v": 0.97,
        "F_c_perp": 0.67,
        "E": 0.9,
    }
    if stress_type not in table:
        raise ValueError(
            f"unknown stress_type {stress_type!r}; valid: {sorted(table)}"
        )
    return table[stress_type]


# ============================================================ C_F size factor

def C_F_size_factor(depth_m: float, grade_category: str = "select_structural") -> float:
    """Size factor for dimension lumber per NDS Table 4A footnotes.

    For sawn dimension lumber (2"-4" thick, 5"-14" wide), the
    reference Fb tabulation is for 12"-wide stock. The size factor
    adjusts for other depths.

    Parameters
    ----------
    depth_m : float
        Section depth (vertical dimension for bending about strong
        axis), in metres. Converted internally to inches.
    grade_category : str, default "select_structural"
        Grade family. Values vary slightly by grade; the bundled
        table covers Select Structural, No. 1, No. 2 (which share
        the same C_F).

    Returns
    -------
    C_F : float
        Size factor (≥ 0.9).
    """
    if depth_m <= 0:
        raise ValueError(f"depth must be positive, got {depth_m}")
    d_in = depth_m / 0.0254
    # NDS Table 4A (Adjustment Factors), Size Factor for 2-4" thick
    # dimension lumber. Same for Select Structural, No. 1, No. 2 grades.
    if d_in <= 4.5:
        return 1.5
    if d_in <= 5.5:
        return 1.4
    if d_in <= 6.5:
        return 1.3
    if d_in <= 8.5:
        return 1.2
    if d_in <= 10.5:
        return 1.1
    if d_in <= 12.5:
        return 1.0       # reference 12" depth
    if d_in <= 14.5:
        return 0.9
    return 0.9            # capped for depths > 14"


# ============================================================ C_r repetitive

def C_r_repetitive_member(n_members_in_row: int = 1) -> float:
    """Repetitive-member factor per NDS §4.3.9.

    Applies to bending members spaced 24 in or less on centre, with
    three or more members in a row that share load through a
    structural deck. For such configurations, ``C_r = 1.15`` per
    NDS §4.3.9.
    """
    if n_members_in_row < 1:
        raise ValueError(f"n_members must be >= 1, got {n_members_in_row}")
    return 1.15 if n_members_in_row >= 3 else 1.0


# ============================================================ C_L lateral stability

def C_L_lateral_stability(
    F_b_star: float,
    *,
    l_e: float,
    d: float,
    b: float,
    E_min: float,
) -> float:
    """Lateral-torsional-stability factor for bending per NDS §3.3.3.

    For solid rectangular bending members:

        R_B = sqrt(l_e · d / b^2)
        F_bE = 1.2 · E_min / R_B^2
        F_bE_ratio = F_bE / F_b*
        C_L = (1 + ratio)/1.9 - sqrt(((1 + ratio)/1.9)^2 - ratio/0.95)

    where ``F_b*`` is the bending stress multiplied by all factors
    except ``C_L`` and ``C_fu``.

    Parameters
    ----------
    F_b_star : float
        Reference bending stress multiplied by all factors except
        ``C_L`` (Pa).
    l_e : float
        Effective unbraced length (m).
    d : float
        Depth (vertical dimension of bending member) (m).
    b : float
        Breadth (m).
    E_min : float
        Reduced modulus for stability calcs (NDS ``E_min``,
        approximately the 5th-percentile E) (Pa).
    """
    if min(l_e, d, b, E_min, F_b_star) <= 0:
        raise ValueError("all inputs must be positive")
    # Slenderness limit per NDS §3.3.3 -- R_B <= 50
    R_B = math.sqrt(l_e * d / (b * b))
    if R_B > 50:
        raise ValueError(
            f"R_B = {R_B:.1f} > 50 limit per NDS §3.3.3.7"
        )
    # No lateral stability issue if R_B small
    if R_B < 1e-6:
        return 1.0
    F_bE = 1.2 * E_min / (R_B * R_B)
    ratio = F_bE / F_b_star
    term1 = (1.0 + ratio) / 1.9
    term2 = ratio / 0.95
    C_L = term1 - math.sqrt(max(term1 * term1 - term2, 0.0))
    return min(1.0, max(0.0, C_L))


# ============================================================ C_P column stability

def C_P_column_stability(
    F_c_star: float,
    *,
    l_e: float,
    d: float,
    E_min: float,
    c: float = 0.8,
) -> float:
    """Column-stability factor per NDS §3.7.

        F_cE = 0.822 · E_min / (l_e/d)^2
        C_P = (1 + F_cE/F_c*)/(2c) - sqrt(((1 + F_cE/F_c*)/(2c))^2 - (F_cE/F_c*)/c)

    Parameters
    ----------
    F_c_star : float
        Reference compression stress multiplied by all factors except
        ``C_P`` (Pa).
    l_e : float
        Effective column length (m).
    d : float
        Least cross-section dimension (the buckling-controlling
        direction) (m).
    E_min : float
        NDS minimum modulus for stability (Pa).
    c : float, default 0.8
        Column buckling shape factor: 0.8 (sawn lumber), 0.9 (glulam,
        SCL), 0.85 (round timber).
    """
    if min(l_e, d, E_min, F_c_star, c) <= 0:
        raise ValueError("all inputs must be positive")
    slenderness = l_e / d
    # NDS §3.7.1 limits l_e/d to 50 for sawn columns; 75 during construction.
    if slenderness > 50:
        raise ValueError(
            f"l_e/d = {slenderness:.1f} exceeds NDS §3.7.1.4 limit of 50"
        )
    F_cE = 0.822 * E_min / (slenderness * slenderness)
    ratio = F_cE / F_c_star
    term1 = (1.0 + ratio) / (2.0 * c)
    term2 = ratio / c
    C_P = term1 - math.sqrt(max(term1 * term1 - term2, 0.0))
    return min(1.0, max(0.0, C_P))
