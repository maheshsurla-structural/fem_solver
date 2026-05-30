"""IS 1893 (Part 1) — Criteria for Earthquake-Resistant Design of
Structures (2016 / 2002).

This module implements the equivalent-static-force procedure (Cl. 7):

* **Design spectrum** ``S_a/g(T)`` for 5%-damped response, with the
  three IS soil types I (rock/hard), II (medium), and III (soft).
* **Design horizontal acceleration coefficient**
  ``A_h = (Z/2)·(I/R)·(S_a/g)`` (Cl. 6.4.2).
* **Base shear** ``V_B = A_h W`` (Cl. 7.5.3).
* **Vertical distribution of base shear** Q_i (Cl. 7.6.3) over storeys.
* **Empirical period formulas** (Cl. 7.6.2): 0.075 h^0.75 for RC MRF,
  0.085 h^0.75 for steel MRF, 0.09 h/sqrt(d) for braced/infill.
* **Drift check** (Cl. 7.11.1): allowable storey-drift ratio 0.004.

References
----------
* IS 1893 (Part 1):2016. *Criteria for Earthquake-Resistant Design --
  Part 1: General Provisions and Buildings*. BIS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ============================================================ tables

# Table 2: Zone Factor
_Z_TABLE = {2: 0.10, 3: 0.16, 4: 0.24, 5: 0.36}

# Table 8: Importance Factors
IMPORTANCE_DEFAULT = 1.0
IMPORTANCE_CRITICAL = 1.5

# Response reduction R (Table 9, common cases)
R_VALUES = {
    "RC_OMRF":            3.0,
    "RC_SMRF":            5.0,
    "RC_DUCTILE_SHEAR":   4.0,
    "STEEL_BRACED":       4.0,
    "STEEL_SMRF":         5.0,
    "STEEL_CBF":          4.0,
    "STEEL_EBF":          5.0,
    "MASONRY":            3.0,
}


def zone_factor(zone: int) -> float:
    """Z from Table 2: zones II..V."""
    if zone not in _Z_TABLE:
        raise ValueError(
            f"zone must be one of {sorted(_Z_TABLE.keys())}, got {zone}"
        )
    return _Z_TABLE[zone]


# ============================================================ design spectrum

def design_spectrum_Sa_g(T: float, *, soil_type: int = 1) -> float:
    """5%-damped design pseudo-spectral acceleration S_a/g for IS 1893.

    Parameters
    ----------
    T : float
        Natural period (s).
    soil_type : int
        1 = Rocky / Hard, 2 = Medium, 3 = Soft.

    Returns
    -------
    Sa_g : float
        Spectral acceleration divided by g (dimensionless).

    Notes
    -----
    Type I:  T < 0.10:  1 + 15T;  0.10..0.40: 2.50;  T > 0.40: 1.00/T
    Type II: T < 0.10:  1 + 15T;  0.10..0.55: 2.50;  T > 0.55: 1.36/T
    Type III:T < 0.10:  1 + 15T;  0.10..0.67: 2.50;  T > 0.67: 1.67/T
    """
    if T < 0.0:
        raise ValueError(f"T must be >= 0, got {T}")
    if soil_type not in (1, 2, 3):
        raise ValueError(f"soil_type must be 1/2/3, got {soil_type}")
    if T < 0.10:
        return 1.0 + 15.0 * T
    # Plateau and decay corners
    if soil_type == 1:
        T_corner, coef = 0.40, 1.00
    elif soil_type == 2:
        T_corner, coef = 0.55, 1.36
    else:
        T_corner, coef = 0.67, 1.67
    if T <= T_corner:
        return 2.50
    return float(coef / T)


def Ah_coefficient(
    *,
    T: float,
    zone: int,
    importance: float,
    R: float,
    soil_type: int = 1,
) -> dict:
    """Design horizontal acceleration coefficient ``A_h``.

    Returns
    -------
    dict
        ``{"A_h": ..., "Z": ..., "I": ..., "R": ..., "Sa_g": ...,
        "T": ..., "soil_type": ...}``
    """
    if importance <= 0.0:
        raise ValueError("importance I must be > 0")
    if R <= 0.0:
        raise ValueError("R must be > 0")
    Z = zone_factor(zone)
    Sa_g = design_spectrum_Sa_g(T, soil_type=soil_type)
    A_h = (Z / 2.0) * (importance / R) * Sa_g
    return {
        "A_h": float(A_h),
        "Z": float(Z),
        "I": float(importance),
        "R": float(R),
        "Sa_g": float(Sa_g),
        "T": float(T),
        "soil_type": int(soil_type),
    }


# ============================================================ time period

def empirical_period(*, h: float, system: str = "RC_MRF",
                       d: float | None = None) -> float:
    """Empirical fundamental period ``T_a`` per IS 1893 Cl. 7.6.2.

    Parameters
    ----------
    h : float
        Building height (m).
    system : str
        ``"RC_MRF"``, ``"STEEL_MRF"``, ``"BRACED_INFILL"``.
    d : float, optional
        Base dimension parallel to ground motion (m). Required for
        ``"BRACED_INFILL"``.
    """
    if h <= 0.0:
        raise ValueError("h must be > 0")
    if system == "RC_MRF":
        return 0.075 * h ** 0.75
    if system == "STEEL_MRF":
        return 0.085 * h ** 0.75
    if system == "BRACED_INFILL":
        if d is None or d <= 0.0:
            raise ValueError("d (base dimension) required for "
                             "BRACED_INFILL")
        return 0.09 * h / math.sqrt(d)
    raise ValueError(f"unknown system {system!r}")


# ============================================================ base shear and distribution

@dataclass
class IS1893BaseShearResult:
    """Result of an equivalent-static base-shear calculation."""

    V_B: float                # total base shear (N)
    A_h: float
    Z: float
    I: float
    R: float
    Sa_g: float
    T: float
    W: float                  # seismic weight (N)


def is1893_base_shear(
    *,
    T: float, W: float,
    zone: int, importance: float, R: float,
    soil_type: int = 1,
) -> IS1893BaseShearResult:
    """Compute the design seismic base shear V_B = A_h W.

    Parameters
    ----------
    T : float
    W : float
        Total seismic weight of the building (N).
    """
    if W <= 0.0:
        raise ValueError("W must be > 0")
    ah = Ah_coefficient(T=T, zone=zone, importance=importance,
                          R=R, soil_type=soil_type)
    V_B = ah["A_h"] * W
    return IS1893BaseShearResult(
        V_B=V_B, A_h=ah["A_h"], Z=ah["Z"],
        I=ah["I"], R=ah["R"], Sa_g=ah["Sa_g"],
        T=T, W=W,
    )


def vertical_force_distribution(
    *,
    V_B: float,
    storey_weights: np.ndarray,
    storey_heights: np.ndarray,
) -> np.ndarray:
    """Distribute V_B over storeys per IS 1893 Cl. 7.6.3.

    Q_i = V_B · (W_i h_i^2) / sum_j(W_j h_j^2).

    Parameters
    ----------
    V_B : float
        Total base shear (N).
    storey_weights : array
        Seismic weight at each floor (N), ordered low to high.
    storey_heights : array
        Floor elevations above ground (m), ordered low to high.

    Returns
    -------
    Q : np.ndarray
        Floor-level lateral force (N).
    """
    W = np.asarray(storey_weights, dtype=float)
    h = np.asarray(storey_heights, dtype=float)
    if W.shape != h.shape:
        raise ValueError("storey_weights and storey_heights must "
                         "have the same shape")
    if np.any(W <= 0.0) or np.any(h <= 0.0):
        raise ValueError("storey weights and heights must all be > 0")
    products = W * h ** 2
    return V_B * products / products.sum()


# ============================================================ drift check

@dataclass
class IS1893DriftResult:
    """Storey-drift ratio check per IS 1893 Cl. 7.11.1.

    Allowable storey-drift ratio: 0.004 of the storey height (under
    factored loads with R = 1, i.e., the "design earthquake"
    deflection -- usually obtained by multiplying the elastic
    deflection by R).
    """

    storey_drifts: np.ndarray         # absolute drifts (m)
    storey_heights: np.ndarray        # m
    drift_ratios: np.ndarray
    max_ratio: float
    passes: bool
    limit: float = 0.004


def is1893_drift_check(
    *,
    floor_disp: np.ndarray,
    storey_heights: np.ndarray,
    R: float,
    limit: float = 0.004,
) -> IS1893DriftResult:
    """Check storey-drift ratios against IS 1893 limit (0.4% by default).

    The input ``floor_disp`` is the elastic floor lateral displacement
    from the analysis (R=1). It is multiplied by ``R`` to obtain the
    design-earthquake deflection.

    Parameters
    ----------
    floor_disp : array
        Elastic lateral displacements at each floor (m), low to high.
        Base assumed at 0.
    storey_heights : array
        Per-storey height (m), same length as ``floor_disp``.
    R : float
    limit : float, default 0.004
    """
    u = np.asarray(floor_disp, dtype=float)
    h = np.asarray(storey_heights, dtype=float)
    if u.shape != h.shape:
        raise ValueError("floor_disp and storey_heights must "
                         "have the same shape")
    u_design = R * u
    drifts = np.empty_like(u_design)
    drifts[0] = u_design[0]
    drifts[1:] = np.diff(u_design)
    drifts = np.abs(drifts)
    ratios = drifts / h
    return IS1893DriftResult(
        storey_drifts=drifts,
        storey_heights=h,
        drift_ratios=ratios,
        max_ratio=float(ratios.max()),
        passes=bool(ratios.max() <= limit),
        limit=limit,
    )
