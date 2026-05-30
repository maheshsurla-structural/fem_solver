"""EN 1998-1 (Eurocode 8) seismic design.

Implements the equivalent-static-force procedure per EN 1998-1:2004
+ A1:2013:

* **Design response spectrum** (Cl. 3.2.2.5) for the five
  ground types A-E and the two spectrum shapes (Type 1: M >= 5.5,
  for high-seismicity regions; Type 2: M < 5.5).
* **Behaviour factor q** -- the Eurocode 8 equivalent of ACI's R
  / IS 1893's R: typical values 1.5 (uncracked, no detailing) to
  6 (high-ductility DCH frames).
* **Base shear** ``F_b = S_d(T_1) · m · lambda``  (Cl. 4.3.3.2.2).
* **Vertical distribution** ``F_i = F_b · z_i m_i / Sum(z_j m_j)``
  (Cl. 4.3.3.2.3).
* **Storey drift check** ``d_r * nu <= 0.005 h`` (Cl. 4.4.3.2).

References
----------
* EN 1998-1:2004 + A1:2013. *Eurocode 8: Design of structures for
  earthquake resistance - Part 1: General rules, seismic actions and
  rules for buildings*.
* Fardis, Carvalho, Elnashai, Faccioli, Pinto, Plumier (2005).
  *Designers' Guide to EN 1998-1 and EN 1998-5*. Thomas Telford.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ============================================================ ground types

# EN 1998-1 Table 3.2 (Type 1 spectrum)
_GROUND_TYPE_PARAMS_TYPE1 = {
    "A": (1.00, 0.15, 0.4, 2.0),
    "B": (1.20, 0.15, 0.5, 2.0),
    "C": (1.15, 0.20, 0.6, 2.0),
    "D": (1.35, 0.20, 0.8, 2.0),
    "E": (1.40, 0.15, 0.5, 2.0),
}

# Type 2 (low / moderate seismicity, M < 5.5)
_GROUND_TYPE_PARAMS_TYPE2 = {
    "A": (1.00, 0.05, 0.25, 1.2),
    "B": (1.35, 0.05, 0.25, 1.2),
    "C": (1.50, 0.10, 0.25, 1.2),
    "D": (1.80, 0.10, 0.30, 1.2),
    "E": (1.60, 0.05, 0.25, 1.2),
}


def ground_type_parameters(
    ground_type: str, *, spectrum_type: int = 1,
) -> tuple[float, float, float, float]:
    """Return ``(S, T_B, T_C, T_D)`` for the given ground type and
    spectrum shape (per Tables 3.2 / 3.3 of EN 1998-1).

    Parameters
    ----------
    ground_type : str
        One of ``"A"`` (rock), ``"B"`` (stiff soil), ``"C"`` (medium),
        ``"D"`` (soft), ``"E"`` (alluvium over rock).
    spectrum_type : {1, 2}, default 1
        Type 1 for high-seismicity / large-magnitude events (M >= 5.5);
        Type 2 for low-seismicity / small events (M < 5.5).
    """
    if ground_type not in ("A", "B", "C", "D", "E"):
        raise ValueError(
            f"ground_type must be A/B/C/D/E, got {ground_type!r}"
        )
    if spectrum_type not in (1, 2):
        raise ValueError("spectrum_type must be 1 or 2")
    table = (_GROUND_TYPE_PARAMS_TYPE1 if spectrum_type == 1
             else _GROUND_TYPE_PARAMS_TYPE2)
    return table[ground_type]


# ============================================================ design spectrum

def design_spectrum_Sd(
    T: float,
    *,
    a_g: float,
    ground_type: str = "C",
    q: float = 1.5,
    spectrum_type: int = 1,
    beta: float = 0.20,
) -> float:
    """Design pseudo-spectral acceleration ``S_d(T)`` for elastic
    response, per EN 1998-1 Cl. 3.2.2.5 Eq. 3.13-16.

    Parameters
    ----------
    T : float
        Natural period (s).
    a_g : float
        Design ground acceleration on type-A rock (m/s^2). Typically
        between 0.5 and 4 m/s^2 across European seismic regions.
    ground_type : {"A", "B", "C", "D", "E"}, default "C"
    q : float, default 1.5
        Behaviour (reduction) factor. Limited to 1.5 in DCL practice;
        3.0 (DCM RC frame), 4.0 (DCM steel frame), up to 6.0 (DCH).
    spectrum_type : {1, 2}, default 1
    beta : float, default 0.20
        Lower-bound factor on the design spectrum (EN 1998-1 Cl. 3.2.2.5).
        S_d shall not fall below ``beta · a_g``.

    Returns
    -------
    S_d : float
        Pseudo-spectral acceleration (m/s^2).
    """
    if T < 0.0:
        raise ValueError("T must be >= 0")
    if a_g <= 0.0:
        raise ValueError("a_g must be > 0")
    if q < 1.0:
        raise ValueError("q must be >= 1.0")
    S, T_B, T_C, T_D = ground_type_parameters(
        ground_type, spectrum_type=spectrum_type,
    )
    if T <= T_B:
        # Rising branch (linear)
        S_d = a_g * S * (2.0 / 3.0
                          + T / T_B * (2.5 / q - 2.0 / 3.0))
    elif T <= T_C:
        # Plateau
        S_d = a_g * S * 2.5 / q
    elif T <= T_D:
        # 1/T decay
        S_d = a_g * S * 2.5 / q * T_C / T
        S_d = max(S_d, beta * a_g)
    else:
        # 1/T^2 decay
        S_d = a_g * S * 2.5 / q * T_C * T_D / T ** 2
        S_d = max(S_d, beta * a_g)
    return float(S_d)


# ============================================================ behaviour factor q

# Per EN 1998-1 Table 5.1 (RC) and Table 6.2 (steel), simplified.
_BEHAVIOUR_FACTOR_DEFAULTS = {
    "RC_DCL":           1.5,
    "RC_DCM_FRAME":     3.0,
    "RC_DCH_FRAME":     4.5,
    "RC_DCM_WALL":      3.0,
    "RC_DCH_WALL":      4.0,
    "STEEL_DCL":        1.5,
    "STEEL_DCM_FRAME":  4.0,
    "STEEL_DCH_FRAME":  6.0,
    "STEEL_DCM_BRACED": 4.0,
    "STEEL_DCH_BRACED": 5.0,
}


def behaviour_factor_default(system: str) -> float:
    """Default ``q`` for a named lateral system.

    Categories: ``DCL`` (low ductility), ``DCM`` (medium), ``DCH`` (high).
    """
    if system not in _BEHAVIOUR_FACTOR_DEFAULTS:
        raise ValueError(
            f"unknown system {system!r}; available: "
            f"{sorted(_BEHAVIOUR_FACTOR_DEFAULTS)}"
        )
    return float(_BEHAVIOUR_FACTOR_DEFAULTS[system])


# ============================================================ base shear

@dataclass
class EC8BaseShearResult:
    """Result of an EC8 equivalent-static base-shear calculation."""

    F_b: float
    S_d: float
    T_1: float
    m_total: float
    lambda_factor: float


def ec8_base_shear(
    *,
    T_1: float,
    m_total: float,
    a_g: float,
    ground_type: str = "C",
    q: float = 1.5,
    spectrum_type: int = 1,
    n_storeys: int = 1,
    importance: float = 1.0,
) -> EC8BaseShearResult:
    """Design seismic base shear ``F_b = S_d(T_1) · m · lambda``
    (EN 1998-1 Cl. 4.3.3.2.2).

    Parameters
    ----------
    T_1 : float
        Fundamental period of the building (s).
    m_total : float
        Total seismic mass = M_DL + psi_E · M_LL (kg).
    a_g : float
        Design ground acceleration (m/s^2).
    ground_type, q, spectrum_type : as for :func:`design_spectrum_Sd`.
    n_storeys : int, default 1
        Used for the EC8 lambda correction factor.
    importance : float, default 1.0
        Importance factor γ_I (1.0 for class II, 1.2 for class III,
        1.4 for class IV).
    """
    if m_total <= 0.0:
        raise ValueError("m_total must be > 0")
    Sd = design_spectrum_Sd(
        T_1, a_g=a_g, ground_type=ground_type, q=q,
        spectrum_type=spectrum_type,
    )
    # lambda correction (Cl. 4.3.3.2.2 Note 3): 0.85 if n >= 3 and
    # T_1 <= 2 T_C; else 1.0.
    _, _, T_C, _ = ground_type_parameters(
        ground_type, spectrum_type=spectrum_type,
    )
    lam = 0.85 if (n_storeys >= 3 and T_1 <= 2.0 * T_C) else 1.0
    F_b = Sd * m_total * lam * importance
    return EC8BaseShearResult(
        F_b=float(F_b), S_d=float(Sd),
        T_1=float(T_1), m_total=float(m_total),
        lambda_factor=float(lam),
    )


# ============================================================ vertical distribution

def vertical_force_distribution(
    *,
    F_b: float,
    storey_masses: np.ndarray,
    storey_heights: np.ndarray,
) -> np.ndarray:
    """Distribute ``F_b`` over storeys per EN 1998-1 Cl. 4.3.3.2.3::

        F_i = F_b · (m_i · z_i) / Sum(m_j · z_j)

    (Linear inverted-triangle, equivalent to assuming the
    fundamental mode shape varies linearly with height.)
    """
    m = np.asarray(storey_masses, dtype=float)
    h = np.asarray(storey_heights, dtype=float)
    if m.shape != h.shape:
        raise ValueError("storey_masses and storey_heights must have "
                         "the same shape")
    if np.any(m <= 0.0) or np.any(h <= 0.0):
        raise ValueError("masses and heights must all be > 0")
    products = m * h
    return F_b * products / products.sum()


# ============================================================ drift check

@dataclass
class EC8DriftResult:
    """Storey-drift check per EN 1998-1 Cl. 4.4.3.2.

    Allowable storey-drift ratio (damage-limitation state):
        nu · d_r / h <= 0.005    (brittle infill)
                        0.0075   (ductile infill)
                        0.010    (no infill)

    where nu = 0.4 (importance class III/IV) or 0.5 (II).
    """

    storey_drifts: np.ndarray
    drift_ratios: np.ndarray
    max_ratio: float
    passes: bool
    limit: float


def ec8_drift_check(
    *,
    floor_disp: np.ndarray,
    storey_heights: np.ndarray,
    q: float,
    importance_class: str = "II",
    infill_type: str = "brittle",
) -> EC8DriftResult:
    """Storey-drift check per EN 1998-1 Cl. 4.4.3.2.

    The elastic interstory drift ``d_r,e`` from the analysis (with
    spectrum reduced by ``q``) is amplified by ``q · nu`` to get the
    design-level deflection used in the limit check.

    Parameters
    ----------
    floor_disp : array
        Elastic floor displacements (m).
    storey_heights : array
        Per-storey heights (m).
    q : float
        Behaviour factor used in the analysis.
    importance_class : {"I", "II", "III", "IV"}, default "II"
    infill_type : {"brittle", "ductile", "no_infill"}, default "brittle"
    """
    nu = {"I": 0.5, "II": 0.5, "III": 0.4, "IV": 0.4}.get(
        importance_class, 0.5
    )
    limit = {
        "brittle": 0.005,
        "ductile": 0.0075,
        "no_infill": 0.010,
    }.get(infill_type, 0.005)
    u = np.asarray(floor_disp, dtype=float)
    h = np.asarray(storey_heights, dtype=float)
    if u.shape != h.shape:
        raise ValueError("floor_disp and storey_heights must have "
                         "the same shape")
    drifts = np.empty_like(u)
    drifts[0] = u[0]
    drifts[1:] = np.diff(u)
    drifts = np.abs(drifts)
    # Amplify by q · nu, then check
    ratios = (q * nu * drifts) / h
    return EC8DriftResult(
        storey_drifts=drifts,
        drift_ratios=ratios,
        max_ratio=float(ratios.max()),
        passes=bool(ratios.max() <= limit),
        limit=float(limit),
    )
