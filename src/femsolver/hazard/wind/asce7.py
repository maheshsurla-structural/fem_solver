"""ASCE 7-22 wind loads -- velocity pressure, pressure coefficients,
gust factor, MWFRS design pressures.

This module implements the Main Wind Force Resisting System (MWFRS)
calculations of ASCE 7-22 Chapter 26-27 for enclosed rectangular
buildings on flat terrain.

Notation
--------

* ``V``      Basic wind speed (m/s) from the ASCE 7 hazard map.
* ``K_z``    Velocity pressure exposure coefficient (height-dependent).
* ``K_zt``   Topographic factor (= 1.0 on flat terrain).
* ``K_d``    Directionality factor (= 0.85 for buildings).
* ``K_e``    Ground-elevation factor (= 1.0 typically).
* ``q_z``    Velocity pressure at height z (Pa).
* ``G``      Gust effect factor (0.85 for rigid buildings).
* ``C_p``    External pressure coefficient.
* ``GC_pi``  Internal pressure coefficient (signed).

Sign convention -- pressures positive when acting **into** the
surface (windward face). Suction (acting **out of** the surface) is
negative.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ============================================================ exposure

#: Exposure categories per ASCE 7-22 26.7.3
#:
#: * **B** -- urban / suburban (closely spaced obstructions of single-
#:   family-dwelling size or larger)
#: * **C** -- open terrain with scattered obstructions less than 9 m tall
#: * **D** -- flat unobstructed areas, water surfaces
#:
#: Each entry: (alpha, z_g_m) where ``alpha`` is the inverse power-law
#: exponent and ``z_g_m`` is the gradient height (m).
_EXPOSURE_CONSTANTS = {
    "B": (7.0, 365.76),     # 1200 ft
    "C": (9.5, 274.32),     # 900 ft
    "D": (11.5, 213.36),    # 700 ft
}


def asce7_exposure_constants(exposure: str) -> tuple[float, float]:
    """Return ``(alpha, z_g)`` for ASCE 7-22 exposure category.

    Parameters
    ----------
    exposure : {"B", "C", "D"}
    """
    e = exposure.upper()
    if e not in _EXPOSURE_CONSTANTS:
        raise ValueError(
            f"exposure must be one of {list(_EXPOSURE_CONSTANTS)}, "
            f"got {exposure!r}"
        )
    return _EXPOSURE_CONSTANTS[e]


def asce7_Kz(z: float, exposure: str) -> float:
    """Velocity-pressure exposure coefficient ``K_z`` (ASCE 7-22
    Table 26.10-1).

    For ``4.57 m <= z <= z_g``::

        K_z = 2.41 * (z / z_g) ** (2 / alpha)

    For ``z < 4.57 m`` the floor at z = 4.57 m (15 ft) is used.
    """
    if z <= 0.0:
        raise ValueError(f"z must be positive, got {z}")
    alpha, z_g = asce7_exposure_constants(exposure)
    z_floor = max(z, 4.572)        # 15 ft = 4.572 m
    return 2.41 * (z_floor / z_g) ** (2.0 / alpha)


# ============================================================ q_z

@dataclass
class Asce7VelocityPressure:
    """Result of a velocity-pressure calculation.

    Attributes
    ----------
    q_z : float
        Velocity pressure at the queried height (Pa).
    K_z : float
        Exposure coefficient at that height.
    V : float
        Basic wind speed (m/s).
    exposure : str
    """

    q_z: float
    K_z: float
    V: float
    exposure: str


def asce7_velocity_pressure(
    z: float,
    V: float,
    *,
    exposure: str = "C",
    K_zt: float = 1.0,
    K_d: float = 0.85,
    K_e: float = 1.0,
) -> Asce7VelocityPressure:
    """ASCE 7-22 velocity pressure at height ``z``::

        q_z = 0.613 * K_z * K_zt * K_d * K_e * V**2    (Pa)

    Parameters
    ----------
    z : float
        Height above ground (m).
    V : float
        Basic wind speed (m/s).
    exposure : {"B", "C", "D"}, default "C"
    K_zt : float, default 1.0
        Topographic factor (1.0 = flat terrain).
    K_d : float, default 0.85
        Directionality factor.
    K_e : float, default 1.0
        Ground elevation factor.
    """
    if V <= 0:
        raise ValueError(f"V must be positive, got {V}")
    if K_zt <= 0 or K_d <= 0 or K_e <= 0:
        raise ValueError("K_zt, K_d, K_e must all be positive")
    K_z = asce7_Kz(z, exposure)
    q_z = 0.613 * K_z * K_zt * K_d * K_e * V * V
    return Asce7VelocityPressure(
        q_z=float(q_z), K_z=float(K_z), V=float(V), exposure=exposure.upper(),
    )


# ============================================================ Cp

@dataclass
class Asce7WallPressureCoefficients:
    """Wall ``C_p`` per ASCE 7-22 Figure 27.3-1.

    All coefficients are positive when acting *into* the building face.

    Attributes
    ----------
    windward : float
        Always ``+0.8``.
    leeward : float
        Depends on plan aspect ratio L/B: ``-0.5`` (L/B <= 1),
        ``-0.3`` (L/B = 2), ``-0.2`` (L/B >= 4).
    side : float
        Always ``-0.7``.
    """

    windward: float
    leeward: float
    side: float


def asce7_wall_Cp(L_over_B: float) -> Asce7WallPressureCoefficients:
    """Wall pressure coefficients for a rectangular building of plan
    aspect ratio ``L / B`` (parallel to wind / perpendicular to wind)."""
    if L_over_B <= 0:
        raise ValueError(f"L_over_B must be positive, got {L_over_B}")
    if L_over_B <= 1.0:
        leeward = -0.5
    elif L_over_B >= 4.0:
        leeward = -0.2
    elif L_over_B <= 2.0:
        # interpolate linearly from -0.5 at 1 to -0.3 at 2
        leeward = -0.5 + 0.2 * (L_over_B - 1.0)
    else:
        # interpolate linearly from -0.3 at 2 to -0.2 at 4
        leeward = -0.3 + 0.1 * (L_over_B - 2.0) / 2.0
    return Asce7WallPressureCoefficients(
        windward=0.8, leeward=float(leeward), side=-0.7,
    )


def asce7_roof_Cp_flat(h_over_L: float) -> tuple[float, float]:
    """Flat-roof external pressure coefficients (zones near the
    windward edge) per ASCE 7-22 Figure 27.3-1.

    Returns ``(Cp_zone_near_edge, Cp_zone_far)``. Both are negative
    (suction) for flat roofs.
    """
    if h_over_L <= 0:
        raise ValueError(f"h_over_L must be positive, got {h_over_L}")
    if h_over_L <= 0.5:
        return (-0.9, -0.5)
    if h_over_L >= 1.0:
        return (-1.3, -0.7)
    # Linear interpolate
    near = -0.9 + (-0.4) * (h_over_L - 0.5) / 0.5
    far  = -0.5 + (-0.2) * (h_over_L - 0.5) / 0.5
    return (float(near), float(far))


# ============================================================ gust factor

def asce7_gust_factor_rigid() -> float:
    """For *rigid* buildings (fundamental natural frequency >= 1 Hz)
    ASCE 7-22 allows ``G = 0.85``."""
    return 0.85


def asce7_gust_factor_flexible(
    *,
    f1: float,
    zeta: float,
    h: float,
    B: float,
    L: float,
    V_bar_z: float,
    exposure: str = "C",
) -> float:
    """Gust-effect factor for *flexible* buildings (n1 < 1 Hz) per
    ASCE 7-22 26.11.

    Parameters
    ----------
    f1 : float
        Building fundamental natural frequency (Hz).
    zeta : float
        Damping ratio (fraction of critical, typically 0.01-0.02).
    h, B, L : float
        Mean roof height, building width perpendicular to wind, and
        building plan length parallel to wind (all m).
    V_bar_z : float
        Mean wind speed at height ``z_bar = 0.6 h`` (m/s).
    exposure : {"B", "C", "D"}
    """
    if f1 <= 0:
        raise ValueError(f"f1 must be positive, got {f1}")
    if not 0.0 < zeta < 1.0:
        raise ValueError(f"zeta must be in (0, 1), got {zeta}")
    # ASCE 7 exposure-dependent constants
    EXP_PARAMS = {
        "B": dict(c=0.30, l_=98.0, eps_=0.33, alpha_bar=0.25),
        "C": dict(c=0.20, l_=152.0, eps_=0.20, alpha_bar=1.0 / 6.5),
        "D": dict(c=0.15, l_=198.0, eps_=0.125, alpha_bar=0.111),
    }
    p = EXP_PARAMS[exposure.upper()]
    z_bar = max(0.6 * h, 9.144)        # 30 ft
    # Intensity of turbulence
    I_z = p["c"] * (10.0 / z_bar) ** (1.0 / 6.0)
    # Integral length scale
    L_z = p["l_"] * (z_bar / 10.0) ** p["eps_"]
    # Background response
    Q = math.sqrt(1.0 / (1.0 + 0.63 * ((B + h) / L_z) ** 0.63))
    # Resonant response
    N1 = f1 * L_z / V_bar_z
    R_n = 7.47 * N1 / (1.0 + 10.3 * N1) ** (5.0 / 3.0)

    def _Rh(eta):
        if eta < 1e-6:
            return 1.0
        return (1.0 / eta) - (1.0 / (2.0 * eta ** 2)) * (1.0 - math.exp(-2.0 * eta))

    eta_h = 4.6 * f1 * h / V_bar_z
    eta_B = 4.6 * f1 * B / V_bar_z
    eta_L = 15.4 * f1 * L / V_bar_z
    R_h = _Rh(eta_h)
    R_B = _Rh(eta_B)
    R_L = _Rh(eta_L)
    R = math.sqrt((1.0 / zeta) * R_n * R_h * R_B * (0.53 + 0.47 * R_L))
    # gQ, gR peak factors (text book values per ASCE 7)
    g_Q = 3.4
    g_R = math.sqrt(2.0 * math.log(3600.0 * f1)) \
        + 0.577 / math.sqrt(2.0 * math.log(3600.0 * f1))
    G_f = 0.925 * ((1.0 + 1.7 * I_z * math.sqrt(g_Q ** 2 * Q ** 2 + g_R ** 2 * R ** 2))
                    / (1.0 + 1.7 * 3.4 * I_z))
    return float(G_f)


# ============================================================ MWFRS

@dataclass
class Asce7MWFRSResult:
    """Combined design pressures for a rectangular building.

    All pressures positive into the surface, negative out (suction).
    """

    p_windward: float       # at the height in question
    p_leeward: float
    p_side: float
    p_roof_near: float      # near windward edge
    p_roof_far: float
    G: float
    GC_pi: float            # net internal coefficient used
    q_z: float
    q_h: float              # velocity pressure at roof height


def asce7_mwfrs_design_pressures(
    *,
    z: float,
    h: float,
    V: float,
    L: float,
    B: float,
    exposure: str = "C",
    K_zt: float = 1.0,
    K_d: float = 0.85,
    K_e: float = 1.0,
    G: float = 0.85,
    GC_pi: float = 0.18,   # +/- value; positive = pressure outward inside
) -> Asce7MWFRSResult:
    """Combined external + internal MWFRS design pressures at height
    ``z`` for a rectangular enclosed building.

    Formula (ASCE 7-22 27.3.1)::

        p_windward = q_z * G * C_p - q_i * (GC_pi)
        p_leeward  = q_h * G * C_p - q_i * (GC_pi)
        p_side     = q_h * G * C_p - q_i * (GC_pi)
        p_roof     = q_h * G * C_p - q_i * (GC_pi)

    where ``q_i = q_h`` for enclosed buildings.

    Parameters
    ----------
    z : float
        Height at which the windward-wall pressure is reported (m).
    h : float
        Mean roof height (m).
    V : float
        Basic wind speed (m/s).
    L, B : float
        Plan length (parallel to wind) and width (perpendicular).
    G : float
        Gust factor; default 0.85 (rigid).
    GC_pi : float
        Internal pressure coefficient magnitude; +/- this value gives
        the two design cases. The returned pressures use ``+GC_pi``
        (internal pressure pushing outward, i.e. *negative* on the
        internal side).
    """
    vp_z = asce7_velocity_pressure(
        z, V, exposure=exposure, K_zt=K_zt, K_d=K_d, K_e=K_e,
    )
    vp_h = asce7_velocity_pressure(
        h, V, exposure=exposure, K_zt=K_zt, K_d=K_d, K_e=K_e,
    )
    q_z = vp_z.q_z
    q_h = vp_h.q_z
    q_i = q_h     # enclosed assumption
    walls = asce7_wall_Cp(L / B)
    roof_near, roof_far = asce7_roof_Cp_flat(h / L)
    # internal pressure contribution acting outward on the internal
    # face of the wall = q_i * (+GC_pi). For windward wall the net is
    # external pressure (into wall) minus this -> subtract q_i*GC_pi.
    p_windward = q_z * G * walls.windward - q_i * GC_pi
    p_leeward  = q_h * G * walls.leeward  - q_i * GC_pi
    p_side     = q_h * G * walls.side     - q_i * GC_pi
    p_roof_near = q_h * G * roof_near - q_i * GC_pi
    p_roof_far  = q_h * G * roof_far  - q_i * GC_pi
    return Asce7MWFRSResult(
        p_windward=float(p_windward),
        p_leeward=float(p_leeward),
        p_side=float(p_side),
        p_roof_near=float(p_roof_near),
        p_roof_far=float(p_roof_far),
        G=float(G), GC_pi=float(GC_pi),
        q_z=float(q_z), q_h=float(q_h),
    )
