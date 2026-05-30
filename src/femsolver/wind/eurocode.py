"""EN 1991-1-4 (Eurocode 1 Part 1-4) wind loading basics.

Computes the peak velocity pressure ``q_p(z)`` (Pa) from the basic
velocity ``v_b`` (10-min mean at 10 m height) via the roughness
factor ``c_r(z)`` and turbulence intensity ``I_v(z)``.

Reference equations (Sections 4.3-4.5):

* ``v_m(z)   = c_r(z) * c_o(z) * v_b``
* ``I_v(z)   = k_l / [c_o(z) * ln(z / z_0)]``
* ``q_p(z)   = [1 + 7 I_v(z)] * 0.5 * rho * v_m(z)**2``

where ``rho = 1.25 kg/m^3`` is air density at standard pressure.

Terrain categories
------------------
==========  =========  ============
Category    z_0 (m)    z_min (m)
0 (sea)     0.003      1
I           0.01       1
II          0.05       2
III         0.3        5
IV (urban)  1.0        10
==========  =========  ============
"""
from __future__ import annotations

import math
from dataclasses import dataclass


_TERRAIN = {
    "0":  (0.003, 1.0),
    "I":  (0.01,  1.0),
    "II": (0.05,  2.0),
    "III": (0.3,  5.0),
    "IV": (1.0,  10.0),
}


def ec1_roughness_factor(z: float, terrain: str = "II") -> float:
    """``c_r(z) = k_r * ln(max(z, z_min) / z_0)`` with
    ``k_r = 0.19 (z_0 / z_0II) ** 0.07``."""
    if z <= 0:
        raise ValueError(f"z must be positive, got {z}")
    if terrain not in _TERRAIN:
        raise ValueError(
            f"terrain must be one of {list(_TERRAIN)}, got {terrain!r}"
        )
    z_0, z_min = _TERRAIN[terrain]
    z_0II = _TERRAIN["II"][0]
    k_r = 0.19 * (z_0 / z_0II) ** 0.07
    z_use = max(z, z_min)
    return float(k_r * math.log(z_use / z_0))


@dataclass
class Ec1PeakVelocityPressure:
    """Result of an EC1 peak-velocity-pressure calculation."""

    q_p: float            # peak velocity pressure (Pa)
    v_m: float            # mean wind velocity at z (m/s)
    I_v: float            # turbulence intensity at z (dimensionless)
    c_r: float            # roughness factor


def ec1_peak_velocity_pressure(
    z: float,
    v_b: float,
    *,
    terrain: str = "II",
    c_o: float = 1.0,
    k_l: float = 1.0,
    rho: float = 1.25,
) -> Ec1PeakVelocityPressure:
    """Peak velocity pressure ``q_p(z)`` per EN 1991-1-4 4.5.

    Parameters
    ----------
    z : float
        Reference height (m).
    v_b : float
        Basic wind velocity (m/s; 10-min mean at 10 m).
    terrain : {"0","I","II","III","IV"}, default "II"
        Terrain category.
    c_o : float, default 1.0
        Orography factor (1.0 = flat terrain).
    k_l : float, default 1.0
        Turbulence factor.
    rho : float, default 1.25
        Air density (kg/m^3).
    """
    if v_b <= 0:
        raise ValueError(f"v_b must be positive, got {v_b}")
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")
    z_0, z_min = _TERRAIN[terrain]
    c_r = ec1_roughness_factor(z, terrain)
    v_m = c_r * c_o * v_b
    z_use = max(z, z_min)
    I_v = k_l / (c_o * math.log(z_use / z_0))
    q_p = (1.0 + 7.0 * I_v) * 0.5 * rho * v_m * v_m
    return Ec1PeakVelocityPressure(
        q_p=float(q_p), v_m=float(v_m),
        I_v=float(I_v), c_r=float(c_r),
    )
