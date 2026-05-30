"""IS 875 Part 3 (2015) wind loading -- Indian standard.

Design wind speed at height ``z``::

    V_z = V_b * k_1 * k_2 * k_3 * k_4

where

* ``V_b`` -- basic wind speed (50-yr return; from IS 875 wind-zone map).
* ``k_1`` -- probability factor / risk coefficient (importance).
* ``k_2(z)`` -- terrain category + height factor (Table 2).
* ``k_3`` -- topography factor (= 1.0 on flat terrain).
* ``k_4`` -- cyclonic-region importance factor (= 1.0 standard).

Design wind pressure::

    p_z = 0.6 * V_z**2    (Pa)
"""
from __future__ import annotations

from dataclasses import dataclass


# IS 875 Part 3 Table 2 (2015): k_2 vs height for terrain categories
# 1 (open), 2 (open with scattered obstructions), 3 (suburban),
# 4 (urban). Heights in m, k_2 dimensionless.
_K2_TABLE = {
    1: [(10.0, 1.05), (15.0, 1.09), (20.0, 1.12), (30.0, 1.15),
        (50.0, 1.20), (100.0, 1.28), (150.0, 1.32), (200.0, 1.34),
        (250.0, 1.36), (300.0, 1.37), (350.0, 1.37), (400.0, 1.38),
        (450.0, 1.39), (500.0, 1.40)],
    2: [(10.0, 1.00), (15.0, 1.05), (20.0, 1.07), (30.0, 1.12),
        (50.0, 1.17), (100.0, 1.24), (150.0, 1.28), (200.0, 1.30),
        (250.0, 1.32), (300.0, 1.34), (350.0, 1.35), (400.0, 1.35),
        (450.0, 1.35), (500.0, 1.36)],
    3: [(10.0, 0.91), (15.0, 0.97), (20.0, 1.01), (30.0, 1.06),
        (50.0, 1.12), (100.0, 1.20), (150.0, 1.24), (200.0, 1.27),
        (250.0, 1.29), (300.0, 1.31), (350.0, 1.32), (400.0, 1.34),
        (450.0, 1.35), (500.0, 1.35)],
    4: [(10.0, 0.80), (15.0, 0.80), (20.0, 0.80), (30.0, 0.97),
        (50.0, 1.10), (100.0, 1.20), (150.0, 1.24), (200.0, 1.27),
        (250.0, 1.28), (300.0, 1.30), (350.0, 1.31), (400.0, 1.32),
        (450.0, 1.33), (500.0, 1.33)],
}


def is875_terrain_category_factor(z: float, category: int) -> float:
    """``k_2`` from IS 875-3 Table 2 for terrain category 1-4. Linearly
    interpolates between tabulated heights."""
    if z <= 0:
        raise ValueError(f"z must be positive, got {z}")
    if category not in _K2_TABLE:
        raise ValueError(
            f"category must be in {sorted(_K2_TABLE.keys())}, got {category}"
        )
    table = _K2_TABLE[category]
    if z <= table[0][0]:
        return float(table[0][1])
    if z >= table[-1][0]:
        return float(table[-1][1])
    for i in range(len(table) - 1):
        z0, k0 = table[i]
        z1, k1 = table[i + 1]
        if z0 <= z <= z1:
            return float(k0 + (k1 - k0) * (z - z0) / (z1 - z0))
    return float(table[-1][1])


@dataclass
class Is875DesignWindPressure:
    """Result of an IS 875 design-pressure calculation."""

    V_z: float          # design wind speed at z (m/s)
    p_z: float          # design wind pressure (Pa)
    k_2: float
    V_b: float


def is875_design_wind_pressure(
    z: float,
    V_b: float,
    *,
    category: int = 2,
    k_1: float = 1.0,
    k_3: float = 1.0,
    k_4: float = 1.0,
) -> Is875DesignWindPressure:
    """Design wind pressure ``p_z = 0.6 V_z^2`` (Pa) at height ``z`` (m)
    per IS 875-3 2015.

    Parameters
    ----------
    z : float
        Height above ground (m).
    V_b : float
        Basic wind speed from IS 875 wind-zone map (m/s; typical 33-55).
    category : 1..4, default 2
        Terrain category.
    k_1 : float, default 1.0
        Risk coefficient (probability factor); 1.0 for general
        building, 1.08 for important structures, 0.92 for temporary.
    k_3 : float, default 1.0
        Topography factor.
    k_4 : float, default 1.0
        Cyclonic-region importance factor; 1.30 for cyclonic
        post-disaster structures, 1.15 for industrial, 1.00 otherwise.
    """
    if V_b <= 0:
        raise ValueError(f"V_b must be positive, got {V_b}")
    if k_1 <= 0 or k_3 <= 0 or k_4 <= 0:
        raise ValueError("k_1, k_3, k_4 must all be positive")
    k_2 = is875_terrain_category_factor(z, category)
    V_z = V_b * k_1 * k_2 * k_3 * k_4
    p_z = 0.6 * V_z * V_z
    return Is875DesignWindPressure(
        V_z=float(V_z), p_z=float(p_z), k_2=float(k_2), V_b=float(V_b),
    )
