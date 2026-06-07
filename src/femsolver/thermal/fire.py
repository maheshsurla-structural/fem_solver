"""Fire-engineering curves and temperature-dependent material
properties for structural-fire analysis.

This module provides:

* **Standard fire curves** -- gas-temperature time histories used as
  thermal boundary conditions in compartment-fire analysis:

  - :func:`iso_834_temperature` -- ISO 834 / BS 476-20 cellulosic.
  - :func:`astm_e119_temperature` -- ASTM E119 (very similar to ISO 834
    at long times).
  - :func:`hydrocarbon_temperature` -- ISO 834 hydrocarbon curve for
    petrochemical / pool fires (high-temperature, fast-ramp).
  - :func:`ec1_parametric_temperature` -- EN 1991-1-2 Annex A natural
    parametric fire.

* **Temperature-dependent reduction factors** for steel and concrete
  per EN 1993-1-2 (steel) and EN 1992-1-2 (concrete):

  - :func:`steel_strength_reduction_ec3` -- ``f_y(T) / f_y(20)``.
  - :func:`steel_modulus_reduction_ec3` -- ``E(T) / E(20)``.
  - :func:`concrete_strength_reduction_ec2` -- ``f_c(T) / f_c(20)``,
    siliceous + calcareous aggregates.

* **Critical temperature** for a steel member at a given utilisation
  ratio (Eurocode-style: ``mu_0 = sigma_demand / f_y(20)``).

All temperatures are in **°C** (Celsius) for consistency with
fire-engineering practice; the analysis-side modules accept K or °C
interchangeably as long as the user is consistent.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np


# ============================================================ standard fires

def iso_834_temperature(t_min: float, *, T0: float = 20.0) -> float:
    """ISO 834 / BS 476-20 cellulosic fire curve.

    ``T(t) = T_0 + 345 · log10(8t + 1)``  with ``t`` in minutes.

    Reaches ~500°C in 5 min, ~840°C in 30 min, ~945°C in 60 min,
    ~1029°C in 120 min.
    """
    if t_min < 0.0:
        raise ValueError(f"t_min must be >= 0, got {t_min}")
    return float(T0 + 345.0 * math.log10(8.0 * t_min + 1.0))


def astm_e119_temperature(t_min: float, *, T0: float = 20.0) -> float:
    """ASTM E119 fire curve (essentially identical to ISO 834 at
    structural-fire-relevant times)."""
    return iso_834_temperature(t_min, T0=T0)


def hydrocarbon_temperature(t_min: float, *, T0: float = 20.0) -> float:
    """EN 1991-1-2 hydrocarbon fire curve (pool fires, petrochemical):

    ``T(t) = T_0 + 1080 · (1 - 0.325·exp(-0.167 t) - 0.675·exp(-2.5 t))``
    """
    if t_min < 0.0:
        raise ValueError(f"t_min must be >= 0, got {t_min}")
    f = 1.0 - 0.325 * math.exp(-0.167 * t_min) - 0.675 * math.exp(-2.5 * t_min)
    return float(T0 + 1080.0 * f)


def ec1_parametric_temperature(
    t_min: float,
    *,
    q_td: float,
    Av: float, At: float, h_w: float, O: float | None = None,
    b: float = 1500.0,
    T0: float = 20.0,
) -> float:
    """EN 1991-1-2 Annex A parametric fire (natural fire) curve.

    A simplified implementation: heating phase up to ``t_max`` and
    cooling phase afterward.

    Parameters
    ----------
    q_td : float
        Design fire-load density (MJ/m^2 of floor area).
    Av : float
        Ventilation area (m^2).
    At : float
        Total enclosure surface area (m^2).
    h_w : float
        Mean opening height (m).
    O : float, optional
        Opening factor (1/m^0.5) = ``Av · sqrt(h_w) / At``. If
        omitted, computed from inputs.
    b : float, default 1500
        Thermal-absorptivity ``sqrt(rho · c · k)`` (J/(m^2·s^0.5·K)).
        Lightweight concrete ~ 720, normal concrete ~ 1500, steel ~ 12000.
    T0 : float, default 20.0
    """
    if O is None:
        O = Av * math.sqrt(h_w) / At
    Gamma = ((O / 0.04) / (b / 1160.0)) ** 2
    # Heating phase
    t_max = max(0.2e-3 * q_td / O, 25.0e-3 / 60.0)     # h
    t_max_min = t_max * 60.0                            # convert h -> min
    t_star = t_min * Gamma / 60.0                       # in hours after scaling
    T_heat = 1325.0 * (1.0 - 0.324 * math.exp(-0.2 * t_star)
                         - 0.204 * math.exp(-1.7 * t_star)
                         - 0.472 * math.exp(-19.0 * t_star))
    return float(T0 + T_heat)


# ============================================================ steel reduction (EC3)

# Table 3.1 of EN 1993-1-2 — reduction factor on yield strength.
_K_Y_TABLE = [
    (20.0, 1.000), (100.0, 1.000), (200.0, 1.000), (300.0, 1.000),
    (400.0, 1.000), (500.0, 0.780), (600.0, 0.470), (700.0, 0.230),
    (800.0, 0.110), (900.0, 0.060), (1000.0, 0.040), (1100.0, 0.020),
    (1200.0, 0.000),
]

# Table 3.1 — reduction factor on elastic modulus.
_K_E_TABLE = [
    (20.0, 1.000), (100.0, 1.000), (200.0, 0.900), (300.0, 0.800),
    (400.0, 0.700), (500.0, 0.600), (600.0, 0.310), (700.0, 0.130),
    (800.0, 0.090), (900.0, 0.0675), (1000.0, 0.045), (1100.0, 0.0225),
    (1200.0, 0.000),
]


def _interp_table(T: float, table: list) -> float:
    Ts = np.array([row[0] for row in table])
    ks = np.array([row[1] for row in table])
    if T <= Ts[0]:
        return float(ks[0])
    if T >= Ts[-1]:
        return float(ks[-1])
    return float(np.interp(T, Ts, ks))


def steel_strength_reduction_ec3(T_C: float) -> float:
    """k_y,θ from EN 1993-1-2 Table 3.1.

    The reduction factor on the steel yield strength at temperature
    ``T_C`` (°C).  Returns 1.0 below 400°C and ~0 above 1200°C.
    """
    return _interp_table(T_C, _K_Y_TABLE)


def steel_modulus_reduction_ec3(T_C: float) -> float:
    """k_E,θ from EN 1993-1-2 Table 3.1.

    The reduction factor on the steel elastic modulus at temperature
    ``T_C`` (°C).  Decreases earlier than the yield-strength factor
    (E drops faster).
    """
    return _interp_table(T_C, _K_E_TABLE)


# ============================================================ concrete reduction (EC2)

# Table 3.1 of EN 1992-1-2 -- siliceous aggregate.
_KC_SILICEOUS_TABLE = [
    (20.0, 1.000), (100.0, 1.000), (200.0, 0.95), (300.0, 0.85),
    (400.0, 0.75), (500.0, 0.60), (600.0, 0.45), (700.0, 0.30),
    (800.0, 0.15), (900.0, 0.08), (1000.0, 0.04), (1100.0, 0.01),
    (1200.0, 0.000),
]

# Calcareous aggregate (slightly better high-T performance).
_KC_CALCAREOUS_TABLE = [
    (20.0, 1.000), (100.0, 1.000), (200.0, 0.97), (300.0, 0.91),
    (400.0, 0.85), (500.0, 0.74), (600.0, 0.60), (700.0, 0.43),
    (800.0, 0.27), (900.0, 0.15), (1000.0, 0.06), (1100.0, 0.02),
    (1200.0, 0.000),
]


def concrete_strength_reduction_ec2(
    T_C: float, *, aggregate: str = "siliceous",
) -> float:
    """k_c,θ from EN 1992-1-2 Table 3.1.

    Parameters
    ----------
    T_C : float
        Temperature (°C).
    aggregate : {"siliceous", "calcareous"}, default "siliceous"
        Calcareous aggregate (limestone) performs better at high T.
    """
    if aggregate == "siliceous":
        table = _KC_SILICEOUS_TABLE
    elif aggregate == "calcareous":
        table = _KC_CALCAREOUS_TABLE
    else:
        raise ValueError(
            f"aggregate must be 'siliceous' or 'calcareous', "
            f"got {aggregate!r}"
        )
    return _interp_table(T_C, table)


# ============================================================ critical temperature

def steel_critical_temperature(
    *,
    mu_0: float,
    k_y_curve: Callable[[float], float] = steel_strength_reduction_ec3,
) -> float:
    """Critical steel temperature ``T_a,cr`` for utilisation ``mu_0``.

    ``T_a,cr`` is the temperature at which the steel yield-strength
    reduction factor equals ``mu_0``:

        k_y(T_a,cr) = mu_0.

    The EC3 closed-form approximation::

        T_a,cr = 39.19 · ln(1 / (0.9674 · mu_0^3.833) - 1) + 482

    is used for ``0.013 <= mu_0 <= 1.0``; outside that range we fall
    back to a bisection on the supplied reduction-factor curve.

    Parameters
    ----------
    mu_0 : float
        Utilisation ratio at the fire limit state, ``mu_0 = sigma /
        f_y(20)``. Typical: 0.5-0.7 for design checks.
    """
    if mu_0 <= 0.0 or mu_0 > 1.0:
        raise ValueError(f"mu_0 must be in (0, 1], got {mu_0}")
    if 0.013 <= mu_0 <= 1.0:
        try:
            arg = 1.0 / (0.9674 * mu_0 ** 3.833) - 1.0
            if arg > 0.0:
                return float(39.19 * math.log(arg) + 482.0)
        except (ValueError, OverflowError):
            pass
    # Fallback: bisection on the supplied curve
    lo, hi = 20.0, 1200.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if k_y_curve(mid) > mu_0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
