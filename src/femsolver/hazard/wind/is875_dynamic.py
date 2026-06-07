"""IS 875 Part 3 (2015) dynamic-wind helpers.

The :mod:`is875` module ships the *static* design wind pressure
``p_z = 0.6 V_z^2``. For tall / slender buildings with fundamental
frequency below ~1 Hz the *dynamic response factor* C_dyn must be
applied (IS 875-3 §10 / Annex C). The factor accounts for three
mechanisms:

* **Background turbulence** (B_s) -- random low-frequency gusts that
  the building responds to quasi-statically.
* **Size reduction** (S) -- larger frontal areas average the gusts
  out so less peak demand survives.
* **Gust energy at the building frequency** (E / beta) -- resonance
  between turbulence and the building's first sway mode.

The combined dynamic factor::

    C_dyn = 1 + I_h * sqrt(g_v^2 * B_s + g_R^2 * S * E / (2 beta))

where ``g_v ~ 3.7`` is the peak factor for background and ``g_R``
is the resonant peak factor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ============================================================ turbulence intensity

# IS 875-3 Table 36 (turbulence intensity at z = 10 m by terrain)
_I_10 = {1: 0.155, 2: 0.180, 3: 0.230, 4: 0.270}


def turbulence_intensity(z: float, category: int) -> float:
    """Turbulence intensity ``I_h(z)`` per IS 875-3 §10::

        I_h(z) = I_h(10) * (z/10)^-0.13

    with a floor at z = 10 m.
    """
    if z <= 0:
        raise ValueError(f"z must be positive, got {z}")
    if category not in _I_10:
        raise ValueError(f"category must be 1-4, got {category}")
    z_use = max(z, 10.0)
    I_10 = _I_10[category]
    return float(I_10 * (z_use / 10.0) ** (-0.13))


# ============================================================ length scale

def integral_length_scale(z: float) -> float:
    """Integral length scale ``L_h`` at height z per IS 875-3 §10::

        L_h = 85 * (z/10)^0.25     (m)
    """
    if z <= 0:
        raise ValueError(f"z must be positive, got {z}")
    return float(85.0 * (max(z, 10.0) / 10.0) ** 0.25)


# ============================================================ component terms

def background_factor(*, h: float, b: float, L_h: float) -> float:
    """Background factor ``B_s`` per IS 875-3 §10::

        B_s = 1 / (1 + sqrt((36 h^2 + 64 b^2)) / L_h)
    """
    if h <= 0 or b <= 0 or L_h <= 0:
        raise ValueError("h, b, L_h must all be positive")
    return float(
        1.0 / (1.0 + math.sqrt(36.0 * h * h + 64.0 * b * b) / L_h)
    )


def size_reduction_factor(
    *, f_a: float, h: float, b: float, V_h_bar: float,
) -> float:
    """Size reduction factor ``S`` per IS 875-3 §10::

        S = 1 / [(1 + 3.5 f_a h / V_h_bar)(1 + 4 f_a b / V_h_bar)]
    """
    if f_a <= 0 or h <= 0 or b <= 0 or V_h_bar <= 0:
        raise ValueError("f_a, h, b, V_h_bar must all be positive")
    denom_h = 1.0 + 3.5 * f_a * h / V_h_bar
    denom_b = 1.0 + 4.0 * f_a * b / V_h_bar
    return float(1.0 / (denom_h * denom_b))


def gust_energy_factor(
    *, f_a: float, L_h: float, V_h_bar: float,
) -> float:
    """Gust energy factor ``E`` at the building frequency::

        N = f_a * L_h / V_h_bar
        E = pi * N / (1 + 70.8 N^2)^(5/6)
    """
    if f_a <= 0 or L_h <= 0 or V_h_bar <= 0:
        raise ValueError("f_a, L_h, V_h_bar must all be positive")
    N = f_a * L_h / V_h_bar
    return float(math.pi * N / (1.0 + 70.8 * N * N) ** (5.0 / 6.0))


# ============================================================ dynamic factor

@dataclass
class Is875DynamicFactor:
    """Result of an IS 875-3 dynamic-response calculation."""
    C_dyn: float
    I_h: float
    L_h: float
    B_s: float
    S: float
    E: float
    N: float
    g_R: float
    g_v: float = 3.7


def is875_dynamic_factor(
    *,
    f_a: float,
    h: float,
    b: float,
    V_h_bar: float,
    beta: float,
    category: int = 2,
    g_v: float = 3.7,
) -> Is875DynamicFactor:
    """Compute the IS 875-3 dynamic response factor ``C_dyn``.

    Parameters
    ----------
    f_a : float
        Fundamental natural frequency of the building (Hz).
    h, b : float
        Building height (m) and width perpendicular to wind (m).
    V_h_bar : float
        Hourly mean wind speed at height ``h`` (m/s).
    beta : float
        Damping ratio (fraction of critical, typ. 0.01-0.05).
    category : int, default 2
        IS 875 terrain category (1 = open ... 4 = urban).
    g_v : float, default 3.7
        Peak factor for background response.
    """
    if f_a <= 0:
        raise ValueError(f"f_a must be positive, got {f_a}")
    if not 0 < beta < 0.5:
        raise ValueError(f"beta must be in (0, 0.5), got {beta}")
    I_h = turbulence_intensity(h, category)
    L_h = integral_length_scale(h)
    B_s = background_factor(h=h, b=b, L_h=L_h)
    S = size_reduction_factor(f_a=f_a, h=h, b=b, V_h_bar=V_h_bar)
    E = gust_energy_factor(f_a=f_a, L_h=L_h, V_h_bar=V_h_bar)
    N = f_a * L_h / V_h_bar
    # Peak factor for resonant response (IS 875-3 §10)
    arg = max(3600.0 * f_a, 1.1)
    g_R = math.sqrt(2.0 * math.log(arg)) \
        + 0.577 / math.sqrt(2.0 * math.log(arg))
    # Combined dynamic factor: 1 + I_h * sqrt(B + R)
    C_dyn = 1.0 + I_h * math.sqrt(
        g_v * g_v * B_s + g_R * g_R * S * E / (2.0 * beta)
    )
    return Is875DynamicFactor(
        C_dyn=float(C_dyn), I_h=float(I_h), L_h=float(L_h),
        B_s=float(B_s), S=float(S), E=float(E), N=float(N),
        g_R=float(g_R), g_v=float(g_v),
    )
