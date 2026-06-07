"""Frequency-dependent (dynamic) Gazetas impedance functions.

The static surface-footing impedance from :mod:`femsolver.analysis.ssi`
is the zero-frequency limit. For a true dynamic SSI analysis, the
foundation stiffness and damping are frequency-dependent:

    K(omega) = K_static · k(a_0),
    C(omega) = (K_static · 2 B / V_s) · c(a_0),

where ``a_0 = omega B / V_s`` is the dimensionless frequency, and
``k(a_0)``, ``c(a_0)`` are Gazetas's empirical curves (1991, Chapter
15 of Foundation Engineering Handbook).

For most modes ``k(a_0)`` is mildly frequency-dependent (varies 0.7-1.0
across the seismic band) and ``c(a_0)`` is the radiation-damping
coefficient that grows from 0 at low frequency.

This module ships polynomial fits to Gazetas's published curves for
the four most common modes of a rigid surface rectangular footing
(half-widths ``B <= L``):

* Vertical (z)
* Horizontal in long direction (y)
* Rocking about long axis x
* Torsion about z

The fits are adequate for ``0 <= a_0 <= 2.0``; beyond that the
literature curves diverge from simple polynomials. Returns the
frequency-dependent multiplier ``k(a_0)`` and the dimensionless
damping ``c(a_0)``, both unitless.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DynamicImpedanceCoefficients:
    """Frequency-dependent multipliers for a rigid surface footing
    at one frequency.

    Attributes
    ----------
    a_0 : float
        Dimensionless frequency ``omega B / V_s``.
    k_z, k_y, k_rx, k_t : float
        Frequency-dependent dynamic-stiffness coefficients (multiplied
        on top of the corresponding static stiffnesses).
    c_z, c_y, c_rx, c_t : float
        Dimensionless damping coefficients.
    """

    a_0: float
    k_z: float
    k_y: float
    k_rx: float
    k_t: float
    c_z: float
    c_y: float
    c_rx: float
    c_t: float


def gazetas_dynamic_coefficients(
    *,
    a_0: float,
    L_over_B: float,
    nu: float = 0.40,
) -> DynamicImpedanceCoefficients:
    """Gazetas (1991) dynamic-stiffness and damping multipliers.

    Polynomial fits valid for ``0 <= a_0 <= 2.0`` and aspect ratio
    ``1 <= L/B <= 4``.

    Parameters
    ----------
    a_0 : float
        Dimensionless frequency.
    L_over_B : float
        Aspect ratio of the footing (L >= B, so L/B >= 1).
    nu : float, default 0.40
        Poisson ratio of the halfspace.
    """
    if a_0 < 0.0:
        raise ValueError("a_0 must be >= 0")
    if L_over_B < 1.0:
        raise ValueError("L_over_B must be >= 1")
    if not (0.0 < nu < 0.5):
        raise ValueError("nu must be in (0, 0.5)")

    # Vertical mode -- Gazetas/Mylonakis fit:
    # k_z(a_0) ~ 1 - 0.10 (a_0)^2  (mild softening with frequency)
    # c_z grows from 0; for square ~ 0.80 a_0; for L/B large up to 0.92 a_0
    k_z = max(1.0 - 0.10 * a_0 * a_0, 0.5)
    c_z = (0.80 + 0.04 * (L_over_B - 1.0)) * a_0

    # Horizontal (long-y direction)
    # k_y(a_0) ~ 1 - 0.05 a_0^2,  c_y ~ 0.65 a_0
    k_y = max(1.0 - 0.05 * a_0 * a_0, 0.5)
    c_y = 0.65 * a_0

    # Rocking about long axis x -- significant frequency dependence
    # k_rx(a_0) ~ 1 - 0.20 a_0   (drops to ~0.6 at a_0=2)
    # c_rx grows slowly; ~ 0.30 a_0
    k_rx = max(1.0 - 0.20 * a_0, 0.3)
    c_rx = 0.30 * a_0

    # Torsion about z
    # k_t(a_0) ~ 1 - 0.14 a_0   c_t ~ 0.55 a_0
    k_t = max(1.0 - 0.14 * a_0, 0.4)
    c_t = 0.55 * a_0

    return DynamicImpedanceCoefficients(
        a_0=float(a_0),
        k_z=float(k_z), k_y=float(k_y), k_rx=float(k_rx), k_t=float(k_t),
        c_z=float(c_z), c_y=float(c_y), c_rx=float(c_rx), c_t=float(c_t),
    )


def dimensionless_frequency(
    *,
    omega: float, B: float, V_s: float,
) -> float:
    """``a_0 = omega B / V_s`` (rad-equivalent dimensionless frequency).

    Parameters
    ----------
    omega : float
        Angular frequency (rad/s).
    B : float
        Footing half-width in the smaller direction (m).
    V_s : float
        Soil shear-wave velocity (m/s).
    """
    if omega < 0.0:
        raise ValueError("omega must be >= 0")
    if B <= 0.0 or V_s <= 0.0:
        raise ValueError("B and V_s must be > 0")
    return float(omega * B / V_s)


@dataclass
class DynamicFootingImpedance:
    """Frequency-dependent footing stiffness AND damping (dimensional).

    Attributes
    ----------
    K_z, K_y, K_rx, K_t : float
        Dynamic stiffnesses (N/m or N.m/rad) at the given frequency.
    C_z, C_y, C_rx, C_t : float
        Dynamic damping coefficients (N.s/m or N.m.s/rad).
    omega : float
    a_0 : float
    """

    K_z: float
    K_y: float
    K_rx: float
    K_t: float
    C_z: float
    C_y: float
    C_rx: float
    C_t: float
    omega: float
    a_0: float


def dynamic_footing_impedance(
    *,
    static_impedance,
    soil,
    omega: float,
) -> DynamicFootingImpedance:
    """Compute dimensional dynamic impedance from a static-impedance
    baseline (via :func:`~femsolver.analysis.ssi.gazetas_surface_footing`)
    multiplied by the Gazetas frequency-dependent coefficients.

    Parameters
    ----------
    static_impedance : FootingImpedance
        From the static Gazetas/Pais-Kausel computation.
    soil : HalfspaceSoil
    omega : float
        Angular frequency (rad/s).
    """
    a_0 = dimensionless_frequency(omega=omega, B=static_impedance.B,
                                     V_s=soil.Vs)
    L_over_B = static_impedance.L / static_impedance.B
    coef = gazetas_dynamic_coefficients(
        a_0=a_0, L_over_B=L_over_B, nu=soil.nu,
    )
    # Damping scaling: c_dim = (K_static · 2 B / V_s) · c_normalized
    scale = 2.0 * static_impedance.B / soil.Vs
    return DynamicFootingImpedance(
        K_z=static_impedance.K_z * coef.k_z,
        K_y=static_impedance.K_y * coef.k_y,
        K_rx=static_impedance.K_rx * coef.k_rx,
        K_t=static_impedance.K_t * coef.k_t,
        C_z=static_impedance.K_z * scale * coef.c_z,
        C_y=static_impedance.K_y * scale * coef.c_y,
        C_rx=static_impedance.K_rx * scale * coef.c_rx,
        C_t=static_impedance.K_t * scale * coef.c_t,
        omega=float(omega), a_0=float(a_0),
    )
