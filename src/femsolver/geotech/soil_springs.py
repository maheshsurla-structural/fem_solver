"""API RP-2A nonlinear soil-spring backbones for pile-soil interaction.

This module provides backbone (monotonic) curves for the three
canonical pile-soil springs in offshore-platform and bridge-pile
practice:

* **p-y** -- lateral soil resistance per unit pile length vs. lateral
  pile displacement at depth ``z``.
* **t-z** -- axial skin friction per unit pile length vs. axial pile
  displacement.
* **q-z** -- tip-bearing force vs. tip displacement at the pile toe.

The functions return :class:`SoilSpringBackbone` objects holding
``(y, p)`` (displacement, force) arrays that can be interpolated, fed
into multi-linear uniaxial materials, or plotted. The forms follow
API RP-2A (sand: Reese et al. 1974, clay: Matlock 1970; t-z and q-z
per API 2014).

References
----------
* API RP-2A WSD, 22nd ed. (2014). "Recommended Practice for Planning,
  Designing and Constructing Fixed Offshore Platforms."
* Reese, L.C., Cox, W.R., and Koop, F.D. (1974). "Analysis of
  laterally loaded piles in sand." *6th OTC*, 2080, 473-483.
* Matlock, H. (1970). "Correlations for design of laterally loaded
  piles in soft clay." *2nd OTC*, 1204, 577-594.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SoilSpringBackbone:
    """Monotonic backbone for a soil spring.

    Attributes
    ----------
    y : np.ndarray
        Displacement axis (m). Sorted ascending, includes 0.
    p : np.ndarray
        Force / line-load axis (N or N/m). Same shape as ``y``.
    description : str
        Short label, e.g., ``"py_sand z=5m phi=35deg"``.
    """

    y: np.ndarray
    p: np.ndarray
    description: str = ""

    def __post_init__(self) -> None:
        self.y = np.asarray(self.y, dtype=float).ravel()
        self.p = np.asarray(self.p, dtype=float).ravel()
        if self.y.shape != self.p.shape:
            raise ValueError("y and p must have the same shape")
        if self.y.size < 2:
            raise ValueError("backbone must have >= 2 points")

    def evaluate(self, y_query: float) -> float:
        """Linear-interpolation lookup ``p(y_query)``.

        For ``|y_query|`` beyond the tabulated range, clamps to the
        endpoint value. Antisymmetric in sign (compression/tension
        treated the same -- the backbone is monotonic in |y|).
        """
        s = 1.0 if y_query >= 0.0 else -1.0
        return float(s * np.interp(abs(y_query), self.y, self.p))

    def initial_stiffness(self) -> float:
        """Initial tangent (chord) stiffness ``dp/dy`` at y -> 0."""
        # Use first interior point as chord estimate
        if self.y[0] == 0.0:
            i = 1
        else:
            i = 0
        return float(self.p[i] / self.y[i])


# ============================================================ p-y curves

# Reese-1974 sand p-y coefficients vs. friction angle (deg)
# Tabulated values from API RP-2A Fig 6.8.6-1 (smooth approximations)

def _reese_C(phi_deg: float) -> tuple[float, float, float]:
    """Reese et al. (1974) shape coefficients ``(C1, C2, C3)`` from
    friction angle ``phi`` (deg).

    Smooth polynomial fits to API Fig 6.8.6-1; valid 20 deg <= phi <= 40 deg.
    """
    phi = math.radians(phi_deg)
    # Wedge geometry: alpha = phi/2, beta = 45 + phi/2
    Ko = 0.4              # at-rest earth pressure coefficient
    Ka = (1.0 - math.sin(phi)) / (1.0 + math.sin(phi))
    alpha = phi / 2.0
    beta = math.pi / 4.0 + phi / 2.0
    # API coefficients (smooth functions of phi)
    C1 = (Ko * math.tan(phi) * math.sin(beta)
          / (math.tan(beta - phi) * math.cos(alpha))
          + math.tan(beta) / math.tan(beta - phi)
            * (math.tan(phi) * math.sin(beta) + math.tan(alpha))
          - Ka)
    C2 = math.tan(beta) / math.tan(beta - phi) - Ka
    # Deep failure factor: 3D wedge
    C3 = Ka * (math.tan(beta) ** 8 - 1.0) + Ko * math.tan(phi) * math.tan(beta) ** 4
    return float(C1), float(C2), float(C3)


def _api_k_py_sand(phi_deg: float) -> float:
    """Modulus of subgrade reaction ``k_py`` for sand (N/m^3).

    API RP-2A Fig 6.8.7-1, linear interpolation between tabulated
    values; clipped to range [20, 40] deg.
    """
    # Values from API tabulation (Pa/m for medium-dense sand)
    phi = max(20.0, min(40.0, phi_deg))
    # API curve (loose to dense), Pa/m × 1e6:
    table_phi = [20.0, 25.0, 30.0, 35.0, 40.0]
    table_k = [5.4e6, 11.0e6, 24.0e6, 45.0e6, 96.0e6]   # Pa/m
    return float(np.interp(phi, table_phi, table_k))


def py_curve_sand(
    *,
    z: float, D: float,
    gamma_eff: float, phi_deg: float,
    A: float = 0.9,
    n_points: int = 60,
    y_max_mult: float = 5.0,
) -> SoilSpringBackbone:
    """API/Reese p-y curve for sand at depth ``z`` below mudline.

    Form (API RP-2A §6.8.6)::

        p(y, z) = A · p_u(z) · tanh( k_py · z / (A · p_u(z)) · y )

    where ``p_u`` is the wedge-failure ultimate soil resistance per
    unit pile length, and ``A`` is the cyclic (0.9) or static
    (max(0.9, 3 - 0.8 z/D)) modifier.

    Parameters
    ----------
    z : float
        Depth below mudline (m), > 0.
    D : float
        Pile outer diameter (m).
    gamma_eff : float
        Effective soil unit weight (N/m^3 = kN/m^3 · 1000).
    phi_deg : float
        Effective friction angle (degrees), 20 to 40.
    A : float, default 0.9
        Cyclic modifier (0.9); for static use ``max(0.9, 3 - 0.8 z/D)``.
    n_points : int, default 60
        Number of backbone points.
    y_max_mult : float, default 5.0
        Maximum displacement = ``y_max_mult · D`` (m).

    Returns
    -------
    SoilSpringBackbone
        ``p`` is in N/m (force per unit pile length).
    """
    if z <= 0.0 or D <= 0.0:
        raise ValueError("z and D must be positive")
    if gamma_eff <= 0.0:
        raise ValueError("gamma_eff must be positive")
    C1, C2, C3 = _reese_C(phi_deg)
    # API: p_u = min(p_us, p_ud)
    # p_us = (C1 z + C2 D) gamma z    (shallow wedge)
    # p_ud = C3 D gamma z              (deep flow-around)
    p_us = (C1 * z + C2 * D) * gamma_eff * z
    p_ud = C3 * D * gamma_eff * z
    p_u = min(p_us, p_ud)
    k = _api_k_py_sand(phi_deg)

    y_max = y_max_mult * D
    y = np.linspace(0.0, y_max, n_points)
    # Avoid 0/0 at y=0
    arg = k * z / max(A * p_u, 1.0e-12) * y
    p = A * p_u * np.tanh(arg)
    desc = (f"py_sand z={z:.2f}m D={D:.3f}m phi={phi_deg:.1f}deg "
            f"A={A:.2f}")
    return SoilSpringBackbone(y=y, p=p, description=desc)


def py_curve_soft_clay(
    *,
    z: float, D: float,
    c_u: float, gamma_eff: float = 0.0,
    eps50: float = 0.02,
    J: float = 0.5,
    n_points: int = 60,
    y_max_mult: float = 16.0,
) -> SoilSpringBackbone:
    """Matlock (1970) static p-y curve for soft clay at depth ``z``.

    Form::

        p_u = min[ (3 + gamma'·z/c_u + J·z/D) c_u D,
                   9 c_u D ]

        y_c = 2.5 · eps50 · D

        Static cubic-root backbone:
            for y <= 8 y_c: p / p_u = 0.5 (y / y_c)^(1/3)
            for y >  8 y_c: p = p_u

    Parameters
    ----------
    z : float
    D : float
    c_u : float
        Undrained shear strength (Pa).
    gamma_eff : float, default 0.0
        Effective unit weight (Pa/m). Set to 0 for short-term/quick.
    eps50 : float, default 0.02
        Strain at 50 percent of undrained strength.
    J : float, default 0.5
        Dimensionless empirical factor (0.5 soft, 0.25 medium).
    n_points, y_max_mult : ints/float

    Returns
    -------
    SoilSpringBackbone
    """
    if z <= 0.0 or D <= 0.0 or c_u <= 0.0:
        raise ValueError("z, D, c_u must be positive")
    if eps50 <= 0.0:
        raise ValueError("eps50 must be positive")

    p_u_shallow = (3.0 + gamma_eff * z / c_u + J * z / D) * c_u * D
    p_u_deep = 9.0 * c_u * D
    p_u = min(p_u_shallow, p_u_deep)
    y_c = 2.5 * eps50 * D

    y_max = y_max_mult * y_c
    y = np.linspace(0.0, y_max, n_points)
    # Avoid 0^(1/3) issue (well-defined but division ok)
    ratio = y / y_c
    p = np.where(
        y <= 8.0 * y_c,
        0.5 * p_u * np.cbrt(ratio),
        p_u,
    )
    desc = (f"py_soft_clay z={z:.2f}m D={D:.3f}m c_u={c_u:.0f}Pa "
            f"eps50={eps50:.3f}")
    return SoilSpringBackbone(y=y, p=p, description=desc)


# ============================================================ t-z curves

def tz_curve_sand(
    *,
    D: float, sigma_v_eff: float,
    delta_deg: float,
    K: float = 0.8,
    z_peak_ratio: float = 0.01,
    n_points: int = 30,
) -> SoilSpringBackbone:
    """API t-z curve for sand (skin friction vs. axial slip).

    Backbone follows the API piecewise-linear shape, normalised by
    the ultimate skin-friction force per unit pile length::

        f_max = K · sigma_v_eff · tan(delta) · pi D

    and the peak displacement at full mobilisation::

        z_peak = z_peak_ratio · D     (default 1% D)

    Linear rise to peak, then constant.

    Parameters
    ----------
    D : float
    sigma_v_eff : float
        Effective overburden stress at the depth of interest (Pa).
    delta_deg : float
        Interface friction angle (deg). Typically 5 deg less than phi.
    K : float, default 0.8
        Lateral earth pressure coefficient.
    z_peak_ratio : float, default 0.01
    """
    if D <= 0.0 or sigma_v_eff < 0.0:
        raise ValueError("D > 0, sigma_v_eff >= 0 required")
    f_max = K * sigma_v_eff * math.tan(math.radians(delta_deg)) * math.pi * D
    z_peak = z_peak_ratio * D
    z_max = 5.0 * z_peak
    y = np.linspace(0.0, z_max, n_points)
    p = np.where(y < z_peak, y / z_peak * f_max, f_max)
    return SoilSpringBackbone(
        y=y, p=p,
        description=f"tz_sand D={D:.3f}m sigma_v={sigma_v_eff:.0f}Pa",
    )


def tz_curve_clay(
    *,
    D: float, c_u: float, alpha: float = 0.8,
    z_peak_ratio: float = 0.005,
    degrade_to_residual: float = 0.9,
    n_points: int = 30,
) -> SoilSpringBackbone:
    """API t-z curve for clay.

    Backbone::

        f_max = alpha · c_u · pi D            (peak)
        f_res = degrade_to_residual · f_max   (post-peak residual)

    Linear rise to ``z_peak = z_peak_ratio · D``, then linear decay to
    ``f_res`` at ``z = 2 z_peak``, then constant residual.
    """
    if D <= 0.0 or c_u <= 0.0:
        raise ValueError("D > 0, c_u > 0 required")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha in [0, 1]")
    if not (0.0 < degrade_to_residual <= 1.0):
        raise ValueError("degrade_to_residual in (0, 1]")
    f_max = alpha * c_u * math.pi * D
    f_res = degrade_to_residual * f_max
    z_peak = z_peak_ratio * D
    z_end = 10.0 * z_peak
    y = np.linspace(0.0, z_end, n_points)
    p = np.where(
        y < z_peak,
        y / z_peak * f_max,
        np.where(
            y < 2.0 * z_peak,
            f_max - (f_max - f_res) * (y - z_peak) / z_peak,
            f_res,
        ),
    )
    return SoilSpringBackbone(
        y=y, p=p,
        description=f"tz_clay D={D:.3f}m c_u={c_u:.0f}Pa alpha={alpha:.2f}",
    )


# ============================================================ q-z curve

def qz_curve(
    *,
    D: float, q_ult: float,
    z_peak_ratio: float = 0.10,
    n_points: int = 30,
) -> SoilSpringBackbone:
    """API tip-bearing q-z curve.

    Cube-root rise to full mobilisation at ``z_peak = 0.10 D``::

        q / q_ult = (z / z_peak)^(1/3)    for z <= z_peak
        q = q_ult                          for z >  z_peak

    Parameters
    ----------
    D : float
    q_ult : float
        Ultimate tip resistance (Pa). For sand: ``N_q · sigma_v_eff``;
        for clay: ``9 c_u``.
    """
    if D <= 0.0 or q_ult <= 0.0:
        raise ValueError("D and q_ult must be positive")
    A_tip = math.pi * D ** 2 / 4.0
    Q_ult = q_ult * A_tip
    z_peak = z_peak_ratio * D
    z_max = 2.0 * z_peak
    y = np.linspace(0.0, z_max, n_points)
    p = np.where(
        y < z_peak,
        Q_ult * np.cbrt(y / z_peak),
        Q_ult,
    )
    return SoilSpringBackbone(
        y=y, p=p,
        description=f"qz D={D:.3f}m q_ult={q_ult:.0f}Pa",
    )
