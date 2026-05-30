"""IS 456:2000 — Plain and Reinforced Concrete, design checks.

Limit-state-of-collapse design per IS 456:2000 (with the 2007 / 2016
amendments). Implements:

* **Beam flexure** (Cl. 38, Annex G): rectangular concrete stress
  block with peak ``0.36 f_ck`` over depth ``x_u`` and tension steel
  at ``0.87 f_y``. Returns required ``A_st`` for a given factored
  moment ``M_u``, plus a doubly-reinforced solution when
  ``M_u > M_u,lim``.
* **Beam shear** (Cl. 40): design shear stress ``tau_v = V_u / (b d)``,
  permissible concrete shear ``tau_c`` interpolated from
  IS 456 Table 19 with ``p_t`` and ``f_ck``, then required vertical-
  stirrup spacing from ``V_us = 0.87 f_y A_sv d / s_v``.
* **Column P-M interaction** (Cl. 39): a numerical strain-compatibility
  routine that sweeps the neutral-axis depth to trace the P-M curve;
  evaluates a (P, M) demand pair for utilisation ratio.

References
----------
* IS 456:2000 (Reaffirmed 2005). *Plain and Reinforced Concrete --
  Code of Practice*. BIS, New Delhi.
* SP-16:1980 (Reprinted 1999). *Design Aids for Reinforced Concrete
  to IS 456:1978*. BIS.

Units
-----
SI throughout: f_ck, f_y in Pa, lengths in m, forces in N, moments
in N.m. Convenience constructors at the bottom take MPa / mm.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ============================================================ constants

# x_u_max / d limits from IS 456 Cl. 38.1 (depending on steel grade).
_XU_MAX_OVER_D = {
    250.0e6: 0.53,    # Fe 250 (mild steel)
    415.0e6: 0.48,    # Fe 415
    500.0e6: 0.46,    # Fe 500
    550.0e6: 0.44,    # Fe 550 (HYSD)
}


def xu_max_over_d(f_y: float) -> float:
    """``x_u,max / d`` from IS 456 Cl. 38.1.

    Linearly interpolates between the four tabulated grades; clamps
    outside that range.
    """
    grades = sorted(_XU_MAX_OVER_D.keys())
    values = [_XU_MAX_OVER_D[g] for g in grades]
    return float(np.interp(f_y, grades, values))


# ============================================================ beam flexure

@dataclass
class IS456FlexureResult:
    """Result of an IS 456 singly- or doubly-reinforced flexural design.

    Attributes
    ----------
    M_u : float
        Demand factored moment (N·m).
    M_u_lim : float
        Singly-reinforced moment capacity at x_u_max (N·m).
    is_doubly : bool
        True if the beam needs compression reinforcement.
    A_st : float
        Required tension steel area (m^2).
    A_sc : float
        Required compression steel area (m^2). Zero for singly.
    x_u : float
        Computed neutral-axis depth (m).
    x_u_over_d : float
        Ratio (must be <= x_u_max/d for under-reinforced).
    utilisation : float
        ``M_u / phi M_n``-equivalent under IS conventions
        (``M_u`` / ``M_n``, since partial factors are already in
        material strengths). Pass iff <= 1.
    note : str
        Free-text design status.
    """

    M_u: float
    M_u_lim: float
    is_doubly: bool
    A_st: float
    A_sc: float
    x_u: float
    x_u_over_d: float
    utilisation: float
    note: str = ""


def is456_beam_flexure(
    *,
    M_u: float,
    f_ck: float,
    f_y: float,
    b: float,
    d: float,
    d_prime: float = 0.0,
    f_y_sc: float | None = None,
) -> IS456FlexureResult:
    """Design a rectangular RC beam for flexure to IS 456:2000.

    Solves for the tension steel ``A_st`` (and compression steel
    ``A_sc`` if needed) under a positive factored moment ``M_u``.

    Parameters
    ----------
    M_u : float
        Factored moment demand (N·m, positive sagging).
    f_ck : float
        Characteristic concrete strength (Pa).
    f_y : float
        Characteristic yield strength of tension reinforcement (Pa).
    b : float
        Beam width (m).
    d : float
        Effective depth to tension steel centroid (m).
    d_prime : float, default 0.0
        Cover-to-centroid of compression steel (m). Required only when
        a doubly-reinforced solution is needed.
    f_y_sc : float, optional
        Yield strength of compression steel (Pa). Defaults to ``f_y``.

    Returns
    -------
    IS456FlexureResult
    """
    if M_u <= 0.0:
        raise ValueError(f"M_u must be > 0, got {M_u}")
    if f_ck <= 0.0 or f_y <= 0.0:
        raise ValueError("f_ck and f_y must be > 0")
    if b <= 0.0 or d <= 0.0:
        raise ValueError("b and d must be > 0")
    if f_y_sc is None:
        f_y_sc = f_y

    k_lim = xu_max_over_d(f_y)
    x_u_max = k_lim * d
    M_u_lim = 0.36 * f_ck * b * x_u_max * (d - 0.42 * x_u_max)

    if M_u <= M_u_lim:
        # Singly-reinforced -- solve quadratic in x_u
        # 0.36 f_ck b x_u (d - 0.42 x_u) = M_u
        # 0.1512 f_ck b x_u^2 - 0.36 f_ck b d x_u + M_u = 0
        A = 0.1512 * f_ck * b
        B = -0.36 * f_ck * b * d
        C = M_u
        disc = B * B - 4.0 * A * C
        if disc < 0.0:
            raise RuntimeError("flexure quadratic discriminant negative")
        x_u = (-B - math.sqrt(disc)) / (2.0 * A)
        if x_u > x_u_max:    # numerical edge case
            x_u = x_u_max
        A_st = 0.36 * f_ck * b * x_u / (0.87 * f_y)
        result = IS456FlexureResult(
            M_u=M_u, M_u_lim=M_u_lim, is_doubly=False,
            A_st=A_st, A_sc=0.0,
            x_u=x_u, x_u_over_d=x_u / d,
            utilisation=M_u / M_u_lim if M_u_lim > 0 else 0.0,
            note=f"Singly-reinforced (M_u <= M_u,lim={M_u_lim:.2f})",
        )
        return result

    # Doubly-reinforced: M_u = M_u,lim + M_u,2
    if d_prime <= 0.0:
        raise ValueError(
            "doubly-reinforced solution needed (M_u > M_u,lim); "
            "supply d_prime > 0"
        )
    M_u_2 = M_u - M_u_lim
    # Tension steel for the limit moment:
    A_st_1 = 0.36 * f_ck * b * x_u_max / (0.87 * f_y)
    lever_arm_2 = d - d_prime
    # Compression-steel stress (approx 0.87 f_y_sc if d_prime/d small)
    f_sc = 0.87 * f_y_sc
    A_sc = M_u_2 / (f_sc * lever_arm_2)
    # Additional tension steel to balance compression steel:
    A_st_2 = (A_sc * f_sc) / (0.87 * f_y)
    A_st = A_st_1 + A_st_2
    return IS456FlexureResult(
        M_u=M_u, M_u_lim=M_u_lim, is_doubly=True,
        A_st=A_st, A_sc=A_sc,
        x_u=x_u_max, x_u_over_d=k_lim,
        utilisation=M_u / M_u_lim,  # informational
        note=f"Doubly-reinforced (M_u > M_u,lim={M_u_lim:.2f})",
    )


def is456_min_max_tension_steel(
    *,
    f_y: float, b: float, d: float,
) -> tuple[float, float]:
    """Min and max tension reinforcement (m^2) per IS 456 Cl. 26.5.1.1.

    * A_st,min / (b·d) = 0.85 / f_y (f_y in MPa).
    * A_st,max / (b·D) = 0.04 -- using D ~ d as a simplification.
    """
    fy_MPa = f_y * 1.0e-6
    A_min = 0.85 / fy_MPa * b * d
    A_max = 0.04 * b * d
    return float(A_min), float(A_max)


# ============================================================ beam shear

# Table 19 of IS 456 -- design shear strength of concrete tau_c (MPa)
# Indexed by [p_t (%)][grade f_ck (MPa)]. Approximate interpolation.
# Format: rows = p_t in [0.15, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00]
# columns = f_ck in [15, 20, 25, 30, 35, 40]
_TAU_C_TABLE_PT = [0.15, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00]
_TAU_C_TABLE_FCK = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
# MPa values from IS 456 Table 19
_TAU_C_TABLE = np.array([
    [0.28, 0.28, 0.29, 0.29, 0.29, 0.30],   # p_t = 0.15
    [0.35, 0.36, 0.36, 0.37, 0.37, 0.38],   # 0.25
    [0.46, 0.48, 0.49, 0.50, 0.50, 0.51],   # 0.50
    [0.54, 0.56, 0.57, 0.59, 0.59, 0.60],   # 0.75
    [0.60, 0.62, 0.64, 0.66, 0.67, 0.68],   # 1.00
    [0.64, 0.67, 0.70, 0.71, 0.73, 0.74],   # 1.25
    [0.68, 0.72, 0.74, 0.76, 0.78, 0.79],   # 1.50
    [0.71, 0.75, 0.78, 0.80, 0.82, 0.84],   # 1.75
    [0.71, 0.79, 0.82, 0.84, 0.86, 0.88],   # 2.00
    [0.71, 0.81, 0.86, 0.88, 0.90, 0.95],   # 2.50
    [0.71, 0.82, 0.88, 0.92, 0.95, 1.00],   # 3.00
])


def is456_tau_c(*, p_t_pct: float, f_ck: float) -> float:
    """Permissible concrete shear stress tau_c (Pa) from IS 456 Table 19.

    Parameters
    ----------
    p_t_pct : float
        Percentage of tension reinforcement: 100 * A_st / (b·d).
    f_ck : float
        Characteristic concrete strength (Pa).

    Returns
    -------
    tau_c : float (Pa)
    """
    if p_t_pct < 0.0:
        raise ValueError("p_t_pct must be >= 0")
    fck_MPa = max(f_ck * 1.0e-6, _TAU_C_TABLE_FCK[0])
    pt = max(min(p_t_pct, _TAU_C_TABLE_PT[-1]), _TAU_C_TABLE_PT[0])

    # 2D linear interpolation
    j = np.clip(np.searchsorted(_TAU_C_TABLE_FCK, fck_MPa) - 1,
                 0, len(_TAU_C_TABLE_FCK) - 2)
    i = np.clip(np.searchsorted(_TAU_C_TABLE_PT, pt) - 1,
                 0, len(_TAU_C_TABLE_PT) - 2)
    f1, f2 = _TAU_C_TABLE_FCK[j], _TAU_C_TABLE_FCK[j + 1]
    p1, p2 = _TAU_C_TABLE_PT[i], _TAU_C_TABLE_PT[i + 1]
    a = (fck_MPa - f1) / (f2 - f1) if f2 > f1 else 0.0
    b = (pt - p1) / (p2 - p1) if p2 > p1 else 0.0
    v00 = _TAU_C_TABLE[i, j]
    v01 = _TAU_C_TABLE[i, j + 1]
    v10 = _TAU_C_TABLE[i + 1, j]
    v11 = _TAU_C_TABLE[i + 1, j + 1]
    val_MPa = (1 - a) * (1 - b) * v00 + a * (1 - b) * v01 \
              + (1 - a) * b * v10 + a * b * v11
    return float(val_MPa * 1.0e6)


def is456_tau_c_max(*, f_ck: float) -> float:
    """tau_c,max from IS 456 Table 20 (Pa).

    Approximate values: 2.5/2.8/3.1/3.5/3.7/4.0 MPa for M15/20/25/30/35/40+.
    """
    grades = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    vals = [2.5, 2.8, 3.1, 3.5, 3.7, 4.0]
    return float(np.interp(f_ck * 1.0e-6, grades, vals) * 1.0e6)


@dataclass
class IS456ShearResult:
    """Result of an IS 456 beam shear design / check."""

    V_u: float
    tau_v: float
    tau_c: float
    tau_c_max: float
    requires_stirrups: bool
    V_us_required: float          # shear to be carried by stirrups (N)
    A_sv_over_s_required: float   # A_sv / s_v required (m^2 / m)
    utilisation_tau: float        # tau_v / tau_c_max
    note: str = ""


def is456_beam_shear(
    *,
    V_u: float,
    f_ck: float,
    f_y_sv: float,
    b: float,
    d: float,
    A_st: float,
) -> IS456ShearResult:
    """Beam shear design per IS 456 Cl. 40.

    Parameters
    ----------
    V_u : float
        Factored shear demand (N).
    f_ck, f_y_sv : float
        Concrete and stirrup steel strengths (Pa).
    b, d : float
        Beam width and effective depth (m).
    A_st : float
        Tension reinforcement area (m^2) at the section.
    """
    if V_u < 0.0:
        raise ValueError("V_u must be >= 0")
    tau_v = V_u / (b * d)
    p_t_pct = 100.0 * A_st / (b * d)
    tau_c = is456_tau_c(p_t_pct=p_t_pct, f_ck=f_ck)
    tau_c_max = is456_tau_c_max(f_ck=f_ck)
    requires = tau_v > tau_c
    util_tau = tau_v / tau_c_max
    if not requires:
        return IS456ShearResult(
            V_u=V_u, tau_v=tau_v, tau_c=tau_c, tau_c_max=tau_c_max,
            requires_stirrups=False,
            V_us_required=0.0, A_sv_over_s_required=0.0,
            utilisation_tau=util_tau,
            note="tau_v <= tau_c -- nominal (minimum) shear reinforcement OK",
        )
    if tau_v > tau_c_max:
        return IS456ShearResult(
            V_u=V_u, tau_v=tau_v, tau_c=tau_c, tau_c_max=tau_c_max,
            requires_stirrups=True,
            V_us_required=float("inf"),
            A_sv_over_s_required=float("inf"),
            utilisation_tau=util_tau,
            note=(f"tau_v ({tau_v*1e-6:.2f} MPa) > tau_c,max "
                  f"({tau_c_max*1e-6:.2f} MPa) -- redesign section"),
        )
    V_us = V_u - tau_c * b * d
    # Vertical-stirrup formula: V_us = 0.87 f_y A_sv d / s_v
    # A_sv / s_v = V_us / (0.87 f_y d)
    A_sv_over_s = V_us / (0.87 * f_y_sv * d)
    return IS456ShearResult(
        V_u=V_u, tau_v=tau_v, tau_c=tau_c, tau_c_max=tau_c_max,
        requires_stirrups=True,
        V_us_required=V_us,
        A_sv_over_s_required=A_sv_over_s,
        utilisation_tau=util_tau,
        note=(f"tau_v ({tau_v*1e-6:.2f} MPa) > tau_c ({tau_c*1e-6:.2f}); "
              f"design stirrups for V_us = {V_us*1e-3:.2f} kN"),
    )


# ============================================================ column P-M

@dataclass
class IS456ColumnPMPoint:
    """One point on the IS 456 column P-M interaction curve."""

    P: float    # axial load capacity (N, compression positive)
    M: float    # moment capacity at this P (N·m)
    x_u: float  # neutral-axis depth (m)


def _column_pm_at_xu(
    *,
    f_ck: float, f_y: float,
    b: float, D: float,
    A_st_total: float,     # total steel area (m^2)
    n_layers: int,
    d_prime: float,
    x_u: float,
) -> tuple[float, float]:
    """For a given neutral-axis depth x_u, integrate stresses and
    return (P, M) capacities about the centroidal axis.

    Steel is distributed equally in ``n_layers`` parallel-to-bending
    layers from cover ``d_prime`` to ``D - d_prime``.
    """
    eps_cu = 0.0035
    E_s = 2.0e11
    # Concrete contribution: rectangular block 0 to min(0.42·x_u, D)
    # over depth a = 0.84 x_u (the IS 456 simplified block uses a
    # parabolic + rectangular form, but the resultant is at 0.42 x_u
    # with magnitude 0.36 f_ck b x_u for x_u entirely inside the
    # section).
    if x_u > D / 0.42:
        # Pure compression: entire section is in compression
        C_c = 0.446 * f_ck * b * D    # 0.67 f_ck / 1.5
        # All steel at 0.87 f_y compression
        # We bypass strain-comp here for simplicity and treat as pure-axial limit
        T_s_total = -A_st_total * 0.87 * f_y    # all compression -> negative T
        # No moment contribution (symmetric)
        P = C_c - T_s_total
        return float(P), 0.0
    C_c = 0.36 * f_ck * b * x_u
    M_c = C_c * (D / 2.0 - 0.42 * x_u)   # about centroid (compression in +x_u side)

    # Steel layers
    A_per_layer = A_st_total / max(n_layers, 1)
    P_s = 0.0
    M_s = 0.0
    for k in range(n_layers):
        if n_layers == 1:
            y = D / 2.0     # arbitrary, treated as a single layer at extreme
        else:
            y_k = d_prime + k * (D - 2.0 * d_prime) / (n_layers - 1)
            y = y_k          # depth from top fiber
        # Strain in bar at depth y:
        eps_s = eps_cu * (x_u - y) / x_u    # positive means compression
        sigma_s = max(min(eps_s * E_s, 0.87 * f_y), -0.87 * f_y)
        F_s = sigma_s * A_per_layer        # compression positive
        # Moment about centroid: arm = (D/2 - y)
        P_s += F_s
        M_s += F_s * (D / 2.0 - y)
    P = C_c + P_s
    M = M_c + M_s
    return float(P), float(M)


def is456_column_pm_curve(
    *,
    f_ck: float, f_y: float,
    b: float, D: float,
    A_st_total: float,
    n_layers: int = 2,
    d_prime: float | None = None,
    n_points: int = 25,
) -> list[IS456ColumnPMPoint]:
    """Trace the P-M interaction curve for an IS 456 rectangular column.

    Parameters
    ----------
    b, D : float
        Section dimensions perpendicular to and along the bending
        axis respectively.
    A_st_total : float
        Total steel area (m^2).
    n_layers : int, default 2
        Number of equally-spaced parallel bar layers (rows of bars).
    d_prime : float, optional
        Cover-to-centroid of outer bars (m). Defaults to 0.06 D.
    n_points : int, default 25
        Number of points on the curve.
    """
    if d_prime is None:
        d_prime = 0.06 * D
    points: list[IS456ColumnPMPoint] = []

    # Pure compression limit
    P_pure_c = 0.446 * f_ck * b * D + A_st_total * 0.87 * f_y
    points.append(IS456ColumnPMPoint(P=P_pure_c, M=0.0, x_u=float("inf")))

    # Sweep x_u from large (near pure compression) to small (tension yielding)
    x_u_values = np.linspace(D * 1.5, D * 0.1, n_points)
    for x_u in x_u_values:
        P, M = _column_pm_at_xu(
            f_ck=f_ck, f_y=f_y, b=b, D=D,
            A_st_total=A_st_total, n_layers=n_layers,
            d_prime=d_prime, x_u=float(x_u),
        )
        points.append(IS456ColumnPMPoint(P=P, M=abs(M), x_u=float(x_u)))

    # Pure tension limit
    P_pure_t = -A_st_total * 0.87 * f_y
    points.append(IS456ColumnPMPoint(P=P_pure_t, M=0.0, x_u=0.0))

    return points


def is456_column_pm_check(
    *,
    P_u: float, M_u: float,
    points: list[IS456ColumnPMPoint],
) -> tuple[bool, float]:
    """Check (P_u, M_u) against the IS 456 P-M curve.

    Linearly interpolates the curve in P to find the capacity moment
    ``M_n(P_u)``, then reports utilisation ``M_u / M_n``.

    Returns
    -------
    (passes, utilisation) : (bool, float)
    """
    Ps = np.array([p.P for p in points])
    Ms = np.array([p.M for p in points])
    if P_u > Ps.max() or P_u < Ps.min():
        return False, float("inf")
    # Sort by P ascending for interpolation
    order = np.argsort(Ps)
    Ps_s, Ms_s = Ps[order], Ms[order]
    M_n = float(np.interp(P_u, Ps_s, Ms_s))
    if M_n <= 0.0:
        return False, float("inf")
    util = abs(M_u) / M_n
    return bool(util <= 1.0), float(util)


# ============================================================ shorthand constructors

def fck_M(grade: int) -> float:
    """Convenience: M20, M25, M30, ... -> f_ck in Pa."""
    if grade not in (15, 20, 25, 30, 35, 40, 45, 50, 55, 60):
        raise ValueError(f"unknown concrete grade M{grade}")
    return float(grade) * 1.0e6


def fy_Fe(grade: int) -> float:
    """Convenience: Fe 250 / 415 / 500 / 550 -> f_y in Pa."""
    if grade not in (250, 415, 500, 550):
        raise ValueError(f"unknown steel grade Fe{grade}")
    return float(grade) * 1.0e6
