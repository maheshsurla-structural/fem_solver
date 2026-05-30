"""Punching shear at slab-column connections.

Three codes covered:

* **ACI 318-19** (US) -- the three v_c equations of 22.6.5.2 with
  size-effect factor ``lambda_s`` and interior/edge/corner
  ``alpha_s``.
* **EC2 / EN 1992-1-1** (Europe) -- v_Rd,c per 6.4.4.
* **IS 456 (2000)** (India) -- ``tau_c = k_s * 0.25 sqrt(f_ck)``.

All three return a :class:`PunchingResult` with the same set of
fields so that engineering code can switch between codes without
restructuring downstream calculations. Demand checks
(``v_u <= phi * v_c``) are exposed separately.

Sign and unit conventions
-------------------------
* SI units (m, N, Pa) everywhere unless an argument's docstring says
  otherwise.
* Concrete compressive strength ``f_ck`` (EC2 / IS) and ``f_c'``
  (ACI) are positive magnitudes in Pa.
* All linear dimensions in m.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ============================================================ shared

#: Column position relative to the slab.
#:
#: * ``"interior"`` -- column with slab on all four sides.
#: * ``"edge"`` -- column on one slab edge (three sides of critical
#:   perimeter active).
#: * ``"corner"`` -- column on two adjacent slab edges.
_POSITIONS = ("interior", "edge", "corner")


@dataclass
class PunchingResult:
    """Result of a punching-shear capacity calculation (one column,
    one code, no shear reinforcement)."""

    v_c: float            # punching-shear capacity stress (Pa)
    V_c: float            # punching-shear capacity force = v_c * b0 * d (N)
    b_0: float            # critical-section perimeter (m)
    d: float              # effective depth (m)
    position: str
    code: str             # "ACI 318-19", "EC2", "IS 456"
    notes: str = ""       # informational notes (which clause governs)


# ============================================================ ACI 318

def _aci_critical_perimeter(
    c_x: float, c_y: float, d: float, position: str,
) -> float:
    """Critical-section perimeter b_0 at d/2 from the column face per
    ACI 318-19 22.6.4.1. Rectangular column of plan c_x x c_y (m)."""
    side_x = c_x + d           # interior side perimeter
    side_y = c_y + d
    if position == "interior":
        return 2.0 * (side_x + side_y)
    if position == "edge":
        # Three sides active (one side is at the slab edge)
        return 2.0 * side_x + side_y
    # corner: two sides active
    return side_x + side_y


def aci318_punching_capacity(
    *,
    c_x: float,
    c_y: float,
    d: float,
    f_c: float,
    position: str = "interior",
    lambda_factor: float = 1.0,
) -> PunchingResult:
    """ACI 318-19 22.6.5 punching-shear capacity stress.

    Parameters
    ----------
    c_x, c_y : float
        Column plan dimensions (m). Long/short ratio enters one of
        the three v_c equations.
    d : float
        Slab effective depth (m).
    f_c : float
        Specified concrete compressive strength ``f'_c`` (Pa).
    position : {"interior", "edge", "corner"}
    lambda_factor : float, default 1.0
        Lightweight-concrete modification factor.

    Returns
    -------
    :class:`PunchingResult`.
    """
    if c_x <= 0 or c_y <= 0:
        raise ValueError("c_x, c_y must be positive")
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    if f_c <= 0:
        raise ValueError(f"f_c must be positive, got {f_c}")
    if position not in _POSITIONS:
        raise ValueError(
            f"position must be one of {_POSITIONS}, got {position!r}"
        )
    if lambda_factor <= 0:
        raise ValueError(f"lambda_factor must be positive, got {lambda_factor}")

    b_0 = _aci_critical_perimeter(c_x, c_y, d, position)
    # alpha_s per 22.6.5.3
    alpha_s = {"interior": 40.0, "edge": 30.0, "corner": 20.0}[position]
    # Long/short side ratio of the column
    beta = max(c_x, c_y) / max(min(c_x, c_y), 1.0e-12)
    # Size-effect factor lambda_s (ACI 318-19 22.5.5.1.3)
    # d in mm for this formula (ACI 318 uses inches; ACI 318M-19 uses mm
    # with d * 1000 to convert)
    d_mm = d * 1000.0
    lambda_s = math.sqrt(2.0 / (1.0 + 0.004 * d_mm))
    lambda_s = min(lambda_s, 1.0)
    # ACI 318M-19 22.6.5.2 (SI units, output in MPa with f_c in MPa)
    # Convert f_c to MPa for the sqrt term
    f_c_MPa = f_c / 1.0e6
    sf = math.sqrt(f_c_MPa)
    v_c_a = (1.0 / 3.0) * lambda_factor * lambda_s * sf
    v_c_b = (1.0 / 6.0) * (1.0 + 2.0 / beta) * lambda_factor * lambda_s * sf
    v_c_c = (1.0 / 12.0) * (alpha_s * d / b_0 + 2.0) \
            * lambda_factor * lambda_s * sf
    v_c_MPa = min(v_c_a, v_c_b, v_c_c)
    v_c = v_c_MPa * 1.0e6      # back to Pa
    # Which clause governs
    if v_c_MPa == v_c_a:
        note = "v_c governed by 22.6.5.2(a)"
    elif v_c_MPa == v_c_b:
        note = "v_c governed by 22.6.5.2(b) [non-square column]"
    else:
        note = "v_c governed by 22.6.5.2(c) [short b_0/d]"
    return PunchingResult(
        v_c=float(v_c),
        V_c=float(v_c * b_0 * d),
        b_0=float(b_0),
        d=float(d),
        position=position,
        code="ACI 318-19",
        notes=note,
    )


def aci318_punching_demand(
    *,
    V_u: float,
    M_unb: float = 0.0,
    c_x: float,
    c_y: float,
    d: float,
    position: str = "interior",
) -> float:
    """Factored shear stress demand at the critical section per ACI
    318-19 22.6.7 (excluding unbalanced-moment shear-stress
    contribution for ``M_unb = 0``)::

        v_u = V_u / (b_0 * d) + gamma_v * M_unb * c / J_c

    For ``M_unb = 0`` returns the pure-shear-force component. The
    ``gamma_v * M_unb * c / J_c`` term is added for non-zero
    ``M_unb`` using the ACI 318 simplification ``c = (c_x + d)/2``
    and ``J_c ≈ b_0 * d * c^2 / 3`` for a square critical section.
    Returns the demand stress in Pa.
    """
    b_0 = _aci_critical_perimeter(c_x, c_y, d, position)
    v_u = V_u / (b_0 * d)
    if M_unb != 0.0:
        # Fraction transferred by eccentric shear (ACI 318-19 8.4.4.2.2)
        b1 = c_x + d
        b2 = c_y + d
        gamma_v = 1.0 - 1.0 / (1.0 + (2.0 / 3.0) * math.sqrt(b1 / b2))
        c = (c_x + d) / 2.0
        J_c = b_0 * d * c * c / 3.0
        v_u += gamma_v * abs(M_unb) * c / max(J_c, 1e-30)
    return float(v_u)


# ============================================================ EC2

def _eurocode_critical_perimeter(
    c_x: float, c_y: float, d: float, position: str,
) -> float:
    """EC2 6.4.2: control perimeter at distance 2 d from the column
    face. Rectangular column approximation: rounded corners are
    ignored (slight conservative simplification)."""
    # The basic control perimeter for a rectangular column at 2d:
    # u_1 = 2*(c_x + c_y) + 2*pi*(2d)/4 corners per interior column
    # For edge/corner we cut off the portion outside the slab.
    if position == "interior":
        return 2.0 * (c_x + c_y) + 4.0 * math.pi * d
    if position == "edge":
        return 2.0 * c_x + c_y + 2.0 * math.pi * d
    return c_x + c_y + math.pi * d


def eurocode_punching_capacity(
    *,
    c_x: float,
    c_y: float,
    d: float,
    f_ck: float,
    rho_l: float = 0.01,
    position: str = "interior",
    gamma_c: float = 1.5,
    k_1: float = 0.1,
    sigma_cp: float = 0.0,
) -> PunchingResult:
    """EC2 (EN 1992-1-1) 6.4.4 punching-shear resistance::

        v_Rd,c = C_Rd,c * k * (100 rho_l f_ck)^(1/3) + k_1 * sigma_cp

    with ``C_Rd,c = 0.18 / gamma_c`` (≈ 0.12), ``k = 1 + sqrt(200/d)``
    capped at 2.0 (``d`` in mm). Minimum ``v_min = 0.035 k^(3/2)
    sqrt(f_ck)`` is enforced.

    Parameters
    ----------
    rho_l : float
        Mean flexural-rebar ratio in the two orthogonal directions,
        capped at 0.02 internally.
    sigma_cp : float
        Mean concrete normal stress (compression positive) at the
        critical section due to longitudinal compression (typical
        zero for non-prestressed slabs).
    """
    if c_x <= 0 or c_y <= 0 or d <= 0 or f_ck <= 0:
        raise ValueError("c_x, c_y, d, f_ck must all be positive")
    if rho_l < 0:
        raise ValueError(f"rho_l must be non-negative, got {rho_l}")
    if position not in _POSITIONS:
        raise ValueError(
            f"position must be one of {_POSITIONS}, got {position!r}"
        )
    rho_l_capped = min(rho_l, 0.02)
    d_mm = d * 1000.0
    k = min(1.0 + math.sqrt(200.0 / d_mm), 2.0)
    C_Rd_c = 0.18 / gamma_c
    f_ck_MPa = f_ck / 1.0e6
    v_Rd_c_MPa = C_Rd_c * k * (100.0 * rho_l_capped * f_ck_MPa) ** (1.0 / 3.0) \
                 + k_1 * (sigma_cp / 1.0e6)
    v_min_MPa = 0.035 * (k ** 1.5) * math.sqrt(f_ck_MPa)
    v_Rd_c_MPa = max(v_Rd_c_MPa, v_min_MPa)
    v_Rd_c = v_Rd_c_MPa * 1.0e6
    u_1 = _eurocode_critical_perimeter(c_x, c_y, d, position)
    return PunchingResult(
        v_c=float(v_Rd_c),
        V_c=float(v_Rd_c * u_1 * d),
        b_0=float(u_1),
        d=float(d),
        position=position,
        code="EC2",
        notes=f"k = {k:.3f}, rho_l_capped = {rho_l_capped:.4f}",
    )


# ============================================================ IS 456

def _is456_critical_perimeter(
    c_x: float, c_y: float, d: float, position: str,
) -> float:
    """IS 456 31.6.2: critical section at d/2 from the column face.
    Same geometric reasoning as ACI 318."""
    return _aci_critical_perimeter(c_x, c_y, d, position)


def is456_punching_capacity(
    *,
    c_x: float,
    c_y: float,
    d: float,
    f_ck: float,
    position: str = "interior",
) -> PunchingResult:
    """IS 456 (2000) Cl 31.6.3 punching-shear capacity::

        tau_c = k_s * 0.25 * sqrt(f_ck)   [MPa, f_ck in MPa]

    with ``k_s = min(0.5 + beta_c, 1.0)`` and ``beta_c = short /
    long`` side ratio of the column.
    """
    if c_x <= 0 or c_y <= 0 or d <= 0 or f_ck <= 0:
        raise ValueError("c_x, c_y, d, f_ck must all be positive")
    if position not in _POSITIONS:
        raise ValueError(
            f"position must be one of {_POSITIONS}, got {position!r}"
        )
    short = min(c_x, c_y)
    long_ = max(c_x, c_y)
    beta_c = short / long_
    k_s = min(0.5 + beta_c, 1.0)
    f_ck_MPa = f_ck / 1.0e6
    tau_c_MPa = k_s * 0.25 * math.sqrt(f_ck_MPa)
    tau_c = tau_c_MPa * 1.0e6
    b_0 = _is456_critical_perimeter(c_x, c_y, d, position)
    return PunchingResult(
        v_c=float(tau_c),
        V_c=float(tau_c * b_0 * d),
        b_0=float(b_0),
        d=float(d),
        position=position,
        code="IS 456",
        notes=f"k_s = {k_s:.3f}, beta_c = {beta_c:.3f}",
    )
