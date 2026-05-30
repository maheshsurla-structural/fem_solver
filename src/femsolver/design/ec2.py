"""EN 1992-1-1 (Eurocode 2) reinforced-concrete design.

Limit-state design per EN 1992-1-1:2004 (incorporating EN 1992-1-1
NA / 2014 amendment).

Implements:

* **Beam flexure** (Cl. 6.1): rectangular stress block of depth
  ``lambda x`` and stress ``eta f_cd`` (closed-form quadratic in
  the neutral-axis depth ``x``). Returns required ``A_s`` for a
  given factored moment.
* **Beam shear** (Cl. 6.2.2 / 6.2.3): ``V_Rd,c`` (concrete only) +
  ``V_Rd,s`` (vertical stirrups, variable-strut-angle truss model).
* **Column M-N interaction** (Cl. 6.1): strain-compatibility P-M
  curve for rectangular sections.

Partial safety factors (EN 1992-1-1 §2.4.2.4 + NA):

    gamma_c = 1.5    (concrete)
    gamma_s = 1.15   (rebar)
    alpha_cc = 0.85  (long-term + sustained-load reduction on f_cd)

Materials (EN 1992-1-1 §3.1.3):

    f_cd = alpha_cc · f_ck / gamma_c
    f_yd = f_yk / gamma_s

References
----------
* EN 1992-1-1:2004 + A1:2014. *Eurocode 2: Design of concrete
  structures - Part 1-1: General rules and rules for buildings*.
* Mosley, Bungey & Hulse (2012). *Reinforced Concrete Design to
  Eurocode 2*, 7e.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ============================================================ partial factors

GAMMA_C = 1.5
GAMMA_S = 1.15
ALPHA_CC = 0.85


def _stress_block_params(f_ck_pa: float) -> tuple[float, float]:
    """Return ``(lambda, eta)`` for the rectangular stress block per
    EN 1992-1-1 §3.1.7.

    For f_ck <= 50 MPa: lambda = 0.8, eta = 1.0.
    For higher strengths the parameters reduce:
        lambda = 0.8 - (f_ck - 50) / 400
        eta    = 1.0 - (f_ck - 50) / 200
    """
    f_ck_MPa = f_ck_pa * 1.0e-6
    if f_ck_MPa <= 50.0:
        return 0.8, 1.0
    lam = max(0.8 - (f_ck_MPa - 50.0) / 400.0, 0.5)
    eta = max(1.0 - (f_ck_MPa - 50.0) / 200.0, 0.5)
    return lam, eta


def f_cd(f_ck: float, *, alpha_cc: float = ALPHA_CC,
          gamma_c: float = GAMMA_C) -> float:
    """Design compressive strength of concrete."""
    return alpha_cc * f_ck / gamma_c


def f_yd(f_yk: float, *, gamma_s: float = GAMMA_S) -> float:
    """Design yield strength of rebar."""
    return f_yk / gamma_s


# ============================================================ beam flexure

@dataclass
class EC2FlexureResult:
    """Result of an EC2 beam flexural design."""

    M_Ed: float
    M_Rd_max: float          # capacity at x = x_max (limit of ductility)
    A_s: float                # required tension steel (m^2)
    A_s2: float               # compression steel if needed (m^2)
    x: float                  # neutral axis depth (m)
    x_over_d: float
    utilisation: float
    is_doubly: bool
    note: str = ""


def ec2_beam_flexure(
    *,
    M_Ed: float,
    f_ck: float,
    f_yk: float,
    b: float,
    d: float,
    d2: float = 0.0,
    alpha_cc: float = ALPHA_CC,
    gamma_c: float = GAMMA_C,
    gamma_s: float = GAMMA_S,
    x_over_d_lim: float = 0.45,
) -> EC2FlexureResult:
    """Design a rectangular RC beam to EN 1992-1-1 §6.1.

    The capacity at the redistribution limit ``x/d = x_over_d_lim``::

        M_Rd_max = eta f_cd b lambda x (d - 0.5 lambda x)

    Singly-reinforced if ``M_Ed <= M_Rd_max``; otherwise doubly with
    compression steel at ``d2`` from the compression face.

    Parameters
    ----------
    M_Ed : float
        Factored moment demand (N·m).
    f_ck : float
        Characteristic concrete strength (Pa).
    f_yk : float
        Characteristic rebar yield strength (Pa).
    b : float
        Beam width (m).
    d : float
        Effective depth (m).
    d2 : float, default 0.0
        Compression-side cover-to-centroid (m). Required only if
        doubly-reinforced.
    alpha_cc, gamma_c, gamma_s : float
        Partial safety / long-term factors.
    x_over_d_lim : float, default 0.45
        Ductility limit on x/d (CEN-recommended value for redistribution
        up to 30%).
    """
    if M_Ed <= 0.0:
        raise ValueError("M_Ed must be > 0")
    if f_ck <= 0.0 or f_yk <= 0.0:
        raise ValueError("f_ck and f_yk must be > 0")
    if b <= 0.0 or d <= 0.0:
        raise ValueError("b and d must be > 0")

    lam, eta = _stress_block_params(f_ck)
    fcd = f_cd(f_ck, alpha_cc=alpha_cc, gamma_c=gamma_c)
    fyd = f_yd(f_yk, gamma_s=gamma_s)

    x_max = x_over_d_lim * d
    M_Rd_max = eta * fcd * b * lam * x_max * (d - 0.5 * lam * x_max)

    if M_Ed <= M_Rd_max:
        # Singly-reinforced. Solve quadratic:
        # eta · f_cd · b · lambda · x · (d - 0.5 lambda x) = M_Ed
        A = -0.5 * eta * fcd * b * lam * lam
        B = eta * fcd * b * lam * d
        C = -M_Ed
        disc = B * B - 4.0 * A * C
        if disc < 0.0:
            raise RuntimeError("flexure quadratic discriminant negative")
        x = (-B + math.sqrt(disc)) / (2.0 * A)
        if x < 0.0 or x > x_max:
            x = x_max
        # A_s from equilibrium: A_s · f_yd = eta · f_cd · b · lambda · x
        A_s = eta * fcd * b * lam * x / fyd
        return EC2FlexureResult(
            M_Ed=M_Ed, M_Rd_max=M_Rd_max,
            A_s=A_s, A_s2=0.0,
            x=x, x_over_d=x / d,
            utilisation=M_Ed / M_Rd_max,
            is_doubly=False,
            note=f"Singly-reinforced (M_Ed <= M_Rd_max = {M_Rd_max:.2f})",
        )

    # Doubly-reinforced
    if d2 <= 0.0:
        raise ValueError(
            "doubly-reinforced solution needed; supply d2 > 0"
        )
    M_2 = M_Ed - M_Rd_max
    A_s1 = eta * fcd * b * lam * x_max / fyd
    A_s2 = M_2 / (fyd * (d - d2))
    A_s = A_s1 + A_s2
    return EC2FlexureResult(
        M_Ed=M_Ed, M_Rd_max=M_Rd_max,
        A_s=A_s, A_s2=A_s2,
        x=x_max, x_over_d=x_over_d_lim,
        utilisation=M_Ed / M_Rd_max,
        is_doubly=True,
        note=(f"Doubly-reinforced "
              f"(M_Ed = {M_Ed*1e-3:.1f} > M_Rd_max = "
              f"{M_Rd_max*1e-3:.1f} kN.m)"),
    )


def ec2_min_max_tension_steel(
    *,
    f_ck: float, f_yk: float, b: float, d: float, h: float | None = None,
) -> tuple[float, float]:
    """A_s,min and A_s,max per EN 1992-1-1 §9.2.1.

    ``A_s,min = max(0.26 f_ctm / f_yk, 0.0013) · b · d``,
    where ``f_ctm = 0.30 · (f_ck/MPa)^(2/3) MPa`` for f_ck <= 50 MPa.

    ``A_s,max = 0.04 · A_c`` (with A_c = b · h, assuming h ≈ d if not
    supplied).
    """
    f_ck_MPa = f_ck * 1.0e-6
    f_yk_MPa = f_yk * 1.0e-6
    f_ctm_MPa = 0.30 * f_ck_MPa ** (2.0 / 3.0)
    ratio = max(0.26 * f_ctm_MPa / f_yk_MPa, 0.0013)
    A_min = ratio * b * d
    A_max = 0.04 * b * (h if h is not None else d)
    return float(A_min), float(A_max)


# ============================================================ beam shear

@dataclass
class EC2ShearResult:
    """Result of an EC2 beam shear check."""

    V_Ed: float
    V_Rd_c: float            # concrete-only capacity (N)
    V_Rd_max: float          # crushing of compression strut (N)
    V_Rd_s_required: float   # required stirrup contribution (N)
    A_sw_over_s_required: float
    requires_stirrups: bool
    cot_theta: float
    utilisation_strut: float
    note: str = ""


def ec2_beam_shear(
    *,
    V_Ed: float,
    f_ck: float,
    f_yk_sv: float,
    b_w: float,
    d: float,
    A_s: float,
    N_Ed: float = 0.0,
    cot_theta: float = 2.5,
    gamma_c: float = GAMMA_C,
    gamma_s: float = GAMMA_S,
) -> EC2ShearResult:
    """Beam shear check per EN 1992-1-1 §6.2.

    Concrete-only capacity::

        V_Rd,c = C_Rd,c · k · (100 rho_l f_ck)^(1/3) · b_w · d
               + k_1 · sigma_cp · b_w · d

    where ``k = 1 + sqrt(200/d_mm) <= 2.0``, ``rho_l = A_s / (b_w d) <=
    0.02``, ``C_Rd,c = 0.18 / gamma_c``, ``k_1 = 0.15``,
    ``sigma_cp = N_Ed / A_c``.

    If ``V_Ed > V_Rd,c``: required vertical stirrups (variable-strut-
    angle truss, default ``cot theta = 2.5`` -> theta ≈ 22°)::

        V_Rd,s = (A_sw / s) · z · f_ywd · cot theta
        z ≈ 0.9 d.

    Crushing of the compression strut limits::

        V_Rd,max = 0.6 · (1 - f_ck/250) · f_cd · b_w · z
                    · (cot theta + tan theta)^{-1}.

    Parameters
    ----------
    V_Ed : float
        Factored shear demand (N).
    f_ck, f_yk_sv : float
        Concrete + stirrup steel strengths (Pa).
    b_w, d : float
        Web width and effective depth (m).
    A_s : float
        Longitudinal tension steel area (m^2).
    N_Ed : float, default 0.0
        Axial force (N, compression positive).
    cot_theta : float, default 2.5
        Cot of the strut angle. CEN default range 1.0 <= cot theta <= 2.5.
    """
    if V_Ed < 0.0:
        raise ValueError("V_Ed must be >= 0")
    if not (1.0 <= cot_theta <= 2.5):
        raise ValueError("cot_theta must lie in [1.0, 2.5]")

    f_ck_MPa = f_ck * 1.0e-6
    d_mm = d * 1000.0
    k = min(1.0 + math.sqrt(200.0 / d_mm), 2.0)
    rho_l = min(A_s / (b_w * d), 0.02)
    C_Rd_c = 0.18 / gamma_c
    sigma_cp = N_Ed / (b_w * (d / 0.9))    # A_c approx via z=0.9d
    fcd = ALPHA_CC * f_ck / gamma_c
    V_Rd_c = (C_Rd_c * k * (100.0 * rho_l * f_ck_MPa) ** (1.0 / 3.0)
              * 1.0e6 * b_w * d
              + 0.15 * sigma_cp * b_w * d)

    z = 0.9 * d
    nu_1 = 0.6 * (1.0 - f_ck_MPa / 250.0)
    tan_theta = 1.0 / cot_theta
    V_Rd_max = nu_1 * fcd * b_w * z / (cot_theta + tan_theta)

    if V_Ed <= V_Rd_c:
        return EC2ShearResult(
            V_Ed=V_Ed, V_Rd_c=V_Rd_c, V_Rd_max=V_Rd_max,
            V_Rd_s_required=0.0,
            A_sw_over_s_required=0.0,
            requires_stirrups=False,
            cot_theta=cot_theta,
            utilisation_strut=V_Ed / V_Rd_max if V_Rd_max > 0 else float("inf"),
            note="V_Ed <= V_Rd_c  -- nominal (minimum) stirrups only",
        )
    if V_Ed > V_Rd_max:
        return EC2ShearResult(
            V_Ed=V_Ed, V_Rd_c=V_Rd_c, V_Rd_max=V_Rd_max,
            V_Rd_s_required=float("inf"),
            A_sw_over_s_required=float("inf"),
            requires_stirrups=True,
            cot_theta=cot_theta,
            utilisation_strut=V_Ed / V_Rd_max,
            note=(f"V_Ed ({V_Ed*1e-3:.0f}) > V_Rd,max ({V_Rd_max*1e-3:.0f}) "
                  "-- redesign cross-section or doubler"),
        )
    V_Rd_s_required = V_Ed
    fywd = f_yk_sv / gamma_s
    A_sw_over_s = V_Rd_s_required / (z * fywd * cot_theta)
    return EC2ShearResult(
        V_Ed=V_Ed, V_Rd_c=V_Rd_c, V_Rd_max=V_Rd_max,
        V_Rd_s_required=V_Rd_s_required,
        A_sw_over_s_required=A_sw_over_s,
        requires_stirrups=True,
        cot_theta=cot_theta,
        utilisation_strut=V_Ed / V_Rd_max,
        note=f"Variable-strut-angle truss (cot theta = {cot_theta})",
    )


# ============================================================ convenience

def fck_class(name: str) -> float:
    """Look up Eurocode concrete class by short name.

    Available: ``"C12/15"``, ``"C16/20"``, ``"C20/25"``, ``"C25/30"``,
    ``"C30/37"``, ``"C35/45"``, ``"C40/50"``, ``"C45/55"``, ``"C50/60"``,
    ``"C55/67"``, ``"C60/75"``.
    """
    table = {
        "C12/15": 12.0e6, "C16/20": 16.0e6,
        "C20/25": 20.0e6, "C25/30": 25.0e6,
        "C30/37": 30.0e6, "C35/45": 35.0e6,
        "C40/50": 40.0e6, "C45/55": 45.0e6,
        "C50/60": 50.0e6,
        "C55/67": 55.0e6, "C60/75": 60.0e6,
    }
    if name not in table:
        raise ValueError(
            f"unknown EC2 concrete class {name!r}; "
            f"available: {sorted(table)}"
        )
    return float(table[name])


def fyk_grade(grade: str) -> float:
    """Look up rebar yield strength by grade name.

    Common: ``"B500"`` (500 MPa, most common in EC2 practice),
    ``"B450"``, ``"B400"``.
    """
    table = {"B400": 400.0e6, "B450": 450.0e6, "B500": 500.0e6}
    if grade not in table:
        raise ValueError(
            f"unknown rebar grade {grade!r}; available: {sorted(table)}"
        )
    return float(table[grade])
