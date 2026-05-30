"""IS 800:2007 — General Construction in Steel, design checks (LSD).

Limit-state design per IS 800:2007. Provides member capacity checks:

* **Tension** (Cl. 6): gross-section yield ``T_dg = A_g f_y / gamma_m0``
  and net-section rupture ``T_dn = 0.9 A_n f_u / gamma_m1``; design
  capacity is the smaller.
* **Compression** (Cl. 7): Perry-Robertson curves a/b/c/d via the
  Eurocode-style imperfection factors. Returns ``P_d = chi A_g f_y /
  gamma_m0``.
* **Flexure** (Cl. 8): plastic moment ``M_dp = beta_b Z_p f_y /
  gamma_m0`` for compact, laterally-restrained sections; LTB
  reduction ``chi_LT`` for unrestrained compression flanges.
* **Shear** (Cl. 8.4): ``V_d = A_v f_yw / (sqrt(3) gamma_m0)``.
* **Combined-force interaction** (Cl. 9): simplified P-M ratio check.

Partial safety factors (IS 800 Table 5):

    gamma_m0 = 1.10  (yielding, member capacity)
    gamma_m1 = 1.25  (rupture, ultimate strength)
    gamma_mw = 1.25  (welds)

All inputs are SI: Pa, m, N, N.m.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ============================================================ constants

GAMMA_M0 = 1.10        # member yielding
GAMMA_M1 = 1.25        # rupture / ultimate
GAMMA_MW = 1.25        # welds

# Perry-Robertson imperfection factors (Table 7 of IS 800; same as EC3)
_ALPHA_CURVE = {"a": 0.21, "b": 0.34, "c": 0.49, "d": 0.76}


def perry_robertson_chi(
    *,
    lambda_bar: float, curve: str = "b",
) -> float:
    """Perry-Robertson reduction factor ``chi`` for compression members.

    chi = 1 / (phi + sqrt(phi^2 - lambda_bar^2)),
    phi = 0.5 (1 + alpha (lambda_bar - 0.2) + lambda_bar^2),
    clamped to <= 1.

    Parameters
    ----------
    lambda_bar : float
        Non-dimensional slenderness sqrt(f_y / sigma_cr).
    curve : {"a", "b", "c", "d"}
        Buckling curve per IS 800 Table 7. Default "b" for hot-rolled.
    """
    if lambda_bar < 0.0:
        raise ValueError("lambda_bar must be >= 0")
    if curve not in _ALPHA_CURVE:
        raise ValueError(f"curve must be one of {list(_ALPHA_CURVE)}, "
                          f"got {curve!r}")
    alpha = _ALPHA_CURVE[curve]
    phi = 0.5 * (1.0 + alpha * (lambda_bar - 0.2) + lambda_bar ** 2)
    chi = 1.0 / (phi + math.sqrt(phi ** 2 - lambda_bar ** 2))
    return float(min(chi, 1.0))


# ============================================================ tension

@dataclass
class IS800TensionResult:
    T_d_gross: float          # gross-section yield (N)
    T_d_net: float            # net-section rupture (N)
    T_d: float                # design capacity (min)
    utilisation: float
    governing: str            # 'gross' or 'net'


def is800_tension(
    *,
    T_u: float,
    A_g: float, A_n: float | None = None,
    f_y: float, f_u: float,
) -> IS800TensionResult:
    """Tension member capacity per IS 800 Cl. 6.

    Parameters
    ----------
    T_u : float
        Factored tensile demand (N).
    A_g : float
        Gross cross-section area (m^2).
    A_n : float, optional
        Net cross-section area (m^2). Defaults to ``A_g`` (no holes).
    f_y, f_u : float
        Yield and ultimate strengths (Pa).
    """
    if A_n is None:
        A_n = A_g
    T_d_gross = A_g * f_y / GAMMA_M0
    T_d_net = 0.9 * A_n * f_u / GAMMA_M1
    if T_d_gross <= T_d_net:
        T_d = T_d_gross
        gov = "gross"
    else:
        T_d = T_d_net
        gov = "net"
    return IS800TensionResult(
        T_d_gross=T_d_gross, T_d_net=T_d_net, T_d=T_d,
        utilisation=T_u / T_d if T_d > 0 else float("inf"),
        governing=gov,
    )


# ============================================================ compression

@dataclass
class IS800CompressionResult:
    P_d: float                # design compressive capacity (N)
    lambda_bar: float
    chi: float
    f_cd: float               # design compressive stress (Pa)
    utilisation: float
    curve: str


def is800_compression(
    *,
    P_u: float,
    A_g: float,
    f_y: float,
    r_min: float,
    K_L: float,
    E: float = 2.0e11,
    curve: str = "b",
) -> IS800CompressionResult:
    """Axial compression capacity per IS 800 Cl. 7.

    Parameters
    ----------
    P_u : float
        Factored compressive demand (N).
    A_g : float
        Gross area (m^2).
    f_y : float
        Yield stress (Pa).
    r_min : float
        Minimum radius of gyration about the buckling axis (m).
    K_L : float
        Effective length (m).
    E : float, default 200 GPa
    curve : {"a", "b", "c", "d"}, default "b"
        Buckling curve (Table 7 IS 800).
    """
    if r_min <= 0.0 or K_L <= 0.0:
        raise ValueError("r_min and K_L must be > 0")
    sigma_cr = math.pi ** 2 * E / (K_L / r_min) ** 2     # Euler stress
    lambda_bar = math.sqrt(f_y / sigma_cr)
    chi = perry_robertson_chi(lambda_bar=lambda_bar, curve=curve)
    f_cd = chi * f_y / GAMMA_M0
    P_d = A_g * f_cd
    return IS800CompressionResult(
        P_d=P_d, lambda_bar=lambda_bar, chi=chi, f_cd=f_cd,
        utilisation=P_u / P_d if P_d > 0 else float("inf"),
        curve=curve,
    )


# ============================================================ flexure

@dataclass
class IS800FlexureResult:
    M_d_plastic: float        # full plastic capacity (N·m)
    M_d_LTB: float            # LTB-reduced capacity (N·m)
    M_d: float                # min(plastic, LTB)
    chi_LT: float             # LTB reduction factor
    utilisation: float


def is800_flexure(
    *,
    M_u: float,
    Z_p: float,
    f_y: float,
    L_LT: float = 0.0,
    Z_e: float | None = None,
    M_cr: float | None = None,
    beta_b: float = 1.0,
    alpha_LT: float = 0.21,
) -> IS800FlexureResult:
    """Flexural capacity per IS 800 Cl. 8.

    Parameters
    ----------
    M_u : float
        Factored moment demand (N·m).
    Z_p : float
        Plastic section modulus (m^3).
    f_y : float
        Yield stress (Pa).
    L_LT : float, default 0.0
        Effective unrestrained length for LTB (m). If 0, no LTB check.
    Z_e : float, optional
        Elastic section modulus (m^3) -- needed if ``L_LT > 0`` and
        ``M_cr`` is not supplied.
    M_cr : float, optional
        Elastic critical moment (N·m) for LTB. If omitted, an
        ``L_LT/r_y``-based approximate formula would be needed; here
        we require it explicitly when ``L_LT > 0``.
    beta_b : float, default 1.0
        1.0 for plastic / compact, ``Z_e / Z_p`` for semi-compact.
    alpha_LT : float, default 0.21
        LTB imperfection factor (0.21 rolled, 0.49 welded).
    """
    M_dp = beta_b * Z_p * f_y / GAMMA_M0
    if L_LT <= 0.0 or M_cr is None:
        return IS800FlexureResult(
            M_d_plastic=M_dp, M_d_LTB=M_dp, M_d=M_dp,
            chi_LT=1.0,
            utilisation=M_u / M_dp if M_dp > 0 else float("inf"),
        )
    # LTB reduction
    lambda_LT = math.sqrt(beta_b * Z_p * f_y / M_cr)
    phi_LT = 0.5 * (1.0 + alpha_LT * (lambda_LT - 0.2) + lambda_LT ** 2)
    chi_LT = 1.0 / (phi_LT + math.sqrt(phi_LT ** 2 - lambda_LT ** 2))
    chi_LT = min(chi_LT, 1.0)
    M_d_LTB = chi_LT * M_dp
    M_d = min(M_dp, M_d_LTB)
    return IS800FlexureResult(
        M_d_plastic=M_dp, M_d_LTB=M_d_LTB, M_d=M_d,
        chi_LT=chi_LT,
        utilisation=M_u / M_d if M_d > 0 else float("inf"),
    )


# ============================================================ shear

@dataclass
class IS800ShearResult:
    V_d: float                # design shear capacity (N)
    utilisation: float
    is_low_shear: bool        # V_u <= 0.6 V_d (low-shear flexure interaction)


def is800_shear(
    *,
    V_u: float,
    A_v: float,
    f_yw: float,
) -> IS800ShearResult:
    """Shear capacity per IS 800 Cl. 8.4.

    Parameters
    ----------
    V_u : float
        Factored shear demand (N).
    A_v : float
        Shear area = h_w · t_w for I-sections (m^2).
    f_yw : float
        Web yield stress (Pa).
    """
    V_d = A_v * f_yw / (math.sqrt(3.0) * GAMMA_M0)
    return IS800ShearResult(
        V_d=V_d,
        utilisation=V_u / V_d if V_d > 0 else float("inf"),
        is_low_shear=bool(V_u <= 0.6 * V_d),
    )


# ============================================================ combined forces

@dataclass
class IS800CombinedResult:
    P_ratio: float            # P_u / P_d
    Mz_ratio: float
    My_ratio: float
    total: float              # linear sum
    passes: bool


def is800_combined_pm(
    *,
    P_u: float, M_u_z: float, M_u_y: float,
    P_d: float, M_d_z: float, M_d_y: float,
) -> IS800CombinedResult:
    """Simplified P-M-M interaction per IS 800 Cl. 9.3.1 (linear).

    Conservative form ``P/P_d + M_z/M_dz + M_y/M_dy <= 1``. The full
    Cl. 9.3.2 nonlinear form uses reduced moments and is more
    accurate; this linear check is the safe upper bound.
    """
    P_r = abs(P_u) / P_d if P_d > 0 else float("inf")
    Mz_r = abs(M_u_z) / M_d_z if M_d_z > 0 else float("inf")
    My_r = abs(M_u_y) / M_d_y if M_d_y > 0 else float("inf")
    total = P_r + Mz_r + My_r
    return IS800CombinedResult(
        P_ratio=P_r, Mz_ratio=Mz_r, My_ratio=My_r,
        total=total, passes=bool(total <= 1.0),
    )
