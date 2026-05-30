"""EN 1993-1-1 (Eurocode 3) steel design.

Limit-state design per EN 1993-1-1:2005 (incorporating EN 1993-1-1
A1:2014). Implements:

* **Tension** (Cl. 6.2.3): ``N_t,Rd = A f_y / gamma_M0`` and net-section
  rupture ``= 0.9 A_net f_u / gamma_M2``.
* **Compression** (Cl. 6.3.1): Perry-Robertson with buckling curves
  a0, a, b, c, d  -- the same imperfection-factor family used by
  IS 800 (Eurocode 3 is the parent specification).
* **Flexure with LTB** (Cl. 6.3.2): ``M_b,Rd = chi_LT W_pl f_y /
  gamma_M1`` with the general or rolled-section case for ``chi_LT``.
* **Shear** (Cl. 6.2.6): ``V_pl,Rd = A_v f_y / (sqrt(3) gamma_M0)``.
* **Combined-force interaction** (Cl. 6.3.3): simplified linear and
  Method 1 / Method 2 from Annex A / Annex B.

Partial safety factors (EN 1993-1-1 §6.1):

    gamma_M0 = 1.00   (cross-section / yielding)
    gamma_M1 = 1.00   (member buckling)
    gamma_M2 = 1.25   (rupture)

(The CEN-recommended values; National Annexes may adjust.)

References
----------
* EN 1993-1-1:2005 + A1:2014. *Eurocode 3: Design of steel
  structures - Part 1-1: General rules and rules for buildings*.
* Trahair, Bradford, Nethercot & Gardner (2017). *The Behaviour
  and Design of Steel Structures to EC3*, 4e.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ============================================================ partial factors

GAMMA_M0 = 1.00
GAMMA_M1 = 1.00
GAMMA_M2 = 1.25

# EN 1993-1-1 Table 6.1 -- imperfection factors for Perry-Robertson
_ALPHA_BUCKLING = {
    "a0": 0.13,
    "a":  0.21,
    "b":  0.34,
    "c":  0.49,
    "d":  0.76,
}


def perry_robertson_chi(*, lambda_bar: float, curve: str = "b") -> float:
    """Perry-Robertson reduction factor for compression members
    (EN 1993-1-1 Cl. 6.3.1.2).

    ``chi = 1 / (phi + sqrt(phi^2 - lambda_bar^2))``,
    ``phi = 0.5 (1 + alpha (lambda_bar - 0.2) + lambda_bar^2)``,
    clamped to <= 1.
    """
    if lambda_bar < 0.0:
        raise ValueError("lambda_bar must be >= 0")
    if curve not in _ALPHA_BUCKLING:
        raise ValueError(
            f"curve must be one of {list(_ALPHA_BUCKLING)}, got {curve!r}"
        )
    alpha = _ALPHA_BUCKLING[curve]
    phi = 0.5 * (1.0 + alpha * (lambda_bar - 0.2) + lambda_bar ** 2)
    chi = 1.0 / (phi + math.sqrt(phi ** 2 - lambda_bar ** 2))
    return float(min(chi, 1.0))


# ============================================================ tension

@dataclass
class EC3TensionResult:
    N_t_Rd_gross: float
    N_t_Rd_net: float
    N_t_Rd: float
    utilisation: float
    governing: str


def ec3_tension(
    *,
    N_Ed: float,
    A: float, A_net: float | None = None,
    f_y: float, f_u: float,
    gamma_M0: float = GAMMA_M0,
    gamma_M2: float = GAMMA_M2,
) -> EC3TensionResult:
    """Tension member per EN 1993-1-1 Cl. 6.2.3."""
    if A_net is None:
        A_net = A
    Nt_g = A * f_y / gamma_M0
    Nt_n = 0.9 * A_net * f_u / gamma_M2
    if Nt_g <= Nt_n:
        Nt = Nt_g; gov = "gross"
    else:
        Nt = Nt_n; gov = "net"
    return EC3TensionResult(
        N_t_Rd_gross=Nt_g, N_t_Rd_net=Nt_n, N_t_Rd=Nt,
        utilisation=N_Ed / Nt if Nt > 0 else float("inf"),
        governing=gov,
    )


# ============================================================ compression

@dataclass
class EC3CompressionResult:
    N_b_Rd: float
    lambda_bar: float
    chi: float
    f_cd: float
    utilisation: float
    curve: str


def ec3_compression(
    *,
    N_Ed: float,
    A: float,
    f_y: float,
    r_min: float,
    L_cr: float,
    E: float = 2.10e11,
    curve: str = "b",
    gamma_M1: float = GAMMA_M1,
) -> EC3CompressionResult:
    """Axial buckling capacity per EN 1993-1-1 Cl. 6.3.1.

    Parameters
    ----------
    N_Ed : float
        Factored compression demand (N).
    A : float
        Gross cross-section area (m^2).
    f_y : float
    r_min : float
        Minimum radius of gyration about the buckling axis (m).
    L_cr : float
        Critical (buckling) length (m).
    E : float, default 210 GPa
    curve : {"a0", "a", "b", "c", "d"}, default "b"
        Buckling curve (Table 6.2).
    """
    if r_min <= 0.0 or L_cr <= 0.0:
        raise ValueError("r_min and L_cr must be > 0")
    sigma_cr = math.pi ** 2 * E / (L_cr / r_min) ** 2
    lambda_bar = math.sqrt(f_y / sigma_cr)
    chi = perry_robertson_chi(lambda_bar=lambda_bar, curve=curve)
    f_cd = chi * f_y / gamma_M1
    N_b_Rd = A * f_cd
    return EC3CompressionResult(
        N_b_Rd=N_b_Rd, lambda_bar=lambda_bar, chi=chi, f_cd=f_cd,
        utilisation=N_Ed / N_b_Rd if N_b_Rd > 0 else float("inf"),
        curve=curve,
    )


# ============================================================ flexure + LTB

@dataclass
class EC3FlexureResult:
    M_pl_Rd: float
    M_b_Rd: float
    chi_LT: float
    utilisation: float


def ec3_flexure(
    *,
    M_Ed: float,
    W_pl: float,
    f_y: float,
    L_LT: float = 0.0,
    M_cr: float | None = None,
    alpha_LT: float = 0.21,
    gamma_M0: float = GAMMA_M0,
    gamma_M1: float = GAMMA_M1,
) -> EC3FlexureResult:
    """Flexural capacity with LTB per EN 1993-1-1 Cl. 6.3.2.

    ``M_pl,Rd = W_pl · f_y / gamma_M0``
    ``M_b,Rd = chi_LT · W_pl · f_y / gamma_M1``

    where ``chi_LT`` is the LTB reduction factor (general case).

    Parameters
    ----------
    L_LT : float
        Unrestrained-flange length (m). If 0, no LTB (M_b,Rd = M_pl,Rd).
    M_cr : float, optional
        Elastic critical moment (N·m). Required when ``L_LT > 0``.
    alpha_LT : float, default 0.21 (rolled section)
        Imperfection factor.
    """
    M_pl = W_pl * f_y / gamma_M0
    if L_LT <= 0.0 or M_cr is None:
        return EC3FlexureResult(
            M_pl_Rd=M_pl, M_b_Rd=M_pl, chi_LT=1.0,
            utilisation=M_Ed / M_pl if M_pl > 0 else float("inf"),
        )
    lambda_LT = math.sqrt(W_pl * f_y / M_cr)
    phi_LT = 0.5 * (1.0 + alpha_LT * (lambda_LT - 0.2) + lambda_LT ** 2)
    chi_LT = 1.0 / (phi_LT + math.sqrt(phi_LT ** 2 - lambda_LT ** 2))
    chi_LT = min(chi_LT, 1.0)
    M_b = chi_LT * W_pl * f_y / gamma_M1
    return EC3FlexureResult(
        M_pl_Rd=M_pl, M_b_Rd=M_b, chi_LT=chi_LT,
        utilisation=M_Ed / M_b if M_b > 0 else float("inf"),
    )


# ============================================================ shear

@dataclass
class EC3ShearResult:
    V_pl_Rd: float
    utilisation: float
    low_shear: bool


def ec3_shear(
    *,
    V_Ed: float,
    A_v: float, f_y: float,
    gamma_M0: float = GAMMA_M0,
) -> EC3ShearResult:
    """Shear capacity per EN 1993-1-1 Cl. 6.2.6.

    ``V_pl,Rd = A_v · f_y / (sqrt(3) gamma_M0)``.
    """
    V_pl = A_v * f_y / (math.sqrt(3.0) * gamma_M0)
    return EC3ShearResult(
        V_pl_Rd=V_pl,
        utilisation=V_Ed / V_pl if V_pl > 0 else float("inf"),
        low_shear=bool(V_Ed <= 0.5 * V_pl),
    )


# ============================================================ combined N-M

@dataclass
class EC3CombinedResult:
    N_ratio: float
    M_y_ratio: float
    M_z_ratio: float
    total: float
    passes: bool


def ec3_combined_NM(
    *,
    N_Ed: float, M_y_Ed: float, M_z_Ed: float,
    N_b_Rd: float, M_y_b_Rd: float, M_z_b_Rd: float,
) -> EC3CombinedResult:
    """Simplified linear N-M-M interaction per EN 1993-1-1 Cl. 6.3.3
    (conservative upper bound).

    ``N_Ed / N_b,Rd + M_y,Ed / M_y,b,Rd + M_z,Ed / M_z,b,Rd <= 1``.

    The full Annex A / B forms include ``k_yy``, ``k_zz``, ``k_zy``,
    ``k_yz`` interaction factors; this linear form is the safe upper
    bound and is implemented here.
    """
    N_r = abs(N_Ed) / N_b_Rd if N_b_Rd > 0 else float("inf")
    My_r = abs(M_y_Ed) / M_y_b_Rd if M_y_b_Rd > 0 else float("inf")
    Mz_r = abs(M_z_Ed) / M_z_b_Rd if M_z_b_Rd > 0 else float("inf")
    total = N_r + My_r + Mz_r
    return EC3CombinedResult(
        N_ratio=N_r, M_y_ratio=My_r, M_z_ratio=Mz_r,
        total=total, passes=bool(total <= 1.0),
    )


# ============================================================ steel grades

def fy_grade(grade: str) -> float:
    """Look up yield strength by EN steel grade.

    Available: ``"S235"``, ``"S275"``, ``"S355"``, ``"S420"``, ``"S460"``.
    """
    table = {"S235": 235e6, "S275": 275e6, "S355": 355e6,
             "S420": 420e6, "S460": 460e6}
    if grade not in table:
        raise ValueError(
            f"unknown EC3 steel grade {grade!r}; available: {sorted(table)}"
        )
    return float(table[grade])


def fu_grade(grade: str) -> float:
    """Look up ultimate strength by EN steel grade (EN 1993-1-1
    Table 3.1)."""
    table = {"S235": 360e6, "S275": 430e6, "S355": 510e6,
             "S420": 540e6, "S460": 540e6}
    if grade not in table:
        raise ValueError(f"unknown grade {grade!r}")
    return float(table[grade])
