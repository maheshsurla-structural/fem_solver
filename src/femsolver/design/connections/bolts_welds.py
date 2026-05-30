"""Bolt and weld strength calculators (AISC 360 + IS 800).

Implements:

* **Bolt shear** (single or double shear plane).
* **Bolt bearing** on connected plies.
* **Bolt block-shear** rupture.
* **Fillet weld** strength on the throat.

Two code variants are provided side by side: AISC 360-22 (LRFD) and
IS 800:2007 (LSD). The interfaces are unified to make code-switching
easy. All inputs in SI (Pa, m, N).

References
----------
* AISC 360-22, Cl. J3 (bolts) and J2 (welds).
* IS 800:2007, Cl. 10.3 (bolts) and Cl. 10.5 (welds).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ============================================================ bolt shear

@dataclass
class BoltShearResult:
    """Bolt shear strength."""

    V_d_single: float           # design strength per bolt per shear plane (N)
    V_d_total: float            # V_d_single * n_bolts * n_shear_planes
    code: str


def bolt_shear_aisc(
    *,
    n_bolts: int,
    A_b: float,
    F_nv: float,
    n_shear_planes: int = 1,
    phi: float = 0.75,
) -> BoltShearResult:
    """AISC 360 Eq. J3-1: ``Rn = F_nv A_b``, ``phi = 0.75``.

    Parameters
    ----------
    n_bolts : int
    A_b : float
        Nominal bolt area (m^2).
    F_nv : float
        Nominal shear stress = 0.45 F_u (threads excluded) or
        0.563 F_u (threads N) per Table J3.2 (Pa).
    n_shear_planes : int, default 1
    phi : float, default 0.75
    """
    if n_bolts < 1 or n_shear_planes < 1:
        raise ValueError("n_bolts and n_shear_planes must be >= 1")
    if A_b <= 0.0 or F_nv <= 0.0:
        raise ValueError("A_b and F_nv must be > 0")
    Rn = F_nv * A_b
    V_d_single = phi * Rn
    V_d_total = V_d_single * n_bolts * n_shear_planes
    return BoltShearResult(
        V_d_single=float(V_d_single),
        V_d_total=float(V_d_total),
        code="AISC 360-22 J3-1",
    )


def bolt_shear_is800(
    *,
    n_bolts: int,
    f_ub: float, A_nb: float,
    A_sb: float | None = None,
    n_shear_planes_thread: int = 0,
    n_shear_planes_shank: int = 1,
    gamma_mb: float = 1.25,
) -> BoltShearResult:
    """IS 800:2007 Cl. 10.3.3: ``V_dsb = (n_n A_nb + n_s A_sb) f_ub /
    (sqrt(3) gamma_mb)`` per bolt.

    Parameters
    ----------
    n_bolts : int
    f_ub : float
        Bolt ultimate stress (Pa). E.g. 400 MPa (Gr 4.6), 800 (Gr 8.8),
        1000 (Gr 10.9).
    A_nb : float
        Bolt area at threaded part (root area) (m^2).
    A_sb : float, optional
        Bolt area at shank (gross) (m^2). Defaults to A_nb.
    n_shear_planes_thread, n_shear_planes_shank : int
        Number of shear planes passing through threads / shank.
    gamma_mb : float, default 1.25
    """
    if A_sb is None:
        A_sb = A_nb
    if n_bolts < 1:
        raise ValueError("n_bolts must be >= 1")
    if f_ub <= 0.0 or A_nb <= 0.0 or A_sb <= 0.0:
        raise ValueError("f_ub, A_nb, A_sb must be > 0")
    V_dsb = (n_shear_planes_thread * A_nb + n_shear_planes_shank * A_sb) \
            * f_ub / (math.sqrt(3.0) * gamma_mb)
    return BoltShearResult(
        V_d_single=float(V_dsb),
        V_d_total=float(V_dsb * n_bolts),
        code="IS 800:2007 Cl. 10.3.3",
    )


# ============================================================ bolt bearing

@dataclass
class BoltBearingResult:
    V_d_single: float           # design bearing strength per bolt per ply (N)
    V_d_total: float
    code: str
    note: str = ""


def bolt_bearing_aisc(
    *,
    n_bolts: int,
    d_b: float, t: float,
    F_u: float,
    L_c: float,
    deformation_considered: bool = True,
    phi: float = 0.75,
) -> BoltBearingResult:
    """AISC 360 Eq. J3-6a: ``Rn = min(1.5 L_c t F_u, 3.0 d_b t F_u)`` if
    deformation is considered (Eq. J3-6a); ``Rn = min(1.5 L_c t F_u,
    3.6 d_b t F_u)`` if not (Eq. J3-6b).

    Parameters
    ----------
    L_c : float
        Clear distance between holes / edge in the direction of force
        (m).
    """
    if d_b <= 0.0 or t <= 0.0 or F_u <= 0.0 or L_c < 0.0:
        raise ValueError("bearing inputs must be > 0")
    coef = 3.0 if deformation_considered else 3.6
    Rn = min(1.2 * L_c * t * F_u, coef * d_b * t * F_u)
    V_d = phi * Rn
    return BoltBearingResult(
        V_d_single=float(V_d),
        V_d_total=float(V_d * n_bolts),
        code="AISC 360-22 J3-6",
    )


def bolt_bearing_is800(
    *,
    n_bolts: int,
    d_b: float, t: float,
    f_u: float, f_ub: float,
    e: float, p: float | None = None,
    d_0: float | None = None,
    gamma_mb: float = 1.25,
) -> BoltBearingResult:
    """IS 800 Cl. 10.3.4: ``V_dpb = 2.5 k_b d t f_u / gamma_mb`` per bolt.

    ``k_b = min(e/(3 d_0), p/(3 d_0) - 0.25, f_ub/f_u, 1.0)``.

    Parameters
    ----------
    e : float
        Edge distance (m) in the direction of force.
    p : float, optional
        Pitch (centre-to-centre) (m). Required for interior bolts.
    d_0 : float, optional
        Hole diameter (m). Defaults to ``d_b + 1.5 mm`` (standard).
    """
    if d_0 is None:
        d_0 = d_b + 1.5e-3
    if d_b <= 0.0 or t <= 0.0 or f_u <= 0.0 or f_ub <= 0.0:
        raise ValueError("bearing inputs must be > 0")
    if e <= 0.0:
        raise ValueError("e must be > 0")
    terms = [e / (3.0 * d_0)]
    if p is not None:
        terms.append(p / (3.0 * d_0) - 0.25)
    terms.append(f_ub / f_u)
    terms.append(1.0)
    k_b = min(terms)
    V_dpb = 2.5 * k_b * d_b * t * f_u / gamma_mb
    return BoltBearingResult(
        V_d_single=float(V_dpb),
        V_d_total=float(V_dpb * n_bolts),
        code="IS 800:2007 Cl. 10.3.4",
        note=f"k_b = {k_b:.3f}",
    )


# ============================================================ block shear

@dataclass
class BlockShearResult:
    R_n: float                  # nominal block-shear strength (N)
    R_d: float                  # design strength (after phi or 1/gamma)
    code: str


def block_shear_aisc(
    *,
    A_gv: float, A_nv: float,
    A_nt: float,
    F_y: float, F_u: float,
    U_bs: float = 1.0,
    phi: float = 0.75,
) -> BlockShearResult:
    """AISC 360 Eq. J4-5: ``Rn = 0.6 F_u A_nv + U_bs F_u A_nt <=
    0.6 F_y A_gv + U_bs F_u A_nt``.
    """
    R_n_1 = 0.60 * F_u * A_nv + U_bs * F_u * A_nt
    R_n_2 = 0.60 * F_y * A_gv + U_bs * F_u * A_nt
    R_n = min(R_n_1, R_n_2)
    return BlockShearResult(
        R_n=float(R_n), R_d=float(phi * R_n),
        code="AISC 360-22 J4-5",
    )


# ============================================================ fillet weld

@dataclass
class WeldStrengthResult:
    R_n_per_length: float       # weld strength per unit length (N/m)
    R_d_per_length: float
    code: str


def fillet_weld_aisc(
    *,
    leg_size: float, F_EXX: float,
    phi: float = 0.75,
) -> WeldStrengthResult:
    """AISC 360 J2-4: ``Rn = 0.60 F_EXX t_e`` per unit length, with
    throat thickness ``t_e = 0.707 leg_size``.

    Parameters
    ----------
    leg_size : float
        Fillet weld leg (m).
    F_EXX : float
        Filler-metal nominal tensile strength (Pa).
    """
    if leg_size <= 0.0 or F_EXX <= 0.0:
        raise ValueError("inputs must be > 0")
    t_e = 0.707 * leg_size
    Rn = 0.60 * F_EXX * t_e
    return WeldStrengthResult(
        R_n_per_length=float(Rn),
        R_d_per_length=float(phi * Rn),
        code="AISC 360-22 J2-4",
    )


def fillet_weld_is800(
    *,
    leg_size: float, f_u_weld: float,
    gamma_mw: float = 1.25,
) -> WeldStrengthResult:
    """IS 800 Cl. 10.5.7: ``P_dw / l_w = f_u_w t_t / (sqrt(3) gamma_mw)``
    per unit length, ``t_t = 0.7 leg_size``.
    """
    if leg_size <= 0.0 or f_u_weld <= 0.0:
        raise ValueError("inputs must be > 0")
    t_t = 0.7 * leg_size
    Rd_per_length = f_u_weld * t_t / (math.sqrt(3.0) * gamma_mw)
    Rn_per_length = Rd_per_length * gamma_mw      # back-calc nominal
    return WeldStrengthResult(
        R_n_per_length=float(Rn_per_length),
        R_d_per_length=float(Rd_per_length),
        code="IS 800:2007 Cl. 10.5.7",
    )
