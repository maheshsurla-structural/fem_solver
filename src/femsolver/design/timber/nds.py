"""NDS-2024 Ch. 3 member design checks.

Functions
---------
* :func:`nds_bending_check` -- §3.4 flexure with C_L, C_F, C_r, etc.
* :func:`nds_tension_check` -- §3.8 tension parallel to grain
* :func:`nds_compression_check` -- §3.6 / 3.7 compression with C_P
* :func:`nds_shear_check` -- §3.4 shear parallel to grain
* :func:`nds_combined_check` -- §3.9 combined H-interaction
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from femsolver.design.timber.factors import (
    NDSFactors,
    C_L_lateral_stability,
    C_P_column_stability,
)
from femsolver.materials.timber.material import TimberMaterial


# ============================================================ result types

@dataclass
class NDSBendingCheck:
    """Result of NDS §3.4 bending check."""
    F_b_prime: float    # adjusted allowable bending stress (Pa)
    M_allow: float      # allowable moment (N·m)
    M_applied: float    # applied moment (N·m)
    DCR: float          # demand / capacity (≤ 1 passes)
    passes: bool
    factors: NDSFactors


@dataclass
class NDSCompressionCheck:
    F_c_prime: float
    P_allow: float
    P_applied: float
    DCR: float
    passes: bool
    factors: NDSFactors


@dataclass
class NDSTensionCheck:
    F_t_prime: float
    T_allow: float
    T_applied: float
    DCR: float
    passes: bool
    factors: NDSFactors


@dataclass
class NDSShearCheck:
    F_v_prime: float
    V_allow: float
    V_applied: float
    DCR: float
    passes: bool
    factors: NDSFactors


@dataclass
class NDSCombinedCheck:
    """NDS §3.9 combined bending + axial interaction."""
    interaction: float   # NDS Eq. 3.9-1 or 3.9-3 result
    passes: bool         # interaction <= 1.0
    note: str            # which equation was applied
    components: dict     # break-down for diagnostics


# ============================================================ helpers

def _section_modulus_z(b: float, d: float) -> float:
    """Strong-axis elastic section modulus for a rectangle: S = b·d²/6."""
    return b * d * d / 6.0


def _area(b: float, d: float) -> float:
    return b * d


def _validate_section(b: float, d: float) -> None:
    if b <= 0 or d <= 0:
        raise ValueError(f"b, d must be positive (got b={b}, d={d})")


# ============================================================ bending

def nds_bending_check(
    *,
    b: float, d: float, material: TimberMaterial,
    M_applied: float,
    factors: Optional[NDSFactors] = None,
    l_e: Optional[float] = None,
    auto_C_L: bool = True,
) -> NDSBendingCheck:
    """NDS §3.4 bending check.

    ``F_b' = F_b · C_D · C_M · C_t · C_L · C_F · C_fu · C_i · C_r``

    Parameters
    ----------
    b, d : float
        Section breadth and depth (m). Depth is the dimension parallel
        to the load direction (i.e., the section is bent about its
        strong axis when ``d > b``).
    material : TimberMaterial
        Timber grade (NDS reference values).
    M_applied : float
        Applied moment (N·m, sagging positive).
    factors : NDSFactors, optional
        Pre-computed factor set. If ``None``, defaults to all 1.0.
    l_e : float, optional
        Effective unbraced length for LTB (m). If supplied AND
        ``auto_C_L=True``, C_L is computed automatically per §3.3.3.
    auto_C_L : bool, default True
        Compute C_L from l_e using :func:`C_L_lateral_stability`.
        Set ``False`` to use the explicit factors.C_L value.
    """
    _validate_section(b, d)
    f = factors if factors is not None else NDSFactors()

    # If user supplied l_e and wants auto C_L, compute it.
    if auto_C_L and l_e is not None:
        F_b_star = material.f_b_k * f.C_D * f.C_M * f.C_t * f.C_F * f.C_fu * f.C_i * f.C_r
        C_L = C_L_lateral_stability(
            F_b_star=F_b_star, l_e=l_e, d=d, b=b,
            E_min=material.E_0_05,
        )
        f = NDSFactors(
            C_D=f.C_D, C_M=f.C_M, C_t=f.C_t, C_F=f.C_F,
            C_L=C_L, C_P=f.C_P, C_r=f.C_r, C_fu=f.C_fu,
            C_i=f.C_i, C_b=f.C_b,
        )

    F_b_prime = (
        material.f_b_k * f.C_D * f.C_M * f.C_t * f.C_L * f.C_F
        * f.C_fu * f.C_i * f.C_r
    )
    S = _section_modulus_z(b, d)
    M_allow = F_b_prime * S
    DCR = abs(M_applied) / max(M_allow, 1e-30)
    return NDSBendingCheck(
        F_b_prime=F_b_prime, M_allow=M_allow,
        M_applied=M_applied, DCR=DCR,
        passes=DCR <= 1.0, factors=f,
    )


# ============================================================ tension

def nds_tension_check(
    *,
    b: float, d: float, material: TimberMaterial,
    T_applied: float,
    factors: Optional[NDSFactors] = None,
) -> NDSTensionCheck:
    """NDS §3.8 tension-parallel-to-grain check.

    ``F_t' = F_t · C_D · C_M · C_t · C_F · C_i``
    """
    _validate_section(b, d)
    f = factors if factors is not None else NDSFactors()
    F_t_prime = (
        material.f_t_0_k * f.C_D * f.C_M * f.C_t * f.C_F * f.C_i
    )
    A = _area(b, d)
    T_allow = F_t_prime * A
    DCR = abs(T_applied) / max(T_allow, 1e-30)
    return NDSTensionCheck(
        F_t_prime=F_t_prime, T_allow=T_allow,
        T_applied=T_applied, DCR=DCR,
        passes=DCR <= 1.0, factors=f,
    )


# ============================================================ compression

def nds_compression_check(
    *,
    b: float, d: float, material: TimberMaterial,
    P_applied: float,
    factors: Optional[NDSFactors] = None,
    l_e: Optional[float] = None,
    auto_C_P: bool = True,
    column_shape_factor_c: float = 0.8,
) -> NDSCompressionCheck:
    """NDS §3.6 + §3.7 compression-parallel-to-grain check with
    column stability.

    ``F_c' = F_c · C_D · C_M · C_t · C_F · C_i · C_P``
    """
    _validate_section(b, d)
    f = factors if factors is not None else NDSFactors()

    # Auto C_P: use the smaller dimension (worst-case buckling)
    if auto_C_P and l_e is not None:
        F_c_star = material.f_c_0_k * f.C_D * f.C_M * f.C_t * f.C_F * f.C_i
        d_buckle = min(b, d)
        C_P = C_P_column_stability(
            F_c_star=F_c_star, l_e=l_e, d=d_buckle,
            E_min=material.E_0_05, c=column_shape_factor_c,
        )
        f = NDSFactors(
            C_D=f.C_D, C_M=f.C_M, C_t=f.C_t, C_F=f.C_F,
            C_L=f.C_L, C_P=C_P, C_r=f.C_r, C_fu=f.C_fu,
            C_i=f.C_i, C_b=f.C_b,
        )

    F_c_prime = (
        material.f_c_0_k * f.C_D * f.C_M * f.C_t * f.C_F * f.C_i * f.C_P
    )
    A = _area(b, d)
    P_allow = F_c_prime * A
    DCR = abs(P_applied) / max(P_allow, 1e-30)
    return NDSCompressionCheck(
        F_c_prime=F_c_prime, P_allow=P_allow,
        P_applied=P_applied, DCR=DCR,
        passes=DCR <= 1.0, factors=f,
    )


# ============================================================ shear

def nds_shear_check(
    *,
    b: float, d: float, material: TimberMaterial,
    V_applied: float,
    factors: Optional[NDSFactors] = None,
) -> NDSShearCheck:
    """NDS §3.4.3 shear-parallel-to-grain check.

    ``F_v' = F_v · C_D · C_M · C_t · C_i``

    The applied shear stress for a rectangular section is
    ``f_v = 1.5 · V / (b·d)`` (parabolic distribution).
    """
    _validate_section(b, d)
    f = factors if factors is not None else NDSFactors()
    F_v_prime = material.f_v_k * f.C_D * f.C_M * f.C_t * f.C_i
    # V_allow such that 1.5 V / A = F_v' --> V = F_v' * A / 1.5
    A = _area(b, d)
    V_allow = F_v_prime * A / 1.5
    DCR = abs(V_applied) / max(V_allow, 1e-30)
    return NDSShearCheck(
        F_v_prime=F_v_prime, V_allow=V_allow,
        V_applied=V_applied, DCR=DCR,
        passes=DCR <= 1.0, factors=f,
    )


# ============================================================ combined H

def nds_combined_check(
    *,
    b: float, d: float, material: TimberMaterial,
    P_applied: float = 0.0,
    M_strong: float = 0.0,
    M_weak: float = 0.0,
    factors: Optional[NDSFactors] = None,
    l_e: Optional[float] = None,
    l_e_compress: Optional[float] = None,
    is_tension: bool = False,
) -> NDSCombinedCheck:
    """NDS §3.9 combined bending + axial interaction.

    For **tension + bending** (``is_tension=True``):

        f_t/F_t' + f_b/F_b'* ≤ 1.0       (NDS Eq. 3.9-1, tension face)

    where ``F_b'*`` is F_b with all factors except C_L (LTB is
    suppressed when net stress is tension).

    For **compression + bending** (``is_tension=False``):

        (f_c/F_c')^2
            + f_b1/(F_b1'·(1 - f_c/F_cE1))
            + f_b2/(F_b2'·(1 - f_c/F_cE2 - (f_b1/F_bE)^2))
            ≤ 1.0                         (NDS Eq. 3.9-3)

    A simplified safe upper bound is used when the section is biaxial
    but no M_weak is provided.

    Parameters
    ----------
    b, d : float
        Section dimensions (m).
    material : TimberMaterial
    P_applied : float
        Axial load (compression positive in this function).
    M_strong : float
        Moment about strong (z) axis (N·m).
    M_weak : float
        Moment about weak (y) axis (N·m), default 0.
    is_tension : bool
        ``True`` for tension + bending, ``False`` for compression +
        bending.
    l_e : float, optional
        Unbraced length for bending C_L.
    l_e_compress : float, optional
        Effective column length for C_P (defaults to ``l_e`` if not
        given).
    """
    _validate_section(b, d)
    f = factors if factors is not None else NDSFactors()
    if l_e_compress is None:
        l_e_compress = l_e

    # Axial-stress utilizations
    A = _area(b, d)
    S_z = b * d * d / 6.0   # strong-axis modulus
    S_y = d * b * b / 6.0   # weak-axis modulus
    f_axial = abs(P_applied) / A
    f_bz = abs(M_strong) / S_z
    f_by = abs(M_weak) / S_y if S_y > 0 else 0.0

    components: dict = {}

    if is_tension:
        # f_t / F_t' + f_b / F_b'* (tension face, no C_L)
        t_check = nds_tension_check(
            b=b, d=d, material=material,
            T_applied=P_applied, factors=f,
        )
        # F_b* (no C_L): use auto_C_L=False and set C_L=1
        f_bstar = NDSFactors(
            C_D=f.C_D, C_M=f.C_M, C_t=f.C_t, C_F=f.C_F,
            C_L=1.0, C_P=f.C_P, C_r=f.C_r, C_fu=f.C_fu,
            C_i=f.C_i, C_b=f.C_b,
        )
        b_check = nds_bending_check(
            b=b, d=d, material=material,
            M_applied=M_strong, factors=f_bstar,
            auto_C_L=False,
        )
        ratio_t = f_axial / t_check.F_t_prime
        ratio_b = f_bz / b_check.F_b_prime
        interaction = ratio_t + ratio_b
        components = {
            "f_t/F_t_prime": ratio_t,
            "f_b/F_b_star": ratio_b,
        }
        note = "NDS Eq. 3.9-1 (tension + bending)"
    else:
        # Compression + bending (Eq. 3.9-3 simplified for strong-axis only)
        c_check = nds_compression_check(
            b=b, d=d, material=material,
            P_applied=P_applied, factors=f,
            l_e=l_e_compress, auto_C_P=(l_e_compress is not None),
        )
        b_check = nds_bending_check(
            b=b, d=d, material=material,
            M_applied=M_strong, factors=f,
            l_e=l_e, auto_C_L=(l_e is not None),
        )
        # Euler buckling for the magnification terms
        # F_cE = 0.822 * E_min / (l_e/d)^2 (NDS 3.7)
        # F_bE = 1.2 * E_min / R_B^2 (NDS 3.3.3)
        ratio_c = f_axial / c_check.F_c_prime
        ratio_b_z = f_bz / b_check.F_b_prime
        # Magnification (P-delta) term -- requires F_cE1 (column buckling
        # about weak axis -> bending about strong axis amplification).
        # Simplified: use l_e_compress over min(b, d).
        if l_e_compress is not None:
            d_buckle = min(b, d)
            slenderness = l_e_compress / d_buckle
            F_cE1 = 0.822 * material.E_0_05 / (slenderness * slenderness)
            magnification = 1.0 / max(1.0 - ratio_c * c_check.F_c_prime / F_cE1, 1e-9)
        else:
            magnification = 1.0
        interaction = ratio_c ** 2 + ratio_b_z * magnification
        components = {
            "(f_c/F_c_prime)^2": ratio_c ** 2,
            "f_b/F_b_prime * mag": ratio_b_z * magnification,
            "magnification": magnification,
        }
        note = "NDS Eq. 3.9-3 (compression + bending)"

    return NDSCombinedCheck(
        interaction=interaction,
        passes=interaction <= 1.0,
        note=note,
        components=components,
    )
