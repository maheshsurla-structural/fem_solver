"""EC5 (EN 1995-1-1) timber design per §6 (Phase D.1.4).

EC5 uses **limit-state design (LSD)** instead of NDS's allowable-
stress-design (ASD). Each design strength is computed as:

    f_d = k_mod · f_k / gamma_M

where:
* ``f_k`` is the characteristic (5th-percentile) strength
* ``k_mod`` accounts for service class + load-duration class
  (EC5 Table 3.1)
* ``gamma_M`` is the partial safety factor for the material
  (EC5 Table 2.3): 1.3 for solid timber, 1.25 for glulam, etc.

Design checks then compare applied stress to ``f_d`` directly,
optionally modified by size factor ``k_h``, lateral-stability
factor ``k_crit`` (analog of NDS C_L), and column-stability factor
``k_c`` (analog of NDS C_P, but with different formulation).

Functions
---------
* :func:`k_mod_factor` -- §3.1.3 modification factor
* :func:`gamma_M_partial_factor` -- §2.3 partial safety factor
* :func:`k_h_solid` / :func:`k_h_glulam` -- size factors
* :func:`k_crit_lateral_stability` -- §6.3.3
* :func:`k_c_column_stability` -- §6.3.2
* :func:`ec5_bending_check` -- §6.1.6 + §6.3.3
* :func:`ec5_tension_check` -- §6.1.2
* :func:`ec5_compression_check` -- §6.1.4 + §6.3.2
* :func:`ec5_shear_check` -- §6.1.7
* :func:`ec5_combined_check` -- §6.2.3 / 6.2.4 / 6.3.2 / 6.3.3
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from femsolver.materials.timber.material import TimberMaterial


# ============================================================ factors

@dataclass
class EC5Factors:
    """EC5 modification + partial-safety factors for a design check.

    Defaults assume **solid timber** in **service class 1**
    (indoor, RH < 65%) under **medium-term** loading (occupancy).
    """
    k_mod: float = 0.8           # §3.1.3
    gamma_M: float = 1.3         # §2.3 Table 2.3 (solid timber)
    k_h: float = 1.0             # §3.2 / §3.3 size factor
    k_crit: float = 1.0          # §6.3.3 lateral stability
    k_c: float = 1.0             # §6.3.2 column stability
    k_cr: float = 0.67           # §6.1.7 crack factor (solid timber)
    k_m: float = 0.7             # §6.2.3 biaxial bending reduction (rect solid)

    def __post_init__(self) -> None:
        for name in ("k_mod", "gamma_M", "k_h", "k_crit", "k_c",
                      "k_cr", "k_m"):
            v = getattr(self, name)
            if v <= 0:
                raise ValueError(f"{name} must be positive, got {v}")


# ============================================================ k_mod

# EC5 Table 3.1 -- k_mod for solid timber, glulam, LVL.
# Keyed by (material_type, service_class, load_duration).
_K_MOD_TABLE = {
    # (material_type, service_class, load_duration) -> k_mod
    ("solid", 1, "permanent"):    0.60,
    ("solid", 1, "long_term"):    0.70,
    ("solid", 1, "medium_term"):  0.80,
    ("solid", 1, "short_term"):   0.90,
    ("solid", 1, "instantaneous"): 1.10,
    ("solid", 2, "permanent"):    0.60,
    ("solid", 2, "long_term"):    0.70,
    ("solid", 2, "medium_term"):  0.80,
    ("solid", 2, "short_term"):   0.90,
    ("solid", 2, "instantaneous"): 1.10,
    ("solid", 3, "permanent"):    0.50,
    ("solid", 3, "long_term"):    0.55,
    ("solid", 3, "medium_term"):  0.65,
    ("solid", 3, "short_term"):   0.70,
    ("solid", 3, "instantaneous"): 0.90,
    # Glulam (and LVL): same as solid for classes 1 and 2
    ("glulam", 1, "permanent"):   0.60,
    ("glulam", 1, "long_term"):   0.70,
    ("glulam", 1, "medium_term"): 0.80,
    ("glulam", 1, "short_term"):  0.90,
    ("glulam", 1, "instantaneous"): 1.10,
    ("glulam", 2, "permanent"):   0.60,
    ("glulam", 2, "long_term"):   0.70,
    ("glulam", 2, "medium_term"): 0.80,
    ("glulam", 2, "short_term"):  0.90,
    ("glulam", 2, "instantaneous"): 1.10,
    ("glulam", 3, "permanent"):   0.50,
    ("glulam", 3, "long_term"):   0.55,
    ("glulam", 3, "medium_term"): 0.65,
    ("glulam", 3, "short_term"):  0.70,
    ("glulam", 3, "instantaneous"): 0.90,
}


def k_mod_factor(
    material_type: str = "solid",
    service_class: int = 1,
    load_duration: str = "medium_term",
) -> float:
    """EC5 Table 3.1 modification factor k_mod.

    Parameters
    ----------
    material_type : {"solid", "glulam"}
        Material family. For LVL and other engineered wood, use
        ``"glulam"`` as a first approximation (values largely match).
    service_class : {1, 2, 3}
        EC5 §2.3.1.3 service class. Class 1 = heated indoor
        (MC < 12%). Class 2 = covered outdoor (MC < 20%). Class 3 =
        exposed outdoor.
    load_duration : str
        Per §2.3.1.2: one of ``"permanent"``, ``"long_term"``,
        ``"medium_term"``, ``"short_term"``, ``"instantaneous"``.
    """
    key = (material_type, service_class, load_duration)
    if key not in _K_MOD_TABLE:
        raise ValueError(
            f"unknown k_mod combination: material={material_type}, "
            f"service_class={service_class}, duration={load_duration}"
        )
    return _K_MOD_TABLE[key]


# ============================================================ gamma_M

_GAMMA_M_TABLE = {
    "solid": 1.30,
    "glulam": 1.25,
    "LVL": 1.20,
    "plywood": 1.20,
    "OSB": 1.20,
}


def gamma_M_partial_factor(material_type: str = "solid") -> float:
    """EC5 Table 2.3 material partial safety factor."""
    if material_type not in _GAMMA_M_TABLE:
        raise ValueError(
            f"unknown material_type {material_type!r}; "
            f"valid: {sorted(_GAMMA_M_TABLE)}"
        )
    return _GAMMA_M_TABLE[material_type]


# ============================================================ k_h

def k_h_solid(h_m: float) -> float:
    """Solid-timber depth factor for bending or tension per §3.2.

    For h < 150 mm: k_h = min((150/h)^0.2, 1.3).
    """
    if h_m <= 0:
        raise ValueError(f"h must be positive, got {h_m}")
    h_mm = h_m * 1000.0
    if h_mm >= 150:
        return 1.0
    return min((150.0 / h_mm) ** 0.2, 1.3)


def k_h_glulam(h_m: float) -> float:
    """Glulam depth factor per §3.3.

    For h < 600 mm: k_h = min((600/h)^0.1, 1.1).
    """
    if h_m <= 0:
        raise ValueError(f"h must be positive, got {h_m}")
    h_mm = h_m * 1000.0
    if h_mm >= 600:
        return 1.0
    return min((600.0 / h_mm) ** 0.1, 1.1)


# ============================================================ k_crit

def k_crit_lateral_stability(
    f_m_k: float,
    sigma_m_crit: float,
) -> float:
    """EC5 §6.3.3 lateral-stability factor for bending.

    Parameters
    ----------
    f_m_k : float
        Characteristic bending strength (Pa).
    sigma_m_crit : float
        Critical bending stress for LTB (Pa). For rectangular beams,
        the simplified formula §6.31:
            sigma_m_crit = 0.78 · b^2 · E_0,05 / (h · l_ef)
        Caller should compute and pass this.

    Returns
    -------
    k_crit : float
        0 to 1, with 1 meaning no LTB reduction.
    """
    if f_m_k <= 0 or sigma_m_crit <= 0:
        raise ValueError("f_m_k and sigma_m_crit must be positive")
    lambda_rel_m = math.sqrt(f_m_k / sigma_m_crit)
    if lambda_rel_m <= 0.75:
        return 1.0
    if lambda_rel_m <= 1.4:
        return 1.56 - 0.75 * lambda_rel_m
    return 1.0 / (lambda_rel_m * lambda_rel_m)


def sigma_m_crit_rectangular(
    b: float, h: float, l_ef: float, E_0_05: float,
) -> float:
    """Critical bending stress for a rectangular cross-section per
    EC5 §6.3.3 Eq. 6.32 (simplified)::

        σ_m,crit = 0.78 · b^2 · E_0,05 / (h · l_ef)
    """
    if b <= 0 or h <= 0 or l_ef <= 0:
        raise ValueError("b, h, l_ef must be positive")
    return 0.78 * b * b * E_0_05 / (h * l_ef)


# ============================================================ k_c

def k_c_column_stability(
    f_c_0_k: float,
    *,
    slenderness: float,
    E_0_05: float,
    beta_c: float = 0.2,
) -> float:
    """EC5 §6.3.2 column stability factor.

    Parameters
    ----------
    f_c_0_k : float
        Characteristic compression strength parallel (Pa).
    slenderness : float
        Geometric slenderness lambda = l_ef / i, where ``i`` is the
        radius of gyration.
    E_0_05 : float
        5th-percentile modulus (Pa).
    beta_c : float, default 0.2
        Member-imperfection factor. 0.2 for solid timber, 0.1 for
        glulam (§6.3.2).
    """
    if min(f_c_0_k, slenderness, E_0_05, beta_c) <= 0:
        raise ValueError("all inputs must be positive")
    lambda_rel = (slenderness / math.pi) * math.sqrt(f_c_0_k / E_0_05)
    if lambda_rel <= 0.3:
        return 1.0
    k = 0.5 * (1.0 + beta_c * (lambda_rel - 0.3) + lambda_rel ** 2)
    denom = k + math.sqrt(max(k * k - lambda_rel ** 2, 0.0))
    return min(1.0, 1.0 / denom)


# ============================================================ result types

@dataclass
class EC5BendingCheck:
    f_m_d: float        # design bending strength (Pa)
    M_Rd: float         # design moment resistance (N·m)
    M_Ed: float         # applied moment (N·m)
    DCR: float
    passes: bool
    factors: EC5Factors


@dataclass
class EC5TensionCheck:
    f_t_0_d: float
    T_Rd: float
    T_Ed: float
    DCR: float
    passes: bool
    factors: EC5Factors


@dataclass
class EC5CompressionCheck:
    f_c_0_d: float
    P_Rd: float
    P_Ed: float
    DCR: float
    passes: bool
    factors: EC5Factors


@dataclass
class EC5ShearCheck:
    f_v_d: float
    V_Rd: float
    V_Ed: float
    DCR: float
    passes: bool
    factors: EC5Factors


@dataclass
class EC5CombinedCheck:
    interaction: float
    interaction_alt: float    # the dual equation per §6.2.3/4
    passes: bool
    note: str
    components: dict


# ============================================================ design helpers

def _A(b: float, h: float) -> float:
    return b * h


def _S_strong(b: float, h: float) -> float:
    return b * h * h / 6.0


def _S_weak(b: float, h: float) -> float:
    return h * b * b / 6.0


def _validate(b, h):
    if b <= 0 or h <= 0:
        raise ValueError(f"b, h must be positive (got b={b}, h={h})")


# ============================================================ bending

def ec5_bending_check(
    *,
    b: float, h: float, material: TimberMaterial,
    M_Ed: float,
    factors: Optional[EC5Factors] = None,
    l_ef: Optional[float] = None,
    auto_k_crit: bool = True,
    auto_k_h: bool = True,
    material_type: str = "solid",
) -> EC5BendingCheck:
    """EC5 §6.1.6 + §6.3.3 bending check.

    ``f_m,d = k_mod · f_m,k / gamma_M``
    Adjusted for size (k_h) and lateral stability (k_crit).
    """
    _validate(b, h)
    f = factors if factors is not None else EC5Factors()

    # Auto k_h
    if auto_k_h and (material_type == "solid"):
        f = EC5Factors(
            k_mod=f.k_mod, gamma_M=f.gamma_M,
            k_h=k_h_solid(h),
            k_crit=f.k_crit, k_c=f.k_c,
            k_cr=f.k_cr, k_m=f.k_m,
        )
    elif auto_k_h and material_type == "glulam":
        f = EC5Factors(
            k_mod=f.k_mod, gamma_M=f.gamma_M,
            k_h=k_h_glulam(h),
            k_crit=f.k_crit, k_c=f.k_c,
            k_cr=f.k_cr, k_m=f.k_m,
        )

    # Auto k_crit
    if auto_k_crit and l_ef is not None:
        sigma_crit = sigma_m_crit_rectangular(
            b=b, h=h, l_ef=l_ef, E_0_05=material.E_0_05,
        )
        kc = k_crit_lateral_stability(
            f_m_k=material.f_b_k, sigma_m_crit=sigma_crit,
        )
        f = EC5Factors(
            k_mod=f.k_mod, gamma_M=f.gamma_M,
            k_h=f.k_h, k_crit=kc, k_c=f.k_c,
            k_cr=f.k_cr, k_m=f.k_m,
        )

    f_m_d = (
        f.k_mod * material.f_b_k / f.gamma_M * f.k_h * f.k_crit
    )
    S = _S_strong(b, h)
    M_Rd = f_m_d * S
    DCR = abs(M_Ed) / max(M_Rd, 1e-30)
    return EC5BendingCheck(
        f_m_d=f_m_d, M_Rd=M_Rd, M_Ed=M_Ed,
        DCR=DCR, passes=DCR <= 1.0, factors=f,
    )


# ============================================================ tension

def ec5_tension_check(
    *,
    b: float, h: float, material: TimberMaterial,
    T_Ed: float,
    factors: Optional[EC5Factors] = None,
    auto_k_h: bool = True,
    material_type: str = "solid",
) -> EC5TensionCheck:
    """EC5 §6.1.2 tension parallel to grain.

    ``f_t,0,d = k_mod · f_t,0,k / gamma_M · k_h``
    """
    _validate(b, h)
    f = factors if factors is not None else EC5Factors()
    if auto_k_h and material_type == "solid":
        f = EC5Factors(
            k_mod=f.k_mod, gamma_M=f.gamma_M,
            k_h=k_h_solid(h),
            k_crit=f.k_crit, k_c=f.k_c, k_cr=f.k_cr, k_m=f.k_m,
        )
    elif auto_k_h and material_type == "glulam":
        f = EC5Factors(
            k_mod=f.k_mod, gamma_M=f.gamma_M,
            k_h=k_h_glulam(h),
            k_crit=f.k_crit, k_c=f.k_c, k_cr=f.k_cr, k_m=f.k_m,
        )
    f_t_d = f.k_mod * material.f_t_0_k / f.gamma_M * f.k_h
    A = _A(b, h)
    T_Rd = f_t_d * A
    DCR = abs(T_Ed) / max(T_Rd, 1e-30)
    return EC5TensionCheck(
        f_t_0_d=f_t_d, T_Rd=T_Rd, T_Ed=T_Ed,
        DCR=DCR, passes=DCR <= 1.0, factors=f,
    )


# ============================================================ compression

def ec5_compression_check(
    *,
    b: float, h: float, material: TimberMaterial,
    P_Ed: float,
    factors: Optional[EC5Factors] = None,
    l_ef: Optional[float] = None,
    auto_k_c: bool = True,
    material_type: str = "solid",
) -> EC5CompressionCheck:
    """EC5 §6.1.4 + §6.3.2 compression parallel + column stability.

    ``f_c,0,d = k_mod · f_c,0,k / gamma_M``, demand ``σ_c,d ≤ k_c · f_c,0,d``.
    """
    _validate(b, h)
    f = factors if factors is not None else EC5Factors()

    if auto_k_c and l_ef is not None:
        # Slenderness about the weak axis (controls buckling)
        d_weak = min(b, h)
        i = d_weak / math.sqrt(12.0)     # radius of gyration for rect
        slenderness = l_ef / i
        beta_c = 0.2 if material_type == "solid" else 0.1
        kc = k_c_column_stability(
            f_c_0_k=material.f_c_0_k,
            slenderness=slenderness,
            E_0_05=material.E_0_05,
            beta_c=beta_c,
        )
        f = EC5Factors(
            k_mod=f.k_mod, gamma_M=f.gamma_M,
            k_h=f.k_h, k_crit=f.k_crit, k_c=kc,
            k_cr=f.k_cr, k_m=f.k_m,
        )

    f_c_d = f.k_mod * material.f_c_0_k / f.gamma_M
    A = _A(b, h)
    P_Rd = f.k_c * f_c_d * A
    DCR = abs(P_Ed) / max(P_Rd, 1e-30)
    return EC5CompressionCheck(
        f_c_0_d=f_c_d, P_Rd=P_Rd, P_Ed=P_Ed,
        DCR=DCR, passes=DCR <= 1.0, factors=f,
    )


# ============================================================ shear

def ec5_shear_check(
    *,
    b: float, h: float, material: TimberMaterial,
    V_Ed: float,
    factors: Optional[EC5Factors] = None,
    material_type: str = "solid",
) -> EC5ShearCheck:
    """EC5 §6.1.7 shear parallel to grain.

    ``f_v,d = k_mod · f_v,k / gamma_M · k_cr``.
    For rectangular section: τ_max = 1.5·V/(b·h).
    ``k_cr = 0.67`` for solid timber (cracking), 1.0 for glulam
    (per EC5 §6.1.7 + amendments).
    """
    _validate(b, h)
    f = factors if factors is not None else EC5Factors()
    if material_type == "glulam":
        f = EC5Factors(
            k_mod=f.k_mod, gamma_M=f.gamma_M,
            k_h=f.k_h, k_crit=f.k_crit, k_c=f.k_c,
            k_cr=1.0, k_m=f.k_m,
        )
    f_v_d = f.k_mod * material.f_v_k / f.gamma_M * f.k_cr
    A = _A(b, h)
    V_Rd = f_v_d * A / 1.5
    DCR = abs(V_Ed) / max(V_Rd, 1e-30)
    return EC5ShearCheck(
        f_v_d=f_v_d, V_Rd=V_Rd, V_Ed=V_Ed,
        DCR=DCR, passes=DCR <= 1.0, factors=f,
    )


# ============================================================ combined

def ec5_combined_check(
    *,
    b: float, h: float, material: TimberMaterial,
    P_Ed: float = 0.0,
    M_strong: float = 0.0,
    M_weak: float = 0.0,
    is_tension: bool = False,
    factors: Optional[EC5Factors] = None,
    l_ef: Optional[float] = None,
    l_ef_compress: Optional[float] = None,
    material_type: str = "solid",
) -> EC5CombinedCheck:
    """EC5 §6.2.3 (tension+bending) or §6.2.4 (compression+bending).

    Tension+bending (§6.2.3 Eqs. 6.17 & 6.18):

        σ_t,0,d/f_t,0,d + σ_m,y,d/f_m,y,d + k_m · σ_m,z,d/f_m,z,d ≤ 1
        σ_t,0,d/f_t,0,d + k_m·σ_m,y,d/f_m,y,d + σ_m,z,d/f_m,z,d ≤ 1

    Compression+bending (§6.2.4 Eqs. 6.19 & 6.20):

        (σ_c,0,d/f_c,0,d)^2 + σ_m,y,d/f_m,y,d + k_m·σ_m,z,d/f_m,z,d ≤ 1
        (σ_c,0,d/f_c,0,d)^2 + k_m·σ_m,y,d/f_m,y,d + σ_m,z,d/f_m,z,d ≤ 1

    Plus when column / lateral stability matters
    (§6.3.2 / 6.3.3 Eqs. 6.23-6.24), the linear σ_c/f_c becomes
    σ_c/(k_c·f_c).
    """
    _validate(b, h)
    f = factors if factors is not None else EC5Factors()
    if l_ef_compress is None:
        l_ef_compress = l_ef

    # Individual demand stresses
    A = _A(b, h)
    S_z = _S_strong(b, h)   # bending about strong (y) axis -> M_y
    S_y = _S_weak(b, h)
    sigma_axial = abs(P_Ed) / A
    sigma_my = abs(M_strong) / S_z
    sigma_mz = abs(M_weak) / S_y

    if is_tension:
        t = ec5_tension_check(
            b=b, h=h, material=material, T_Ed=P_Ed,
            factors=f, material_type=material_type,
        )
        # Bending: no k_crit (tension face)
        f_no_crit = EC5Factors(
            k_mod=f.k_mod, gamma_M=f.gamma_M,
            k_h=k_h_solid(h) if material_type == "solid" else k_h_glulam(h),
            k_crit=1.0, k_c=f.k_c, k_cr=f.k_cr, k_m=f.k_m,
        )
        b_y = ec5_bending_check(
            b=b, h=h, material=material, M_Ed=M_strong,
            factors=f_no_crit, auto_k_crit=False, auto_k_h=False,
            material_type=material_type,
        )
        b_z = ec5_bending_check(
            b=h, h=b, material=material, M_Ed=M_weak,
            factors=f_no_crit, auto_k_crit=False, auto_k_h=False,
            material_type=material_type,
        )
        r_t = sigma_axial / t.f_t_0_d
        r_my = sigma_my / b_y.f_m_d
        r_mz = sigma_mz / b_z.f_m_d if b_z.f_m_d > 0 else 0
        eq1 = r_t + r_my + f.k_m * r_mz
        eq2 = r_t + f.k_m * r_my + r_mz
        components = {"r_t": r_t, "r_my": r_my, "r_mz": r_mz}
        note = "EC5 §6.2.3 (tension + bending)"
    else:
        c = ec5_compression_check(
            b=b, h=h, material=material, P_Ed=P_Ed,
            factors=f, l_ef=l_ef_compress,
            auto_k_c=(l_ef_compress is not None),
            material_type=material_type,
        )
        b_y = ec5_bending_check(
            b=b, h=h, material=material, M_Ed=M_strong,
            factors=f, l_ef=l_ef,
            auto_k_crit=(l_ef is not None),
            material_type=material_type,
        )
        b_z = ec5_bending_check(
            b=h, h=b, material=material, M_Ed=M_weak,
            factors=f, l_ef=l_ef,
            auto_k_crit=False, auto_k_h=False,
            material_type=material_type,
        )
        # Use squared for compression per EC5 6.2.4
        r_c = sigma_axial / c.f_c_0_d
        r_c_sq = r_c ** 2
        r_my = sigma_my / b_y.f_m_d
        r_mz = sigma_mz / b_z.f_m_d if b_z.f_m_d > 0 else 0
        eq1 = r_c_sq + r_my + f.k_m * r_mz
        eq2 = r_c_sq + f.k_m * r_my + r_mz
        components = {
            "(r_c)^2": r_c_sq, "r_c": r_c,
            "r_my": r_my, "r_mz": r_mz,
        }
        note = "EC5 §6.2.4 (compression + bending)"

    interaction = max(eq1, eq2)
    return EC5CombinedCheck(
        interaction=interaction,
        interaction_alt=min(eq1, eq2),
        passes=interaction <= 1.0,
        note=note,
        components=components,
    )
