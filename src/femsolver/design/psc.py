"""Prestressed-concrete (PSC) limit-state design checks.

Bridge prestressed-concrete design lives or dies on a handful of
limit-state checks that all consume the **same** internal actions:

* the effective prestress force ``P`` (after losses),
* the *primary* prestress moment ``P·e`` (the eccentric tendon), and
* the *secondary* (hyperstatic / parasitic) prestress moment
  ``M_sec`` -- which in a continuous structure is a real applied moment
  from the redundant reactions (see
  :func:`femsolver.bridges.tendon_secondary_forces`).

This module provides:

* :func:`psc_extreme_fiber_stresses` -- the canonical service stress
  equation ``f = P/A ∓ P·e/S ± M/S`` evaluated at the top and bottom
  fibres, with ``M`` the *total* applied moment ``M_external + M_sec``.
* **AASHTO LRFD** §5.9.2.3 stress limits at transfer and at service
  (Class U / T), and the §3.4.1 factored-moment demand (secondary
  prestress at load factor 1.0).
* **EN 1992-1-1 / EN 1992-2** §5.10.2.2 (transfer) and §7.2 stress
  limits (characteristic + quasi-permanent combinations) and the
  §7.3.1 decompression check.

Sign convention
---------------
Stresses are **compression-positive** here (the prevailing PSC design
convention). ``M`` is sagging-positive (tension on the bottom fibre);
``e`` is the tendon eccentricity **below** the centroid (positive).
``P`` is the (positive) effective prestress force.

Units are SI throughout (Pa, N, N·m, m). The AASHTO ``√f'c`` limits
are defined in US customary ksi, so those helpers convert through ksi
internally and return Pa -- exactly, no approximation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

_KSI = 6.894757e6        # 1 ksi in Pa
_MPA = 1.0e6


# ============================================================ section

@dataclass
class PscSection:
    """Geometric section properties for a PSC stress check.

    Attributes
    ----------
    A : float
        Cross-section area (m²).
    I : float
        Second moment of area about the bending axis (m⁴).
    y_top, y_bot : float
        Distances from the centroid to the top and bottom extreme
        fibres (m, both positive).
    """

    A: float
    I: float
    y_top: float
    y_bot: float

    def __post_init__(self) -> None:
        if min(self.A, self.I, self.y_top, self.y_bot) <= 0.0:
            raise ValueError("A, I, y_top, y_bot must all be > 0")

    @property
    def S_top(self) -> float:
        """Section modulus to the top fibre (m³)."""
        return self.I / self.y_top

    @property
    def S_bot(self) -> float:
        """Section modulus to the bottom fibre (m³)."""
        return self.I / self.y_bot

    @classmethod
    def rectangular(cls, b: float, h: float) -> "PscSection":
        A = b * h
        I = b * h ** 3 / 12.0
        return cls(A=A, I=I, y_top=h / 2.0, y_bot=h / 2.0)

    @classmethod
    def from_section(cls, section, *, axis: str = "z") -> "PscSection":
        """Build a :class:`PscSection` from a unified
        :class:`~femsolver.sections.Section` of **any** geometry
        (I, box, custom polygon with holes, ...).

        Pulls the gross area, the second moment of area about the
        bending axis, and the **exact** extreme-fibre distances from the
        polygon bounds and centroid -- so the SLS stress checks read the
        right `S_top`/`S_bot` for an asymmetric or voided section
        without any manual centroid arithmetic.

        Parameters
        ----------
        section : Section
            Any unified section (holes are already reflected in its
            gross properties).
        axis : {"z", "y"}, default "z"
            Bending axis. ``"z"`` (the usual strong axis) bends in the
            ``y`` direction → fibres are the top/bottom in ``y``;
            ``"y"`` bends in ``z``.

        Returns
        -------
        PscSection
        """
        cz, cy = section.centroid
        minz, miny, maxz, maxy = section.geometry.polygon.bounds
        if axis == "z":
            return cls(A=float(section.area), I=float(section.I_zz),
                       y_top=float(maxy - cy), y_bot=float(cy - miny))
        if axis == "y":
            return cls(A=float(section.area), I=float(section.I_yy),
                       y_top=float(maxz - cz), y_bot=float(cz - minz))
        raise ValueError("axis must be 'z' or 'y'")


# ============================================================ fiber stresses

@dataclass
class PscFiberStress:
    """Extreme-fibre stresses (compression-positive, Pa)."""
    f_top: float
    f_bot: float
    M_total: float        # external + secondary moment used (N·m)


def psc_extreme_fiber_stresses(
    section: PscSection, *, P: float, e: float, M: float,
) -> PscFiberStress:
    """Top / bottom fibre stresses of a PSC section.

    Compression-positive::

        f_top = P/A - P·e/S_top + M/S_top
        f_bot = P/A + P·e/S_bot - M/S_bot

    where ``M`` is the **total** applied moment, i.e. the external
    moment *plus the secondary prestress moment* ``M_sec``. The primary
    prestress moment ``P·e`` is carried by the eccentricity term.

    Parameters
    ----------
    section : PscSection
    P : float
        Effective prestress force (N, > 0 = compression on the section).
    e : float
        Tendon eccentricity below the centroid (m, positive down).
    M : float
        Total applied moment ``M_external + M_secondary`` (N·m, sagging
        positive).
    """
    A, St, Sb = section.A, section.S_top, section.S_bot
    f_top = P / A - P * e / St + M / St
    f_bot = P / A + P * e / Sb - M / Sb
    return PscFiberStress(f_top=float(f_top), f_bot=float(f_bot),
                          M_total=float(M))


# ============================================================ limit checks

@dataclass
class PscStressCheck:
    """Outcome of a PSC stress-limit check (compression-positive)."""
    f_top: float
    f_bot: float
    comp_limit: float        # allowable compression (Pa, positive)
    tens_limit: float        # allowable tension magnitude (Pa, >= 0)
    dcr_top: float
    dcr_bot: float
    DCR: float
    passes: bool
    code: str = ""
    state: str = ""

    @property
    def governing_fiber(self) -> str:
        return "top" if self.dcr_top >= self.dcr_bot else "bottom"


def _fiber_dcr(f: float, comp_limit: float, tens_limit: float) -> float:
    """Demand/capacity for one fibre. ``f`` compression-positive."""
    if f >= 0.0:                                  # compression
        return f / comp_limit if comp_limit > 0 else math.inf
    return (-f) / tens_limit if tens_limit > 0 else math.inf


def check_psc_stresses(
    section: PscSection,
    *,
    P: float,
    e: float,
    M_external: float,
    M_secondary: float = 0.0,
    comp_limit: float,
    tens_limit: float,
    code: str = "",
    state: str = "",
) -> PscStressCheck:
    """Evaluate fibre stresses (consuming primary + secondary moments)
    and compare to compression / tension allowables.

    ``M`` used in the stress equation is ``M_external + M_secondary``;
    the primary prestress moment ``P·e`` is in the eccentricity term.
    """
    M = M_external + M_secondary
    s = psc_extreme_fiber_stresses(section, P=P, e=e, M=M)
    dcr_t = _fiber_dcr(s.f_top, comp_limit, tens_limit)
    dcr_b = _fiber_dcr(s.f_bot, comp_limit, tens_limit)
    DCR = max(dcr_t, dcr_b)
    return PscStressCheck(
        f_top=s.f_top, f_bot=s.f_bot,
        comp_limit=comp_limit, tens_limit=tens_limit,
        dcr_top=dcr_t, dcr_bot=dcr_b, DCR=DCR,
        passes=DCR <= 1.0 + 1e-9, code=code, state=state,
    )


# ============================================================ AASHTO limits

def _sqrt_ksi(coeff: float, f_c: float, lam: float) -> float:
    """``coeff · λ · √(f'c)`` with the √ taken in ksi (AASHTO form),
    returned in Pa."""
    f_c_ksi = f_c / _KSI
    return lam * coeff * math.sqrt(f_c_ksi) * _KSI


def aashto_transfer_limits(
    f_ci: float, *, bonded_reinforcement: bool = False, lam: float = 1.0,
) -> tuple[float, float]:
    """AASHTO LRFD §5.9.2.3.1 stress limits *at transfer* (release).

    Parameters
    ----------
    f_ci : float
        Concrete strength at transfer (Pa).
    bonded_reinforcement : bool
        Whether bonded reinforcement is provided to resist the tensile
        force in the tension zone (raises the tensile allowable).
    lam : float
        Concrete density modification factor λ (1.0 normal weight).

    Returns
    -------
    (comp_limit, tens_limit) : tuple[float, float]
        Compression allowable ``0.60 f'ci`` and tension allowable
        (Pa, both positive magnitudes).
    """
    comp = 0.60 * f_ci
    if bonded_reinforcement:
        tens = _sqrt_ksi(0.24, f_ci, lam)
    else:
        tens = min(_sqrt_ksi(0.0948, f_ci, lam), 0.2 * _KSI)
    return float(comp), float(tens)


def aashto_service_limits(
    f_c: float, *, prestress_class: str = "U", transient: bool = True,
    lam: float = 1.0,
) -> tuple[float, float]:
    """AASHTO LRFD §5.9.2.3.2 stress limits *at service* (after losses).

    Parameters
    ----------
    f_c : float
        28-day concrete strength (Pa).
    prestress_class : {"U", "T"}
        Class U (uncracked): tension ≤ ``0.19λ√f'c`` (ksi form).
        Class T (transition): tension ≤ ``0.38λ√f'c``.
        (Class C is cracked -- use a cracked-section / crack-width
        check instead of a stress limit.)
    transient : bool
        If True the compression limit is ``0.60 f'c`` (effective
        prestress + permanent + transient loads); if False it is
        ``0.45 f'c`` (effective prestress + permanent loads).
    lam : float

    Returns
    -------
    (comp_limit, tens_limit) : tuple[float, float]
    """
    comp = (0.60 if transient else 0.45) * f_c
    if prestress_class == "U":
        tens = _sqrt_ksi(0.19, f_c, lam)
    elif prestress_class == "T":
        tens = _sqrt_ksi(0.38, f_c, lam)
    else:
        raise ValueError(
            "prestress_class must be 'U' or 'T' (Class C is cracked -- "
            "use a crack-width check)"
        )
    return float(comp), float(tens)


# ============================================================ EN 1992 limits

def ec2_f_ctm(f_ck: float) -> float:
    """EN 1992-1-1 Table 3.1 mean tensile strength ``f_ctm`` (Pa).

    ``f_ctm = 0.30 · f_ck^(2/3)`` (MPa) for ``f_ck ≤ 50 MPa``;
    ``f_ctm = 2.12 · ln(1 + f_cm/10)`` above, with ``f_cm = f_ck + 8``.
    """
    f_ck_MPa = f_ck / _MPA
    if f_ck_MPa <= 50.0:
        f_ctm_MPa = 0.30 * f_ck_MPa ** (2.0 / 3.0)
    else:
        f_cm = f_ck_MPa + 8.0
        f_ctm_MPa = 2.12 * math.log(1.0 + f_cm / 10.0)
    return f_ctm_MPa * _MPA


def ec2_transfer_limits(
    f_ck_t: float, *, allow_tension: bool = False,
) -> tuple[float, float]:
    """EN 1992-1-1 §5.10.2.2 stress limits *at transfer*.

    Compression ≤ ``0.60 f_ck(t)`` (the §5.10.2.2(5) limit; above
    ``0.45 f_ck(t)`` non-linear creep should be considered). Tension is
    normally limited to zero (decompression) unless cracking-controlled
    reinforcement is provided, in which case ``f_ctm(t)`` is allowed.
    """
    comp = 0.60 * f_ck_t
    tens = ec2_f_ctm(f_ck_t) if allow_tension else 0.0
    return float(comp), float(tens)


def ec2_service_limits(
    f_ck: float, *, combination: str = "characteristic",
) -> tuple[float, float]:
    """EN 1992-1-1 §7.2 compressive stress limits *at service*.

    * ``"characteristic"`` -> ``0.60 f_ck`` (limit longitudinal cracking
      in aggressive exposure, §7.2(2)).
    * ``"quasi-permanent"`` -> ``0.45 f_ck`` (keep creep linear,
      §7.2(3)).

    Tension allowable is 0 here (use :func:`ec2_decompression_check`
    for the decompression limit state, or a crack-width check).
    """
    if combination == "characteristic":
        comp = 0.60 * f_ck
    elif combination == "quasi-permanent":
        comp = 0.45 * f_ck
    else:
        raise ValueError(
            "combination must be 'characteristic' or 'quasi-permanent'"
        )
    return float(comp), 0.0


def ec2_decompression_check(
    section: PscSection, *, P: float, e: float,
    M_external: float, M_secondary: float = 0.0,
) -> PscStressCheck:
    """EN 1992-1-1 §7.3.1 decompression check (Table 7.1N, frequent
    combination for XC2-XS3 exposure): the concrete at the extreme
    fibre nearest the tendon must remain in compression (≥ 0).

    Returns a :class:`PscStressCheck` with ``tens_limit = 0`` -- it
    passes only if *both* fibres are non-tensile (the controlling
    bottom fibre under positive moment in particular).
    """
    return check_psc_stresses(
        section, P=P, e=e, M_external=M_external, M_secondary=M_secondary,
        comp_limit=math.inf, tens_limit=0.0,
        code="EN1992", state="decompression",
    )


# ============================================================ ULS demand

def psc_factored_moment(
    *,
    factored_external: float,
    M_secondary: float,
    gamma_secondary: float = 1.0,
) -> float:
    """Factored design moment ``M_Ed`` / ``M_u`` for the ULS flexure
    check, **including the secondary prestress moment**.

        M_u = (factored external moment) + γ_sec · M_secondary

    The secondary (hyperstatic) prestress moment is a genuine action on
    the indeterminate structure and is carried at a load factor of
    1.0 (AASHTO LRFD §3.4.1, the ``1.0·PS`` term; EN 1992 likewise
    applies ``γ_P = 1.0`` to the hyperstatic prestress effect at ULS).

    Parameters
    ----------
    factored_external : float
        Σ γ_i · M_i over the external load cases (N·m), already factored
        by the caller for the relevant strength combination.
    M_secondary : float
        Secondary prestress moment at the section (N·m).
    gamma_secondary : float, default 1.0
        Load factor on the secondary moment.

    Returns
    -------
    float : the factored design moment to compare against ``φM_n`` /
    ``M_Rd``.
    """
    return float(factored_external + gamma_secondary * M_secondary)


# ============================================================ convenience checks

def aashto_transfer_check(
    section: PscSection, *, P: float, e: float,
    M_external: float, M_secondary: float = 0.0,
    f_ci: float, bonded_reinforcement: bool = False, lam: float = 1.0,
) -> PscStressCheck:
    """AASHTO transfer stress check (consumes primary + secondary)."""
    comp, tens = aashto_transfer_limits(
        f_ci, bonded_reinforcement=bonded_reinforcement, lam=lam
    )
    c = check_psc_stresses(
        section, P=P, e=e, M_external=M_external, M_secondary=M_secondary,
        comp_limit=comp, tens_limit=tens, code="AASHTO", state="transfer",
    )
    return c


def aashto_service_check(
    section: PscSection, *, P: float, e: float,
    M_external: float, M_secondary: float = 0.0,
    f_c: float, prestress_class: str = "U", transient: bool = True,
    lam: float = 1.0,
) -> PscStressCheck:
    """AASHTO service stress check (consumes primary + secondary)."""
    comp, tens = aashto_service_limits(
        f_c, prestress_class=prestress_class, transient=transient, lam=lam
    )
    return check_psc_stresses(
        section, P=P, e=e, M_external=M_external, M_secondary=M_secondary,
        comp_limit=comp, tens_limit=tens, code="AASHTO", state="service",
    )


def ec2_transfer_check(
    section: PscSection, *, P: float, e: float,
    M_external: float, M_secondary: float = 0.0,
    f_ck_t: float, allow_tension: bool = False,
) -> PscStressCheck:
    """EN 1992 transfer stress check (consumes primary + secondary)."""
    comp, tens = ec2_transfer_limits(f_ck_t, allow_tension=allow_tension)
    return check_psc_stresses(
        section, P=P, e=e, M_external=M_external, M_secondary=M_secondary,
        comp_limit=comp, tens_limit=tens, code="EN1992", state="transfer",
    )


def ec2_service_check(
    section: PscSection, *, P: float, e: float,
    M_external: float, M_secondary: float = 0.0,
    f_ck: float, combination: str = "characteristic",
) -> PscStressCheck:
    """EN 1992 §7.2 service compression check (consumes primary +
    secondary)."""
    comp, tens = ec2_service_limits(f_ck, combination=combination)
    return check_psc_stresses(
        section, P=P, e=e, M_external=M_external, M_secondary=M_secondary,
        comp_limit=comp, tens_limit=tens, code="EN1992",
        state=f"service-{combination}",
    )


# ============================================================ ULS flexure capacity

@dataclass
class FlexureCapacity:
    """ULS flexural capacity of a bonded PSC section.

    Attributes
    ----------
    M_n : float
        Nominal moment capacity (N·m).
    phi : float
        Resistance / strength-reduction factor (AASHTO φ, or 1.0 for the
        EN 1992 partial-factor formulation where γ's are inside).
    phi_M_n : float
        Design moment capacity ``φ·M_n`` (AASHTO) or ``M_Rd`` (EN 1992).
    c : float
        Neutral-axis depth (m).
    a : float
        Equivalent stress-block depth (m).
    f_ps : float
        Stress in the prestressing steel at flexural capacity (Pa).
    section_type : str
        ``"rectangular"`` or ``"flanged"`` compression zone.
    controlled : str
        ``"tension"`` / ``"transition"`` / ``"compression"`` (AASHTO).
    code : str
    """
    M_n: float
    phi: float
    phi_M_n: float
    c: float
    a: float
    f_ps: float
    section_type: str = "rectangular"
    controlled: str = ""
    code: str = ""


def aashto_beta1(f_c: float) -> float:
    """AASHTO LRFD §5.6.2.2 stress-block factor ``β1``.

    ``0.85`` for ``f'c ≤ 28 MPa``; reduce ``0.05`` per ``7 MPa`` above,
    not below ``0.65``.
    """
    f_c_MPa = f_c / _MPA
    if f_c_MPa <= 28.0:
        return 0.85
    return max(0.85 - 0.05 * (f_c_MPa - 28.0) / 7.0, 0.65)


def aashto_flexure_capacity(
    *,
    A_ps: float, f_pu: float, f_py: float, d_p: float,
    b: float, f_c: float,
    A_s: float = 0.0, f_y: float = 0.0, d_s: float = 0.0,
    b_w: Optional[float] = None, h_f: Optional[float] = None,
    eps_cl: float = 0.002, eps_tl: float = 0.005,
) -> FlexureCapacity:
    """AASHTO LRFD §5.6.3 flexural capacity of a bonded PSC member,
    approximate (``f_ps``) method.

    ``f_ps = f_pu (1 - k·c/d_p)``, ``k = 2(1.04 - f_py/f_pu)``; the
    neutral axis ``c`` follows from horizontal equilibrium with the
    Whitney stress block. Rectangular behaviour is assumed unless a
    flange (``b``, ``b_w``, ``h_f``) is given and ``a = β1·c > h_f``, in
    which case the T-section formula is used.

    The strength-reduction factor φ is interpolated from the net tensile
    strain ``ε_t`` (AASHTO §5.5.4.2): 1.0 (tension-controlled,
    ``ε_t ≥ ε_tl``) down to 0.75 (compression-controlled,
    ``ε_t ≤ ε_cl``).

    Parameters
    ----------
    A_ps, f_pu, f_py, d_p : prestressing steel area, ultimate / yield
        stress, and depth to its centroid.
    b, f_c : compression-face width and concrete strength.
    A_s, f_y, d_s : optional bonded mild steel (tension).
    b_w, h_f : web width and flange thickness for a flanged section.
    """
    if A_ps <= 0 or f_pu <= 0 or d_p <= 0 or b <= 0 or f_c <= 0:
        raise ValueError("A_ps, f_pu, d_p, b, f_c must be > 0")
    k = 2.0 * (1.04 - f_py / f_pu)
    b1 = aashto_beta1(f_c)
    kdp = k * A_ps * f_pu / d_p

    # rectangular trial
    c = (A_ps * f_pu + A_s * f_y) / (0.85 * f_c * b1 * b + kdp)
    section_type = "rectangular"
    if h_f is not None and b_w is not None and b1 * c > h_f:
        # T-section: subtract the overhang-flange compression force
        flange_force = 0.85 * f_c * (b - b_w) * h_f
        c = (A_ps * f_pu + A_s * f_y - flange_force) / (
            0.85 * f_c * b1 * b_w + kdp
        )
        section_type = "flanged"
    a = b1 * c
    f_ps = f_pu * (1.0 - k * c / d_p)

    if section_type == "flanged":
        M_n = (A_ps * f_ps * (d_p - a / 2.0)
               + A_s * f_y * (d_s - a / 2.0)
               + 0.85 * f_c * (b - b_w) * h_f * (a / 2.0 - h_f / 2.0))
    else:
        M_n = (A_ps * f_ps * (d_p - a / 2.0)
               + A_s * f_y * (d_s - a / 2.0))

    # φ from net tensile strain at the extreme prestressing/steel layer
    d_t = max(d_p, d_s)
    eps_t = 0.003 * (d_t - c) / c
    if eps_t >= eps_tl:
        phi, controlled = 1.0, "tension"
    elif eps_t <= eps_cl:
        phi, controlled = 0.75, "compression"
    else:
        phi = 0.75 + 0.25 * (eps_t - eps_cl) / (eps_tl - eps_cl)
        controlled = "transition"

    return FlexureCapacity(
        M_n=float(M_n), phi=float(phi), phi_M_n=float(phi * M_n),
        c=float(c), a=float(a), f_ps=float(f_ps),
        section_type=section_type, controlled=controlled, code="AASHTO",
    )


def ec2_flexure_capacity(
    *,
    A_p: float, f_p01k: float, d_p: float,
    b: float, f_ck: float,
    A_s: float = 0.0, f_yk: float = 0.0, d_s: float = 0.0,
    gamma_c: float = 1.5, gamma_s: float = 1.15, alpha_cc: float = 1.0,
) -> FlexureCapacity:
    """EN 1992-1-1 §6.1 flexural capacity ``M_Rd`` of a bonded PSC
    section (simplified rectangular stress block, ``f_ck ≤ 50 MPa``).

    Bonded ductile tendons are taken to reach the design strength
    ``f_pd = f_p01k / γ_s``; mild steel reaches ``f_yd = f_yk / γ_s``.
    Concrete design strength ``f_cd = α_cc f_ck / γ_c``, stress block
    ``λ = 0.8``, ``η = 1.0``. The neutral axis ``x`` follows from
    horizontal equilibrium.

    Parameters
    ----------
    A_p, f_p01k, d_p : tendon area, 0.1%-proof characteristic stress,
        and depth.
    b, f_ck : width and characteristic concrete strength.
    A_s, f_yk, d_s : optional mild steel.
    gamma_c, gamma_s, alpha_cc : partial / long-term factors.

    Returns
    -------
    FlexureCapacity (``phi`` reported as 1.0; the safety is inside the
    γ-factors, so ``phi_M_n == M_Rd``).
    """
    if A_p <= 0 or f_p01k <= 0 or d_p <= 0 or b <= 0 or f_ck <= 0:
        raise ValueError("A_p, f_p01k, d_p, b, f_ck must be > 0")
    f_cd = alpha_cc * f_ck / gamma_c
    f_pd = f_p01k / gamma_s
    f_yd = f_yk / gamma_s
    lam, eta = 0.8, 1.0
    T = A_p * f_pd + A_s * f_yd
    x = T / (eta * f_cd * b * lam)
    a = lam * x
    M_Rd = A_p * f_pd * (d_p - a / 2.0) + A_s * f_yd * (d_s - a / 2.0)
    return FlexureCapacity(
        M_n=float(M_Rd), phi=1.0, phi_M_n=float(M_Rd),
        c=float(x), a=float(a), f_ps=float(f_pd),
        section_type="rectangular", controlled="", code="EN1992",
    )


@dataclass
class PscFlexureCheck:
    """ULS flexure check ``M_u ≤ φM_n`` (consumes the secondary moment
    through ``M_u``)."""
    M_u: float
    phi_M_n: float
    DCR: float
    passes: bool
    capacity: FlexureCapacity


def psc_flexure_check(*, M_u: float, capacity: FlexureCapacity) -> PscFlexureCheck:
    """Compare a factored design moment ``M_u`` (from
    :func:`psc_factored_moment`, which already folds in the secondary
    prestress moment) against the design capacity ``φM_n`` / ``M_Rd``.
    """
    cap = capacity.phi_M_n
    dcr = abs(M_u) / cap if cap > 0 else math.inf
    return PscFlexureCheck(
        M_u=float(M_u), phi_M_n=float(cap), DCR=float(dcr),
        passes=bool(dcr <= 1.0 + 1e-9), capacity=capacity,
    )
