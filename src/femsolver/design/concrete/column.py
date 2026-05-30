"""Rectangular tied-column P-M interaction surface per ACI 318-19 Ch. 22.4.

For a given trial neutral-axis depth ``c`` (measured from the extreme
compression fiber), strain compatibility and equilibrium give one
point ``(P_n, M_n)`` on the nominal interaction surface:

* Concrete contributes through the Whitney stress block: uniform
  stress ``0.85 f_c'`` over depth ``a = β_1 c`` (capped at ``h``).
* Each rebar layer contributes ``A_si · (f_si - 0.85 f_c')`` when it
  lies in the compression block (the bar displaces concrete), or
  ``A_si · f_si`` otherwise, where ``f_si = clip(E_s ε_i, -f_y, +f_y)``
  and ``ε_i = ε_cu (c - d_i) / c``.

Convention
----------
* Compression is **positive** for both axial force ``P_n`` and the
  stress at each fiber.
* Positive moment ``M_n`` produces tension at the bottom of the
  section (sagging) -- i.e., compression at the top.
* The "extreme compression fiber" is at the top of the section
  (depth 0); ``d_i`` is the depth from this fiber to each rebar
  layer.

ACI provisions
--------------
* §22.4.2.1: ``P_n_max = 0.80 · P_o`` for **tied** columns;
  ``0.85 · P_o`` for **spirally-reinforced** columns.
* §22.4.2.2: ``P_o = 0.85 f_c' (A_g - A_st) + f_y A_st``.
* Strain compatibility: ``ε_cu = 0.003`` at extreme compression
  fiber.
* §21.2.2: ``φ`` is interpolated from 0.65 (tied, compression-
  controlled) / 0.75 (spiral) to 0.90 (tension-controlled) based on
  the extreme tension steel's strain ``ε_t``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from femsolver.design.concrete.section import (
    EPSILON_CU,
    E_STEEL,
    ConcreteSection,
    phi_for_strain,
)


# ============================================================ result types

@dataclass
class InteractionPoint:
    """One point on the P-M interaction surface.

    Attributes
    ----------
    c : float
        Neutral-axis depth from extreme compression fiber (m). Can be
        ``inf`` for the pure-axial-compression cap.
    P_n : float
        Nominal axial capacity (N), compression positive.
    M_n : float
        Nominal moment capacity (N·m).
    phi : float
        Strength-reduction factor per Table 21.2.2.
    phi_P_n : float
        Design axial capacity = ``φ · P_n``.
    phi_M_n : float
        Design moment capacity = ``φ · M_n``.
    epsilon_t : float
        Strain in the extreme tension steel (positive when tensile,
        zero when no tension steel exists in tension zone).
    section_type : str
        ``"tension-controlled"`` / ``"transition"`` /
        ``"compression-controlled"`` / ``"pure-compression-cap"``.
    """

    c: float
    P_n: float
    M_n: float
    phi: float
    phi_P_n: float
    phi_M_n: float
    epsilon_t: float
    section_type: str


@dataclass
class InteractionSurface:
    """Full P-M interaction surface for a column section.

    Attributes
    ----------
    points : list[InteractionPoint]
        Surface points sorted by descending ``P_n`` (top of curve
        first). The list starts with the ``P_n_max`` cap, sweeps
        through the balanced and tension-controlled regions, and ends
        at the pure-tension point.
    P_o : float
        Pure-axial-compression capacity per 22.4.2.2:
        ``0.85 f_c' (A_g - A_st) + f_y A_st``.
    P_n_max : float
        Cap on the axial capacity per 22.4.2.1: ``0.80 P_o`` (tied)
        or ``0.85 P_o`` (spiral).
    P_n_pure_tension : float
        Pure axial-tension capacity = ``-A_st · f_y`` (negative).
    M_o : float
        Pure flexural capacity (M_n at P_n = 0, approximate by
        interpolation of the surface).
    spiral : bool
        Whether the column is spirally reinforced.
    """

    points: list[InteractionPoint] = field(default_factory=list)
    P_o: float = 0.0
    P_n_max: float = 0.0
    P_n_pure_tension: float = 0.0
    M_o: float = 0.0
    spiral: bool = False

    def __post_init__(self) -> None:
        # Ensure sorted by descending P_n
        self.points = sorted(self.points, key=lambda p: -p.P_n)

    def phi_M_n_at_P_u(self, P_u: float) -> float:
        """Design moment capacity at axial demand ``P_u`` (linear
        interpolation across the surface points)."""
        if not self.points:
            return 0.0
        # Sort by P_n ascending for searchsorted-style interpolation
        pts = sorted(self.points, key=lambda p: p.P_n)
        Ps = [p.P_n for p in pts]
        if P_u >= Ps[-1]:
            # Above max P_n on the surface: capacity is essentially
            # the P_n_max cap with M_n -> 0
            return 0.0
        if P_u <= Ps[0]:
            return pts[0].phi_M_n
        # Find bracketing pair
        for i in range(len(Ps) - 1):
            if Ps[i] <= P_u <= Ps[i + 1]:
                t = ((P_u - Ps[i]) / (Ps[i + 1] - Ps[i])
                     if Ps[i + 1] != Ps[i] else 0.0)
                return pts[i].phi_M_n + t * (pts[i + 1].phi_M_n
                                                - pts[i].phi_M_n)
        return 0.0

    def dcr(self, P_u: float, M_u: float) -> float:
        """Demand-capacity ratio for a demand ``(P_u, M_u)``.

        Uses the standard "moment usage at axial demand" check:
        ``DCR = |M_u| / φ M_n(P_u)``. Returns ``inf`` if
        ``φ M_n(P_u) = 0`` (i.e., demand is at or above the P_n_max
        cap).
        """
        phi_M = self.phi_M_n_at_P_u(P_u)
        if phi_M <= 0.0:
            return float("inf") if abs(M_u) > 0.0 else 0.0
        return abs(M_u) / phi_M


# ============================================================ point evaluator

def column_interaction_point(section: ConcreteSection, c: float, *,
                                spiral: bool = False) -> InteractionPoint:
    """Compute one point ``(P_n, M_n)`` on the interaction surface for a
    given neutral-axis depth ``c``.

    The bar layout used is from ``section.rebar``: ``top_bars`` at
    depth ``top_cover`` from the extreme compression fiber, and
    ``bottom_bars`` at depth ``h - bottom_cover``. Steel-area centroid
    is treated as concentrated at each layer's depth (acceptable for
    typical layouts).
    """
    if c <= 0.0:
        raise ValueError(f"c must be positive, got {c}")
    fc = section.material.fc_prime
    fy = section.material.fy
    beta_1 = section.material.beta_1
    eps_ty = section.material.epsilon_ty
    b = section.b
    h = section.h

    # Bar layers (depth from compression fiber, area)
    bar_layers: list[tuple[float, float]] = []
    if section.rebar.As_top > 0.0:
        bar_layers.append((section.rebar.top_cover, section.rebar.As_top))
    if section.rebar.As_bottom > 0.0:
        bar_layers.append((h - section.rebar.bottom_cover,
                            section.rebar.As_bottom))

    # Concrete force (Whitney block, depth a from top, capped at h)
    a = min(beta_1 * c, h)
    F_c = 0.85 * fc * b * a
    M_c = F_c * (0.5 * h - 0.5 * a)        # arm = h/2 - a/2

    # Steel forces -- with concrete displacement correction for bars
    # in the compression block.
    F_s_total = 0.0
    M_s_total = 0.0
    eps_t_signed = float("inf")     # track most-tensile (most-negative) strain
    for d_i, A_si in bar_layers:
        eps_i = EPSILON_CU * (c - d_i) / c
        # Clip stress to yield
        f_si = max(-fy, min(fy, E_STEEL * eps_i))
        # Concrete-displacement correction: bar inside the Whitney
        # block displaces concrete that we already counted in F_c
        if d_i <= a:
            f_si_eff = f_si - 0.85 * fc
        else:
            f_si_eff = f_si
        F_si = A_si * f_si_eff
        F_s_total += F_si
        M_s_total += F_si * (0.5 * h - d_i)
        if eps_i < eps_t_signed:
            eps_t_signed = eps_i

    P_n = F_c + F_s_total
    M_n = M_c + M_s_total

    # Extreme tension steel strain (positive when tensile)
    eps_t = -eps_t_signed if eps_t_signed < 0.0 else 0.0
    phi = phi_for_strain(eps_t, eps_ty, spiral=spiral)

    if eps_t >= 0.005:
        section_type = "tension-controlled"
    elif eps_t <= eps_ty:
        section_type = "compression-controlled"
    else:
        section_type = "transition"

    return InteractionPoint(
        c=c, P_n=P_n, M_n=M_n, phi=phi,
        phi_P_n=phi * P_n, phi_M_n=phi * M_n,
        epsilon_t=eps_t, section_type=section_type,
    )


# ============================================================ surface builder

def _Po(section: ConcreteSection) -> float:
    """Pure axial-compression capacity per ACI 22.4.2.2:
    ``P_o = 0.85 f_c' (A_g - A_st) + f_y A_st``."""
    fc = section.material.fc_prime
    fy = section.material.fy
    A_st = section.rebar.As_top + section.rebar.As_bottom
    A_g = section.b * section.h
    return 0.85 * fc * (A_g - A_st) + fy * A_st


def column_interaction_surface(
    section: ConcreteSection,
    *,
    n_points: int = 40,
    spiral: bool = False,
) -> InteractionSurface:
    """Construct the full nominal+design P-M interaction surface for a
    rectangular tied column with the given section.

    Parameters
    ----------
    section : ConcreteSection
    n_points : int, default 40
        Number of neutral-axis depths to sweep. The c values are
        chosen on a logarithmic-style grid that captures the
        compression-controlled, balanced, and tension-controlled
        regions evenly.
    spiral : bool, default False
        ``True`` for spirally-reinforced columns (changes the
        ``P_n_max`` cap and the compression-controlled φ from 0.65
        to 0.75).

    Returns
    -------
    InteractionSurface
        Points sorted by descending P_n, including the
        ``P_n_max`` cap and the pure-tension end.
    """
    h = section.h
    d_t = h - section.rebar.bottom_cover     # extreme tension steel depth
    if d_t <= 0.0:
        raise ValueError(
            "section has no usable tension steel for P-M surface"
        )

    # Range of c: from very small (deep tension-controlled, eps_t huge)
    # to very large (whole section in compression).
    # c_balanced ~ ε_cu / (ε_cu + ε_ty) · d_t
    eps_ty = section.material.epsilon_ty
    c_bal = EPSILON_CU * d_t / (EPSILON_CU + eps_ty)
    c_tc = EPSILON_CU * d_t / (EPSILON_CU + 0.005)    # tension-ctrl boundary
    # Sweep c from a small fraction of c_tc to a large multiple of h.
    c_min = 0.05 * c_tc
    c_max = 5.0 * h
    cs = []
    # Concentrate points near balanced/transition zones
    n_low = n_points // 3
    n_high = n_points - n_low
    for i in range(n_low):
        t = i / (n_low - 1 if n_low > 1 else 1)
        cs.append(c_min * (c_tc / c_min) ** t)
    for i in range(n_high):
        t = (i + 1) / n_high
        cs.append(c_tc * (c_max / c_tc) ** t)

    # Build points
    pts = [column_interaction_point(section, c, spiral=spiral) for c in cs]

    # Add pure-tension end (M = 0, P = -A_st · f_y, phi for ε_t → ∞)
    A_st = section.rebar.As_top + section.rebar.As_bottom
    P_pure_tens = -A_st * section.material.fy
    pts.append(InteractionPoint(
        c=0.0,
        P_n=P_pure_tens,
        M_n=0.0,
        phi=0.90,        # tension-controlled
        phi_P_n=0.90 * P_pure_tens,
        phi_M_n=0.0,
        epsilon_t=float("inf"),
        section_type="tension-controlled",
    ))

    # Add the pure-compression cap (P_n_max) per 22.4.2.1
    P_o = _Po(section)
    cap_factor = 0.85 if spiral else 0.80
    P_n_max = cap_factor * P_o
    phi_comp = 0.75 if spiral else 0.65
    pts.append(InteractionPoint(
        c=float("inf"),
        P_n=P_n_max,
        M_n=0.0,
        phi=phi_comp,
        phi_P_n=phi_comp * P_n_max,
        phi_M_n=0.0,
        epsilon_t=0.0,
        section_type="pure-compression-cap",
    ))

    # Cap any computed P_n above P_n_max at the cap value, with M_n -> 0
    # (this enforces ACI 22.4.2.1 over the high-axial region).
    pts = _apply_axial_cap(pts, P_n_max, phi_comp)

    # M_o (capacity at P_n ~ 0): interpolate
    # Find points bracketing P_n = 0
    sorted_pts = sorted(pts, key=lambda p: p.P_n)
    M_o = 0.0
    for i in range(len(sorted_pts) - 1):
        if sorted_pts[i].P_n <= 0.0 <= sorted_pts[i + 1].P_n:
            t = (0.0 - sorted_pts[i].P_n) / (
                sorted_pts[i + 1].P_n - sorted_pts[i].P_n
            ) if sorted_pts[i + 1].P_n != sorted_pts[i].P_n else 0.0
            M_o = (sorted_pts[i].phi_M_n
                   + t * (sorted_pts[i + 1].phi_M_n
                          - sorted_pts[i].phi_M_n))
            break

    return InteractionSurface(
        points=pts,
        P_o=P_o,
        P_n_max=P_n_max,
        P_n_pure_tension=P_pure_tens,
        M_o=M_o,
        spiral=spiral,
    )


def _apply_axial_cap(points: list[InteractionPoint], P_n_max: float,
                      phi_comp: float) -> list[InteractionPoint]:
    """Clip any point with ``P_n > P_n_max`` down to ``(P_n_max, 0)``
    with compression-controlled φ -- per ACI 22.4.2.1."""
    capped: list[InteractionPoint] = []
    for p in points:
        if p.P_n > P_n_max:
            capped.append(InteractionPoint(
                c=p.c, P_n=P_n_max, M_n=0.0,
                phi=phi_comp,
                phi_P_n=phi_comp * P_n_max,
                phi_M_n=0.0,
                epsilon_t=p.epsilon_t,
                section_type="pure-compression-cap",
            ))
        else:
            capped.append(p)
    return capped
