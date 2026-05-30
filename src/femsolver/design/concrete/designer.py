"""Iterative member designer for reinforced-concrete beams and columns.

Given a section geometry (``b, h``), material, cover, and a factored
force demand envelope, the designer searches over standard bar sizes
and counts to find a rebar layout that:

* Satisfies flexural strength: ``φ M_n >= M_u`` (both sagging and
  hogging if applicable).
* Satisfies shear strength: ``φ V_n >= V_u`` and minimum stirrup
  requirements (ACI 9.6.3, 9.7.6.2).
* For columns: satisfies the P-M interaction surface (``DCR <= 1``)
  within the ACI 10.6.1.1 limits ``0.01 <= ρ <= 0.08``.

The search is breadth-first over (bar size, bar count) combinations
and returns the **lightest** layout (minimum steel area) that
satisfies every check. The user supplies a candidate bar-size list;
the default list covers the common #5-#11 range.

This is a deliberately simple greedy designer. More sophisticated
strategies (asymmetric reinforcement, optimised bar grouping, etc.)
are future extensions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from femsolver.design.concrete.column import (
    InteractionSurface,
    column_interaction_surface,
)
from femsolver.design.concrete.flexure import (
    FlexuralCheck,
    beam_flexural_strength,
)
from femsolver.design.concrete.section import (
    ConcreteMaterial,
    ConcreteSection,
    RebarLayout,
    rebar_area,
)
from femsolver.design.concrete.shear import (
    ShearCheck,
    ShearDesign,
    beam_shear_strength,
    design_stirrup_spacing,
)


DEFAULT_BEAM_BAR_OPTIONS = ("#5", "#6", "#7", "#8", "#9", "#10", "#11")
DEFAULT_COLUMN_BAR_OPTIONS = ("#7", "#8", "#9", "#10", "#11")


# ============================================================ demand types

@dataclass
class BeamDesignDemand:
    """Factored force demand envelope on a beam member.

    Attributes
    ----------
    M_u_positive : float
        Maximum positive (sagging) factored moment (N·m, positive).
    M_u_negative : float
        Magnitude of the maximum negative (hogging) factored moment
        (N·m, supply as a positive number).
    V_u : float
        Maximum factored shear (N).
    """

    M_u_positive: float = 0.0
    M_u_negative: float = 0.0
    V_u: float = 0.0


@dataclass
class ColumnDesignDemand:
    """Factored force demand on a column member.

    Attributes
    ----------
    P_u : float
        Axial force (N, compression positive).
    M_u : float
        Bending moment magnitude (N·m).
    V_u : float, default 0
        Shear demand (N).
    """

    P_u: float
    M_u: float
    V_u: float = 0.0


# ============================================================ result types

@dataclass
class BeamDesignResult:
    section: ConcreteSection | None
    flexure_positive: FlexuralCheck | None
    flexure_negative: FlexuralCheck | None
    shear: ShearCheck | None
    success: bool
    notes: str = ""


@dataclass
class ColumnDesignResult:
    section: ConcreteSection | None
    interaction_surface: InteractionSurface | None
    dcr: float
    rho: float                     # ρ = A_st / A_g
    success: bool
    notes: str = ""


# ============================================================ beam designer

def design_beam(
    b: float,
    h: float,
    material: ConcreteMaterial,
    demand: BeamDesignDemand,
    *,
    cover: float = 0.040,
    bar_options: tuple = DEFAULT_BEAM_BAR_OPTIONS,
    stirrup_designation: str = "#3",
    stirrup_legs: int = 2,
    max_bars_per_layer: int = 8,
    fallback_stirrup_spacing: float = 0.150,
) -> BeamDesignResult:
    """Design a rectangular RC beam: bottom/top flexural rebar +
    stirrups for a given (M_u, V_u) demand.

    Strategy
    --------
    For each candidate bar size (smallest first):
      For ``n`` = 2, 3, ..., max_bars_per_layer:
        Build a candidate bottom-bar layout (and matching top layout
        if hogging demand exists), evaluate flexural capacity at
        both faces, design stirrup spacing for shear, and check
        all conditions. Track the lightest layout that satisfies
        every check.

    Returns
    -------
    BeamDesignResult
        With ``success = True`` and a ``ConcreteSection`` you can use
        directly; or ``success = False`` and notes explaining why the
        demand could not be satisfied.
    """
    best: BeamDesignResult | None = None
    best_As_total = float("inf")

    for bar in bar_options:
        A_bar = rebar_area(bar)
        # Rough required-As for positive moment (assume tension-controlled,
        # lever arm ~ 0.9 d, phi = 0.9):
        d_est = h - cover
        if d_est <= 0.0:
            continue
        As_req_pos = (
            abs(demand.M_u_positive) / (0.9 * material.fy * 0.9 * d_est)
            if abs(demand.M_u_positive) > 0.0 else 0.0
        )
        n_bot_min = max(2, math.ceil(As_req_pos / A_bar)
                          if A_bar > 0.0 else 2)

        # Top steel estimate (for hogging)
        if abs(demand.M_u_negative) > 0.0:
            As_req_neg = abs(demand.M_u_negative) / (
                0.9 * material.fy * 0.9 * d_est
            )
            n_top_min = max(2, math.ceil(As_req_neg / A_bar))
        else:
            n_top_min = 0

        for n_bot in range(n_bot_min, max_bars_per_layer + 1):
            for n_top in range(max(n_top_min, 0),
                                  max_bars_per_layer + 1):
                # Build candidate
                bottom = tuple(bar for _ in range(n_bot))
                top = tuple(bar for _ in range(n_top)) if n_top > 0 else ()
                rebar = RebarLayout(
                    bottom_bars=bottom, bottom_cover=cover,
                    top_bars=top, top_cover=cover,
                    stirrup_designation=stirrup_designation,
                    stirrup_legs=stirrup_legs,
                    stirrup_spacing=fallback_stirrup_spacing,
                )
                sec_init = ConcreteSection(
                    b=b, h=h, material=material, rebar=rebar,
                )

                # Positive-moment check
                fc_pos = beam_flexural_strength(sec_init)
                if fc_pos.phi_M_n < abs(demand.M_u_positive) - 1.0e-6:
                    # Not enough -> bump n_top first (won't help pos
                    # significantly), then n_bot at next iteration
                    if n_top >= n_top_min and abs(demand.M_u_negative) == 0.0:
                        break        # n_top doesn't affect pos meaningfully
                    continue
                # Negative-moment check: flip the section so the
                # original top becomes "tension"
                if abs(demand.M_u_negative) > 0.0:
                    rebar_flipped = RebarLayout(
                        bottom_bars=top, bottom_cover=cover,
                        top_bars=bottom, top_cover=cover,
                        stirrup_designation=stirrup_designation,
                        stirrup_legs=stirrup_legs,
                        stirrup_spacing=fallback_stirrup_spacing,
                    )
                    sec_flipped = ConcreteSection(
                        b=b, h=h, material=material, rebar=rebar_flipped,
                    )
                    fc_neg = beam_flexural_strength(sec_flipped)
                    if fc_neg.phi_M_n < abs(demand.M_u_negative) - 1.0e-6:
                        continue
                else:
                    fc_neg = None

                # Design stirrup spacing for V_u
                shear_design = design_stirrup_spacing(sec_init, demand.V_u)
                s_final = shear_design.s_recommended
                if not math.isfinite(s_final) or s_final <= 0.0:
                    s_final = fallback_stirrup_spacing
                # Rebuild section with the chosen stirrup spacing
                rebar_final = RebarLayout(
                    bottom_bars=bottom, bottom_cover=cover,
                    top_bars=top, top_cover=cover,
                    stirrup_designation=stirrup_designation,
                    stirrup_legs=stirrup_legs,
                    stirrup_spacing=s_final,
                )
                sec_final = ConcreteSection(
                    b=b, h=h, material=material, rebar=rebar_final,
                )
                shear_check = beam_shear_strength(sec_final, V_u=demand.V_u)
                if shear_check.phi_V_n < demand.V_u - 1.0e-6:
                    continue

                # Accept candidate; track total As
                As_total = (sec_final.rebar.As_bottom
                            + sec_final.rebar.As_top)
                if As_total < best_As_total:
                    best_As_total = As_total
                    notes_list = []
                    if abs(demand.M_u_negative) > 0.0:
                        notes_list.append(
                            f"hogging design with top steel "
                            f"({n_top} × {bar})"
                        )
                    best = BeamDesignResult(
                        section=sec_final,
                        flexure_positive=fc_pos,
                        flexure_negative=fc_neg,
                        shear=shear_check,
                        success=True,
                        notes="; ".join(notes_list) or "OK",
                    )
                # Once we found a valid layout at this (bar, n_bot, n_top),
                # we don't need to check larger n_top for the same n_bot.
                break

    if best is None:
        return BeamDesignResult(
            section=None, flexure_positive=None,
            flexure_negative=None, shear=None,
            success=False,
            notes=(
                "could not satisfy demand within max_bars_per_layer = "
                f"{max_bars_per_layer} with bar options "
                f"{list(bar_options)}; try doubly-reinforced or larger "
                "section."
            ),
        )
    return best


# ============================================================ column designer

def design_column(
    b: float,
    h: float,
    material: ConcreteMaterial,
    demand: ColumnDesignDemand,
    *,
    cover: float = 0.060,
    bar_options: tuple = DEFAULT_COLUMN_BAR_OPTIONS,
    spiral: bool = False,
    max_bars_per_face: int = 6,
    tie_designation: str = "#3",
    tie_spacing: float = 0.150,
) -> ColumnDesignResult:
    """Design a rectangular tied (or spirally reinforced) RC column:
    symmetric longitudinal rebar.

    Strategy
    --------
    Iterate over (bar size, n_per_face) with symmetric top+bottom
    layout. At each candidate, build the section, compute the
    P-M interaction surface, and evaluate ``DCR(P_u, M_u)``. Pick
    the lightest layout (smallest ``ρ = A_st / A_g``) that satisfies
    ``DCR <= 1`` and the ACI 10.6.1.1 ratio limits ``0.01 <= ρ <= 0.08``.

    Returns
    -------
    ColumnDesignResult
    """
    A_g = b * h
    best: ColumnDesignResult | None = None
    best_rho = float("inf")

    for bar in bar_options:
        A_bar = rebar_area(bar)
        for n_per_face in range(2, max_bars_per_face + 1):
            n_total = 2 * n_per_face      # top + bottom (symmetric)
            A_st = n_total * A_bar
            rho = A_st / A_g
            if rho < 0.01:
                continue                  # below ACI 10.6.1.1 minimum
            if rho > 0.08:
                break                     # exceed ACI 10.6.1.1 maximum
            rebar = RebarLayout(
                top_bars=tuple(bar for _ in range(n_per_face)),
                top_cover=cover,
                bottom_bars=tuple(bar for _ in range(n_per_face)),
                bottom_cover=cover,
                stirrup_designation=tie_designation,
                stirrup_spacing=tie_spacing,
            )
            sec = ConcreteSection(
                b=b, h=h, material=material, rebar=rebar,
            )
            surface = column_interaction_surface(sec, spiral=spiral)
            dcr = surface.dcr(demand.P_u, abs(demand.M_u))
            if dcr <= 1.0 - 1.0e-6:
                if rho < best_rho:
                    best_rho = rho
                    best = ColumnDesignResult(
                        section=sec,
                        interaction_surface=surface,
                        dcr=dcr,
                        rho=rho,
                        success=True,
                        notes=(
                            f"ρ = {rho * 100:.2f}% with {n_total} × {bar} "
                            f"symmetric layout"
                        ),
                    )
                break       # smaller n_per_face for next bar size

    if best is None:
        return ColumnDesignResult(
            section=None, interaction_surface=None,
            dcr=float("inf"), rho=0.0,
            success=False,
            notes=(
                "could not satisfy demand within ρ <= 0.08 (ACI 10.6.1.1) "
                f"with bar options {list(bar_options)}; enlarge section "
                "or change materials."
            ),
        )
    return best


# ============================================================ unified driver

class RcMemberDesigner:
    """Facade over :func:`design_beam` and :func:`design_column`.

    Provides a single entry point so calling code doesn't need to
    distinguish beams from columns at the API level.

    Examples
    --------
    >>> mat = ConcreteMaterial(fc_prime=28e6, fy=420e6)
    >>> beam = RcMemberDesigner.design_beam(
    ...     b=0.30, h=0.55, material=mat,
    ...     demand=BeamDesignDemand(M_u_positive=200e3, V_u=120e3),
    ... )
    >>> beam.success
    True
    """

    design_beam = staticmethod(design_beam)
    design_column = staticmethod(design_column)
