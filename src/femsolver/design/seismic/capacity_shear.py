"""Capacity-design beam shear per ACI 318-19 §18.6.5.

For special moment frames, the design shear at the beam ends must be
based on the **probable moment** ``M_pr`` -- the moment the beam can
actually develop after over-strength and strain-hardening of the
flexural reinforcement -- not just the factored demand from
analysis. Otherwise the beam would fail in shear (non-ductile) before
its flexural hinge forms (ductile).

Per ACI 18.6.5.1, the design shear ``V_e`` accounts for:

1. Sway-induced shear from probable beam-end moments:

       V_e_sway = (M_pr_left + M_pr_right) / L_n

2. Plus the factored gravity shear ``V_g`` from the tributary
   distributed load on the beam clear span ``L_n``:

       V_g = w_u · L_n / 2

The total capacity-design shear at each end is

    V_e = V_e_sway + V_g   (for end where sway shear adds to gravity)

The "probable moment" M_pr per ACI 18.6.5.1 commentary is computed
using a tensile-stress multiplier of **1.25 f_y** in place of f_y:

    M_pr = M_n (computed with f_y replaced by 1.25 f_y)

This effectively represents the maximum moment that the section can
develop accounting for strain hardening, with no strength-reduction
factor (φ = 1.0).

When the analysis-derived factored shear V_u exceeds V_e, the
analysis governs. When V_e exceeds V_u (the common case for SMF),
V_e governs and the beam stirrups must be designed for V_e per ACI
9.6.3 + 22.5 (Phase 29.3 routines).
"""
from __future__ import annotations

from dataclasses import dataclass

from femsolver.design.concrete.flexure import beam_flexural_strength
from femsolver.design.concrete.section import (
    ConcreteMaterial,
    ConcreteSection,
)


@dataclass
class CapacityShearCheck:
    """Result of an ACI 18.6.5 capacity-design beam-shear evaluation.

    Attributes
    ----------
    M_pr_left, M_pr_right : float
        Probable moments at the left and right ends (N·m, magnitudes).
    L_n : float
        Beam clear span (m).
    w_u : float
        Factored gravity UDL on the span (N/m).
    V_g : float
        Gravity-induced shear ``w_u · L_n / 2`` (N).
    V_e_sway : float
        Sway-induced shear ``(M_pr_left + M_pr_right) / L_n`` (N).
    V_e : float
        Total capacity-design shear ``V_e_sway + V_g`` (N).
    V_u_analysis : float
        Factored shear from analysis (optional; used for comparison).
    V_design : float
        Final design shear = max(V_e, V_u_analysis) -- the value
        the stirrup design (Phase 29.3) must resist.
    notes : str
    """

    M_pr_left: float
    M_pr_right: float
    L_n: float
    w_u: float
    V_g: float
    V_e_sway: float
    V_e: float
    V_u_analysis: float
    V_design: float
    notes: str = ""


def probable_moment(section: ConcreteSection) -> float:
    """Compute the **probable** flexural moment ``M_pr`` of a section,
    using ``1.25 f_y`` in place of ``f_y`` and no φ reduction.

    Returns the nominal moment (N·m) of a modified section in which
    the rebar yield stress is bumped up by 25%. This represents the
    most likely overstrength the section can deliver under cyclic
    inelastic action (ACI 18.6.5 commentary).
    """
    boosted_mat = ConcreteMaterial(
        fc_prime=section.material.fc_prime,
        fy=1.25 * section.material.fy,
        Ec=section.material.Ec,
    )
    boosted_section = ConcreteSection(
        b=section.b, h=section.h,
        material=boosted_mat,
        rebar=section.rebar,
    )
    fc = beam_flexural_strength(boosted_section)
    return fc.M_n


def capacity_design_shear(
    section_left: ConcreteSection,
    section_right: ConcreteSection,
    *,
    L_n: float,
    w_u: float = 0.0,
    V_u_analysis: float = 0.0,
) -> CapacityShearCheck:
    """Compute the ACI 18.6.5 capacity-design beam shear.

    Parameters
    ----------
    section_left, section_right : ConcreteSection
        Beam sections **as designed** at the left and right ends
        (typically with hogging reinforcement on top for negative
        moment at the joint). The probable moment ``M_pr`` is the
        actual capacity the section can develop after strain
        hardening, computed with f_y → 1.25 f_y.
    L_n : float
        Beam clear span between column faces (m).
    w_u : float, default 0
        Factored gravity UDL on the span (N/m).
    V_u_analysis : float, default 0
        Factored shear from the analysis envelope (for comparison
        with V_e). The returned ``V_design`` is the larger.

    Returns
    -------
    CapacityShearCheck
        Includes ``V_e`` (capacity-design shear) and ``V_design`` (the
        governing value for stirrup design).
    """
    if L_n <= 0.0:
        raise ValueError(f"L_n must be positive, got {L_n}")

    M_pr_left = probable_moment(section_left)
    M_pr_right = probable_moment(section_right)
    V_e_sway = (abs(M_pr_left) + abs(M_pr_right)) / L_n
    V_g = abs(w_u) * L_n / 2.0
    V_e = V_e_sway + V_g
    V_design = max(V_e, abs(V_u_analysis))

    notes_list: list[str] = []
    if V_e > abs(V_u_analysis) + 1.0e-6:
        notes_list.append(
            f"capacity-design V_e ({V_e/1e3:.1f} kN) governs over "
            f"analysis V_u ({V_u_analysis/1e3:.1f} kN)"
        )

    return CapacityShearCheck(
        M_pr_left=M_pr_left,
        M_pr_right=M_pr_right,
        L_n=L_n,
        w_u=w_u,
        V_g=V_g,
        V_e_sway=V_e_sway,
        V_e=V_e,
        V_u_analysis=abs(V_u_analysis),
        V_design=V_design,
        notes="; ".join(notes_list),
    )
