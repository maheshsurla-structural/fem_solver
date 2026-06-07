"""Wall shear-flexibility utilities.

For tall walls (``H / L_w >= 2``), bending dominates and the
flexure-only fiber section (:mod:`femsolver.sections.response.wall`) captures
the response. For **squat** walls (``H / L_w < 1.5``, with ``1.5 < H / L_w < 2``
in the transition zone), shear deformation contributes significantly to
the lateral deflection and the bending-only model over-predicts the
stiffness.

This module provides two pragmatic patches for that case:

1. **Closed-form lateral-stiffness check** with both flexure and shear
   contributions, so the user can quickly see how much shear matters
   for their wall.
2. **Cracked-section factors** per ACI 318-19 Table 6.6.3.1.1 and
   ASCE 41-17 Table 10-5 (for nonlinear seismic analysis), reducing
   ``I_eff = alpha_I I_g`` and ``A_v,eff = alpha_v A_g`` to model the
   stiffness loss from cracking.
3. **Shear-spring helper** -- the stiffness ``K_v = G A_v / H`` to be
   installed as a horizontal :class:`UniaxialElastic` between the wall
   base node and a fixed support node via :class:`ZeroLengthElement`.
   The bending-only fiber-section element is then attached above this
   shear spring.

References
----------
* ACI 318-19 Table 6.6.3.1.1 (cracked-section moments of inertia for
  linear analysis).
* ASCE 41-17 Table 10-5 (cracked-section factors for nonlinear seismic
  analysis of walls).
* PEER/ATC-72-1 (2010) "Modeling and Acceptance Criteria for Seismic
  Design and Analysis of Tall Buildings."
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ============================================================ shear stiffness

def wall_shear_area(L_w: float, t_w: float,
                    *, shear_shape_factor: float = 5.0 / 6.0) -> float:
    """Effective shear area ``A_v = k_s * L_w * t_w``.

    The shape factor ``k_s = 5/6`` is the standard Timoshenko value
    for a rectangular cross-section. For T- or I-shaped walls, the
    user supplies the actual ``A_v`` directly.
    """
    if L_w <= 0.0 or t_w <= 0.0:
        raise ValueError("L_w and t_w must be positive")
    return float(shear_shape_factor * L_w * t_w)


def wall_lateral_stiffness(
    *,
    L_w: float, t_w: float, H: float,
    E: float, G: float,
    I_eff_factor: float = 1.0,
    A_v_eff_factor: float = 1.0,
) -> dict:
    """Closed-form cantilever-wall lateral stiffness.

    Decomposes the tip-deflection compliance into a flexure and a
    shear part::

        delta_tip = V · ( H^3 / (3 E I_eff)  +  H / (G A_v,eff) )
        k_lat    = 1 / (delta_tip / V)
        alpha_shear = shear-only compliance / total

    Parameters
    ----------
    L_w : float
        Wall length in plan (m).
    t_w : float
        Wall thickness (m).
    H : float
        Wall height (m).
    E : float
        Concrete Young's modulus (Pa).
    G : float
        Concrete shear modulus (Pa). Use ``E / (2 (1 + nu))``.
    I_eff_factor : float, default 1.0
        Multiplier on ``I_g`` (use ``0.35`` for cracked-section per
        ACI 318-19 §6.6.3.1.1 walls).
    A_v_eff_factor : float, default 1.0
        Multiplier on ``A_v`` (use ``0.5`` for cracked walls).

    Returns
    -------
    dict
        ``{"k_lat": ..., "k_flex": ..., "k_shear": ...,
        "alpha_shear": ..., "alpha_flex": ...,
        "I_eff": ..., "A_v_eff": ...}``
    """
    if H <= 0.0:
        raise ValueError(f"H must be positive, got {H}")
    if E <= 0.0 or G <= 0.0:
        raise ValueError("E and G must be positive")
    I_g = t_w * L_w ** 3 / 12.0
    A_v = wall_shear_area(L_w, t_w)
    I_eff = I_eff_factor * I_g
    A_v_eff = A_v_eff_factor * A_v
    c_flex = H ** 3 / (3.0 * E * I_eff)
    c_shear = H / (G * A_v_eff)
    c_total = c_flex + c_shear
    k_lat = 1.0 / c_total
    k_flex = 1.0 / c_flex
    k_shear = 1.0 / c_shear
    return {
        "k_lat": float(k_lat),
        "k_flex": float(k_flex),
        "k_shear": float(k_shear),
        "alpha_flex": float(c_flex / c_total),
        "alpha_shear": float(c_shear / c_total),
        "I_eff": float(I_eff),
        "A_v_eff": float(A_v_eff),
    }


# ============================================================ cracked-section factors

@dataclass
class CrackedSectionFactors:
    """Cracked-section stiffness reduction factors.

    Attributes
    ----------
    I_eff_over_I_g : float
        Effective moment of inertia divided by gross.
    A_v_eff_over_A_g : float
        Effective shear area divided by gross.
    code : str
        Origin code reference.
    """

    I_eff_over_I_g: float
    A_v_eff_over_A_g: float
    code: str


def aci318_cracked_factors(member_type: str) -> CrackedSectionFactors:
    """ACI 318-19 Table 6.6.3.1.1 cracked-section factors for
    linear-elastic analysis.

    Parameters
    ----------
    member_type : str
        ``"wall_cracked"`` (0.35), ``"wall_uncracked"`` (0.70),
        ``"beam"`` (0.35), ``"column"`` (0.70), ``"slab"`` (0.25).
    """
    table = {
        "wall_cracked":    (0.35, 1.0),    # walls in significant tension
        "wall_uncracked":  (0.70, 1.0),    # walls with low tension
        "beam":            (0.35, 1.0),
        "column":          (0.70, 1.0),
        "slab":            (0.25, 1.0),
    }
    if member_type not in table:
        raise ValueError(
            f"unknown member_type '{member_type}', must be one of: "
            f"{list(table.keys())}"
        )
    I, A = table[member_type]
    return CrackedSectionFactors(
        I_eff_over_I_g=I,
        A_v_eff_over_A_g=A,
        code="ACI 318-19 Table 6.6.3.1.1",
    )


def asce41_wall_factors(
    *,
    axial_load_ratio: float = 0.0,
    flexure_or_shear: str = "flexure",
) -> CrackedSectionFactors:
    """ASCE 41-17 Table 10-5 wall stiffness factors for nonlinear
    seismic analysis.

    Walls controlled by flexure use ``I_eff = 0.5 I_g`` (cracked) and
    ``A_v,eff = 0.4 A_g``; walls controlled by shear use ``I_eff = 0.8 I_g``
    and ``A_v,eff = 1.0 A_g``.

    Parameters
    ----------
    axial_load_ratio : float, default 0.0
        ``P / (A_g f_c')`` -- currently informational only.
    flexure_or_shear : str
        ``"flexure"`` or ``"shear"`` controlled behaviour.
    """
    if flexure_or_shear == "flexure":
        I, A = 0.5, 0.4
    elif flexure_or_shear == "shear":
        I, A = 0.8, 1.0
    else:
        raise ValueError(
            f"flexure_or_shear must be 'flexure' or 'shear', "
            f"got '{flexure_or_shear}'"
        )
    return CrackedSectionFactors(
        I_eff_over_I_g=I,
        A_v_eff_over_A_g=A,
        code="ASCE 41-17 Table 10-5",
    )


# ============================================================ shear-spring stiffness

def wall_base_shear_spring_stiffness(
    *,
    L_w: float, t_w: float, H: float,
    G: float,
    A_v_eff_factor: float = 1.0,
) -> float:
    """Lumped shear-spring stiffness ``K_v = G A_v_eff / H`` (N/m).

    Install this as the stiffness of a horizontal
    :class:`UniaxialElastic` between the wall base and a fixed support
    via :class:`ZeroLengthElement` (DOF 0). The fiber-section wall
    element then attaches above this spring, capturing axial-flexural
    response while the spring captures shear flexibility.
    """
    if H <= 0.0 or G <= 0.0:
        raise ValueError("H and G must be positive")
    A_v_eff = A_v_eff_factor * wall_shear_area(L_w, t_w)
    return float(G * A_v_eff / H)
