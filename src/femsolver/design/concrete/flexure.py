"""Rectangular concrete beam flexural strength per ACI 318-19 Ch. 22.2.

Implements the Whitney rectangular stress-block analysis:

* Uniform concrete stress ``0.85 f_c'`` over compression depth ``a = β_1 c``.
* Tension steel assumed to yield (``f_s = f_y``); the algorithm checks
  that ε_s ≥ ε_ty and warns if not.
* Strain-compatibility gives tension-steel strain ``ε_t = ε_cu (d - c) / c``
  used to set the φ factor per Table 21.2.2.

Singly-reinforced
-----------------
Force equilibrium with ``A_s f_y = 0.85 f_c' b a`` gives:

    a = A_s f_y / (0.85 f_c' b)
    c = a / β_1
    M_n = A_s f_y (d - a / 2)

Doubly-reinforced
-----------------
With compression steel ``A_s'`` at depth ``d'``:

1. **First** try assuming compression steel yields (``f_s' = f_y``).
   Then equilibrium gives:

       a = (A_s - A_s') f_y / (0.85 f_c' b),   c = a / β_1
       M_n = (A_s - A_s') f_y (d - a/2) + A_s' f_y (d - d')

2. **Verify** that ``ε_s' = ε_cu (c - d') / c ≥ ε_ty``. If not,
   compression steel does **not** yield and we solve the quadratic
   in ``c``:

       0.85 f_c' b β_1 c² + (A_s' E_s ε_cu - A_s f_y) c
                          - A_s' E_s ε_cu d' = 0

   then recompute M_n with the actual ``f_s' = E_s ε_cu (c - d') / c``.

Sign conventions
----------------
* Positive ``M_n`` corresponds to tension at the bottom of the
  section (sagging) -- the standard convention. The section is
  treated as symmetric; for hogging (negative) moment, swap top and
  bottom bars before calling.
* Units: all inputs in SI (Pa, m). Result ``M_n`` in N·m.
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


# ============================================================ result

@dataclass
class FlexuralCheck:
    """Result of a beam flexural-strength evaluation.

    Attributes
    ----------
    M_n : float
        Nominal moment capacity (N·m).
    phi : float
        Strength-reduction factor per ACI Table 21.2.2.
    phi_M_n : float
        Design moment capacity ``φ · M_n`` (N·m).
    c : float
        Neutral-axis depth from extreme compression fiber (m).
    a : float
        Whitney stress-block depth = ``β_1 · c`` (m).
    epsilon_t : float
        Extreme-tension-steel strain at nominal flexural strength.
    section_type : str
        ``"tension-controlled"`` (ε_t ≥ 0.005), ``"transition"``
        (ε_ty < ε_t < 0.005), or ``"compression-controlled"``
        (ε_t ≤ ε_ty).
    compression_steel_yields : bool
        For doubly-reinforced sections: whether the compression steel
        has reached yield at nominal strength. False for singly-
        reinforced sections (no compression steel) by convention.
    tension_steel_yields : bool
        Whether the tension steel has reached yield. Should be True
        for any conventional design; False indicates a severely
        over-reinforced section.
    notes : str
        Warnings about A_s_min or A_s_max violations, etc.
    """

    M_n: float
    phi: float
    phi_M_n: float
    c: float
    a: float
    epsilon_t: float
    section_type: str
    compression_steel_yields: bool
    tension_steel_yields: bool
    notes: str = ""


# ============================================================ doubly-reinforced solver

def _solve_doubly_reinforced_c(
    *, b: float, d: float, d_prime: float,
    As: float, As_prime: float,
    fc_prime: float, fy: float, beta_1: float,
) -> float:
    """Solve the quadratic in ``c`` when compression steel doesn't yield.

    From equilibrium:
        0.85 f_c' b β_1 c² + (A_s' E_s ε_cu - A_s f_y) c
                            - A_s' E_s ε_cu d' = 0

    Returns the positive root.
    """
    A_q = 0.85 * fc_prime * b * beta_1
    B_q = As_prime * E_STEEL * EPSILON_CU - As * fy
    C_q = -As_prime * E_STEEL * EPSILON_CU * d_prime
    disc = B_q * B_q - 4.0 * A_q * C_q
    if disc < 0.0:
        raise RuntimeError(
            f"doubly-reinforced solve: negative discriminant "
            f"(A={A_q:g}, B={B_q:g}, C={C_q:g}); the section may have "
            f"compatibility issues -- check inputs."
        )
    sd = math.sqrt(disc)
    c1 = (-B_q + sd) / (2.0 * A_q)
    c2 = (-B_q - sd) / (2.0 * A_q)
    # Choose the positive root (there's exactly one for physical sections)
    if c1 > 0.0 and c2 > 0.0:
        # Both positive: pick the one with the larger compression-steel
        # strain (the more physically reasonable). For typical sections
        # only one is positive.
        return min(c1, c2)
    if c1 > 0.0:
        return c1
    if c2 > 0.0:
        return c2
    raise RuntimeError(
        "doubly-reinforced solve: no positive root for neutral-axis "
        "depth -- the section may be severely over-reinforced."
    )


# ============================================================ main entry

def beam_flexural_strength(section: ConcreteSection) -> FlexuralCheck:
    """Compute nominal and design flexural strength of a rectangular
    concrete beam per ACI 318-19 Ch. 22.2.

    Parameters
    ----------
    section : ConcreteSection
        Rectangular section with rebar layout. Bottom bars are
        treated as tension steel (sagging-moment case).

    Returns
    -------
    FlexuralCheck

    Notes
    -----
    * For hogging moments, build the section with bars swapped (top
      bars become "tension") -- a higher-level wrapper can handle
      both signs automatically.
    * The compression-steel-yields check is checked exactly; if it
      doesn't yield, the routine falls back to the quadratic solve
      with the actual elastic compression-steel stress.
    """
    fc = section.material.fc_prime
    fy = section.material.fy
    beta_1 = section.material.beta_1
    eps_ty = section.material.epsilon_ty
    b = section.b
    d = section.d
    d_prime = section.d_prime
    As = section.rebar.As_bottom
    As_prime = section.rebar.As_top

    # Degenerate: no tension steel -> no flexural capacity
    if As <= 0.0:
        return FlexuralCheck(
            M_n=0.0, phi=0.90, phi_M_n=0.0,
            c=0.0, a=0.0, epsilon_t=float("inf"),
            section_type="no-tension-steel",
            compression_steel_yields=False,
            tension_steel_yields=False,
            notes="no tension steel; flexural capacity = 0",
        )

    if As_prime <= 0.0:
        # ---- Singly-reinforced ----
        a = As * fy / (0.85 * fc * b)
        c = a / beta_1
        M_n = As * fy * (d - 0.5 * a)
        compression_steel_yields = False
    else:
        # ---- Doubly-reinforced ----
        # Assume compression steel yields, compute c, then verify.
        a_assumed = (As - As_prime) * fy / (0.85 * fc * b)
        if a_assumed <= 0.0:
            # As' >= As: rare case; section behaves like compression
            # member -- not a flexural problem. Fall back to elastic c.
            c_assumed = 0.0
        else:
            c_assumed = a_assumed / beta_1
        if c_assumed > d_prime:
            # Check compression-steel strain
            eps_s_prime = EPSILON_CU * (c_assumed - d_prime) / c_assumed
            if eps_s_prime >= eps_ty:
                # Compression steel yields -- the assumption holds
                a = a_assumed
                c = c_assumed
                M_n = (
                    (As - As_prime) * fy * (d - 0.5 * a)
                    + As_prime * fy * (d - d_prime)
                )
                compression_steel_yields = True
            else:
                # Compression steel doesn't yield -- solve quadratic
                c = _solve_doubly_reinforced_c(
                    b=b, d=d, d_prime=d_prime,
                    As=As, As_prime=As_prime,
                    fc_prime=fc, fy=fy, beta_1=beta_1,
                )
                a = beta_1 * c
                eps_s_prime = EPSILON_CU * (c - d_prime) / c
                f_s_prime = E_STEEL * eps_s_prime
                M_n = (
                    0.85 * fc * b * a * (d - 0.5 * a)
                    + As_prime * f_s_prime * (d - d_prime)
                )
                compression_steel_yields = False
        else:
            # c is so shallow that compression steel is actually in
            # tension (below the neutral axis): treat as singly-
            # reinforced with the top bars ignored (conservative).
            a = As * fy / (0.85 * fc * b)
            c = a / beta_1
            M_n = As * fy * (d - 0.5 * a)
            compression_steel_yields = False

    # Tension-steel strain at ultimate, for phi
    if c > 0.0:
        eps_t = EPSILON_CU * (d - c) / c
    else:
        eps_t = float("inf")
    tension_steel_yields = (eps_t >= eps_ty)

    phi = phi_for_strain(eps_t, epsilon_ty=eps_ty)

    if eps_t >= 0.005:
        section_type = "tension-controlled"
    elif eps_t <= eps_ty:
        section_type = "compression-controlled"
    else:
        section_type = "transition"

    # Validation warnings
    notes: list[str] = []
    As_min = section.As_min_flexure()
    if As < As_min:
        notes.append(
            f"As ({As * 1e6:.0f} mm²) < As_min ({As_min * 1e6:.0f} mm²) "
            f"per ACI 9.6.1.2"
        )
    if section_type == "transition":
        notes.append(
            f"section is in transition zone (ε_t = {eps_t:.4f}); "
            f"consider adding compression steel or reducing tension steel"
        )
    elif section_type == "compression-controlled" and As_prime <= 0.0:
        notes.append(
            f"section is over-reinforced (ε_t = {eps_t:.4f}); add "
            f"compression reinforcement or increase section depth"
        )
    if not tension_steel_yields:
        notes.append(
            f"tension steel has NOT yielded (ε_s = {eps_t:.4f} < "
            f"ε_ty = {eps_ty:.4f}); the As·fy assumption is violated"
        )

    return FlexuralCheck(
        M_n=M_n,
        phi=phi,
        phi_M_n=phi * M_n,
        c=c,
        a=a,
        epsilon_t=eps_t,
        section_type=section_type,
        compression_steel_yields=compression_steel_yields,
        tension_steel_yields=tension_steel_yields,
        notes="; ".join(notes),
    )
