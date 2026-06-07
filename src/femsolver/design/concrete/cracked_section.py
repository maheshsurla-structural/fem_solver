"""Cracked transformed section properties for serviceability (Phase II.14).

Provides cracked-elastic section properties for deflection and crack-
width calculations under SLS loading:

* :func:`cracked_section_properties` -- given (P, M_z, M_y), solve for
  the elastic strain distribution treating concrete as cracked (zero
  tension, linear in compression) and steel/tendons as linear elastic.
  Returns the cracked moment of inertia ``I_cr``, the neutral-axis
  depth ``c``, and extreme-fibre concrete / steel stresses.

* :func:`branson_I_e` -- ACI 318-19 §24.2.3.5 effective moment of
  inertia for deflection: ``I_e = I_cr + (I_g - I_cr)(M_cr/M_a)^3``.

* :func:`ec2_mean_curvature` -- EC2 §7.4.3 tension-stiffening mean
  curvature interpolation between cracked and uncracked states.

Algorithm
---------
Reuses the 3-DOF Newton driver from :func:`stress_field` (II.15) but
swaps in cracked-elastic constitutives:

* :class:`CrackedElasticConcrete` -- linear in compression, zero in
  tension (clean rupture)
* :class:`UniaxialElastic` -- for rebar and (with the existing
  :class:`PrestressedUniaxial` wrapper) for tendons

For typical SLS loadings the result is a converged elastic state with
the cracked moment of inertia given by ``I_cr = M / (E_c * kappa)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from femsolver.design.concrete.section import ConcreteMaterial
from femsolver.design.concrete.stress_field import (
    SectionStressField,
    stress_field,
)
from femsolver.materials.uniaxial.base import UniaxialMaterial
from femsolver.materials.uniaxial.elastic import UniaxialElastic
from femsolver.sections.section import Section


# ============================================================ constitutive

class CrackedElasticConcrete(UniaxialMaterial):
    """Linear elastic in compression (sigma = E·eps for eps <= 0),
    zero in tension (eps > 0).

    Used as the concrete constitutive for cracked-section SLS analysis
    (no tension-stiffening). For the uncracked elastic comparison,
    pass a regular :class:`UniaxialElastic`.
    """

    def __init__(self, E: float):
        if E <= 0:
            raise ValueError(f"E must be positive, got {E}")
        self.E = float(E)

    def get_response(self, eps: float) -> tuple[float, float]:
        if eps <= 0.0:
            return self.E * eps, self.E
        return 0.0, 0.0

    def commit_state(self) -> None: return None
    def revert_state(self) -> None: return None
    def clone(self) -> "CrackedElasticConcrete":
        return CrackedElasticConcrete(self.E)


# ============================================================ result

@dataclass
class CrackedSectionProperties:
    """Cracked-section elastic properties under (P, M_z, M_y)."""
    P: float
    M_z: float
    M_y: float

    # Strain distribution
    eps_0: float
    kappa_z: float
    kappa_y: float

    # Cracked I about each axis (computed from M / (E_c * kappa) when
    # applicable; None if curvature is too small to be meaningful)
    I_cr_z: Optional[float]
    I_cr_y: Optional[float]

    # Materials
    E_c: float
    E_s: float

    # Geometry
    neutral_axis_depth_from_top: float    # c, positive (m)
    extreme_compression_strain: float     # most-negative (= -|eps|)
    extreme_compression_stress: float     # Pa
    max_steel_tensile_strain: float
    max_steel_tensile_stress: float

    # Full per-fibre field for further queries / visualization
    field: SectionStressField


# ============================================================ driver

def cracked_section_properties(
    section: Section,
    *,
    P: float = 0.0,
    M_z: float = 0.0,
    M_y: float = 0.0,
    E_c: Optional[float] = None,
    E_s: float = 200e9,
    n_z: int = 16,
    n_y: int = 40,
    P_convention: str = "compression",
) -> CrackedSectionProperties:
    """Solve for the cracked-elastic section state under (P, M_z, M_y).

    Parameters
    ----------
    section : Section
        Unified Section with rebar (and optionally tendons).
    P, M_z, M_y : float
        Axial load (compression-positive default) and moments.
    E_c : float, optional
        Concrete elastic modulus. If ``None``, read from the section's
        primary material (must be a :class:`ConcreteMaterial` or
        provide ``Ec``). Otherwise raises.
    E_s : float, default 200 GPa
        Steel elastic modulus.
    n_z, n_y : int
        Polygon discretization grid resolution for accurate cracked
        moment-of-inertia integration.
    """
    if E_c is None:
        prim = section.primary_material
        if prim is not None and hasattr(prim, "Ec"):
            E_c = float(prim.Ec)
        else:
            raise ValueError(
                "no E_c available; pass E_c explicitly or attach a "
                "ConcreteMaterial with Ec to the section"
            )

    concrete = CrackedElasticConcrete(E_c)
    steel = UniaxialElastic(E_s)

    sf = stress_field(
        section,
        P=P, M_z=M_z, M_y=M_y,
        concrete_uniaxial=concrete,
        steel_uniaxial=steel,
        n_z=n_z, n_y=n_y,
        P_convention=P_convention,
    )

    # I_cr from M / (E_c * kappa) -- valid when there's curvature
    I_cr_z = None
    if abs(sf.kappa_z) > 1e-10 and abs(M_z) > 1.0:
        I_cr_z = float(M_z / (E_c * sf.kappa_z))
    I_cr_y = None
    if abs(sf.kappa_y) > 1e-10 and abs(M_y) > 1.0:
        I_cr_y = float(M_y / (E_c * sf.kappa_y))

    # Neutral-axis depth from top (only meaningful for primary bending
    # about z-axis; for biaxial, use the curvature direction)
    minz, miny, maxz, maxy = section.geometry.polygon.bounds
    y_top = maxy
    if abs(sf.kappa_z) > 1e-10:
        # eps = eps_0 - y * kappa_z = 0 at y_NA = eps_0 / kappa_z
        y_NA = sf.eps_0 / sf.kappa_z
        c = float(y_top - y_NA)
    else:
        c = float(maxy - miny)

    # Extreme fibre values from the stress field
    eps_top = sf.extreme_compression_strain()
    sigma_top = E_c * eps_top   # linear elastic; matches the field's sigma

    rebar_states = [f for f in sf.fibers if f.kind == "rebar"]
    if rebar_states:
        max_tension = max(rebar_states, key=lambda f: f.eps)
        eps_s = max_tension.eps
        sigma_s = max_tension.sigma
    else:
        eps_s = 0.0
        sigma_s = 0.0

    return CrackedSectionProperties(
        P=float(P), M_z=float(M_z), M_y=float(M_y),
        eps_0=float(sf.eps_0),
        kappa_z=float(sf.kappa_z),
        kappa_y=float(sf.kappa_y),
        I_cr_z=I_cr_z, I_cr_y=I_cr_y,
        E_c=float(E_c), E_s=float(E_s),
        neutral_axis_depth_from_top=c,
        extreme_compression_strain=float(eps_top),
        extreme_compression_stress=float(sigma_top),
        max_steel_tensile_strain=float(eps_s),
        max_steel_tensile_stress=float(sigma_s),
        field=sf,
    )


# ============================================================ Branson

def branson_I_e(I_g: float, I_cr: float, M_cr: float, M_a: float) -> float:
    """ACI 318-19 §24.2.3.5 effective moment of inertia for deflection.

    ``I_e = I_cr + (I_g - I_cr) * (M_cr / M_a)^3``

    Valid for ``M_a >= M_cr``. Below cracking returns ``I_g`` (uncracked).
    Capped at ``I_g`` above.
    """
    if I_g <= 0 or I_cr <= 0 or M_cr <= 0:
        raise ValueError(
            "I_g, I_cr, M_cr must be positive"
        )
    if M_a <= M_cr:
        return float(I_g)
    ratio = M_cr / M_a
    I_e = I_cr + (I_g - I_cr) * (ratio ** 3)
    return float(min(I_e, I_g))


def ec2_mean_curvature(
    kappa_uncracked: float, kappa_cracked: float,
    *, M_cr: float, M_a: float, beta: float = 1.0,
) -> float:
    """EC2 §7.4.3 (3) tension-stiffening mean curvature.

    ``kappa_mean = zeta * kappa_cracked + (1 - zeta) * kappa_uncracked``

    where ``zeta = 1 - beta * (M_cr / M_a)^2`` is the distribution
    factor (0 when uncracked, ~1 when heavily cracked).

    Parameters
    ----------
    kappa_uncracked, kappa_cracked : float
        Curvatures computed for the uncracked and fully-cracked states
        under the same loading ``M_a``.
    M_cr, M_a : float
        Cracking moment and applied moment (same sign convention).
    beta : float, default 1.0
        Bond + load-duration factor (EC2 7.19: 1.0 for short-term,
        0.5 for sustained / repeated loading).
    """
    if M_a <= 0:
        return float(kappa_uncracked)
    if M_a <= M_cr:
        return float(kappa_uncracked)
    zeta = 1.0 - beta * (M_cr / M_a) ** 2
    zeta = max(0.0, min(1.0, zeta))
    return float(zeta * kappa_cracked + (1.0 - zeta) * kappa_uncracked)
