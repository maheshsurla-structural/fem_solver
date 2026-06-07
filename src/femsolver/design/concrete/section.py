"""Concrete section + rebar layout dataclasses and ACI 318-19 constants.

This module is the data-layer foundation for ACI 318-19 design checks
(Phase 29.2-29.5). Geometry is described in **SI units**: lengths in
meters, stresses in pascals (Pa). Helper classes convert to/from
imperial (in, psi) where convenient.

ACI 318 references in comments cite the **2019 edition** unless
otherwise noted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ============================================================ material constants


# Standard imperial bar designations and their nominal diameters (in) +
# nominal cross-section areas (in^2). Source: ACI 318-19 Appendix A
# (Standard Rebar Sizes).
_US_REBAR_TABLE = {
    "#3": (0.375, 0.11),
    "#4": (0.500, 0.20),
    "#5": (0.625, 0.31),
    "#6": (0.750, 0.44),
    "#7": (0.875, 0.60),
    "#8": (1.000, 0.79),
    "#9": (1.128, 1.00),
    "#10": (1.270, 1.27),
    "#11": (1.410, 1.56),
    "#14": (1.693, 2.25),
    "#18": (2.257, 4.00),
}

_IN_TO_M = 0.0254
_IN2_TO_M2 = _IN_TO_M * _IN_TO_M


def rebar_diameter(designation: str) -> float:
    """Nominal bar diameter in **meters** for a US-customary bar
    designation (e.g., ``"#5"`` -> 0.0159 m).
    """
    if designation not in _US_REBAR_TABLE:
        raise ValueError(
            f"unknown rebar designation {designation!r}; valid: "
            f"{list(_US_REBAR_TABLE.keys())}"
        )
    db_in, _ = _US_REBAR_TABLE[designation]
    return db_in * _IN_TO_M


def rebar_area(designation: str) -> float:
    """Nominal bar area in **m²** for a US-customary bar designation."""
    if designation not in _US_REBAR_TABLE:
        raise ValueError(
            f"unknown rebar designation {designation!r}; valid: "
            f"{list(_US_REBAR_TABLE.keys())}"
        )
    _, A_in2 = _US_REBAR_TABLE[designation]
    return A_in2 * _IN2_TO_M2


def standard_rebar_designations() -> list[str]:
    """All available standard bar designations, smallest to largest."""
    return list(_US_REBAR_TABLE.keys())


# ============================================================ ACI 318 phi factors

@dataclass(frozen=True)
class PhiFactors:
    """Strength-reduction factors per ACI 318-19 Table 21.2.1.

    Used directly when the section is **tension-controlled**. The
    transition zone (``ε_t`` between ``ε_ty`` and ``0.005``) uses a
    linear interpolation -- see :func:`phi_for_strain`.
    """

    #: Tension-controlled flexure (or flexure + small axial), phi = 0.90.
    flexure_tension_controlled: float = 0.90

    #: Compression-controlled members (tied), phi = 0.65 (21.2.2).
    compression_tied: float = 0.65

    #: Compression-controlled with spiral reinforcement, phi = 0.75.
    compression_spiral: float = 0.75

    #: Shear and torsion, phi = 0.75 (Table 21.2.1).
    shear: float = 0.75

    #: Bearing on concrete, phi = 0.65 (22.8).
    bearing: float = 0.65


def phi_for_strain(epsilon_t: float, epsilon_ty: float = 0.002,
                    *, spiral: bool = False) -> float:
    """Strength-reduction factor ``φ`` for a given tension-steel
    strain ``ε_t`` per ACI 318-19 Table 21.2.2 (the transition zone
    between compression-controlled and tension-controlled sections).

    * ``ε_t <= ε_ty`` (yield strain of steel): compression-controlled
      -> ``φ = 0.65`` (tied) or ``0.75`` (spiral).
    * ``ε_t >= 0.005``: tension-controlled -> ``φ = 0.90``.
    * In between: linear interpolation per Table 21.2.2.

    For Grade 60 steel ``ε_ty ≈ 0.00207`` (= 60 ksi / 29000 ksi). For
    Grade 80 ``ε_ty ≈ 0.00276``.
    """
    phi_comp = 0.75 if spiral else 0.65
    phi_tens = 0.90
    eps_t_lim = 0.005
    if epsilon_t <= epsilon_ty:
        return phi_comp
    if epsilon_t >= eps_t_lim:
        return phi_tens
    # Linear interpolation (per 21.2.2)
    t = (epsilon_t - epsilon_ty) / (eps_t_lim - epsilon_ty)
    return phi_comp + t * (phi_tens - phi_comp)


def beta_1_aci(fc_prime_Pa: float) -> float:
    """Whitney stress-block depth factor ``β_1`` per ACI 318-19
    22.2.2.4.3.

    * ``β_1 = 0.85`` for ``f_c' <= 4000 psi (27.6 MPa)``
    * ``β_1 = 0.85 - 0.05 ((f_c' - 4000) / 1000)`` for 4000 < f_c' < 8000 psi
    * ``β_1 = 0.65`` for ``f_c' >= 8000 psi (55.2 MPa)``

    Input is in pascals; conversions handled internally.
    """
    fc_psi = fc_prime_Pa / 6894.757
    if fc_psi <= 4000.0:
        return 0.85
    if fc_psi >= 8000.0:
        return 0.65
    return 0.85 - 0.05 * (fc_psi - 4000.0) / 1000.0


#: Concrete ultimate compressive strain per ACI 318-19 22.2.2.1.
EPSILON_CU = 0.003

#: Modulus of elasticity of reinforcement (ACI 318-19 20.2.2.2),
#: 29000 ksi = 200000 MPa.
E_STEEL = 200000.0e6     # Pa


# ============================================================ ConcreteSection

@dataclass
class ConcreteMaterial:
    """Concrete material strength properties.

    Attributes
    ----------
    fc_prime : float
        Specified compressive strength ``f_c'`` (Pa).
    fy : float
        Specified yield strength of reinforcement (Pa).
    Ec : float, optional
        Modulus of elasticity. Defaults to ACI 318-19 19.2.2.1
        normal-weight concrete: ``Ec = 4700 * sqrt(f_c' in MPa) MPa``.
    """

    fc_prime: float
    fy: float
    Ec: float | None = None

    def __post_init__(self) -> None:
        if self.fc_prime <= 0.0:
            raise ValueError(f"fc_prime must be positive, got {self.fc_prime}")
        if self.fy <= 0.0:
            raise ValueError(f"fy must be positive, got {self.fy}")
        if self.Ec is None:
            # ACI 19.2.2.1: Ec = 4700 sqrt(f_c'[MPa]) MPa
            fc_MPa = self.fc_prime / 1.0e6
            self.Ec = 4700.0 * math.sqrt(fc_MPa) * 1.0e6
        if self.Ec <= 0.0:
            raise ValueError(f"Ec must be positive, got {self.Ec}")

    @property
    def beta_1(self) -> float:
        """Whitney stress-block depth factor per ACI 22.2.2.4.3."""
        return beta_1_aci(self.fc_prime)

    @property
    def epsilon_ty(self) -> float:
        """Yield strain of reinforcement: ``ε_ty = f_y / E_s``."""
        return self.fy / E_STEEL

    def __repr__(self) -> str:
        return (
            f"ConcreteMaterial(fc'={self.fc_prime/1e6:.1f} MPa, "
            f"fy={self.fy/1e6:.0f} MPa, Ec={self.Ec/1e9:.1f} GPa)"
        )


@dataclass
class RebarLayout:
    """Reinforcement layout in a rectangular section.

    Bars are described per *layer* (top, bottom), with stirrups
    described separately. Distances are measured from the **extreme
    compression fiber** at the top.

    Attributes
    ----------
    bottom_bars : tuple of str
        Standard bar designations for tension steel (e.g.,
        ``("#8", "#8", "#8")`` for three #8 bars).
    bottom_cover : float
        Clear cover from the bottom face to the surface of the
        bottom-bar centroid (m).
    top_bars : tuple of str, optional
        Compression-steel designations (default empty).
    top_cover : float, default same as bottom_cover
        Clear cover from the top face to the top-bar centroid (m).
    stirrup_designation : str, optional
        Stirrup bar designation (e.g., ``"#3"``). Defaults to ``"#3"``.
    stirrup_spacing : float, default 0.15 m
        Center-to-center stirrup spacing along the beam axis (m).
    stirrup_legs : int, default 2
        Number of vertical legs in a stirrup (typically 2 for closed
        stirrups, more for compound).
    """

    bottom_bars: tuple = ()
    bottom_cover: float = 0.040
    top_bars: tuple = ()
    top_cover: float | None = None
    stirrup_designation: str = "#3"
    stirrup_spacing: float = 0.150
    stirrup_legs: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.bottom_bars, tuple):
            self.bottom_bars = tuple(self.bottom_bars)
        if not isinstance(self.top_bars, tuple):
            self.top_bars = tuple(self.top_bars)
        for d in self.bottom_bars:
            rebar_area(d)     # validates
        for d in self.top_bars:
            rebar_area(d)
        if self.bottom_cover <= 0.0:
            raise ValueError(
                f"bottom_cover must be positive, got {self.bottom_cover}"
            )
        if self.top_cover is None:
            self.top_cover = self.bottom_cover
        if self.top_cover <= 0.0:
            raise ValueError(
                f"top_cover must be positive, got {self.top_cover}"
            )
        rebar_area(self.stirrup_designation)
        if self.stirrup_spacing <= 0.0:
            raise ValueError(
                f"stirrup_spacing must be positive, got {self.stirrup_spacing}"
            )
        if self.stirrup_legs < 2:
            raise ValueError(
                f"stirrup_legs must be >= 2, got {self.stirrup_legs}"
            )

    @property
    def As_bottom(self) -> float:
        """Total tension-steel area (m²)."""
        return sum(rebar_area(d) for d in self.bottom_bars)

    @property
    def As_top(self) -> float:
        """Total compression-steel area (m²)."""
        return sum(rebar_area(d) for d in self.top_bars)

    @property
    def Av(self) -> float:
        """Total transverse-shear-steel area per stirrup (m²)."""
        return self.stirrup_legs * rebar_area(self.stirrup_designation)


@dataclass
class ConcreteSection:
    """Rectangular concrete cross-section.

    Attributes
    ----------
    b : float
        Cross-section width (m).
    h : float
        Cross-section depth (m). Distance from extreme tension to
        extreme compression fiber.
    material : ConcreteMaterial
        f_c', f_y, E_c.
    rebar : RebarLayout
        Top, bottom, stirrup layout.

    Derived properties
    ------------------
    d : float
        Effective depth from extreme compression fiber to the
        centroid of *tension* reinforcement (m).
    d_prime : float
        Distance from extreme compression fiber to the centroid of
        *compression* reinforcement (m). Equal to ``top_cover``.
    Ag : float
        Gross cross-section area = b * h (m²).
    """

    b: float
    h: float
    material: ConcreteMaterial
    rebar: RebarLayout

    def __post_init__(self) -> None:
        if self.b <= 0.0:
            raise ValueError(f"b must be positive, got {self.b}")
        if self.h <= 0.0:
            raise ValueError(f"h must be positive, got {self.h}")
        if self.rebar.bottom_cover >= self.h:
            raise ValueError(
                f"bottom_cover ({self.rebar.bottom_cover}) must be "
                f"less than h ({self.h})"
            )

    @property
    def d(self) -> float:
        """Effective depth to centroid of tension reinforcement (m)."""
        return self.h - self.rebar.bottom_cover

    @property
    def d_prime(self) -> float:
        """Distance from extreme compression fiber to centroid of
        compression reinforcement (m). Returns ``top_cover``."""
        return self.rebar.top_cover

    @property
    def Ag(self) -> float:
        """Gross cross-section area (m²)."""
        return self.b * self.h

    @property
    def Ec(self) -> float:
        return self.material.Ec

    # ---------------------------------------------------- migration (II.7)
    def to_unified(self, *, steel_material=None):
        """Convert this legacy :class:`ConcreteSection` to a unified
        :class:`femsolver.sections.Section`.

        The reverse direction --
        :meth:`femsolver.sections.Section.as_aci_concrete_section` --
        was added in II.6. Together they let user code mix the two
        worlds freely during the migration window.

        Parameters
        ----------
        steel_material : optional
            ``UniaxialMaterial`` to attach to every rebar. Required
            only if the resulting unified Section will drive a
            nonlinear fiber analysis (then each bar needs a
            UniaxialMaterial). For design-only use, omit it.
        """
        from femsolver.sections import (
            ReinforcementLayout,
            rc_rectangular_section,
        )

        bottom_bar_specs = [
            (rebar_area(d), d) for d in self.rebar.bottom_bars
        ]
        top_bar_specs = [
            (rebar_area(d), d) for d in self.rebar.top_bars
        ]
        rl = ReinforcementLayout.from_rectangular_layers(
            b=self.b, h=self.h,
            bottom_bars=bottom_bar_specs,
            top_bars=top_bar_specs,
            bottom_cover=self.rebar.bottom_cover,
            top_cover=self.rebar.top_cover,
            stirrup_designation=self.rebar.stirrup_designation,
            stirrup_spacing=self.rebar.stirrup_spacing,
            stirrup_legs=self.rebar.stirrup_legs,
            steel_material=steel_material,
        )
        return rc_rectangular_section(
            b=self.b, h=self.h,
            concrete=self.material,
            reinforcement=rl,
        )

    def neutral_axis_balanced(self) -> float:
        """Depth of the neutral axis at *balanced* strain conditions
        per ACI 318-19 22.2.1: ε_c = 0.003 at extreme compression
        fiber AND ε_s = ε_ty at the tension steel.

        c_b = (ε_cu / (ε_cu + ε_ty)) · d
        """
        return EPSILON_CU * self.d / (EPSILON_CU + self.material.epsilon_ty)

    def As_min_flexure(self) -> float:
        """Minimum tension-steel area for flexure per ACI 318M-19 9.6.1.2:

            A_s,min = max(0.25 sqrt(f_c'[MPa]) / f_y[MPa],
                            1.4 / f_y[MPa]) · b · d

        f_c' and f_y are converted from Pa to MPa internally; the
        result is in m² (since b·d is in m²). This is the SI form of
        the imperial ``max(3 sqrt(f_c'[psi])/f_y[psi], 200/f_y[psi])``.
        """
        fc_MPa = self.material.fc_prime / 1.0e6
        fy_MPa = self.material.fy / 1.0e6
        ratio1 = 0.25 * math.sqrt(fc_MPa) / fy_MPa
        ratio2 = 1.4 / fy_MPa
        return max(ratio1, ratio2) * self.b * self.d

    def As_max_tension_controlled(self) -> float:
        """Maximum tension steel that keeps the section tension-controlled
        (``ε_t >= 0.005``) for a singly-reinforced section. From strain
        compatibility:

            c_max = ε_cu / (ε_cu + 0.005) · d = 3/8 · d
            a_max = β_1 · c_max
            A_s,max = 0.85 · f_c' · b · a_max / f_y
        """
        c_max = EPSILON_CU * self.d / (EPSILON_CU + 0.005)
        a_max = self.material.beta_1 * c_max
        return 0.85 * self.material.fc_prime * self.b * a_max / self.material.fy

    def __repr__(self) -> str:
        return (
            f"ConcreteSection(b={self.b:g} m, h={self.h:g} m, "
            f"d={self.d:g} m, As_bot={self.rebar.As_bottom*1e6:.0f} mm², "
            f"As_top={self.rebar.As_top*1e6:.0f} mm², "
            f"{self.material!r})"
        )
