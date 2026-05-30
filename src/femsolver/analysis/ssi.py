"""Soil-structure interaction (SSI): Gazetas (1991) / Pais-Kausel (1988)
static impedance functions for rigid surface footings on a homogeneous
elastic halfspace.

For a rectangular footing of half-widths ``B <= L`` resting on a
homogeneous halfspace with shear modulus ``G``, Poisson ratio ``nu``,
and mass density ``rho``, this module returns the six static
foundation stiffnesses (translation in x/y/z, rocking about x/y, and
torsion about z) plus the associated radiation-damping coefficients
in the Lysmer-Richart sense.

Conventions
-----------
* ``x`` is the short in-plane direction, ``y`` the long in-plane
  direction, ``z`` is vertical (up).
* ``L >= B`` always; if you swap them you swap which formula you use.
* All static stiffnesses are expressed per unit rotation (rocking) or
  unit translation (translational).

References
----------
* Gazetas, G. (1991) "Foundation vibrations." *Foundation Engineering
  Handbook*, 2e, Chapter 15.
* Pais, A. & Kausel, E. (1988) "Approximate formulas for dynamic
  stiffnesses of rigid foundations." *Soil Dyn. and Earthq. Eng.*,
  7(4), 213-227.
* Wolf, J.P. (1985) *Dynamic Soil-Structure Interaction*. Prentice-
  Hall (radiation-damping forms used here).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class HalfspaceSoil:
    """Homogeneous halfspace properties.

    Attributes
    ----------
    G : float
        Shear modulus (Pa).
    nu : float
        Poisson ratio (-).
    rho : float
        Mass density (kg/m^3).
    """

    G: float
    nu: float
    rho: float

    def __post_init__(self) -> None:
        if self.G <= 0.0:
            raise ValueError(f"G must be > 0, got {self.G}")
        if not (0.0 < self.nu < 0.5):
            raise ValueError(f"nu must be in (0, 0.5), got {self.nu}")
        if self.rho <= 0.0:
            raise ValueError(f"rho must be > 0, got {self.rho}")

    @property
    def Vs(self) -> float:
        """Shear-wave velocity sqrt(G/rho)."""
        return math.sqrt(self.G / self.rho)

    @property
    def Vla(self) -> float:
        """Lysmer's analog velocity for the P-wave halfspace::

            V_la = 3.4 / (pi (1 - nu)) · V_s

        Used in radiation-damping of vertical and rocking modes.
        """
        return 3.4 / (math.pi * (1.0 - self.nu)) * self.Vs


@dataclass
class FootingImpedance:
    """Result of a Gazetas / Pais-Kausel impedance evaluation.

    Attributes
    ----------
    K_z : float
        Vertical stiffness (N/m).
    K_x : float
        Horizontal stiffness in short direction (N/m).
    K_y : float
        Horizontal stiffness in long direction (N/m).
    K_rx : float
        Rocking about long-axis x (tipping in y-direction) (N·m/rad).
    K_ry : float
        Rocking about short-axis y (tipping in x-direction) (N·m/rad).
    K_t : float
        Torsion about vertical z (N·m/rad).
    C_z, C_x, C_y, C_rx, C_ry, C_t : float
        Radiation-damping coefficients (N·s/m or N·m·s/rad).
    B, L : float
        Footing half-widths (m); ``L >= B``.
    """

    K_z: float
    K_x: float
    K_y: float
    K_rx: float
    K_ry: float
    K_t: float
    C_z: float
    C_x: float
    C_y: float
    C_rx: float
    C_ry: float
    C_t: float
    B: float
    L: float


def gazetas_surface_footing(
    soil: HalfspaceSoil,
    *,
    B: float,
    L: float,
) -> FootingImpedance:
    """Pais-Kausel (1988) static impedances for a rigid surface
    rectangular footing of half-widths ``B`` (short) and ``L`` (long),
    plus Lysmer-Wolf radiation damping.

    Parameters
    ----------
    soil : HalfspaceSoil
    B, L : float
        Footing half-widths (m). Must satisfy ``L >= B > 0``.

    Returns
    -------
    FootingImpedance

    Notes
    -----
    Static stiffnesses (Pais-Kausel Table 1)::

        K_z  = (G B / (1 - nu)) · [3.1 (L/B)^0.75 + 1.6]
        K_y  = (G B / (2 - nu)) · [6.8 (L/B)^0.65 + 0.8 (L/B) + 1.6]
        K_x  = (G B / (2 - nu)) · [6.8 (L/B)^0.65 + 2.4]
        K_rx = (G B^3 / (1 - nu)) · [3.2 (L/B) + 0.8]
        K_ry = (G B^3 / (1 - nu)) · [3.73 (L/B)^2.4 + 0.27]
        K_t  =  G B^3 · [4.25 (L/B)^2.45 + 4.06]

    Radiation damping is the static-limit Wolf form::

        C_z = rho V_la · A_b              (A_b = 4 B L)
        C_x = C_y = rho V_s · A_b
        C_rx = rho V_la · I_x             (I_x = (2L)(2B)^3 / 3)
        C_ry = rho V_la · I_y             (I_y = (2B)(2L)^3 / 3)
        C_t  = rho V_s · (I_x + I_y)
    """
    if B <= 0.0:
        raise ValueError(f"B must be > 0, got {B}")
    if L < B:
        raise ValueError(f"L must be >= B (got L={L}, B={B})")

    G = soil.G
    nu = soil.nu
    rho = soil.rho
    Vs = soil.Vs
    Vla = soil.Vla
    aspect = L / B           # >= 1

    # Translational stiffnesses
    K_z = (G * B / (1.0 - nu)) * (
        3.1 * aspect ** 0.75 + 1.6
    )
    K_y = (G * B / (2.0 - nu)) * (
        6.8 * aspect ** 0.65 + 0.8 * aspect + 1.6
    )
    K_x = (G * B / (2.0 - nu)) * (
        6.8 * aspect ** 0.65 + 2.4
    )

    # Rocking and torsion stiffnesses
    B3 = B ** 3
    K_rx = (G * B3 / (1.0 - nu)) * (
        3.2 * aspect + 0.8
    )
    K_ry = (G * B3 / (1.0 - nu)) * (
        3.73 * aspect ** 2.4 + 0.27
    )
    K_t = G * B3 * (
        4.25 * aspect ** 2.45 + 4.06
    )

    # Radiation damping (Wolf static-limit form)
    A_b = 4.0 * B * L
    # Area moments of the full 2B x 2L footing about its own centroidal axes
    # Long axis x (along y direction width is 2B): I_x = (2L)(2B)^3 / 12
    # Short axis y (along x direction width is 2L): I_y = (2B)(2L)^3 / 12
    I_x = (2.0 * L) * (2.0 * B) ** 3 / 12.0
    I_y = (2.0 * B) * (2.0 * L) ** 3 / 12.0
    C_z = rho * Vla * A_b
    C_x = rho * Vs * A_b
    C_y = rho * Vs * A_b
    C_rx = rho * Vla * I_x
    C_ry = rho * Vla * I_y
    C_t = rho * Vs * (I_x + I_y)

    return FootingImpedance(
        K_z=K_z, K_x=K_x, K_y=K_y,
        K_rx=K_rx, K_ry=K_ry, K_t=K_t,
        C_z=C_z, C_x=C_x, C_y=C_y,
        C_rx=C_rx, C_ry=C_ry, C_t=C_t,
        B=B, L=L,
    )


def embedment_correction(
    impedance: FootingImpedance,
    soil: HalfspaceSoil,
    *,
    D: float,
) -> FootingImpedance:
    """Apply Gazetas (1991) approximate corrections for a footing
    embedded to depth ``D`` (m).

    Per Gazetas (1991), the embedded-stiffness coefficients are
    multiplicative factors applied to the surface values::

        K_z (embedded)  = K_z (surface)  · [1 + (1/21)(D/B)(1 + 1.3 chi)]
        K_y (embedded)  = K_y (surface)  · [1 + 0.15 (D/L)^0.5]
        K_x (embedded)  = K_x (surface)  · [1 + 0.15 (D/B)^0.5]
        K_rx (embedded) = K_rx (surface) · [1 + 0.92 (D/B)^0.6 ·
                                              (1.5 + (D/B)^1.9 (L/B)^-0.6)]

    Vertical and rocking get the largest multipliers; horizontal modes
    get a modest 1+ factor. Damping coefficients are unchanged in
    this static-limit approximation.

    Parameters
    ----------
    impedance : FootingImpedance
        From :func:`gazetas_surface_footing`.
    soil : HalfspaceSoil
    D : float
        Embedment depth (m). Must be >= 0.

    Returns
    -------
    FootingImpedance
        New instance with embedded stiffnesses.
    """
    if D < 0.0:
        raise ValueError(f"D must be >= 0, got {D}")
    if D == 0.0:
        return impedance     # no embedment

    B = impedance.B
    L = impedance.L
    chi = B / L              # in (0, 1]

    f_z = 1.0 + (1.0 / 21.0) * (D / B) * (1.0 + 1.3 * chi)
    f_y = 1.0 + 0.15 * math.sqrt(D / L)
    f_x = 1.0 + 0.15 * math.sqrt(D / B)
    f_rx = 1.0 + 0.92 * (D / B) ** 0.6 * (
        1.5 + (D / B) ** 1.9 * (L / B) ** -0.6
    )
    f_ry = 1.0 + 0.92 * (D / L) ** 0.6 * (
        1.5 + (D / L) ** 1.9 * (B / L) ** -0.6
    )
    f_t = 1.0 + 1.4 * (1.0 + B / L) * (D / B) ** 0.9

    return FootingImpedance(
        K_z=impedance.K_z * f_z,
        K_x=impedance.K_x * f_x,
        K_y=impedance.K_y * f_y,
        K_rx=impedance.K_rx * f_rx,
        K_ry=impedance.K_ry * f_ry,
        K_t=impedance.K_t * f_t,
        C_z=impedance.C_z, C_x=impedance.C_x, C_y=impedance.C_y,
        C_rx=impedance.C_rx, C_ry=impedance.C_ry, C_t=impedance.C_t,
        B=B, L=L,
    )
