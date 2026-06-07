"""Orthotropic lamina material for composite laminate analysis.

Each ply (lamina) in a composite laminate is treated as an
orthotropic plane-stress material with a principal "1-direction"
(fiber direction) and a perpendicular "2-direction" (matrix
direction). The plane-stress constitutive in the material (1-2) axes
is:

    Q = [[E1 / (1 - nu12 nu21), nu12 E2 / (1 - nu12 nu21), 0],
         [nu12 E2 / (1 - nu12 nu21), E2 / (1 - nu12 nu21), 0],
         [0, 0, G12]]

with the Maxwell relation ``nu21 = nu12 * E2 / E1``.

A lamina rotated by angle ``theta`` (measured CCW from the global x
axis to the fiber direction) contributes its ``Q_bar`` matrix to the
laminate. The rotation transforms the lamina (1-2) stiffness to the
global (x-y) basis:

    Q_bar = T^-1 Q T_eps

where ``T`` is the stress-rotation matrix (engineering Voigt) and
``T_eps`` is the strain-rotation matrix. The standard equivalent
form is ``Q_bar = R Q R^T`` with ``R`` the Reuter-adjusted rotation
matrix (Jones, "Mechanics of Composite Materials" ch. 2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class OrthotropicLamina:
    """Single-ply orthotropic plane-stress material.

    Attributes
    ----------
    E1 : float
        Young's modulus in the fiber direction.
    E2 : float
        Young's modulus transverse to the fiber direction.
    G12 : float
        In-plane shear modulus.
    nu12 : float
        Major Poisson's ratio (-1 < nu12 < sqrt(E1/E2)).
    G13 : float, optional
        Out-of-plane shear modulus in the 1-3 plane. Defaults to G12.
        Used for transverse-shear stiffness in the laminate's shear
        constitutive.
    G23 : float, optional
        Out-of-plane shear modulus in the 2-3 plane. Defaults to G12.
    rho : float, default 0.0
        Mass density.

    Examples
    --------
    Typical CFRP T300/5208 ply:

    >>> lamina = OrthotropicLamina(
    ...     E1=181e9, E2=10.3e9, G12=7.17e9, nu12=0.28, rho=1600.0,
    ... )

    Isotropic equivalence: setting ``E1 = E2 = E``, ``G12 = E/(2(1+nu))``,
    and ``nu12 = nu`` recovers an isotropic ply.
    """

    E1: float
    E2: float
    G12: float
    nu12: float
    G13: float | None = None
    G23: float | None = None
    rho: float = 0.0

    def __post_init__(self) -> None:
        if self.E1 <= 0.0:
            raise ValueError(f"E1 must be positive, got {self.E1}")
        if self.E2 <= 0.0:
            raise ValueError(f"E2 must be positive, got {self.E2}")
        if self.G12 <= 0.0:
            raise ValueError(f"G12 must be positive, got {self.G12}")
        if not (-1.0 < self.nu12 < math.sqrt(self.E1 / self.E2)):
            raise ValueError(
                f"nu12 must satisfy -1 < nu12 < sqrt(E1/E2), "
                f"got {self.nu12}"
            )
        if self.rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {self.rho}")
        if self.G13 is None:
            self.G13 = self.G12
        if self.G23 is None:
            self.G23 = self.G12

    @property
    def nu21(self) -> float:
        return self.nu12 * self.E2 / self.E1

    def Q_lamina(self) -> np.ndarray:
        """Plane-stress D-matrix in lamina (1-2) axes. 3x3, ordering
        (sigma_11, sigma_22, sigma_12) on engineering strains."""
        nu12 = self.nu12
        nu21 = self.nu21
        denom = 1.0 - nu12 * nu21
        Q11 = self.E1 / denom
        Q22 = self.E2 / denom
        Q12 = nu12 * self.E2 / denom
        Q66 = self.G12
        return np.array([
            [Q11, Q12, 0.0],
            [Q12, Q22, 0.0],
            [0.0, 0.0, Q66],
        ])

    def Q_bar(self, theta_deg: float) -> np.ndarray:
        """In-plane D-matrix rotated by ``theta_deg`` from the global
        x axis to the fiber direction. ``theta_deg = 0`` means fibers
        aligned with the global x axis.
        """
        theta = math.radians(theta_deg)
        c = math.cos(theta)
        s = math.sin(theta)
        c2, s2 = c * c, s * s
        cs = c * s
        Q = self.Q_lamina()
        Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]
        # Standard Jones / Tsai formulas
        Q11b = Q11 * c2 ** 2 + 2.0 * (Q12 + 2.0 * Q66) * s2 * c2 + Q22 * s2 ** 2
        Q22b = Q11 * s2 ** 2 + 2.0 * (Q12 + 2.0 * Q66) * s2 * c2 + Q22 * c2 ** 2
        Q12b = (Q11 + Q22 - 4.0 * Q66) * s2 * c2 + Q12 * (s2 ** 2 + c2 ** 2)
        Q66b = (Q11 + Q22 - 2.0 * Q12 - 2.0 * Q66) * s2 * c2 + Q66 * (s2 ** 2 + c2 ** 2)
        Q16b = (Q11 - Q12 - 2.0 * Q66) * c2 * cs + (Q12 - Q22 + 2.0 * Q66) * s2 * cs
        Q26b = (Q11 - Q12 - 2.0 * Q66) * s2 * cs + (Q12 - Q22 + 2.0 * Q66) * c2 * cs
        return np.array([
            [Q11b, Q12b, Q16b],
            [Q12b, Q22b, Q26b],
            [Q16b, Q26b, Q66b],
        ])

    def Qs_bar(self, theta_deg: float) -> np.ndarray:
        """Out-of-plane (transverse) shear D-matrix rotated by
        ``theta_deg``. 2x2, ordering (sigma_xz, sigma_yz)."""
        theta = math.radians(theta_deg)
        c = math.cos(theta)
        s = math.sin(theta)
        Q44 = self.G23     # shear in 2-3 plane (gamma_23)
        Q55 = self.G13     # shear in 1-3 plane (gamma_13)
        # Rotation of the 2D shear stiffness; standard Reuter form
        Qs44b = Q44 * c * c + Q55 * s * s
        Qs55b = Q44 * s * s + Q55 * c * c
        Qs45b = (Q55 - Q44) * c * s
        return np.array([
            [Qs55b, Qs45b],
            [Qs45b, Qs44b],
        ])
