"""Linear elastic isotropic material."""
from __future__ import annotations

import numpy as np

from femsolver.materials.base import Material


class ElasticIsotropic(Material):
    """Hooke's law for an isotropic medium.

    Parameters
    ----------
    tag : int
        Unique material identifier.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio (-1 < nu < 0.5).
    rho : float, optional
        Mass density (default 0.0).
    """

    def __init__(self, tag: int, E: float, nu: float, rho: float = 0.0):
        super().__init__(tag)
        if E <= 0:
            raise ValueError(f"E must be positive, got {E}")
        if not (-1.0 < nu < 0.5):
            raise ValueError(f"nu must be in (-1, 0.5), got {nu}")
        if rho < 0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        self._E = float(E)
        self._nu = float(nu)
        self._rho = float(rho)

    @property
    def E(self) -> float:
        return self._E

    @property
    def nu(self) -> float:
        return self._nu

    @property
    def G(self) -> float:
        return self._E / (2.0 * (1.0 + self._nu))

    @property
    def rho(self) -> float:
        return self._rho

    @property
    def K(self) -> float:
        """Bulk modulus."""
        return self._E / (3.0 * (1.0 - 2.0 * self._nu))

    def D_plane_stress(self) -> np.ndarray:
        E, nu = self._E, self._nu
        f = E / (1.0 - nu * nu)
        return f * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - nu)],
        ])

    def D_plane_strain(self) -> np.ndarray:
        E, nu = self._E, self._nu
        f = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return f * np.array([
            [1.0 - nu, nu, 0.0],
            [nu, 1.0 - nu, 0.0],
            [0.0, 0.0, 0.5 - nu],
        ])

    def D_3d(self) -> np.ndarray:
        E, nu = self._E, self._nu
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = self.G
        D = np.zeros((6, 6))
        D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * mu
        D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = lam
        D[3, 3] = D[4, 4] = D[5, 5] = mu
        return D

    def __repr__(self) -> str:
        return f"ElasticIsotropic(tag={self.tag}, E={self._E:g}, nu={self._nu:g}, rho={self._rho:g})"
