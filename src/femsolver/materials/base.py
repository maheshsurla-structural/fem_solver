"""Material abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Material(ABC):
    """Abstract material. Subclasses provide stress-strain behaviour and
    a constitutive matrix (D) for the appropriate stress state."""

    def __init__(self, tag: int):
        self.tag = int(tag)

    @abstractmethod
    def D_plane_stress(self) -> np.ndarray:
        """3x3 constitutive matrix for plane stress (sigma_xx, sigma_yy, tau_xy)."""

    @abstractmethod
    def D_plane_strain(self) -> np.ndarray:
        """3x3 constitutive matrix for plane strain."""

    @abstractmethod
    def D_3d(self) -> np.ndarray:
        """6x6 constitutive matrix for full 3D stress/strain (Voigt order:
        xx, yy, zz, xy, yz, zx)."""

    @property
    @abstractmethod
    def E(self) -> float:
        """Effective Young's modulus (used by 1D elements such as truss/beam)."""

    @property
    @abstractmethod
    def G(self) -> float:
        """Effective shear modulus (used by beam torsion)."""

    @property
    @abstractmethod
    def rho(self) -> float:
        """Mass density."""
