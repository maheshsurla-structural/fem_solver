"""Shell-section contract (the 2-D analogue of ``SectionBase``).

A *shell section* abstracts the through-thickness cross-section response
of a plate/shell element in the generalized-strain to generalized-stress
sense, exactly parallel to how :class:`femsolver.sections.SectionBase`
does it for beam-columns. Where a beam section maps (axial strain,
curvatures) -> (axial force, moments), a shell section exposes four
D-matrices:

* ``D_membrane`` (3, 3) -- in-plane membrane stiffness.
  N = D_m * eps_m,  eps_m = (eps_xx, eps_yy, gamma_xy)
* ``D_bending`` (3, 3) -- out-of-plane bending stiffness.
  M = D_b * kappa,   kappa = (kappa_xx, kappa_yy, kappa_xy)
* ``D_coupling`` (3, 3) -- membrane-bending coupling. Zero for
  symmetric mid-surface sections; nonzero for asymmetric stacks
  (e.g. sandwich with different face sheets).
* ``D_shear`` (2, 2) -- transverse-shear stiffness.
  Q = D_s * gamma_s,  gamma_s = (gamma_xz, gamma_yz)

Concrete implementations live in :mod:`femsolver.shell_sections.layered`
(single-layer isotropic and multi-layer laminate).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ShellSectionBase(ABC):
    """Abstract base for shell sections (membrane + bending + shear)."""

    @property
    @abstractmethod
    def thickness(self) -> float:
        """Total through-thickness measure (sum of layer thicknesses)."""

    @property
    @abstractmethod
    def density(self) -> float:
        """Effective mass per unit area / thickness = mean density."""

    @abstractmethod
    def D_membrane(self) -> np.ndarray: ...

    @abstractmethod
    def D_bending(self) -> np.ndarray: ...

    @abstractmethod
    def D_coupling(self) -> np.ndarray: ...

    @abstractmethod
    def D_shear(self) -> np.ndarray: ...
