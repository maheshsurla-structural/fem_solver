"""Shell sections — through-thickness integration of layered shells.

A shell section abstracts the cross-section response in the
generalized-strain ↔ generalized-stress sense, parallel to how
``SectionBase`` does for beam-columns. Shell sections expose four
D-matrices:

* ``D_membrane`` (3, 3) — in-plane membrane stiffness.
  N = D_m * eps_m,  eps_m = (eps_xx, eps_yy, gamma_xy)
* ``D_bending`` (3, 3) — out-of-plane bending stiffness.
  M = D_b * kappa,   kappa = (kappa_xx, kappa_yy, kappa_xy)
* ``D_coupling`` (3, 3) — membrane-bending coupling. Zero for
  symmetric mid-surface sections; nonzero for asymmetric stacks
  (e.g. sandwich with different face sheets).
* ``D_shear`` (2, 2) — transverse-shear stiffness.
  Q = D_s * gamma_s,  gamma_s = (gamma_xz, gamma_yz)

The simple isotropic single-layer section is the default and matches
the hard-coded constitutive used inside ``ShellMITC4``/``ShellTri3``.
A layered section sums each layer's contribution analytically:

    D_m = Sum_i D_m_i * t_i
    D_c = Sum_i D_m_i * t_i * z_mid_i
    D_b = Sum_i D_m_i * (z_top_i^3 - z_bot_i^3) / 3
    D_s = Sum_i k_i * G_i * t_i

This is the workhorse RC-slab / composite-sandwich formulation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

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


# ============================================================ isotropic helper

def _isotropic_plane_stress_D(E: float, nu: float) -> np.ndarray:
    """Plane-stress D (3x3) per unit thickness for an isotropic layer."""
    f = E / (1.0 - nu * nu)
    return f * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, 0.5 * (1.0 - nu)],
    ])


# ============================================================ single layer

class ElasticShellSection(ShellSectionBase):
    """Isotropic, single-layer elastic shell section.

    Parameters
    ----------
    material : ElasticIsotropic
    thickness : float
    k_shear : float, default 5/6
        Transverse-shear correction factor.

    Notes
    -----
    Backward-compatibility shell for the case ``ShellMITC4`` /
    ``ShellTri3`` already handle directly. Useful for explicit
    section passing and for comparison tests with
    ``LayeredShellSection``.
    """

    def __init__(self, material, thickness: float, *,
                 k_shear: float = 5.0 / 6.0):
        if thickness <= 0:
            raise ValueError(f"thickness must be positive, got {thickness}")
        if not (0.0 < k_shear <= 1.0):
            raise ValueError(f"k_shear must be in (0, 1], got {k_shear}")
        self.material = material
        self._thickness = float(thickness)
        self.k_shear = float(k_shear)

    @property
    def thickness(self) -> float:
        return self._thickness

    @property
    def density(self) -> float:
        return getattr(self.material, "rho", 0.0)

    def D_membrane(self) -> np.ndarray:
        return _isotropic_plane_stress_D(
            self.material.E, self.material.nu
        ) * self._thickness

    def D_bending(self) -> np.ndarray:
        return _isotropic_plane_stress_D(
            self.material.E, self.material.nu
        ) * (self._thickness ** 3 / 12.0)

    def D_coupling(self) -> np.ndarray:
        return np.zeros((3, 3))      # symmetric -> no coupling

    def D_shear(self) -> np.ndarray:
        return self.k_shear * self.material.G * self._thickness * np.eye(2)


# ============================================================ layered

class ShellLayer:
    """A single layer in a layered shell section.

    Parameters
    ----------
    material : ElasticIsotropic
    thickness : float
        Through-thickness extent of this layer.
    z_mid : float, optional
        z-coordinate of the layer's mid-surface relative to the
        section reference plane (default 0). The stacking helper
        ``LayeredShellSection.from_layers_centered`` fills these in
        for you given an ordered list of (material, thickness)
        layers.
    """

    def __init__(self, material, thickness: float, z_mid: float = 0.0):
        if thickness <= 0:
            raise ValueError(f"layer thickness must be positive, got {thickness}")
        self.material = material
        self.thickness = float(thickness)
        self.z_mid = float(z_mid)

    @property
    def z_top(self) -> float:
        return self.z_mid + 0.5 * self.thickness

    @property
    def z_bot(self) -> float:
        return self.z_mid - 0.5 * self.thickness


class LayeredShellSection(ShellSectionBase):
    """Multi-layer shell section. Each layer is an isotropic linear-
    elastic slab; the section integrates membrane / coupling / bending
    / shear analytically across the layered stack.

    Parameters
    ----------
    layers : sequence of :class:`ShellLayer`
        Layers in any order; their ``z_mid`` values define the
        through-thickness layout. Use
        :meth:`from_layers_centered` to auto-stack centered on the
        mid-surface.
    k_shear : float, default 5/6

    Notes
    -----
    For a symmetric (same material, same thickness on both sides of
    the mid-surface) stack, the coupling matrix is identically zero
    and membrane decouples from bending — the standard case for
    slabs and reinforced-concrete decks with symmetric reinforcement.

    For an asymmetric stack (e.g. RC slab with bottom-only steel,
    or sandwich plate with different face sheets), ``D_coupling`` is
    nonzero and membrane forces produce out-of-plane curvature.
    """

    def __init__(self, layers: Sequence[ShellLayer], *,
                 k_shear: float = 5.0 / 6.0):
        if len(layers) == 0:
            raise ValueError("need at least one layer")
        if not (0.0 < k_shear <= 1.0):
            raise ValueError(f"k_shear must be in (0, 1], got {k_shear}")
        self.layers = list(layers)
        self.k_shear = float(k_shear)

    @classmethod
    def from_layers_centered(
        cls,
        layers: Sequence[tuple],
        *,
        k_shear: float = 5.0 / 6.0,
    ) -> "LayeredShellSection":
        """Build a layered section from ``[(material, thickness), ...]``,
        stacking layers bottom-to-top centered on the reference plane.
        """
        total = sum(t for _, t in layers)
        z = -0.5 * total
        built: list[ShellLayer] = []
        for material, thickness in layers:
            z_mid = z + 0.5 * thickness
            built.append(ShellLayer(material, thickness, z_mid=z_mid))
            z += thickness
        return cls(built, k_shear=k_shear)

    @property
    def thickness(self) -> float:
        return sum(layer.thickness for layer in self.layers)

    @property
    def density(self) -> float:
        # Thickness-weighted average density
        T = self.thickness
        if T == 0.0:
            return 0.0
        return sum(layer.material.rho * layer.thickness
                   for layer in self.layers) / T

    def D_membrane(self) -> np.ndarray:
        D = np.zeros((3, 3))
        for layer in self.layers:
            D += _isotropic_plane_stress_D(
                layer.material.E, layer.material.nu
            ) * layer.thickness
        return D

    def D_coupling(self) -> np.ndarray:
        D = np.zeros((3, 3))
        for layer in self.layers:
            D += _isotropic_plane_stress_D(
                layer.material.E, layer.material.nu
            ) * (layer.thickness * layer.z_mid)
        return D

    def D_bending(self) -> np.ndarray:
        D = np.zeros((3, 3))
        for layer in self.layers:
            # Layer contribution to bending: integral of D_m * z^2 dz
            # = D_m * (z_top^3 - z_bot^3) / 3
            D += _isotropic_plane_stress_D(
                layer.material.E, layer.material.nu
            ) * (layer.z_top ** 3 - layer.z_bot ** 3) / 3.0
        return D

    def D_shear(self) -> np.ndarray:
        Gt = sum(self.k_shear * layer.material.G * layer.thickness
                 for layer in self.layers)
        return Gt * np.eye(2)
