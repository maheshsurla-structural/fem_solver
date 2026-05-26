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
    material : ElasticIsotropic or OrthotropicLamina
        Layer constitutive. Isotropic materials use the legacy
        E/nu path; orthotropic laminae produce a rotated Q-bar
        matrix using ``theta_deg``.
    thickness : float
        Through-thickness extent of this layer.
    z_mid : float, optional
        z-coordinate of the layer's mid-surface relative to the
        section reference plane (default 0). The stacking helper
        ``LayeredShellSection.from_layers_centered`` fills these in
        for you given an ordered list of (material, thickness)
        layers.
    theta_deg : float, default 0.0
        Fiber orientation in degrees (CCW from global x). Used only
        for orthotropic ``OrthotropicLamina`` materials; ignored for
        isotropic materials.
    """

    def __init__(self, material, thickness: float, z_mid: float = 0.0,
                 theta_deg: float = 0.0):
        if thickness <= 0:
            raise ValueError(f"layer thickness must be positive, got {thickness}")
        self.material = material
        self.thickness = float(thickness)
        self.z_mid = float(z_mid)
        self.theta_deg = float(theta_deg)

    @property
    def z_top(self) -> float:
        return self.z_mid + 0.5 * self.thickness

    @property
    def z_bot(self) -> float:
        return self.z_mid - 0.5 * self.thickness

    def _Q_inplane(self) -> np.ndarray:
        """Return the rotated 3x3 in-plane D-matrix of this layer."""
        mat = self.material
        if hasattr(mat, "Q_bar"):
            # OrthotropicLamina
            return mat.Q_bar(self.theta_deg)
        # Fall back to isotropic
        return _isotropic_plane_stress_D(mat.E, mat.nu)

    def _Qs_transverse(self, k_shear: float) -> np.ndarray:
        """Return the rotated 2x2 transverse-shear D-matrix of this
        layer (with shear-correction factor applied)."""
        mat = self.material
        if hasattr(mat, "Qs_bar"):
            return k_shear * mat.Qs_bar(self.theta_deg)
        # Isotropic fall-back: G * I_2
        return k_shear * mat.G * np.eye(2)

    @property
    def density(self) -> float:
        return getattr(self.material, "rho", 0.0)


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
        """Build a layered section from a list of layer specs.

        Each spec may be either:

        * ``(material, thickness)`` -- isotropic layer, fiber angle 0.
        * ``(material, thickness, theta_deg)`` -- orthotropic lamina
          with the given fiber orientation (CCW from global x).

        Stacks layers bottom-to-top centered on the reference plane.
        """
        total = sum(spec[1] for spec in layers)
        z = -0.5 * total
        built: list[ShellLayer] = []
        for spec in layers:
            if len(spec) == 2:
                material, thickness = spec
                theta_deg = 0.0
            elif len(spec) == 3:
                material, thickness, theta_deg = spec
            else:
                raise ValueError(
                    f"layer spec must be (material, thickness) or "
                    f"(material, thickness, theta_deg); got {spec!r}"
                )
            z_mid = z + 0.5 * thickness
            built.append(ShellLayer(
                material, thickness, z_mid=z_mid, theta_deg=theta_deg,
            ))
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
        return sum(layer.density * layer.thickness
                   for layer in self.layers) / T

    def D_membrane(self) -> np.ndarray:
        """Membrane (A) matrix: integral of Q_bar over thickness."""
        D = np.zeros((3, 3))
        for layer in self.layers:
            D += layer._Q_inplane() * layer.thickness
        return D

    def D_coupling(self) -> np.ndarray:
        """Coupling (B) matrix: integral of Q_bar * z dz.

        For each layer, contribution = Q_bar_layer * (z_top^2 - z_bot^2)/2
        which equals Q_bar * t * z_mid for layers of constant Q_bar.
        """
        D = np.zeros((3, 3))
        for layer in self.layers:
            D += layer._Q_inplane() * (
                (layer.z_top ** 2 - layer.z_bot ** 2) / 2.0
            )
        return D

    def D_bending(self) -> np.ndarray:
        """Bending (D) matrix: integral of Q_bar * z^2 dz over thickness.
        Each layer contributes Q_bar * (z_top^3 - z_bot^3) / 3.
        """
        D = np.zeros((3, 3))
        for layer in self.layers:
            D += layer._Q_inplane() * (
                (layer.z_top ** 3 - layer.z_bot ** 3) / 3.0
            )
        return D

    def D_shear(self) -> np.ndarray:
        """Transverse-shear (2x2) stiffness summed over layers.

        Each isotropic layer contributes ``k_shear * G * t * I``; each
        orthotropic layer contributes its rotated Q_s matrix scaled by
        thickness. The shear-correction factor ``k_shear`` is applied
        uniformly (composite shear-correction factors are an active
        area of research; the default 5/6 is widely accepted).
        """
        D = np.zeros((2, 2))
        for layer in self.layers:
            D += layer._Qs_transverse(self.k_shear) * layer.thickness
        return D

    # ------------------------------------------------------ stress recovery
    def ply_stresses(self, eps_membrane, kappa, *,
                       z: str = "all") -> list[dict]:
        """Recover per-ply stresses from the laminate's generalized
        strains using Classical Laminate Theory.

        Parameters
        ----------
        eps_membrane : sequence of 3 floats
            Mid-plane membrane strains ``(eps_xx, eps_yy, gamma_xy)``
            in the global (laminate) axes -- the strains returned by
            the shell element's ``recover`` at any Gauss point.
        kappa : sequence of 3 floats
            Curvatures ``(kappa_xx, kappa_yy, kappa_xy)`` in the same
            global axes.
        z : ``"all"``, ``"top"``, ``"bot"``, or ``"mid"``, default ``"all"``
            Which through-thickness positions to evaluate per ply.

            * ``"all"``: top, mid, bottom of each ply (3 stations per ply)
            * ``"top"`` / ``"bot"`` / ``"mid"``: just that station

        Returns
        -------
        list of dict, one per (ply, z-station) pair, each containing:

        * ``"layer"`` : 0-based layer index in the laminate
        * ``"z"``     : signed z-coordinate from the mid-surface
        * ``"theta_deg"`` : ply fiber orientation
        * ``"sigma_global"`` : (sigma_xx, sigma_yy, sigma_xy) in global axes
        * ``"sigma_local"``  : (sigma_11, sigma_22, sigma_12) in the
                                ply's material (1-2) axes
        * ``"eps_global"``   : the laminate strain at this z
        """
        eps_m = np.asarray(eps_membrane, dtype=float).reshape(3)
        kap = np.asarray(kappa, dtype=float).reshape(3)
        results: list[dict] = []
        station_keys = {
            "top": ("top",),
            "bot": ("bot",),
            "mid": ("mid",),
            "all": ("top", "mid", "bot"),
        }
        if z not in station_keys:
            raise ValueError(
                f"z must be 'all', 'top', 'bot', or 'mid'; got {z!r}"
            )
        for k, layer in enumerate(self.layers):
            Q_bar = layer._Q_inplane()
            theta = layer.theta_deg
            T = _stress_rotation_to_local(theta)
            for station in station_keys[z]:
                z_val = {
                    "top": layer.z_top,
                    "bot": layer.z_bot,
                    "mid": layer.z_mid,
                }[station]
                eps_at_z = eps_m + z_val * kap
                sigma_global = Q_bar @ eps_at_z
                sigma_local = T @ sigma_global
                results.append({
                    "layer": k,
                    "station": station,
                    "z": z_val,
                    "theta_deg": theta,
                    "sigma_global": sigma_global,
                    "sigma_local": sigma_local,
                    "eps_global": eps_at_z,
                })
        return results


# ============================================================ helpers

def _stress_rotation_to_local(theta_deg: float) -> np.ndarray:
    """Stress-rotation matrix T that takes global-axis stresses
    ``(sigma_xx, sigma_yy, sigma_xy)`` to lamina (1-2) material axes
    ``(sigma_11, sigma_22, sigma_12)`` for a ply rotated by ``theta_deg``
    CCW from global x to fiber direction.

    The transformation is

        T(theta) = [[c^2,   s^2,    2 c s],
                    [s^2,   c^2,   -2 c s],
                    [-c s,  c s,   c^2 - s^2]]

    where c = cos(theta), s = sin(theta). (Reuter form, engineering
    Voigt convention.)
    """
    import math
    theta = math.radians(theta_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([
        [c * c,     s * s,    2.0 * c * s],
        [s * s,     c * c,   -2.0 * c * s],
        [-c * s,    c * s,    c * c - s * s],
    ])
