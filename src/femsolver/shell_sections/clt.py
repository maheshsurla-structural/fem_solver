"""Cross-Laminated Timber (CLT) panel section (Phase D.1.2).

CLT is a layered orthotropic plate built from alternating layers of
dimension lumber rotated 90° between layers. Engineering description
follows EC5 Annex B (mechanically-jointed beams / γ-method) and
APA PRG-320 (Standard for Performance-Rated CLT in North America).

A typical 5-ply CLT panel:

   +-----------------------+  top
   |   layer 1 (0°)        |  ↕ grain parallel to strong axis
   +-----------------------+
   |   layer 2 (90°)       |  cross-grain
   +-----------------------+
   |   layer 3 (0°)        |
   +-----------------------+
   |   layer 4 (90°)       |
   +-----------------------+
   |   layer 5 (0°)        |
   +-----------------------+  bottom

* **Strong axis** (= y-axis of the section, parallel to outer-layer
  grain): the 0° layers (1, 3, 5) carry most of the bending.
* **Weak axis**: roles reverse -- the 90° layers carry the bending.

Effective stiffness model
-------------------------
For each layer, treat its modulus as:
* ``E_0_mean`` if the layer's fibre angle matches the bending axis
* ``E_90_mean`` if perpendicular

Then apply parallel-axis theorem to get composite EI. This is the
**γ = 1 limit** of EC5 Annex B (no joint slip / no rolling-shear
softening). For span-dependent γ < 1 reduction, see
:meth:`CLTSection.gamma_method`.

For shear: cross layers experience **rolling shear** with
``G_R ≈ G / 10`` (≈ ``E_0 / 160``) — much weaker than longitudinal
shear. This dominates shear-strain in CLT bending.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from femsolver.materials.timber.material import TimberMaterial


# ============================================================ CLTLayer

@dataclass
class CLTLayer:
    """One lamination of a CLT panel.

    Attributes
    ----------
    thickness : float
        Layer thickness (m).
    material : TimberMaterial
        Timber grade (typically same across all layers, but the design
        framework allows mixing).
    angle_deg : float, default 0
        Fibre direction relative to the panel's strong axis. ``0`` =
        parallel (longitudinal lamination), ``90`` = perpendicular
        (cross lamination). For a standard CLT layup these alternate.
    """
    thickness: float
    material: TimberMaterial
    angle_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.thickness <= 0:
            raise ValueError(f"thickness must be positive, got {self.thickness}")
        if self.angle_deg not in (0.0, 90.0):
            # CLT in practice only uses 0/90 -- raise for clarity.
            raise ValueError(
                f"angle_deg must be 0 or 90 (CLT convention); got {self.angle_deg}"
            )


# ============================================================ CLTSection

@dataclass
class CLTSection:
    """Cross-laminated timber panel section.

    Built from a list of :class:`CLTLayer` from top to bottom. The
    section's "strong axis" is parallel to the OUTER layers' grain
    direction. Use :meth:`beam_strip` to adapt to a unified
    :class:`~femsolver.sections.Section` for member-level analysis.

    Parameters
    ----------
    layers : list[CLTLayer]
        Top-to-bottom list of laminations. For a 5-ply panel use 5
        layers; typical CLT has 3, 5, 7, or 9 layers, always odd to
        give symmetric strong-axis bending.
    name : str
        Identifier (e.g. "100mm-5ply-C24", "175mm-7ply-V1").
    """
    layers: list[CLTLayer]
    name: str = "CLT"

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("CLT section needs at least one layer")
        # Check alternating-angle convention (standard CLT)
        # We allow non-alternating for research / unusual layups but
        # warn via the symmetry property.

    # ----------------------------------------------------------- geometry
    @property
    def total_thickness(self) -> float:
        """Total panel thickness (m)."""
        return float(sum(l.thickness for l in self.layers))

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def is_symmetric(self) -> bool:
        """``True`` if the layup mirrors about the panel mid-plane
        (standard for engineered CLT). Most engineering relations
        assume this."""
        n = len(self.layers)
        for i in range(n // 2):
            li = self.layers[i]
            lj = self.layers[n - 1 - i]
            if abs(li.thickness - lj.thickness) > 1e-9:
                return False
            if li.angle_deg != lj.angle_deg:
                return False
            if li.material.name != lj.material.name:
                return False
        return True

    def _layer_centroids_from_top(self) -> list[float]:
        """Centroidal y-coordinate of each layer measured from the
        TOP surface (positive going down)."""
        out = []
        y = 0.0
        for layer in self.layers:
            out.append(y + layer.thickness / 2.0)
            y += layer.thickness
        return out

    # ----------------------------------------------------------- mass
    def mass_per_area(self) -> float:
        """Panel mass per unit area (kg/m^2). Uses each layer's
        material density and thickness."""
        return float(sum(
            l.thickness * l.material.density_mean for l in self.layers
        ))

    # ----------------------------------------------------------- stiffness
    def _E_for_layer(self, layer: CLTLayer, *, strong_axis: bool) -> float:
        """Effective modulus for this layer in bending about the
        named axis. Strong-axis bending sees E_0 if layer angle=0,
        E_90 if angle=90. Weak-axis sees the reverse."""
        if strong_axis:
            return (layer.material.E_0_mean if layer.angle_deg == 0.0
                    else layer.material.E_90_mean)
        else:
            return (layer.material.E_0_mean if layer.angle_deg == 90.0
                    else layer.material.E_90_mean)

    def EA_per_width(self, *, strong_axis: bool = True) -> float:
        """Effective axial stiffness per unit panel width (N/m)."""
        return float(sum(
            self._E_for_layer(l, strong_axis=strong_axis) * l.thickness
            for l in self.layers
        ))

    def neutral_axis_from_top(self, *, strong_axis: bool = True) -> float:
        """Position of the elastic neutral axis from the top surface
        (m). For symmetric layups this is exactly the mid-thickness."""
        centroids = self._layer_centroids_from_top()
        num = 0.0
        den = 0.0
        for layer, y_c in zip(self.layers, centroids):
            E = self._E_for_layer(layer, strong_axis=strong_axis)
            EA = E * layer.thickness
            num += EA * y_c
            den += EA
        return float(num / max(den, 1e-30))

    def EI_eff_per_width(self, *, strong_axis: bool = True) -> float:
        """Composite effective bending stiffness per unit panel width
        (N·m^2/m = N·m). Sum of layer ``E · (I_self + A · d^2)``."""
        y_NA = self.neutral_axis_from_top(strong_axis=strong_axis)
        centroids = self._layer_centroids_from_top()
        EI = 0.0
        for layer, y_c in zip(self.layers, centroids):
            E = self._E_for_layer(layer, strong_axis=strong_axis)
            t = layer.thickness
            I_self = t ** 3 / 12.0          # per unit width
            A = t                           # per unit width
            d = y_c - y_NA
            EI += E * (I_self + A * d * d)
        return float(EI)

    def G_R_per_width(self) -> float:
        """Effective rolling-shear modulus weighted across cross-grain
        layers. APA PRG-320: G_R ≈ G / 10 for typical softwoods."""
        # Use the first cross-grain layer's material as a proxy
        # (typical CLT uses one species so this is unique).
        for layer in self.layers:
            if layer.angle_deg == 90.0:
                return float(layer.material.G_mean / 10.0)
        # No cross layer -> rolling shear doesn't apply
        for layer in self.layers:
            return float(layer.material.G_mean / 10.0)
        return 0.0

    # ----------------------------------------------------------- gamma method
    def gamma_method(
        self,
        span: float,
        *,
        strong_axis: bool = True,
    ) -> dict:
        """EC5 Annex B mechanically-jointed beam theory ("γ-method")
        accounting for rolling-shear softening of cross layers.

        For each *outer* longitudinal layer i, compute:
            γ_i = 1 / (1 + π² E_i A_i / (K_i · ℓ²))
        where K_i is the slip stiffness of the cross layer below /
        above (K = G_R · b / t_cross).

        Returns a dict with γ factors per layer and the effective
        EI accounting for them.

        Parameters
        ----------
        span : float
            Effective member span (m). Required for γ; smaller spans
            give smaller γ (less composite action).
        strong_axis : bool
            Bending axis.
        """
        if span <= 0:
            raise ValueError(f"span must be positive, got {span}")
        y_NA = self.neutral_axis_from_top(strong_axis=strong_axis)
        centroids = self._layer_centroids_from_top()
        G_R = self.G_R_per_width()

        gammas = []
        for i, (layer, y_c) in enumerate(zip(self.layers, centroids)):
            E_i = self._E_for_layer(layer, strong_axis=strong_axis)
            A_i = layer.thickness
            # If this is a load-carrying (longitudinal) layer, find the
            # adjacent cross layer thickness to compute K.
            is_load = (
                (strong_axis and layer.angle_deg == 0.0)
                or (not strong_axis and layer.angle_deg == 90.0)
            )
            if not is_load:
                gammas.append(1.0)
                continue
            # Adjacent cross layer (toward panel centre)
            mid_idx = len(self.layers) // 2
            if i < mid_idx:
                cross_layer = (
                    self.layers[i + 1] if i + 1 < len(self.layers) else None
                )
            elif i > mid_idx:
                cross_layer = self.layers[i - 1] if i - 1 >= 0 else None
            else:
                # Middle layer of an odd layup -- assumed to fully
                # participate; γ = 1
                gammas.append(1.0)
                continue
            if cross_layer is None or cross_layer.angle_deg == 0:
                gammas.append(1.0)
                continue
            K = G_R / cross_layer.thickness    # per unit width
            gamma = 1.0 / (1.0 + (math.pi ** 2 * E_i * A_i) /
                            (K * span ** 2))
            gammas.append(gamma)

        # Effective EI with γ factors
        EI = 0.0
        for gamma, layer, y_c in zip(gammas, self.layers, centroids):
            E = self._E_for_layer(layer, strong_axis=strong_axis)
            t = layer.thickness
            I_self = t ** 3 / 12.0
            A = t
            d = y_c - y_NA
            EI += E * I_self + gamma * E * A * d * d

        return {
            "gammas": gammas,
            "EI_eff": float(EI),
            "EI_full": self.EI_eff_per_width(strong_axis=strong_axis),
            "span": float(span),
            "G_R": float(G_R),
        }

    # ----------------------------------------------------------- adapter
    def beam_strip(
        self,
        *,
        width: float = 1.0,
        strong_axis: bool = True,
        name: Optional[str] = None,
    ):
        """Adapt this CLT panel to a unified :class:`Section` for
        member-level analysis (e.g. a beam-element pushover of a CLT
        floor strip).

        Returns a :class:`Section` whose elastic adapters give the
        correct EI and EA for the chosen bending axis. The Section's
        geometry is a rectangle of ``width × total_thickness`` with
        an attached "virtual material" carrying the effective E.
        """
        from femsolver.sections.parametric import rectangular_section
        from femsolver.sections.section import MaterialZone

        b = width
        h = self.total_thickness
        # Effective E for the section: choose so that EI of the
        # rectangle matches EI_eff_per_width * width.
        # Rectangle I = b h^3 / 12, so:
        # E_eff = EI_panel * width / (b * h^3 / 12) = 12 * EI_panel / h^3
        # (since b cancels — EI_panel is per unit width)
        EI_panel = self.EI_eff_per_width(strong_axis=strong_axis)
        EA_panel = self.EA_per_width(strong_axis=strong_axis)
        # Use EI to define E (more important for beam analysis)
        E_eff = 12.0 * EI_panel / (h ** 3)

        # Build a synthetic material with the effective modulus
        # Pull density from average of layer materials
        rho_eff = self.mass_per_area() / h
        synthetic = _CLTSyntheticMaterial(
            E=E_eff,
            density=rho_eff,
            EA_target=EA_panel * width,
        )

        sec = rectangular_section(
            b=b, h=h, material=synthetic,
            name=name or f"{self.name}-strip-{'strong' if strong_axis else 'weak'}",
        )
        return sec


# ============================================================ synthetic material

@dataclass
class _CLTSyntheticMaterial:
    """Lightweight material descriptor for the CLT beam-strip adapter.

    Exposes ``E``, ``nu``, ``density`` so it satisfies the duck-typing
    needed by :meth:`Section.elastic_section_3d`."""
    E: float
    density: float
    EA_target: float
    nu: float = 0.30

    @property
    def G(self) -> float:
        return self.E / (2 * (1 + self.nu))
