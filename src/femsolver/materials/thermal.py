"""Thermal-material model: conductivity, heat capacity, expansion.

Used by the thermal-element library
(:mod:`femsolver.elements.thermal`) and by the thermo-mechanical
coupling layer (:mod:`femsolver.analysis.thermal_strain`).

The model is deliberately a small dataclass: the constitutive
"matrix" for heat conduction is just ``k * I`` (isotropic) or a
``ndm x ndm`` diagonal/anisotropic tensor, and the heat capacity is
the scalar ``rho * c``.

Temperature-dependent properties (``k(T)``, ``c(T)``, ``f_y(T)``, …)
for fire engineering are provided separately in
:mod:`femsolver.analysis.fire_materials` so they can be composed with
this baseline model without complicating its interface.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ThermalMaterial:
    """Isotropic thermal properties at a single (reference) temperature.

    Attributes
    ----------
    tag : int
        User-facing identifier (mirrors mechanical-material tags).
    k : float
        Thermal conductivity (W/(m·K)).  Steel ~ 45, concrete ~ 1.5-2.0,
        aluminium ~ 230.
    rho : float
        Mass density (kg/m^3).  Steel ~ 7850, concrete ~ 2400.
    c : float
        Specific heat capacity (J/(kg·K)).  Steel ~ 460, concrete
        ~ 900-1000.
    alpha : float, default 0.0
        Linear thermal expansion coefficient (1/K).  Steel ~ 1.2e-5,
        concrete ~ 1.0e-5.  Set 0 if the model is only used for heat
        transfer.
    """

    tag: int
    k: float
    rho: float
    c: float
    alpha: float = 0.0

    def __post_init__(self) -> None:
        if self.k <= 0.0:
            raise ValueError(f"k must be > 0, got {self.k}")
        if self.rho <= 0.0:
            raise ValueError(f"rho must be > 0, got {self.rho}")
        if self.c <= 0.0:
            raise ValueError(f"c must be > 0, got {self.c}")
        if self.alpha < 0.0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")

    @property
    def rho_c(self) -> float:
        """Volumetric heat capacity ``rho · c`` (J/(m^3·K))."""
        return self.rho * self.c

    @property
    def diffusivity(self) -> float:
        """Thermal diffusivity ``k / (rho · c)`` (m^2/s)."""
        return self.k / self.rho_c
