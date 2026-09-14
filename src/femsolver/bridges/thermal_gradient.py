"""Bridge temperature-gradient loads (bridge plan T1.3).

A bridge superstructure heated by the sun develops a **nonlinear** temperature
distribution ``T(y)`` through its depth. Because plane sections stay plane, the
section cannot strain to match a nonlinear ``T(y)`` freely, so three effects
arise (Ghali & Neville; AASHTO LRFD §3.12.3 commentary):

1. an **equivalent uniform temperature** ``ΔT_N`` → axial strain ``ε₀ = α·ΔT_N``;
2. an **equivalent linear gradient** → curvature ``κ`` (free camber);
3. a **self-equilibrated (eigen) stress** ``σ_self(y)`` — present even in a
   simply-supported (determinate) girder, integrating to zero axial force and
   zero moment.

For a **continuous** (indeterminate) bridge the free curvature ``κ`` is
restrained at the interior supports, so **continuity (secondary) moments**
develop — obtained by imposing ``(ε₀, κ)`` on the frame as equivalent nodal
loads (:func:`apply_beam_thermal_actions`) and solving.

This module provides the code gradient profiles (:func:`aashto_gradient`,
:func:`linear_gradient`), the section reduction
(:func:`equivalent_thermal_actions` → ``ΔT_N``, ``κ``, ``σ_self``), and the
frame application. It is the temperature-gradient counterpart of the tendon /
prestress machinery (equivalent loads → primary + secondary effects).

References
----------
* AASHTO LRFD Bridge Design Specifications §3.12.3 (vertical temperature
  gradient; Table 3.12.3-1 zone values; Fig. 3.12.3-2 profile).
* EN 1991-1-5 §6.1.4 / Annex B (vertical temperature components).
* Ghali, Favre & Elbadry, *Concrete Structures: Stresses and Deformations*.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# AASHTO LRFD Table 3.12.3-1 — positive vertical gradient, concrete
# superstructure, no asphalt overlay: {solar zone: (T1, T2) in °C}. T3 is
# taken 0 °C unless the owner specifies otherwise (0–3 °C).
_AASHTO_ZONE_T1_T2 = {1: (30.0, 7.8), 2: (25.0, 6.7),
                      3: (23.0, 6.0), 4: (21.0, 5.0)}


@dataclass
class TemperatureGradient:
    """A vertical temperature profile ``T`` vs depth-from-top.

    Parameters
    ----------
    depths : array-like
        Depths below the top fibre (m), non-decreasing, starting at 0.
    temps : array-like
        Temperature (°C above the reference) at each depth. Piecewise-linear
        between points; clamped to the end values outside the range.
    alpha : float, default 1.0e-5
        Coefficient of thermal expansion (1/°C).
    name : str
    """

    depths: np.ndarray
    temps: np.ndarray
    alpha: float = 1.0e-5
    name: str = "gradient"

    def __post_init__(self) -> None:
        self.depths = np.asarray(self.depths, dtype=float).ravel()
        self.temps = np.asarray(self.temps, dtype=float).ravel()
        if self.depths.size != self.temps.size or self.depths.size < 2:
            raise ValueError("depths and temps must be equal length >= 2")
        if np.any(np.diff(self.depths) < 0):
            raise ValueError("depths must be non-decreasing")

    def T(self, depth) -> np.ndarray:
        """Temperature at ``depth`` below the top (clamped outside range)."""
        return np.interp(depth, self.depths, self.temps)


def linear_gradient(dT_top: float, dT_bot: float, depth: float,
                    *, alpha: float = 1.0e-5) -> TemperatureGradient:
    """A simple linear top-to-bottom gradient (a linear profile produces no
    self-equilibrated stress — useful as a baseline / for EN linear cases)."""
    return TemperatureGradient([0.0, depth], [dT_top, dT_bot], alpha=alpha,
                               name="linear")


def aashto_gradient(zone: int, depth: float, *, alpha: float = 1.0e-5,
                    T3: float = 0.0) -> TemperatureGradient:
    """AASHTO LRFD §3.12.3 positive vertical gradient for a concrete
    superstructure (no asphalt), solar ``zone`` 1–4.

    Profile (Fig. 3.12.3-2): ``T1`` at the top surface, linear to ``T2`` at
    100 mm depth, linear to ``T3`` at depth ``A`` (300 mm for depth ≥ 400 mm,
    else ``depth − 100 mm``), and ``T3`` held to the soffit.
    """
    if zone not in _AASHTO_ZONE_T1_T2:
        raise ValueError("zone must be 1, 2, 3 or 4")
    T1, T2 = _AASHTO_ZONE_T1_T2[zone]
    A = 0.3 if depth >= 0.4 else max(depth - 0.1, 0.1)
    depths = [0.0, 0.1, A]
    temps = [T1, T2, T3]
    if depth > A:                                   # hold T3 to the soffit
        depths.append(depth)
        temps.append(T3)
    return TemperatureGradient(depths, temps, alpha=alpha,
                               name=f"AASHTO zone {zone}")


@dataclass
class SectionThermalActions:
    """Result of reducing a :class:`TemperatureGradient` on a section.

    Attributes
    ----------
    dT_uniform : float
        Equivalent uniform temperature ``ΔT_N`` (°C) → axial strain.
    curvature : float
        Free thermal curvature ``κ`` (1/m). Positive = the profile makes the
        member sag (hog) per the section's sign convention; validated so a
        top-hotter gradient cambers a simple span upward.
    eps0 : float
        Axial strain ``α·ΔT_N``.
    dT_gradient_equiv : float
        Equivalent linear temperature difference top-to-bottom that produces
        the same curvature (``κ·h/α``), for reporting alongside code checks.
    area, inertia, centroid_depth : float
        Section properties used (from the top fibre).
    self_stress_top, self_stress_bottom : float
        Self-equilibrated stress at the extreme fibres (``E`` × strain diff).
    """

    dT_uniform: float
    curvature: float
    eps0: float
    dT_gradient_equiv: float
    area: float
    inertia: float
    centroid_depth: float
    self_stress_top: float
    self_stress_bottom: float
    _E: float = 0.0
    _alpha: float = 1.0e-5
    _height: float = 0.0
    _dc: float = 0.0
    _kappa: float = 0.0
    _eps0: float = 0.0

    def self_stress(self, depth) -> np.ndarray:
        """Self-equilibrated stress at ``depth`` below the top (Pa).

        ``σ_self(y) = E·[ε₀ + κ·y − α·T(y)]`` with ``y`` measured up from the
        centroid. It integrates to zero N and zero M over the section."""
        raise NotImplementedError  # replaced by a closure in the builder


def equivalent_thermal_actions(
    gradient: TemperatureGradient,
    *,
    height: float,
    width,
    E: float,
    n: int = 400,
) -> SectionThermalActions:
    """Reduce a temperature gradient on a section to ``(ΔT_N, κ, σ_self)``.

    Parameters
    ----------
    gradient : TemperatureGradient
    height : float
        Section depth (m).
    width : float or callable
        Section width — a constant (rectangular) or ``b(depth_from_top)``.
    E : float
        Elastic modulus (Pa).
    n : int, default 400
        Integration sub-divisions through the depth.
    """
    alpha = gradient.alpha
    d = np.linspace(0.0, height, n + 1)             # depth from top
    b = (np.full_like(d, float(width)) if not callable(width)
         else np.array([float(width(di)) for di in d]))
    T = gradient.T(d)

    A = float(np.trapezoid(b, d))
    dc = float(np.trapezoid(b * d, d) / A)          # centroid depth from top
    y = dc - d                                       # +y upward from centroid
    I = float(np.trapezoid(b * y * y, d))

    dT_N = float(np.trapezoid(T * b, d) / A)         # equivalent uniform temp
    # curvature: κ = α/I · ∫ T·b·y dd  (y up from centroid)
    kappa = alpha * float(np.trapezoid(T * b * y, d)) / I
    eps0 = alpha * dT_N
    dT_grad_equiv = kappa * height / alpha if alpha else 0.0

    def _self_stress(depth):
        yy = dc - np.asarray(depth, dtype=float)
        return E * (eps0 + kappa * yy - alpha * gradient.T(depth))

    s_top = float(_self_stress(0.0))
    s_bot = float(_self_stress(height))
    res = SectionThermalActions(
        dT_uniform=dT_N, curvature=kappa, eps0=eps0,
        dT_gradient_equiv=dT_grad_equiv, area=A, inertia=I,
        centroid_depth=dc, self_stress_top=s_top, self_stress_bottom=s_bot,
        _E=E, _alpha=alpha, _height=height, _dc=dc, _kappa=kappa, _eps0=eps0)
    res.self_stress = _self_stress                   # bind the closure
    return res


def apply_beam_thermal_actions(model, *, eps0: float, kappa: float,
                               elements=None) -> int:
    """Apply an axial strain ``eps0`` and curvature ``kappa`` to 2-D beam
    elements as equivalent nodal loads, so a subsequent linear solve gives the
    thermal deflections and — for indeterminate structures — the continuity
    (secondary) moments. Returns the number of elements loaded.

    The equivalent local end-force vector for imposed ``(ε₀, κ)`` on a 2-D
    Euler beam (DOF order ``[u,v,θ]`` per node) is
    ``[-EA·ε₀, 0, +EI·κ, +EA·ε₀, 0, -EI·κ]``; it is rotated to global and
    scattered to the element's nodes. A free (determinate) member then simply
    elongates and cambers with zero stress — a top-hotter gradient (``κ`` > 0)
    cambers a simple span upward.
    """
    tags = (set(elements) if elements is not None
            else set(model.elements.keys()))
    n_done = 0
    for el in model.elements.values():
        if el.tag not in tags:
            continue
        if not (hasattr(el, "area") and hasattr(el, "Iz")
                and hasattr(el, "transform_matrix")):
            continue
        E = el.material.E
        EA = E * el.area
        EI = E * el.Iz
        f_local = np.array([-EA * eps0, 0.0, +EI * kappa,
                            +EA * eps0, 0.0, -EI * kappa])
        f_global = el.transform_matrix().T @ f_local
        n1, n2 = el.node_tags
        model.add_nodal_load(n1, list(f_global[0:3]))
        model.add_nodal_load(n2, list(f_global[3:6]))
        n_done += 1
    return n_done
