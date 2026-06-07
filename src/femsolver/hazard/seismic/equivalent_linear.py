"""Equivalent-linear (SHAKE-style) site response with G/G_max and
damping curves.

Under strong shaking, soil shear modulus G decreases dramatically and
damping xi increases, both as functions of cyclic shear strain
gamma. The *equivalent-linear* method (Schnabel-Lysmer-Seed 1972,
SHAKE) uses a strain-compatible iteration:

1. Start with elastic G_max and small-strain damping xi_min.
2. Run the linear transfer-function analysis to estimate peak shear
   strain in each layer.
3. Use the **effective strain** gamma_eff = R * gamma_max (R ~ 0.65)
   to look up updated G/G_max and damping from soil-specific curves.
4. Repeat until G and xi converge layer-by-layer.

This module ships:

* :class:`NonlinearSoilCurves` -- Vucetic-Dobry 1991 G/G_max and
  damping curves indexed by plasticity index (PI). Hyperbolic form
  ``G/G_max = 1 / (1 + (gamma/gamma_r)^alpha)`` with ``gamma_r(PI)``.
* :func:`vucetic_dobry_curves` -- factory returning standard curves
  for a given PI.
* :func:`equivalent_linear_iterate` -- main driver that takes a
  layered profile and a target input PGA, returns the converged
  ``(G_eff, xi_eff)`` per layer plus the surface amplification.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from femsolver.hazard.seismic.site_response import SoilLayer, SoilProfile


@dataclass
class NonlinearSoilCurves:
    """G/G_max and damping curves for one soil type.

    Attributes
    ----------
    gamma_r : float
        Reference shear strain (G/G_max = 0.5 at gamma = gamma_r).
        Vucetic-Dobry 1991:
        - PI = 0:   ~1.8e-4
        - PI = 15:  ~2.9e-4
        - PI = 30:  ~5.0e-4
        - PI = 50:  ~8.1e-4
        - PI = 100: ~1.7e-3
        - PI = 200: ~4.5e-3
    alpha : float, default 0.92
        Hyperbolic-curve shape exponent (Darendeli 2001 best fit).
    xi_min : float, default 0.01
        Small-strain damping (1% typical for sand, 1-2% for clay).
    xi_max : float, default 0.25
        Large-strain (asymptotic) damping (~25-30% for fully cyclic).
    damping_exponent : float, default 0.5
        Power-law exponent on ``(1 - G/G_max)`` in the damping curve.
    name : str
        Free-text label.
    """
    gamma_r: float
    alpha: float = 0.92
    xi_min: float = 0.01
    xi_max: float = 0.25
    damping_exponent: float = 0.5
    name: str = ""

    def G_over_Gmax(self, gamma: float) -> float:
        """G/G_max at the supplied cyclic shear strain (positive)."""
        if gamma <= 0:
            return 1.0
        ratio = (gamma / self.gamma_r) ** self.alpha
        return 1.0 / (1.0 + ratio)

    def damping(self, gamma: float) -> float:
        """Damping ratio at the supplied cyclic shear strain."""
        gG = self.G_over_Gmax(gamma)
        return float(self.xi_min + (self.xi_max - self.xi_min)
                      * (1.0 - gG) ** self.damping_exponent)


def vucetic_dobry_curves(PI: float) -> NonlinearSoilCurves:
    """Vucetic-Dobry 1991 nonlinear curves for plasticity index ``PI``.

    Interpolates ``gamma_r`` between the published PI = 0, 15, 30, 50,
    100, 200 anchor points; clay (higher PI) has a larger gamma_r and
    thus stays linear at higher strains.

    Parameters
    ----------
    PI : float
        Plasticity index (%). 0 = sand, 100+ = high-plasticity clay.
    """
    if PI < 0:
        raise ValueError(f"PI must be non-negative, got {PI}")
    # Published anchor data (PI, gamma_r)
    anchors = [
        (0.0, 1.8e-4),
        (15.0, 2.9e-4),
        (30.0, 5.0e-4),
        (50.0, 8.1e-4),
        (100.0, 1.7e-3),
        (200.0, 4.5e-3),
    ]
    if PI >= anchors[-1][0]:
        gamma_r = anchors[-1][1]
    elif PI <= anchors[0][0]:
        gamma_r = anchors[0][1]
    else:
        # Log-linear interpolation in PI -> gamma_r
        for i in range(len(anchors) - 1):
            PI0, gr0 = anchors[i]
            PI1, gr1 = anchors[i + 1]
            if PI0 <= PI <= PI1:
                w = (PI - PI0) / (PI1 - PI0)
                gamma_r = math.exp(
                    math.log(gr0) + w * (math.log(gr1) - math.log(gr0))
                )
                break
    # Damping: small-strain rises slightly with PI (Vucetic-Dobry)
    xi_min = max(0.005, 0.005 + PI * 5.0e-5)    # 0.5% at PI=0, ~1.5% at PI=200
    return NonlinearSoilCurves(
        gamma_r=gamma_r,
        alpha=0.92,
        xi_min=xi_min,
        xi_max=0.25,
        damping_exponent=0.55,
        name=f"Vucetic-Dobry PI={PI:.0f}",
    )


@dataclass
class EquivalentLinearResult:
    """Output of the equivalent-linear iteration."""
    converged: bool
    iterations: int
    G_eff: np.ndarray              # (n_layers,) effective shear modulus
    G_over_Gmax: np.ndarray         # (n_layers,) ratios
    xi_eff: np.ndarray              # (n_layers,) effective damping ratios
    gamma_eff: np.ndarray           # (n_layers,) effective shear strains
    surface_amplification: float    # ratio of surface PGA to input PGA
    Vs_eff: np.ndarray              # (n_layers,) effective shear-wave velocity


def equivalent_linear_iterate(
    layers: list,
    rock_Vs: float,
    rock_rho: float,
    curves: list,
    *,
    input_pga: float = 0.1,
    R_factor: float = 0.65,
    max_iter: int = 25,
    tol: float = 0.02,
) -> EquivalentLinearResult:
    """SHAKE-style iteration to strain-compatible G_eff and xi_eff.

    Parameters
    ----------
    layers : sequence of SoilLayer
        Top layer first. ``layer.Vs`` is the small-strain (G_max)
        shear-wave velocity.
    rock_Vs, rock_rho : float
        Bedrock properties.
    curves : sequence of :class:`NonlinearSoilCurves`
        One per layer (same order). Each must implement
        ``G_over_Gmax(gamma)`` and ``damping(gamma)``.
    input_pga : float, default 0.1
        Estimated peak input acceleration at bedrock (units of g).
        Used to estimate layer strains via PGA * H_amp / Vs.
    R_factor : float, default 0.65
        Effective-strain reduction factor (typ. 0.65 per SHAKE convention).
    max_iter : int, default 25
    tol : float, default 0.02
        Convergence tolerance on max relative change in G_eff.
    """
    if len(layers) != len(curves):
        raise ValueError(
            f"layers ({len(layers)}) and curves ({len(curves)}) must "
            "have the same length"
        )
    if input_pga <= 0:
        raise ValueError(f"input_pga must be positive, got {input_pga}")
    n_lay = len(layers)
    # Initialize at small-strain (G_max)
    G_max = np.array([lyr.rho * lyr.Vs ** 2 for lyr in layers])
    G_eff = G_max.copy()
    xi_eff = np.array([c.xi_min for c in curves])
    G_ratio = np.ones(n_lay)
    gamma_eff = np.zeros(n_lay)
    converged = False
    iters = 0
    surface_amp = 1.0
    # Frequency grid for the transfer function (engineering 0-15 Hz)
    freqs = np.linspace(0.1, 15.0, 200)
    for it in range(1, max_iter + 1):
        iters = it
        # Build a transient profile with current G_eff, xi_eff
        eff_layers = []
        for j, (lyr, c) in enumerate(zip(layers, curves)):
            Vs_eff = math.sqrt(G_eff[j] / lyr.rho)
            eff_layers.append(SoilLayer(
                thickness=lyr.thickness,
                Vs=Vs_eff,
                rho=lyr.rho,
                damping=xi_eff[j],
            ))
        prof = SoilProfile(
            layers=eff_layers, rock_Vs=rock_Vs, rock_rho=rock_rho,
        )
        from femsolver.hazard.seismic.site_response import (
            site_amplification_spectrum,
        )
        amp = site_amplification_spectrum(prof, freqs)
        # Layer-strain estimate: use Vucetic's simplified formula
        #   gamma_eff[j] = R * (PGA_top[j] * g) / (omega_eff * Vs_eff[j])
        # where omega_eff ~ 2*pi * f_dominant. We use the peak-amplification
        # frequency as a proxy.
        idx_peak = int(np.argmax(amp))
        f_dom = float(freqs[idx_peak])
        amp_peak = float(amp[idx_peak])
        # Approximate strain in each layer using a kinematic estimate:
        # gamma_j = R * acceleration / (omega^2 * H_to_bedrock_j * Vs_eff_j)
        # Simpler: gamma_j ~ R * input_pga * 9.81 * amp_layer_j / Vs_eff_j^2 / depth_j
        # Use a clean form: peak velocity v ~ PGA * g / (2 pi f)
        # gamma ~ v / Vs
        v_input = input_pga * 9.81 / (2.0 * math.pi * f_dom)
        v_surface = v_input * amp_peak
        new_gamma_eff = np.zeros(n_lay)
        for j in range(n_lay):
            Vs_eff_j = math.sqrt(G_eff[j] / layers[j].rho)
            # Layer strain decays with depth from the surface
            depth_frac = (j + 1) / n_lay
            # Estimate layer peak velocity as fraction of surface velocity
            v_layer = v_surface * (1.0 - 0.5 * depth_frac)
            new_gamma_eff[j] = R_factor * abs(v_layer) / Vs_eff_j
        # Update G/G_max and damping from the curves
        new_G_ratio = np.array([
            curves[j].G_over_Gmax(new_gamma_eff[j])
            for j in range(n_lay)
        ])
        new_G_eff = G_max * new_G_ratio
        new_xi = np.array([
            curves[j].damping(new_gamma_eff[j])
            for j in range(n_lay)
        ])
        # Convergence check on G_eff
        diff = np.abs(new_G_eff - G_eff) / np.maximum(G_eff, 1e-30)
        max_diff = float(np.max(diff))
        # Under-relaxation factor for stability (~0.5)
        relax = 0.5
        G_eff = G_eff + relax * (new_G_eff - G_eff)
        xi_eff = xi_eff + relax * (new_xi - xi_eff)
        G_ratio = G_eff / G_max
        gamma_eff = new_gamma_eff
        surface_amp = amp_peak
        if max_diff < tol:
            converged = True
            break
    return EquivalentLinearResult(
        converged=converged,
        iterations=iters,
        G_eff=G_eff.copy(),
        G_over_Gmax=G_ratio.copy(),
        xi_eff=xi_eff.copy(),
        gamma_eff=gamma_eff.copy(),
        surface_amplification=float(surface_amp),
        Vs_eff=np.array([math.sqrt(G_eff[j] / layers[j].rho)
                          for j in range(n_lay)]),
    )
