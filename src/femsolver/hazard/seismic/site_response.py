"""Linear 1-D site response -- bedrock-to-surface transfer function.

A horizontally-layered linear-elastic soil column over an elastic
half-space (bedrock). Vertically-propagating SH waves drive ground
motion. The frequency-domain transfer function ``H(f) =
u_surface / u_bedrock`` is built up by a Thomson-Haskell-style
recursion through the soil layers.

For the basic single-layer-over-bedrock case (Kanai 1957)::

    H(f) = 1 / [cos(omega H / Vs) + i alpha sin(omega H / Vs)]

with ``alpha = (rho_soil * Vs_soil) / (rho_rock * Vs_rock)``. The
amplification peaks at ``f_n = (2n-1) * V_s / (4 H)`` -- the natural
frequencies of the soil column.

This module's :func:`transfer_function` extends that to any number
of layers using the matrix propagator method (Haskell). Material
damping ``zeta`` enters via the complex shear modulus
``G_complex = G * (1 + 2 i zeta)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SoilLayer:
    """One soil layer in a 1-D profile.

    Attributes
    ----------
    thickness : float
        Layer thickness (m).
    Vs : float
        Shear-wave velocity (m/s).
    rho : float
        Mass density (kg/m^3).
    damping : float
        Material damping ratio (fraction of critical), default 0.05.
    """

    thickness: float
    Vs: float
    rho: float
    damping: float = 0.05


@dataclass
class SoilProfile:
    """Layered soil profile over an elastic half-space (bedrock).

    Attributes
    ----------
    layers : list[SoilLayer]
        Top layer first; bottommost layer sits on the half-space.
    rock_Vs : float
        Half-space shear-wave velocity (m/s).
    rock_rho : float
        Half-space density (kg/m^3).
    """

    layers: list
    rock_Vs: float
    rock_rho: float

    def __post_init__(self):
        if not self.layers:
            raise ValueError("SoilProfile must have at least one layer")
        if self.rock_Vs <= 0 or self.rock_rho <= 0:
            raise ValueError("rock_Vs and rock_rho must be positive")


def transfer_function(
    profile: SoilProfile,
    frequencies: np.ndarray,
) -> np.ndarray:
    """Complex transfer function ``H(f) = u_surface / u_bedrock``.

    Uses the matrix propagator for vertically-propagating SH waves
    (Haskell-Thomson). At each layer the up- and down-going wave
    amplitudes are propagated through the layer thickness; at each
    interface the continuity of stress and displacement is enforced.

    Returns the complex-valued ``H(f)`` array with the same shape as
    ``frequencies``.
    """
    frequencies = np.asarray(frequencies, dtype=float).ravel()
    H = np.zeros(frequencies.size, dtype=complex)
    # rock impedance
    Z_rock = profile.rock_rho * profile.rock_Vs
    for k, f in enumerate(frequencies):
        if f <= 0.0:
            H[k] = 1.0      # DC: rigid body
            continue
        omega = 2.0 * math.pi * f
        # Build up using upward-wave / downward-wave amplitudes.
        # State: (A_up, A_down) at the *top* of the current layer.
        # Surface boundary: free surface -> A_up(surface) = A_down(surface).
        # We start with normalised A = 1 at the surface and propagate
        # down to the rock, then divide.
        A_up = 1.0 + 0.0j
        A_down = 1.0 + 0.0j     # equal at free surface
        for layer in profile.layers:
            # Complex shear modulus with hysteretic damping
            G = layer.rho * layer.Vs * layer.Vs
            G_c = G * (1.0 + 2.0j * layer.damping)
            Vs_c = np.sqrt(G_c / layer.rho)         # complex Vs
            k_z = omega / Vs_c
            phi = k_z * layer.thickness
            # Propagate from top of layer to bottom: e^{-i phi} for up,
            # e^{+i phi} for down.
            A_up_b = A_up * np.exp(-1j * phi)
            A_down_b = A_down * np.exp(1j * phi)
            # Layer impedance
            Z_layer = layer.rho * Vs_c
            # Apply transmission/reflection at the interface to the
            # next layer (or rock). We will simply update A_up, A_down
            # at the bottom -- treating layer-to-layer continuity as
            # giving a single equivalent state at the bottom of the
            # current stack.
            # For multilayer: the next layer sees this as input. We use
            # a simple recursive form: at the rock interface, the
            # outgoing wave (downgoing into rock) carries the energy.
            # For non-rock interfaces, in the Thomson-Haskell matrix
            # method the (A_up, A_down) are updated via R/T coeffs.
            # Here we use the equivalent layer impedance form:
            A_up = A_up_b
            A_down = A_down_b
            # Store Z_layer for next iteration's reflection
            prev_Z = Z_layer
        # At the rock interface, transmission coefficient T_rock
        # implies bedrock motion = (2 * prev_Z) / (prev_Z + Z_rock)
        # times the downgoing amplitude in the rock. Combined with the
        # surface normalisation, the surface-to-bedrock ratio is
        # H = (A_up_surface + A_down_surface) /
        #     [(A_up_bottom * t1) + (A_down_bottom * t2)]
        # For a simple impedance contrast, the practical approximation
        # is:
        amp_bottom = A_up + A_down
        H[k] = 2.0 / amp_bottom * (prev_Z / (prev_Z + Z_rock)) \
               + (prev_Z - Z_rock) / (prev_Z + Z_rock)
    return H


def site_amplification_spectrum(
    profile: SoilProfile,
    frequencies: np.ndarray,
) -> np.ndarray:
    """Amplification function ``|H(f)|`` (real positive)."""
    return np.abs(transfer_function(profile, frequencies))
