"""Beam-on-Winkler-foundation (BOEF / BOWF) support.

A mat / strip footing resting on a continuous elastic bed of modulus
``k_s`` (force / area / length-deflection -- units N/m^3 if working
per unit footprint area; engineers also tabulate ``k_s`` as
N/m^2 / m = N/m^3) is modelled by adding a distributed transverse
spring term to the beam-bending equation:

    EI w'''' + k_s b w = q(x),

where ``b`` is the in-plane footing width.

At the FE level, the Winkler bed contributes an extra stiffness
matrix block::

    K_w_local = k_s · b · ∫₀ᴸ N_w^T N_w dx

with ``N_w`` the 4-component Hermite cubic shape functions for the
transverse-displacement / end-rotation DOFs ``(v_i, theta_i,
v_j, theta_j)``. The exact integral is::

    K_w_local = (k_s · b · L / 420) ·
        [[ 156,   22 L,    54,  -13 L ],
         [  22 L,   4 L^2,  13 L, -3 L^2 ],
         [   54,   13 L,   156,  -22 L  ],
         [ -13 L,  -3 L^2, -22 L,  4 L^2 ]]

This module ships:

* :class:`BeamOnWinklerFoundation2D` -- a thin subclass of
  :class:`~femsolver.elements.beam.BeamColumn2D` that augments
  ``K_global`` with the Winkler block.
* :func:`hetenyi_characteristic_length` -- ``L_c = (4 E I / (k_s b))^(1/4)``.
* :func:`hetenyi_infinite_beam_point_load` -- closed-form deflection
  for an infinite beam under a concentrated load (verification
  reference for the FE element).
* :func:`subgrade_modulus_table` -- typical k_s ranges for sand /
  clay / rock per Bowles 1996 (a useful lookup for first-pass
  design).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from femsolver.elements.beam import BeamColumn2D


# ============================================================ Hermite N^T N integral

def _hermite_NTN_block(L: float) -> np.ndarray:
    """4 x 4 integral ``∫₀ᴸ N_w^T N_w dx`` for Hermite cubic shape
    functions in beam DOF order ``(v_i, theta_i, v_j, theta_j)``.

    Returned matrix is already multiplied by ``L/420``.
    """
    return (L / 420.0) * np.array([
        [156.0,    22.0 * L,    54.0,   -13.0 * L],
        [22.0 * L,  4.0 * L * L, 13.0 * L, -3.0 * L * L],
        [54.0,     13.0 * L,   156.0,  -22.0 * L],
        [-13.0 * L,-3.0 * L * L,-22.0 * L,  4.0 * L * L],
    ])


# ============================================================ Winkler beam

class BeamOnWinklerFoundation2D(BeamColumn2D):
    """``BeamColumn2D`` augmented with a continuous Winkler bed.

    Parameters
    ----------
    tag, nodes, material, area, Iz :
        As for :class:`~femsolver.elements.beam.BeamColumn2D`.
    k_s : float
        Modulus of subgrade reaction (N/m^3 or N/m^2 / m).
    b : float
        Footing in-plane width (m). For a beam, this is the width
        across which the Winkler bed acts.
    """

    def __init__(
        self,
        tag: int,
        nodes,
        material,
        area: float | None = None,
        Iz: float | None = None,
        *,
        section=None,
        k_s: float,
        b: float,
    ):
        super().__init__(tag, nodes, material, area, Iz, section=section)
        if k_s <= 0.0:
            raise ValueError(f"k_s must be > 0, got {k_s}")
        if b <= 0.0:
            raise ValueError(f"b must be > 0, got {b}")
        self.k_s = float(k_s)
        self.b = float(b)

    def K_local(self) -> np.ndarray:
        """Local stiffness with the Winkler block added on the
        transverse-rotation DOFs.

        Beam local DOF order: ``(u_i, v_i, theta_i, u_j, v_j, theta_j)``.
        Winkler affects only ``(v_i, theta_i, v_j, theta_j)``
        (rows/cols 1, 2, 4, 5 in zero-indexed terms).
        """
        K = super().K_local()
        L, _, _ = self.length_and_angle()
        K_w = self.k_s * self.b * _hermite_NTN_block(L)
        idx = [1, 2, 4, 5]
        for i, ii in enumerate(idx):
            for j, jj in enumerate(idx):
                K[ii, jj] += K_w[i, j]
        return K


# ============================================================ closed-form refs

@dataclass
class HetenyiInfiniteBeamResult:
    """Closed-form result for an infinite beam on a Winkler bed under a
    concentrated point load ``P`` at ``x = 0``.

    Attributes
    ----------
    L_c : float
        Hetenyi characteristic length ``(4 E I / (k_s b))^(1/4)`` (m).
    w_max : float
        Deflection directly under the load (m).
    M_max : float
        Bending moment under the load (N·m).
    V_max : float
        Shear just to one side of the load (N).
    """

    L_c: float
    w_max: float
    M_max: float
    V_max: float


def hetenyi_characteristic_length(
    *, E: float, I: float, k_s: float, b: float,
) -> float:
    """Hetenyi characteristic length ``L_c = (4 E I / (k_s b))^(1/4)``.

    The beam is considered "long" when its actual length ``L > pi L_c``;
    "rigid" when ``L < pi L_c / 4``.
    """
    if E <= 0.0 or I <= 0.0 or k_s <= 0.0 or b <= 0.0:
        raise ValueError("E, I, k_s, b must be > 0")
    return float((4.0 * E * I / (k_s * b)) ** 0.25)


def hetenyi_infinite_beam_point_load(
    *, P: float, E: float, I: float, k_s: float, b: float,
) -> HetenyiInfiniteBeamResult:
    """Closed-form max-deflection / max-moment for an infinite beam
    on a Winkler bed under a concentrated load.

    Hetenyi (1946):

        w(0)  = P beta / (2 k_s b),
        M(0)  = P / (4 beta),
        V(0+) = -P / 2,

    where ``beta = 1 / L_c = (k_s b / (4 E I))^(1/4)``.

    These are the canonical references used for verification of the
    FE :class:`BeamOnWinklerFoundation2D` element.
    """
    if P <= 0.0:
        raise ValueError("P must be > 0")
    L_c = hetenyi_characteristic_length(E=E, I=I, k_s=k_s, b=b)
    beta = 1.0 / L_c
    w0 = P * beta / (2.0 * k_s * b)
    M0 = P / (4.0 * beta)
    V0 = P / 2.0
    return HetenyiInfiniteBeamResult(
        L_c=float(L_c), w_max=float(w0), M_max=float(M0), V_max=float(V0),
    )


# ============================================================ subgrade-modulus table

# Bowles 1996 / Coduto 2001 typical ranges (MN/m^3).
_K_S_TABLE = {
    "loose_sand":        (4.8, 16.0),
    "medium_sand":       (9.6, 80.0),
    "dense_sand":        (64.0, 128.0),
    "clayey_sand":       (32.0, 80.0),
    "silty_sand":        (24.0, 48.0),
    "stiff_clay_dry":    (12.0, 24.0),
    "stiff_clay_wet":    (24.0, 48.0),
    "very_stiff_clay":   (48.0, 96.0),
    "hard_clay":         (96.0, 200.0),
    "weathered_rock":    (200.0, 800.0),
    "sound_rock":        (800.0, 5000.0),
}


def subgrade_modulus_table(soil_type: str) -> tuple[float, float]:
    """Typical modulus-of-subgrade-reaction range (low, high) in N/m^3.

    Source: Bowles 1996 / Coduto 2001 -- order-of-magnitude design
    values for first-pass selection. For final design, calibrate
    against plate-load test data.

    Parameters
    ----------
    soil_type : str
        One of: ``"loose_sand"``, ``"medium_sand"``, ``"dense_sand"``,
        ``"clayey_sand"``, ``"silty_sand"``, ``"stiff_clay_dry"``,
        ``"stiff_clay_wet"``, ``"very_stiff_clay"``, ``"hard_clay"``,
        ``"weathered_rock"``, ``"sound_rock"``.

    Returns
    -------
    (low, high) : tuple of floats (N/m^3)
    """
    if soil_type not in _K_S_TABLE:
        raise ValueError(
            f"unknown soil_type {soil_type!r}; available: "
            f"{sorted(_K_S_TABLE)}"
        )
    lo_MN, hi_MN = _K_S_TABLE[soil_type]
    return float(lo_MN * 1.0e6), float(hi_MN * 1.0e6)
