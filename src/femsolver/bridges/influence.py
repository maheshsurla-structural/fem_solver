"""Influence lines and moving-load envelopes for bridge analysis.

An *influence line* gives the value of a structural response (reaction,
shear, or moment at a fixed section) as a unit point load moves
across the structure. For determinate beams the influence lines are
closed-form polynomials; for continuous or indeterminate structures,
they are obtained numerically via the Müller-Breslau principle (the
deflected shape produced by a unit virtual displacement at the
response location, with sign convention to match the response).

Moving-load analysis convolves a vehicle's wheel loads with the
influence line to determine the maximum response. Vehicles include
AASHTO HL-93 (design truck + tandem + lane) and IRC Class A/AA/70R.

This module provides:

* :func:`influence_line_simple_span_moment` -- M(x) influence for a
  simply-supported beam.
* :func:`influence_line_simple_span_shear` -- V(x) influence for a SS
  beam (sign convention: + just to the right of x).
* :class:`MovingLoad` -- a collection of axle loads + spacings.
* :func:`max_response_for_moving_load` -- sweep the moving load
  across the span and return the maximum response.
* :func:`max_truck_envelope_simple_span` -- end-to-end shortcut for
  HL-93 truck + tandem + lane envelope on a simple span.

References
----------
* AASHTO LRFD Bridge Design Specifications, 9e (2020), Sec. 3.6.
* IRC 6:2017, Sec. 204 (vehicle loads).
* Hibbeler, R.C. (2017). *Structural Analysis*, 10e. Pearson.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ============================================================ simple-span IL

def influence_line_simple_span_moment(
    *,
    L: float, x: float, xi: np.ndarray,
) -> np.ndarray:
    """Influence line for bending moment at section ``x`` on a SS span ``L``,
    evaluated at load positions ``xi``.

    Formula::
        M(x; xi) = xi (L - x) / L      for xi <= x
                 = x (L - xi) / L      for xi >  x

    Returns
    -------
    eta : np.ndarray
        Influence-line ordinates (m) -- so a unit load (N) at xi gives
        the moment eta · 1 (N·m).
    """
    if L <= 0.0:
        raise ValueError(f"L must be > 0, got {L}")
    if not (0.0 <= x <= L):
        raise ValueError(f"x = {x} must lie in [0, L = {L}]")
    xi = np.asarray(xi, dtype=float).ravel()
    if np.any((xi < 0.0) | (xi > L)):
        raise ValueError("all xi must lie in [0, L]")
    eta = np.where(
        xi <= x,
        xi * (L - x) / L,
        x * (L - xi) / L,
    )
    return eta


def influence_line_simple_span_shear(
    *,
    L: float, x: float, xi: np.ndarray,
) -> np.ndarray:
    """Influence line for shear at section ``x`` on a SS span ``L``.

    Sign convention: ``V`` positive when the resultant of forces to
    the LEFT of ``x`` acts upward (standard "left-up = positive").
    Formula::
        V(x; xi) = -xi / L        for xi <  x   (load to the left of x)
                 = (L - xi) / L   for xi >  x   (load to the right of x)
        V(x; x)  = +0.5/-0.5 (discontinuity); we return the right-limit
                   value here.
    """
    if L <= 0.0:
        raise ValueError("L must be > 0")
    if not (0.0 <= x <= L):
        raise ValueError(f"x must lie in [0, L = {L}]")
    xi = np.asarray(xi, dtype=float).ravel()
    eta = np.where(
        xi < x,
        -xi / L,
        (L - xi) / L,
    )
    return eta


# ============================================================ moving loads

@dataclass
class MovingLoad:
    """A train of point loads (axles) with spacings.

    Attributes
    ----------
    axle_loads : np.ndarray
        Axle weights (N), positive downward.
    axle_offsets : np.ndarray
        Distance of each axle from the LEFT-MOST axle (m). The
        leftmost has offset 0; offsets must be non-decreasing.
    name : str
    """

    axle_loads: np.ndarray
    axle_offsets: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        self.axle_loads = np.asarray(self.axle_loads, dtype=float).ravel()
        self.axle_offsets = np.asarray(self.axle_offsets, dtype=float).ravel()
        if self.axle_loads.size != self.axle_offsets.size:
            raise ValueError(
                "axle_loads and axle_offsets must have same length"
            )
        if self.axle_loads.size == 0:
            raise ValueError("need at least one axle")
        if not np.all(np.diff(self.axle_offsets) >= 0.0):
            raise ValueError("axle_offsets must be non-decreasing")
        if self.axle_offsets[0] != 0.0:
            raise ValueError("first axle offset must be 0")

    @property
    def total_length(self) -> float:
        """Span of the load train (distance from first to last axle)."""
        return float(self.axle_offsets[-1])

    @property
    def total_load(self) -> float:
        return float(np.sum(self.axle_loads))

    @classmethod
    def preset(cls, name: str) -> "MovingLoad":
        """Look up a named code vehicle by short string.

        Available names::

            "hl93_truck"   -- AASHTO HL-93 design truck
            "hl93_tandem"  -- AASHTO HL-93 design tandem
            "irc_class_a"  -- IRC Class A vehicle train
            "irc_70r"      -- IRC Class 70R tracked vehicle
        """
        if name not in _MOVING_LOAD_PRESETS:
            raise ValueError(
                f"unknown moving-load preset {name!r}; "
                f"available: {sorted(_MOVING_LOAD_PRESETS)}"
            )
        return _MOVING_LOAD_PRESETS[name]()


def evaluate_response_for_position(
    *,
    head_position: float,
    moving_load: MovingLoad,
    influence_line: callable,
    L: float,
    drop_off_span: bool = True,
) -> float:
    """Compute the structural response when the leftmost axle is at
    ``head_position`` along the span.

    Parameters
    ----------
    head_position : float
        Position of the leftmost axle (m).
    moving_load : MovingLoad
    influence_line : callable
        ``f(xi: np.ndarray) -> np.ndarray`` giving the IL ordinate(s).
    L : float
        Span length (m).
    drop_off_span : bool, default True
        Axles outside [0, L] contribute zero (have left the bridge).
    """
    axle_x = head_position + moving_load.axle_offsets
    if drop_off_span:
        mask = (axle_x >= 0.0) & (axle_x <= L)
    else:
        mask = np.ones_like(axle_x, dtype=bool)
    if not mask.any():
        return 0.0
    eta = influence_line(axle_x[mask])
    return float(np.sum(moving_load.axle_loads[mask] * eta))


def max_response_for_moving_load(
    *,
    moving_load: MovingLoad,
    influence_line: callable,
    L: float,
    n_positions: int = 401,
) -> tuple[float, float]:
    """Sweep the moving load across the structure and return
    ``(max_response, head_position_at_max)``.

    The leftmost axle is positioned at sample points from
    ``-train_length`` (load entering) to ``L`` (load leaving), so the
    full envelope is captured.

    Vectorised: builds the ``(n_positions, n_axles)`` matrix of axle
    positions, evaluates the IL once over the flattened array, masks
    off-bridge axles, and sums ``axle_loads · eta`` per position in a
    single matrix-vector product.
    """
    train_L = moving_load.total_length
    positions = np.linspace(-train_L, L, n_positions)
    # (n_positions, n_axles) matrix of every axle position to evaluate.
    axle_x = positions[:, None] + moving_load.axle_offsets[None, :]
    on_bridge = (axle_x >= 0.0) & (axle_x <= L)
    # Evaluate the IL over the flat array of on-bridge positions only;
    # for off-bridge cells store 0 (axle contributes nothing).
    eta = np.zeros_like(axle_x)
    if on_bridge.any():
        eta[on_bridge] = influence_line(axle_x[on_bridge])
    responses = eta @ moving_load.axle_loads
    i = int(np.argmax(np.abs(responses)))
    return float(responses[i]), float(positions[i])


# ============================================================ AASHTO HL-93

def aashto_hl93_truck() -> MovingLoad:
    """AASHTO HL-93 design truck (HS20-44 derivative).

    Three axles: 35 kN, 145 kN, 145 kN at spacings 4.3 m (front-to-middle)
    and a variable 4.3-9.0 m (middle-to-rear). We use the 4.3 m minimum
    for the maximum positive moment in simple spans (governs).
    """
    return MovingLoad(
        axle_loads=np.array([35.0e3, 145.0e3, 145.0e3]),
        axle_offsets=np.array([0.0, 4.3, 8.6]),
        name="AASHTO HL-93 design truck (4.3 m rear spacing)",
    )


def aashto_hl93_tandem() -> MovingLoad:
    """AASHTO HL-93 design tandem: 2 axles of 110 kN at 1.2 m spacing."""
    return MovingLoad(
        axle_loads=np.array([110.0e3, 110.0e3]),
        axle_offsets=np.array([0.0, 1.2]),
        name="AASHTO HL-93 design tandem",
    )


def aashto_hl93_lane_load_kN_per_m() -> float:
    """AASHTO HL-93 lane load: 9.34 kN/m (640 plf), uniformly distributed."""
    return 9.34e3


def aashto_lane_moment_simple_span(*, w: float, L: float, x: float) -> float:
    """Moment at section x on a SS span L under uniform lane load w (N/m).

    Integral of M influence line · w gives ``M = w x (L - x) / 2``.
    """
    return float(w * x * (L - x) / 2.0)


def max_truck_envelope_simple_span(
    *,
    L: float, x: float,
    impact_factor: float = 1.33,
) -> dict:
    """End-to-end AASHTO HL-93 envelope for moment at section ``x`` on
    a SS span of length ``L``.

    Combines:
        max[ truck + lane, tandem + lane ] · impact_factor.

    Parameters
    ----------
    impact_factor : float, default 1.33
        Dynamic load allowance per AASHTO 3.6.2 (33% for the truck).

    Returns
    -------
    dict
        ``{"M_truck_plus_lane": ..., "M_tandem_plus_lane": ...,
        "M_governing": ..., "M_with_impact": ...,
        "vehicle_governing": ...}``
    """
    def il_M(xi):
        return influence_line_simple_span_moment(L=L, x=x, xi=xi)

    M_truck, _ = max_response_for_moving_load(
        moving_load=aashto_hl93_truck(),
        influence_line=il_M, L=L,
    )
    M_tandem, _ = max_response_for_moving_load(
        moving_load=aashto_hl93_tandem(),
        influence_line=il_M, L=L,
    )
    M_lane = aashto_lane_moment_simple_span(
        w=aashto_hl93_lane_load_kN_per_m(), L=L, x=x,
    )
    M_t = M_truck + M_lane
    M_d = M_tandem + M_lane
    governing = "truck+lane" if M_t >= M_d else "tandem+lane"
    M_gov = max(M_t, M_d)
    return {
        "M_truck": float(M_truck),
        "M_tandem": float(M_tandem),
        "M_lane": float(M_lane),
        "M_truck_plus_lane": float(M_t),
        "M_tandem_plus_lane": float(M_d),
        "M_governing": float(M_gov),
        "M_with_impact": float(impact_factor * M_gov),
        "vehicle_governing": governing,
    }


# ============================================================ IRC vehicles

def irc_class_70r_truck() -> MovingLoad:
    """IRC Class 70R tracked vehicle (approximate equivalent as 7
    axles to keep the API uniform).

    A simplified 7-axle representation of the 70R tracked vehicle,
    total weight 700 kN over a 4.57 m base, 100 kN per axle.
    """
    n = 7
    base = 4.57
    loads = np.full(n, 100.0e3)
    offsets = np.linspace(0.0, base, n)
    return MovingLoad(
        axle_loads=loads, axle_offsets=offsets,
        name="IRC Class 70R (simplified 7-axle)",
    )


def irc_class_a() -> MovingLoad:
    """IRC Class A train: 2 driving + 2 front + bogie of 4 rear axles.

    Loads (kN): 27, 27, 114, 114, 68, 68, 68, 68.
    Spacings (m): 1.1 (front pair) -> 3.2 -> 1.2 (driving pair) -> 4.3
    -> 3.0 -> 3.0 -> 3.0 (rear bogie).
    """
    loads = np.array([27, 27, 114, 114, 68, 68, 68, 68]) * 1.0e3
    spacings = [0.0, 1.1, 3.2, 1.2, 4.3, 3.0, 3.0, 3.0]
    offsets = np.cumsum(spacings)
    return MovingLoad(
        axle_loads=loads, axle_offsets=offsets,
        name="IRC Class A vehicle train",
    )


# Name -> factory registry used by :meth:`MovingLoad.preset`.
_MOVING_LOAD_PRESETS = {
    "hl93_truck":  aashto_hl93_truck,
    "hl93_tandem": aashto_hl93_tandem,
    "irc_class_a": irc_class_a,
    "irc_70r":     irc_class_70r_truck,
}
