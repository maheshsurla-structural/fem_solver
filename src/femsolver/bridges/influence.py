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

from dataclasses import dataclass, field
from typing import Optional

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


# ============================================================ 2-D vehicles

@dataclass
class Vehicle2D:
    """A vehicle as a set of point wheel loads at in-plan ``(x, y)``
    positions, for placement on an :class:`InfluenceSurface`.

    ``x`` runs longitudinally (direction of travel), ``y`` transversely.
    The positions are relative to the vehicle's own reference point; the
    envelope helper slides that reference over the deck.

    Parameters
    ----------
    wheel_loads : array-like (k,)
        Wheel loads (N), positive magnitudes.
    wheel_xy : array-like (k, 2)
        ``(x, y)`` of each wheel relative to the reference point (m).
    name : str
    """

    wheel_loads: np.ndarray
    wheel_xy: np.ndarray
    name: str = "vehicle"

    def __post_init__(self) -> None:
        self.wheel_loads = np.asarray(self.wheel_loads, dtype=float).ravel()
        self.wheel_xy = np.asarray(self.wheel_xy, dtype=float)
        if self.wheel_xy.shape != (self.wheel_loads.size, 2):
            raise ValueError("wheel_xy must be (k, 2) matching wheel_loads")
        if self.wheel_loads.size == 0:
            raise ValueError("need at least one wheel")

    @classmethod
    def from_axle_train(cls, train: "MovingLoad", *, track_width: float = 1.8,
                        name: str | None = None) -> "Vehicle2D":
        """Build a 2-D vehicle from a 1-D axle train by splitting each axle
        into two wheels a ``track_width`` apart transversely (AASHTO wheel
        gauge = 1.8 m). The axle offsets become longitudinal positions."""
        offs = np.asarray(train.axle_offsets, dtype=float)
        loads = np.asarray(train.axle_loads, dtype=float)
        half = track_width / 2.0
        xy, w = [], []
        for x, p in zip(offs, loads):
            xy.append((x, +half)); w.append(p / 2.0)
            xy.append((x, -half)); w.append(p / 2.0)
        return cls(wheel_loads=np.array(w), wheel_xy=np.array(xy),
                   name=name or f"{train.name} (2-D)")


def moving_load_surface_envelope(
    surface,
    vehicle: Vehicle2D,
    *,
    x_positions=None,
    y_positions=None,
    n_x: int = 61,
    n_y: int = 21,
    both_directions: bool = True,
) -> dict:
    """Slide a 2-D vehicle over an influence surface and return the maximum
    and minimum response and the governing placement.

    The vehicle reference point is placed at each ``(X, Y)`` on the search
    grid; the response is ``sum_k P_k * IS(x_k + X, y_k + Y)`` (wheels off
    the deck contribute zero). By default the search grid spans the deck's
    bounding box; pass ``x_positions`` / ``y_positions`` to control it.

    Parameters
    ----------
    surface : InfluenceSurface
    vehicle : Vehicle2D
    x_positions, y_positions : array-like, optional
        Reference-point search stations (m). Default: ``n_x`` × ``n_y`` grid
        over the deck bounding box, expanded so the whole vehicle can enter
        and leave.
    both_directions : bool, default True
        Also test the vehicle reversed (governs for asymmetric vehicles).

    Returns
    -------
    dict with ``max``, ``min``, ``max_xy``, ``min_xy`` (governing reference
    positions), and ``vehicle``.
    """
    pts = surface.points
    if x_positions is None:
        wx = vehicle.wheel_xy[:, 0]
        x0 = float(pts[:, 0].min() - wx.max())
        x1 = float(pts[:, 0].max() - wx.min())
        x_positions = np.linspace(x0, x1, n_x)
    if y_positions is None:
        wy = vehicle.wheel_xy[:, 1]
        y0 = float(pts[:, 1].min() - wy.max())
        y1 = float(pts[:, 1].max() - wy.min())
        y_positions = np.linspace(y0, y1, n_y)

    trials = [vehicle]
    if both_directions:
        rev = Vehicle2D(wheel_loads=vehicle.wheel_loads.copy(),
                        wheel_xy=vehicle.wheel_xy * np.array([-1.0, 1.0]),
                        name=vehicle.name + " (reversed)")
        trials.append(rev)

    best_max, best_min = -np.inf, np.inf
    max_xy = min_xy = (0.0, 0.0)
    for veh in trials:
        for X in x_positions:
            for Y in y_positions:
                q = veh.wheel_xy + np.array([X, Y])
                r = float(np.dot(veh.wheel_loads, surface(q)))
                if r > best_max:
                    best_max, max_xy = r, (float(X), float(Y))
                if r < best_min:
                    best_min, min_xy = r, (float(X), float(Y))
    return {"max": best_max, "min": best_min,
            "max_xy": max_xy, "min_xy": min_xy, "vehicle": vehicle.name}


# ============================================================ multi-lane

# AASHTO LRFD Table 3.6.1.1.2-1 — multiple-presence factors m.
AASHTO_MULTI_PRESENCE = {1: 1.20, 2: 1.00, 3: 0.85}


def multi_presence_factor(n_loaded_lanes: int) -> float:
    """AASHTO LRFD multiple-presence factor ``m`` for ``n`` loaded lanes
    (Table 3.6.1.1.2-1): 1→1.20, 2→1.00, 3→0.85, ≥4→0.65."""
    if n_loaded_lanes < 1:
        raise ValueError("n_loaded_lanes must be >= 1")
    return AASHTO_MULTI_PRESENCE.get(n_loaded_lanes, 0.65)


def number_of_design_lanes(roadway_width: float, *,
                           lane_width: float = 3.6) -> int:
    """Number of design lanes for a clear roadway width (AASHTO LRFD
    §3.6.1.1.1): ``INT(w / 3.6 m)``, with the special rule that a width
    between 6.0 and 7.2 m carries **two** lanes of ``w/2``."""
    if roadway_width <= 0.0:
        return 0
    if 6.0 <= roadway_width < 7.2:
        return 2
    return max(1, int(roadway_width / lane_width))


@dataclass
class DesignLane:
    """A transverse design-lane band on the deck. The vehicle travels
    longitudinally along it and may be positioned transversely within the
    band (keeping wheels a ``lane_margin`` inside the edges).

    Parameters
    ----------
    y_center : float
        Transverse centreline of the lane (m).
    width : float, default 3.6
        Design-lane width (AASHTO 12 ft = 3.6 m).
    name : str
    """

    y_center: float
    width: float = 3.6
    name: str = "lane"

    @property
    def y_lo(self) -> float:
        return self.y_center - self.width / 2.0

    @property
    def y_hi(self) -> float:
        return self.y_center + self.width / 2.0


def generate_design_lanes(y_min: float, y_max: float, *,
                          lane_width: float = 3.6) -> list:
    """Tile ``N = number_of_design_lanes`` lanes across a roadway spanning
    ``[y_min, y_max]``, centred within the roadway. Within-lane transverse
    vehicle placement and lane-selection (in :func:`multi_lane_envelope`)
    then cover the AASHTO "position lanes to maximise" requirement."""
    w = float(y_max - y_min)
    n = number_of_design_lanes(w, lane_width=lane_width)
    if n == 0:
        return []
    used = n * lane_width
    start = y_min + (w - used) / 2.0            # centre the lane block
    return [DesignLane(y_center=start + (i + 0.5) * lane_width,
                       width=lane_width, name=f"lane {i + 1}")
            for i in range(n)]


def _lane_center_positions(lane: DesignLane, vehicle: Vehicle2D,
                           lane_margin: float, n_y: int) -> np.ndarray:
    """Allowed transverse positions for the vehicle *reference* so the
    outermost wheels stay ``lane_margin`` inside the lane edges."""
    wy = vehicle.wheel_xy[:, 1]
    lo = lane.y_lo + lane_margin - float(wy.min())
    hi = lane.y_hi - lane_margin - float(wy.max())
    if hi < lo:                                  # vehicle wider than the band
        return np.array([lane.y_center - 0.5 * float(wy.min() + wy.max())])
    return np.linspace(lo, hi, max(1, n_y))


def multi_lane_envelope(
    surface,
    vehicle: Vehicle2D,
    lanes,
    *,
    n_x: int = 61,
    n_y: int = 9,
    lane_margin: float = 0.6,
    both_directions: bool = True,
    multi_presence: bool = True,
) -> dict:
    """Governing multi-lane moving-load response with AASHTO multiple-presence.

    Each lane's vehicle is placed to maximise (and minimise) its own
    contribution — longitudinally over the deck, transversely within the lane
    band. The governing response then loads the ``k`` most-favourable lanes and
    applies the multiple-presence factor ``m(k)``; the analysis reports the
    ``k`` (and factor) that governs, searching ``k = 1 … N``.

    ``response_max = max_k [ m(k) · Σ (k largest lane contributions) ]`` and
    symmetrically for the minimum.

    Returns
    -------
    dict with ``max`` / ``min`` (factored governing responses),
    ``max_num_lanes`` / ``min_num_lanes``, ``max_factor`` / ``min_factor``,
    and ``per_lane`` (each lane's unfactored max/min contribution + position).
    """
    lanes = list(lanes)
    if not lanes:
        raise ValueError("provide at least one design lane")

    per_lane = []
    for lane in lanes:
        ys = _lane_center_positions(lane, vehicle, lane_margin, n_y)
        env = moving_load_surface_envelope(
            surface, vehicle, y_positions=ys, n_x=n_x,
            both_directions=both_directions)
        per_lane.append({"lane": lane.name, "max": env["max"],
                         "min": env["min"], "max_xy": env["max_xy"],
                         "min_xy": env["min_xy"]})

    def _govern(values, reverse):
        # add lanes best-first; m(k) penalises loading more lanes
        ordered = sorted(values, reverse=reverse)
        best, best_k, best_m, cum = (None, 0, 1.0, 0.0)
        for k, v in enumerate(ordered, start=1):
            cum += v
            m = multi_presence_factor(k) if multi_presence else 1.0
            tot = m * cum
            if best is None or (tot > best if reverse else tot < best):
                best, best_k, best_m = tot, k, m
        return best, best_k, best_m

    max_v, max_k, max_m = _govern([d["max"] for d in per_lane], reverse=True)
    min_v, min_k, min_m = _govern([d["min"] for d in per_lane], reverse=False)
    return {"max": max_v, "max_num_lanes": max_k, "max_factor": max_m,
            "min": min_v, "min_num_lanes": min_k, "min_factor": min_m,
            "per_lane": per_lane, "vehicle": vehicle.name}


def lane_load_surface_envelope(
    surface,
    lanes,
    pressure: float,
    *,
    loaded_width: float = 3.0,
    x_range=None,
    nx: int = 81,
    ny: int = 9,
    multi_presence: bool = True,
) -> dict:
    """Governing multi-lane **uniform lane load** response with multiple
    presence. Integrates the influence surface over each lane's loaded strip
    (``loaded_width``, AASHTO 3.0 m) times the ``pressure`` (force/area — for
    the AASHTO 9.3 kN/m design lane load use 9.3/3.0 = 3.1 kN/m²), keeps the
    positive area for the maximum and the negative area for the minimum, then
    combines lanes with ``m(k)`` exactly as :func:`multi_lane_envelope`."""
    lanes = list(lanes)
    if x_range is None:
        x_range = (float(surface.points[:, 0].min()),
                   float(surface.points[:, 0].max()))
    x0, x1 = x_range
    per_lane = []
    for lane in lanes:
        y0 = lane.y_center - loaded_width / 2.0
        y1 = lane.y_center + loaded_width / 2.0
        a_pos = surface.integrate_patch(x0, x1, y0, y1, sign="positive",
                                        nx=nx, ny=ny)
        a_neg = surface.integrate_patch(x0, x1, y0, y1, sign="negative",
                                        nx=nx, ny=ny)
        per_lane.append({"lane": lane.name, "max": pressure * a_pos,
                         "min": pressure * a_neg})

    def _govern(values, reverse):
        ordered = sorted(values, reverse=reverse)
        best, best_k, cum = None, 0, 0.0
        for k, v in enumerate(ordered, start=1):
            cum += v
            m = multi_presence_factor(k) if multi_presence else 1.0
            tot = m * cum
            if best is None or (tot > best if reverse else tot < best):
                best, best_k = tot, k
        return best, best_k

    max_v, max_k = _govern([d["max"] for d in per_lane], reverse=True)
    min_v, min_k = _govern([d["min"] for d in per_lane], reverse=False)
    return {"max": max_v, "max_num_lanes": max_k,
            "min": min_v, "min_num_lanes": min_k, "per_lane": per_lane}
