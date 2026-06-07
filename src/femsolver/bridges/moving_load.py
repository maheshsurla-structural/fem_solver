"""General moving-load / influence-line engine (Phase B.1).

The closed-form influence lines in :mod:`femsolver.bridges.influence`
only cover a single simply-supported span. Real bridges -- continuous
girders, frames, grillages, integral abutments -- need influence
lines built **directly from the assembled finite-element model**.

This module provides that, via the classical *unit-load traversal*
(Müller-Breslau in its most literal, brute-force form):

    For each position along a defined **lane**, place a unit load on
    the structure, solve, and record the response of interest. The
    collection of (position, response) pairs IS the influence line for
    that response.

Because influence lines are a linear-superposition concept they are
only meaningful for a linear-elastic structure, so the engine always
uses a single linear solve per position. To make the traversal cheap,
:class:`InfluenceLineEngine` assembles the global stiffness **once**
and factorises it **once** (``scipy.sparse.linalg.splu``); each lane
position is then a back-substitution against a unit right-hand side.
For a model with *N* lane points this costs one factorisation plus
*N* back-substitutions, instead of *N* full solves.

Workflow
--------
::

    from femsolver.bridges.moving_load import (
        InfluenceLineEngine, Lane, BeamForce, Reaction,
        aashto_hl93_envelope,
    )

    # 1. Build the engine on any model (continuous girder, frame, ...)
    engine = InfluenceLineEngine(model)

    # 2. Define the lane the load travels along (ordered node tags)
    lane = Lane(node_tags=girder_nodes, load_dof=1)   # vertical

    # 3. Ask for influence lines of any responses, in ONE traversal
    ils = engine.influence_lines(lane, {
        "M_mid":  BeamForce(element_tag=midspan_elem, component="M", end="j"),
        "R_left": Reaction(node_tag=1, dof=1),
    })

    # 4. Run code vehicles over the IL to get the design envelope
    env = aashto_hl93_envelope(ils["M_mid"])
    print(env["max"], env["min"])

The :class:`InfluenceLine` produced is itself a callable
``il(x) -> ordinate`` (linear interpolation, zero off the lane), so it
plugs directly into the existing vehicle-convolution helpers in
:mod:`femsolver.bridges.influence`.

Sign convention
---------------
The influence-line ordinate is the response produced by a unit load
applied in the lane's gravity direction (``load_dof`` with
``gravity_sign``, default a downward unit load). Vehicle axle loads
are positive magnitudes, so a vehicle response is simply
``sum(P_k * il(x_k))``.

References
----------
* Müller-Breslau, H. (1886). *Die graphische Statik der Baukonstruktionen.*
* Hibbeler, R.C. *Structural Analysis* -- influence lines for statically
  indeterminate structures (Müller-Breslau principle).
* AASHTO LRFD Bridge Design Specifications, §3.6.1 (HL-93 live load,
  dynamic load allowance IM).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from femsolver.analysis.assembler import assemble_reactions, assemble_stiffness
from femsolver.analysis.linear_static import LinearStaticAnalysis
from femsolver.bridges.influence import MovingLoad
from femsolver.results.diagrams import beam_force_diagram


# ============================================================ InfluenceLine

@dataclass
class InfluenceLine:
    """An influence line: response ordinate vs load position.

    Attributes
    ----------
    stations : np.ndarray
        Load positions along the lane (m), strictly increasing.
    values : np.ndarray
        Response ordinate at each station (response per unit downward
        load placed at that station).
    response_name : str
    lane_name : str

    The object is callable: ``il(x)`` linearly interpolates the
    ordinate, returning 0 for positions outside the lane extent
    (the traversing axle has left the structure).
    """

    stations: np.ndarray
    values: np.ndarray
    response_name: str = ""
    lane_name: str = ""

    def __post_init__(self) -> None:
        s = np.asarray(self.stations, dtype=float).ravel()
        v = np.asarray(self.values, dtype=float).ravel()
        if s.size != v.size:
            raise ValueError("stations and values must have equal length")
        if s.size < 2:
            raise ValueError("influence line needs at least 2 stations")
        order = np.argsort(s)
        self.stations = s[order]
        self.values = v[order]

    def __call__(self, x) -> np.ndarray:
        """Interpolate the ordinate at position(s) ``x``. Positions
        outside ``[stations[0], stations[-1]]`` return 0."""
        x_arr = np.asarray(x, dtype=float)
        out = np.interp(
            x_arr, self.stations, self.values, left=0.0, right=0.0
        )
        return out

    # ---- scalar extrema -------------------------------------------------
    @property
    def max_value(self) -> float:
        return float(np.max(self.values))

    @property
    def min_value(self) -> float:
        return float(np.min(self.values))

    @property
    def max_station(self) -> float:
        return float(self.stations[int(np.argmax(self.values))])

    @property
    def min_station(self) -> float:
        return float(self.stations[int(np.argmin(self.values))])

    @property
    def span(self) -> float:
        return float(self.stations[-1] - self.stations[0])

    # ---- integration (lane / uniform loads) -----------------------------
    def integrate(
        self,
        *,
        sign: str = "all",
        x0: Optional[float] = None,
        x1: Optional[float] = None,
        n_refine: int = 2001,
    ) -> float:
        """Area under the influence line (∫ IL dx).

        Parameters
        ----------
        sign : {"all", "positive", "negative"}
            ``"all"`` integrates the signed area. ``"positive"`` keeps
            only the positive ordinates (used for the *maximum* effect
            of a patch / lane load that can be placed wherever it
            increases the response). ``"negative"`` keeps the negative
            ordinates (minimum effect).
        x0, x1 : float, optional
            Integration limits (default: full lane).
        n_refine : int
            Number of sample points for the trapezoidal integration of
            the (clipped) interpolated line.
        """
        a = self.stations[0] if x0 is None else max(x0, self.stations[0])
        b = self.stations[-1] if x1 is None else min(x1, self.stations[-1])
        if b <= a:
            return 0.0
        xs = np.linspace(a, b, n_refine)
        ys = self(xs)
        if sign == "positive":
            ys = np.clip(ys, 0.0, None)
        elif sign == "negative":
            ys = np.clip(ys, None, 0.0)
        elif sign != "all":
            raise ValueError("sign must be 'all', 'positive', or 'negative'")
        return float(np.trapezoid(ys, xs))


# ============================================================ response extractors

class ResponseExtractor:
    """Base class for a scalar response read from a solved model.

    Subclasses declare ``needs_elements`` / ``needs_reactions`` so the
    engine performs the minimum post-processing per lane position.
    """

    needs_elements: bool = False
    needs_reactions: bool = False
    name: str = "response"

    def evaluate(self, model) -> float:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass
class Displacement(ResponseExtractor):
    """A nodal displacement / rotation DOF value."""

    node_tag: int
    dof: int
    name: str = "displacement"
    needs_elements: bool = False
    needs_reactions: bool = False

    def evaluate(self, model) -> float:
        return float(model.node(self.node_tag).disp[self.dof])


@dataclass
class Reaction(ResponseExtractor):
    """A support reaction component."""

    node_tag: int
    dof: int
    name: str = "reaction"
    needs_elements: bool = False
    needs_reactions: bool = True

    def evaluate(self, model) -> float:
        return float(model.node(self.node_tag).reaction[self.dof])


@dataclass
class BeamForce(ResponseExtractor):
    """An internal force (N / V / M) in a 2-D beam-column element.

    Evaluated at a fractional position ``xi`` along the element axis
    (0 = node i, 1 = node j). End sections (``end="i"`` / ``"j"``) are
    *nodally exact*; interior sections use linear interpolation of the
    end forces and are exact except while the traversing unit load sits
    within this same element (a local error that vanishes with mesh
    refinement -- mesh a node at every section of interest).

    Parameters
    ----------
    element_tag : int
    component : {"N", "V", "M"}
    end : {"i", "j"}, optional
        Convenience for ``xi = 0`` (i) or ``xi = 1`` (j).
    xi : float, optional
        Explicit fractional position in [0, 1]. Overrides ``end``.
    """

    element_tag: int
    component: str = "M"
    end: Optional[str] = "j"
    xi: Optional[float] = None
    name: str = "beam_force"
    needs_elements: bool = True
    needs_reactions: bool = False

    def __post_init__(self) -> None:
        if self.component not in ("N", "V", "M"):
            raise ValueError("component must be 'N', 'V', or 'M'")
        if self.xi is None:
            if self.end == "i":
                self._frac = 0.0
            elif self.end == "j":
                self._frac = 1.0
            else:
                raise ValueError("end must be 'i' or 'j' (or pass xi)")
        else:
            if not (0.0 <= self.xi <= 1.0):
                raise ValueError("xi must be in [0, 1]")
            self._frac = float(self.xi)

    def evaluate(self, model) -> float:
        elem = model.element(self.element_tag)
        diag = beam_force_diagram(elem, n_points=3)
        L = diag["length"]
        target_s = self._frac * L
        return float(np.interp(target_s, diag["s"], diag[self.component]))


# ============================================================ Lane

@dataclass
class Lane:
    """The path a moving load travels along, as an ordered node list.

    Parameters
    ----------
    node_tags : Sequence[int]
        Ordered node tags the load passes over. The unit load is placed
        at each of these nodes in turn. For a fine influence line, mesh
        the girder finely; the influence line is exact at every node.
    stations : Sequence[float], optional
        Distance along the lane for each node (m). If omitted, computed
        as the cumulative straight-line distance between consecutive
        node coordinates (correct for straight or polyline alignments).
    load_dof : int, default 1
        The DOF index the moving (gravity) load drives. For a 2-D frame
        (ndf = 3) the vertical DOF is index 1. For a 3-D model set this
        to your vertical DOF explicitly.
    gravity_sign : float, default -1.0
        Sign of the unit load on ``load_dof`` (-1 = downward). The
        influence-line ordinate is the response per unit load in this
        direction, so positive axle magnitudes convolve directly.
    name : str
    """

    node_tags: Sequence[int]
    stations: Optional[Sequence[float]] = None
    load_dof: int = 1
    gravity_sign: float = -1.0
    name: str = "lane"

    def __post_init__(self) -> None:
        self.node_tags = list(self.node_tags)
        if len(self.node_tags) < 2:
            raise ValueError("lane needs at least 2 nodes")
        if self.stations is not None:
            st = np.asarray(self.stations, dtype=float).ravel()
            if st.size != len(self.node_tags):
                raise ValueError("stations length must match node_tags")
            self.stations = st

    def resolve_stations(self, model) -> np.ndarray:
        """Return the per-node stations, computing them from geometry
        if they were not supplied."""
        if self.stations is not None:
            return np.asarray(self.stations, dtype=float)
        coords = [model.node(t).coords for t in self.node_tags]
        st = np.zeros(len(coords))
        for i in range(1, len(coords)):
            st[i] = st[i - 1] + float(np.linalg.norm(coords[i] - coords[i - 1]))
        return st


# ============================================================ engine

class InfluenceLineEngine:
    """Build influence lines on an arbitrary linear-elastic model.

    Parameters
    ----------
    model : Model
        Any femsolver model. Its geometry and stiffness define the
        influence lines; the model's *applied loads are ignored*
        (the engine applies its own unit loads). Nodal loads are
        snapshotted and restored; element distributed loads are cleared
        for the duration -- build influence lines on the bare structure,
        then apply gravity / other loads separately.
    constraints : str, default "transformation"
        Constraint handler used for the MP-constraint fallback path.
    numberer : {"default", "rcm"}, default "default"

    Notes
    -----
    * **Fast path** (no MP constraints): K is assembled and LU-factorised
      once at construction; each lane position is a back-substitution.
    * **Fallback path** (model has rigid links / diaphragms / equalDOF):
      a full :class:`LinearStaticAnalysis` is run per position. Correct
      for every model feature, just slower; bridges rarely need it.
    """

    def __init__(self, model, *, constraints: str = "transformation",
                 numberer: str = "default"):
        self.model = model
        self._constraints = constraints
        self._has_mp = bool(model.mp_constraints)

        model.reset_results()
        if numberer == "rcm":
            from femsolver.numerics.dof_numbering import rcm_renumber
            rcm_renumber(model)
        elif numberer == "default":
            model.number_dofs()
        else:
            raise ValueError("numberer must be 'default' or 'rcm'")
        self.neq = model.neq
        if self.neq == 0:
            raise RuntimeError("model has no free DOFs")

        # Assemble stiffness once; keep element-K list for reactions.
        K, elem_K_list = assemble_stiffness(model, return_element_K=True)
        self._K = K
        self._elem_K_list = elem_K_list
        self._factor = None
        if not self._has_mp:
            try:
                self._factor = splu(K.tocsc())
            except Exception as exc:
                raise RuntimeError(
                    f"stiffness factorisation failed: {exc}. Likely a "
                    "singular matrix (insufficient supports / mechanism)."
                ) from exc

    # ----------------------------------------------------------- public API
    def influence_line(
        self, lane: Lane, response: ResponseExtractor
    ) -> InfluenceLine:
        """Influence line for a single response over ``lane``."""
        return self.influence_lines(lane, {response.name: response})[
            response.name
        ]

    def influence_lines(
        self, lane: Lane, responses: dict
    ) -> dict:
        """Influence lines for several responses in a single traversal.

        Parameters
        ----------
        lane : Lane
        responses : dict[str, ResponseExtractor]
            Named responses to record at every lane position.

        Returns
        -------
        dict[str, InfluenceLine]
        """
        if not responses:
            raise ValueError("provide at least one response")
        stations = lane.resolve_stations(self.model)
        need_elem = any(r.needs_elements for r in responses.values())
        need_react = any(r.needs_reactions for r in responses.values())

        n = len(lane.node_tags)
        recorded = {name: np.zeros(n) for name in responses}

        # Snapshot + clear loads so unit-load reactions are uncontaminated.
        snap = {nd.tag: nd._load.copy() for nd in self.model.nodes.values()}
        self.model.clear_loads()
        try:
            for i, node_tag in enumerate(lane.node_tags):
                self._solve_unit_load(
                    node_tag, lane.load_dof, lane.gravity_sign,
                    need_elem=need_elem, need_react=need_react,
                )
                for name, r in responses.items():
                    recorded[name][i] = r.evaluate(self.model)
        finally:
            # restore nodal loads
            for nd in self.model.nodes.values():
                nd._load[:] = snap[nd.tag]

        return {
            name: InfluenceLine(
                stations=stations.copy(), values=vals,
                response_name=name, lane_name=lane.name,
            )
            for name, vals in recorded.items()
        }

    # ----------------------------------------------------------- solve cores
    def _solve_unit_load(self, node_tag, load_dof, gravity_sign, *,
                          need_elem, need_react) -> None:
        if self._has_mp:
            self._solve_unit_load_mp(node_tag, load_dof, gravity_sign)
        else:
            self._solve_unit_load_fast(
                node_tag, load_dof, gravity_sign,
                need_elem=need_elem, need_react=need_react,
            )

    def _solve_unit_load_fast(self, node_tag, load_dof, gravity_sign, *,
                               need_elem, need_react) -> None:
        m = self.model
        eq = int(m.node(node_tag).eqn[load_dof])
        if eq < 0:
            # Load DOF is a support: the unit load is reacted directly,
            # the flexible structure does not deform. All displacement
            # and internal-force ordinates are zero; the reaction at
            # this very DOF balances the applied load (= 1 per unit
            # downward load), every other reaction is zero.
            for node in m.nodes.values():
                node.disp[:] = 0.0
            if need_react:
                for node in m.nodes.values():
                    node.reaction[:] = 0.0
                m.node(node_tag).reaction[load_dof] = -gravity_sign * 1.0
            return
        F = np.zeros(self.neq)
        F[eq] = gravity_sign * 1.0
        u = self._factor.solve(F)
        # scatter to nodes
        for node in m.nodes.values():
            for k in range(node.ndf):
                e = node.eqn[k]
                node.disp[k] = u[e] if e >= 0 else 0.0
        if need_elem or need_react:
            for el in m.elements.values():
                el.recover()
        if need_react:
            assemble_reactions(m, elem_K_list=self._elem_K_list)

    def _solve_unit_load_mp(self, node_tag, load_dof, gravity_sign) -> None:
        m = self.model
        m.clear_loads()
        fvec = np.zeros(m.ndf)
        fvec[load_dof] = gravity_sign * 1.0
        m.add_nodal_load(node_tag, fvec)
        LinearStaticAnalysis(m, constraints=self._constraints).run()
        m.clear_loads()


# ============================================================ vehicle convolution

def _reverse_vehicle(vehicle: MovingLoad) -> MovingLoad:
    """Return the vehicle travelling in the opposite direction
    (axle order mirrored). Used so asymmetric trains are checked both
    ways."""
    offs = vehicle.axle_offsets
    new_offs = offs[-1] - offs[::-1]
    return MovingLoad(
        axle_loads=vehicle.axle_loads[::-1].copy(),
        axle_offsets=new_offs.copy(),
        name=vehicle.name + " (reversed)",
    )


def moving_load_envelope(
    il: InfluenceLine,
    vehicle: MovingLoad,
    *,
    n_positions: int = 801,
    both_directions: bool = True,
) -> dict:
    """Sweep a vehicle (axle train) across an influence line and return
    the maximum and minimum response.

    The lead axle is swept from before the lane (``station[0] -
    train_length``) to the lane end, so every placement is covered.
    Axles off the lane contribute zero (``il`` returns 0 there).

    Parameters
    ----------
    il : InfluenceLine
    vehicle : MovingLoad
    n_positions : int, default 801
        Number of lead-axle positions in the sweep.
    both_directions : bool, default True
        Also sweep the mirrored train (governs for asymmetric vehicles
        such as the HL-93 truck).

    Returns
    -------
    dict with ``max``, ``min``, ``pos_max``, ``pos_min`` (lead-axle
    station at the governing placement).
    """
    s0, s1 = il.stations[0], il.stations[-1]
    train_L = vehicle.total_length

    def sweep(veh: MovingLoad) -> tuple:
        positions = np.linspace(s0 - train_L, s1, n_positions)
        axle_x = positions[:, None] + veh.axle_offsets[None, :]
        eta = il(axle_x.ravel()).reshape(axle_x.shape)
        responses = eta @ veh.axle_loads
        i_max = int(np.argmax(responses))
        i_min = int(np.argmin(responses))
        return (responses[i_max], positions[i_max],
                responses[i_min], positions[i_min])

    rmax, pmax, rmin, pmin = sweep(vehicle)
    if both_directions:
        rmax2, pmax2, rmin2, pmin2 = sweep(_reverse_vehicle(vehicle))
        if rmax2 > rmax:
            rmax, pmax = rmax2, pmax2
        if rmin2 < rmin:
            rmin, pmin = rmin2, pmin2

    return {
        "max": float(rmax), "min": float(rmin),
        "pos_max": float(pmax), "pos_min": float(pmin),
        "vehicle": vehicle.name,
    }


def lane_load_response(il: InfluenceLine, w: float) -> dict:
    """Maximum / minimum response from a uniform lane load ``w`` (N/m).

    The lane load is placed only where it *increases* the response
    (positive IL area for the max, negative IL area for the min) --
    the standard treatment for a patch / lane load on an influence
    line (AASHTO §3.6.1.3.1).

    Returns
    -------
    dict with ``max`` and ``min``.
    """
    return {
        "max": float(w * il.integrate(sign="positive")),
        "min": float(w * il.integrate(sign="negative")),
    }


def aashto_hl93_envelope(
    il: InfluenceLine,
    *,
    im: float = 0.33,
    lane_load: float = 9.34e3,
    n_positions: int = 801,
    include_lane: bool = True,
) -> dict:
    """AASHTO LRFD HL-93 live-load envelope on an influence line.

    HL-93 design live load (§3.6.1.2) is the worse of:

    * design **truck** + design lane load, or
    * design **tandem** + design lane load,

    with the dynamic load allowance ``IM`` applied to the vehicular
    part only (not the lane load).  Here::

        effect = max(truck, tandem) * (1 + IM)  +  lane_load_effect

    Parameters
    ----------
    il : InfluenceLine
    im : float, default 0.33
        Dynamic load allowance (33 % for the strength/service limit
        states; use 0.15 for fatigue, 0.75 for deck joints).
    lane_load : float, default 9.34e3
        Uniform lane load (N/m) -- 9.34 kN/m per AASHTO.
    n_positions : int
        Sweep resolution for the vehicle convolution.
    include_lane : bool, default True
        Add the lane-load effect. Set False to get the bare vehicular
        envelope.

    Returns
    -------
    dict with ``max``, ``min``, ``governing`` ("truck" or "tandem"),
    and the component breakdown.
    """
    from femsolver.bridges.influence import (
        aashto_hl93_tandem,
        aashto_hl93_truck,
    )

    truck = moving_load_envelope(il, aashto_hl93_truck(),
                                  n_positions=n_positions)
    tandem = moving_load_envelope(il, aashto_hl93_tandem(),
                                   n_positions=n_positions)

    dyn = 1.0 + im
    # Governing vehicular for max and min independently.
    veh_max = max(truck["max"], tandem["max"]) * dyn
    veh_min = min(truck["min"], tandem["min"]) * dyn
    gov_max = "truck" if truck["max"] >= tandem["max"] else "tandem"
    gov_min = "truck" if truck["min"] <= tandem["min"] else "tandem"

    lane = lane_load_response(il, lane_load) if include_lane else {
        "max": 0.0, "min": 0.0
    }

    return {
        "max": float(veh_max + lane["max"]),
        "min": float(veh_min + lane["min"]),
        "vehicular_max": float(veh_max),
        "vehicular_min": float(veh_min),
        "lane_max": float(lane["max"]),
        "lane_min": float(lane["min"]),
        "governing_max": gov_max,
        "governing_min": gov_min,
        "im": im,
    }
