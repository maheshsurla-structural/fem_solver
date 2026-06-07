"""EnvelopeAnalysis -- per-member and per-node envelopes across many
load combinations.

Given a model, a dict of named :class:`LoadPattern`s, and a list of
:class:`LoadCombination`s (typically from
:func:`asce7_lrfd_combinations`), this driver:

1. For each combination:

   a. Clears the model's loads and applies the combination via
      :func:`apply_combination`.
   b. Runs :class:`LinearStaticAnalysis`.
   c. Records per-member end forces and per-node displacements.

2. After the sweep, computes for each element the **envelope** of
   its force components: max/min across combos plus the governing
   combo name for each.
3. Same for nodes (displacement envelope).

The output ``EnvelopeResult`` carries enough information to feed
straight into the AISC / ACI design drivers (Phase 29 + 30): each
member's worst-case M, V, P across all combos is available as a
``ForceEnvelope`` object.

Only BeamColumn2D / BeamColumn3D elements are envelope-tracked at
present (those are the elements with a meaningful ``end_forces_local``
recovery). Other elements appear in the per-combo raw results dict
but not in the per-member envelopes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from femsolver.analysis.linear_static import LinearStaticAnalysis
from femsolver.analysis.load_combinations import (
    LoadCombination,
    LoadPattern,
    apply_combination,
)


# ============================================================ envelope types

@dataclass
class ForceEnvelope:
    """Per-component max/min force envelope for one member across the
    full set of load combinations.

    Components are indexed by their position in ``end_forces_local``:
    for BeamColumn2D this is ``(F_xi, F_yi, M_zi, F_xj, F_yj, M_zj)``
    -- six components. Each component has both a maximum and a minimum
    value, plus the name of the load combination that produced each.

    Attributes
    ----------
    element_tag : int
    n_components : int
    max_values, min_values : np.ndarray
        Length-``n_components`` arrays of the worst-case (max, min)
        values across all combinations.
    max_combos, min_combos : list[str]
        Length-``n_components`` lists giving the combination name
        that produced each ``max_values[i]`` / ``min_values[i]``.
    """

    element_tag: int
    n_components: int
    max_values: np.ndarray
    min_values: np.ndarray
    max_combos: list
    min_combos: list

    @property
    def abs_max_per_component(self) -> np.ndarray:
        """Max of |min|, |max| per component -- the worst-case absolute
        force for design checks."""
        return np.maximum(np.abs(self.max_values),
                            np.abs(self.min_values))


@dataclass
class DispEnvelope:
    """Per-DOF max/min displacement envelope for one node across all
    combinations."""

    node_tag: int
    ndf: int
    max_values: np.ndarray
    min_values: np.ndarray
    max_combos: list
    min_combos: list

    @property
    def abs_max_per_dof(self) -> np.ndarray:
        return np.maximum(np.abs(self.max_values),
                            np.abs(self.min_values))


@dataclass
class EnvelopeResult:
    """Result of an envelope analysis across many load combinations.

    Attributes
    ----------
    combinations : list[LoadCombination]
        The combos that were run, in order.
    member_envelopes : dict[int, ForceEnvelope]
        Per-element ``ForceEnvelope`` (only for elements with
        ``end_forces_local`` -- typically BeamColumn2D / 3D).
    node_envelopes : dict[int, DispEnvelope]
    raw_member_forces : dict[int, dict[str, np.ndarray]]
        Per-element, per-combo raw end-force arrays. Useful if a
        caller wants to recompute envelopes on a different subset.
    raw_node_disps : dict[int, dict[str, np.ndarray]]
    """

    combinations: list = field(default_factory=list)
    member_envelopes: dict = field(default_factory=dict)
    node_envelopes: dict = field(default_factory=dict)
    raw_member_forces: dict = field(default_factory=dict)
    raw_node_disps: dict = field(default_factory=dict)


# ============================================================ driver

class EnvelopeAnalysis:
    """Run a model through many load combinations and produce per-
    member / per-node force / displacement envelopes.

    Parameters
    ----------
    model : Model
    patterns : dict[str, LoadPattern]
        Named load patterns. Keys must match the pattern names used
        by the combinations (e.g. ``"D"``, ``"L"``, ``"W"``, ``"E"``).
    combinations : Iterable[LoadCombination]
        Load combinations to evaluate, in order.

    Notes
    -----
    Each combination is applied independently via
    :func:`apply_combination` (clears loads then re-applies). Between
    combinations the model is otherwise unchanged. For nonlinear
    analyses use a per-combo model factory pattern (similar to
    ``ModalPushoverAnalysis``); this driver assumes linear static.
    """

    def __init__(self, model, patterns: dict,
                 combinations: Iterable):
        self.model = model
        self.patterns = dict(patterns)
        self.combinations = list(combinations)

    def run(self) -> EnvelopeResult:
        m = self.model
        # Per-combo raw recordings
        raw_forces: dict[int, dict[str, np.ndarray]] = {}
        raw_disps: dict[int, dict[str, np.ndarray]] = {}

        for combo in self.combinations:
            apply_combination(m, self.patterns, combo)
            LinearStaticAnalysis(m).run()
            # Record element end forces
            for etag, el in m.elements.items():
                ef = getattr(el, "end_forces_local", None)
                if ef is None:
                    continue
                raw_forces.setdefault(etag, {})[combo.name] = np.asarray(
                    ef, dtype=float
                ).copy()
            # Record node displacements
            for ntag, node in m.nodes.items():
                raw_disps.setdefault(ntag, {})[combo.name] = node.disp.copy()

        # Build envelopes
        member_envelopes: dict[int, ForceEnvelope] = {}
        for etag, by_combo in raw_forces.items():
            names = list(by_combo.keys())
            stack = np.array([by_combo[n] for n in names])     # (n_combos, n_comp)
            max_idx = np.argmax(stack, axis=0)
            min_idx = np.argmin(stack, axis=0)
            max_values = stack[max_idx, np.arange(stack.shape[1])]
            min_values = stack[min_idx, np.arange(stack.shape[1])]
            max_combos = [names[i] for i in max_idx]
            min_combos = [names[i] for i in min_idx]
            member_envelopes[etag] = ForceEnvelope(
                element_tag=etag,
                n_components=stack.shape[1],
                max_values=max_values,
                min_values=min_values,
                max_combos=max_combos,
                min_combos=min_combos,
            )

        node_envelopes: dict[int, DispEnvelope] = {}
        for ntag, by_combo in raw_disps.items():
            names = list(by_combo.keys())
            stack = np.array([by_combo[n] for n in names])
            max_idx = np.argmax(stack, axis=0)
            min_idx = np.argmin(stack, axis=0)
            max_values = stack[max_idx, np.arange(stack.shape[1])]
            min_values = stack[min_idx, np.arange(stack.shape[1])]
            max_combos = [names[i] for i in max_idx]
            min_combos = [names[i] for i in min_idx]
            node_envelopes[ntag] = DispEnvelope(
                node_tag=ntag,
                ndf=stack.shape[1],
                max_values=max_values,
                min_values=min_values,
                max_combos=max_combos,
                min_combos=min_combos,
            )

        return EnvelopeResult(
            combinations=list(self.combinations),
            member_envelopes=member_envelopes,
            node_envelopes=node_envelopes,
            raw_member_forces=raw_forces,
            raw_node_disps=raw_disps,
        )
