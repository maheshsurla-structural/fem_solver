"""Construction-stage bridge workflows (bridge plan T1.4).

The incremental staged-erection core (:class:`IncrementalStagedAnalysis`,
element birth/death) already carries the hard mechanics. This module adds the
bridge-specific workflow layer on top of it:

* **Camber / geometry control** (:func:`staged_camber`) — from a forward
  staged analysis, the deflection each node will still accumulate *after it is
  cast* (born) is the pre-camber to build that segment with, so the completed
  bridge reaches the target profile. Both the final camber (relative to the
  datum) and the per-segment residual camber (the number a segmental bridge is
  actually built to) are returned. This is the "backward" geometry-control
  information read off the forward solve.

* **Tendon stressing sequence** (:func:`tendon_stage_loads`) — turns a
  :class:`~femsolver.bridges.tendon.Tendon` into the ``{node: [F...]}`` nodal
  loads for one :class:`ErectionStage`, so post-tensioning can be stressed at
  the construction stage it actually happens (cantilever tendons per segment,
  continuity tendons after closure) rather than all at once.

* **Composite (wet → hardened) staging** — supported directly by element
  birth: keep the composite/deck element in the model and *add* it at the
  hardening stage. Wet-concrete load applied before that stage acts on the bare
  girder (deck inactive); load after acts on the composite section (deck
  active). :func:`merge_stage_loads` helps combine several load sources on one
  stage. See ``tests/test_bridge_construction_stage.py`` for the pattern.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def merge_stage_loads(*load_dicts) -> dict:
    """Merge several ``{node_tag: [F...]}`` stage-load dicts, summing the
    force vectors of any node that appears in more than one."""
    out: dict = {}
    for d in load_dicts:
        for tag, f in (d or {}).items():
            f = np.asarray(f, dtype=float)
            if tag in out:
                acc = np.asarray(out[tag], dtype=float)
                n = max(acc.size, f.size)
                merged = np.zeros(n)
                merged[:acc.size] += acc
                merged[:f.size] += f
                out[tag] = merged.tolist()
            else:
                out[tag] = f.tolist()
    return out


def tendon_stage_loads(tendon, model) -> dict:
    """Equivalent nodal loads of ``tendon`` as a ``{node_tag: [F...]}`` dict,
    ready to drop into an :class:`ErectionStage`'s ``loads`` — so the tendon is
    stressed at that stage. The model's own loads are left untouched.
    """
    snap = {t: nd._load.copy() for t, nd in model.nodes.items()}
    try:
        model.clear_loads()
        tendon.apply_to(model)
        loads = {}
        for t, nd in model.nodes.items():
            v = np.asarray(nd._load, dtype=float)
            if np.any(v != 0.0):
                loads[t] = v.copy().tolist()
    finally:
        for t, nd in model.nodes.items():
            nd._load[:] = snap[t]
    return loads


@dataclass
class StagedCamber:
    """Camber / geometry-control result from a forward staged analysis.

    Attributes
    ----------
    nodes : list[int]
        Node tags, ordered by ``x`` (longitudinal).
    x : ndarray
        Longitudinal coordinate of each node.
    final_deflection : ndarray
        Cumulative vertical deflection of each node at the end of construction.
    final_camber : ndarray
        ``-final_deflection`` — the pre-set (build-high) geometry relative to
        the datum so the finished bridge sits on the target profile.
    stage_deflection : ndarray, shape (n_stages, n_nodes)
        Cumulative vertical deflection of each node at the end of every stage
        (the "stage deflection" table). A node's row is 0 until its segment is
        cast. Lets any casting-camber convention be derived transparently.
    birth_stage : ndarray
        Stage index at which each node was first cast (``-1`` = initially
        active before stage 1).
    """

    nodes: list
    x: np.ndarray
    final_deflection: np.ndarray
    final_camber: np.ndarray
    stage_deflection: np.ndarray
    birth_stage: np.ndarray


def _element_birth_stage(result) -> dict:
    """Stage index at which each element first became active (``-1`` if active
    before the first stage), from the result's ``active_history``."""
    birth = {}
    prev = set()
    for k, active in enumerate(result.active_history):
        for tag in active - prev:
            birth.setdefault(tag, k)
        prev = active
    return birth


def staged_camber(result, model, stages, *, dof: int = 1, x_axis: int = 0,
                  nodes=None) -> StagedCamber:
    """Compute camber / geometry-control values from an
    :class:`IncrementalStagedResult`.

    Parameters
    ----------
    result : IncrementalStagedResult
    model : Model
        The same model the staged analysis ran on (DOF numbering intact).
    stages : list[ErectionStage]
        The stage script (used to attribute nodes to their casting stage).
    dof : int, default 1
        Vertical DOF index (2-D frame: 1).
    x_axis : int, default 0
        Coordinate index used as the longitudinal axis for ordering/plots.
    nodes : sequence[int], optional
        Restrict to these node tags (default: all model nodes), ordered by x.
    """
    # per-stage cumulative displacement (prefix sums of the increments)
    cum = []
    running = np.zeros_like(result.u_cumulative)
    for du in result.u_increments:
        running = running + du
        cum.append(running.copy())

    elem_birth = _element_birth_stage(result)
    # node casting stage = earliest birth stage over incident elements
    node_birth: dict = {}
    for el in model.elements.values():
        b = elem_birth.get(el.tag, -1)
        for nt in el.node_tags:
            if nt not in node_birth or b < node_birth[nt]:
                node_birth[nt] = b

    tags = list(nodes) if nodes is not None else list(model.nodes.keys())
    tags.sort(key=lambda t: float(model.node(t).coords[x_axis]))

    x = np.array([float(model.node(t).coords[x_axis]) for t in tags])
    final = np.zeros(len(tags))
    birth = np.zeros(len(tags), dtype=int)
    stage_defl = np.zeros((len(cum), len(tags)))
    for i, t in enumerate(tags):
        eq = int(model.node(t).eqn[dof])
        birth[i] = node_birth.get(t, -1)
        if eq < 0:
            continue                        # supported DOF → no deflection
        final[i] = float(result.u_cumulative[eq])
        for k in range(len(cum)):
            stage_defl[k, i] = float(cum[k][eq])

    return StagedCamber(
        nodes=tags, x=x, final_deflection=final, final_camber=-final,
        stage_deflection=stage_defl, birth_stage=birth)
