"""Cable-stayed initial-force optimisation — the unknown-load-factor method
(bridge plan T2.2).

The stay-cable pretensions of a cable-stayed bridge are chosen so that, under
permanent (dead) load, the deck and pylon reach a **target geometry** (usually
near-zero deck deflection) and/or **target forces**. MIDAS calls this the
*Unknown Load Factor* method; CSiBridge the *target-force* iteration.

Because the structure is linear, the response is a superposition of the dead
load plus each cable's unit pretension:

    r_i(x) = b0_i + Σ_j A_ij x_j

with ``b0_i`` the target response ``i`` under dead load alone, ``A_ij`` its
response to a **unit pretension** in cable ``j`` (a self-equilibrated pair of
forces pulling the cable ends together), and ``x_j`` the unknown cable
tensions. :func:`unknown_load_factors` builds ``A`` and ``b0`` with one
stiffness factorisation (a solve per cable), then solves ``b0 + A x = t`` for
the tensions ``x`` (exact when square and full-rank, least-squares otherwise).

Targets are ordinary :class:`~femsolver.bridges.moving_load.ResponseExtractor`
objects (``Displacement`` / ``Reaction`` / ``BeamForce``), so a target can be a
deck deflection, a reaction, or a member force.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import splu

from femsolver.analysis.assembler import (assemble_force, assemble_reactions,
                                          assemble_stiffness)


@dataclass
class Cable:
    """A tunable stay cable, identified by its two end nodes. A unit
    pretension pulls ``node_a`` and ``node_b`` toward each other."""

    node_a: int
    node_b: int
    name: str = "cable"


def _unit_cable_force(model, cable: Cable) -> np.ndarray:
    """Free-DOF load vector for a **unit tension** in ``cable`` — end forces
    pulling the two ends together along the chord."""
    F = np.zeros(model.neq)
    na, nb = model.node(cable.node_a), model.node(cable.node_b)
    d = np.asarray(nb.coords, dtype=float) - np.asarray(na.coords, dtype=float)
    L = float(np.linalg.norm(d))
    if L <= 0.0:
        raise ValueError(f"cable {cable.name}: zero length")
    e = d / L
    ndm = model.ndm
    for node, sign in ((na, +1.0), (nb, -1.0)):     # tension pulls A→B, B→A
        for k in range(ndm):
            eq = int(node.eqn[k])
            if eq >= 0:
                F[eq] += sign * e[k]
    return F


@dataclass
class CableTuningResult:
    """Outcome of :func:`unknown_load_factors`.

    Attributes
    ----------
    tensions : ndarray
        Cable pretensions ``x`` that best meet the targets (N; positive =
        tension).
    achieved : ndarray
        Target responses actually achieved (``b0 + A x``).
    target : ndarray
        Requested target values.
    residual : ndarray
        ``achieved − target`` (≈ 0 when the system is square + full-rank).
    dead_load_response : ndarray
        Target responses under dead load alone (``b0``).
    influence : ndarray
        The ``(n_target, n_cable)`` unit-pretension influence matrix ``A``.
    """

    tensions: np.ndarray
    achieved: np.ndarray
    target: np.ndarray
    residual: np.ndarray
    dead_load_response: np.ndarray
    influence: np.ndarray


def unknown_load_factors(model, cables, targets) -> CableTuningResult:
    """Solve for the cable pretensions that meet the target conditions.

    Parameters
    ----------
    model : Model
        Cable-stayed structure with **dead load already applied** (its loads
        are the permanent-load case). Cables should be present as elements.
    cables : sequence[Cable]
        The tunable cables (unknowns).
    targets : sequence[(ResponseExtractor, float)]
        Target conditions: each response is driven toward its target value.
        Over-determined (more targets than cables) → least squares;
        under-determined → minimum-norm tensions.
    """
    cables = list(cables)
    exts = [t[0] for t in targets]
    tvals = np.array([float(t[1]) for t in targets])
    if not cables or not exts:
        raise ValueError("need at least one cable and one target")

    model.reset_results()
    model.number_dofs()
    if model.neq == 0:
        raise RuntimeError("model has no free DOFs")

    K, elem_K = assemble_stiffness(model, return_element_K=True)
    lu = splu(K.tocsc())
    need_elem = any(getattr(e, "needs_elements", False) for e in exts)
    need_react = any(getattr(e, "needs_reactions", False) for e in exts)

    def _solve_and_eval(F: np.ndarray) -> np.ndarray:
        u = lu.solve(F)
        for node in model.nodes.values():
            for k in range(node.ndf):
                eq = int(node.eqn[k])
                node.disp[k] = u[eq] if eq >= 0 else 0.0
        if need_elem or need_react:
            for el in model.elements.values():
                el.recover()
        if need_react:
            assemble_reactions(model, elem_K_list=elem_K)
        return np.array([e.evaluate(model) for e in exts])

    # dead-load response (b0) and the unit-pretension influence matrix (A)
    b0 = _solve_and_eval(assemble_force(model))
    A = np.zeros((len(exts), len(cables)))
    for j, cable in enumerate(cables):
        A[:, j] = _solve_and_eval(_unit_cable_force(model, cable))

    # solve b0 + A x = t  (least squares handles non-square / rank-deficient)
    x, *_ = np.linalg.lstsq(A, tvals - b0, rcond=None)
    achieved = b0 + A @ x
    return CableTuningResult(
        tensions=x, achieved=achieved, target=tvals,
        residual=achieved - tvals, dead_load_response=b0, influence=A)


def apply_cable_tensions(model, cables, tensions) -> None:
    """Add the tuned cable pretensions to ``model`` as equivalent nodal loads
    (so a subsequent solve includes the stay forces alongside the dead load)."""
    for cable, T in zip(cables, np.asarray(tensions, dtype=float).ravel()):
        F = _unit_cable_force(model, cable) * T
        for node in model.nodes.values():
            comps = []
            any_nz = False
            for k in range(node.ndf):
                eq = int(node.eqn[k])
                v = float(F[eq]) if eq >= 0 else 0.0
                comps.append(v)
                any_nz = any_nz or v != 0.0
            if any_nz:
                model.add_nodal_load(node.tag, comps)
