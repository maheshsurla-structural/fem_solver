"""Cable / cable-net form-finding by the Force-Density Method (FDM).

Cable-stayed and suspension bridges (and every tensile / cable-net
roof) pose an *inverse* problem the ordinary stiffness method cannot
answer directly: **what initial geometry and pretension put the cable
system in equilibrium under its dead load?** A cable has no bending
stiffness, so its rest shape is not a design input -- it is an output
that must be *found*.

The **Force-Density Method** (Schek, 1974) is the classical, robust
solution. Its key idea: parametrise each cable branch *j* by its
**force density**

    q_j = T_j / L_j        (tension / length, units N/m)

rather than by its (unknown) length. With the force densities chosen,
the equilibrium of the free nodes becomes a **linear** system in the
nodal coordinates -- no iteration, no initial-shape sensitivity, no
singular tangent from the zero bending stiffness:

    (C_N^T Q C_N) x_N = p_N - (C_N^T Q C_F) x_F

where

* ``C`` is the branch-node connectivity matrix (``C = [C_N | C_F]``
  split into free / fixed columns),
* ``Q = diag(q)``,
* ``p`` is the applied nodal load,
* ``x_F`` are the prescribed (anchor) coordinates.

Solving the same system independently for each coordinate direction
gives the equilibrium shape directly. The branch tensions follow from
``T_j = q_j · L_j`` evaluated on the found geometry.

The found coordinates + tensions then seed a finite-element model
(e.g. :class:`~femsolver.bridges.cable.CableElement3D` with
``T_operating`` from the form-finding) so that a subsequent stiffness
analysis starts from a self-equilibrated dead-load state.

References
----------
* Schek, H.-J. (1974). "The force density method for form finding and
  computation of general networks." *Computer Methods in Applied
  Mechanics and Engineering*, 3(1), 115-134.
* Lewis, W.J. (2003). *Tension Structures: Form and Behaviour.*
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FormFindingResult:
    """Result of a force-density form-finding.

    Attributes
    ----------
    coords : np.ndarray, shape (n_nodes, ndm)
        Equilibrium coordinates. Fixed nodes keep their input values;
        free nodes are the solved positions.
    tensions : np.ndarray, shape (n_branches,)
        Branch tensions ``T_j = q_j · L_j`` on the found geometry (N).
    lengths : np.ndarray, shape (n_branches,)
        Branch lengths on the found geometry (m).
    force_densities : np.ndarray, shape (n_branches,)
        The force densities used (N/m).
    free_nodes, fixed_nodes : np.ndarray
        Index arrays.
    residual : float
        Infinity-norm of the free-node equilibrium residual
        ``C_N^T Q C x - p`` (≈ machine zero for a successful solve).
    """

    coords: np.ndarray
    tensions: np.ndarray
    lengths: np.ndarray
    force_densities: np.ndarray
    free_nodes: np.ndarray
    fixed_nodes: np.ndarray
    residual: float


def _connectivity(branches: np.ndarray, n_nodes: int) -> np.ndarray:
    """Branch-node connectivity matrix C (m x n): +1 at the start node,
    -1 at the end node of each branch."""
    m = branches.shape[0]
    C = np.zeros((m, n_nodes))
    for j, (a, b) in enumerate(branches):
        C[j, a] = 1.0
        C[j, b] = -1.0
    return C


def force_density_form_find(
    coords_init,
    branches,
    fixed,
    q,
    *,
    loads=None,
) -> FormFindingResult:
    """Find the equilibrium shape of a cable net by the force-density method.

    Parameters
    ----------
    coords_init : array (n_nodes, ndm)
        Initial coordinates. Only the **fixed** node rows matter for the
        result (they are the anchors); free-node rows are overwritten by
        the solution but are still used to report the initial guess.
    branches : array (n_branches, 2) of int
        Node-index pairs (start, end) for each cable branch.
    fixed : sequence of int
        Indices of the fixed (anchor / support) nodes.
    q : float or array (n_branches,)
        Force density ``T/L`` (N/m) for each branch. A scalar applies to
        all branches. Must be > 0 for cables in tension.
    loads : array (n_nodes, ndm), optional
        Applied nodal loads (N). Defaults to zero. Use a downward
        component (e.g. negative on the vertical axis) for self-weight /
        dead load lumped to nodes.

    Returns
    -------
    FormFindingResult

    Notes
    -----
    The free-node sub-matrix ``D_N = C_N^T Q C_N`` is positive-definite
    whenever every free node is connected to the structure by at least
    one branch with ``q > 0``; an isolated free node makes it singular
    (raised as a clear error).
    """
    coords = np.array(coords_init, dtype=float)
    if coords.ndim != 2:
        raise ValueError("coords_init must be (n_nodes, ndm)")
    n_nodes, ndm = coords.shape
    branches = np.asarray(branches, dtype=int)
    if branches.ndim != 2 or branches.shape[1] != 2:
        raise ValueError("branches must be (n_branches, 2)")
    m = branches.shape[0]

    q_arr = np.full(m, float(q)) if np.isscalar(q) else np.asarray(q, dtype=float)
    if q_arr.shape != (m,):
        raise ValueError("q must be scalar or length n_branches")
    if np.any(q_arr <= 0.0):
        raise ValueError("force densities q must be > 0 (cables in tension)")

    fixed_arr = np.asarray(sorted(set(int(i) for i in fixed)), dtype=int)
    if fixed_arr.size == 0:
        raise ValueError("need at least one fixed (anchor) node")
    free_mask = np.ones(n_nodes, dtype=bool)
    free_mask[fixed_arr] = False
    free_arr = np.nonzero(free_mask)[0]
    if free_arr.size == 0:
        raise ValueError("all nodes fixed -- nothing to form-find")

    P = np.zeros((n_nodes, ndm)) if loads is None else np.array(loads, dtype=float)
    if P.shape != (n_nodes, ndm):
        raise ValueError("loads must be (n_nodes, ndm)")

    C = _connectivity(branches, n_nodes)
    Q = np.diag(q_arr)
    Cn = C[:, free_arr]
    Cf = C[:, fixed_arr]
    Dn = Cn.T @ Q @ Cn        # (nfree, nfree)
    Df = Cn.T @ Q @ Cf        # (nfree, nfixed)

    # Solve each coordinate direction independently.
    x_free = np.zeros((free_arr.size, ndm))
    rhs = P[free_arr, :] - Df @ coords[fixed_arr, :]
    try:
        x_free = np.linalg.solve(Dn, rhs)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "force-density matrix is singular -- likely an isolated free "
            "node or a free node connected only by zero-force-density "
            f"branches. ({exc})"
        ) from exc

    coords[free_arr, :] = x_free

    # Branch lengths + tensions on the found geometry.
    vec = coords[branches[:, 0], :] - coords[branches[:, 1], :]
    lengths = np.linalg.norm(vec, axis=1)
    tensions = q_arr * lengths

    # Free-node equilibrium residual: C_N^T Q C x - p
    resid = Cn.T @ Q @ (C @ coords) - P[free_arr, :]
    residual = float(np.max(np.abs(resid))) if resid.size else 0.0

    return FormFindingResult(
        coords=coords,
        tensions=tensions,
        lengths=lengths,
        force_densities=q_arr,
        free_nodes=free_arr,
        fixed_nodes=fixed_arr,
        residual=residual,
    )
