"""Assembly of the global stiffness matrix and force vector.

Vectorized COO build: for each element we take the outer-product DOF index
pattern via ``np.repeat`` / ``np.tile`` and write the flattened ``Ke`` into
preallocated arrays — this avoids the Python-level inner loops that
dominate runtime on medium-to-large meshes.

When ``return_element_K=True`` the assembler also returns a list of
``(element, dof_map, K_e)`` triples so that downstream stages (force
assembly with element-equivalent loads, reaction recovery) can reuse the
already-computed element stiffnesses.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _empty_K(neq: int) -> sp.csc_matrix:
    return sp.csc_matrix((neq, neq))


def assemble_stiffness(model, *, return_element_K: bool = False):
    """Assemble the global stiffness matrix restricted to free DOFs.

    Parameters
    ----------
    model : Model
    return_element_K : bool, optional
        If True, also return ``[(element, dof_map, K_e), ...]`` for reuse by
        force assembly and reaction recovery.

    Returns
    -------
    csc_matrix or (csc_matrix, list)
    """
    neq = model.neq
    elements = list(model.elements.values())
    if not elements:
        empty = _empty_K(neq)
        return (empty, []) if return_element_K else empty

    elem_K_list: list[tuple] = []
    total = 0
    for e in elements:
        Ke = e.K_global()
        dofs = model.element_dof_map(e)
        elem_K_list.append((e, dofs, Ke))
        total += dofs.size * dofs.size

    rows_all = np.empty(total, dtype=np.int64)
    cols_all = np.empty(total, dtype=np.int64)
    vals_all = np.empty(total, dtype=float)
    pos = 0
    for (_, dofs, Ke) in elem_K_list:
        n = dofs.size
        nn = n * n
        # outer-product pattern: row[i*n+j] = dofs[i], col[i*n+j] = dofs[j]
        rows_all[pos : pos + nn] = np.repeat(dofs, n)
        cols_all[pos : pos + nn] = np.tile(dofs, n)
        vals_all[pos : pos + nn] = np.asarray(Ke, dtype=float).ravel()
        pos += nn

    # drop entries that touch fixed DOFs (eq == -1)
    mask = (rows_all >= 0) & (cols_all >= 0)
    if not mask.any():
        K = _empty_K(neq)
    else:
        K = sp.coo_matrix(
            (vals_all[mask], (rows_all[mask], cols_all[mask])),
            shape=(neq, neq),
        ).tocsc()

    if return_element_K:
        return K, elem_K_list
    return K


def assemble_force(model, *, elem_K_list: list | None = None) -> np.ndarray:
    """Build the right-hand-side force vector for free DOFs.

    Includes nodal point loads and element-equivalent nodal forces (e.g.,
    from distributed beam loads, body forces). If ``elem_K_list`` is given
    we reuse its precomputed DOF maps to skip the per-element lookup.
    """
    neq = model.neq
    F = np.zeros(neq)
    for n in model.nodes.values():
        eq = n.eqn
        free = eq >= 0
        if free.any():
            F[eq[free]] += n._load[free]

    if elem_K_list is None:
        for e in model.elements.values():
            f_eq = e.f_eq_global()
            if f_eq is None or not np.any(f_eq):
                continue
            dofs = model.element_dof_map(e)
            free = dofs >= 0
            if free.any():
                F[dofs[free]] += np.asarray(f_eq, dtype=float)[free]
    else:
        for (e, dofs, _) in elem_K_list:
            f_eq = e.f_eq_global()
            if f_eq is None or not np.any(f_eq):
                continue
            free = dofs >= 0
            if free.any():
                F[dofs[free]] += np.asarray(f_eq, dtype=float)[free]
    return F


def assemble_reactions(
    model, u_full: np.ndarray | None = None, *, elem_K_list: list | None = None
) -> None:
    """Compute reactions at fixed DOFs by accumulating element contributions.

    If ``elem_K_list`` is provided we reuse its cached element stiffnesses
    (``K_e`` produced by :func:`assemble_stiffness`).
    """
    for n in model.nodes.values():
        n.reaction[:] = 0.0

    if elem_K_list is None:
        for e in model.elements.values():
            Ke = e.K_global()
            u_e = e.gather_u()
            _accumulate_element_reaction(model, e, Ke, u_e)
    else:
        for (e, _dofs, Ke) in elem_K_list:
            u_e = e.gather_u()
            _accumulate_element_reaction(model, e, Ke, u_e)

    for n in model.nodes.values():
        for j in range(n.ndf):
            if n.fixity[j]:
                n.reaction[j] -= n._load[j]
            else:
                n.reaction[j] = 0.0


def _accumulate_element_reaction(model, e, Ke, u_e) -> None:
    f_int_e = Ke @ u_e
    f_eq_e = e.f_eq_global()
    f_net = f_int_e - (f_eq_e if f_eq_e is not None else 0.0)
    dofs_per_node = e.dofs_per_node
    for k, nt in enumerate(e.node_tags):
        node = model.node(nt)
        node.reaction[:dofs_per_node] += f_net[k * dofs_per_node : (k + 1) * dofs_per_node]


def assemble_mass(model, *, lumped: bool = False) -> sp.csc_matrix:
    """Assemble the global mass matrix restricted to free DOFs.

    Vectorized COO build mirrors :func:`assemble_stiffness`. Returns a CSC
    matrix of shape ``(neq, neq)``. Elements whose material has ``rho==0``
    contribute zero blocks; the resulting matrix may be singular for
    rho==0 portions of the model (caller's responsibility to constrain
    away the corresponding DOFs in eigen analyses).
    """
    neq = model.neq
    elements = list(model.elements.values())
    if not elements:
        return _empty_K(neq)

    elem_M_list: list[tuple] = []
    total = 0
    for e in elements:
        Me = e.M_global(lumped=lumped)
        dofs = model.element_dof_map(e)
        elem_M_list.append((dofs, Me))
        total += dofs.size * dofs.size

    rows_all = np.empty(total, dtype=np.int64)
    cols_all = np.empty(total, dtype=np.int64)
    vals_all = np.empty(total, dtype=float)
    pos = 0
    for (dofs, Me) in elem_M_list:
        n = dofs.size
        nn = n * n
        rows_all[pos : pos + nn] = np.repeat(dofs, n)
        cols_all[pos : pos + nn] = np.tile(dofs, n)
        vals_all[pos : pos + nn] = np.asarray(Me, dtype=float).ravel()
        pos += nn

    mask = (rows_all >= 0) & (cols_all >= 0)
    if not mask.any():
        return _empty_K(neq)
    return sp.coo_matrix(
        (vals_all[mask], (rows_all[mask], cols_all[mask])),
        shape=(neq, neq),
    ).tocsc()
