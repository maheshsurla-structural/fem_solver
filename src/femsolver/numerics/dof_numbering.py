"""DOF numbering schemes.

The default sequential numbering (``Model.number_dofs``) is fine for small
models but produces wide-band stiffness matrices on structured meshes.
:func:`rcm_renumber` replaces the equation numbering with a reverse
Cuthill-McKee ordering derived from the node-adjacency graph (each pair of
nodes that share an element is adjacent). For symmetric positive-definite
direct solves this typically reduces fill-in and speeds up the solve.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee


def _node_adjacency(model) -> sp.csr_matrix:
    """Boolean (n_nodes x n_nodes) graph: an edge for each pair of nodes
    that appears together in any element."""
    tags = list(model.nodes.keys())
    index = {t: i for i, t in enumerate(tags)}
    rows: list[int] = []
    cols: list[int] = []
    for e in model.elements.values():
        idxs = [index[t] for t in e.node_tags]
        for i in idxs:
            for j in idxs:
                if i != j:
                    rows.append(i)
                    cols.append(j)
    n = len(tags)
    if not rows:
        return sp.csr_matrix((n, n))
    data = np.ones(len(rows), dtype=np.int8)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def rcm_renumber(model) -> None:
    """Apply reverse Cuthill-McKee renumbering to the model's DOFs.

    Idempotent — calling twice on the same model has no further effect
    beyond the first call's reordering.

    Notes
    -----
    Operates on the node-level adjacency graph (cheaper to build than the
    full DOF graph). Within a node, DOFs remain in their original order; we
    only reorder *which* node's DOFs are numbered first.
    """
    A = _node_adjacency(model)
    if A.nnz == 0:
        # no elements — nothing to reorder
        model.number_dofs()
        return
    perm = np.asarray(reverse_cuthill_mckee(A, symmetric_mode=True), dtype=np.int64)
    tags = list(model.nodes.keys())
    eq = 0
    for k in perm:
        node = model.node(tags[int(k)])
        for i in range(node.ndf):
            if node.fixity[i]:
                node.eqn[i] = -1
            else:
                node.eqn[i] = eq
                eq += 1
    model._neq = eq
    model._numbered = True
