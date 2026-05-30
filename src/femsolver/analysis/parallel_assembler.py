"""Parallel element-level assembly of K, M, f using a thread pool.

For models with many elements (≥ a few thousand), the per-element
``K_global()`` evaluation can dominate assembly cost. The default
serial assembler walks one element at a time; this module fans the
per-element work out across a thread pool and rejoins the per-chunk
COO triplets.

We use **threads** rather than processes because:

* NumPy releases the GIL during BLAS calls, so threads provide real
  parallelism for compute-heavy element matrices (Hex8, MITC4, fiber
  beams).
* Threads share the model graph by reference -- no pickling overhead.
* The COO merge is a cheap NumPy concatenation, not a heavy Python
  reduction.

The serial path remains in :func:`femsolver.analysis.assembler.assemble_stiffness`;
this module provides a drop-in :func:`assemble_stiffness_parallel`
that can be selected when the model is large enough to benefit.
"""
from __future__ import annotations

import concurrent.futures as cf
import os

import numpy as np
import scipy.sparse as sp


def _element_chunks(elements: list, n_chunks: int) -> list[list]:
    """Split ``elements`` into ``n_chunks`` roughly-equal slices."""
    n = len(elements)
    if n_chunks <= 1 or n_chunks >= n:
        return [elements] if elements else []
    chunk_size = (n + n_chunks - 1) // n_chunks
    return [elements[i : i + chunk_size]
            for i in range(0, n, chunk_size)]


def _process_chunk(chunk_with_model):
    """Compute (rows, cols, vals, elem_K_list_chunk) for one element chunk."""
    chunk, model = chunk_with_model
    total = 0
    K_list_local: list[tuple] = []
    for e in chunk:
        Ke = e.K_global()
        dofs = model.element_dof_map(e)
        K_list_local.append((e, dofs, Ke))
        total += dofs.size * dofs.size
    rows = np.empty(total, dtype=np.int64)
    cols = np.empty(total, dtype=np.int64)
    vals = np.empty(total, dtype=float)
    pos = 0
    for (_, dofs, Ke) in K_list_local:
        n = dofs.size
        nn = n * n
        rows[pos : pos + nn] = np.repeat(dofs, n)
        cols[pos : pos + nn] = np.tile(dofs, n)
        vals[pos : pos + nn] = np.asarray(Ke, dtype=float).ravel()
        pos += nn
    return rows, cols, vals, K_list_local


def assemble_stiffness_parallel(
    model,
    *,
    n_workers: int | None = None,
    return_element_K: bool = False,
):
    """Parallel sibling of
    :func:`femsolver.analysis.assembler.assemble_stiffness`.

    Parameters
    ----------
    model : Model
    n_workers : int, optional
        Number of worker threads. Defaults to ``os.cpu_count() // 2``
        (leave room for other concurrent work in the process).
    return_element_K : bool, default False
        If True, return ``[(element, dof_map, K_e), ...]`` alongside
        the assembled matrix (mirrors the serial API).

    Returns
    -------
    csc_matrix or (csc_matrix, list)

    Notes
    -----
    Speedup is only meaningful when the per-element cost is large and
    the model has thousands of elements. For small models the
    threading overhead dominates and the serial path is faster.
    """
    neq = model.neq
    elements = list(model.elements.values())
    if not elements:
        empty = sp.csc_matrix((neq, neq))
        return (empty, []) if return_element_K else empty
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) // 2)
    n_workers = max(1, min(n_workers, len(elements)))
    chunks = _element_chunks(elements, n_workers)
    inputs = [(c, model) for c in chunks]

    if n_workers == 1:
        # Skip the executor entirely
        results = [_process_chunk(inputs[0])]
    else:
        with cf.ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_process_chunk, inputs))

    # Merge per-chunk COO triples
    total = sum(r[0].size for r in results)
    rows_all = np.empty(total, dtype=np.int64)
    cols_all = np.empty(total, dtype=np.int64)
    vals_all = np.empty(total, dtype=float)
    elem_K_list: list[tuple] = []
    pos = 0
    for (rows, cols, vals, K_list_chunk) in results:
        sz = rows.size
        rows_all[pos : pos + sz] = rows
        cols_all[pos : pos + sz] = cols
        vals_all[pos : pos + sz] = vals
        pos += sz
        elem_K_list.extend(K_list_chunk)

    mask = (rows_all >= 0) & (cols_all >= 0)
    if not mask.any():
        K = sp.csc_matrix((neq, neq))
    else:
        K = sp.coo_matrix(
            (vals_all[mask], (rows_all[mask], cols_all[mask])),
            shape=(neq, neq),
        ).tocsc()

    if return_element_K:
        return K, elem_K_list
    return K
