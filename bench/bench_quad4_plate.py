"""Microbenchmark: NxN Quad4 plate under uniform tip tension.

Measures wall-clock for the four phases of a linear-static run on a
configurable mesh:

  - DOF numbering
  - global K assembly
  - global F assembly
  - sparse direct solve
  - element response recovery + reactions

Usage::

    python bench/bench_quad4_plate.py            # default 80x80
    python bench/bench_quad4_plate.py 120        # 120x120 mesh
    python bench/bench_quad4_plate.py 80 --rcm   # try RCM numbering
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from femsolver import (
    ElasticIsotropic,
    LinearStaticAnalysis,
    Model,
    Quad4,
)
from femsolver.analysis.assembler import (
    assemble_force,
    assemble_reactions,
    assemble_stiffness,
)


def build_plate(n: int, *, E: float = 2.0e11, nu: float = 0.3, t: float = 0.01) -> Model:
    """Square plate, n x n quad mesh, left edge fixed, uniform tension on right."""
    nx, ny = n, n
    m = Model(ndm=2, ndf=2)
    mat = ElasticIsotropic(1, E=E, nu=nu)
    m.add_material(mat)
    # nodes
    tag = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            m.add_node(tag, float(i), float(j))
            tag += 1

    def nidx(i, j):
        return j * (nx + 1) + i + 1

    # elements
    etag = 1
    for j in range(ny):
        for i in range(nx):
            n1 = nidx(i, j)
            n2 = nidx(i + 1, j)
            n3 = nidx(i + 1, j + 1)
            n4 = nidx(i, j + 1)
            m.add_element(Quad4(etag, (n1, n2, n3, n4), mat, thickness=t))
            etag += 1

    # boundary: fix left edge
    for j in range(ny + 1):
        m.fix(nidx(0, j), [1, 1])
    # uniform tension on right edge: split equally between nodes
    P = 1.0e6
    n_right = ny + 1
    for j in range(n_right):
        scale = 0.5 if (j == 0 or j == ny) else 1.0
        m.add_nodal_load(nidx(nx, j), [P / ny * scale, 0.0])
    return m


def _bench(n: int, *, rcm: bool = False) -> dict:
    print(f"--- {n}x{n} Quad4 plate ---")
    t0 = time.perf_counter()
    m = build_plate(n)
    t_build = time.perf_counter() - t0
    n_elem = len(m.elements)
    n_node = len(m.nodes)
    print(f"  build:        {t_build:8.3f} s   ({n_node} nodes, {n_elem} elements)")

    t0 = time.perf_counter()
    if rcm:
        from femsolver.numerics.dof_numbering import rcm_renumber
        rcm_renumber(m)
    else:
        m.number_dofs()
    t_number = time.perf_counter() - t0
    label = "RCM" if rcm else "default"
    print(f"  number DOFs:  {t_number:8.3f} s   (neq = {m.neq}, scheme = {label})")

    t0 = time.perf_counter()
    K, elem_K_list = assemble_stiffness(m, return_element_K=True)
    t_assemble_K = time.perf_counter() - t0
    print(f"  assemble K:   {t_assemble_K:8.3f} s   (nnz = {K.nnz})")

    t0 = time.perf_counter()
    F = assemble_force(m, elem_K_list=elem_K_list)
    t_assemble_F = time.perf_counter() - t0
    print(f"  assemble F:   {t_assemble_F:8.3f} s")

    t0 = time.perf_counter()
    from scipy.sparse.linalg import spsolve
    u = spsolve(K, F)
    t_solve = time.perf_counter() - t0
    print(f"  solve:        {t_solve:8.3f} s")

    t0 = time.perf_counter()
    for node in m.nodes.values():
        for i in range(node.ndf):
            eq = node.eqn[i]
            node.disp[i] = u[eq] if eq >= 0 else 0.0
    t_scatter = time.perf_counter() - t0

    t0 = time.perf_counter()
    for e in m.elements.values():
        e.recover()
    t_elem_recover = time.perf_counter() - t0

    t0 = time.perf_counter()
    assemble_reactions(m, elem_K_list=elem_K_list)
    t_reactions = time.perf_counter() - t0
    print(f"  scatter u:    {t_scatter:8.3f} s")
    print(f"  elem recover: {t_elem_recover:8.3f} s")
    print(f"  reactions:    {t_reactions:8.3f} s")
    total = (
        t_build + t_number + t_assemble_K + t_assemble_F + t_solve
        + t_scatter + t_elem_recover + t_reactions
    )
    print(f"  TOTAL:        {total:8.3f} s")
    t_recover = t_scatter + t_elem_recover + t_reactions
    return dict(
        n=n,
        neq=m.neq,
        nnz=K.nnz,
        build=t_build,
        number=t_number,
        assemble_K=t_assemble_K,
        assemble_F=t_assemble_F,
        solve=t_solve,
        recover=t_recover,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("n", nargs="?", type=int, default=80, help="mesh size N (NxN)")
    p.add_argument("--rcm", action="store_true", help="use RCM numbering")
    args = p.parse_args()
    _bench(args.n, rcm=args.rcm)
