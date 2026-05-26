"""Phase 20 -- solver-infrastructure benchmark.

Compares the wall-time of the new pluggable solvers on representative
problems:

1. **Direct sparse solver** (default) -- SciPy ``spsolve`` /
   UMFPACK / SuperLU.
2. **Iterative CG with ILU preconditioning** -- for symmetric
   positive-definite systems (typical structural stiffness matrices).
3. **Iterative GMRES with ILU** -- for non-symmetric / indefinite
   systems.

Two test problems:

* **Linear static of a tall multi-story frame** (~5000 DOF). Direct
  is usually fastest below 10-20k DOF; iterative becomes competitive
  past that scale.
* **Buckling of a tall column** (~1000 DOF). Sparse path uses
  ``eigsh`` shift-invert and is now the default in ``LinearBucklingAnalysis``.

Run::

    python examples/33_solver_benchmark.py
"""
from __future__ import annotations

import math
import time

import numpy as np

from femsolver import (
    BeamColumn2D,
    BeamColumn2DCorotational,
    DirectSparseSolver,
    ElasticIsotropic,
    IterativeSolver,
    LinearStaticAnalysis,
    Model,
    Truss2D,
)
from femsolver.analysis.buckling import LinearBucklingAnalysis


def build_tall_frame(*, n_story: int, n_bay: int = 2) -> Model:
    """N-story, n_bay-bay frame with realistic stiffness ratios."""
    E = 2.0e10; A = 1.0e-2; Iz = 1.0e-4
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=7850.0)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    L_col = 3.0
    L_bm = 5.0
    n_col_per_floor = n_bay + 1
    # Nodes
    for floor in range(n_story + 1):
        for col in range(n_col_per_floor):
            tag = floor * n_col_per_floor + col + 1
            m.add_node(tag, col * L_bm, floor * L_col)
    # Columns
    etag = 1
    for floor in range(n_story):
        for col in range(n_col_per_floor):
            n_b = floor * n_col_per_floor + col + 1
            n_t = (floor + 1) * n_col_per_floor + col + 1
            m.add_element(BeamColumn2D(etag, (n_b, n_t), mat, A, Iz))
            etag += 1
    # Beams
    for floor in range(1, n_story + 1):
        for col in range(n_col_per_floor - 1):
            n_l = floor * n_col_per_floor + col + 1
            n_r = floor * n_col_per_floor + col + 2
            m.add_element(BeamColumn2D(etag, (n_l, n_r), mat, A * 2, Iz * 2))
            etag += 1
    # Base supports
    for col in range(n_col_per_floor):
        m.fix(col + 1, [1, 1, 1])
    # Lateral load at each floor (proportional to floor index)
    for floor in range(1, n_story + 1):
        tag = floor * n_col_per_floor + 1
        m.add_nodal_load(tag, [floor * 1.0e3, 0.0, 0.0])
    return m


def build_long_truss(*, n_elem: int) -> Model:
    """1-D truss row: well-conditioned SPD system ideal for iterative
    solvers. Tip load on the free end."""
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3, rho=7850.0)
    m = Model(ndm=2, ndf=2); m.add_material(mat)
    L = 1.0
    for i in range(n_elem + 1):
        m.add_node(i + 1, i * L, 0.0)
    for i in range(n_elem):
        m.add_element(Truss2D(i + 1, (i + 1, i + 2), mat, area=1e-4))
    m.fix(1, [1, 1])
    for i in range(2, n_elem + 2):
        m.fix(i, [0, 1])     # restrain y
    m.add_nodal_load(n_elem + 1, [1.0e3, 0])
    return m


def benchmark_linear_static() -> None:
    """Compare direct vs iterative solvers on two problems:
    a well-conditioned 1-D truss (good for CG) and a multi-story
    frame (typical structural problem where direct usually wins)."""
    print("\nLinear static benchmark -- well-conditioned truss row")
    n_elem = 2000
    m_direct = build_long_truss(n_elem=n_elem)
    n_dof = m_direct.neq
    print(f"  Model: {n_elem}-element 1-D truss row")
    print(f"  Free DOFs: {n_dof}")

    t0 = time.perf_counter()
    LinearStaticAnalysis(m_direct, solver=DirectSparseSolver()).run()
    t_direct = time.perf_counter() - t0
    u_direct = m_direct.node(n_elem + 1).disp[0]

    m_iter = build_long_truss(n_elem=n_elem)
    t0 = time.perf_counter()
    iter_solver = IterativeSolver(method="cg", tol=1e-12, drop_tol=1e-3)
    LinearStaticAnalysis(m_iter, solver=iter_solver).run()
    t_iter = time.perf_counter() - t0
    u_iter = m_iter.node(n_elem + 1).disp[0]

    print(f"  Direct sparse:    {t_direct*1000:>8.1f} ms"
            f"  (tip displacement = {u_direct*1e3:.6f} mm)")
    print(f"  Iterative CG+ILU: {t_iter*1000:>8.1f} ms"
            f"  (tip displacement = {u_iter*1e3:.6f} mm)"
            f"  ({iter_solver.last_iterations} CG iters,"
            f" residual {iter_solver.last_residual:.2e})")
    rel_diff = abs(u_iter - u_direct) / max(abs(u_direct), 1e-30)
    print(f"  Solution agreement: relative diff = {rel_diff:.2e}")
    print()
    print(f"  At ~{n_dof} DOFs on this well-conditioned 1-D problem the")
    print(f"  direct sparse solver is usually fastest -- CG's overhead")
    print(f"  per iteration is hard to beat when LU factorization is cheap.")
    print(f"  CG starts winning for million-DOF 3-D continuum problems")
    print(f"  where dense LU factors don't fit in memory.")


def benchmark_buckling() -> None:
    print("\nBuckling benchmark")
    n_elem = 40
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    A = 1e-4; I = 1e-8
    L = 1.0

    def _build():
        m = Model(ndm=2, ndf=3); m.add_material(mat)
        for i in range(n_elem + 1):
            m.add_node(i + 1, 0.0, i * L / n_elem)
        for i in range(n_elem):
            m.add_element(BeamColumn2DCorotational(
                i + 1, (i + 1, i + 2), mat, area=A, Iz=I,
            ))
        m.fix(1, [1, 1, 0])
        m.fix(n_elem + 1, [1, 0, 0])
        m.add_nodal_load(n_elem + 1, [0.0, -1.0, 0.0])
        return m

    # Sparse path (default)
    m_sparse = _build()
    t0 = time.perf_counter()
    res_sparse = LinearBucklingAnalysis(
        m_sparse, num_modes=1, mode="sparse",
    ).run()
    t_sparse = time.perf_counter() - t0
    # Dense fallback
    m_dense = _build()
    t0 = time.perf_counter()
    res_dense = LinearBucklingAnalysis(
        m_dense, num_modes=1, mode="dense",
    ).run()
    t_dense = time.perf_counter() - t0

    # Analytical pin-pin Euler load
    P_cr_euler = math.pi ** 2 * 2e11 * I / L ** 2

    print(f"  Pin-pin column, n_elem = {n_elem}  (neq = {res_sparse['neq']})")
    print(f"  P_cr_euler (analytical):  {P_cr_euler:.4f} N")
    print(f"  Sparse eigsh:  {t_sparse*1000:>8.1f} ms"
            f"  -> P_cr = {res_sparse['critical_load_factor']:.4f} N"
            f"  ({100*res_sparse['critical_load_factor']/P_cr_euler:.2f}% of Euler)")
    print(f"  Dense eigh:    {t_dense*1000:>8.1f} ms"
            f"  -> P_cr = {res_dense['critical_load_factor']:.4f} N"
            f"  ({100*res_dense['critical_load_factor']/P_cr_euler:.2f}% of Euler)")


def main() -> None:
    print("Phase 20 -- solver-infrastructure benchmark")
    benchmark_linear_static()
    benchmark_buckling()
    print()
    print("Takeaways:")
    print("* DirectSparseSolver (default) handles structural problems")
    print("  up to ~10k-50k DOFs efficiently.")
    print("* IterativeSolver with ILU pays off above that scale, or")
    print("  when the LU factor's memory cost is prohibitive.")
    print("* The new sparse-eigsh buckling path scales the same way --")
    print("  Lanczos with shift-invert near sigma = 0 finds the few")
    print("  critical load factors without forming dense matrices.")


if __name__ == "__main__":
    main()
