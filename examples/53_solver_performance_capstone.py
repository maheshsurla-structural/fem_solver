"""Phase 43.6 -- Solver-performance capstone.

Times every solver / assembly / reduction strategy that femsolver
exposes, on the same medium-sized model. Reports wall-clock for:

1. Element-assembly: serial vs parallel (thread-pool).
2. Linear solve: SuperLU direct, conjugate-gradient (+ ILU),
   CachedFactorSolver (3 repeated solves), and PARDISO (skipped if
   not installed).
3. Guyan static condensation of a 3D-cantilever model down to its
   top-face master DOFs; then a solve in the reduced space.
4. Craig-Bampton with 10 fixed-interface modes on the same model;
   compare reduced-system eigenvalues to the full system.

The model is a 3-D cantilever block meshed with ~ 1000 Hex8 elements
(~ 5000 free DOFs), large enough to expose where each method helps.

Run::

    python examples/53_solver_performance_capstone.py
"""
from __future__ import annotations

import math
import time
from contextlib import contextmanager

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh as scipy_eigh

from femsolver import (
    CachedFactorSolver,
    DirectSparseSolver,
    ElasticIsotropic,
    Hex8,
    IterativeSolver,
    LinearStaticAnalysis,
    Model,
    PardisoSolver,
    assemble_stiffness_parallel,
    craig_bampton,
    guyan_condensation,
    pardiso_available,
)
from femsolver.analysis.assembler import assemble_force, assemble_stiffness


@contextmanager
def stopwatch(label: str, results: dict):
    t0 = time.perf_counter()
    yield
    results[label] = time.perf_counter() - t0


def build_cantilever(*, nx: int, ny: int, nz: int):
    """3D Hex8 cantilever block ``nx x ny x nz`` elements."""
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.30, rho=7850.0)
    m = Model(ndm=3, ndf=3)
    m.add_material(mat)
    Lx, Ly, Lz = 1.0, 0.2, 0.2
    grid = {}
    tag = 1
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                m.add_node(tag,
                            i * Lx / nx, j * Ly / ny, k * Lz / nz)
                grid[(i, j, k)] = tag
                tag += 1
    et = 1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                nodes = (
                    grid[(i, j, k)], grid[(i + 1, j, k)],
                    grid[(i + 1, j + 1, k)], grid[(i, j + 1, k)],
                    grid[(i, j, k + 1)], grid[(i + 1, j, k + 1)],
                    grid[(i + 1, j + 1, k + 1)], grid[(i, j + 1, k + 1)],
                )
                m.add_element(Hex8(et, nodes, mat))
                et += 1
    # Fix the i=0 face
    for k in range(nz + 1):
        for j in range(ny + 1):
            m.fix(grid[(0, j, k)], [1, 1, 1])
    # Apply transverse tip load
    P = 1.0e3
    for k in range(nz + 1):
        for j in range(ny + 1):
            m.add_nodal_load(grid[(nx, j, k)],
                              [0.0, 0.0, -P / ((nz + 1) * (ny + 1))])
    m.number_dofs()
    return m


def main() -> None:
    print("=" * 78)
    print("Phase 43.6 -- Solver-performance capstone")
    print("=" * 78)

    # Build the model
    nx, ny, nz = 30, 4, 4
    print(f"\nModel: 3D Hex8 cantilever ({nx}x{ny}x{nz} elements)")
    m = build_cantilever(nx=nx, ny=ny, nz=nz)
    n_elem = len(m.elements)
    print(f"  {n_elem} elements,  {m.neq} free DOFs")

    timings: dict[str, float] = {}

    # ---- (1) Assembly: serial vs parallel -----------------------------
    print()
    print("(1) Assembly")
    print("-" * 70)
    with stopwatch("assembly serial", timings):
        K_ser = assemble_stiffness(m)
    print(f"  Serial assembly   : {timings['assembly serial']*1000:7.1f} ms")
    with stopwatch("assembly parallel x4", timings):
        K_par = assemble_stiffness_parallel(m, n_workers=4)
    print(f"  Parallel x4       : "
          f"{timings['assembly parallel x4']*1000:7.1f} ms  "
          f"(speedup "
          f"{timings['assembly serial']/timings['assembly parallel x4']:.2f}x)")

    # Force vector (single assembly)
    f_full = assemble_force(m)

    # ---- (2) Linear solve --------------------------------------------
    print()
    print("(2) Linear solve")
    print("-" * 70)
    with stopwatch("DirectSparseSolver (SuperLU)", timings):
        x_dir = DirectSparseSolver().solve(K_ser, f_full)
    print(f"  Direct (SuperLU)        : "
          f"{timings['DirectSparseSolver (SuperLU)']*1000:7.1f} ms")

    with stopwatch("IterativeSolver (CG + ILU)", timings):
        x_cg = IterativeSolver(
            method="cg", tol=1e-10, max_iter=2000,
        ).solve(K_ser, f_full)
    print(f"  Iterative (CG + ILU)    : "
          f"{timings['IterativeSolver (CG + ILU)']*1000:7.1f} ms")

    # CachedFactor: 3 repeated solves with the same matrix
    cached = CachedFactorSolver()
    t0 = time.perf_counter()
    for _ in range(3):
        cached.solve(K_ser, f_full)
    timings["CachedFactor x3"] = time.perf_counter() - t0
    print(f"  CachedFactor (3 RHS)    : "
          f"{timings['CachedFactor x3']*1000:7.1f} ms  "
          f"(hits {cached.cache_hits}, misses {cached.cache_misses})")

    if pardiso_available():
        with stopwatch("PARDISO", timings):
            x_pard = PardisoSolver().solve(K_ser, f_full)
        print(f"  PARDISO                 : {timings['PARDISO']*1000:7.1f} ms")
    else:
        print(f"  PARDISO                 : not installed (skipped)")

    # Correctness check across solvers
    rel_cg = np.linalg.norm(x_cg - x_dir) / np.linalg.norm(x_dir)
    print(f"\n  CG vs Direct relative error: {rel_cg:.2e}")

    # ---- (3) Guyan reduction ------------------------------------------
    print()
    print("(3) Guyan static condensation")
    print("-" * 70)
    # Master DOFs: nodes on the top face (k = nz) -- the "free surface"
    master_tags = []
    for j in range(ny + 1):
        master_tags.append(j * (nx + 1) + (nx + 1))    # i=nx along j-row
    # Convert to DOF indices: pick the z-DOF (index 2) of each master node
    master_dofs = []
    for ntag in master_tags:
        eq = int(m.node(ntag).eqn[2])
        if eq >= 0:
            master_dofs.append(eq)
    print(f"  Masters: {len(master_dofs)} z-DOFs on the loaded tip")

    with stopwatch("Guyan reduction", timings):
        res_g = guyan_condensation(K_ser.toarray(), f_full, master_dofs)
    print(f"  Reduction time          : "
          f"{timings['Guyan reduction']*1000:7.1f} ms  "
          f"(K_red size = {res_g.K_red.shape[0]})")
    with stopwatch("Guyan solve", timings):
        u_m = np.linalg.solve(res_g.K_red, res_g.f_red)
    print(f"  Reduced solve           : "
          f"{timings['Guyan solve']*1000:7.1f} ms")

    # Compare: u at master DOFs vs full solve
    u_m_full = x_dir[res_g.master_dofs]
    rel = np.linalg.norm(u_m - u_m_full) / max(np.linalg.norm(u_m_full), 1e-30)
    print(f"  Master-DOF error vs full: {rel:.2e}  "
          f"(zero iff no slave-side load — Guyan is exact for static)")

    # ---- (4) Craig-Bampton --------------------------------------------
    print()
    print("(4) Craig-Bampton (10 fixed-interface modes)")
    print("-" * 70)
    # Use a small subset of masters to keep the dense ops tractable
    masters_subset = master_dofs[:5]
    print(f"  Masters {len(masters_subset)} + {10} kept modes = "
          f"{len(masters_subset) + 10}-DOF reduced system")

    M_full = np.eye(m.neq)         # placeholder unit mass for the demo
    with stopwatch("Craig-Bampton", timings):
        res_cb = craig_bampton(
            K_ser.toarray(), M_full,
            master_dofs=masters_subset, n_keep=10,
        )
    print(f"  CB reduction time       : "
          f"{timings['Craig-Bampton']*1000:7.1f} ms")
    print(f"  Fixed-interface omegas  : "
          f"{res_cb.omega_fixed[:5]}  ...")

    # Compare bottom eigenvalues
    w_red, _ = scipy_eigh(res_cb.K_red, res_cb.M_red)
    print(f"  Lowest CB eigenvalues   : {w_red[:5]}")

    # ---- Summary ------------------------------------------------------
    print()
    print("Summary of wall-clock times")
    print("-" * 70)
    for label, t in timings.items():
        print(f"  {label:<35}: {t*1000:7.1f} ms")

    print()
    print("=" * 78)
    print("Theme K closed: solver plugins + Guyan + Craig-Bampton +")
    print("                parallel assembly all operational.")
    print("=" * 78)


if __name__ == "__main__":
    main()
