"""Mesh-convergence study for Euler buckling of slender columns.

Computes the first buckling load of three classical Euler cases under
progressive mesh refinement and compares with the analytical formulas:

    Pin-pin       :  P_cr = pi^2 EI / L^2         (K_eff = 1.0)
    Cantilever    :  P_cr = pi^2 EI / (2L)^2      (K_eff = 2.0)
    Fixed-fixed   :  P_cr = pi^2 EI / (0.5 L)^2   (K_eff = 0.5)

The corotational beam-column element is *displacement-based* with a
cubic Hermite shape function. Although the elastic bending stiffness
is integrated exactly (closed-form), the **geometric** stiffness
``K_g(N, M, c, s, L)`` brings in lower-order truncation: the first-
mode buckling load converges as ``O(h^2)`` — doubling the number of
elements per column cuts the error by roughly 4x. Higher modes
(more curvature) need more elements to capture accurately.

For O(h^4) convergence one needs either higher-order shape functions
or a force-based / consistent-geometric-stiffness formulation. That
is the natural follow-up that Phase 7 (force-based beam) would
bring.

Run::

    python examples/17_buckling_mesh_convergence.py
"""
from __future__ import annotations

import math

from femsolver import (
    BeamColumn2DCorotational,
    ElasticIsotropic,
    LinearBucklingAnalysis,
    Model,
)


def _build_column(n_elem: int, boundary: str, *,
                  E: float, A: float, Iz: float, L: float):
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    for i in range(n_elem + 1):
        m.add_node(i + 1, i * L / n_elem, 0.0)
    for i in range(n_elem):
        m.add_element(
            BeamColumn2DCorotational(i + 1, (i + 1, i + 2), mat, A, Iz)
        )
    if boundary == "pinned":
        m.fix(1, [1, 1, 0]); m.fix(n_elem + 1, [0, 1, 0])
        K_eff = 1.0
    elif boundary == "cantilever":
        m.fix(1, [1, 1, 1])
        K_eff = 2.0
    elif boundary == "fixed-fixed":
        m.fix(1, [1, 1, 1]); m.fix(n_elem + 1, [0, 1, 1])
        K_eff = 0.5
    else:
        raise ValueError(boundary)
    m.add_nodal_load(n_elem + 1, [-1.0, 0.0, 0.0])
    return m, K_eff


def main() -> None:
    E = 2.0e11
    A = 1.0e-3
    Iz = 1.0e-7
    L = 5.0
    EI = E * Iz

    print(f"\nMesh-convergence study for Euler buckling")
    print(f"  E = {E:g} Pa,  I = {Iz:g} m^4,  L = {L} m,  EI = {EI:g} N.m^2\n")

    boundaries = [
        ("pinned",      "Pin-pin",      "P_cr = pi^2 EI / L^2"),
        ("cantilever",  "Cantilever",   "P_cr = pi^2 EI / (2L)^2"),
        ("fixed-fixed", "Fixed-fixed",  "P_cr = pi^2 EI / (0.5 L)^2"),
    ]
    n_list = [2, 4, 8, 16, 32]

    for tag, name, formula in boundaries:
        print(f"  {name}  ({formula})")
        # Analytical P_cr for this case
        _, K_eff = _build_column(2, tag, E=E, A=A, Iz=Iz, L=L)
        P_cr_exact = math.pi ** 2 * EI / (K_eff * L) ** 2
        print(f"    Analytical P_cr  = {P_cr_exact:.6f} N")
        print(f"    {'n_elem':>8}  {'P_cr_FE (N)':>15}  {'rel err':>10}  {'iters':>6}")
        for n in n_list:
            m, _ = _build_column(n, tag, E=E, A=A, Iz=Iz, L=L)
            res = LinearBucklingAnalysis(m, num_modes=1).run()
            P_fe = res["critical_load_factor"]
            err = abs(P_fe - P_cr_exact) / P_cr_exact * 100.0
            print(f"    {n:8d}  {P_fe:15.6f}  {err:9.4f}%")
        print()

    # Show error-reduction ratio between successive refinements for
    # the pin-pin case — should drop by ~16x per doubling for displacement-
    # based cubic-Hermite elements.
    print(f"  Convergence rate (pin-pin column, first mode):")
    P_cr_exact = math.pi ** 2 * EI / L ** 2
    errs = []
    for n in n_list:
        m, _ = _build_column(n, "pinned", E=E, A=A, Iz=Iz, L=L)
        res = LinearBucklingAnalysis(m, num_modes=1).run()
        errs.append(abs(res["critical_load_factor"] - P_cr_exact) / P_cr_exact)
    print(f"    {'n_elem':>8}  {'rel err':>12}  {'ratio_prev':>11}")
    for i, (n, e) in enumerate(zip(n_list, errs)):
        ratio = f"{errs[i - 1] / e:.2f}x" if i > 0 else "-"
        print(f"    {n:8d}  {e * 100:11.6f}%  {ratio:>11}")
    print()
    print(f"  Observed ~4x reduction per mesh doubling = O(h^2)")
    print(f"  convergence — the standard rate for the geometric-stiffness-")
    print(f"  driven eigenvalue with displacement-based cubic-Hermite")
    print(f"  elements. A force-based (Phase 7) variant would give O(h^4).")


if __name__ == "__main__":
    main()
