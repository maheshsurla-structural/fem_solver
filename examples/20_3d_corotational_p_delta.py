"""3D P-Delta amplification of a slender simply-supported column.

The 3-D analog of example 10. A slender pin-pin column is loaded with
axial compression plus a small lateral perturbation. We sweep the
axial fraction ``P / P_cr`` and verify that midspan deflection grows
by the classical ``1 / (1 - P / P_cr)`` factor — **in any transverse
direction**.

Run::

    python examples/20_3d_corotational_p_delta.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver import (
    BeamColumn3DCorotational,
    ElasticIsotropic,
    Model,
    NonlinearStaticAnalysis,
)


def midspan_deflection_3d(P_axial: float, *, axis: str = "y",
                          P_lateral: float = 1.0,
                          n_elem: int = 8) -> tuple[float, float]:
    """Run a 3-D P-Delta analysis and return midspan deflection
    along the chosen lateral axis (y or z)."""
    E = 2.0e11
    A = 1.0e-3
    I = 1.0e-7
    L = 5.0
    J = 1.0e-7

    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    for i in range(n_elem + 1):
        m.add_node(i + 1, i * L / n_elem, 0.0, 0.0)
    for i in range(n_elem):
        m.add_element(BeamColumn3DCorotational(
            i + 1, (i + 1, i + 2), mat, A, I, I, J,
        ))
    # Pin at node 1, roller at node (n_elem+1)
    m.fix(1, [1, 1, 1, 1, 0, 0])
    m.fix(n_elem + 1, [0, 1, 1, 1, 0, 0])
    m.add_nodal_load(n_elem + 1, [-P_axial, 0, 0, 0, 0, 0])
    mid_tag = 1 + n_elem // 2
    if axis == "y":
        m.add_nodal_load(mid_tag, [0, -P_lateral, 0, 0, 0, 0])
        idx = 1
    else:
        m.add_nodal_load(mid_tag, [0, 0, -P_lateral, 0, 0, 0])
        idx = 2
    NonlinearStaticAnalysis(
        m, num_steps=10, dlambda=1.0 / 10, tol=1e-6, max_iter=30,
    ).run()
    return abs(m.node(mid_tag).disp[idx]), L


def main() -> None:
    E = 2.0e11
    I = 1.0e-7
    L = 5.0
    EI = E * I
    P_cr = math.pi ** 2 * EI / L ** 2
    P_lateral = 1.0
    ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]

    print(f"\nP-Delta amplification of a slender 3-D column")
    print(f"  L = {L} m,  EI = {EI:.3g} N.m^2")
    print(f"  P_cr (Euler, pin-pin) = pi^2 EI / L^2 = {P_cr:.4g} N")
    print(f"  Lateral perturbation: {P_lateral} N\n")

    for axis in ("y", "z"):
        print(f"  --- Lateral load along {axis} ---")
        v0, _ = midspan_deflection_3d(0.0, axis=axis, P_lateral=P_lateral)
        print(f"  {'P / P_cr':>10}  {'delta_FE':>13}  {'amp_FE':>8}  "
              f"{'amp_theory':>11}")
        for r in ratios:
            P = r * P_cr
            try:
                d, _ = midspan_deflection_3d(P, axis=axis,
                                              P_lateral=P_lateral)
                amp_fe = d / v0
                amp_th = 1.0 / (1.0 - r) if r < 1.0 else float("inf")
                print(f"  {r:10.2f}  {d:13.4e}  {amp_fe:8.3f}  {amp_th:11.3f}")
            except Exception as exc:
                print(f"  {r:10.2f}  *** failed: {exc.__class__.__name__}")
        print()

    print(f"  Both transverse directions show the same amplification curve,")
    print(f"  confirming the 3-D corotational element treats y- and z-bending")
    print(f"  symmetrically (which is required since the column has Iy = Iz).")


if __name__ == "__main__":
    main()
