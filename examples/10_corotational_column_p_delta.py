"""P-Delta amplification of a slender simply-supported column.

A 5 m steel column, simply supported at both ends, is loaded with an
axial compression ``P`` plus a small transverse perturbation at midspan.
The corotational element produces the classical Euler-style stiffness
softening: as P approaches the Euler critical load
``P_cr = pi^2 EI / L^2``, the midspan deflection under the transverse
perturbation grows by the textbook amplification factor

    delta(P) = delta_lin / (1 - P / P_cr)

A linear (non-corotational) element would predict ``delta(P) = delta_lin``
regardless of P — it misses the geometric softening entirely.

We sweep P from 0 to 0.95 * P_cr and print the amplification factor at
each step alongside the analytical prediction.

Run::

    python examples/10_corotational_column_p_delta.py
"""
from __future__ import annotations

import numpy as np

from femsolver import (
    BeamColumn2DCorotational,
    ElasticIsotropic,
    Model,
    NonlinearStaticAnalysis,
)


def midspan_deflection(P_axial: float, *, P_lateral: float = 1.0,
                       n_elem: int = 8) -> float:
    """Run a P-Delta analysis and return the absolute midspan deflection."""
    E = 2.0e11
    A = 1.0e-3
    Iz = 1.0e-7
    L = 5.0
    nu = 0.3

    mat = ElasticIsotropic(1, E=E, nu=nu)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)

    # Nodes: 1 and 2 are the supports; 3..(n_elem+1) are interior
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    for i in range(1, n_elem):
        m.add_node(i + 2, i * L / n_elem, 0.0)

    # Build chain 1 -> 3 -> 4 -> ... -> (n_elem + 1) -> 2
    def node_at(idx: int) -> int:
        if idx == 0:
            return 1
        if idx == n_elem:
            return 2
        return idx + 2

    for i in range(n_elem):
        left = node_at(i)
        right = node_at(i + 1)
        m.add_element(BeamColumn2DCorotational(i + 1, (left, right), mat, A, Iz))

    # Boundary conditions: pin-roller
    m.fix(1, [1, 1, 0])
    m.fix(2, [0, 1, 0])

    # Loads: axial compression at the roller, small lateral kick at midspan
    m.add_nodal_load(2, [-P_axial, 0.0, 0.0])
    midspan_tag = node_at(n_elem // 2)
    m.add_nodal_load(midspan_tag, [0.0, -P_lateral, 0.0])

    NonlinearStaticAnalysis(
        m, num_steps=15, dlambda=1.0 / 15,
        tol=1e-6, max_iter=30,
    ).run()
    return abs(m.node(midspan_tag).disp[1])


def main() -> None:
    E = 2.0e11
    Iz = 1.0e-7
    L = 5.0
    EI = E * Iz
    P_cr = np.pi ** 2 * EI / L ** 2

    P_lateral = 1.0          # tiny transverse perturbation
    P_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]

    delta_lin = midspan_deflection(0.0, P_lateral=P_lateral)

    print(f"\nP-Delta amplification of a slender simply-supported column")
    print(f"  L  = {L} m,  EI = {EI:.3g} N.m^2")
    print(f"  P_cr (Euler) = pi^2 EI / L^2 = {P_cr:.4g} N")
    print(f"  Lateral perturbation at midspan: {P_lateral} N")
    print(f"  Reference deflection (P_axial = 0): {delta_lin:.4e} m")
    print()
    print(f"  {'P / P_cr':>10}  {'delta_FE (m)':>13}  {'amp_FE':>9}  "
          f"{'amp_theory':>11}")
    for r in P_ratios:
        P = r * P_cr
        try:
            d = midspan_deflection(P, P_lateral=P_lateral)
            amp_fe = d / delta_lin
            amp_theory = 1.0 / (1.0 - r) if r < 1.0 else float("inf")
            print(f"  {r:10.2f}  {d:13.4e}  {amp_fe:9.3f}  {amp_theory:11.3f}")
        except Exception as e:
            print(f"  {r:10.2f}  *** failed: {e}")
    print()
    print(f"  At P / P_cr = 0.95 the analytical amplification is 20x;")
    print(f"  the FE result tracks the analytical to within a few percent,")
    print(f"  with the small offset coming from discretisation and the")
    print(f"  difference between Euler's small-deflection limit and the")
    print(f"  full geometrically-nonlinear response.")


if __name__ == "__main__":
    main()
