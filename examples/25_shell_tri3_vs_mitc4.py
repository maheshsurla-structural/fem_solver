"""ShellTri3 vs ShellMITC4 -- triangular vs quadrilateral shells.

This example compares the two shell elements on a simply-supported
plate under a central point load, for a moderately thick plate
(L/t = 10 and L/t = 20). The sweep highlights:

* **ShellMITC4** -- robust across all L/t ratios thanks to MITC tying.
* **ShellTri3** -- excellent for thick to moderate shells; starts to
  lock as L/t grows past ~30. This is the textbook limitation of a
  Reissner-Mindlin triangle with reduced-point shear and *no* edge
  tying / bubble enrichment.

The headline guidance:

    For thin (L/t > 30) shells: use ShellMITC4 on a quad mesh.
    For moderate-thickness shells on unstructured meshes:
      ShellTri3 is fast and well-behaved.

Run::

    python examples/25_shell_tri3_vs_mitc4.py
"""
from __future__ import annotations

import math

from femsolver import (
    ElasticIsotropic,
    Model,
    ShellMITC4,
    ShellTri3,
)
from femsolver.analysis.linear_static import LinearStaticAnalysis


def build_plate(N: int, *, t: float, L: float, E: float, nu: float,
                element_type: str = "quad"):
    mat = ElasticIsotropic(1, E=E, nu=nu)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    nL = N + 1
    for j in range(nL):
        for i in range(nL):
            m.add_node(j * nL + i + 1, i * L / N, j * L / N, 0.0)
    etag = 1
    for j in range(N):
        for i in range(N):
            n1 = j * nL + i + 1; n2 = n1 + 1; n3 = n2 + nL; n4 = n1 + nL
            if element_type == "quad":
                m.add_element(ShellMITC4(etag, (n1, n2, n3, n4), mat, t))
                etag += 1
            else:
                m.add_element(ShellTri3(etag, (n1, n2, n3), mat, t)); etag += 1
                m.add_element(ShellTri3(etag, (n1, n3, n4), mat, t)); etag += 1
    for j in range(nL):
        for i in range(nL):
            if i in (0, N) or j in (0, N):
                m.fix(j * nL + i + 1, [0, 0, 1, 0, 0, 0])
    m.fix(1, [1, 1, 1, 0, 0, 0])
    m.fix(N + 1, [0, 1, 1, 0, 0, 0])
    return m


def main() -> None:
    E, nu = 2.0e11, 0.3
    L = 1.0
    P = 1.0

    print(f"\nSS square plate under center point load -- shell element comparison")
    print(f"  L = {L} m,  E = {E:g} Pa,  nu = {nu}")
    print(f"  Closed forms:")
    print(f"    Timoshenko thin plate:  w_max = 0.01160 * P L^2 / D")
    print(f"    where D = E t^3 / (12 (1-nu^2)) is the flexural rigidity.")
    print()

    for t in (0.1, 0.05, 0.02, 0.01, 0.005):
        D = E * t ** 3 / (12.0 * (1.0 - nu ** 2))
        w_thin = 0.01160 * P * L ** 2 / D
        print(f"  L/t = {int(L/t):>4d}  (t = {t})")
        print(f"    Thin-plate Timoshenko w_max = {w_thin:.3e} m")
        print(f"    {'N':>3s}   {'Tri3 ratio':>12s}   {'MITC4 ratio':>13s}")
        for N in (8, 12, 16):
            m_t = build_plate(N=N, t=t, L=L, E=E, nu=nu, element_type="tri")
            m_q = build_plate(N=N, t=t, L=L, E=E, nu=nu, element_type="quad")
            ic = (N // 2) * (N + 1) + N // 2 + 1
            m_t.add_nodal_load(ic, [0, 0, -P, 0, 0, 0])
            m_q.add_nodal_load(ic, [0, 0, -P, 0, 0, 0])
            LinearStaticAnalysis(m_t).run()
            LinearStaticAnalysis(m_q).run()
            w_tri = -m_t.node(ic).disp[2]
            w_quad = -m_q.node(ic).disp[2]
            print(f"    {N:>3d}   {w_tri / w_thin:>12.4f}   {w_quad / w_thin:>13.4f}")
        print()

    print(f"  Reading the table:")
    print(f"  * L/t = 10 (thick): both elements predict ~1.18 * thin-plate")
    print(f"    deflection -- the extra ~18% is the Mindlin transverse")
    print(f"    shear contribution Timoshenko's thin formula omits.")
    print(f"  * L/t = 20-50: ShellTri3 starts to under-predict; the")
    print(f"    reduced-point shear no longer fully cures locking.")
    print(f"  * L/t = 100+: ShellTri3 severely locked; ShellMITC4 stays")
    print(f"    accurate thanks to the MITC4 tying scheme.")
    print(f"  * Practical guidance: use ShellMITC4 with quad meshes for")
    print(f"    thin shells; use ShellTri3 only for unstructured-mesh")
    print(f"    work on thick (L/t < 20) shells.")


if __name__ == "__main__":
    main()
