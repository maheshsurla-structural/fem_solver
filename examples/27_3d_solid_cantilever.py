"""3-D solid cantilever -- Hex8 vs Tet4 vs Euler beam theory.

A cantilever block of dimensions ``L x b x h`` is fixed at one end
and loaded with a transverse point load distributed across the tip
face. The same problem is solved with three meshes:

* **Hex8** with full 2x2x2 Gauss integration -- the standard 3-D
  brick.
* **Tet4** with the 6-tet diagonal-fan decomposition of each cube.
* **Euler-Bernoulli beam theory** as the analytical reference
  (valid for slender beams, L/h >> 1).

The convergence study illustrates two textbook results:

1. Hex8 with full integration is notoriously stiff in bending due
   to spurious shear locking. Convergence is slow per DOF; the
   industry cure is incompatible-modes / B-bar / reduced-integration
   with hourglass control (future Phase 15.x).

2. Tet4 is even stiffer because it has only constant-strain
   kinematic behavior. It is best used for bulk / volumetric
   problems (soil, mass concrete) or as filler around brick meshes,
   not for bending-dominated structural problems.

Run::

    python examples/27_3d_solid_cantilever.py
"""
from __future__ import annotations

import numpy as np

from femsolver import ElasticIsotropic, Hex8, Model, Tet4
from femsolver.analysis.linear_static import LinearStaticAnalysis


# Standard 6-tet decomposition of a unit hex (diagonal 1-7 fan)
_TET_FAN_LOCAL = [
    (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
    (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6),
]


def build_hex_cantilever(N_x, N_y, N_z, *, L, b, h, E, nu, P):
    mat = ElasticIsotropic(1, E=E, nu=nu)
    m = Model(ndm=3, ndf=3); m.add_material(mat)
    nx, ny, nz = N_x + 1, N_y + 1, N_z + 1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                tag = k * nx * ny + j * nx + i + 1
                m.add_node(tag, i * L / N_x, j * b / N_y, k * h / N_z)

    def n(i, j, k): return k * nx * ny + j * nx + i + 1

    etag = 1
    for k in range(N_z):
        for j in range(N_y):
            for i in range(N_x):
                m.add_element(Hex8(etag, (
                    n(i, j, k), n(i+1, j, k), n(i+1, j+1, k), n(i, j+1, k),
                    n(i, j, k+1), n(i+1, j, k+1), n(i+1, j+1, k+1), n(i, j+1, k+1),
                ), mat)); etag += 1
    for k in range(nz):
        for j in range(ny):
            m.fix(n(0, j, k), [1, 1, 1])
    tip = [n(N_x, j, k) for k in range(nz) for j in range(ny)]
    for nt in tip:
        m.add_nodal_load(nt, [0, 0, -P / len(tip)])
    LinearStaticAnalysis(m).run()
    return -np.mean([m.node(nt).disp[2] for nt in tip])


def build_tet_cantilever(N_x, N_y, N_z, *, L, b, h, E, nu, P):
    mat = ElasticIsotropic(1, E=E, nu=nu)
    m = Model(ndm=3, ndf=3); m.add_material(mat)
    nx, ny, nz = N_x + 1, N_y + 1, N_z + 1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                tag = k * nx * ny + j * nx + i + 1
                m.add_node(tag, i * L / N_x, j * b / N_y, k * h / N_z)

    def n(i, j, k): return k * nx * ny + j * nx + i + 1

    etag = 1
    for k in range(N_z):
        for j in range(N_y):
            for i in range(N_x):
                # 8 corners of this hex cell (0-indexed -> 1-indexed)
                corners = [
                    n(i, j, k), n(i+1, j, k), n(i+1, j+1, k), n(i, j+1, k),
                    n(i, j, k+1), n(i+1, j, k+1), n(i+1, j+1, k+1), n(i, j+1, k+1),
                ]
                for local_tet in _TET_FAN_LOCAL:
                    tags = tuple(corners[idx] for idx in local_tet)
                    m.add_element(Tet4(etag, tags, mat)); etag += 1
    for k in range(nz):
        for j in range(ny):
            m.fix(n(0, j, k), [1, 1, 1])
    tip = [n(N_x, j, k) for k in range(nz) for j in range(ny)]
    for nt in tip:
        m.add_nodal_load(nt, [0, 0, -P / len(tip)])
    LinearStaticAnalysis(m).run()
    return -np.mean([m.node(nt).disp[2] for nt in tip])


def main() -> None:
    E, nu = 2.0e11, 0.0       # nu=0 to match Euler exactly
    L, b, h = 1.0, 0.1, 0.1
    P = 1.0
    I = b * h ** 3 / 12.0
    w_beam = P * L ** 3 / (3.0 * E * I)

    print(f"\n3-D cantilever block under transverse tip load")
    print(f"  L = {L} m, b = {b} m, h = {h} m, E = {E:g} Pa, nu = {nu}")
    print(f"  P_tip = {P} N (distributed across tip face)")
    print(f"  Beam theory: w_tip = P L^3 / (3 E I) = {w_beam:.4e} m")
    print()

    print(f"  Hex8 mesh refinement (N_y = N_z = 1):")
    print(f"  {'N_x':>4s}   {'w_tip (m)':>14s}   {'ratio':>8s}")
    for N_x in (2, 4, 8, 16, 32):
        w = build_hex_cantilever(N_x, 1, 1, L=L, b=b, h=h, E=E, nu=nu, P=P)
        print(f"  {N_x:>4d}   {w:>14.4e}   {w / w_beam:>8.4f}")

    print(f"\n  Hex8 mesh refinement (N_y = N_z = 2):")
    print(f"  {'N_x':>4s}   {'w_tip (m)':>14s}   {'ratio':>8s}")
    for N_x in (4, 8, 16, 32):
        w = build_hex_cantilever(N_x, 2, 2, L=L, b=b, h=h, E=E, nu=nu, P=P)
        print(f"  {N_x:>4d}   {w:>14.4e}   {w / w_beam:>8.4f}")

    print(f"\n  Tet4 mesh refinement (6 tets per hex cell, N_y = N_z = 1):")
    print(f"  {'N_x':>4s}   {'w_tip (m)':>14s}   {'ratio':>8s}")
    for N_x in (2, 4, 8, 16):
        w = build_tet_cantilever(N_x, 1, 1, L=L, b=b, h=h, E=E, nu=nu, P=P)
        print(f"  {N_x:>4d}   {w:>14.4e}   {w / w_beam:>8.4f}")
    print()

    print(f"  Reading the result:")
    print(f"  * Hex8 with full integration is markedly stiff in bending --")
    print(f"    even at 32x1x1 the ratio is only ~0.9 of beam theory.")
    print(f"    This is the famous 'parasitic shear' phenomenon for the")
    print(f"    trilinear hex; the industry cures are incompatible modes")
    print(f"    (Wilson) or B-bar with selective integration. These will")
    print(f"    be added in a future Phase 15.x.")
    print(f"  * Refining transverse (N_y = N_z = 2) does not help -- the")
    print(f"    locking is in the longitudinal aspect ratio of each cell.")
    print(f"  * Tet4 is far stiffer per DOF -- a 6-tet decomposition of a")
    print(f"    single hex inherits CST's poor bending behavior. Tet4 is")
    print(f"    best for bulk / volumetric problems, not bending.")
    print(f"  * For slender structural bending, the BeamColumn family")
    print(f"    (Phases 1, 5, 6, 6.5, 7, 13, 13.5) is the right tool.")
    print(f"    Hex8 / Tet4 unlock 3-D continuum effects: confined")
    print(f"    soil, mass concrete, connection details.")


if __name__ == "__main__":
    main()
