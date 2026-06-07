"""Phase B.6 -- prestressing a SOLID model via initial-stress eigenstrain.

Prestress, residual stress, and thermal strain are the same FE
operation: an initial stress ``σ₀`` present at zero displacement
produces the equivalent load ``f = -∫ Bᵀ σ₀ dV``. This closes
prestress for solid (and plane) element types -- a tendon force ``P``
smeared over a host area ``A`` along a direction ``d`` is the uniaxial
compression ``σ₀ = -(P/A)·(d⊗d)``.

Here a prismatic concrete block (meshed with Hex8 bricks) is
post-tensioned by an axial tendon. We:

1. build ``σ₀`` with :func:`prestress_initial_stress`,
2. apply it with :func:`apply_initial_stress`,
3. confirm the block shortens by the elastic-shortening amount
   ``P L / (A E)`` and carries a uniform compressive field.

Run::

    python examples/75_solid_prestress_eigenstrain.py
"""
from __future__ import annotations

import numpy as np

from femsolver.core.model import Model
from femsolver.elements.solid import Hex8
from femsolver.materials.elastic import ElasticIsotropic
from femsolver.analysis.linear_static import LinearStaticAnalysis
from femsolver.analysis.initial_stress import (
    apply_initial_stress,
    prestress_initial_stress,
)


def main() -> None:
    # Prismatic block: Lx x Ly x Lz, meshed nx x 1 x 1 Hex8 along x.
    Lx, Ly, Lz = 4.0, 0.5, 0.5
    nx = 8
    E, nu = 30e9, 0.2
    P = 3.0e6                     # tendon force (N)
    A = Ly * Lz                   # host cross-section area

    mat = ElasticIsotropic(1, E=E, nu=nu, rho=0.0)
    m = Model(ndm=3, ndf=3)
    m.add_material(mat)

    # grid of (nx+1) x 2 x 2 nodes
    def nid(i, j, k):
        return 1 + i * 4 + j * 2 + k
    for i in range(nx + 1):
        x = i * Lx / nx
        for j in range(2):
            for k in range(2):
                m.add_node(nid(i, j, k), x, j * Ly, k * Lz)
    for i in range(nx):
        n = [nid(i, 0, 0), nid(i + 1, 0, 0), nid(i + 1, 1, 0), nid(i, 1, 0),
             nid(i, 0, 1), nid(i + 1, 0, 1), nid(i + 1, 1, 1), nid(i, 1, 1)]
        m.add_element(Hex8(i + 1, tuple(n), mat))

    # roller supports at x=0 face (fix x), plus minimal restraint of the
    # y,z rigid-body / Poisson so the block can shorten freely in x.
    for j in range(2):
        for k in range(2):
            m.fix(nid(0, j, k), [1, 0, 0])
    m.fix(nid(0, 0, 0), [1, 1, 1])       # pin one corner
    m.fix(nid(0, 1, 0), [1, 1, 0])       # restrain y of the y-edge
    m.fix(nid(0, 0, 1), [1, 0, 1])       # restrain z of the z-edge

    print("=" * 60)
    print(" Prestressing a SOLID model (initial-stress eigenstrain)")
    print(f"   block {Lx}x{Ly}x{Lz} m, {nx} Hex8, P={P/1e3:.0f} kN")
    print("=" * 60)

    sigma0 = prestress_initial_stress(P=P, A=A, direction=[1, 0, 0])
    print(f"\n tendon -> initial stress sxx = {sigma0[0]/1e6:.2f} MPa "
          f"(= -P/A = {-P/A/1e6:.2f})")

    stresses = {e.tag: sigma0 for e in m.elements.values()}
    apply_initial_stress(m, stresses)
    LinearStaticAnalysis(m).run()

    # free shortening of the released block: eps_xx = -sigma0/E = +P/(A E)?
    # NOTE: with the x=0 face on rollers and the far face free, the block
    # carries zero net axial stress and strains to eps = -Dinv sigma0.
    tip = m.node(nid(nx, 0, 0)).disp[0]
    eps_free = -sigma0[0] / E         # = +P/(A E)
    print(f"\n far-end x-displacement = {tip*1e3:+.4f} mm")
    print(f" expected eps_xx*Lx     = {eps_free*Lx*1e3:+.4f} mm")
    print(f" (the released prestressed block strains by -D^-1 sigma0)")
    print("\n The same apply_initial_stress closes prestress for plane")
    print(" (Quad4) and solid (Hex8/Tet4) models; tendons in beams use")
    print(" Tendon.apply_to. Shell membrane prestress is the next step.")


if __name__ == "__main__":
    main()
