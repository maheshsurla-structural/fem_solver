"""Phase B.4 -- "define a tendon, apply the prestress" (load balancing).

Demonstrates the one-call tendon workflow on a two-span continuous PT
girder:

1. Define a parabolic post-tensioned `Tendon` (profile + jacking force).
2. `tendon.apply_to(model)` lowers it to equivalent nodal loads.
3. Solve and read the **primary**, **total**, and **secondary
   (parasitic)** prestress moments -- the split every indeterminate PT
   design needs.

Also shows Lin's load balancing: the parabolic tendon's upward
equivalent load offsets the applied dead load.

Run::

    python examples/74_tendon_load_balancing.py
"""
from __future__ import annotations

import numpy as np

from femsolver.core.model import Model
from femsolver.elements.beam import BeamColumn2D
from femsolver.materials.elastic import ElasticIsotropic
from femsolver.analysis.linear_static import LinearStaticAnalysis
from femsolver.io.diagrams import beam_force_diagram
from femsolver.bridges.tendon import Tendon, tendon_secondary_moment


def main() -> None:
    Lspan, nps = 25.0, 20            # two 25 m spans
    A, I, E = 0.6, 0.18, 34e9
    P = 3.5e6                        # effective prestress (N)
    a = 0.35                         # parabolic drape per span (m)

    mat = ElasticIsotropic(1, E=E, nu=0.2, rho=0.0)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    nn = 2 * nps + 1
    for i in range(nn):
        m.add_node(i + 1, i * Lspan / nps, 0.0)
    for i in range(nn - 1):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, I))
    m.fix(1, [1, 1, 0]); m.fix(nps + 1, [0, 1, 0]); m.fix(nn, [0, 1, 0])

    # parabolic tendon: sagging (below centroid) in each span, passing
    # over the pier at the centroid.
    ecc = np.zeros(nn)
    for i in range(nn):
        x = i * Lspan / nps
        xs = x % Lspan
        ecc[i] = -4.0 * a * xs * (Lspan - xs) / Lspan ** 2

    tendon = Tendon(
        nodes=list(range(1, nn + 1)), eccentricity=ecc,
        area=0.0042, jacking_force=P, effective_force=P,
        tendon_type="post-tension", name="PT",
    )

    print("=" * 64)
    print(" Define-a-tendon, apply-the-prestress  (2-span PT girder)")
    print(f"   2 x {Lspan:.0f} m, P = {P/1e3:.0f} kN, drape a = {a:.2f} m")
    print("=" * 64)

    # ---- prestress only -----------------------------------------------
    tendon.apply_to(m)
    LinearStaticAnalysis(m).run()

    # total moment over the interior pier (end of element nps)
    M_total = beam_force_diagram(m.element(nps))["M"][-1]
    M_prim = P * ecc[nps]                       # primary = P*e at the pier
    M_sec = tendon_secondary_moment(total_moment=M_total, P=P, e=ecc[nps])
    print("\nPrestress moments over the interior pier:")
    print(f"   primary    P*e        = {M_prim/1e3:+8.1f} kN.m")
    print(f"   total (analysis)       = {M_total/1e3:+8.1f} kN.m")
    print(f"   secondary  (parasitic) = {M_sec/1e3:+8.1f} kN.m")
    print("   (secondary = total - primary, from the redundant pier)")

    # ---- load balancing -----------------------------------------------
    w_bal = 8.0 * P * a / Lspan ** 2
    print(f"\nLoad balancing: tendon upward UDL = 8Pa/L^2 = "
          f"{w_bal/1e3:.1f} kN/m")
    mid1 = nps // 2 + 1
    camber = m.node(mid1).disp[1]
    print(f"   span-1 mid camber (prestress only) = {camber*1e3:+.2f} mm (up)")

    m.clear_loads()
    tendon.apply_to(m)
    for e in m.elements.values():
        e.add_uniform_load(-w_bal)               # apply matching dead load
    LinearStaticAnalysis(m).run()
    bal = m.node(mid1).disp[1]
    print(f"   span-1 mid defl (prestress + {w_bal/1e3:.0f} kN/m DL) = "
          f"{bal*1e3:+.3f} mm  (~balanced)")


if __name__ == "__main__":
    main()
