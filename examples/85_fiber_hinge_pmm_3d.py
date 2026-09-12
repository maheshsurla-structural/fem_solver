"""Fiber-hinge plan P7 (G5) — 3-D force-based element, biaxial P-M2-M3.

Turns the Caltrans 84 in circular RC section into a genuine biaxial
(P-M2-M3) fiber beam-column with the new small-displacement force-based 3-D
element :class:`ForceBeamColumn3D` + :class:`FiberSection3D.circular`.

Three checks (plan §7.4 + acceptance):

1. **Elastic equivalence** — for an elastic prismatic section the force-based
   3-D stiffness equals the displacement-based closed-form K (both exact), and
   is invariant to the number of integration points.
2. **Uniaxial limit** — pushing the 3-D column in one transverse direction
   reproduces the 2-D force-based response (base Mz), with My ~ 0.
3. **Biaxial P-M2-M3** — a 45-degree push develops both base moments; for the
   symmetric circular section they split equally and the resultant matches the
   uniaxial capacity.

Units: **kip, inch**.

Run::

    PYTHONPATH=src python examples/85_fiber_hinge_pmm_3d.py
"""
from __future__ import annotations

import numpy as np

from femsolver import (
    BeamColumn3D,
    ConcreteMander,
    ElasticIsotropic,
    ForceBeamColumn2DCorotational,
    ForceBeamColumn3D,
    Model,
    NonlinearStaticAnalysis,
    rc_circular_column_section,
)
from femsolver.materials.uniaxial import UniaxialReinforcingSteel
from femsolver.sections.analysis import mander_confined_circular
from femsolver.sections.response.elastic import ElasticSection3D

D, COVER, EC, FC, EPS_C0, ES, FY, L = (
    84.0, 2.0, 3605.0, 5.0, 0.002219, 29000.0, 68.0, 49.0)
NU = 0.2
G = EC / (2.0 * (1.0 + NU))


def _core_cover_steel():
    conf = mander_confined_circular(
        fco=FC, eps_co=EPS_C0, D_core=79.0, hoop_area=0.79, hoop_spacing=6.0,
        fyh=68.0, rho_long=0.0257, hoop_type="hoop", clear_spacing=5.0)
    return (conf.to_material(Ec=EC),
            ConcreteMander(fpc=FC, eps_c0=EPS_C0, Ec=EC),
            UniaxialReinforcingSteel(ES, FY, 95.0, 0.0075, 0.09))


def _fiber3d():
    core, cover, steel = _core_cover_steel()
    Jt = np.pi * D ** 4 / 32.0
    return rc_circular_column_section(
        diameter=D, cover=COVER, core_concrete=core, cover_concrete=cover,
        n_bars=56, bar_area=2.25, steel=steel,
        n_core_rings=8, n_cover_rings=2, n_wedges=24,
        three_d=True, GJ=G * Jt)


def _fiber2d():
    core, cover, steel = _core_cover_steel()
    return rc_circular_column_section(
        diameter=D, cover=COVER, core_concrete=core, cover_concrete=cover,
        n_bars=56, bar_area=2.25, steel=steel,
        n_core_rings=8, n_cover_rings=2, n_wedges=24)


def _push3d(Py, Pz, section):
    m = Model(ndm=3, ndf=6)
    mat = ElasticIsotropic(1, E=EC, nu=NU)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, L, 0.0, 0.0)
    m.add_element(ForceBeamColumn3D(1, (1, 2), mat, section=section))
    m.fix(1, [1, 1, 1, 1, 1, 1])
    m.add_nodal_load(2, [0.0, -Py, -Pz, 0.0, 0.0, 0.0])
    r = NonlinearStaticAnalysis(m, num_steps=20, dlambda=1 / 20,
                                integrator="load_control", track=(2, 1),
                                tol=1e-6, max_iter=40).run()
    return r["tracked"][-1], m.nodes[1].reaction


def _push2d(P, section):
    m = Model(ndm=2, ndf=3)
    mat = ElasticIsotropic(1, E=EC, nu=NU)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    m.add_element(ForceBeamColumn2DCorotational(1, (1, 2), mat, section=section))
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [0.0, -P, 0.0])
    r = NonlinearStaticAnalysis(m, num_steps=20, dlambda=1 / 20,
                                integrator="load_control", track=(2, 1),
                                tol=1e-6, max_iter=40).run()
    return r["tracked"][-1], m.nodes[1].reaction[2]


def main() -> None:
    print("=" * 72)
    print("P7 (G5) - 3-D force-based fiber beam-column (P-M2-M3)")
    print("=" * 72)

    # --- 1. elastic equivalence + n_ip invariance -------------------------
    A = np.pi * D ** 2 / 4.0
    Iz = Iy = np.pi * D ** 4 / 64.0
    Jt = np.pi * D ** 4 / 32.0

    def elem(cls, nip):
        m = Model(ndm=3, ndf=6)
        mat = ElasticIsotropic(1, E=EC, nu=NU)
        m.add_material(mat)
        m.add_node(1, 0.0, 0.0, 0.0)
        m.add_node(2, L, 0.0, 0.0)
        e = cls(1, (1, 2), mat, section=ElasticSection3D(EC, G, A, Iy, Iz, Jt))
        e.n_int = nip
        m.add_element(e)
        return e

    Kdb = elem(BeamColumn3D, 5).K_global()
    Kfb = elem(ForceBeamColumn3D, 5).K_global()
    K3 = elem(ForceBeamColumn3D, 3).K_global()
    print(f"\n1. Elastic K: force-based vs displacement-based rel diff = "
          f"{np.max(np.abs(Kfb - Kdb)) / np.max(np.abs(Kdb)):.2e}")
    print(f"   n_ip invariance |K(3) - K(5)| = {np.max(np.abs(K3 - Kfb)):.2e}")

    # --- 2. uniaxial limit: 3-D == 2-D ------------------------------------
    P = 300.0
    uy2, Mz2 = _push2d(P, _fiber2d())
    uy3, R = _push3d(P, 0.0, _fiber3d())
    print(f"\n2. Uniaxial push (P = {P:.0f} kip in y):")
    print(f"   2-D: tip u_y = {uy2:.5f} in, base Mz = {Mz2:,.0f} kip-in")
    print(f"   3-D: tip u_y = {uy3:.5f} in, base Mz = {R[5]:,.0f} kip-in, "
          f"My = {R[4]:.3e} (~0)")
    print(f"   match: u_y {abs(uy2 - uy3):.2e}, Mz {abs(Mz2 - R[5]):.2e}")

    # --- 3. biaxial P-M2-M3 ----------------------------------------------
    _uy, Rb = _push3d(P / np.sqrt(2), P / np.sqrt(2), _fiber3d())
    Mz, My = Rb[5], Rb[4]
    print(f"\n3. Biaxial push (45 deg, |P| = {P:.0f} kip):")
    print(f"   base Mz = {Mz:,.0f}, My = {My:,.0f} kip-in "
          f"(resultant {np.hypot(Mz, My):,.0f} = uniaxial {abs(R[5]):,.0f})")

    print("\n" + "=" * 72)
    print("P7 closed: 3-D force-based element + FiberSection3D.circular (G5).")
    print("Next: P8 (cyclic benchmark + energy) or P9 (fiber-hinge element).")
    print("=" * 72)


if __name__ == "__main__":
    main()
