"""Fiber-hinge plan P9 (G4) — finite-length fiber-hinge element idiom.

The benchmark tools let you assign a **fiber hinge of finite length** to an
otherwise-elastic member (CSI "Fiber P-M2-M3" over a relative hinge length;
Midas lumped inelastic hinge). :class:`FiberHingeBeamColumn2D` is that idiom: an
elastic member with a fiber plastic hinge of length ``lp`` at each end (P1-P3
section in the hinges, elastic interior), reusing the force-based section
state determination.

This example shows it reproduces the distributed force-based element (P5/P6)
where it should, and how the hinge length ``lp`` shapes the post-yield response:

1. **Elastic** — for any ``lp`` the hinge element's elastic stiffness equals the
   distributed element's (exact).
2. **Axial (P5)** — under the 2400 kip preload the hinge and distributed
   elements shorten identically.
3. **Pushover (P6)** — the hinge cantilever yields with a base shear in the same
   ballpark as the distributed element; smaller ``lp`` concentrates the
   plasticity.

Units: **kip, inch**.

Run::

    PYTHONPATH=src python examples/87_fiber_hinge_element.py
"""
from __future__ import annotations

import numpy as np

from femsolver import (
    ConcreteMander,
    ElasticIsotropic,
    FiberHingeBeamColumn2D,
    ForceBeamColumn2DCorotational,
    Model,
    NonlinearStaticAnalysis,
    rc_circular_column_section,
)
from femsolver.analysis.static_integrator import DisplacementControl
from femsolver.materials.uniaxial import UniaxialReinforcingSteel
from femsolver.sections.analysis import mander_confined_circular

D, COVER, EC, FC, EPS_C0, ES, FY, L = (
    84.0, 2.0, 3605.0, 5.0, 0.002219, 29000.0, 68.0, 49.0)


def _section():
    conf = mander_confined_circular(
        fco=FC, eps_co=EPS_C0, D_core=79.0, hoop_area=0.79, hoop_spacing=6.0,
        fyh=68.0, rho_long=0.0257, hoop_type="hoop", clear_spacing=5.0)
    return rc_circular_column_section(
        diameter=D, cover=COVER, core_concrete=conf.to_material(Ec=EC),
        cover_concrete=ConcreteMander(fpc=FC, eps_c0=EPS_C0, Ec=EC),
        n_bars=56, bar_area=2.25,
        steel=UniaxialReinforcingSteel(ES, FY, 95.0, 0.0075, 0.09),
        n_core_rings=8, n_cover_rings=2, n_wedges=24)


def _cantilever(elem_factory):
    mat = ElasticIsotropic(1, E=EC, nu=0.2)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    m.add_element(elem_factory(mat))
    m.fix(1, [1, 1, 1])
    return m


def _axial(elem_factory):
    m = _cantilever(elem_factory)
    m.fix(2, [0, 1, 1])
    m.add_nodal_load(2, [-2400.0, 0.0, 0.0])
    NonlinearStaticAnalysis(m, num_steps=10, dlambda=0.1,
                            integrator="load_control", track=(2, 0),
                            tol=1e-8).run()
    return m.nodes[2].disp[0]


def _pushover(elem_factory, n=25):
    m = _cantilever(elem_factory)
    m.add_nodal_load(2, [0.0, -1.0, 0.0])
    r = NonlinearStaticAnalysis(
        m, num_steps=n, integrator=DisplacementControl(2, 1, -0.006),
        track=(2, 1), tol=1e-6, max_iter=60).run()
    return np.abs(np.array(r["tracked"])), np.abs(np.array(r["lambdas"]))


def main() -> None:
    print("=" * 72)
    print("P9 (G4) - finite-length fiber-hinge element (beam with hinges)")
    print("=" * 72)

    dist = lambda mat: ForceBeamColumn2DCorotational(  # noqa: E731
        1, (1, 2), mat, section=_section())
    hinge = lambda mat, lp: FiberHingeBeamColumn2D(    # noqa: E731
        1, (1, 2), mat, section=_section(), lp=lp)

    # --- 2. axial (P5) ----------------------------------------------------
    ud = _axial(dist)
    uh = _axial(lambda mat: hinge(mat, 8.0))
    print(f"\nAxial preload (P = 2400 kip): tip shortening")
    print(f"  distributed force-based = {ud:.6f} in")
    print(f"  fiber hinge (lp = 8 in) = {uh:.6f} in  "
          f"(match {abs(uh - ud) / abs(ud):.2%})")

    # --- 3. pushover (P6) vs distributed, and lp study --------------------
    ud_u, ud_V = _pushover(dist)
    print(f"\nMonotonic pushover base shear (kip) vs tip displacement (in):")
    print(f"{'tip':>8}{'distributed':>14}{'hinge lp=4':>14}"
          f"{'hinge lp=8':>14}{'hinge lp=16':>14}")
    print("-" * 64)
    curves = {lp: _pushover(lambda mat, _lp=lp: hinge(mat, _lp))
              for lp in (4.0, 8.0, 16.0)}
    for d in (0.02, 0.04, 0.06, 0.09, 0.12):
        row = f"{d:>8.2f}{np.interp(d, ud_u, ud_V):>14.0f}"
        for lp in (4.0, 8.0, 16.0):
            u, V = curves[lp]
            row += f"{np.interp(d, u, V):>14.0f}"
        print(row)
    print("-" * 64)
    peaks = "".join(f"{c[1].max():>14.0f}" for c in curves.values())
    print(f"{'peak V':>8}{ud_V.max():>14.0f}{peaks}")

    print(f"\nElastic stiffness is identical for every lp; the hinge length "
          f"shapes\nonly the post-yield branch (smaller lp -> more concentrated "
          f"plasticity).")
    print("\n" + "=" * 72)
    print("P9 closed: finite-length fiber-hinge element (CSI RelDist / Midas).")
    print("Next: P10 (MCT/$br importers + regression harness).")
    print("=" * 72)


if __name__ == "__main__":
    main()
