"""Fiber-hinge plan P6 — staged analysis, protocols, monotonic pushover.

Stage-2 of the benchmark workflow (plan §2.6/§7.2): hold the P = 2400 kip
axial preload, then push the column tip laterally and trace the base-shear vs
tip-displacement pushover curve.

Shows the three P6 pieces working together:

* **G6 staged analysis** (:class:`StagedAnalysis`) — stage 1 applies the axial
  load under load control; stage 2 *continues from that committed state* and
  *holds the axial load constant* while pushing laterally under displacement
  control. (The axial reaction stays at 2400 kip throughout.)
* **G7 protocols** (:func:`monotonic`) — the lateral target history.
* the fibre section + force-based element from P1-P5.

Base shear * L is reported against the section moment capacity (P4 M-phi) as a
consistency check. Golden Midas/CSI pushover numbers: paste into plan §7.2.

Units: **kip, inch**.

Run::

    PYTHONPATH=src python examples/84_fiber_hinge_pushover.py
"""
from __future__ import annotations

import numpy as np

from femsolver import (
    ConcreteMander,
    ElasticIsotropic,
    ForceBeamColumn2DCorotational,
    Model,
    NonlinearStaticAnalysis,
    StagedAnalysis,
    monotonic,
    rc_circular_column_section,
)
from femsolver.analysis.static_integrator import DisplacementControl
from femsolver.materials.uniaxial import UniaxialReinforcingSteel
from femsolver.sections.analysis import mander_confined_circular

D, COVER, EC, FC, EPS_C0, ES, FY, L = (
    84.0, 2.0, 3605.0, 5.0, 0.002219, 29000.0, 68.0, 49.0)
P_PRELOAD = 2400.0
TIP_TARGET = 0.15                # in lateral push


def main() -> None:
    print("=" * 72)
    print("P6 - Caltrans column staged pushover (hold P = 2400 kip, push tip)")
    print("=" * 72)

    conf = mander_confined_circular(
        fco=FC, eps_co=EPS_C0, D_core=79.0, hoop_area=0.79, hoop_spacing=6.0,
        fyh=68.0, rho_long=0.0257, hoop_type="hoop", clear_spacing=5.0)
    sec = rc_circular_column_section(
        diameter=D, cover=COVER, core_concrete=conf.to_material(Ec=EC),
        cover_concrete=ConcreteMander(fpc=FC, eps_c0=EPS_C0, Ec=EC),
        n_bars=56, bar_area=2.25,
        steel=UniaxialReinforcingSteel(ES, FY, 95.0, 0.0075, 0.09),
        n_core_rings=8, n_cover_rings=2, n_wedges=24)
    mat = ElasticIsotropic(1, E=EC, nu=0.2)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    m.add_element(ForceBeamColumn2DCorotational(1, (1, 2), mat, section=sec))
    m.fix(1, [1, 1, 1])

    # G7: monotonic lateral target -> a fixed displacement-control step
    n_push = 25
    targets = monotonic(-TIP_TARGET, n_push)     # 0 -> -0.15 in
    du = targets[1] - targets[0]

    sa = StagedAnalysis(m)
    sa.add_stage("axial", lambda mm: (
        mm.add_nodal_load(2, [-P_PRELOAD, 0.0, 0.0]),
        NonlinearStaticAnalysis(mm, num_steps=8, dlambda=0.125,
                                integrator="load_control", tol=1e-8))[1])
    sa.add_stage("push", lambda mm: (
        mm.add_nodal_load(2, [0.0, -1.0, 0.0]),
        NonlinearStaticAnalysis(
            mm, num_steps=n_push, integrator=DisplacementControl(2, 1, du),
            track=(2, 1), tol=1e-6, max_iter=60))[1])
    out = sa.run()

    u = np.abs(np.array(out["push"]["tracked"]))
    V = np.abs(np.array(out["push"]["lambdas"]))       # ref lateral load = -1
    print(f"\nStage 1 (axial): base reaction Fx = "
          f"{abs(m.nodes[1].reaction[0]):.1f} kip (held at {P_PRELOAD:.0f})")
    print(f"Stage 2 (push):  {len(u)} steps to tip = {u[-1]:.3f} in\n")
    print(f"{'tip (in)':>10}{'base shear (kip)':>18}{'M=V*L (kip-ft)':>18}")
    print("-" * 46)
    for d in (0.02, 0.04, 0.06, 0.09, 0.12, 0.15):
        Vi = float(np.interp(d, u, V))
        print(f"{d:>10.2f}{Vi:>18.0f}{Vi * L / 12.0:>18.0f}")

    print(f"\nAxial held constant during push: base Fx = "
          f"{abs(m.nodes[1].reaction[0]):.1f} kip")
    print(f"Peak base shear = {V.max():.0f} kip -> "
          f"M = {V.max() * L / 12.0:.0f} kip-ft "
          f"(cf. section M-phi peak ~32,900 kip-ft, P4)")

    print("\n" + "=" * 72)
    print("P6 closed: staged analysis (G6) + protocols (G7) + pushover.")
    print("Golden Midas/CSI pushover: paste into plan 7.2 when exported.")
    print("Next: P7 (3-D force-based + circular 3-D; P-M2-M3).")
    print("=" * 72)


if __name__ == "__main__":
    main()
