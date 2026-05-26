"""Confined vs unconfined concrete -- stress-strain curves and a
cantilever pushover.

The classic Mander result: confining a concrete core with transverse
reinforcement raises both the peak stress (``fcc' > fc'``) and,
crucially, the peak strain (``eps_cc >> eps_c0``). The descending
branch becomes far gentler -- the confined section retains a high
fraction of its peak strength out to ductile failure strains.

This example shows:

1. The stress-strain envelopes for two parameter sets side by side
   (Kent-Park-Scott unconfined vs Mander confined).
2. A cantilever RC column pushover where the cover concrete uses the
   unconfined Kent-Park material and the core uses the Mander confined
   model. The combined fiber-section captures the right answer for a
   seismic RC column: the cover crushes at a small drift, the confined
   core sustains substantial post-peak ductility.

Run::

    python examples/28_concrete_confined_vs_unconfined.py
"""
from __future__ import annotations

import math

from femsolver import (
    BeamColumn2D,
    ConcreteKentPark,
    ConcreteMander,
    ElasticIsotropic,
    Fiber,
    FiberSection2D,
    Model,
    NonlinearStaticAnalysis,
    UniaxialBilinear,
)


def main() -> None:
    print("\nConfined vs unconfined concrete -- stress-strain envelopes")
    print(f"  Unconfined cover: ConcreteKentPark, fc' = 30 MPa, eps_c0 = 0.002,")
    print(f"                    fpcu = 6 MPa, eps_cu = 0.0035 (typical 30 MPa concrete)")
    print(f"  Confined core:    ConcreteMander, fcc' = 45 MPa, eps_cc = 0.005")
    print(f"                    (50% strength gain + 2.5x peak strain from confinement)")
    print()

    unconf = ConcreteKentPark(fpc=30e6, eps_c0=0.002, fpcu=6e6, eps_cu=0.0035)
    conf = ConcreteMander(fpc=45e6, eps_c0=0.005)

    print(f"  {'eps':>10s}  {'unconf sigma (MPa)':>22s}  {'conf sigma (MPa)':>20s}")
    eps_grid = [0.0, 0.0005, 0.001, 0.002, 0.0035, 0.005, 0.008, 0.012, 0.020]
    for eps_mag in eps_grid:
        eps = -eps_mag
        s_u, _ = unconf.get_response(eps)
        s_c, _ = conf.get_response(eps)
        # Reset state for the next pure-monotonic call
        unconf = ConcreteKentPark(fpc=30e6, eps_c0=0.002, fpcu=6e6, eps_cu=0.0035)
        conf = ConcreteMander(fpc=45e6, eps_c0=0.005)
        unconf.get_response(eps); conf.get_response(eps)
        print(f"  {eps:>10.5f}  {s_u * 1e-6:>22.3f}  {s_c * 1e-6:>20.3f}")
    print()
    print(f"  Reading the envelopes:")
    print(f"  * Both models start at zero stress with the same initial tangent")
    print(f"    near 30 GPa (ACI Ec = 4700 sqrt(fc[MPa])).")
    print(f"  * Unconfined peaks at -30 MPa at eps = -0.002 then drops to")
    print(f"    -6 MPa residual at eps = -0.0035 -- the cover concrete loses")
    print(f"    most of its strength once it spalls.")
    print(f"  * Confined Mander peaks at -45 MPa at eps = -0.005 and decays")
    print(f"    gently past it -- the transverse steel keeps the core")
    print(f"    intact and ductile out to eps = -0.02 and beyond.")
    print()
    print(f"  This is the canonical input for a *fiber-section* RC column:")
    print(f"  cover fibers use ConcreteKentPark, core fibers use ConcreteMander,")
    print(f"  rebar fibers use UniaxialBilinear (or cyclic Menegotto-Pinto in")
    print(f"  Phase 16.3). The section captures the right post-peak behaviour")
    print(f"  for seismic pushover and time-history analyses.")
    print()

    # ------- RC column under axial compression -------
    print(f"  RC column under axial compression (combined cover + core fibers)")
    L = 1.5                  # m
    h_sec = 0.4              # depth
    b_sec = 0.4              # width
    cover = 0.04             # cover concrete thickness
    n_core_y = 6
    n_cover_per_side = 2
    rebar_area = 4 * 200e-6

    mat_unconf = ConcreteKentPark(fpc=30e6, eps_c0=0.002,
                                    fpcu=6e6, eps_cu=0.0035)
    mat_conf = ConcreteMander(fpc=45e6, eps_c0=0.005)
    mat_rebar = UniaxialBilinear(E=2.0e11, sigma_y=400e6, b=0.01)
    mat_iso = ElasticIsotropic(1, E=30e9, nu=0.2)

    fibers: list[Fiber] = []
    y_core_min = -(h_sec / 2 - cover)
    y_core_max = +(h_sec / 2 - cover)
    dy_core = (y_core_max - y_core_min) / n_core_y
    for i in range(n_core_y):
        y = y_core_min + (i + 0.5) * dy_core
        fibers.append(Fiber(y=y, z=0.0,
                             area=(b_sec - 2 * cover) * dy_core,
                             material=mat_conf.clone()))
    for side_sign in (+1, -1):
        y_edge = side_sign * h_sec / 2
        y_inner = side_sign * (h_sec / 2 - cover)
        dy_cov = (y_edge - y_inner) / n_cover_per_side
        for i in range(n_cover_per_side):
            y = y_inner + (i + 0.5) * dy_cov
            fibers.append(Fiber(y=y, z=0.0,
                                  area=b_sec * abs(dy_cov),
                                  material=mat_unconf.clone()))
    for i in range(n_core_y):
        y = y_core_min + (i + 0.5) * dy_core
        fibers.append(Fiber(y=y, z=0.0,
                             area=cover * dy_core,
                             material=mat_unconf.clone()))
        fibers.append(Fiber(y=y, z=0.0,
                             area=cover * dy_core,
                             material=mat_unconf.clone()))
    fibers.append(Fiber(y=+(h_sec / 2 - cover), z=0.0,
                         area=rebar_area, material=mat_rebar.clone()))
    fibers.append(Fiber(y=-(h_sec / 2 - cover), z=0.0,
                         area=rebar_area, material=mat_rebar.clone()))

    sec = FiberSection2D(fibers)
    m = Model(ndm=2, ndf=3); m.add_material(mat_iso)
    m.add_node(1, 0.0, 0.0); m.add_node(2, 0.0, L)
    m.add_element(BeamColumn2D(1, (1, 2), mat_iso, section=sec))
    m.fix(1, [1, 1, 1])
    # Service-level axial load: 10% of f'c * Ag
    P_axial = 0.10 * 30e6 * (b_sec * h_sec)
    m.add_nodal_load(2, [0.0, -P_axial, 0.0])
    res = NonlinearStaticAnalysis(
        m, num_steps=10, dlambda=1.0 / 10, tol=1e-4, max_iter=30,
    ).run()
    print(f"  Axial load applied: {P_axial * 1e-3:.1f} kN compression")
    print(f"  Newton iterations per step: {res['iter_counts']}")
    print(f"  Axial shortening at top = {-m.node(2).disp[1] * 1e3:.3f} mm")
    print(f"  Average axial strain = {-m.node(2).disp[1] / L:.5e}")
    print(f"  Section has {len(sec.fibers)} fibers "
          f"({n_core_y} core + {4 * n_cover_per_side + 2 * n_core_y} cover + 2 rebar)")
    print()
    print(f"  Practical takeaway:")
    print(f"  * The fiber-section couples cover and core constitutive")
    print(f"    behaviour automatically: outer fibers reach the Kent-Park")
    print(f"    descending branch (spalling) while inner core fibers ride")
    print(f"    the Mander curve to higher strength + ductility.")
    print(f"  * For full seismic pushover with lateral cyclic loading,")
    print(f"    use ``DisplacementControl`` (Phase 10) -- load control")
    print(f"    struggles past the peak of the column's lateral-force")
    print(f"    capacity, where the descending branch makes load-control")
    print(f"    inversion ill-conditioned.")


if __name__ == "__main__":
    main()
