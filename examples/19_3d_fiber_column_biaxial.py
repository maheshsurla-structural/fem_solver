"""3D cantilever column with a fiber cross-section under biaxial loading.

A vertical 3 m steel column with a 200 x 100 mm rectangular fiber
section (40 x 20 = 800 bilinear fibers through the depth) is loaded at
its tip with:

* axial compression ``P_axial`` (along the local x = beam axis)
* lateral force ``P_y`` (along the local y axis — causes ``Mz``)
* lateral force ``P_z`` (along the local z axis — causes ``My``)

The combined load drives the cross section to develop biaxial bending
(both ``Mz`` and ``My``) plus axial. As the loading progresses, fibers
on the corners yield first (the corners see the largest combined
strain ``eps_axial - y kappa_z + z kappa_y``). The result is a
genuine **P-Mz-My interaction**: yielding in one direction reduces
the elastic stiffness available to carry the others, and the section
tangent picks up cross-coupling terms.

This is the 3D analog of the Phase 5 fiber-column pushover (example
09). With ``FiberSection3D`` + ``BeamColumn3D``, the solver now
supports the full distributed-plasticity story in 3D, which is what
performance-based seismic design needs for biaxial column response.

Run::

    python examples/19_3d_fiber_column_biaxial.py
"""
from __future__ import annotations

import math

from femsolver import (
    BeamColumn3D,
    ElasticIsotropic,
    FiberSection3D,
    Model,
    NonlinearStaticAnalysis,
    UniaxialBilinear,
)


def main() -> None:
    # ----- inputs ----------------------------------------------------
    E = 2.0e11
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    sy = 400.0e6
    b_post = 0.05

    width_y = 0.20         # depth in y (the "tall" dimension)
    width_z = 0.10         # breadth in z
    L = 3.0
    n_y, n_z = 20, 10
    # St. Venant torsional constant for a rectangle (approx).
    # For a 2:1 rectangle, J ≈ 0.229 * width_y * width_z^3
    J_StVenant = 0.229 * width_y * width_z ** 3
    GJ = G * J_StVenant

    mat_iso = ElasticIsotropic(1, E=E, nu=nu)
    mat_u = UniaxialBilinear(E=E, sigma_y=sy, b=b_post)
    sec = FiberSection3D.rectangular(
        width_y=width_y, width_z=width_z,
        n_y=n_y, n_z=n_z,
        material=mat_u, GJ=GJ,
    )

    # ----- reference loads ------------------------------------------
    # First-yield moment about each axis (assuming outermost fiber at
    # the geometric extreme; actual yield slightly later due to fiber
    # discretisation).
    Mz_y = sy * width_z * width_y ** 2 / 6.0    # bending about z
    My_y = sy * width_y * width_z ** 2 / 6.0    # bending about y
    Py_yield = Mz_y / L
    Pz_yield = My_y / L
    A_section = width_y * width_z
    P_axial_yield = sy * A_section              # plain-axial yield force

    # Loading: 25 % of axial yield + 60 % of each lateral yield force.
    # This combination drives biaxial bending past first yield in the
    # corner fibers but stays well clear of full plastification.
    P_axial = 0.25 * P_axial_yield
    P_y = 0.60 * Py_yield
    P_z = 0.60 * Pz_yield

    # ----- model ----------------------------------------------------
    m = Model(ndm=3, ndf=6)
    m.add_material(mat_iso)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, L, 0.0, 0.0)
    elem = BeamColumn3D(1, (1, 2), mat_iso, section=sec)
    m.add_element(elem)
    m.fix(1, [1, 1, 1, 1, 1, 1])
    m.add_nodal_load(2, [-P_axial, -P_y, -P_z, 0.0, 0.0, 0.0])

    # ----- solve ----------------------------------------------------
    n_steps = 30
    res = NonlinearStaticAnalysis(
        m, num_steps=n_steps, dlambda=1.0 / n_steps,
        tol=1.0e-5, max_iter=40,
        track=(2, 1),    # track tip y-displacement
    ).run()

    # ----- report ---------------------------------------------------
    print(f"\n3D cantilever fiber-column under biaxial loading")
    print(f"  Cross section: {width_y*1000:.0f} x {width_z*1000:.0f} mm, "
          f"{n_y} x {n_z} bilinear fibers")
    print(f"  E = {E:g} Pa, sigma_y = {sy:g} Pa, b = {b_post}")
    print(f"  L = {L} m")
    print(f"  GJ (St. Venant) = {GJ:.4g} N.m^2\n")

    print(f"  Reference (per-axis-only) loads:")
    print(f"    First yield force in y (Mz_y / L) = {Py_yield:.1f} N")
    print(f"    First yield force in z (My_y / L) = {Pz_yield:.1f} N")
    print(f"    Axial-yield force                 = {P_axial_yield:.1f} N")
    print(f"  Applied:")
    print(f"    P_axial = {P_axial:.1f} N "
          f"({P_axial / P_axial_yield:.1%} of axial yield)")
    print(f"    P_y     = {P_y:.1f} N "
          f"({P_y / Py_yield:.1%} of Py_yield)")
    print(f"    P_z     = {P_z:.1f} N "
          f"({P_z / Pz_yield:.1%} of Pz_yield)\n")

    print(f"  Tip state at the end of pushover (lambda = 1.0):")
    print(f"    u_x (axial) = {m.node(2).disp[0]:+.4e} m")
    print(f"    u_y         = {m.node(2).disp[1]:+.4e} m")
    print(f"    u_z         = {m.node(2).disp[2]:+.4e} m\n")

    print(f"  Newton convergence:")
    print(f"    Total iterations across {n_steps} steps: "
          f"{sum(res['iter_counts'])}")
    print(f"    Max iters in a single step: {max(res['iter_counts'])}\n")

    # Inspect fiber yield pattern at the most-stressed IP (fixed end).
    fixed_sec = elem.sections[0]
    yielded = [f for f in fixed_sec.fibers
               if f.material.eps_p_committed != 0.0]
    yielded_y_pos = sum(1 for f in yielded if f.y > 0.0)
    yielded_y_neg = sum(1 for f in yielded if f.y < 0.0)
    yielded_z_pos = sum(1 for f in yielded if f.z > 0.0)
    yielded_z_neg = sum(1 for f in yielded if f.z < 0.0)

    print(f"  Yielded fibers at the fixed-end IP "
          f"({len(yielded)} / {n_y * n_z}):")
    print(f"    +y side: {yielded_y_pos:3d}   -y side: {yielded_y_neg:3d}")
    print(f"    +z side: {yielded_z_pos:3d}   -z side: {yielded_z_neg:3d}\n")

    # The asymmetry of yielding (more on one side than the other)
    # reflects the biaxial-bending pattern — the corner fiber at
    # (+y_max, +z_max) feels the largest combined compressive strain
    # from the downward+lateral load combination.
    # "Corner" fiber = outermost in BOTH y and z directions (one per quadrant).
    fy_max = max(f.y for f in fixed_sec.fibers)
    fz_max = max(f.z for f in fixed_sec.fibers)
    corner_strains = [
        (f.y, f.z, f.material.eps_p_committed)
        for f in fixed_sec.fibers
        if abs(f.y) >= 0.99 * fy_max and abs(f.z) >= 0.99 * fz_max
    ]
    print(f"  Corner-fiber plastic strain (4 corners):")
    for y, z, ep in sorted(corner_strains, key=lambda t: (t[0], t[1])):
        flag = "*" if ep != 0.0 else " "
        print(f"    (y = {y:+.4f}, z = {z:+.4f})  "
              f"eps_p = {ep:+.4e}  {flag}")
    print()
    print(f"  The asymmetric yield pattern is the P-Mz-My interaction")
    print(f"  in action: corner fibers where axial compression + both")
    print(f"  bending compressions stack up yield first, while opposite")
    print(f"  corners (where they partially cancel) remain elastic.")


if __name__ == "__main__":
    main()
