"""Cantilever pushover with a fiber-section column — distributed
plasticity, the natural successor to the lumped-hinge example in
`examples/08_hinged_cantilever_pushover.py`.

A 3 m steel cantilever (W-like rectangular cross-section, 100 x 200 mm)
is discretised into 20 horizontal strips, each carrying a bilinear
uniaxial material with kinematic hardening (b = 0.05). A transverse
tip load is ramped up; the analysis traces the load-deflection curve
from elastic into the post-plastic plateau.

Two reference quantities are computed from the section algebra:

* First-fiber-yield moment ``My = sigma_y * b * h^2 / 6``
  (linear stress profile, outermost fiber reaches sigma_y)
* Full plastic moment ``Mp  = sigma_y * b * h^2 / 4``
  (uniform stress profile of magnitude sigma_y, shape factor 1.5)

The corresponding tip forces are ``My/L`` and ``Mp/L``. The fiber
section transitions smoothly between the two as plasticity spreads
through the depth of the cross-section (unlike a lumped plastic hinge,
which has a single abrupt yield knee).

Run::

    python examples/09_fiber_cantilever_pushover.py
"""
from __future__ import annotations

from femsolver import (
    BeamColumn2D,
    ElasticIsotropic,
    FiberSection2D,
    Model,
    NonlinearStaticAnalysis,
    UniaxialBilinear,
)


def main() -> None:
    # ----------------------------------------------------------- inputs
    E = 2.0e11           # Young's modulus (Pa)
    sigma_y = 400.0e6    # yield stress (Pa)
    b_post = 0.05        # kinematic-hardening slope ratio
    b_section = 0.1      # cross-section width  (m)
    h_section = 0.2      # cross-section height (m)
    L = 3.0              # cantilever length    (m)
    n_fibers = 20        # discretisation through the depth

    # Reference quantities
    My = sigma_y * b_section * h_section ** 2 / 6.0
    Mp = sigma_y * b_section * h_section ** 2 / 4.0
    P_yield = My / L
    P_plastic = Mp / L

    # Push to 1.4 * Mp / L so we cross the plateau and ride the
    # kinematic-hardening branch beyond it.
    P_target = 1.4 * P_plastic
    n_steps = 60

    # ----------------------------------------------------------- build
    mat_iso = ElasticIsotropic(1, E=E, nu=0.3)
    uniaxial = UniaxialBilinear(E=E, sigma_y=sigma_y, b=b_post)
    section = FiberSection2D.rectangular(
        width=b_section, height=h_section,
        n_fibers=n_fibers, material=uniaxial,
    )

    model = Model(ndm=2, ndf=3)
    model.add_material(mat_iso)
    model.add_node(1, 0.0, 0.0)
    model.add_node(2, L, 0.0)
    elem = BeamColumn2D(1, (1, 2), mat_iso, section=section)
    model.add_element(elem)
    model.fix(1, [1, 1, 1])
    model.add_nodal_load(2, [0.0, -P_target, 0.0])

    # ----------------------------------------------------------- solve
    analysis = NonlinearStaticAnalysis(
        model, num_steps=n_steps, dlambda=1.0 / n_steps,
        track=(2, 1), tol=1e-6, max_iter=40,
    )
    analysis.run()

    # ----------------------------------------------------------- print
    print("\nFiber-section cantilever pushover")
    print(f"  Cross section : {b_section} x {h_section} m, {n_fibers} fibers")
    print(f"  Material      : E={E:g} Pa, sigma_y={sigma_y:g} Pa, b={b_post}")
    print(f"  Length        : {L} m")
    print()
    print(f"  Reference moments:")
    print(f"    My (first fiber yield) = {My:.4g} N.m  ->  P_y = {P_yield:.4g} N")
    print(f"    Mp (full plastic)      = {Mp:.4g} N.m  ->  P_p = {P_plastic:.4g} N")
    print(f"    Shape factor Mp / My   = {Mp / My:.4f}  (analytical: 1.5)")
    print()
    print(f"  Load history (every 5 steps):")
    print(f"  {'step':>5}  {'lambda':>7}  {'P (N)':>11}  {'v_tip (m)':>14}")
    for i, (lmbd, v_tip) in enumerate(zip(analysis.lambdas, analysis.tracked), 1):
        if i % 5 == 0 or i == 1:
            print(f"  {i:5d}  {lmbd:7.4f}  {lmbd * P_target:11.4g}  {v_tip:14.6e}")
    print()

    # ----------------------------------------------------------- inspect
    # The fixed-end integration point has the largest moment, so it's
    # where plasticity is deepest. Report the plastic-strain profile
    # through the depth at that section.
    fixed_end_sec = elem.sections[0]
    fibers_sorted = sorted(fixed_end_sec.fibers, key=lambda f: f.y)
    print(f"  Plastic-strain profile at the fixed-end IP:")
    print(f"  {'y (m)':>9}  {'eps_p':>13}  {'sigma (Pa)':>13}")
    for f in fibers_sorted:
        print(
            f"  {f.y:9.4f}  {f.material.eps_p_committed:13.5e}  "
            f"{f.material.sigma_trial:13.4g}"
        )

    # Count yielded fibers across the most-stressed IP
    yielded = sum(
        1 for f in fixed_end_sec.fibers
        if f.material.eps_p_committed != 0.0
    )
    print()
    print(f"  Fibers yielded at fixed-end IP: {yielded} / {n_fibers}")
    if yielded == n_fibers:
        print("  Cross section is fully plastic at the fixed end.")
    elif yielded > 0:
        print("  Plasticity has spread part-way through the depth — "
              "the unyielded core in the middle carries elastic stress.")
    else:
        print("  Section is still elastic.")


if __name__ == "__main__":
    main()
