"""Force-based vs displacement-based beam-column — fiber-section pushover.

Two cantilever models, both with the same bilinear fiber section and
the same compressive-bending loading. We compare:

* Force-based: 1 element (the FB headline claim).
* Displacement-based: 1, 2, 4, 8, 16 elements.

At the load level chosen (95 % of the full plastic moment), the FB
element with **one** element gives a tip displacement that matches the
DB element with 8-16 elements. DB with 1 element is several percent
off; with 4 elements it's close; with 8+ it has converged.

This is the canonical demonstration of force-based formulations'
*one-element-per-member* property under distributed plasticity — and
the main reason ``forceBeamColumn`` is the default frame element in
OpenSees for performance-based earthquake engineering.

Run::

    python examples/18_force_based_vs_displacement_based.py
"""
from __future__ import annotations

import math
import time

from femsolver import (
    BeamColumn2DCorotational,
    ElasticIsotropic,
    FiberSection2D,
    ForceBeamColumn2DCorotational,
    Model,
    NonlinearStaticAnalysis,
    UniaxialBilinear,
)


def fiber_section_template():
    """Bilinear-kinematic-hardening rectangular fiber section."""
    mat = UniaxialBilinear(E=2.0e11, sigma_y=400.0e6, b=0.05)
    return FiberSection2D.rectangular(
        width=0.1, height=0.2, n_fibers=20, material=mat,
    )


def build_fb_cantilever(L: float):
    """FB cantilever with one element."""
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
    m.add_element(ForceBeamColumn2DCorotational(
        1, (1, 2), mat, section=fiber_section_template().clone()))
    m.fix(1, [1, 1, 1])
    return m, 2


def build_db_cantilever(L: float, n_elem: int):
    """DB cantilever with n_elem elements."""
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    for i in range(n_elem + 1):
        m.add_node(i + 1, i * L / n_elem, 0.0)
    for i in range(n_elem):
        m.add_element(BeamColumn2DCorotational(
            i + 1, (i + 1, i + 2), mat,
            section=fiber_section_template().clone()))
    m.fix(1, [1, 1, 1])
    return m, n_elem + 1


def run_pushover(m, tip_tag, P_max, *, n_steps=30):
    """Return (tip_disp, total_newton_iters, wall_time_s)."""
    m.add_nodal_load(tip_tag, [0.0, -P_max, 0.0])
    t0 = time.perf_counter()
    res = NonlinearStaticAnalysis(
        m, num_steps=n_steps, dlambda=1.0 / n_steps,
        tol=1.0e-5, max_iter=40,
    ).run()
    wall = time.perf_counter() - t0
    return m.node(tip_tag).disp[1], int(sum(res["iter_counts"])), wall


def main() -> None:
    E = 2.0e11
    sigma_y = 400.0e6
    b_section, h_section, L = 0.1, 0.2, 3.0

    My = sigma_y * b_section * h_section ** 2 / 6.0   # first-yield moment
    Mp = sigma_y * b_section * h_section ** 2 / 4.0   # plastic moment
    P_y = My / L
    P_max = 0.95 * Mp / L

    print(f"\nForce-based vs displacement-based beam — fiber-section pushover")
    print(f"  Cross section: 100 x 200 mm, 20 bilinear fibers")
    print(f"  Material:      E = {E:g} Pa, sigma_y = {sigma_y:g} Pa, b = 0.05")
    print(f"  Length:        L = {L} m")
    print(f"  Reference loads:")
    print(f"    My / L (first fiber yield) = {P_y:.1f} N")
    print(f"    Mp / L (full plastic)      = {Mp / L:.1f} N")
    print(f"  Pushover target P_max = {P_max:.1f} N ({P_max / (Mp/L):.1%} of Mp/L)")
    print()

    # FB with 1 element
    m_fb, tip_fb = build_fb_cantilever(L)
    v_fb, iters_fb, t_fb = run_pushover(m_fb, tip_fb, P_max)
    print(f"  Force-based:                                                                ")
    print(f"  {'n_elem':>7}  {'tip disp (m)':>14}  {'iters':>7}  {'wall (s)':>9}")
    print(f"  {1:7d}  {v_fb:14.6e}  {iters_fb:>7d}  {t_fb:>9.3f}")
    print()

    print(f"  Displacement-based mesh-convergence study:")
    print(f"  {'n_elem':>7}  {'tip disp (m)':>14}  {'iters':>7}  {'wall (s)':>9}  {'err vs FB':>10}")
    results_db = []
    for n in (1, 2, 4, 8, 16):
        m_db, tip_db = build_db_cantilever(L, n)
        v_db, iters_db, t_db = run_pushover(m_db, tip_db, P_max)
        err = abs(v_db - v_fb) / abs(v_fb) * 100
        results_db.append((n, v_db, iters_db, t_db, err))
        print(f"  {n:7d}  {v_db:14.6e}  {iters_db:>7d}  {t_db:>9.3f}  {err:>9.3f}%")
    print()

    # Pick the DB count whose tip disp is closest to FB
    closest_n = min(results_db, key=lambda r: r[4])
    print(f"  Closest DB result to 1-element FB: "
          f"{closest_n[0]} elements ({closest_n[4]:.3f}% diff)")
    print()
    print(f"  Reading the table:")
    print(f"  * 1 FB element matches 1 DB element to {results_db[0][4]:.1f}%.")
    print(f"  * 1 FB element matches {closest_n[0]} DB element(s) to "
          f"{closest_n[4]:.2f}%.")
    print(f"  * The FB element captures the exact linear-moment distribution")
    print(f"    from equilibrium; the DB element approximates it via the")
    print(f"    cubic-Hermite second derivative, which converges as O(h^2) for")
    print(f"    the plateau region of M-kappa.")


if __name__ == "__main__":
    main()
