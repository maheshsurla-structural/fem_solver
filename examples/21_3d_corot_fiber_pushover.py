"""Combined material + geometric nonlinearity in 3-D.

A 3-D steel cantilever with a rectangular fiber section is loaded
with axial compression + lateral pushover. We compare:

* **BeamColumn3D + FiberSection3D** (Phase 5.5) — distributed plasticity
  but no P-Delta.
* **BeamColumn3DCorotational + FiberSection3D** (Phase 13.5) — both
  distributed plasticity *and* P-Delta.

The corotational variant predicts larger tip deflection (P-Delta
amplifies the lateral motion) AND more fibers past first yield (the
M-P interaction means axial compression shifts the yield boundary).
This is the same canonical pair of effects as the 2-D Phase 6.5
example, now demonstrated in 3-D.

Run::

    python examples/21_3d_corot_fiber_pushover.py
"""
from __future__ import annotations

import math

from femsolver import (
    BeamColumn3D,
    BeamColumn3DCorotational,
    ElasticIsotropic,
    FiberSection3D,
    Model,
    NonlinearStaticAnalysis,
    UniaxialBilinear,
)


def build_cantilever(elem_cls, *, axial_ratio: float):
    """Build a 3-D cantilever of either DB or corotational type, with
    fiber section. Returns (model, element, constants_dict)."""
    E = 2.0e11
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    sy = 400.0e6
    b_post = 0.05
    width_y, width_z = 0.2, 0.1
    L = 3.0
    n_y, n_z = 20, 10
    GJ = G * 0.229 * width_y * width_z ** 3
    mat_iso = ElasticIsotropic(1, E=E, nu=nu)
    mat_u = UniaxialBilinear(E=E, sigma_y=sy, b=b_post)
    sec = FiberSection3D.rectangular(
        width_y=width_y, width_z=width_z, n_y=n_y, n_z=n_z,
        material=mat_u, GJ=GJ,
    )
    m = Model(ndm=3, ndf=6); m.add_material(mat_iso)
    m.add_node(1, 0.0, 0.0, 0.0); m.add_node(2, L, 0.0, 0.0)
    elem = elem_cls(1, (1, 2), mat_iso, section=sec)
    m.add_element(elem)
    m.fix(1, [1, 1, 1, 1, 1, 1])

    Iz = width_z * width_y ** 3 / 12.0
    Mz_y = sy * width_z * width_y ** 2 / 6.0
    # Cantilever Euler load
    P_cr_cantilever = math.pi ** 2 * E * Iz / (4.0 * L ** 2)
    cn = dict(
        E=E, sy=sy, b_post=b_post, width_y=width_y, width_z=width_z, L=L,
        Iz=Iz, Mz_y=Mz_y, P_cr_cantilever=P_cr_cantilever,
        P_lateral_yield=Mz_y / L,
    )
    # Compose loads on the model
    P_axial = axial_ratio * P_cr_cantilever
    P_lateral_max = 1.1 * cn["P_lateral_yield"]
    m.add_nodal_load(2, [-P_axial, -P_lateral_max, 0.0, 0.0, 0.0, 0.0])
    cn["P_axial"] = P_axial
    cn["P_lateral_max"] = P_lateral_max
    return m, elem, cn


def run_pushover(m, elem):
    """Run the pushover; gracefully report non-convergence as a limit
    point caused by combined material + geometric nonlinearity."""
    try:
        res = NonlinearStaticAnalysis(
            m, num_steps=20, dlambda=1.0 / 20, tol=1e-5, max_iter=30,
            track=(2, 1),
        ).run()
        converged = True
        iters = sum(res["iter_counts"])
    except Exception:
        converged = False
        iters = -1
    n_yielded = sum(
        1 for f in elem.sections[0].fibers
        if f.material.eps_p_committed != 0.0
    )
    return {
        "converged": converged,
        "tip_disp_x": m.node(2).disp[0],
        "tip_disp_y": m.node(2).disp[1],
        "tip_disp_z": m.node(2).disp[2],
        "n_yielded": n_yielded,
        "newton_iters": iters,
    }


def main() -> None:
    print(f"\nCombined material + geometric nonlinearity in 3-D")
    print(f"  100 x 200 mm cross section, 20 x 10 = 200 bilinear fibers")
    print(f"  E = 2.0e11 Pa, sigma_y = 400 MPa, b_post = 0.05")
    print(f"  Cantilever length L = 3.0 m\n")

    axial_ratios = [0.0, 0.15, 0.30, 0.45]
    print(f"  Pushover sweep: axial load fraction of cantilever P_cr")
    print(f"  Lateral target P_lat = 1.10 * P_yield "
          f"(just past first fiber yield)\n")

    print(f"  Method                 P/P_cr   tip v (m)      "
          f"yielded   iters")
    for r in axial_ratios:
        # Displacement-based (no P-Delta)
        m_db, e_db, cn = build_cantilever(BeamColumn3D, axial_ratio=r)
        res_db = run_pushover(m_db, e_db)

        # Corotational (with P-Delta)
        m_co, e_co, _ = build_cantilever(BeamColumn3DCorotational, axial_ratio=r)
        res_co = run_pushover(m_co, e_co)

        db_status = "" if res_db["converged"] else "  (DID NOT CONVERGE)"
        co_status = "" if res_co["converged"] else "  (DID NOT CONVERGE)"
        print(f"  BeamColumn3D           {r:6.2f}   "
              f"{res_db['tip_disp_y']:+12.4e}   "
              f"{res_db['n_yielded']:>5d}/200   "
              f"{res_db['newton_iters']:>5}{db_status}")
        print(f"  BeamColumn3DCorot      {r:6.2f}   "
              f"{res_co['tip_disp_y']:+12.4e}   "
              f"{res_co['n_yielded']:>5d}/200   "
              f"{res_co['newton_iters']:>5}{co_status}")
        if r > 0.0 and res_db["converged"] and res_co["converged"]:
            amp = res_co["tip_disp_y"] / res_db["tip_disp_y"]
            print(f"  -> corotational amplification: {amp:.3f} x "
                  f"(P-Delta signature)")
        elif not res_co["converged"]:
            print(f"  -> Combined material + geometric softening past a "
                  f"limit point under load control.")
        print()

    print(f"  Reading the sweep:")
    print(f"  * For no axial load (P/P_cr = 0), corotational and DB agree —")
    print(f"    chord rotation is essentially zero, no P-Delta contribution.")
    print(f"  * As axial fraction grows, corotational tip deflection grows")
    print(f"    faster than DB (P-Delta amplification).")
    print(f"  * Yielded-fiber counts grow with axial fraction (M-P interaction):")
    print(f"    axial compression already loads the +y fibers, so the lateral")
    print(f"    load needed to push them past yield is smaller.")
    print(f"  * The corotational element captures both effects simultaneously —")
    print(f"    this is the canonical input for performance-based seismic")
    print(f"    design of 3-D frames.")


if __name__ == "__main__":
    main()
