"""Corotational fiber-section column — distributed plasticity with P-Delta.

A 3 m steel cantilever (100 x 200 mm rectangular section, 20 fibers
through the depth, bilinear-kinematic uniaxial law) is loaded with a
fixed axial compression PLUS a ramped lateral force at the tip. We
sweep over three axial-load levels and show how the lateral
force-displacement response changes:

* **No axial load** — pure fiber-section pushover. Smooth elastic-to-
  plastic transition, no geometric amplification.
* **Moderate axial compression** — P-Delta amplifies the lateral
  deflection AND advances first yield (axial compression already
  loads the +y fibers, so they hit the yield surface sooner). The
  M-P interaction reduces the effective lateral capacity.
* **Higher axial compression** — even more pronounced P-Delta, more
  pronounced reduction in lateral capacity, until the structure
  reaches a limit point under load control and the analysis stops.

This is the canonical nonlinear column analysis used in performance-
based seismic design — what OpenSees ``dispBeamColumn`` with a fiber
section + ``geomTransf Corotational`` produces, and what MIDAS
Civil's nonlinear-frame elements do under the hood.

Run::

    python examples/11_corotational_fiber_column.py
"""
from __future__ import annotations

import numpy as np

from femsolver import (
    BeamColumn2DCorotational,
    ElasticIsotropic,
    FiberSection2D,
    Model,
    NonlinearStaticAnalysis,
    UniaxialBilinear,
)


def build_cantilever(P_axial: float):
    """Build a fresh cantilever model with a fiber-corotational element
    and the requested axial preload."""
    E = 2.0e11
    sigma_y = 400.0e6
    b_post = 0.05
    b_section = 0.1
    h_section = 0.2
    L = 3.0
    n_fibers = 20

    uniaxial = UniaxialBilinear(E=E, sigma_y=sigma_y, b=b_post)
    section = FiberSection2D.rectangular(
        width=b_section, height=h_section,
        n_fibers=n_fibers, material=uniaxial,
    )
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    elem = BeamColumn2DCorotational(1, (1, 2), mat, section=section)
    m.add_element(elem)
    m.fix(1, [1, 1, 1])

    My = sigma_y * b_section * h_section ** 2 / 6.0
    Mp = sigma_y * b_section * h_section ** 2 / 4.0
    EI = E * b_section * h_section ** 3 / 12.0
    P_cr_cantilever = np.pi ** 2 * EI / (4.0 * L ** 2)
    return m, elem, L, My, Mp, EI, P_cr_cantilever


def run_pushover(P_axial_ratio: float, P_lat_max_fraction_of_My_over_L: float = 1.05):
    """Run a tip pushover with the given axial preload ratio.

    Returns ``(P_lat_max, lambdas, tip_displacements, yielded_at_end)``.
    """
    m, elem, L, My, Mp, EI, P_cr = build_cantilever(0.0)
    P_axial = P_axial_ratio * P_cr
    P_lat_max = P_lat_max_fraction_of_My_over_L * (My / L)
    m.add_nodal_load(2, [-P_axial, -P_lat_max, 0.0])
    n_steps = 30
    try:
        res = NonlinearStaticAnalysis(
            m, num_steps=n_steps, dlambda=1.0 / n_steps,
            track=(2, 1), tol=1e-5, max_iter=40,
        ).run()
        completed = True
    except Exception as exc:
        # Limit point reached under load control — return what we have
        return {
            "P_lat_max": P_lat_max,
            "lambdas": [],
            "tracked": [],
            "yielded": 0,
            "completed": False,
            "msg": str(exc),
            "My": My, "Mp": Mp, "P_cr": P_cr,
            "P_axial": P_axial, "EI": EI, "L": L,
        }
    yielded = sum(
        1 for f in elem.sections[0].fibers
        if f.material.eps_p_committed != 0.0
    )
    return {
        "P_lat_max": P_lat_max,
        "lambdas": res["lambdas"],
        "tracked": res["tracked"],
        "yielded": yielded,
        "completed": True,
        "My": My, "Mp": Mp, "P_cr": P_cr,
        "P_axial": P_axial, "EI": EI, "L": L,
    }


def main() -> None:
    print("\nCorotational fiber-section column under axial + lateral load\n")

    # Pick a few axial-load ratios that produce distinguishable curves
    ratios = [0.0, 0.2, 0.4]

    print(f"  Cross section : 100 x 200 mm, 20 fibers, bilinear (b = 0.05)")
    print(f"  Length        : 3.0 m, cantilever\n")

    results = []
    for r in ratios:
        result = run_pushover(P_axial_ratio=r,
                              P_lat_max_fraction_of_My_over_L=1.03)
        results.append(result)
        L = result["L"]
        My = result["My"]
        Mp = result["Mp"]
        P_cr = result["P_cr"]
        P_yield = My / L
        if r == ratios[0]:
            print(f"  Reference moments (from cross section):")
            print(f"    My (first fiber yield) = {My:.4g} N.m")
            print(f"    Mp (full plastic)      = {Mp:.4g} N.m")
            print(f"    Lateral force at first yield (no axial): "
                  f"{P_yield:.4g} N")
            print(f"    P_cr (cantilever Euler): {P_cr:.4g} N\n")

        if not result["completed"]:
            print(f"  P_axial / P_cr = {r:.2f}: did not converge — "
                  f"limit point under load control")
            print(f"     {result['msg']}")
            continue

        lambdas = np.array(result["lambdas"])
        disps = np.array(result["tracked"])
        forces = lambdas * result["P_lat_max"]
        # Tangent stiffness around the endpoints
        K_initial = (forces[1] - forces[0]) / abs(disps[1] - disps[0])
        if len(forces) >= 5:
            K_final = (forces[-1] - forces[-5]) / abs(disps[-1] - disps[-5])
        else:
            K_final = K_initial
        print(f"  P_axial / P_cr = {r:.2f}  (P_axial = {result['P_axial']:.3g} N)")
        print(f"    Final lateral force  : {forces[-1]:.4g} N "
              f"({forces[-1]/P_yield:.3f} x P_yield)")
        print(f"    Final tip deflection : {disps[-1]:.5e} m")
        print(f"    Tangent K (early)    : {K_initial:.3g} N/m")
        print(f"    Tangent K (late)     : {K_final:.3g} N/m  "
              f"({K_final/K_initial*100:.1f} % of early)")
        print(f"    Fibers yielded (base): {result['yielded']} / 20")
        print()

    # Quick summary of the M-P interaction
    if all(r["completed"] for r in results):
        print(f"  Summary — M-P interaction signature:\n")
        print(f"  {'P_axial/P_cr':>12}  {'tip deflection (m)':>20}  "
              f"{'fibers yielded':>16}")
        for r, res in zip(ratios, results):
            tip = res["tracked"][-1]
            print(f"  {r:12.2f}  {tip:20.5e}  {res['yielded']:>16d}")
        print()
        print(f"  Going from P_axial / P_cr = 0 to {ratios[-1]:.2f}, the tip")
        print(f"  deflection under the SAME lateral load is amplified, and")
        print(f"  more fibers have yielded — both signatures of the M-P")
        print(f"  interaction the corotational + fiber combination captures.")


if __name__ == "__main__":
    main()
