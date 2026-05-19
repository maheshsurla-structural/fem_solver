"""Trace the EPP-hinge plateau with displacement control.

This is the example that motivated Phase 10. In Phase 4 we built
``HingedBeamColumn2D`` with concentrated plastic hinges, and showed
that under load control an EPP hinge at the base of a cantilever
**cannot** be pushed past ``P_y`` — the column becomes a kinematic
mechanism the instant the hinge yields, and Newton diverges. The
canonical work-around in commercial practice is to switch to
displacement control: prescribe the tip displacement and let the load
factor ``lambda`` come out as a solution variable.

Here we ramp the tip down by 4 x its yield deflection, well past the
mechanism point, and print the load-displacement trace. The expected
shape is a clean bilinear: elastic rise to ``P_y`` followed by an
absolutely flat plateau at ``P_y``.

Run::

    python examples/12_displacement_control_epp_plateau.py
"""
from __future__ import annotations

import numpy as np

from femsolver import (
    BilinearMomentRotationSpring,
    DisplacementControl,
    ElasticIsotropic,
    HingedBeamColumn2D,
    Model,
    NonlinearStaticAnalysis,
)


def main() -> None:
    # Cantilever with an EPP hinge at the base
    E = 2.0e11
    A = 1.0e-2
    Iz = 8.333e-6
    L = 3.0
    K_h = 4.0 * E * Iz / L        # comparable to beam's 4 EI/L
    My = 5.0e3                     # yield moment (N.m)
    P_yield = My / L               # tip force at first yield

    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    hinge = BilinearMomentRotationSpring(K0=K_h, My=My, b=0.0)   # EPP
    elem = HingedBeamColumn2D(1, (1, 2), mat, A, Iz, hinge_i=hinge)
    m.add_element(elem)
    m.fix(1, [1, 1, 1])
    # Reference load: -1 N (downward). lambda is then the tip force.
    m.add_nodal_load(2, [0.0, -1.0, 0.0])

    # Elastic tip deflection at yield:
    #   v_y = -P_y * L^3 / (3 EI) - P_y * L^2 / K_h
    v_y = -P_yield * L ** 3 / (3.0 * E * Iz) - P_yield * L ** 2 / K_h
    # Push 4x past yield in 40 steps so the plateau is clearly visible
    v_target = 4.0 * v_y
    n_steps = 40
    integrator = DisplacementControl(
        node_tag=2, dof_index=1, du_step=v_target / n_steps,
    )

    res = NonlinearStaticAnalysis(
        m, num_steps=n_steps, integrator=integrator,
        algorithm="newton", tol=1e-6, max_iter=20,
        track=(2, 1),
    ).run()

    lambdas = np.array(res["lambdas"])
    disps = np.array(res["tracked"])

    print("\nCantilever with EPP base hinge — displacement-control pushover\n")
    print(f"  L  = {L} m")
    print(f"  EI = {E * Iz:.3g} N.m^2")
    print(f"  M_y / L (first yield force) = {P_yield:.4g} N")
    print(f"  v_y (elastic deflection at yield) = {v_y:.4e} m")
    print(f"  v_target (4 x v_y)                = {v_target:.4e} m\n")

    # Print every fifth step plus first and last
    print(f"  {'step':>5}  {'v_tip (m)':>13}  {'P (N)':>10}  "
          f"{'P / P_y':>9}  {'theta_p (rad)':>14}")
    for i, (v, lam) in enumerate(zip(disps, lambdas), 1):
        # theta_p at the hinge at the end of the analysis is the committed
        # value (we only have one value to report after .run()).
        if i == 1 or i % 5 == 0 or i == n_steps:
            tp_str = (f"{hinge.theta_p_committed:14.4e}"
                      if i == n_steps else " " * 14)
            print(f"  {i:5d}  {v:13.4e}  {lam:10.3f}  {lam/P_yield:9.4f}  {tp_str}")
    print()

    # Confirm plateau: after yield, lambda stays at P_yield to within
    # round-off.
    plateau = lambdas[len(lambdas) // 2:]
    plateau_mean = plateau.mean()
    plateau_spread = plateau.max() - plateau.min()
    print(f"  Plateau-region statistics (second half of steps):")
    print(f"    mean lambda = {plateau_mean:.6f}  "
          f"(P_y = {P_yield:.6f})")
    print(f"    spread      = {plateau_spread:.3e}  "
          f"({plateau_spread / P_yield * 100:.4f} % of P_y)")
    print()
    print(f"  The plateau is flat to within {plateau_spread / P_yield * 100:.2f} %, ")
    print(f"  matching the EPP analytical behavior: once the hinge yields,")
    print(f"  the structure carries no additional load — only plastic")
    print(f"  rotation accumulates.")
    print()
    print(f"  Compare with Phase 4's test_epp_load_past_yield_diverges:")
    print(f"  under *load* control the same problem cannot be pushed past")
    print(f"  P_y at all (the column becomes a kinematic mechanism). The")
    print(f"  displacement-control formulation handles this region cleanly.")


if __name__ == "__main__":
    main()
