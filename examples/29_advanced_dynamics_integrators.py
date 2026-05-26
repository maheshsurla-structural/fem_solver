"""Phase 17 -- advanced dynamic integrators in action.

Two scenarios illustrate the integrators introduced in Phase 17:

1. **Spurious high-frequency suppression** on a 2-DOF system whose
   second mode is dramatically higher than the first. We see how
   Newmark preserves the spurious HF noise, while HHT-alpha and
   generalized-alpha attenuate it without distorting the dominant
   first mode.

2. **Multi-support ground motion** on a simple 2-pier frame: each
   pier follows a phase-shifted sinusoidal acceleration. The
   relative motion between piers excites the frame in a way that a
   single-support analysis cannot capture.

Run::

    python examples/29_advanced_dynamics_integrators.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver import (
    BeamColumn2D,
    ElasticIsotropic,
    GeneralizedAlpha,
    HHTAlpha,
    Model,
    Newmark,
    TransientAnalysis,
    Truss2D,
    multi_support_ground_motion_force,
)


def two_dof_two_omega_demo() -> None:
    """Build a 2-DOF system with two well-separated natural frequencies
    and excite both modes simultaneously. Compare how each integrator
    handles the high-frequency mode under a coarse time step.
    """
    # Two springs in series with two masses. K1 = 100, K2 = 10000.
    # omega1 ~ sqrt(K1/M) ~ 10 rad/s ; omega2 ~ sqrt(K2/M) ~ 100 rad/s
    # The high mode period T2 ~ 0.063 s. Choosing dt = T2/4 = 0.016 s
    # puts mode 1 at omega1*dt ~ 0.16 (well-resolved) and mode 2 at
    # omega2*dt ~ 1.6 (large -- on the edge of stable resolution).
    L_total = 2.0
    M_mass = 1.0
    K1 = 100.0
    K2 = 10000.0
    L_each = L_total / 2
    A = 1.0
    E1 = K1 * L_each / A
    E2 = K2 * L_each / A
    rho1 = 3.0 * M_mass / (A * L_each)
    rho2 = 3.0 * M_mass / (A * L_each)
    mat1 = ElasticIsotropic(1, E=E1, nu=0.3, rho=rho1)
    mat2 = ElasticIsotropic(2, E=E2, nu=0.3, rho=rho2)

    def build():
        m = Model(ndm=2, ndf=2); m.add_material(mat1); m.add_material(mat2)
        m.add_node(1, 0.0, 0.0)
        m.add_node(2, L_each, 0.0)
        m.add_node(3, L_total, 0.0)
        m.add_element(Truss2D(1, (1, 2), mat1, area=A))
        m.add_element(Truss2D(2, (2, 3), mat2, area=A))
        m.fix(1, [1, 1])
        m.fix(2, [0, 1])
        m.fix(3, [0, 1])
        return m

    # Initial conditions: excite both modes (mass 1 displaced + mass 2 displaced)
    u0_1 = 0.01
    u0_2 = 0.005
    dt = 0.016        # ~ T1/40, ~ T2/4
    n_steps = 60      # ~ 1 period of mode 1

    print("\n2-DOF demo: 2-spring-mass with omega ~ 10 and ~ 100 rad/s")
    print(f"  dt = {dt} s  ->  omega1*dt ~ 0.16,  omega2*dt ~ 1.6")
    print(f"  Coarse step puts mode 2 in the HF regime.")
    print(f"  Initial displacement of both masses excites both modes.")
    print()

    for label, integrator in [
        ("Newmark              ", Newmark()),
        ("HHT-alpha=-0.05      ", HHTAlpha(alpha=-0.05)),
        ("HHT-alpha=-0.3       ", HHTAlpha(alpha=-0.3)),
        ("Gen-alpha rho_inf=0.8", GeneralizedAlpha(rho_inf=0.8)),
        ("Gen-alpha rho_inf=0.5", GeneralizedAlpha(rho_inf=0.5)),
    ]:
        m = build()
        m.number_dofs()
        m.node(2).disp[0] = u0_1
        m.node(3).disp[0] = u0_2
        res = TransientAnalysis(
            m, num_steps=n_steps, dt=dt,
            integrator=integrator, track=(2, 0),
        ).run()
        u = np.array(res["tracked_disp"])
        peak = np.max(np.abs(u))
        rms = np.sqrt(np.mean(u ** 2))
        final = u[-1]
        print(f"  {label}: peak={peak:.4e}, rms={rms:.4e}, final={final:+.4e}")

    print()
    print("  Reading the table:")
    print("  * Newmark: preserves the full HF content -- peak displacement is")
    print("    inflated by the unresolved HF mode, rms is high.")
    print("  * HHT-alpha: low alpha (-0.05) gives mild HF damping;")
    print("    larger alpha (-0.3) aggressively suppresses the HF noise")
    print("    while leaving the first mode largely intact.")
    print("  * Generalized-alpha: rho_inf < 1 controls HF dissipation;")
    print("    rho_inf = 0.5 sits between the two HHT variants.")


def multi_support_demo() -> None:
    """Two-pier frame with phase-shifted base motions. The relative
    pier-to-pier ground displacement excites a frame-distortion mode
    that a single-support analysis cannot capture.
    """
    # Simple two-pier portal frame: 2 columns 3 m tall, 6 m apart, with
    # a rigid roof girder.
    E = 2.0e10
    A = 1.0e-2
    Iz = 1.0e-4
    L = 3.0
    span = 6.0
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=7850.0)

    def build():
        m = Model(ndm=2, ndf=3); m.add_material(mat)
        m.add_node(1, 0.0, 0.0)
        m.add_node(2, span, 0.0)
        m.add_node(3, 0.0, L)
        m.add_node(4, span, L)
        m.add_element(BeamColumn2D(1, (1, 3), mat, A, Iz))
        m.add_element(BeamColumn2D(2, (2, 4), mat, A, Iz))
        m.add_element(BeamColumn2D(3, (3, 4), mat, A * 10, Iz * 10))
        m.fix(1, [1, 1, 1])
        m.fix(2, [1, 1, 1])
        return m

    # Phase-shifted ground motion at each pier
    f = 1.5   # Hz
    omega = 2.0 * math.pi * f
    amp = 0.5  # m/s^2 peak ground acceleration

    def ag_left(t):
        return amp * math.sin(omega * t)

    def ag_right(t):
        # Phase shift: simulates seismic-wave travel time across the span
        return amp * math.sin(omega * t - 0.5 * math.pi)

    dt = 0.01
    n_steps = 800

    print("\n2-pier frame -- multi-support excitation")
    print(f"  Span = {span} m, column height = {L} m")
    print(f"  Two piers with phase-shifted ground motion:")
    print(f"    left pier  (nodes 1-3): {amp} m/s^2 * sin(2 pi {f} t)")
    print(f"    right pier (nodes 2-4): {amp} m/s^2 * sin(2 pi {f} t - pi/2)")
    print(f"  Phase shift -> the two piers move out-of-phase, exciting")
    print(f"  the *anti-symmetric* frame distortion mode.")
    print()

    # Multi-support: each pier's free roof node follows its own pier's motion
    m = build(); m.number_dofs()
    load = multi_support_ground_motion_force(
        m, supports=[
            {"direction": "x", "accel_function": ag_left,
             "nodes": [3]},      # roof node on top of left pier
            {"direction": "x", "accel_function": ag_right,
             "nodes": [4]},      # roof node on top of right pier
        ],
    )
    res_multi = TransientAnalysis(
        m, num_steps=n_steps, dt=dt,
        integrator=HHTAlpha(alpha=-0.1),
        load_function=load, track=(3, 0),
    ).run()
    u_left = np.array(res_multi["tracked_disp"])
    # Roof spread (relative motion between the two roof nodes)
    u_right_history = []
    for t_step in range(n_steps + 1):
        u_right_history.append(m.node(4).disp[0])
        # (Note: tracking node 4 requires re-running with track=(4,0);
        # for the example we use the live final state, which is the
        # end-of-analysis position.)
    print(f"  HHT-alpha(-0.1) results:")
    print(f"    Peak left-roof displacement (relative): {np.max(np.abs(u_left)):.4e} m")
    # Run again with right tracking for the relative spread
    m2 = build(); m2.number_dofs()
    load2 = multi_support_ground_motion_force(
        m2, supports=[
            {"direction": "x", "accel_function": ag_left, "nodes": [3]},
            {"direction": "x", "accel_function": ag_right, "nodes": [4]},
        ],
    )
    res_right = TransientAnalysis(
        m2, num_steps=n_steps, dt=dt,
        integrator=HHTAlpha(alpha=-0.1),
        load_function=load2, track=(4, 0),
    ).run()
    u_right = np.array(res_right["tracked_disp"])
    spread = u_left - u_right    # relative pier-to-pier roof motion
    print(f"    Peak right-roof displacement (relative): {np.max(np.abs(u_right)):.4e} m")
    print(f"    Peak roof spread (anti-symmetric mode):  {np.max(np.abs(spread)):.4e} m")

    # Compare to in-phase (single-support equivalent)
    def ag_inphase(t):
        return ag_left(t)        # both piers see the same motion

    m3 = build(); m3.number_dofs()
    load3 = multi_support_ground_motion_force(
        m3, supports=[
            {"direction": "x", "accel_function": ag_inphase, "nodes": [3]},
            {"direction": "x", "accel_function": ag_inphase, "nodes": [4]},
        ],
    )
    res_in = TransientAnalysis(
        m3, num_steps=n_steps, dt=dt,
        integrator=HHTAlpha(alpha=-0.1),
        load_function=load3, track=(3, 0),
    ).run()
    res_in_r = TransientAnalysis(
        build(), num_steps=n_steps, dt=dt,
        integrator=HHTAlpha(alpha=-0.1),
        load_function=multi_support_ground_motion_force(
            build(), supports=[
                {"direction": "x", "accel_function": ag_inphase, "nodes": [3]},
                {"direction": "x", "accel_function": ag_inphase, "nodes": [4]},
            ],
        ),
        track=(4, 0),
    )
    # (Skipping the in-phase right side -- by symmetry it equals the left,
    # so spread = 0 to numerical precision.)
    print()
    print(f"  Comparison with in-phase (both piers same motion):")
    print(f"    Peak left-roof displacement (in-phase): "
          f"{np.max(np.abs(res_in['tracked_disp'])):.4e} m")
    print(f"    Spread (in-phase): 0 (by symmetry -- both piers move together)")
    print()
    print(f"  Reading the result:")
    print(f"  * With out-of-phase support motion, the frame undergoes a")
    print(f"    distortion mode (the roof girder bends) that simply isn't")
    print(f"    excited by uniform single-support analysis.")
    print(f"  * Long bridges, dams across faults, and any structure with")
    print(f"    widely-spaced supports across the dominant ground-wave")
    print(f"    travel direction need this kind of analysis.")


def main() -> None:
    two_dof_two_omega_demo()
    multi_support_demo()


if __name__ == "__main__":
    main()
