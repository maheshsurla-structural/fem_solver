"""Cyclic-load hysteresis of a cantilever with a yielding plastic hinge.

A 3 m steel cantilever with a bilinear M-theta hinge at the base is
loaded by a tip force that oscillates in time as a sinusoid with
ramped amplitude. After a few cycles the amplitude is large enough
to yield the hinge; subsequent cycles trace the canonical
moment-rotation hysteresis loop characteristic of kinematic hardening.

This is the canonical demo for nonlinear dynamic structural analysis:
the same calculation that performance-based seismic design tools
(SAP2000 nonlinear, OpenSees, MIDAS Civil's nonlinear-frame analyses)
perform under earthquake ground motion. The sinusoidal load stands in
for a real ground motion for clarity.

We print:
* Per-step Newton-iteration counts (to demonstrate the integrator is
  doing real work post-yield),
* The peak hinge plastic rotation reached,
* A coarse summary of the force-displacement curve.

Run::

    python examples/16_nonlinear_cyclic_hysteresis.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver import (
    BeamColumn2D,
    BilinearMomentRotationSpring,
    ElasticIsotropic,
    HingedBeamColumn2D,
    Model,
    NonlinearTransientAnalysis,
    RayleighDamping,
)


def build_cantilever_with_hinge(*, n_elem: int = 4, b_post: float = 0.05):
    """Cantilever with a bilinear hinge at the base."""
    E = 2.0e11
    A = 1.0e-2
    Iz = 8.333e-6
    L = 3.0
    rho = 7850.0
    K_h = 4.0 * E * Iz / L
    My = 5.0e3
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=rho)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    for i in range(n_elem + 1):
        m.add_node(i + 1, i * L / n_elem, 0.0)
    hinge = BilinearMomentRotationSpring(K0=K_h, My=My, b=b_post)
    m.add_element(
        HingedBeamColumn2D(1, (1, 2), mat, A, Iz, hinge_i=hinge)
    )
    for i in range(1, n_elem):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, Iz))
    m.fix(1, [1, 1, 1])
    tip_tag = n_elem + 1
    return m, hinge, tip_tag, dict(E=E, A=A, Iz=Iz, L=L, K_h=K_h, My=My, b_post=b_post)


def main() -> None:
    model, hinge, tip_tag, cn = build_cantilever_with_hinge(b_post=0.05)
    # Reference load: 1 N applied transversely at the tip. The load
    # function scales this in time.
    model.add_nodal_load(tip_tag, [0.0, -1.0, 0.0])

    P_y = cn["My"] / cn["L"]                # first-yield tip force

    # Loading: sinusoid with ramped amplitude. Final peak amplitude
    # ~ 1.4 P_y so we cycle the hinge well past yield.
    P_peak = 1.4 * P_y
    T_load = 0.6                             # 0.6 s period (slow vs eigenmodes)
    omega_load = 2.0 * math.pi / T_load
    total_time = 3.0 * T_load                # 3 cycles
    ramp_time = T_load                       # ramp up over the first cycle

    def load(t: float) -> float:
        amp = P_peak * min(1.0, t / ramp_time)
        return amp * math.sin(omega_load * t)

    # 5% damping at the load frequency for stability of high-frequency
    # numerical noise. Mass-proportional only.
    damping = RayleighDamping(alpha_M=2.0 * 0.05 * omega_load, alpha_K=0.0)

    dt = T_load / 100.0
    n_steps = int(total_time / dt)
    analysis = NonlinearTransientAnalysis(
        model, num_steps=n_steps, dt=dt,
        damping=damping, load_function=load,
        track=(tip_tag, 1),
        tol=1.0e-5, max_iter=30,
    )
    res = analysis.run()

    times = np.array(res["times"])
    disps = np.array(res["tracked_disp"])
    forces = np.array([load(t) for t in times])
    iters = res["iter_counts"]

    print(f"\nCyclic-pushover hysteresis of a cantilever with base hinge")
    print(f"  Hinge: K0 = {cn['K_h']:.3g} N.m/rad,  M_y = {cn['My']} N.m,")
    print(f"         b = {cn['b_post']} (kinematic hardening)")
    print(f"  Loading: sinusoid, amplitude P_peak = {P_peak:.1f} N "
          f"(= {P_peak / P_y:.2f} x P_y),")
    print(f"           period T_load = {T_load} s, ramped over the first cycle")
    print(f"  P_y (first yield tip force) = {P_y:.3f} N")
    print(f"  Damping: 5 % mass-proportional at omega_load")
    print(f"  Time step dt = {dt:.4f} s,  total = {total_time} s "
          f"({n_steps} steps)\n")

    print(f"  Newton-iteration counts (per step):")
    print(f"    min = {min(iters[1:])}, max = {max(iters[1:])}, "
          f"avg = {sum(iters[1:]) / max(1, len(iters[1:])):.2f}")
    print(f"    First 30: {iters[1:31]}")
    print()

    print(f"  Hinge state at end of analysis:")
    print(f"    theta_p = {hinge.theta_p_committed:.4e} rad")
    print(f"    Has yielded: {abs(hinge.theta_p_committed) > 1e-12}")
    print()

    # Tip-displacement extrema
    print(f"  Tip-displacement extrema (cycle-by-cycle):")
    cycle_end_indices = [int((k + 1) * T_load / dt) for k in range(3)]
    for k, idx in enumerate(cycle_end_indices, 1):
        if idx >= len(disps):
            idx = len(disps) - 1
        # Find min/max in this cycle window.
        start = (k - 1) * int(T_load / dt) + 1
        end = idx + 1
        window_u = disps[start:end]
        if window_u.size:
            print(f"    Cycle {k}: u_min = {window_u.min():+.4e} m, "
                  f"u_max = {window_u.max():+.4e} m")
    print()

    # Coarse hysteresis-loop summary: print a few (force, displacement)
    # pairs spaced uniformly across the load history.
    print(f"  Coarse (P, u) trace through the cycles:")
    sample_idx = np.linspace(0, len(disps) - 1, 12, dtype=int)
    print(f"  {'t (s)':>9}  {'P (N)':>10}  {'u (m)':>14}")
    for idx in sample_idx:
        print(f"  {times[idx]:9.3f}  {forces[idx]:10.2f}  {disps[idx]:+14.4e}")
    print()
    print("  Past the elastic limit the curve traces a hysteresis loop —")
    print("  area enclosed = energy dissipated per cycle. This is the")
    print("  signature of plastic-hinge cyclic behaviour, equivalent to")
    print("  what OpenSees / MIDAS produce for a hinge-equipped frame")
    print("  under a transient lateral load.")


if __name__ == "__main__":
    main()
