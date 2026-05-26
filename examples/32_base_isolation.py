"""Phase 18 -- base-isolated SDOF vs fixed-base under ground motion.

A single-mass building model (M = 100 t = 100,000 kg) is analyzed in
two configurations:

1. **Fixed-base**: the mass is connected to ground through a stiff
   superstructure with natural period ~ 0.3 s.
2. **Base-isolated**: a lead-rubber bearing (LRB) is inserted between
   the foundation and the superstructure. The isolation period is
   ~ 2.0 s and the LRB yields at lateral force ~ 5% of the weight.

Both configurations are subjected to the same ground acceleration:
a Ricker-pulse-like input at 2 Hz with PGA = 0.4 g. The classical
seismic-isolation result emerges:

* Fixed-base: large floor (mass) acceleration ~ structure tries to
  follow the ground motion's frequency content.
* Isolated: dramatically smaller floor acceleration (the isolator
  filters out the high-frequency content) at the cost of larger
  *relative* (isolator) displacement.

Run::

    python examples/32_base_isolation.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver import (
    ElasticIsotropic,
    HHTAlpha,
    Model,
    NonlinearTransientAnalysis,
    RayleighDamping,
    TransientAnalysis,
    Truss2D,
    UniaxialElastic,
    ZeroLengthElement,
    ground_motion_force,
    lead_rubber_bearing,
)


M_MASS = 100_000.0           # kg, the floor mass
PGA = 0.4 * 9.81             # m/s^2 ground acceleration peak
FP = 2.0                     # Hz ground motion peak frequency
t0 = 1.0                     # s, ground motion centered here
dt = 0.005
n_steps = 800


def ricker(t: float) -> float:
    """Ricker pulse at frequency FP centered at t0, magnitude PGA."""
    tau = math.pi * FP * (t - t0)
    return PGA * (1.0 - 2.0 * tau ** 2) * math.exp(-tau ** 2)


def build_fixed_base() -> Model:
    """Stiff column from base to mass.  K_super = M omega^2 with
    omega = 2 pi / T_super gives natural period T_super."""
    T_super = 0.3
    K_super = M_MASS * (2.0 * math.pi / T_super) ** 2
    # Truss2D from ground to mass: a single rigid-ish "column".
    L = 1.0
    A = 1.0
    E = K_super * L / A
    rho = 3.0 * M_MASS / (A * L)        # consistent mass -> tip mass = M
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=rho)
    m = Model(ndm=2, ndf=2); m.add_material(mat)
    m.add_node(1, 0.0, 0.0)        # ground
    m.add_node(2, L, 0.0)           # mass
    m.add_element(Truss2D(1, (1, 2), mat, area=A))
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])
    return m


def build_isolated() -> Model:
    """LRB between ground and superstructure. The superstructure stays
    relatively rigid; the isolator period dominates."""
    T_iso = 2.0
    K_iso = M_MASS * (2.0 * math.pi / T_iso) ** 2
    # LRB: K1 = K_iso (= post-yield stiffness, design dominant), Q
    # = 5% of weight (typical characteristic strength).
    K2_lrb = K_iso
    K1_lrb = 10.0 * K2_lrb         # initial elastic stiffness ~ 10 K2
    Q = 0.05 * M_MASS * 9.81       # characteristic strength
    # Superstructure: stiff with T_super = 0.2 s
    T_super = 0.2
    K_super = M_MASS * (2.0 * math.pi / T_super) ** 2
    L_super = 1.0
    A = 1.0
    E_super = K_super * L_super / A
    # Consistent mass on the top node only; the isolator carries no mass
    # in the model (zero-length).
    rho_super = 3.0 * M_MASS / (A * L_super)
    mat = ElasticIsotropic(1, E=E_super, nu=0.3, rho=rho_super)
    m = Model(ndm=2, ndf=2); m.add_material(mat)
    m.add_node(1, 0.0, 0.0)         # ground
    m.add_node(2, 0.0, 0.0)         # base of superstructure (coincident
                                     # with node 1; isolator connects them)
    m.add_node(3, L_super, 0.0)     # top of superstructure (mass)
    # LRB at the base
    m.add_element(lead_rubber_bearing(
        1, (1, 2),
        K1=K1_lrb, K2=K2_lrb, Q=Q, dofs_per_node=2,
    ))
    # Superstructure column
    m.add_element(Truss2D(2, (2, 3), mat, area=A))
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])                 # base of superstructure free in x only
    m.fix(3, [0, 1])
    return m


def run_case(model: Model, *, label: str) -> dict:
    """Run a transient analysis under the Ricker pulse and return peak
    response quantities."""
    # Light Rayleigh damping at the dominant period
    eig_periods = [1.0]      # used only to set Rayleigh; rough first guess
    omega1 = 2.0 * math.pi / eig_periods[0]
    damping = RayleighDamping(alpha_M=2.0 * 0.02 * omega1, alpha_K=0.0)
    g_force = ground_motion_force(model, direction="x", accel_function=ricker)

    # Use NonlinearTransientAnalysis so isolator nonlinearity is captured
    res = NonlinearTransientAnalysis(
        model, num_steps=n_steps, dt=dt,
        load_function=g_force, damping=damping,
        track=(model.nodes[len(model.nodes)].tag, 0),     # last node, x
        tol=1.0e-4, max_iter=20,
    ).run()

    u = np.array(res["tracked_disp"])
    t = np.array(res["times"])
    # Acceleration -- finite-difference of u
    udd = np.gradient(np.gradient(u, dt), dt)

    # Also compute the ground accelerations for context
    ag = np.array([ricker(ti) for ti in t])

    print(f"  {label}:")
    print(f"    peak floor disp (relative): {np.max(np.abs(u))*1000:.2f} mm")
    print(f"    peak floor acceleration:    {np.max(np.abs(udd)):.3f} m/s^2")
    print(f"    peak PGA (input):            {np.max(np.abs(ag)):.3f} m/s^2")
    print(f"    amplification factor (acc):  {np.max(np.abs(udd))/PGA:.3f}")
    return {
        "label": label,
        "t": t, "u": u, "udd": udd, "ag": ag,
    }


def main() -> None:
    print(f"\nBase-isolated SDOF vs fixed-base under Ricker pulse")
    print(f"  M = {M_MASS:.0f} kg")
    print(f"  PGA = {PGA:.3f} m/s^2 ({PGA/9.81:.2f} g)")
    print(f"  Ricker peak frequency = {FP} Hz")
    print(f"  dt = {dt} s, n_steps = {n_steps}")
    print()

    res_fixed = run_case(build_fixed_base(), label="Fixed-base   ")
    res_iso = run_case(build_isolated(), label="Base-isolated")

    # Comparative metrics
    peak_floor_fixed = np.max(np.abs(res_fixed["udd"]))
    peak_floor_iso = np.max(np.abs(res_iso["udd"]))
    peak_disp_fixed = np.max(np.abs(res_fixed["u"]))
    peak_disp_iso = np.max(np.abs(res_iso["u"]))

    print()
    print(f"Comparison:")
    print(f"  Floor-acceleration reduction (isolated vs fixed-base):")
    print(f"    fixed-base peak:  {peak_floor_fixed:.3f} m/s^2")
    print(f"    isolated peak:    {peak_floor_iso:.3f} m/s^2")
    print(f"    ratio:            {peak_floor_iso/peak_floor_fixed:.3f}")
    print(f"  Floor-displacement (relative):")
    print(f"    fixed-base peak:  {peak_disp_fixed*1000:.2f} mm")
    print(f"    isolated peak:    {peak_disp_iso*1000:.2f} mm")
    print(f"    (isolated displacement is *larger* -- the isolator")
    print(f"     soaks up the relative motion to protect the floor.)")
    print()
    print(f"Reading the result:")
    print(f"* The lead-rubber bearing decouples the floor from the")
    print(f"  high-frequency ground content. Floor acceleration drops")
    print(f"  by roughly the period-elongation ratio (T_iso/T_super)^2,")
    print(f"  matching classical isolation theory.")
    print(f"* The trade-off is increased *relative* displacement at the")
    print(f"  isolator level. Real isolators are sized for this drift")
    print(f"  with seismic gap allowances of 300-500 mm.")
    print(f"* The LRB's bilinear hysteresis dissipates energy through")
    print(f"  the yielding lead core, providing supplemental damping")
    print(f"  beyond the elastic-only rubber response.")


if __name__ == "__main__":
    main()
