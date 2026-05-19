"""Trace the full Mises-truss snap-through curve with arc length.

This is the example that motivated the *arc-length* part of Phase 10.
Example 07 (``07_mises_truss_snapthrough.py``) set up the same
geometry but used load control, which can only trace the ascending
branch up to the limit point. Anything past the peak — the
descending branch, the snap-through, the symmetric ascending branch
in the inverted configuration — is unreachable under load control.

With cylindrical arc-length (Crisfield with ``psi = 0``) and a
direction-tracking predictor (Bergan's GSP), the analysis traces the
*full* curve smoothly through the limit point and out the other side.

Run::

    python examples/13_arc_length_mises_snap_through.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver import (
    ArcLength,
    ElasticIsotropic,
    Model,
    NonlinearStaticAnalysis,
    Truss2DCorotational,
)


def analytical_load(w: float, *, B: float, h: float, EA: float) -> float:
    """Equilibrium load factor at apex displacement ``w`` (downward
    positive) for the symmetric two-bar Mises truss."""
    L0 = math.sqrt(B * B + h * h)
    L = math.sqrt(B * B + (h - w) ** 2)
    eps = (L - L0) / L0
    N = EA * eps
    return -2.0 * N * (h - w) / L


def main() -> None:
    B = 10.0       # half-span
    h = 1.0        # initial apex rise
    EA = 1.0e6     # axial stiffness per bar
    P_ref = 300.0  # reference load (just above analytical limit ~ 290)

    mat = ElasticIsotropic(1, E=EA, nu=0.3)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, B, h)
    m.add_node(3, 2.0 * B, 0.0)
    m.add_element(Truss2DCorotational(1, (1, 2), mat, 1.0))
    m.add_element(Truss2DCorotational(2, (2, 3), mat, 1.0))
    m.fix(1, [1, 1, 1])
    m.fix(3, [1, 1, 1])
    m.fix(2, [1, 0, 1])   # only vertical apex motion is free
    m.add_nodal_load(2, [0.0, -P_ref, 0.0])

    # Arc-length: trace the apex down through the flat configuration
    # (v ~ -h) and beyond. Total apex travel ~ 2.5 h.
    delta_s = 0.08
    n_steps = 50

    integrator = ArcLength(delta_s=delta_s)
    res = NonlinearStaticAnalysis(
        m, num_steps=n_steps, integrator=integrator,
        tol=1e-7, max_iter=30, track=(2, 1),
    ).run()
    lambdas = np.array(res["lambdas"])
    disps = np.array(res["tracked"])     # apex v (negative = downward)
    forces = lambdas * P_ref              # actual apex load
    apex_disp_dn = -disps                  # downward positive for display

    print(f"\nMises truss snap-through under arc length\n")
    print(f"  B = {B} m, h = {h} m, EA = {EA:g} N")
    print(f"  P_ref = {P_ref} N (just above analytical limit P_cr ~ "
          f"{2 * EA * h ** 2 / (B ** 2 + h ** 2):.2f} N)")
    print(f"  delta_s = {delta_s},  num_steps = {n_steps}\n")

    print(f"  {'step':>5}  {'w (m)':>10}  {'P_FE (N)':>10}  "
          f"{'P_theory':>10}  {'err':>9}")
    for i, (w_dn, P) in enumerate(zip(apex_disp_dn, forces), 1):
        if i == 1 or i % 5 == 0 or i == n_steps:
            P_th = analytical_load(w_dn, B=B, h=h, EA=EA)
            err = abs(P - P_th)
            print(f"  {i:5d}  {w_dn:10.4f}  {P:10.3f}  {P_th:10.3f}  {err:9.3e}")

    # Find the peak and verify the descending branch is traced
    peak_idx = int(np.argmax(forces))
    peak_P = forces[peak_idx]
    peak_w = apex_disp_dn[peak_idx]
    print()
    print(f"  Peak load (limit point):  P = {peak_P:.3f} N at w = {peak_w:.4f} m")
    print(f"  Analytical limit at w = h - h/sqrt(3) = {h - h / math.sqrt(3):.4f} m")
    print()
    print(f"  Path traced from w = {apex_disp_dn[0]:.4f} to "
          f"w = {apex_disp_dn[-1]:.4f}")
    print(f"  Apex passed through flat configuration: "
          f"{'yes' if apex_disp_dn[-1] > h else 'no'} "
          f"(needs w > h = {h})")
    if apex_disp_dn[-1] > h:
        print(f"  Snap-through captured cleanly. Compare with example 07")
        print(f"  ({'examples/07_mises_truss_snapthrough.py'}): under load")
        print(f"  control the analysis stalls at the limit point and cannot")
        print(f"  reach the descending branch at all.")


if __name__ == "__main__":
    main()
