"""Shallow von-Mises (Mises) truss — geometrically nonlinear load-deflection
trace under load control.

Two corotational trusses meeting at an apex carry a downward load. As the
apex moves down, the trusses stiffen geometrically at first, then soften
as they rotate toward horizontal — eventually reaching a *limit point*
beyond which the structure snaps through to a new equilibrium. With a
plain ``LoadControl`` integrator we can only trace the stable, ascending
branch up to (but not past) the limit point. Capturing the full
snap-through requires arc-length control, which is the natural Phase-2
follow-up.

The reference equilibrium relation for the apex displacement ``w``
(downward positive) is

    P(w) = -2 N (h - w) / L,    L = sqrt(B^2 + (h - w)^2),
    N = E A (L - L0) / L0,      L0 = sqrt(B^2 + h^2)

Run::

    python examples/07_mises_truss_snapthrough.py
"""
from __future__ import annotations

import math

from femsolver import (
    ElasticIsotropic,
    Model,
    NonlinearStaticAnalysis,
    NotConvergedError,
    Truss2DCorotational,
)


def analytical_load(w: float, *, B: float, h: float, EA: float) -> float:
    L0 = math.sqrt(B * B + h * h)
    L = math.sqrt(B * B + (h - w) ** 2)
    eps = (L - L0) / L0
    N = EA * eps
    return -2.0 * N * (h - w) / L


def main() -> None:
    B = 10.0      # half-span
    h = 1.0       # initial rise
    EA = 1.0e6    # axial stiffness per member
    P_target = 280.0   # target load — close to but below the analytical limit (~290)
    n_steps = 28
    dlambda = 1.0 / n_steps

    m = Model(ndm=2, ndf=2)
    mat = ElasticIsotropic(1, E=EA, nu=0.0)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 2.0 * B, 0.0)
    m.add_node(3, B, h)
    m.fix(1, [1, 1])
    m.fix(2, [1, 1])
    m.fix(3, [1, 0])  # constrain horizontal motion at apex (symmetric anyway)
    m.add_element(Truss2DCorotational(1, (1, 3), mat, area=1.0))
    m.add_element(Truss2DCorotational(2, (3, 2), mat, area=1.0))
    m.add_nodal_load(3, [0.0, -P_target])

    a = NonlinearStaticAnalysis(
        m, num_steps=n_steps, dlambda=dlambda,
        convergence="unbalance", tol=1e-6, max_iter=40,
        track=(3, 1),
    )
    try:
        info = a.run()
        terminated_at = "all steps converged"
    except NotConvergedError as exc:
        info = {
            "final_lambda": a.integrator.lambd,
            "lambdas": list(a.lambdas),
            "tracked": list(a.tracked),
            "iter_counts": list(a.iter_counts),
        }
        terminated_at = f"diverged at step {len(a.lambdas) + 1}: {exc}"

    print("Mises shallow truss — load control trace")
    print(f"  geometry: B={B} m, h={h} m, EA={EA:.1e} N")
    print(f"  target P = {P_target:.1f} N over {n_steps} steps")
    print(f"  termination: {terminated_at}")
    print(f"  final lambda: {info['final_lambda']:.4f}  ({info['final_lambda'] * P_target:.2f} N)")
    print(f"  total Newton iterations: {sum(info['iter_counts'])}")
    print()
    print(f"  {'step':>4} {'lambda':>8} {'P (N)':>10} {'-w (m)':>10} "
          f"{'P_analytic':>12} {'rel err':>10} {'iters':>6}")
    for k, (lam, w_obs, n_it) in enumerate(zip(info['lambdas'], info['tracked'], info['iter_counts'])):
        w = -w_obs
        P_app = lam * P_target
        P_an = analytical_load(w, B=B, h=h, EA=EA)
        rel = (P_app - P_an) / P_an if P_an != 0 else 0.0
        print(f"  {k+1:>4} {lam:8.4f} {P_app:10.3f} {w:10.5f} {P_an:12.3f} "
              f"{rel:10.2e} {n_it:>6}")


if __name__ == "__main__":
    main()
