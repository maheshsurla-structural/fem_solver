"""Theme Q capstone -- adaptive arc-length on the Mises truss.

A polished version of example 13 that exercises the Phase 50 extensions:

* **Adaptive arc-length stepping** -- delta_s auto-tuned by iteration
  count so the same script handles the stiff ascending branch and
  the gently-curving post-buckling branch.
* **Spherical variant** (``psi > 0``) -- shows that with a tuned psi
  the analysis still traces the curve cleanly.
* **Automatic limit-point detection** -- the integrator reports
  which steps crossed limit points (snap-through peak / trough).

The script prints a tabular load-deflection trace and the
auto-detected limit points, and saves a plot when matplotlib is
available.
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


def _build_truss(P_ref: float = 300.0):
    B = 10.0
    h = 1.0
    EA = 1.0e6
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
    m.fix(2, [1, 0, 1])
    m.add_nodal_load(2, [0.0, -P_ref, 0.0])
    return m, B, h, EA


def _analytical_load(w: float, B: float, h: float, EA: float) -> float:
    L0 = math.sqrt(B * B + h * h)
    L = math.sqrt(B * B + (h - w) ** 2)
    eps = (L - L0) / L0
    N = EA * eps
    return -2.0 * N * (h - w) / L


def main():
    print("=" * 72)
    print("Theme Q capstone -- adaptive arc-length on Mises truss")
    print("=" * 72)

    m, B, h, EA = _build_truss(P_ref=300.0)

    integrator = ArcLength(
        delta_s=0.05,
        psi=1e-3,              # mild spherical weighting
        adaptive=True,
        target_iterations=4,
        delta_s_min=0.01,
        delta_s_max=0.20,
    )
    res = NonlinearStaticAnalysis(
        m, num_steps=35, integrator=integrator,
        tol=1e-7, max_iter=30, track=(2, 1),
    ).run()

    lambdas = np.array(res["lambdas"])
    disps = np.array(res["tracked"])
    iter_counts = res["iter_counts"]
    forces = lambdas * 300.0
    apex_dn = -disps

    print()
    print(f"  Truss geometry:  B = {B} m, h = {h} m, EA = {EA:g} N")
    print(f"  Reference load:  P_ref = 300 N "
          f"(analytical limit ~ {2 * EA * h ** 2 / (B ** 2 + h ** 2):.2f} N)")
    print()
    print(f"  Initial delta_s = 0.05, adaptive range [0.01, 0.20]")
    print(f"  Final delta_s   = {integrator.delta_s:.4f}")
    print(f"  Limit points    = {integrator.limit_points}")
    print()
    print(f"  {'step':>4} {'iters':>6} {'w_dn (m)':>10}"
          f" {'P_FE (N)':>10} {'P_anal (N)':>10}")
    for i, (w, P, nit) in enumerate(zip(apex_dn, forces, iter_counts), 1):
        Pa = _analytical_load(w, B, h, EA)
        marker = ""
        if (i - 1) in integrator.limit_points:
            marker = "  <-- limit point"
        if i == 1 or i % 5 == 0 or i == len(forces):
            print(f"  {i:4d} {nit:6d} {w:10.4f} "
                  f"{P:10.3f} {Pa:10.3f}{marker}")
        elif marker:
            print(f"  {i:4d} {nit:6d} {w:10.4f} "
                  f"{P:10.3f} {Pa:10.3f}{marker}")

    # Optional plot
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(apex_dn, forces, "-o", markersize=4,
                 label="Arc-length FE", color="C0")
        # Analytical reference
        w_ref = np.linspace(0.0, max(apex_dn) + 0.05, 200)
        P_ref_curve = np.array(
            [_analytical_load(w, B, h, EA) for w in w_ref]
        )
        ax.plot(w_ref, P_ref_curve, "--", color="0.5",
                 label="Analytical equilibrium")
        for lp in integrator.limit_points:
            if lp < len(apex_dn):
                ax.axvline(apex_dn[lp], color="red", linestyle=":",
                            alpha=0.7,
                            label="Limit point" if lp == integrator.limit_points[0] else None)
        ax.set_xlabel("Apex downward displacement w (m)")
        ax.set_ylabel("Apex load P (N)")
        ax.set_title("Mises truss snap-through (adaptive arc-length)")
        ax.legend(loc="best", frameon=False)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = "mises_snapthrough_adaptive.png"
        fig.savefig(out, dpi=120)
        print()
        print(f"  Figure saved: {out}")
    except ImportError:
        print()
        print("  (matplotlib not available -- skipping figure)")

    print()
    print("Theme Q capstone DONE.")


if __name__ == "__main__":
    main()
