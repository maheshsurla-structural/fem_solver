"""Phase 22.4 / 22.5 -- ply failure analysis of a CFRP laminate.

Two demonstrations:

1. **First-ply-failure envelope** -- sweep eps_xx for a [0/90]s
   cross-ply CFRP and find the load level at which the FIRST ply
   reaches Tsai-Wu FI = 1. Show that the matrix-dominated (90-deg)
   plies fail before the fiber-dominated (0-deg) plies.

2. **Failure criterion comparison** -- at a fixed in-plane biaxial +
   shear stress state, compare max-stress, Tsai-Hill, and Tsai-Wu
   failure indices. Demonstrates how interaction terms make the
   quadratic criteria more (or less) conservative depending on the
   stress quadrant.

Run::

    python examples/35_composite_failure.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver import (
    LayeredShellSection,
    OrthotropicLamina,
    PlyStrength,
    evaluate_laminate,
    max_stress_index,
    tsai_hill_index,
    tsai_wu_index,
    tsai_wu_strength_ratio,
)


def first_ply_failure_sweep() -> None:
    print("First-ply failure sweep -- [0/90/90/0]s CFRP cross-ply")
    print("=" * 60)
    ply = OrthotropicLamina(
        E1=181.0e9, E2=10.3e9, G12=7.17e9, nu12=0.28, rho=1600.0,
    )
    sec = LayeredShellSection.from_layers_centered([
        (ply, 0.25e-3, 0.0),
        (ply, 0.25e-3, 90.0),
        (ply, 0.25e-3, 90.0),
        (ply, 0.25e-3, 0.0),
    ])
    strength = PlyStrength(
        Xt=1500.0e6, Xc=1500.0e6, Yt=40.0e6, Yc=246.0e6, S=68.0e6,
    )

    print(f"  Laminate: {len(sec.layers)} plies, total thickness "
          f"{sec.thickness * 1e3:.2f} mm")
    print(f"  Strengths: Xt = {strength.Xt/1e6:.0f} MPa, "
          f"Yt = {strength.Yt/1e6:.0f} MPa, S = {strength.S/1e6:.0f} MPa")
    print()
    print(f"  {'eps_xx':>10} | {'max FI':>8} | {'fail theta':>10} | "
          f"{'FI(0)':>8} | {'FI(90)':>8}")
    print("  " + "-" * 58)
    for eps_xx in (1e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 7e-3, 1e-2):
        results = evaluate_laminate(
            sec, eps_membrane=(eps_xx, 0.0, 0.0), kappa=(0.0, 0.0, 0.0),
            strengths=strength, criterion="tsai_wu", z="mid",
        )
        fi_0 = max(r["FI"] for r in results if r["theta_deg"] == 0.0)
        fi_90 = max(r["FI"] for r in results if r["theta_deg"] == 90.0)
        max_rec = max(results, key=lambda r: r["FI"])
        print(f"  {eps_xx:>10.4f} | {max_rec['FI']:>8.3f} | "
              f"{max_rec['theta_deg']:>10.1f} | "
              f"{fi_0:>8.3f} | {fi_90:>8.3f}")
    print()
    print("Reading the result:")
    print("* For modest eps_xx (matrix-cracking regime), FI(90) > FI(0):")
    print("  the 90-deg plies (transverse-tension on matrix) fail first.")
    print("* FI scales quadratically with strain, so first-ply failure")
    print("  occurs around eps_xx ~ 0.004 - 0.005 here (FI = 1).")
    print("* Past first failure, post-failure analysis would degrade the")
    print("  90-deg plies' stiffness and continue (progressive damage).")


def failure_criterion_comparison() -> None:
    print()
    print("Failure criterion comparison at a fixed biaxial stress state")
    print("=" * 60)
    strength = PlyStrength(
        Xt=1500.0e6, Xc=1500.0e6, Yt=40.0e6, Yc=246.0e6, S=68.0e6,
    )

    cases = [
        ("pure sigma_11 = +Xt",      (1500.0e6, 0.0, 0.0)),
        ("pure sigma_11 = -Xc",      (-1500.0e6, 0.0, 0.0)),
        ("pure sigma_22 = +Yt",      (0.0, 40.0e6, 0.0)),
        ("pure sigma_22 = -Yc",      (0.0, -246.0e6, 0.0)),
        ("pure shear = +S",          (0.0, 0.0, 68.0e6)),
        ("biaxial tension (+,+)",    (750.0e6, 20.0e6, 0.0)),
        ("biaxial tension+shear",    (750.0e6, 20.0e6, 30.0e6)),
        ("longitudinal comp+trans tens", (-1000.0e6, 30.0e6, 0.0)),
    ]
    print(f"  {'stress state':<30} | {'maxstress':>9} | "
          f"{'TsaiHill':>9} | {'TsaiWu':>9} | {'SR':>7}")
    print("  " + "-" * 75)
    for label, sigma in cases:
        fi_max = max_stress_index(sigma, strength)
        fi_hill = tsai_hill_index(sigma, strength)
        fi_wu = tsai_wu_index(sigma, strength)
        sr = tsai_wu_strength_ratio(sigma, strength)
        sr_str = f"{sr:>7.3f}" if math.isfinite(sr) else "    inf"
        print(f"  {label:<30} | {fi_max:>9.3f} | "
              f"{fi_hill:>9.3f} | {fi_wu:>9.3f} | {sr_str}")
    print()
    print("Reading the result:")
    print("* All criteria agree on uniaxial failure (FI = 1 at strength).")
    print("* Tsai-Hill / Tsai-Wu pick up stress-interaction effects that")
    print("  max-stress misses (e.g. biaxial tension+shear FI > sum of")
    print("  pure-axis ratios).")
    print("* Tsai-Wu's strength ratio SR tells you how much the current")
    print("  stress can be scaled before reaching the failure envelope.")
    print("  SR > 1 means safe; SR = 1 means on the envelope; SR < 1")
    print("  means failed (current stress already past the envelope).")


def main() -> None:
    first_ply_failure_sweep()
    failure_criterion_comparison()


if __name__ == "__main__":
    main()
