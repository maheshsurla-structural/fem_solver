"""Phase 44.6 -- Reliability capstone: steel column under uncertain
gravity + wind, compared across FORM, SORM, and three Monte-Carlo
methods.

A simply-supported steel column carries:

* dead load ``D ~ Normal``  (well-characterised mean, low CoV);
* live load ``L ~ Gumbel``  (extreme-value process);
* yield strength ``f_y ~ Lognormal`` (material variability).

The cross-section is a W14x90 (gross area = 17,200 mm^2). Failure
is reached when the axial demand exceeds the yield-defined capacity:

    g(D, L, f_y) = f_y · A_g - (D + L).

This is a **mildly nonlinear** limit-state function (product term
``f_y · A_g``), so SORM and FORM may differ slightly. The
capstone reports:

1. FORM β + design point + variable importances.
2. SORM β + curvatures.
3. Three Monte-Carlo estimates: crude, Latin hypercube, importance
   sampling around the FORM design point. Variance reduction is
   reported.
4. Sensitivity to dead-load CoV (a sweep).

Run::

    python examples/54_reliability_capstone.py
"""
from __future__ import annotations

import math
import time
from contextlib import contextmanager

import numpy as np
from scipy.stats import norm

from femsolver.reliability import (
    Gumbel,
    Lognormal,
    Normal,
    RandomVariableVector,
    crude_monte_carlo,
    form_hlrf,
    importance_sampling_around_u_star,
    latin_hypercube_monte_carlo,
    sorm_breitung,
)


@contextmanager
def timed(label: str, table: dict):
    t0 = time.perf_counter()
    yield
    table[label] = time.perf_counter() - t0


# ============================================================ problem

# Column cross-section
A_g = 0.01720      # 17,200 mm^2 -- W14x90 (m^2)

# Mean + CoV inputs
mu_D = 1.0e6         # 1000 kN dead-load mean
cov_D = 0.10
mu_L = 0.8e6         # 800 kN live-load mean
cov_L = 0.25
mu_fy = 345.0e6      # 345 MPa
cov_fy = 0.10

# Convert to distribution parameters
sigma_D = mu_D * cov_D                  # Normal CoV
sigma_L = mu_L * cov_L                  # used to fit Gumbel
# Gumbel: mu = mu_L - gamma_E * beta, sigma = beta * pi / sqrt(6)
beta_L = sigma_L * math.sqrt(6.0) / math.pi
mu_gumbel = mu_L - 0.5772156649 * beta_L
sigma_fy = mu_fy * cov_fy               # Lognormal CoV


def g(x):
    """Limit-state: capacity (f_y A_g) - demand (D + L)."""
    D, L, fy = x[0], x[1], x[2]
    return fy * A_g - (D + L)


def main() -> None:
    print("=" * 78)
    print("Phase 44.6 -- Reliability of a steel column under uncertain loads")
    print("=" * 78)

    print(f"\nLimit state:  g(D, L, f_y) = f_y * A_g - (D + L) "
          f"<= 0  =>  failure")
    print(f"Cross-section A_g = {A_g*1e4:.0f} cm^2  (W14x90 = "
          f"17.2 in^2 = 110 cm^2)")
    print()
    print(f"Random variables:")
    print(f"  D ~ Normal  (mu={mu_D*1e-3:.0f} kN, "
          f"CoV={cov_D*100:.0f}%)")
    print(f"  L ~ Gumbel  (mu={mu_L*1e-3:.0f} kN, "
          f"CoV={cov_L*100:.0f}%)")
    print(f"  f_y ~ Lognormal (mu={mu_fy*1e-6:.0f} MPa, "
          f"CoV={cov_fy*100:.0f}%)")

    rvs = RandomVariableVector([
        Normal(mu_D, sigma_D),
        Gumbel(mu_gumbel, beta_L),
        Lognormal(mu_fy, sigma_fy),
    ])

    times: dict[str, float] = {}

    # ---- (1) FORM ------------------------------------------------------
    print()
    print("-" * 70)
    print("(1) FORM (Hasofer-Lind-Rackwitz-Fiessler)")
    print("-" * 70)
    with timed("FORM", times):
        res_F = form_hlrf(g=g, rvs=rvs, tol_g=1.0e-3, tol_u=1.0e-6)
    print(f"  Reliability index   beta  = {res_F.beta:.4f}")
    print(f"  Failure probability P_f   = {res_F.pf:.3e}")
    print(f"  Iterations                = {res_F.n_iter}, "
          f"converged = {res_F.converged}")
    print(f"  Design point (X-space):")
    print(f"    D*   = {res_F.x_star[0]*1e-3:.1f} kN  "
          f"(mean = {mu_D*1e-3:.0f}, lifted to "
          f"{res_F.x_star[0]/mu_D*100:.0f}% of mean)")
    print(f"    L*   = {res_F.x_star[1]*1e-3:.1f} kN")
    print(f"    f_y* = {res_F.x_star[2]*1e-6:.1f} MPa  "
          f"(mean = {mu_fy*1e-6:.0f}, drawn down to "
          f"{res_F.x_star[2]/mu_fy*100:.0f}% of mean)")
    print(f"  Direction cosines (importance factors):")
    labels = ["D", "L", "f_y"]
    for lab, a in zip(labels, res_F.alpha):
        print(f"    alpha_{lab:<3} = {a:>+7.3f}  "
              f"(squared {a**2*100:>5.1f}%)")

    # ---- (2) SORM ------------------------------------------------------
    print()
    print("-" * 70)
    print("(2) SORM (Breitung curvature correction)")
    print("-" * 70)
    with timed("SORM", times):
        res_S = sorm_breitung(form_result=res_F, g=g, rvs=rvs)
    print(f"  beta_SORM = {res_S.beta_SORM:.4f}  "
          f"(vs FORM {res_F.beta:.4f})")
    print(f"  P_f_SORM  = {res_S.pf_SORM:.3e}  "
          f"(vs FORM {res_F.pf:.3e})")
    print(f"  Principal curvatures kappa = {res_S.kappa}")

    # ---- (3) Monte Carlo with three methods ---------------------------
    print()
    print("-" * 70)
    print("(3) Monte-Carlo estimators")
    print("-" * 70)
    with timed("crude MC (100k)", times):
        res_MC = crude_monte_carlo(
            g=g, rvs=rvs, n_samples=100000, seed=42,
        )
    print(f"  Crude MC (100k samples):     P_f = {res_MC.pf_estimate:.3e}  "
          f"(CI 95% [{res_MC.pf_ci95_low:.2e}, "
          f"{res_MC.pf_ci95_high:.2e}], "
          f"{res_MC.n_failures} failures)")
    with timed("LHS (100k)", times):
        res_LHS = latin_hypercube_monte_carlo(
            g=g, rvs=rvs, n_samples=100000, seed=42,
        )
    print(f"  LHS    (100k samples):       P_f = {res_LHS.pf_estimate:.3e}  "
          f"({res_LHS.n_failures} failures)")
    with timed("Imp. sampling (2k)", times):
        res_IS = importance_sampling_around_u_star(
            g=g, rvs=rvs, u_star=res_F.u_star,
            n_samples=2000, seed=42,
        )
    print(f"  Imp.Sampling (2k samples):   P_f = {res_IS.pf_estimate:.3e}  "
          f"(CI 95% [{res_IS.pf_ci95_low:.2e}, "
          f"{res_IS.pf_ci95_high:.2e}])")
    print()
    var_ratio = ((res_MC.pf_std_error ** 2 * res_MC.n_samples)
                 / max(res_IS.pf_std_error ** 2 * res_IS.n_samples,
                       1.0e-30))
    print(f"  Importance-sampling variance reduction vs crude MC: "
          f"{var_ratio:.0f}x")

    # ---- (4) Sensitivity to dead-load CoV ------------------------------
    print()
    print("-" * 70)
    print("(4) Sensitivity of beta to dead-load CoV")
    print("-" * 70)
    print(f"  {'CoV_D':>8}{'beta':>10}{'P_f':>14}")
    print("  " + "-" * 32)
    for cov in [0.05, 0.10, 0.15, 0.20, 0.30]:
        sig = mu_D * cov
        rvs_local = RandomVariableVector([
            Normal(mu_D, sig),
            Gumbel(mu_gumbel, beta_L),
            Lognormal(mu_fy, sigma_fy),
        ])
        res_local = form_hlrf(g=g, rvs=rvs_local, tol_g=1.0e-3)
        print(f"  {cov*100:>7.1f}%{res_local.beta:>10.3f}"
              f"{res_local.pf:>14.3e}")

    # ---- Wall clocks ---------------------------------------------------
    print()
    print("-" * 70)
    print("Wall-clock summary:")
    for lab, t in times.items():
        print(f"  {lab:<30}: {t*1000:>7.1f} ms")

    print()
    print("=" * 78)
    print("Theme D closed: random variables + FORM + SORM + Monte Carlo all")
    print("                operational.")
    print("=" * 78)


if __name__ == "__main__":
    main()
