"""Theme R capstone -- soil and concrete constitutive models compared.

Three single-material-point ("driver") tests that exercise the
new constitutive models against canonical stress-strain envelopes:

1. **Drained triaxial compression** on three soils:

   * :class:`DruckerPrager3D` (existing) -- pressure-dependent, smooth
   * :class:`MohrCoulomb3D` (new) -- pressure-dependent, hexagonal
   * :class:`ModifiedCamClay3D` (new) -- critical-state hardening

   under increasing axial compression with constant lateral
   confinement, plotted as ``q`` vs ``eps_axial``.

2. **Uniaxial cyclic concrete** with the new
   :class:`ConcreteDamage3D` showing the canonical tension-cracking
   and compression-crushing envelopes plus stiffness degradation on
   reversal.

A single matplotlib figure is saved when available.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from femsolver import (
    ConcreteDamage3D,
    DruckerPrager3D,
    MohrCoulomb3D,
    ModifiedCamClay3D,
)


# ============================================================ helpers

def _triaxial_drive(material, *, sigma_3: float, n_steps: int = 50,
                     eps_max: float = -0.02):
    """Run an axisymmetric triaxial compression: apply axial strain
    ``eps_axial`` from 0 to ``eps_max`` while keeping lateral
    confinement constant at ``sigma_3``. We use *strain control* with
    Poisson coupling tweaked so that lateral stress stays close to
    ``sigma_3``."""
    out_eps_ax = []
    out_q = []
    out_p_eff = []
    # Initial confining state: apply hydrostatic eps to reach sigma_3
    # in all three directions.
    # eps_0 = sigma_3 / E_bulk -> approximate via Poisson coupling.
    nu = material.nu
    E = material.E
    K = E / (3.0 * (1.0 - 2.0 * nu))
    eps_init = sigma_3 / (3.0 * K)        # hydrostatic
    eps = np.array([eps_init, eps_init, eps_init, 0.0, 0.0, 0.0])
    sigma_now, _ = material.get_response(eps)
    material.commit_state()
    eps_axial = eps_init
    for i in range(1, n_steps + 1):
        eps_axial_target = eps_init + (eps_max - eps_init) * i / n_steps
        eps[2] = eps_axial_target
        # Iterative correction on lateral strains to keep sigma_xx,
        # sigma_yy = sigma_3 (drained triaxial). 6 sub-iterations
        # is plenty for the smooth materials.
        eps_lat = eps_init
        for _ in range(10):
            eps[0] = eps_lat
            eps[1] = eps_lat
            sigma, _ = material.get_response(eps)
            err_x = sigma[0] - sigma_3
            if abs(err_x) < 10.0:          # within 10 Pa
                break
            # Tangent correction (rough secant): change in sigma_x per
            # change in eps_lat dominated by (lambda + 2G) -> use D[0,0]
            eps_lat -= err_x / (E * (1.0 - nu) /
                                  ((1.0 + nu) * (1.0 - 2.0 * nu)))
        sigma, _ = material.get_response(eps)
        material.commit_state()
        p_v = (sigma[0] + sigma[1] + sigma[2]) / 3.0
        s = sigma.copy()
        s[0] -= p_v; s[1] -= p_v; s[2] -= p_v
        q = math.sqrt(1.5 * (
            s[0] ** 2 + s[1] ** 2 + s[2] ** 2
            + 2.0 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2)
        ))
        out_eps_ax.append(eps_axial_target)
        out_q.append(q)
        out_p_eff.append(-p_v)
    return (
        np.array(out_eps_ax),
        np.array(out_q),
        np.array(out_p_eff),
    )


def _uniaxial_concrete_drive(material, eps_history):
    """Drive a single concrete point through a sequence of axial strains
    in 1-direction with Poisson coupling. Returns sigma_axial history."""
    nu = material.nu
    sigma_hist = []
    for eps_ax in eps_history:
        eps = np.array([eps_ax, -nu * eps_ax, -nu * eps_ax, 0, 0, 0])
        sigma, _ = material.get_response(eps)
        material.commit_state()
        sigma_hist.append(sigma[0])
    return np.array(sigma_hist)


# ============================================================ main

def main():
    print("=" * 72)
    print("Theme R capstone -- soil + concrete constitutive models")
    print("=" * 72)
    print()

    # ============================ triaxial: three soils, sigma_3 = -100 kPa
    sigma_3 = -100e3       # 100 kPa confining (tension positive sign)

    dp = DruckerPrager3D.from_mohr_coulomb(
        E=30e6, nu=0.30, cohesion=10e3, phi_deg=30.0,
    )
    mc = MohrCoulomb3D(E=30e6, nu=0.30, cohesion=10e3, phi_deg=30.0)
    mcc = ModifiedCamClay3D(
        E=30e6, nu=0.30, M=1.0,
        lambda_cc=0.20, kappa_cc=0.05, p_c0=200e3,
    )

    eps_dp, q_dp, p_dp = _triaxial_drive(dp, sigma_3=sigma_3)
    eps_mc, q_mc, p_mc = _triaxial_drive(mc, sigma_3=sigma_3)
    eps_mcc, q_mcc, p_mcc = _triaxial_drive(mcc, sigma_3=sigma_3)

    print("Drained triaxial compression at sigma_3 = -100 kPa:")
    print(f"  {'eps_ax':>9} | {'DP q':>9} | {'MC q':>9} | {'MCC q':>9}  "
          f"(all kPa)")
    for i in (0, 9, 19, 29, 39, 49):
        if i < len(eps_dp):
            print(f"  {eps_dp[i]:+9.4f} | "
                  f"{q_dp[i]/1e3:9.2f} | "
                  f"{q_mc[i]/1e3:9.2f} | "
                  f"{q_mcc[i]/1e3:9.2f}")
    print()
    print(f"  DP   final q = {q_dp[-1]/1e3:.2f} kPa")
    print(f"  MC   final q = {q_mc[-1]/1e3:.2f} kPa")
    print(f"  MCC  final q = {q_mcc[-1]/1e3:.2f} kPa  "
          f"(p_c grew {200:.0f} -> {mcc.p_c*1e-3:.0f} kPa)")

    # ============================ uniaxial concrete envelopes
    print()
    print("Uniaxial concrete (C30, monotonic):")
    conc = ConcreteDamage3D(
        E=30e9, nu=0.20, eps_t0=1.0e-4, eps_c0=1.0e-3,
        A_t=1.0, B_t=1.0e4, A_c=1.0, B_c=300.0,
    )
    eps_t = np.linspace(0, 5e-4, 30)
    sig_t = _uniaxial_concrete_drive(conc, eps_t)

    conc2 = ConcreteDamage3D(
        E=30e9, nu=0.20, eps_t0=1.0e-4, eps_c0=1.0e-3,
        A_t=1.0, B_t=1.0e4, A_c=1.0, B_c=300.0,
    )
    eps_c = np.linspace(0, -5e-3, 40)
    sig_c = _uniaxial_concrete_drive(conc2, eps_c)
    print(f"  peak tensile stress     = {sig_t.max()/1e6:.2f} MPa "
          f"(expected ~ {30e3 * 1e-4 / 1e6:.2f} MPa)")
    print(f"  peak compressive stress = {-sig_c.min()/1e6:.2f} MPa")

    # ============================ optional plot
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_t = axes[0]
        ax_t.plot(-eps_dp, q_dp / 1e3, label="Drucker-Prager", lw=2)
        ax_t.plot(-eps_mc, q_mc / 1e3, label="Mohr-Coulomb", lw=2, ls="--")
        ax_t.plot(-eps_mcc, q_mcc / 1e3, label="Modified Cam-Clay",
                   lw=2, ls=":")
        ax_t.set_xlabel(r"$-\varepsilon_\mathrm{axial}$")
        ax_t.set_ylabel(r"$q$ (kPa)")
        ax_t.set_title(r"Drained triaxial @ $\sigma_3 = -100$ kPa")
        ax_t.legend(loc="lower right")
        ax_t.grid(True, alpha=0.3)
        ax_c = axes[1]
        ax_c.plot(eps_t, sig_t / 1e6, label="Tension", color="C3", lw=2)
        ax_c.plot(eps_c, sig_c / 1e6, label="Compression",
                   color="C0", lw=2)
        ax_c.set_xlabel(r"$\varepsilon_\mathrm{axial}$")
        ax_c.set_ylabel(r"$\sigma$ (MPa)")
        ax_c.set_title("Concrete damage (Mazars)")
        ax_c.legend(loc="best")
        ax_c.grid(True, alpha=0.3)
        ax_c.axhline(0, color="0.5", lw=0.5)
        ax_c.axvline(0, color="0.5", lw=0.5)
        fig.tight_layout()
        out = "theme_r_constitutive.png"
        fig.savefig(out, dpi=120)
        print()
        print(f"  Figure saved: {out}")
    except ImportError:
        print("  (matplotlib not available -- skipping plot)")

    print()
    print("Theme R capstone DONE.")


if __name__ == "__main__":
    sys.exit(main())
