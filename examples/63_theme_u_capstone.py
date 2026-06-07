"""Theme U capstone -- site-specific seismic hazard for a soft-soil site.

A complete PBSE workflow:

1. Define a regional seismic source (one Gutenberg-Richter point
   source 15 km away).
2. Compute a PGA hazard curve and a UHS at 5 periods.
3. Deaggregate the 475-yr PGA to identify the dominant (M, R, eps)
   scenario.
4. Apply 1-D linear site response to convert a rock UHS to a
   soft-soil-surface UHS via the transfer function.
5. Compute the risk-targeted MCE_R (ASCE 7 Ch 21) for two structures
   of different fragility.

Outputs a numerical summary + a 2-panel matplotlib figure showing
the hazard curve and the rock vs surface UHS comparison.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from femsolver.seismic import (
    BooreAtkinsonLike,
    GutenbergRichterMFD,
    PointSource,
    SoilLayer,
    SoilProfile,
    annual_collapse_rate,
    compute_hazard_curve,
    compute_uhs,
    deaggregate,
    risk_targeted_im,
    site_amplification_spectrum,
)


def main():
    print("=" * 76)
    print("Theme U capstone -- site-specific seismic for a soft-soil site")
    print("=" * 76)

    # ============================ regional seismicity =========================
    src = PointSource(
        name="Fault A",
        R_jb_km=15.0,
        mfd=GutenbergRichterMFD(a=4.2, b=0.9, M_min=5.0, M_max=7.5),
    )
    nu_min = src.mfd.nu_M_min
    print()
    print(f"  Source 'Fault A' at R_jb = 15 km")
    print(f"  Gutenberg-Richter (a, b) = (4.2, 0.9), M in [5.0, 7.5]")
    print(f"  Mean annual rate (M >= 5.0): {nu_min:.4f}/yr  "
          f"({1/nu_min:.0f} yr return)")

    # ============================ PGA hazard curve =========================
    from femsolver.seismic import bssa14
    gmpe_pga = bssa14(0.01)   # BSSA14 PGA period
    ims = np.geomspace(0.001, 5.0, 60)
    curve_pga_rock = compute_hazard_curve(
        gmpe=gmpe_pga, sources=[src], im_levels=ims, V_s30=760.0,
    )
    pga_475 = curve_pga_rock.im_at_return_period(475)
    pga_2475 = curve_pga_rock.im_at_return_period(2475)
    print()
    print(f"  PGA hazard curve (rock V_s30 = 760 m/s):")
    print(f"    PGA @ 475 yr  = {pga_475:.3f} g")
    print(f"    PGA @ 2475 yr = {pga_2475:.3f} g")

    # ============================ deaggregation =========================
    d = deaggregate(
        gmpe=gmpe_pga, sources=[src], im_target=pga_475,
        R_edges=np.arange(0.0, 100.0, 10.0),
    )
    print(f"  Deaggregation @ PGA_475:")
    print(f"    modal (M, R, eps) = ({d.modal_M:.2f}, "
          f"{d.modal_R:.0f} km, {d.modal_eps:+.2f})")
    print(f"    mean  (M, R, eps) = ({d.mean_M:.2f}, "
          f"{d.mean_R:.1f} km, {d.mean_eps:+.2f})")

    # ============================ UHS =========================
    # Use BSSA14 period-by-period coefficients for a properly-shaped UHS
    from femsolver.seismic import bssa14
    periods = [0.01, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0]
    gmpes = {T: bssa14(T) for T in periods}
    uhs_rock = compute_uhs(
        gmpes_by_period=gmpes, sources=[src],
        return_period=475, im_levels=ims, V_s30=760.0,
    )
    print()
    print(f"  Uniform Hazard Spectrum @ 475 yr (rock):")
    print(f"    {'T (s)':>6}  {'Sa (g)':>8}")
    for T, Sa in zip(uhs_rock.periods, uhs_rock.sa_values):
        print(f"    {T:>6.2f}  {Sa:>8.3f}")

    # ============================ soft-soil amplification =========================
    # Soft-soil profile: 20 m deep, Vs_avg = 180 m/s -> Class E site
    soil = SoilProfile(
        layers=[
            SoilLayer(thickness=10.0, Vs=150.0, rho=1900, damping=0.05),
            SoilLayer(thickness=10.0, Vs=220.0, rho=2000, damping=0.05),
        ],
        rock_Vs=760.0, rock_rho=2300,
    )
    freqs = np.linspace(0.1, 15.0, 400)
    amp = site_amplification_spectrum(soil, freqs)
    f_peak = freqs[np.argmax(amp)]
    print()
    print(f"  Soft-soil site response (20 m, Vs_avg ~ 185 m/s):")
    print(f"    Peak amplification = {amp.max():.2f} at f = {f_peak:.2f} Hz "
          f"(T = {1/f_peak:.2f} s)")

    # Apply amplification at each spectrum period using the soil
    # transfer function at that frequency.
    # For each period T, compute amplification at f = 1/T.
    Sa_surface = []
    for T, Sa_rock in zip(uhs_rock.periods, uhs_rock.sa_values):
        if T == 0.0:
            # PGA amplification at site fundamental frequency
            # (approximation: use the broadband amp at peak)
            amp_T = amp.max() / 2  # half-peak for PGA (broadband)
        else:
            f_target = 1.0 / T
            i = int(np.argmin(np.abs(freqs - f_target)))
            amp_T = amp[i]
        Sa_surface.append(Sa_rock * amp_T)
    Sa_surface = np.array(Sa_surface)

    print()
    print(f"  Surface UHS (rock UHS x site amplification):")
    print(f"    {'T (s)':>6}  {'Sa_rock':>8}  {'AF':>5}  {'Sa_surf':>8}")
    for i, T in enumerate(uhs_rock.periods):
        if T == 0.0:
            af = amp.max() / 2
        else:
            af = amp[int(np.argmin(np.abs(freqs - 1.0 / T)))]
        print(f"    {T:>6.2f}  {uhs_rock.sa_values[i]:>8.3f}  "
              f"{af:>5.2f}  {Sa_surface[i]:>8.3f}")

    # ============================ MCE_R =========================
    print()
    print(f"  Risk-targeted MCE_R (ASCE 7-22 Ch 21, beta = 0.6):")
    for beta in (0.4, 0.6, 0.8):
        mce = risk_targeted_im(curve_pga_rock, target_collapse_prob=0.01,
                                window_years=50, beta=beta)
        lc = annual_collapse_rate(curve_pga_rock, theta=mce, beta=beta)
        Pc = 1.0 - math.exp(-lc * 50)
        print(f"    beta = {beta}: MCE_R = {mce:.3f} g  "
              f"(verify: P_C(50yr) = {Pc*100:.3f}%)")

    # ============================ plot =========================
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax_h = axes[0]
        ax_h.loglog(curve_pga_rock.im_levels, curve_pga_rock.annual_rates, "-")
        ax_h.axhline(1 / 475, ls="--", color="C1", label="475-yr ($\\lambda$ = 1/475)")
        ax_h.axhline(1 / 2475, ls="--", color="C3", label="2475-yr ($\\lambda$ = 1/2475)")
        ax_h.set_xlabel("PGA (g)")
        ax_h.set_ylabel(r"Annual exceedance rate $\lambda(PGA > x)$")
        ax_h.set_title("Hazard curve (rock)")
        ax_h.legend()
        ax_h.grid(True, which="both", alpha=0.3)

        ax_u = axes[1]
        ax_u.plot(uhs_rock.periods, uhs_rock.sa_values, "-o",
                   label="Rock UHS")
        ax_u.plot(uhs_rock.periods, Sa_surface, "-s",
                   label="Surface UHS (soft soil)")
        ax_u.set_xlabel("Period T (s)")
        ax_u.set_ylabel("Sa @ 475 yr (g)")
        ax_u.set_title("Uniform Hazard Spectrum")
        ax_u.legend()
        ax_u.grid(True, alpha=0.3)
        fig.tight_layout()
        out = "site_specific_uhs.png"
        fig.savefig(out, dpi=120)
        print()
        print(f"  Figure saved: {out}")
    except ImportError:
        print()
        print("  (matplotlib not installed -- skipping plot)")

    print()
    print("Theme U capstone DONE.")


if __name__ == "__main__":
    sys.exit(main())
