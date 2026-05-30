"""Phase 41.6 -- Foundation-design walkthrough.

A 12-storey building (W = 24 MN total) on a 12 m x 8 m mat foundation
above a layered profile of medium-dense sand over soft clay with the
water table at -2 m. The site has design PGA = 0.35 g for M_w = 7.5
seismic demand (Zone IV).

The walkthrough exercises all four Phase 41 modules:

1. **Winkler mat** -- effective-stiffness check using the classic
   Hetenyi closed form for a strip footing of equivalent
   characteristic length.
2. **Pile group** -- 4 x 4 group of 600 mm-diameter bored piles
   below the mat; AASHTO LRFD p-multipliers + Converse-Labarre
   axial-efficiency.
3. **Liquefaction screening** -- Idriss-Boulanger 2014 at three
   depths in the loose-sand layer to flag the depth range that
   needs ground improvement.
4. **Dynamic SSI** -- the same mat treated as a rigid surface
   footing with frequency-dependent Gazetas impedance at the
   building's first-mode frequency (~0.8 Hz typical for a
   12-storey RC building).

Run::

    python examples/51_foundation_design_walkthrough.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver import (
    HalfspaceSoil,
    dimensionless_frequency,
    dynamic_footing_impedance,
    evaluate_liquefaction,
    gazetas_surface_footing,
    group_efficiency_converse_labarre,
    group_p_multipliers,
    hetenyi_characteristic_length,
    hetenyi_infinite_beam_point_load,
    p_multiplier,
    subgrade_modulus_table,
)


def line(width: int = 76) -> None:
    print("-" * width)


def main() -> None:
    print("=" * 78)
    print("Phase 41.6 -- Foundation design walkthrough")
    print("=" * 78)

    # ---- Inputs ---------------------------------------------------------
    W_building = 24.0e6                # N total building weight
    B_mat, L_mat = 8.0, 12.0            # m, mat plan dimensions
    A_mat = B_mat * L_mat
    h_mat = 1.5                          # m mat thickness
    E_concrete = 30.0e9                  # Pa
    G_concrete = E_concrete / (2.0 * 1.20)
    nu_concrete = 0.20

    # Site soil profile
    gamma_total = 19.0e3                 # N/m^3
    gamma_w = 9.81e3                     # water unit weight
    z_wt = 2.0                            # m depth to water table

    # Seismic / liquefaction inputs
    M_w = 7.5
    PGA_g = 0.35

    # Soil for SSI: stiff stratum 5 m below mat
    soil = HalfspaceSoil(G=80.0e6, nu=0.35, rho=1900.0)

    print(f"\nBuilding:  W = {W_building*1e-6:.1f} MN")
    print(f"Mat:       {B_mat} m x {L_mat} m  (A = {A_mat:.0f} m^2)")
    print(f"Bearing pressure  q = W/A = "
          f"{W_building/A_mat*1e-3:.0f} kPa")
    print(f"Site:      Zone IV, M_w = {M_w}, PGA = {PGA_g} g")

    # ---- (1) Winkler mat check ----------------------------------------
    print()
    line()
    print("(1) Winkler mat -- effective lateral stiffness")
    line()
    # Use medium-dense sand subgrade modulus as the bearing layer
    k_lo, k_hi = subgrade_modulus_table("medium_sand")
    k_s = 0.5 * (k_lo + k_hi)
    print(f"Subgrade modulus k_s = {k_s*1e-6:.0f} MN/m^3 "
          f"(medium sand: {k_lo*1e-6:.0f}-{k_hi*1e-6:.0f})")

    # Effective characteristic length of a strip of width = B_mat
    I_mat_per_m = h_mat ** 3 / 12.0       # strip flexural moment (m^4/m)
    L_c = hetenyi_characteristic_length(
        E=E_concrete, I=I_mat_per_m * B_mat, k_s=k_s, b=B_mat,
    )
    print(f"Hetenyi characteristic length L_c = {L_c:.2f} m  "
          f"(mat L = {L_mat} m, ratio L/L_c = {L_mat/L_c:.1f})")
    print(f"  Mat is " +
          ("RIGID" if L_mat / L_c < 1.0
           else "FLEXIBLE" if L_mat / L_c > 5.0
           else "INTERMEDIATE"))

    # If a point load of 4 MN (a column) acts on this mat:
    P_col = 4.0e6
    res_w = hetenyi_infinite_beam_point_load(
        P=P_col, E=E_concrete,
        I=I_mat_per_m * B_mat, k_s=k_s, b=B_mat,
    )
    print(f"\nUnder a P = {P_col*1e-3:.0f} kN column load (Hetenyi infinite-beam):")
    print(f"  Local settlement  w_max  = {res_w.w_max*1000:.2f} mm")
    print(f"  Local sagging M   M_max  = {res_w.M_max*1e-3:.1f} kN.m")

    # ---- (2) Pile group -------------------------------------------------
    print()
    line()
    print("(2) Pile group below the mat")
    line()
    n_rows, n_cols = 4, 4
    s_x, s_y = 1.8, 2.4
    D_pile = 0.6
    n_piles = n_rows * n_cols
    print(f"4 x 4 group of {D_pile*1000:.0f}-mm piles, "
          f"spacings s_x = {s_x}, s_y = {s_y} m")
    print(f"Spacing-to-diameter ratio s_x/D = {s_x/D_pile:.2f}")
    print()
    print("p-multipliers per column (lateral load in +x):")
    fm = group_p_multipliers(
        n_rows=n_rows, n_cols=n_cols, s_x=s_x, s_y=s_y, D=D_pile,
    )
    for col in range(n_cols):
        print(f"  Row {col+1}: {fm[col, 0]:.2f}  (s/D = {s_x/D_pile:.1f})")
    print()
    E_g = group_efficiency_converse_labarre(
        n_rows=n_rows, n_cols=n_cols, s_x=s_x, s_y=s_y, D=D_pile,
    )
    P_per_pile_static = W_building / n_piles
    print(f"Converse-Labarre group efficiency  E_g = {E_g:.3f}")
    print(f"Average axial demand per pile      = "
          f"{P_per_pile_static*1e-3:.0f} kN")
    # Required single-pile capacity: Q_single = P_per_pile / E_g (gross)
    Q_single_required = P_per_pile_static / E_g
    print(f"Required single-pile capacity      "
          f"(accounting for E_g): {Q_single_required*1e-3:.0f} kN")

    # ---- (3) Liquefaction screening ------------------------------------
    print()
    line()
    print("(3) Liquefaction screening (Idriss-Boulanger 2014)")
    line()
    print(f"Soil profile: {gamma_total*1e-3:.1f} kN/m^3 saturated above "
          f"water table at z={z_wt} m")
    print(f"  Loose-sand layer N_60 = 10-14 between z = 3 and 8 m")
    print()
    print(f"  {'z (m)':>8}{'sigma_v':>12}{'sigma_eff':>14}"
          f"{'N_60':>8}{'CSR':>10}{'CRR':>10}{'FS':>10}{'verdict':>14}")
    line(86)
    for z, N60 in [(3.0, 10.0), (5.0, 12.0), (8.0, 14.0)]:
        sv_tot = z * gamma_total
        sv_eff = z * gamma_total - max(0.0, z - z_wt) * gamma_w
        res_liq = evaluate_liquefaction(
            z=z, M=M_w, a_max_g=PGA_g,
            sigma_v_total=sv_tot, sigma_v_eff=sv_eff,
            N_60=N60, FC_percent=5.0,
        )
        verdict = "LIQUEFIES" if res_liq.liquefies else "stable"
        print(f"  {z:>8.1f}{sv_tot*1e-3:>12.0f}{sv_eff*1e-3:>14.0f}"
              f"{N60:>8.0f}{res_liq.CSR:>10.3f}{res_liq.CRR:>10.3f}"
              f"{res_liq.FS:>10.3f}{verdict:>14}")
    print()
    print("Mitigation: ground improvement (stone columns, jet-grouting)")
    print("           required for the loose-sand layer flagged above.")

    # ---- (4) Dynamic SSI ------------------------------------------------
    print()
    line()
    print("(4) Dynamic SSI -- frequency-dependent Gazetas impedance")
    line()
    imp_static = gazetas_surface_footing(soil, B=B_mat / 2, L=L_mat / 2)
    print(f"Soil halfspace: V_s = {soil.Vs:.0f} m/s, G = {soil.G*1e-6:.0f} MPa")
    print(f"Static K_z  = {imp_static.K_z*1e-6:.0f} MN/m")
    print(f"Static K_rx = {imp_static.K_rx*1e-9:.2f} GN.m/rad")
    print()
    print(f"  {'f (Hz)':>9}{'a_0':>8}{'K_z dyn (MN/m)':>18}"
          f"{'C_z (MN.s/m)':>16}{'K_rx dyn (GN.m/rad)':>22}")
    line(74)
    for f_Hz in [0.5, 1.0, 2.0, 5.0]:
        omega = 2.0 * math.pi * f_Hz
        a_0 = dimensionless_frequency(
            omega=omega, B=B_mat / 2, V_s=soil.Vs,
        )
        imp_dyn = dynamic_footing_impedance(
            static_impedance=imp_static, soil=soil, omega=omega,
        )
        print(f"  {f_Hz:>9.1f}{a_0:>8.3f}"
              f"{imp_dyn.K_z*1e-6:>18.0f}"
              f"{imp_dyn.C_z*1e-6:>16.2f}"
              f"{imp_dyn.K_rx*1e-9:>22.3f}")

    print()
    print("Use these K(omega), C(omega) at the building's modal frequency")
    print("for SSI-coupled response-spectrum or NLTHA analysis.")

    print()
    print("=" * 78)
    print("Theme F closed: Winkler beam + pile group + liquefaction +")
    print("                dynamic Gazetas impedance all operational.")
    print("=" * 78)


if __name__ == "__main__":
    main()
