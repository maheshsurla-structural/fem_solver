"""Phase 36.6 -- RC + steel frame design to Indian codes.

End-to-end design walkthrough of a 4-storey RC moment frame in
Zone IV, including:

1. **IS 1893 seismic demand** -- empirical period, design spectrum,
   base shear, vertical distribution, storey-drift check.
2. **IS 456 member design** -- typical beam (flexure + shear),
   typical column (P-M interaction).
3. **IS 13920 capacity design** -- SCWB joint check, beam capacity
   shear, column capacity shear, plastic-hinge confinement.
4. **IS 800 alternate steel design** -- the same beam re-checked as
   an I-section for comparison.

Run::

    python examples/47_indian_codes_frame_design.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver.design import is456, is800, is1893, is13920


# ============================================================ building data

N_STOREYS = 4
H_STOREY = 3.0                    # m
H_TOTAL = N_STOREYS * H_STOREY
ZONE = 4                          # Z = 0.24
IMPORTANCE = 1.0
R = 5.0                           # RC SMRF
SOIL = 2                          # medium
W_FLOOR = 500.0e3                 # N per floor (DL + 25% LL)
W_TOTAL = W_FLOOR * N_STOREYS


# ============================================================ helpers

def line(width=72):
    print("-" * width)


def main() -> None:
    print("=" * 80)
    print("Phase 36.6 -- 4-storey RC frame to Indian codes (IS 456 / "
          "1893 / 13920)")
    print("=" * 80)

    # ---- IS 1893 seismic ---------------------------------------------
    print(f"\nBuilding: {N_STOREYS} storeys x {H_STOREY} m = {H_TOTAL} m, "
          f"W = {W_TOTAL*1e-3:.0f} kN")
    print(f"Site: Zone {ZONE} (Z = {is1893.zone_factor(ZONE)}), "
          f"I = {IMPORTANCE}, R = {R} (RC SMRF), Soil = type {SOIL}")
    line()

    T = is1893.empirical_period(h=H_TOTAL, system="RC_MRF")
    print(f"Empirical period   T = 0.075 h^0.75 = {T:.3f} s")
    sa_g = is1893.design_spectrum_Sa_g(T, soil_type=SOIL)
    print(f"Spectral coefficient Sa/g(T) = {sa_g:.3f}")
    res_bs = is1893.is1893_base_shear(
        T=T, W=W_TOTAL, zone=ZONE, importance=IMPORTANCE,
        R=R, soil_type=SOIL,
    )
    print(f"Design coefficient A_h = (Z/2)(I/R)(Sa/g) = {res_bs.A_h:.4f}")
    print(f"Base shear V_B = A_h W = {res_bs.V_B*1e-3:.1f} kN")

    # Vertical distribution
    W_s = np.full(N_STOREYS, W_FLOOR)
    h_s = np.arange(1, N_STOREYS + 1) * H_STOREY
    Q = is1893.vertical_force_distribution(
        V_B=res_bs.V_B, storey_weights=W_s, storey_heights=h_s,
    )
    print(f"\nFloor-level lateral forces Q_i (low to high):")
    for i, (h, Qi) in enumerate(zip(h_s, Q)):
        print(f"  Floor {i+1} (h = {h:.1f} m): Q_{i+1} = "
              f"{Qi*1e-3:.2f} kN")

    # Hypothetical elastic floor displacements (from analysis)
    u_elastic = np.array([0.0010, 0.0022, 0.0032, 0.0040])
    drift = is1893.is1893_drift_check(
        floor_disp=u_elastic,
        storey_heights=np.full(N_STOREYS, H_STOREY),
        R=R,
    )
    print(f"\nDrift check (hypothetical elastic disp, R-amplified):")
    print(f"  Max ratio = {drift.max_ratio*100:.3f} % "
          f"(limit {drift.limit*100:.1f} %)")
    print(f"  Status: {'PASS' if drift.passes else 'FAIL'}")

    # ---- IS 456 typical beam (300 x 500) -----------------------------
    print()
    line()
    print("IS 456 RC BEAM DESIGN: 300 x 500 mm, M25 / Fe415")
    line()
    M_u_beam = 150.0e3        # demand from analysis
    V_u_beam = 120.0e3
    b_beam, d_beam = 0.300, 0.460
    flex = is456.is456_beam_flexure(
        M_u=M_u_beam,
        f_ck=is456.fck_M(25), f_y=is456.fy_Fe(415),
        b=b_beam, d=d_beam,
    )
    print(f"M_u = {M_u_beam*1e-3:.0f} kN.m -> A_st = "
          f"{flex.A_st*1e6:.0f} mm^2 ({flex.note})")
    print(f"  x_u / d = {flex.x_u_over_d:.3f}  "
          f"(limit {is456.xu_max_over_d(is456.fy_Fe(415))})")

    shear = is456.is456_beam_shear(
        V_u=V_u_beam,
        f_ck=is456.fck_M(25), f_y_sv=is456.fy_Fe(415),
        b=b_beam, d=d_beam, A_st=flex.A_st,
    )
    print(f"V_u = {V_u_beam*1e-3:.0f} kN -> {shear.note}")
    if shear.requires_stirrups and not math.isinf(shear.V_us_required):
        # 2-leg 8 mm stirrup A_sv = 2 * pi/4 * 8^2 = 100.5 mm^2
        A_sv = 2.0 * math.pi / 4.0 * (0.008) ** 2
        s_max = A_sv / shear.A_sv_over_s_required
        print(f"  2-leg 8 mm stirrups: A_sv = "
              f"{A_sv*1e6:.1f} mm^2, s_max = {s_max*1000:.0f} mm")

    # ---- IS 456 typical column (300 x 500) ---------------------------
    print()
    line()
    print("IS 456 RC COLUMN DESIGN: 300 x 500 mm, M25 / Fe415")
    line()
    P_u_col = 1200.0e3        # kN -> N
    M_u_col = 200.0e3
    A_st_col = 1885e-6         # 6-Y20 ~ 1885 mm^2 (~ 1.26 % of A_g)
    pts = is456.is456_column_pm_curve(
        f_ck=is456.fck_M(25), f_y=is456.fy_Fe(415),
        b=0.300, D=0.500, A_st_total=A_st_col, n_layers=3,
    )
    passes, util = is456.is456_column_pm_check(
        P_u=P_u_col, M_u=M_u_col, points=pts,
    )
    print(f"P_u = {P_u_col*1e-3:.0f} kN, M_u = {M_u_col*1e-3:.0f} kN.m")
    print(f"A_st = {A_st_col*1e6:.0f} mm^2 ({100*A_st_col/(0.300*0.500):.2f}% of A_g)")
    print(f"P-M interaction: utilisation = {util:.3f} -> "
          f"{'PASS' if passes else 'FAIL'}")

    # ---- IS 13920 capacity design ------------------------------------
    print()
    line()
    print("IS 13920 CAPACITY DESIGN")
    line()
    # Joint SCWB: assume each column at the joint has M_n = 250 kNm (top
    # and bottom), each beam has M_n = 150 kNm (left and right)
    sum_Mc = 2 * 250.0e3
    sum_Mb = 2 * 150.0e3
    scwb = is13920.is13920_scwb_check(sum_Mc=sum_Mc, sum_Mb=sum_Mb)
    print(f"SCWB at interior joint:  sum_Mc = {sum_Mc*1e-3:.0f} kN.m, "
          f"sum_Mb = {sum_Mb*1e-3:.0f} kN.m")
    print(f"  Ratio = {scwb.ratio:.3f} (limit {scwb.limit}) "
          f"-> {'PASS' if scwb.passes else 'FAIL'}")

    # Beam capacity shear: clear span 4.5 m, gravity V = 50 kN
    cap_v_b = is13920.is13920_capacity_shear_beam(
        M_n_pos_left=120e3, M_n_neg_left=180e3,
        M_n_pos_right=120e3, M_n_neg_right=180e3,
        L_n=4.5, V_gravity=50e3,
    )
    print(f"\nBeam capacity shear (clear span 4.5 m):")
    print(f"  V_p = {cap_v_b.V_p*1e-3:.1f} kN, "
          f"V_design = {cap_v_b.V_design*1e-3:.1f} kN")

    # Column capacity shear: clear height 2.5 m, analysis V = 80 kN
    cap_v_c = is13920.is13920_capacity_shear_column(
        M_n_top=250e3, M_n_bot=250e3,
        h_clear=2.5, V_analysis=80e3,
    )
    print(f"\nColumn capacity shear (clear height 2.5 m):")
    print(f"  V_p = {cap_v_c.V_p*1e-3:.1f} kN, "
          f"V_design = {cap_v_c.V_design*1e-3:.1f} kN")

    # Confinement at column plastic hinge
    conf = is13920.is13920_confinement(
        A_g=0.300 * 0.500,
        A_k=(0.300 - 2 * 0.040) * (0.500 - 2 * 0.040),
        f_ck=is456.fck_M(25), f_yh=is456.fy_Fe(415),
        h_clear=2.5, D=0.500,
    )
    print(f"\nColumn confinement (plastic-hinge zone):")
    print(f"  rho_st (required) = {conf.rho_st_required*100:.3f} %")
    print(f"  Hoop spacing s_max = {conf.s_max_required*1000:.0f} mm")
    for n in conf.notes:
        print(f"  - {n}")

    # ---- IS 800 alternate: steel beam --------------------------------
    print()
    line()
    print("IS 800 ALTERNATE: same beam as steel section (Z_p needed)")
    line()
    # ISMB 450: Z_p = 1408 cm^3, A = 92.27 cm^2, h_w*t_w = 9.4*450 = 4230 mm^2
    res_st_flex = is800.is800_flexure(
        M_u=M_u_beam, Z_p=1408e-6, f_y=250e6,
    )
    res_st_shr = is800.is800_shear(V_u=V_u_beam, A_v=4230e-6, f_yw=250e6)
    print(f"ISMB 450 (Z_p = 1408 cm^3): M_d = {res_st_flex.M_d*1e-3:.1f} kN.m "
          f"-> util {res_st_flex.utilisation:.3f}")
    print(f"  Shear: V_d = {res_st_shr.V_d*1e-3:.1f} kN -> "
          f"util {res_st_shr.utilisation:.3f}")

    print()
    print("=" * 80)
    print("Theme G2 closed: Indian codes (IS 456 / 800 / 1893 / 13920) "
          "operational.")
    print("=" * 80)


if __name__ == "__main__":
    main()
