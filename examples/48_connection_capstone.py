"""Phase 37.6 -- Connection-modelling capstone.

A pre-Northridge vs. RBS-detailed steel moment-frame joint, plus a
side-by-side moment-rotation comparison of three PR connection
presets and the corresponding bolt + weld design check.

Demonstrates:

1. **Krawinkler panel-zone** sizing for a real W14x90 / W24x84 joint.
2. **AISC 358 RBS** dimensions and the resulting reduction in plastic
   moment at the column face (the "dog-bone" protection mechanism).
3. **Richard-Abbott PR connections** plotted on a single M-theta
   table -- showing why "rigid" vs "semi-rigid" matter.
4. **AISC + IS 800 bolt/weld design check** for the same fastener
   geometry under the two codes.

Run::

    python examples/48_connection_capstone.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver.design.connections import (
    Pr_preset,
    aisc358_recommended_RBS,
    block_shear_aisc,
    bolt_bearing_aisc,
    bolt_bearing_is800,
    bolt_shear_aisc,
    bolt_shear_is800,
    fillet_weld_aisc,
    fillet_weld_is800,
    krawinkler_panel_zone,
)


def line(width: int = 72) -> None:
    print("-" * width)


def main() -> None:
    print("=" * 80)
    print("Phase 37.6 -- Steel/RC connection capstone")
    print("=" * 80)

    # ---- Section data: W14x90 column, W24x84 beam (real AISC W shapes)
    # W14x90
    d_c, t_w_c, b_cf, t_cf = 0.355, 0.0112, 0.368, 0.018
    # W24x84
    d_b, t_w_b, b_fb, t_fb = 0.612, 0.0117, 0.229, 0.0236
    Z_x_beam = 3.99e-3       # m^3
    f_y_st = 345e6           # Gr 50

    print(f"\nMaterials: f_y = {f_y_st*1e-6:.0f} MPa")
    print(f"Column:   W14x90    d_c={d_c*1000:.0f} mm, t_w={t_w_c*1000:.1f} mm")
    print(f"Beam:     W24x84    d_b={d_b*1000:.0f} mm, b_f={b_fb*1000:.0f} mm")
    print(f"          Z_x = {Z_x_beam*1e6:.0f} cm^3, M_p = "
          f"{f_y_st * Z_x_beam * 1e-3:.0f} kN.m")

    # ---- (1) Krawinkler panel zone ----------------------------------
    print()
    line()
    print("(1) Krawinkler panel zone (no doubler)")
    line()
    pz = krawinkler_panel_zone(
        f_y=f_y_st, d_c=d_c, t_p=t_w_c,
        d_b=d_b, b_cf=b_cf, t_cf=t_cf,
    )
    print(f"V_y = {pz.V_y*1e-3:.0f} kN    "
          f"(boundary contribution + {pz.b_over_a*100:.0f}%)")
    print(f"K_e = {pz.K_e*1e-6:.1f} MN/rad,   K_p = {pz.K_p*1e-6:.1f} MN/rad")
    print(f"M_y at joint = {pz.M_y_joint*1e-3:.0f} kN.m")
    print(f"gamma_y     = {pz.gamma_y*1000:.2f} milliradians")
    print()
    print("Compare to beam M_p = "
          f"{f_y_st * Z_x_beam * 1e-3:.0f} kN.m :")
    M_p_beam = f_y_st * Z_x_beam
    if pz.M_y_joint > 1.1 * M_p_beam:
        print("  Panel-zone strength exceeds beam strength by > 10% -- ")
        print("  beam will yield first (preferred ductile behaviour).")
    else:
        print("  Panel zone may yield before beam -- consider adding")
        print("  a doubler plate.")

    # ---- (2) AISC 358 RBS ------------------------------------------
    print()
    line()
    print("(2) AISC 358 Reduced Beam Section")
    line()
    L_clear = 6.0    # m clear span
    rbs = aisc358_recommended_RBS(
        d=d_b, b_f=b_fb, t_f=t_fb,
        f_y=f_y_st, Z_x=Z_x_beam, L_clear=L_clear,
    )
    print(f"Recommended dimensions (mid-point of AISC ranges):")
    print(f"  a = {rbs.a*1000:.0f} mm,  b = {rbs.b*1000:.0f} mm,  "
          f"c = {rbs.c*1000:.0f} mm")
    print(f"  AISC limits OK: a={rbs.aisc_a_ok}, b={rbs.aisc_b_ok}, "
          f"c={rbs.aisc_c_ok}")
    print(f"Reduced flange width b_f_red = {rbs.b_f_reduced*1000:.0f} mm "
          f"(from {b_fb*1000:.0f} mm)")
    print(f"Z_RBS = {rbs.Z_RBS*1e6:.0f} cm^3  (= "
          f"{rbs.Z_RBS / Z_x_beam * 100:.0f} % of Z_x)")
    print(f"M_p,RBS = {rbs.M_p_RBS*1e-3:.0f} kN.m   "
          f"(at the dog-bone)")
    print(f"M_pr,face = {rbs.M_pr_face*1e-3:.0f} kN.m   "
          f"(used for column / panel-zone capacity design)")

    # ---- (3) PR connection M-theta comparison ----------------------
    print()
    line()
    print("(3) Richard-Abbott PR connection M-theta comparison")
    line()
    presets = [
        ("top_seat_double_web", "Top+seat angle, double web"),
        ("end_plate_4_bolts",   "End-plate, 4 bolts"),
        ("end_plate_extended",  "Extended end-plate"),
        ("tee_stub",             "T-stub"),
    ]
    pr_models = {label: Pr_preset(name) for name, label in presets}

    thetas = np.array([0.001, 0.005, 0.010, 0.020, 0.040])
    print(f"\n  {'theta (rad)':>14}", end="")
    for label in pr_models:
        print(f"{label[:18]:>20}", end="")
    print()
    line(80)
    for t in thetas:
        print(f"  {t:>14.4f}", end="")
        for ra in pr_models.values():
            M_val = ra.M(t)
            print(f"{M_val*1e-3:>18.0f} kN", end="")
        print()

    print()
    print("Initial rotational stiffness (for system stiffness assembly):")
    for label, ra in pr_models.items():
        print(f"  {label:<32}: R_ki = "
              f"{ra.R_ki*1e-6:6.1f} MN.m/rad")

    # ---- (4) Bolt + weld design (AISC vs IS 800) -------------------
    print()
    line()
    print("(4) Bolt + weld design check (M22 bolts, 8 mm fillet)")
    line()

    # 4-M22 bolts in shear (single shear, threads excluded)
    bs_aisc = bolt_shear_aisc(
        n_bolts=4, A_b=3.80e-4, F_nv=0.563 * 825e6,
    )
    bs_is = bolt_shear_is800(
        n_bolts=4, f_ub=800e6, A_nb=303e-6, A_sb=380e-6,
        n_shear_planes_thread=0, n_shear_planes_shank=1,
    )
    print(f"4-M22 shear (1 plane):")
    print(f"  AISC 360 (A325-X) : V_d = {bs_aisc.V_d_total*1e-3:.0f} kN")
    print(f"  IS 800 (Gr 8.8)   : V_d = {bs_is.V_d_total*1e-3:.0f} kN")

    # Bearing on 10 mm shear tab, end distance 40 mm
    bb_aisc = bolt_bearing_aisc(
        n_bolts=4, d_b=0.022, t=0.010,
        F_u=450e6, L_c=0.040,
    )
    bb_is = bolt_bearing_is800(
        n_bolts=4, d_b=0.022, t=0.010,
        f_u=450e6, f_ub=800e6, e=0.040,
    )
    print(f"\n4-M22 bearing on 10 mm plate:")
    print(f"  AISC 360 : V_d = {bb_aisc.V_d_total*1e-3:.0f} kN")
    print(f"  IS 800   : V_d = {bb_is.V_d_total*1e-3:.0f} kN  "
          f"({bb_is.note})")

    # Fillet weld on the shear tab
    fw_aisc = fillet_weld_aisc(leg_size=0.008, F_EXX=480e6)
    fw_is = fillet_weld_is800(leg_size=0.008, f_u_weld=410e6)
    print(f"\n8 mm fillet weld:")
    print(f"  AISC 360 (E70XX) : R_d/length = "
          f"{fw_aisc.R_d_per_length*1e-3:.2f} N/mm")
    print(f"  IS 800           : R_d/length = "
          f"{fw_is.R_d_per_length*1e-3:.2f} N/mm")

    # Block shear example
    bs = block_shear_aisc(
        A_gv=600e-6, A_nv=400e-6, A_nt=200e-6,
        F_y=345e6, F_u=450e6,
    )
    print(f"\nBlock-shear rupture (AISC J4-5): R_d = {bs.R_d*1e-3:.0f} kN")

    print()
    print("=" * 80)
    print("Theme I closed: panel zone + RBS + PR + bolts + welds operational.")
    print("=" * 80)


if __name__ == "__main__":
    main()
