"""Phase 46.5 -- Eurocode-compliant frame design walkthrough.

A 4-storey building in southern Europe is designed to EC8 seismic
demand + EC2 RC beam/column flexure + EC2 shear + EC3 steel beam
as an alternate. Demonstrates the parallel design pipelines across
Eurocodes.

Pipeline:

1. **EC8 seismic demand** -- design spectrum (ground B, q=3.0 for
   DCM RC frame), base shear, vertical force distribution, drift
   check.
2. **EC2 RC beam** -- typical 300x500 mm beam designed for the
   factored moment + shear.
3. **EC2 RC column** -- typical 400x400 mm column designed for the
   factored axial.
4. **EC3 alternate** -- the same beam loaded as an HEB300 steel
   I-section, with LTB-reduced bending capacity and shear check.

The vignettes also report the **cross-code consistency** between EC3
and IS 800 (both share the Perry-Robertson formulation).

Run::

    python examples/56_eurocode_frame_design.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver.design import ec2, ec3, ec8


def line(width: int = 76) -> None:
    print("-" * width)


# ============================================================ inputs

N_STOREYS = 4
H_STOREY = 3.0
H_TOTAL = N_STOREYS * H_STOREY
W_FLOOR = 600.0e3            # kg per floor (DL + 30% LL)
M_TOTAL = W_FLOOR * N_STOREYS
A_G = 2.5                     # 0.25 g design ground acceleration
GROUND = "B"                  # stiff soil
SYSTEM = "RC_DCM_FRAME"       # → q = 3.0


def main() -> None:
    print("=" * 78)
    print("Phase 46.5 -- 4-storey building designed to EC2 / EC3 / EC8")
    print("=" * 78)

    print(f"\nBuilding:  {N_STOREYS} storeys x {H_STOREY} m = "
          f"{H_TOTAL:.0f} m, m_total = {M_TOTAL*1e-3:.0f} t")
    print(f"Site:      ground type {GROUND}, a_g = {A_G:.2f} m/s^2 (~ "
          f"{A_G/9.81*100:.0f}%g)")
    q = ec8.behaviour_factor_default(SYSTEM)
    print(f"System:    {SYSTEM}, behaviour factor q = {q}")

    # ---- EC8 seismic --------------------------------------------------
    print()
    line()
    print("(1) EC8 seismic demand")
    line()
    T_1 = 0.075 * H_TOTAL ** 0.75    # EC8 Cl. 4.3.3.2.2(3) (empirical)
    print(f"Empirical period T_1 = 0.075 H^0.75 = {T_1:.3f} s")
    Sd = ec8.design_spectrum_Sd(T_1, a_g=A_G, ground_type=GROUND, q=q)
    print(f"Design spectrum Sd(T_1, ground={GROUND}, q={q}) = "
          f"{Sd:.3f} m/s^2")

    res_bs = ec8.ec8_base_shear(
        T_1=T_1, m_total=M_TOTAL, a_g=A_G,
        ground_type=GROUND, q=q, n_storeys=N_STOREYS,
    )
    print(f"Base shear  F_b = Sd · m · lambda = "
          f"{res_bs.F_b*1e-3:.0f} kN  (lambda = {res_bs.lambda_factor})")

    masses = np.full(N_STOREYS, W_FLOOR)
    heights = np.arange(1, N_STOREYS + 1) * H_STOREY
    F = ec8.vertical_force_distribution(
        F_b=res_bs.F_b, storey_masses=masses, storey_heights=heights,
    )
    print(f"\nFloor forces (EC8 Cl. 4.3.3.2.3 linear distribution):")
    for i, (h, Fi) in enumerate(zip(heights, F)):
        print(f"  Floor {i+1} (h = {h:.1f} m): F_{i+1} = "
              f"{Fi*1e-3:.1f} kN")

    # Hypothetical elastic floor displacements + drift check
    u_elastic = np.array([0.002, 0.005, 0.008, 0.011])
    drift = ec8.ec8_drift_check(
        floor_disp=u_elastic,
        storey_heights=np.full(N_STOREYS, H_STOREY),
        q=q, importance_class="II", infill_type="brittle",
    )
    print(f"\nDrift check (brittle infill, II): max ratio = "
          f"{drift.max_ratio*1000:.2f} per mille  (limit "
          f"{drift.limit*1000:.1f} per mille)")
    print(f"  Status: {'PASS' if drift.passes else 'FAIL'}")

    # ---- EC2 beam -----------------------------------------------------
    print()
    line()
    print("(2) EC2 RC beam (300 x 500 mm, C30/37 + B500)")
    line()
    M_Ed = 180.0e3
    V_Ed = 130.0e3
    b_beam, d_beam = 0.300, 0.460
    f_ck = ec2.fck_class("C30/37")
    f_yk = ec2.fyk_grade("B500")

    res_flex = ec2.ec2_beam_flexure(
        M_Ed=M_Ed, f_ck=f_ck, f_yk=f_yk, b=b_beam, d=d_beam,
    )
    print(f"M_Ed = {M_Ed*1e-3:.0f} kN.m -> A_s = "
          f"{res_flex.A_s*1e6:.0f} mm^2 ({res_flex.note})")
    print(f"  x/d = {res_flex.x_over_d:.3f}, utilisation = "
          f"{res_flex.utilisation:.3f}")

    A_min, A_max = ec2.ec2_min_max_tension_steel(
        f_ck=f_ck, f_yk=f_yk, b=b_beam, d=d_beam,
    )
    print(f"  A_s,min = {A_min*1e6:.0f} mm^2, A_s,max = {A_max*1e6:.0f} mm^2 "
          f"(EC2 Cl. 9.2.1)")

    res_v = ec2.ec2_beam_shear(
        V_Ed=V_Ed, f_ck=f_ck, f_yk_sv=f_yk,
        b_w=b_beam, d=d_beam, A_s=res_flex.A_s,
    )
    print(f"\nV_Ed = {V_Ed*1e-3:.0f} kN -> V_Rd,c = "
          f"{res_v.V_Rd_c*1e-3:.1f} kN, V_Rd,max = "
          f"{res_v.V_Rd_max*1e-3:.0f} kN")
    if res_v.requires_stirrups and not math.isinf(res_v.A_sw_over_s_required):
        # 2-leg 8mm stirrup A_sw = 100.5 mm^2
        A_sw = 2.0 * math.pi / 4.0 * (0.008) ** 2
        s_max = A_sw / res_v.A_sw_over_s_required
        print(f"  A_sw/s required = "
              f"{res_v.A_sw_over_s_required*1e3:.2f} mm^2/mm")
        print(f"  Use 8 mm 2-leg stirrups at s = "
              f"{s_max*1000:.0f} mm c/c")

    # ---- EC3 alternate ------------------------------------------------
    print()
    line()
    print("(3) EC3 alternate: HEB300 steel beam (S275)")
    line()
    # HEB300 plastic-section modulus Z_pl = 1869 cm^3
    W_pl = 1869e-6
    f_y_s = ec3.fy_grade("S275")
    res_flex_s = ec3.ec3_flexure(
        M_Ed=M_Ed, W_pl=W_pl, f_y=f_y_s,
    )
    print(f"M_pl,Rd = W_pl · f_y / gamma_M0 = "
          f"{res_flex_s.M_pl_Rd*1e-3:.0f} kN.m  (no LTB)")
    print(f"  utilisation = {res_flex_s.utilisation:.3f}")

    # With LTB: assume L_LT = 3 m, M_cr made-up
    res_ltb = ec3.ec3_flexure(
        M_Ed=M_Ed, W_pl=W_pl, f_y=f_y_s, L_LT=3.0, M_cr=400e3,
    )
    print(f"With LTB (L_LT = 3 m, M_cr = 400 kN.m): "
          f"chi_LT = {res_ltb.chi_LT:.3f}, "
          f"M_b,Rd = {res_ltb.M_b_Rd*1e-3:.0f} kN.m")

    # HEB300 shear area approximate: h_w·t_w = 244·11 = 2684 mm^2
    res_shr = ec3.ec3_shear(
        V_Ed=V_Ed, A_v=2684e-6, f_y=f_y_s,
    )
    print(f"\nV_pl,Rd = A_v · f_y / (sqrt(3) gamma_M0) = "
          f"{res_shr.V_pl_Rd*1e-3:.0f} kN, util = "
          f"{res_shr.utilisation:.3f}")

    # ---- Cross-code consistency check --------------------------------
    print()
    line()
    print("(4) Cross-code consistency: EC3 vs IS 800")
    line()
    from femsolver.design import is800
    print(f"Perry-Robertson reduction chi at lambda_bar = 1.0, curve b:")
    chi_ec3 = ec3.perry_robertson_chi(lambda_bar=1.0, curve="b")
    chi_is = is800.perry_robertson_chi(lambda_bar=1.0, curve="b")
    print(f"  EC3:    chi = {chi_ec3:.4f}")
    print(f"  IS 800: chi = {chi_is:.4f}")
    print(f"  Difference = {abs(chi_ec3 - chi_is):.2e}")
    print(f"  (Same parent specification -- IS 800 adopted the EC3 form.)")

    print()
    print("=" * 78)
    print("Theme G1 closed: EC2 + EC3 + EC8 operational alongside ACI/AISC/IS.")
    print("=" * 78)


if __name__ == "__main__":
    main()
