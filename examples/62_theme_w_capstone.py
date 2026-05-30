"""Theme W capstone -- flat-plate slab design.

A typical office bay:

* Plan: 6 m x 6 m (square interior panel)
* Slab: 250 mm thick flat plate
* Columns: 400 x 400 mm at each corner of a 3 x 3 grid of panels
* Concrete: f_c' = 30 MPa (ACI), f_ck = 30 MPa (EC2 / IS)
* Steel:    f_y = 420 MPa
* Loads:    DL = 6 kPa (self-weight + super-imposed), LL = 4 kPa
            -> w_u = 1.2 * 6 + 1.6 * 4 = 13.6 kPa

The capstone:

1. **DDM moments** for the interior panel.
2. **Punching shear** at an interior column (V_u from tributary
   load), checked against ACI / EC2 / IS 456.
3. **Diaphragm classification** assuming the 250 mm slab is
   automatically rigid (span/depth shortcut).
4. **Lateral force transfer** to a 3-wall shear core through the
   rigid diaphragm under a 1000 kN seismic floor force.
"""
from __future__ import annotations

import sys

from femsolver.design.diaphragm import (
    classify_diaphragm,
    rigid_transfer,
)
from femsolver.design.punching import (
    aci318_punching_capacity,
    aci318_punching_demand,
    eurocode_punching_capacity,
    is456_punching_capacity,
)
from femsolver.design.two_way_slab import (
    ddm_minimum_thickness,
    ddm_panel,
)


def main():
    print("=" * 78)
    print("Theme W capstone -- flat-plate slab design (6 x 6 m bay)")
    print("=" * 78)

    # Parameters
    l_long = 6.0
    l_short = 6.0
    col_size = 0.4
    h_slab = 0.25
    cover = 0.025
    bar_dia = 0.012
    d = h_slab - cover - bar_dia        # effective depth
    f_c = 30e6
    f_y = 420e6
    DL = 6e3
    LL = 4e3
    w_u = 1.2 * DL + 1.6 * LL
    print()
    print(f"  Slab h = {h_slab*1e3:.0f} mm, d = {d*1e3:.0f} mm")
    print(f"  Column 400x400 mm, f_c' = 30 MPa, f_y = 420 MPa")
    print(f"  w_u = 1.2 DL + 1.6 LL = {w_u/1e3:.2f} kPa")

    # ===== minimum thickness check =====
    l_n = l_long - col_size
    h_min = ddm_minimum_thickness(l_n=l_n, f_y=f_y, interior_panel=True)
    print(f"  h_min (l_n={l_n:.2f} m, interior) = {h_min*1e3:.0f} mm  "
          f"{'OK' if h_slab >= h_min else 'NG'}")

    # ===== DDM moments =====
    res = ddm_panel(w_u=w_u, l_long=l_long, l_short=l_short, col_size=col_size)
    print()
    print(f"  DDM interior-panel moments (panel area {l_long}x{l_short} m)")
    print(f"     M_o            = {res.M_o/1e3:7.1f} kN.m")
    print(f"     M_neg_int      = {res.M_neg_int/1e3:7.1f} kN.m  "
          f"(col strip {res.M_col_strip_neg_int/1e3:.1f}, "
          f"mid {res.M_mid_strip_neg_int/1e3:.1f})")
    print(f"     M_pos_int      = {res.M_pos_int/1e3:7.1f} kN.m  "
          f"(col strip {res.M_col_strip_pos_int/1e3:.1f}, "
          f"mid {res.M_mid_strip_pos_int/1e3:.1f})")

    # ===== punching shear =====
    # Tributary area = l_long * l_short = 36 m^2
    # V_u at interior column = w_u * (tributary - column footprint)
    A_trib = l_long * l_short - col_size * col_size
    V_u = w_u * A_trib
    print()
    print(f"  Punching at interior column:  V_u = {V_u/1e3:.1f} kN")
    print(f"  {'Code':<14} {'v_c (MPa)':>12} {'V_c (kN)':>12} {'DCR':>8}")
    for code_fn, code_args, code_label in [
        (aci318_punching_capacity,
         dict(c_x=col_size, c_y=col_size, d=d, f_c=f_c, position="interior"),
         "ACI 318-19"),
        (eurocode_punching_capacity,
         dict(c_x=col_size, c_y=col_size, d=d, f_ck=f_c, rho_l=0.01,
              position="interior"),
         "EC2"),
        (is456_punching_capacity,
         dict(c_x=col_size, c_y=col_size, d=d, f_ck=f_c, position="interior"),
         "IS 456"),
    ]:
        r = code_fn(**code_args)
        # demand stress (no unbalanced moment for symmetric loading)
        if code_label == "ACI 318-19":
            v_u_stress = aci318_punching_demand(
                V_u=V_u, c_x=col_size, c_y=col_size, d=d, position="interior",
            )
            phi = 0.75
            dcr = v_u_stress / (phi * r.v_c)
        else:
            v_u_stress = V_u / (r.b_0 * r.d)
            dcr = v_u_stress / r.v_c
        print(f"  {code_label:<14} {r.v_c/1e6:12.3f} "
              f"{r.V_c/1e3:12.1f} {dcr:8.3f}")

    # ===== diaphragm classification + transfer =====
    print()
    span_over_depth = 6.0 / h_slab
    cls = classify_diaphragm(
        delta_d=0.001, delta_drift_avg=0.005,
        span_over_depth=span_over_depth, material="concrete",
    )
    print(f"  Diaphragm: 6 m span / {h_slab*1e3:.0f} mm thick = "
          f"span/depth = {span_over_depth:.1f}")
    print(f"  Classification: {cls.upper()}")

    # ===== rigid transfer to 3-wall core =====
    # Imagine 3 shear walls in the 6 m direction:
    # wall A at x=0, wall B at x=3 (centre), wall C at x=6
    # Stiffnesses K_A = K_C = 1e8 N/m, K_B = 4e8 N/m (longer middle wall)
    # Apply a 1000 kN floor force at x = 4 m (eccentric)
    F_floor = 1000e3
    shares, e_x = rigid_transfer(
        F_total=F_floor,
        elements=[("A", 0.0, 1e8), ("B", 3.0, 4e8), ("C", 6.0, 1e8)],
        F_position=4.0,
    )
    print()
    print(f"  Rigid-diaphragm transfer of F={F_floor/1e3:.0f} kN at x=4 m")
    print(f"  Eccentricity from centre of rigidity: e_x = {e_x*1e3:.0f} mm")
    print(f"  {'wall':>4} {'K (N/m)':>12} {'direct (kN)':>12} "
          f"{'torsion':>10} {'total':>10}")
    for s in shares:
        print(f"  {s.element_id:>4} {s.K:12.2e} {s.F_direct/1e3:12.1f} "
              f"{s.F_torsion/1e3:10.1f} {s.F_total/1e3:10.1f}")

    print()
    print("Theme W capstone DONE.")


if __name__ == "__main__":
    sys.exit(main())
