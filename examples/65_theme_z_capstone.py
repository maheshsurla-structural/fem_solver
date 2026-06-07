"""Theme Z capstone -- catalogue-driven beam auto-selection.

A 6-m simply-supported steel beam carries a uniform factored
service load. The capstone:

1. Picks materials and concrete/steel grades from the catalogue.
2. Computes the required plastic modulus W_pl from EC3.
3. Auto-selects the lightest IPE, HEA, HEB, and ISMB that satisfies
   the demand.
4. Reports the deflection check using the catalogue's I_y.
5. Picks a bolt grade for the end connection (4 M20 8.8 in single
   shear) and reports its capacity.
"""
from __future__ import annotations

import sys

from femsolver.data import (
    auto_select_ec_section,
    bolt_lookup,
    concrete_grade,
    eurocode_section,
    indian_section,
    list_indian_sections,
    steel_grade,
)


def _auto_select_ismb(W_pl_required: float):
    """Auto-select lightest ISMB (we don't have a generic Indian
    selector yet; do it inline)."""
    pool = list_indian_sections("ISMB")
    feasible = [
        indian_section(n) for n in pool
        if indian_section(n).W_pl_y >= W_pl_required
    ]
    if not feasible:
        return None
    return min(feasible, key=lambda s: s.mass)


def main():
    print("=" * 78)
    print("Theme Z capstone -- catalogue-driven steel-beam auto-selection")
    print("=" * 78)
    print()

    # ============================ Loads + materials =========================
    L = 6.0           # m
    w_u = 30.0e3      # N/m factored UDL
    M_u = w_u * L * L / 8.0
    V_u = w_u * L / 2.0
    grade = steel_grade("S355")
    gamma_M0 = 1.0
    print(f"  Span L = {L} m, w_u = {w_u/1e3:.1f} kN/m")
    print(f"  Steel: {grade.name} (f_y = {grade.f_y/1e6:.0f} MPa)")
    print(f"  Design moment M_u = {M_u/1e3:.1f} kN.m")
    print(f"  Design shear  V_u = {V_u/1e3:.1f} kN")
    print()

    # ============================ EC3 plastic-section demand ==================
    # W_pl_required is in m^3 (M / f_y); catalogues store in mm^3.
    W_pl_req_m3 = M_u * gamma_M0 / grade.f_y
    W_pl_req_mm3 = W_pl_req_m3 * 1e9
    print(f"  Required W_pl = M_u / f_y = {W_pl_req_mm3/1e3:.1f} cm^3")
    print()

    # ============================ EC families ===============================
    print(f"  {'Family':<8s} {'Section':<10s} {'mass':>8s} "
          f"{'W_pl_y':>10s} {'DCR':>6s} {'delta/L':>10s}")
    for family in ("IPE", "HEA", "HEB"):
        sec = auto_select_ec_section(
            W_pl_required=W_pl_req_mm3, family=family, minimise="mass",
        )
        # Deflection check (5wL^4/384EI for simply supported UDL)
        E = grade.E
        delta = 5.0 * w_u * L ** 4 / (384.0 * E * sec.I_y * 1e-12)
        # I in mm^4 -> convert to m^4 via *1e-12
        # delta = 5 w L^4 / (384 E I)
        delta_m = 5.0 * w_u * L ** 4 / (384.0 * E * (sec.I_y * 1e-12))
        ratio = delta_m / L
        DCR_M = M_u / (sec.W_pl_y * 1e-9 * grade.f_y / gamma_M0)
        print(f"  {family:<8s} {sec.name:<10s} {sec.mass:>6.1f} kg/m "
              f"{sec.W_pl_y/1e3:>8.0f}cm^3 "
              f"{DCR_M:>6.3f} {ratio:>10.5f}")

    # ============================ Indian family =============================
    print()
    ismb = _auto_select_ismb(W_pl_req_mm3)
    if ismb is not None:
        delta_is = 5.0 * w_u * L ** 4 / (384.0 * 200e9 * (ismb.I_y * 1e-12))
        DCR_is = M_u / (ismb.W_pl_y * 1e-9 * 250e6 / 1.10)
        print(f"  Indian:    {ismb.name:<10s} {ismb.mass:>6.1f} kg/m "
              f"{ismb.W_pl_y/1e3:>8.0f}cm^3 "
              f"{DCR_is:>6.3f} {delta_is/L:>10.5f}")
        print(f"             (Fe410 steel, gamma_m0 = 1.10 per IS 800)")

    # ============================ Bolts =============================
    print()
    bolt = bolt_lookup("8.8", 20)
    n_bolts = 4
    # Single shear strength per ISO bolt: 0.6 * f_ub * A_t (simplification).
    # f_ub in Pa, A_t in mm^2 -> convert to m^2.
    V_per_bolt_single_shear = 0.6 * bolt.f_ub * bolt.A_t * 1e-6
    V_capacity = n_bolts * V_per_bolt_single_shear
    print(f"  End connection: {n_bolts}x M{int(bolt.d_mm)} bolt {bolt.grade}")
    print(f"  A_t = {bolt.A_t} mm^2, f_ub = {bolt.f_ub/1e6:.0f} MPa")
    print(f"  V_capacity = {V_capacity/1e3:.1f} kN  vs  V_u = {V_u/1e3:.1f} kN  "
          f"DCR = {V_u/V_capacity:.3f}")

    # ============================ Material database peek ====================
    print()
    print("  Concrete grades referenced:")
    for g_name in ("C30", "M30", "4000 psi"):
        c = concrete_grade(g_name)
        print(f"    {c.name:<12s}: f_ck = {c.f_ck/1e6:>5.1f} MPa, "
              f"E_cm = {c.E_cm/1e9:>5.1f} GPa, "
              f"f_ctm = {c.f_ctm/1e6:>4.2f} MPa")

    print()
    print("Theme Z capstone DONE.")


if __name__ == "__main__":
    sys.exit(main())
