"""Theme S capstone -- wind on a 4-storey building, three codes side-by-side.

Building geometry (rectangular, enclosed):

* Plan: 30 m (parallel to wind, L) x 20 m (perpendicular, B)
* Height: 16 m (4 storeys at 4 m each), flat roof
* Exposure: ASCE 7 category C / IS 875 category 2 / EC1 terrain II
* Basic wind speed:
  * ASCE 7-22:  V = 50 m/s (~ 112 mph; coastal Florida)
  * IS 875:     V_b = 50 m/s (typical zone IV; Mumbai region)
  * EC1:        v_b = 27 m/s (typical UK / central Europe)

The script:

1. Computes the windward-face pressure profile under each code at
   the four floor heights (4, 8, 12, 16 m).
2. Calculates per-floor wind shears from tributary area.
3. Vortex-shedding check on a chimney mounted on the roof.

Output: a side-by-side floor-shear table + optional bar plot.
"""
from __future__ import annotations

import sys

import numpy as np

from femsolver.hazard.wind import (
    asce7_mwfrs_design_pressures,
    asce7_velocity_pressure,
    asce7_wall_Cp,
    ec1_peak_velocity_pressure,
    is875_design_wind_pressure,
    is_lock_in_risk,
    scruton_number,
    vortex_shedding_frequency,
)


def main():
    print("=" * 78)
    print("Theme S capstone -- wind loads on a 4-storey building")
    print("=" * 78)
    print()
    print("  Plan L x B = 30 m x 20 m  (L parallel to wind)")
    print("  Height H = 16 m (4 storeys x 4 m, flat roof)")
    print()

    # Storey midheights (where we report pressure)
    floors = [(1, 4.0), (2, 8.0), (3, 12.0), (4, 16.0)]
    # Tributary length (full storey height per floor)
    h_trib = 4.0
    # Building plan width perpendicular to wind = 20 m
    B = 20.0
    H = 16.0
    L = 30.0

    # ===== ASCE 7-22 =====
    V = 50.0
    print(f"ASCE 7-22 (V = {V} m/s, exposure C, G = 0.85)")
    print(f"  {'floor':>5} {'z (m)':>6} {'q_z (kPa)':>10} "
          f"{'p_wind (kPa)':>13} {'F_storey (kN)':>14}")
    F_asce = []
    walls = asce7_wall_Cp(L / B)
    for f, z in floors:
        res = asce7_mwfrs_design_pressures(
            z=z, h=H, V=V, L=L, B=B, exposure="C",
        )
        F = res.p_windward * B * h_trib
        F_asce.append(F)
        print(f"  {f:5d} {z:6.1f} {res.q_z/1e3:10.3f} "
              f"{res.p_windward/1e3:13.3f} {F/1e3:14.3f}")
    base_shear_asce = sum(F_asce) - sum(
        asce7_mwfrs_design_pressures(z=z, h=H, V=V, L=L, B=B, exposure="C").p_leeward
        * B * h_trib for _, z in floors
    )
    print(f"  Total base shear (windward - leeward) = "
          f"{base_shear_asce/1e3:.1f} kN")

    # ===== IS 875 =====
    print()
    print(f"IS 875 Part 3 (V_b = {V} m/s, terrain category 2)")
    print(f"  {'floor':>5} {'z (m)':>6} {'p_z (kPa)':>10} "
          f"{'p_design':>9} {'F_storey':>10}")
    # IS 875 uses a similar windward Cp = +0.8 on a tall building
    Cp_win = 0.8
    F_is = []
    for f, z in floors:
        res = is875_design_wind_pressure(z=z, V_b=V, category=2)
        p_design = res.p_z * Cp_win
        F = p_design * B * h_trib
        F_is.append(F)
        print(f"  {f:5d} {z:6.1f} {res.p_z/1e3:10.3f} "
              f"{p_design/1e3:9.3f} {F/1e3:10.3f}")
    print(f"  Total windward force = {sum(F_is)/1e3:.1f} kN")

    # ===== EC1 =====
    print()
    v_b = 27.0
    print(f"EN 1991-1-4 (v_b = {v_b} m/s, terrain II)")
    print(f"  {'floor':>5} {'z (m)':>6} {'q_p (kPa)':>10} "
          f"{'p_design':>9} {'F_storey':>10}")
    # EC1 also uses external windward c_pe = +0.8 for rectangular plan
    cpe_win = 0.8
    F_ec1 = []
    for f, z in floors:
        res = ec1_peak_velocity_pressure(z=z, v_b=v_b, terrain="II")
        p_design = res.q_p * cpe_win
        F = p_design * B * h_trib
        F_ec1.append(F)
        print(f"  {f:5d} {z:6.1f} {res.q_p/1e3:10.3f} "
              f"{p_design/1e3:9.3f} {F/1e3:10.3f}")
    print(f"  Total windward force = {sum(F_ec1)/1e3:.1f} kN")

    # ===== Vortex shedding on a roof chimney =====
    print()
    print("Vortex shedding on a 4 m dia, 20 m tall roof chimney:")
    print("  (steel, m_e = 600 kg/m, zeta = 0.005)")
    U_design = 30.0       # m/s mean wind at chimney top
    vs = vortex_shedding_frequency(U=U_design, D=4.0, St=0.20)
    Sc = scruton_number(m_e=600.0, zeta=0.005, D=4.0)
    f_n_chimney = 1.5      # natural lateral frequency (Hz)
    print(f"  U = {U_design} m/s, D = 4 m -> f_s = {vs.f_s:.3f} Hz")
    print(f"  Scruton number Sc = {Sc:.2f}")
    print(f"  Chimney lateral natural freq f_n = {f_n_chimney} Hz")
    risk = is_lock_in_risk(f_s=vs.f_s, f_n=f_n_chimney, Sc=Sc)
    print(f"  Lock-in risk: {'YES' if risk else 'no'} "
          f"(Sc < 10 and |f_s - f_n|/f_n < 20%)")

    # Optional plot
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(1, 5)
        width = 0.25
        ax.bar(x - width, np.array(F_asce) / 1e3, width, label="ASCE 7-22")
        ax.bar(x, np.array(F_is) / 1e3, width, label="IS 875")
        ax.bar(x + width, np.array(F_ec1) / 1e3, width, label="EC1")
        ax.set_xlabel("Floor")
        ax.set_ylabel("Storey wind force (kN, windward only)")
        ax.set_title("Per-floor wind shear: code comparison")
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = "wind_code_comparison.png"
        fig.savefig(out, dpi=120)
        print()
        print(f"  Figure saved: {out}")
    except ImportError:
        print()
        print("  (matplotlib not installed -- skipping plot)")

    print()
    print("Theme S capstone DONE.")


if __name__ == "__main__":
    sys.exit(main())
