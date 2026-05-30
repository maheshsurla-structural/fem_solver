"""Phase 38.6 -- Simply-supported PSC girder bridge under HL-93.

End-to-end design walkthrough of a 30 m simply-supported precast
prestressed concrete (PSC) girder + cast-in-place RC deck under
AASHTO HL-93 live load with time-dependent prestress losses.

Pipeline:

1. **Composite section** (transformed) -- precast I-girder + RC deck.
2. **Dead-load moments** -- girder self-weight + deck weight +
   superimposed dead load.
3. **HL-93 live-load envelope** -- truck + tandem + lane, governing
   chosen and amplified by impact (1.33).
4. **PT tendon** -- parabolic profile + friction losses + anchorage
   slip.
5. **Long-term losses** -- CEB-FIP 2010 creep + shrinkage and steel
   relaxation, summed for the effective prestress at infinity.
6. **Stress checks** -- top of girder, top of deck, bottom of girder
   at midspan under (DL + LL + PS).

Run::

    python examples/49_psc_girder_bridge.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver.bridges import (
    aashto_lane_moment_simple_span,
    aashto_hl93_lane_load_kN_per_m,
    anchorage_slip_loss,
    cebfip_creep_coefficient,
    cebfip_shrinkage,
    composite_fiber_stresses,
    composite_girder_deck,
    equivalent_uniform_load_parabolic,
    friction_loss,
    max_truck_envelope_simple_span,
    parabolic_drape_profile,
    prestress_long_term_loss,
    steel_relaxation_loss_ratio,
)


# ============================================================ inputs

# Span
L = 30.0                     # m

# Girder (precast I-girder, ~1.2 m deep)
A_g = 0.45                   # m^2
I_g = 0.06                   # m^4   (precast I-girder reference)
y_cg_girder = 0.55           # m, from girder bottom
h_g = 1.20                   # m
E_g = 34.0e9                 # Pa
f_ck_g = 50.0e6              # girder concrete characteristic strength
f_cm_g = f_ck_g + 8.0e6      # mean strength
gamma_concrete = 24.0e3      # N/m^3

# Deck (CIP RC)
b_d = 2.40                   # m, deck effective width
h_d = 0.20                   # m
E_d = 28.0e9                 # Pa
f_ck_d = 30.0e6
gamma_deck = 24.0e3

# Strand
A_ps = 2.0e-3                # m^2 (about 14 strands x 140 mm^2)
E_p = 1.95e11
f_pi = 0.75 * 1860e6         # 1395 MPa initial stress
f_py = 0.9 * 1860e6
P_0_total = f_pi * A_ps      # initial jacking force per tendon
strand_y_from_bottom = 0.15   # m

# Tendon profile (drape)
drape = 0.70                  # m (mid-span eccentricity below cg girder)

# Time-dependent
RH = 70.0                     # %
h_0 = 2 * A_g / (2 * (1.2 + 0.5))   # crude perimeter
t_load = 28.0                 # days
t_drying = 3.0                # days
t_50yr = 18250.0              # days


# ============================================================ helpers

def line(width: int = 76) -> None:
    print("-" * width)


def main() -> None:
    print("=" * 80)
    print("Phase 38.6 -- PSC girder bridge, L = "
          f"{L} m, HL-93 + CEB-FIP losses")
    print("=" * 80)

    # ---- (1) Composite section ----------------------------------------
    print()
    line()
    print("(1) Composite girder + deck section")
    line()
    props = composite_girder_deck(
        girder_area=A_g, girder_I=I_g,
        girder_y_centroid=y_cg_girder, girder_height=h_g,
        deck_width=b_d, deck_thickness=h_d,
        E_girder=E_g, E_deck=E_d,
    )
    print(f"  Modular ratio n = E_d/E_g  = {props.n:.3f}")
    print(f"  Transformed area A_t       = {props.A_t:.3f} m^2")
    print(f"  Centroid y_bar (from bot)  = {props.y_bar:.3f} m")
    print(f"  Transformed I_t            = {props.I_t:.4f} m^4")
    print(f"  S_t,top  = {props.S_t_top:.4f} m^3   "
          f"S_t,bot  = {props.S_t_bot:.4f} m^3")

    # ---- (2) Dead-load moments ----------------------------------------
    print()
    line()
    print("(2) Dead-load moments")
    line()
    w_girder = gamma_concrete * A_g                  # N/m
    w_deck = gamma_deck * b_d * h_d                  # N/m
    w_sdl = 2.5e3                                    # N/m superimposed
    M_girder = w_girder * L ** 2 / 8.0
    M_deck = w_deck * L ** 2 / 8.0
    M_sdl = w_sdl * L ** 2 / 8.0
    M_DL = M_girder + M_deck + M_sdl
    print(f"  w_girder = {w_girder*1e-3:.2f} kN/m  -> "
          f"M_girder = {M_girder*1e-3:.0f} kN.m")
    print(f"  w_deck   = {w_deck*1e-3:.2f} kN/m  -> "
          f"M_deck   = {M_deck*1e-3:.0f} kN.m")
    print(f"  w_sdl    = {w_sdl*1e-3:.2f} kN/m  -> "
          f"M_sdl    = {M_sdl*1e-3:.0f} kN.m")
    print(f"  Total DL moment at midspan = {M_DL*1e-3:.0f} kN.m")

    # ---- (3) HL-93 envelope at midspan --------------------------------
    print()
    line()
    print("(3) HL-93 live-load envelope at midspan")
    line()
    env = max_truck_envelope_simple_span(L=L, x=L/2, impact_factor=1.33)
    print(f"  Truck      moment     = {env['M_truck']*1e-3:>6.0f} kN.m")
    print(f"  Tandem     moment     = {env['M_tandem']*1e-3:>6.0f} kN.m")
    print(f"  Lane       moment     = {env['M_lane']*1e-3:>6.0f} kN.m")
    print(f"  Governing vehicle      = {env['vehicle_governing']}")
    print(f"  Governing total moment = {env['M_governing']*1e-3:>6.0f} kN.m")
    print(f"  with impact (×1.33)    = {env['M_with_impact']*1e-3:>6.0f} kN.m")
    M_LL = env['M_with_impact']

    # ---- (4) PT tendon: friction + anchorage slip ---------------------
    print()
    line()
    print("(4) PT tendon: friction + anchorage slip")
    line()
    profile = parabolic_drape_profile(L=L, drape=drape, n_segments=40)
    fric = friction_loss(profile, mu=0.20, k=0.0066)
    print(f"  Drape = {drape} m, profile total length = "
          f"{profile.total_length:.2f} m")
    print(f"  Initial jacking force P_0  = {P_0_total*1e-3:.0f} kN")
    print(f"  P/P_0 at midspan           = {fric.P_over_P0[20]:.4f}")
    print(f"  P/P_0 at far end           = {fric.P_over_P0[-1]:.4f}")
    ank = anchorage_slip_loss(
        profile, P_0=P_0_total, mu=0.20, k=0.0066,
        slip=0.006, E_s=E_p, A_ps=A_ps,
    )
    print(f"  Anchorage slip length l_a  = {ank.l_a:.2f} m")
    print(f"  P at anchor after seating  = "
          f"{ank.P0_after_seating*1e-3:.0f} kN")
    # Effective prestress (after instantaneous losses, before long-term)
    P_eff_inst = float(ank.P_profile.mean())
    print(f"  Effective P after inst. losses = {P_eff_inst*1e-3:.0f} kN")

    # ---- (5) Long-term losses (CEB-FIP + relaxation) -----------------
    print()
    line()
    print("(5) Long-term losses (50-yr)")
    line()
    creep = cebfip_creep_coefficient(
        t_days=t_50yr, t0_days=t_load,
        f_cm=f_cm_g, RH=RH, h_0=h_0,
    )
    shr = cebfip_shrinkage(
        t_days=t_50yr, t_s_days=t_drying,
        f_cm=f_cm_g, RH=RH, h_0=h_0,
    )
    rel = steel_relaxation_loss_ratio(
        t_hours=t_50yr * 24.0,
        fpi_over_fpy=f_pi / f_py,
        relaxation_class="low",
    )
    sigma_c_at_strand = -P_eff_inst / A_g          # Pa, crude axial only
    loss = prestress_long_term_loss(
        P_initial=P_eff_inst,
        A_ps=A_ps, E_p=E_p,
        sigma_c_at_strand=sigma_c_at_strand,
        E_c=E_g,
        creep=creep, shrinkage=shr,
        relaxation_loss_ratio=rel,
        f_pi=f_pi,
    )
    P_eff_long = loss.P_effective
    print(f"  Creep phi(50yr, 28d) = {creep.phi:.3f}")
    print(f"  Shrinkage eps_cs     = {shr.eps_cs*1e6:.0f} microstrain")
    print(f"  Relaxation ratio     = {rel*100:.2f}%")
    print()
    print(f"  Long-term creep loss     = "
          f"{loss.delta_P_creep*1e-3:>+7.1f} kN")
    print(f"  Long-term shrinkage loss = "
          f"{loss.delta_P_shrinkage*1e-3:>+7.1f} kN")
    print(f"  Long-term relaxation     = "
          f"{loss.delta_P_relaxation*1e-3:>+7.1f} kN")
    print(f"  Total long-term Delta P  = "
          f"{loss.delta_P_total*1e-3:>+7.1f} kN")
    print(f"  Effective P at 50 yr     = {P_eff_long*1e-3:.0f} kN  "
          f"({P_eff_long/P_0_total*100:.0f}% of P_0)")

    # ---- (6) Stress check at midspan ---------------------------------
    print()
    line()
    print("(6) Fiber stresses at midspan (DL + LL + PS, effective at 50yr)")
    line()
    # Equivalent prestress moment from drape:
    # P_eff acts along tendon (drape from centroid eccentricity ~ drape - (y_cg - y_strand))
    # For our composite section, strand is at y_strand_from_bottom; composite y_bar is
    # higher; eccentricity e = y_bar - strand_y_from_bottom
    e = props.y_bar - strand_y_from_bottom
    M_PS = P_eff_long * e         # +ve: prestress creates upward moment (-M is sagging)
    # Combined moment: -DL - LL + PS  (sign convention: sagging positive)
    M_total = M_DL + M_LL - M_PS
    sigma = composite_fiber_stresses(
        props=props,
        P=-P_eff_long, M=M_total,
        strand_y_from_bottom=strand_y_from_bottom,
    )
    print(f"  Eccentricity e (cg to strand) = {e*1000:.0f} mm")
    print(f"  M_PS from drape = P_eff * e   = {M_PS*1e-3:.0f} kN.m")
    print(f"  Net moment = M_DL + M_LL - M_PS = "
          f"{M_total*1e-3:.0f} kN.m")
    print()
    print(f"  Deck top   sigma = {sigma.sigma_top_deck*1e-6:>+6.2f} MPa")
    print(f"  Girder top sigma = {sigma.sigma_top_girder*1e-6:>+6.2f} MPa")
    print(f"  Girder bot sigma = {sigma.sigma_bot_girder*1e-6:>+6.2f} MPa")
    print(f"  At strand  sigma = {sigma.sigma_at_strand*1e-6:>+6.2f} MPa")
    print()
    bot_status = ("PASS (compression)" if sigma.sigma_bot_girder < 0
                  else "TENSION at bot fibre -- check serviceability")
    print(f"  Bottom-fibre status: {bot_status}")

    print()
    print("=" * 80)
    print("Theme E closed: bridges, PT, creep/shrinkage, composite "
          "section operational.")
    print("=" * 80)


if __name__ == "__main__":
    main()
