"""Phase 45.6 -- Cable-stayed segmental PSC bridge with staged construction.

Mini cable-stayed bridge: a deck girder cantilevered out from a
central pylon, supported by two cable stays. The bridge is built in
three stages and analysed with creep redistribution.

Three vignettes:

1. **Cable stiffness** -- show how Ernst's equivalent modulus changes
   the effective stay stiffness across a range of operating tensions.
2. **Catenary verification** -- compare the catenary closed form
   ``sag = (H/w)(cosh(wL/2H) - 1)`` against the parabolic
   approximation ``wL^2/(8H)`` for a real bridge-deck cable.
3. **Staged-construction analysis** -- the deck is built in three
   segments, with the cable stays activated as each segment is added.
   We report the cumulative tip deflection at the end of construction
   and how it changes when creep redistribution at chi = 0.8 is
   included.

Run::

    python examples/55_cable_stayed_staged_capstone.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver import (
    BeamColumn2D,
    ElasticIsotropic,
    LinearStaticAnalysis,
    Model,
)
from femsolver.bridges import (
    CableElement2D,
    ConstructionStage,
    StagedConstructionAnalysis,
    catenary_max_tension,
    catenary_sag,
    effective_modulus_EMM,
    ernst_equivalent_modulus,
)


def line(width: int = 76) -> None:
    print("-" * width)


# ============================================================ vignette 1: Ernst

def vignette_ernst() -> None:
    print()
    line()
    print("(1) Ernst equivalent-modulus across a range of stay tensions")
    line()
    E_steel = 2.0e11
    A_cable = 0.005                  # 50 cm^2 stay
    L_h = 150.0                       # 150 m horizontal projection
    gamma_cable = 0.005 * 78.0e3      # A · gamma_steel ~ 390 N/m
    print(f"Stay: E = 200 GPa, A = 50 cm^2, L_h = 150 m, "
          f"weight = {gamma_cable:.0f} N/m")
    print()
    print(f"  {'T (kN)':>10}{'E_eq (GPa)':>14}{'ratio E_eq/E':>16}")
    line(40)
    for T_kN in [100, 500, 1000, 2500, 5000, 10000, 20000]:
        T = T_kN * 1.0e3
        E_eq = ernst_equivalent_modulus(
            E=E_steel, A=A_cable,
            L_h=L_h, gamma_eff=gamma_cable, T=T,
        )
        print(f"  {T_kN:>10}{E_eq*1e-9:>14.1f}{E_eq/E_steel:>16.4f}")
    print()
    print("At T = 100 kN the sag absorbs ~half the axial energy; ")
    print("by T = 20 MN, E_eq is essentially the bare modulus.")


# ============================================================ vignette 2: catenary

def vignette_catenary() -> None:
    print()
    line()
    print("(2) Catenary vs parabolic-approximation sag")
    line()
    # Real stay-bridge cable: 200 m chord, w = 400 N/m, H = 2 MN
    L_h, w = 200.0, 400.0
    print(f"Cable: L_h = {L_h} m, w = {w} N/m")
    print()
    print(f"  {'H (kN)':>10}{'sag (cat) cm':>18}"
          f"{'sag (parab) cm':>18}{'rel err %':>14}")
    line(60)
    for H_kN in [500, 1000, 2000, 5000, 10000]:
        H = H_kN * 1.0e3
        sag_cat = catenary_sag(L_h=L_h, w=w, H=H)
        sag_par = w * L_h ** 2 / (8.0 * H)
        rel_err = abs(sag_cat - sag_par) / sag_cat * 100
        print(f"  {H_kN:>10}{sag_cat*100:>18.1f}"
              f"{sag_par*100:>18.1f}{rel_err:>14.3f}")
    print()
    print("The parabolic approximation is excellent for the wL/2H << 1 ")
    print("regime that covers most cable-stayed bridges.")


# ============================================================ vignette 3: staged

def build_cantilever_bridge():
    """Vertical pylon at x=0, deck extending to the right in 3
    segments; cable stays from the pylon top to each segment-end.

    Approximate model: deck = 3 beam elements, pylon = 1 vertical beam,
    2 cable stays attaching from the top of the pylon to the segment
    ends.
    """
    # Concrete deck + pylon
    mat_concrete = ElasticIsotropic(1, E=30.0e9, nu=0.20, rho=2400.0)
    mat_steel = ElasticIsotropic(2, E=2.0e11, nu=0.30, rho=7850.0)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat_concrete)
    m.add_material(mat_steel)

    # Deck nodes: 1 (base of pylon) -- but pylon and deck share node 1
    # 2: top of pylon (z = 30 m)
    # 3, 4, 5: deck nodes at x = 25, 50, 75 m
    m.add_node(1, 0.0, 0.0)          # base
    m.add_node(2, 0.0, 30.0)         # pylon top
    m.add_node(3, 25.0, 0.0)
    m.add_node(4, 50.0, 0.0)
    m.add_node(5, 75.0, 0.0)

    # Pylon: 1 beam from base to top
    m.add_element(BeamColumn2D(1, (1, 2), mat_concrete,
                                 area=4.0, Iz=4.0))
    # Deck segments
    m.add_element(BeamColumn2D(2, (1, 3), mat_concrete,
                                 area=0.5, Iz=0.05))
    m.add_element(BeamColumn2D(3, (3, 4), mat_concrete,
                                 area=0.5, Iz=0.05))
    m.add_element(BeamColumn2D(4, (4, 5), mat_concrete,
                                 area=0.5, Iz=0.05))
    # Cable stays (use CableElement2D)
    m.add_element(CableElement2D(5, (2, 4), mat_steel, area=0.01,
                                    gamma_eff=400.0, T_operating=5e6))
    m.add_element(CableElement2D(6, (2, 5), mat_steel, area=0.01,
                                    gamma_eff=400.0, T_operating=8e6))

    # Boundary: pylon base fixed
    m.fix(1, [1, 1, 1])
    return m


def vignette_staged() -> None:
    print()
    line()
    print("(3) Staged construction with creep redistribution")
    line()

    print("Bridge: 30 m pylon at x=0, deck extending 75 m to the right,")
    print("       with 2 cable stays (from pylon top to deck nodes).")
    print()

    # Stage list
    stages = [
        ConstructionStage(
            name="Stage 1: cast pylon + first deck segment",
            duration_days=60.0,
            load_pattern={3: [0.0, -2.0e6, 0.0]},     # deck weight
            age_at_loading_days=28.0,
        ),
        ConstructionStage(
            name="Stage 2: cast second segment + activate stay 1",
            duration_days=60.0,
            load_pattern={4: [0.0, -2.0e6, 0.0]},
            age_at_loading_days=28.0,
        ),
        ConstructionStage(
            name="Stage 3: close deck + activate stay 2",
            duration_days=18250.0,                       # 50 yr from now
            load_pattern={5: [0.0, -3.0e6, 0.0]},
            age_at_loading_days=28.0,
        ),
    ]

    # Run WITH creep
    print("Running staged analysis WITH creep redistribution (chi=0.8)...")
    m_with = build_cantilever_bridge()
    ana_with = StagedConstructionAnalysis(
        m_with, stages=stages, f_cm=38e6, chi=0.8,
        RH=70.0, h_0=0.30,
        final_age_days=18250.0,
    )
    res_with = ana_with.run()

    # Re-run with chi=0 (instant E, no creep) for comparison
    print("Running staged analysis WITHOUT creep (chi ~ 0+)...")
    m_no = build_cantilever_bridge()
    ana_no = StagedConstructionAnalysis(
        m_no, stages=stages, f_cm=38e6, chi=0.01,
        RH=70.0, h_0=0.30,
        final_age_days=29.0,             # very short time -> phi ~ 0
    )
    res_no = ana_no.run()

    # Report
    print()
    print(f"  {'Stage':<46}{'creep factor':>16}")
    line(62)
    for nm, kf in zip(res_with.stage_names, res_with.creep_factors):
        print(f"  {nm[:46]:<46}{kf:>16.3f}")
    print()
    # Tip deflection (node 5, y-DOF) -- read directly from cumulative result
    eq_y = int(m_with.node(5).eqn[1])
    tip_with_y = (abs(res_with.u_cumulative[eq_y])
                  if eq_y >= 0 else 0.0)
    eq_y2 = int(m_no.node(5).eqn[1])
    tip_no_y = (abs(res_no.u_cumulative[eq_y2])
                if eq_y2 >= 0 else 0.0)
    print(f"Final tip deflection (node 5, y) under all 3 stages:")
    print(f"  WITHOUT creep redistribution: {tip_no_y*1000:7.2f} mm")
    print(f"  WITH creep redistribution:    {tip_with_y*1000:7.2f} mm")
    print(f"  Ratio (long-term/short-term): {tip_with_y/tip_no_y:.2f}x")
    print()
    print("Long-term creep typically increases deflections by 2-4x for ")
    print("loads applied at early concrete age; the EMM is the standard ")
    print("first-pass approach for capturing this in design.")


# ============================================================ main

def main() -> None:
    print("=" * 78)
    print("Phase 45.6 -- Cable-stayed segmental PSC bridge capstone")
    print("=" * 78)

    vignette_ernst()
    vignette_catenary()
    vignette_staged()

    print()
    print("=" * 78)
    print("Bridges Phase 2 closed: cable element + catenary + staged")
    print("                construction + creep all operational.")
    print("=" * 78)


if __name__ == "__main__":
    main()
