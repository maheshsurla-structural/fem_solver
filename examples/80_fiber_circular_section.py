"""Fiber-hinge plan P1 (G1) — circular / Caltrans fiber section builder.

Builds the benchmark specimen's cross-section — a Caltrans 84 in (7 ft)
circular RC bridge column: confined core + unconfined cover concrete + a
ring of 56 #14 longitudinal bars — as a :class:`FiberSection2D` via
:func:`rc_circular_column_section`, and exercises it three ways:

1. **Geometry check** — the fiber mesh reproduces the analytic gross area
   exactly and the second moment ``Iz`` converges to ``pi D^4 / 64``.
2. **Transformed axial stiffness** — ``EA`` from the fiber sum matches
   ``Ec*A_concrete + Es*A_steel`` by hand.
3. **Axial response preview** — drive the section through monotonic axial
   compression and report the N–strain curve, previewing Stage 1 of the
   fiber-hinge axial-load benchmark (P5). The working-point strain at the
   P = 2400 kip preload is reported.

Units: **kip, inch** (the benchmark's natural units).

The confined-core law comes from the **Mander confinement pre-processor** (P2,
``mander_confined_circular``), which derives ``f'cc, eps_cc, eps_cu`` from the
#8 @ 6 in hoops — verified against the Midas card (fcc = 6.36 vs 6.36 ksi).

Run::

    PYTHONPATH=src python examples/80_fiber_circular_section.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver import FiberSection2D, ConcreteMander, rc_circular_column_section
from femsolver.materials.uniaxial.menegotto_pinto import UniaxialMenegottoPinto
from femsolver.sections.analysis import mander_confined_circular


def main() -> None:
    D = 84.0            # diameter (in)
    cover = 2.0         # cover to core boundary (in)
    Ec = 3605.0         # concrete modulus (ksi)
    fc = 5.0            # expected f'c (ksi)
    eps_c0 = 0.002219
    Es = 29000.0
    fy = 68.0

    print("=" * 72)
    print("P1 (G1) - Caltrans 84in circular RC column fiber section")
    print("=" * 72)

    # --- 1. geometry: exact area, Iz convergence (plain circle) -----------
    A_exact = math.pi * D**2 / 4.0
    I_exact = math.pi * D**4 / 64.0
    from femsolver.materials.uniaxial.elastic import UniaxialElastic
    print(f"\nAnalytic circle: A = {A_exact:.2f} in^2, Iz = {I_exact:.0f} in^4")
    print(f"\n{'mesh (rings x wedges)':>22}{'A error':>12}{'Iz error':>12}")
    print("-" * 46)
    for nr, nw in [(2, 8), (4, 16), (8, 24), (16, 48), (24, 72)]:
        s = FiberSection2D.circular(D, nr, nw, UniaxialElastic(Ec))
        ae = (s.gross_area - A_exact) / A_exact
        ie = (s.gross_Iz - I_exact) / I_exact
        print(f"{f'{nr} x {nw}':>22}{ae:>11.3%}{ie:>12.3%}")

    # --- 2. the composite RC section --------------------------------------
    # Confined core law from the Mander pre-processor (P2): 79 in core,
    # #8 hoop @ 6 in, f_yh = 68 ksi -> f'cc, eps_cc, eps_cu.
    conf = mander_confined_circular(
        fco=fc, eps_co=eps_c0, D_core=79.0, hoop_area=0.79, hoop_spacing=6.0,
        fyh=68.0, rho_long=0.0257, hoop_type="hoop", clear_spacing=5.0,
    )
    print(f"\nMander confinement (P2): f'cc = {conf.fcc:.3f} ksi "
          f"(Midas 6.356), eps_cc = {conf.eps_cc:.5f}, eps_cu = {conf.eps_cu:.4f}, "
          f"ke = {conf.ke:.4f}")
    core = conf.to_material(Ec=Ec)
    coverc = ConcreteMander(fpc=fc, eps_c0=eps_c0, Ec=Ec)   # unconfined cover
    steel = UniaxialMenegottoPinto(E=Es, sigma_y=fy, b=0.01)

    sec = rc_circular_column_section(
        diameter=D, cover=cover,
        core_concrete=core, cover_concrete=coverc,
        n_bars=56, bar_area=2.25, steel=steel,
        n_core_rings=8, n_cover_rings=2, n_wedges=24,
    )
    As = 56 * 2.25
    print(f"\nComposite section: {len(sec.fibers)} fibers "
          f"(192 core + 48 cover + 56 bars)")
    print(f"  total fiber area = {sec.gross_area:.2f} in^2 "
          f"(gross {A_exact:.2f}; steel {As:.1f} subtracted from core)")

    # transformed EA from the initial tangent (probe in slight COMPRESSION:
    # concrete carries no tension, so a tensile probe would show only steel)
    _, ks = sec.get_response(np.array([-1.0e-8, 0.0]))
    EA = ks[0, 0]
    core_r = 0.5 * D - cover
    A_core_conc = math.pi * core_r**2 - As
    A_cover = math.pi * ((0.5 * D) ** 2 - core_r**2)
    EA_hand = Ec * (A_core_conc + A_cover) + Es * As
    print(f"  EA (fiber sum) = {EA:,.0f} kip   EA (by hand) = {EA_hand:,.0f} kip"
          f"   -> match {EA/EA_hand:.4f}")

    # --- 3. monotonic axial compression preview ---------------------------
    print(f"\nMonotonic axial compression (N vs axial strain):")
    print(f"{'eps_a':>12}{'N (kip)':>14}")
    print("-" * 26)
    N_at_2400 = None
    eps_prev = N_prev = 0.0
    for eps in np.linspace(0.0, -0.006, 61):
        s_vec, _ = sec.get_response(np.array([eps, 0.0]))
        N = s_vec[0]
        sec.commit_state()
        # capture strain where |N| crosses the 2400 kip preload
        if N_at_2400 is None and -N >= 2400.0:
            frac = (-2400.0 - N_prev) / (N - N_prev) if N != N_prev else 0.0
            N_at_2400 = eps_prev + frac * (eps - eps_prev)
        eps_prev, N_prev = eps, N
        if abs(round(eps / -0.001) * -0.001 - eps) < 1e-9:  # ~ every 0.001
            print(f"{eps:>12.5f}{N:>14.1f}")

    print(f"\nWorking point: axial strain at P = 2400 kip preload "
          f"= {N_at_2400:.3e}" if N_at_2400 else "\n(2400 kip not reached)")
    print("(small and elastic - axial-load ratio ~8.7% of f'c*Ag, as expected)")

    print("\n" + "=" * 72)
    print("P1 closed: circular fiber builders reproduce the benchmark section.")
    print("Next: P2 (Mander confinement pre-processor) to derive the core law.")
    print("=" * 72)


if __name__ == "__main__":
    main()
