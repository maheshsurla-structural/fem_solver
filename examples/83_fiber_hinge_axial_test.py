"""Fiber-hinge plan P5 — axial-load benchmark (Stage 1).

The benchmark's "Axial Load Test" isolates Stage 1: hold the P = 2400 kip
axial preload on the Caltrans 84 in circular RC column and validate the axial
force vs axial shortening/strain (plan §2.6/§7.1) before any moment is applied.

This example does both halves of that:

1. **FE model preload** — build the two-node column with a force-based fiber
   beam-column (:class:`ForceBeamColumn2DCorotational`) carrying the Caltrans
   fiber section (P1-P2 section, Park steel P3), apply P = 2400 kip under load
   control, and report the base reaction, tip shortening, and the transformed
   axial stiffness ``EA`` (cross-checked against ``Ec*A_concrete + Es*A_steel``).

2. **Section axial capacity** — impose axial strain on the section
   (``N = sum sigma*A``) to trace the full N-vs-strain curve, which the
   force-based element cannot follow past crushing (its flexibility goes
   singular). This is the pure material/section validation §7.1 describes:
   the capacity peaks near the confined-core strain and softens afterwards as
   the cover spalls and the confined core degrades. Logged with a
   :class:`SectionRecorder` (G8).

Golden Midas/CSI axial numbers: paste into plan §7.1 when exported.

Units: **kip, inch**.

Run::

    PYTHONPATH=src python examples/83_fiber_hinge_axial_test.py
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

from femsolver import (
    ConcreteMander,
    ElasticIsotropic,
    ForceBeamColumn2DCorotational,
    Model,
    NonlinearStaticAnalysis,
    rc_circular_column_section,
)
from femsolver.materials.uniaxial import UniaxialReinforcingSteel
from femsolver.results import SectionRecorder
from femsolver.sections.analysis import mander_confined_circular

D, COVER, EC, FC, EPS_C0, ES, FY, L = (
    84.0, 2.0, 3605.0, 5.0, 0.002219, 29000.0, 68.0, 49.0)
P_PRELOAD = 2400.0


def build_section():
    conf = mander_confined_circular(
        fco=FC, eps_co=EPS_C0, D_core=79.0, hoop_area=0.79, hoop_spacing=6.0,
        fyh=68.0, rho_long=0.0257, hoop_type="hoop", clear_spacing=5.0)
    return rc_circular_column_section(
        diameter=D, cover=COVER, core_concrete=conf.to_material(Ec=EC),
        cover_concrete=ConcreteMander(fpc=FC, eps_c0=EPS_C0, Ec=EC),
        n_bars=56, bar_area=2.25,
        steel=UniaxialReinforcingSteel(ES, FY, 95.0, 0.0075, 0.09),
        n_core_rings=8, n_cover_rings=2, n_wedges=24)


def main() -> None:
    print("=" * 72)
    print("P5 - Caltrans column axial-load benchmark (Stage 1)")
    print("=" * 72)

    # --- 1. FE model preload (load control) -------------------------------
    sec = build_section()
    mat = ElasticIsotropic(1, E=EC, nu=0.2)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    m.add_element(ForceBeamColumn2DCorotational(1, (1, 2), mat, section=sec))
    m.fix(1, [1, 1, 1])
    m.fix(2, [0, 1, 1])                                  # only axial free
    m.add_nodal_load(2, [-P_PRELOAD, 0.0, 0.0])
    res = NonlinearStaticAnalysis(
        m, num_steps=10, dlambda=0.1, integrator="load_control",
        track=(2, 0), tol=1e-8).run()

    u = res["tracked"][-1]
    strain = u / L
    As = 56 * 2.25
    core_r = 0.5 * D - COVER
    A_core = math.pi * core_r ** 2 - As
    A_cover = math.pi * ((0.5 * D) ** 2 - core_r ** 2)
    EA_hand = EC * (A_core + A_cover) + ES * As
    EA_model = P_PRELOAD / (-strain)
    Ag = math.pi * D ** 2 / 4.0
    print(f"\nPreload applied: {P_PRELOAD:.0f} kip "
          f"({100 * P_PRELOAD / (FC * Ag):.1f}% of f'c*Ag)")
    print(f"  base reaction (node 1)  = {m.nodes[1].reaction[0]:.1f} kip")
    print(f"  tip shortening          = {u:.5f} in  (strain {strain:.3e})")
    print(f"  EA  fiber model         = {EA_model:,.0f} kip")
    print(f"  EA  Ec*Ac + Es*As       = {EA_hand:,.0f} kip"
          f"   -> match {EA_model / EA_hand:.4f}")

    # --- 2. section axial capacity curve (imposed strain) -----------------
    sec2 = build_section()
    rec = SectionRecorder(sec2, name="axial")
    eps = np.linspace(0.0, -0.02, 200)
    N = []
    for i, e in enumerate(eps):
        s, _ = sec2.get_response(np.array([e, 0.0]))
        sec2.commit_state()
        rec.record(np.array([e, 0.0]), step=i)
        N.append(-float(s[0]))                          # compression positive
    N = np.array(N)
    ip = int(N.argmax())
    print("\nSection axial capacity (N = sum sigma*A, imposed strain):")
    print(f"  peak N     = {N.max():,.0f} kip at strain {-eps[ip]:.4f} "
          f"({100 * N.max() / (FC * Ag):.0f}% of f'c*Ag)")
    print(f"  N at 0.005 = {N[np.argmin(abs(eps + 0.005))]:,.0f} kip "
          f"(cover spalling underway)")
    print(f"  N at 0.020 = {N[-1]:,.0f} kip  (confined core, softened)")

    csv = Path(tempfile.gettempdir()) / "83_axial_capacity.csv"
    rec.to_csv(csv)
    print(f"\nSection recorder -> {csv} ({len(rec.rows)} rows)")

    print("\nCross-checks (plan 7.4):")
    print(f"  - EA fiber model vs hand: {EA_model / EA_hand:.4f}")
    print(f"  - working point strain < eps_c0: "
          f"{abs(strain) < EPS_C0} ({abs(strain):.2e} < {EPS_C0})")
    print(f"  - capacity softens after peak: {N[-1] < N.max()}")

    print("\n" + "=" * 72)
    print("P5 closed: axial-load benchmark model + section capacity.")
    print("Golden Midas/CSI axial: paste into plan 7.1 when exported.")
    print("Next: P6 (staged analysis + protocols; monotonic pushover).")
    print("=" * 72)


if __name__ == "__main__":
    main()
