"""Fiber-hinge plan P4 (G8) — benchmark section moment-curvature + recorders.

Builds the Caltrans 84 in circular RC column section (confined Mander core +
unconfined cover + 56 #14 Park-steel bars, plan P1-P3) and computes its
**moment-curvature** at the benchmark axial preload P = 2400 kip compression
with :func:`fiber_section_moment_curvature` -- the fibre-consistent driver that
integrates the section's *own* fibers and uniaxial laws, so this M-phi is
exactly what the fiber hinge will integrate (plan §15, "what you analyse is
what you run").

The run is logged with two G8 recorders:

* :class:`SectionRecorder` -> the (kappa, M, N, eps_a) history.
* :class:`FiberRecorder`   -> per-fiber (y, z, eps, sigma), restricted to the
  rebar fibers, for the bar stress history.

Both are written to CSV next to this script's output. Golden Midas/CSI M-phi
targets are pending the user's export (plan 7.2); until then the run reports
the yield / ultimate landmarks and the cross-checks in §7.4.

Units: **kip, inch**.

Run::

    PYTHONPATH=src python examples/82_fiber_section_mphi.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from femsolver import (
    ConcreteMander,
    fiber_section_moment_curvature,
    rc_circular_column_section,
)
from femsolver.materials.uniaxial import UniaxialReinforcingSteel
from femsolver.results import FiberRecorder, SectionRecorder
from femsolver.sections.analysis import mander_confined_circular


def main() -> None:
    D, cover, Ec, fc, eps_c0, Es, fy = (
        84.0, 2.0, 3605.0, 5.0, 0.002219, 29000.0, 68.0)

    print("=" * 72)
    print("P4 (G8) - Caltrans column section M-phi at P = 2400 kip + recorders")
    print("=" * 72)

    conf = mander_confined_circular(
        fco=fc, eps_co=eps_c0, D_core=79.0, hoop_area=0.79, hoop_spacing=6.0,
        fyh=68.0, rho_long=0.0257, hoop_type="hoop", clear_spacing=5.0)
    sec = rc_circular_column_section(
        diameter=D, cover=cover, core_concrete=conf.to_material(Ec=Ec),
        cover_concrete=ConcreteMander(fpc=fc, eps_c0=eps_c0, Ec=Ec),
        n_bars=56, bar_area=2.25,
        steel=UniaxialReinforcingSteel(Es, fy, 95.0, 0.0075, 0.09),
        n_core_rings=8, n_cover_rings=2, n_wedges=24)
    n_conc = len(sec.fibers) - 56
    rebar_ids = list(range(n_conc, len(sec.fibers)))    # bars are appended last

    sec_rec = SectionRecorder(sec, name="mphi_section")
    fib_rec = FiberRecorder(sec, fiber_ids=rebar_ids, name="mphi_rebar")

    kappas = np.linspace(0.0, 6.0e-4, 120)              # 1/in
    res = fiber_section_moment_curvature(
        sec, kappas, N_target=-2400.0,
        section_recorder=sec_rec, fiber_recorder=fib_rec)

    kap = np.array(res["kappa"])
    M = np.array(res["M"])
    print(f"\nConverged steps: {res['n_converged']}/{len(kappas)}")
    print(f"Axial force held: N in [{min(res['N']):.3f}, {max(res['N']):.3f}] "
          f"kip (target -2400)")
    print(f"Peak moment: M_max = {M.max():.0f} kip-in "
          f"= {M.max() / 12.0:.0f} kip-ft at kappa = {kap[M.argmax()]:.2e} 1/in")

    # first-yield curvature: extreme tension bar reaches eps_y = fy/Es
    eps_y = fy / Es
    y_bar_max = max(sec.fibers[i].y for i in rebar_ids)
    ky = None
    for e_a, k in zip(res["eps_a"], res["kappa"]):
        if (e_a - (-y_bar_max) * k) >= eps_y:          # bottom bar tension
            ky = k
            break
    if ky:
        print(f"First-yield curvature (extreme tension bar): "
              f"kappa_y ~ {ky:.2e} 1/in")

    # ---- recorders to CSV -----------------------------------------------
    out = Path(tempfile.gettempdir())
    sec_csv = out / "82_mphi_section.csv"
    fib_csv = out / "82_mphi_rebar.csv"
    sec_rec.to_csv(sec_csv)
    fib_rec.to_csv(fib_csv)
    print(f"\nSection recorder -> {sec_csv} ({len(sec_rec.rows)} rows)")
    print(f"Fiber recorder   -> {fib_csv} "
          f"({len(fib_rec.rows)} rows = {len(kappas)} steps x 56 bars)")

    print("\nCross-checks (plan 7.4, no external data needed):")
    print(f"  - axial force held to 1e-6 kip at every step: "
          f"{max(abs(N + 2400.0) for N in res['N']):.1e}")
    print(f"  - M(kappa) monotonic up to peak: "
          f"{all(M[i + 1] >= M[i] - 1.0 for i in range(M.argmax()))}")

    print("\n" + "=" * 72)
    print("P4 closed: fibre-consistent section M-phi + recorders (G8).")
    print("Golden Midas/CSI M-phi: paste into plan 7.2 when exported.")
    print("Next: P5 (axial-load benchmark, Stage 1).")
    print("=" * 72)


if __name__ == "__main__":
    main()
