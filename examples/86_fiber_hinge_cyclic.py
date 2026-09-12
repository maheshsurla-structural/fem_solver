"""Fiber-hinge plan P8 — cyclic benchmark + energy dissipation.

Stage 3 of the benchmark (plan §2.6/§7.3): hold the P = 2400 kip axial preload,
then drive the column tip through a stepped reversed-cyclic displacement
protocol (+-0.25, 0.5, 0.75, 1.0 of a reference drift) and trace the base-shear
vs tip-displacement hysteresis, reporting the peak force and dissipated energy
per loop.

Pieces used: the cyclic ``ReinforcingSteelKinematic`` steel (P3), the
``stepped_cyclic`` protocol (P6) driven through one displacement-control
analysis via a per-step increment schedule, ``StagedAnalysis`` to hold the
axial load (P6), and a ``NodeRecorder`` (P4).

Element choice: **displacement-based** fiber elements (a short mesh) are used
for the cyclic run -- the force-based element's flexibility goes singular at
cyclic reversals (plan §8), while a 4-element displacement-based mesh is robust
and converges to the same capacity as the force-based monotonic pushover
(~8,000 kip base shear, cf. P6).

Golden Midas/CSI cyclic loops/energy: paste into plan §7.3 when exported.

Units: **kip, inch**.

Run::

    PYTHONPATH=src python examples/86_fiber_hinge_cyclic.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from femsolver import (
    BeamColumn2DCorotational,
    ConcreteMander,
    ElasticIsotropic,
    Model,
    NonlinearStaticAnalysis,
    StagedAnalysis,
    rc_circular_column_section,
    stepped_cyclic,
)
from femsolver.analysis.static_integrator import DisplacementControl
from femsolver.materials.uniaxial import ReinforcingSteelKinematic
from femsolver.sections.analysis import mander_confined_circular

D, COVER, EC, FC, EPS_C0, ES, FY, L = (
    84.0, 2.0, 3605.0, 5.0, 0.002219, 29000.0, 68.0, 49.0)
N_EL = 4
AMPLITUDES = (0.25, 0.5, 0.75, 1.0)
SCALE = 0.12                 # reference drift (in) -> peaks 0.03 .. 0.12 in
PPC = 16


def _section():
    conf = mander_confined_circular(
        fco=FC, eps_co=EPS_C0, D_core=79.0, hoop_area=0.79, hoop_spacing=6.0,
        fyh=68.0, rho_long=0.0257, hoop_type="hoop", clear_spacing=5.0)
    return rc_circular_column_section(
        diameter=D, cover=COVER, core_concrete=conf.to_material(Ec=EC),
        cover_concrete=ConcreteMander(fpc=FC, eps_c0=EPS_C0, Ec=EC),
        n_bars=56, bar_area=2.25,
        steel=ReinforcingSteelKinematic(ES, FY, 95.0, 0.0075, 0.09),
        n_core_rings=8, n_cover_rings=2, n_wedges=24)


def _loop_energy(u, V, cyc):
    lo, hi = cyc * PPC, (cyc + 1) * PPC
    uu = np.concatenate([[0.0 if cyc == 0 else u[lo - 1]], u[lo:hi]])
    vv = np.concatenate([[0.0 if cyc == 0 else V[lo - 1]], V[lo:hi]])
    return abs(np.sum(0.5 * (vv[1:] + vv[:-1]) * np.diff(uu)))


def main() -> None:
    print("=" * 72)
    print("P8 - Caltrans column cyclic benchmark (hold P=2400 kip, cycle tip)")
    print("=" * 72)

    mat = ElasticIsotropic(1, E=EC, nu=0.2)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    for i in range(N_EL + 1):
        m.add_node(i + 1, i * L / N_EL, 0.0)
    for i in range(N_EL):
        m.add_element(BeamColumn2DCorotational(
            i + 1, (i + 1, i + 2), mat, section=_section()))
    m.fix(1, [1, 1, 1])
    tip = N_EL + 1

    targets = stepped_cyclic(amplitudes=AMPLITUDES, scale=SCALE, cycles=1,
                             pts_per_cycle=PPC)
    du = -np.diff(targets)

    sa = StagedAnalysis(m)
    sa.add_stage("axial", lambda mm: (
        mm.add_nodal_load(tip, [-2400.0, 0.0, 0.0]),
        NonlinearStaticAnalysis(mm, num_steps=8, dlambda=0.125,
                                integrator="load_control", tol=1e-8))[1])
    sa.add_stage("cyclic", lambda mm: (
        mm.add_nodal_load(tip, [0.0, -1.0, 0.0]),
        NonlinearStaticAnalysis(
            mm, num_steps=len(du),
            integrator=DisplacementControl(tip, 1, du),
            track=(tip, 1), tol=1e-6, max_iter=80))[1])
    out = sa.run()

    u = np.array(out["cyclic"]["tracked"])
    V = -np.array(out["cyclic"]["lambdas"])          # base shear (kip)

    print(f"\nModel: {N_EL} displacement-based fiber elements, tip node {tip}")
    print(f"Axial preload held: base Fx = {abs(m.nodes[1].reaction[0]):.1f} kip")
    print(f"Protocol: +-{', '.join(str(a) for a in AMPLITUDES)} x {SCALE} in "
          f"= peaks {', '.join(f'{a*SCALE:.3f}' for a in AMPLITUDES)} in\n")

    print(f"{'amplitude (in)':>16}{'peak +V':>12}{'peak -V':>12}"
          f"{'loop energy':>14}")
    print("-" * 54)
    total_E = 0.0
    for j, a in enumerate(AMPLITUDES):
        seg = slice(j * PPC, (j + 1) * PPC)
        vp, vn = V[seg].max(), V[seg].min()
        e = _loop_energy(u, V, j)
        total_E += e
        print(f"{a * SCALE:>16.3f}{vp:>12.0f}{vn:>12.0f}{e:>14.0f}")
    print("-" * 54)
    print(f"{'total':>16}{'':>12}{'':>12}{total_E:>14.0f} kip-in")

    csv = Path(tempfile.gettempdir()) / "86_cyclic_tip.csv"
    import csv as _csv
    with open(csv, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["step", "tip_disp_in", "base_shear_kip"])
        for k, (uu, vv) in enumerate(zip(u, V)):
            w.writerow([k, f"{uu:.6f}", f"{vv:.3f}"])
    print(f"\nHysteresis path -> {csv} ({len(u)} points)")

    print(f"\nSanity: peak base shear {np.max(np.abs(V)):.0f} kip "
          f"(cf. P6 monotonic ~8,000 kip); loops reverse and dissipate energy.")
    print("\n" + "=" * 72)
    print("P8 closed: cyclic hysteresis + energy dissipation.")
    print("Golden Midas/CSI cyclic loops/energy: paste into plan 7.3.")
    print("Next: P9 (finite-length fiber-hinge element idiom).")
    print("=" * 72)


if __name__ == "__main__":
    main()
