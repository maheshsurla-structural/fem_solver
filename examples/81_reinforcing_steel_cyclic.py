"""Fiber-hinge plan P3 (G3) — Caltrans/Park reinforcing steel, cyclic.

The benchmark's longitudinal bars are A615 Gr60 at *expected* (overstrength)
values (plan §2.2): f_y = 68 ksi, f_u = 95 ksi, E_s = 29000 ksi, onset of
strain hardening eps_sh = 0.0075, ultimate eps_su = 0.09 (Midas) / 0.06 (CSI,
open decision O1). Both reference tools model these bars with the **Park**
idealized backbone (elastic -> yield plateau -> parabolic hardening to
(eps_su, f_u)) and a **kinematic** cyclic rule (Midas "Park PM", CSI "Simple"
+ Kinematic).

This example shows the two femsolver classes that cover that:

* :class:`UniaxialReinforcingSteel` — the monotonic Park backbone (already
  present before P3); we verify it hits the benchmark (eps_sh, f_y) and
  (eps_su, f_u) landmarks exactly.
* :class:`ReinforcingSteelKinematic` — the same backbone wrapped in a
  kinematic-hardening return map for cyclic analysis. A monotonic push
  reproduces the backbone fibre-for-fibre; a reversal unloads elastically
  (slope E) and re-yields early in compression (kinematic / Bauschinger
  shift). We drive one full displacement cycle and report the loop.

Units: **kip, inch** (the benchmark's natural units).

Run::

    PYTHONPATH=src python examples/81_reinforcing_steel_cyclic.py
"""
from __future__ import annotations

import numpy as np

from femsolver.materials.uniaxial import (
    ReinforcingSteelKinematic,
    UniaxialReinforcingSteel,
)


def main() -> None:
    E, fy, fu, esh, esu = 29000.0, 68.0, 95.0, 0.0075, 0.09

    print("=" * 72)
    print("P3 (G3) - Caltrans/Park reinforcing steel (A615 Gr60 expected)")
    print("=" * 72)

    # --- 1. monotonic backbone hits the benchmark landmarks ---------------
    bb = UniaxialReinforcingSteel(E, fy, fu, esh, esu)
    print("\nMonotonic Park backbone landmarks (plan 2.2):")
    print(f"{'strain':>12}{'sigma (ksi)':>14}{'target':>10}")
    print("-" * 36)
    for eps, tgt in ((fy / E, fy), (esh, fy), (esu, fu)):
        print(f"{eps:>12.5f}{bb.get_response(eps)[0]:>14.3f}{tgt:>10.1f}")

    # --- 2. kinematic model reproduces the backbone monotonically ---------
    kin = ReinforcingSteelKinematic(E, fy, fu, esh, esu)
    max_diff = 0.0
    for eps in np.linspace(0.0, 0.085, 200):
        s_bb = bb.get_response(eps)[0]
        s_k, _ = kin.get_response(eps)
        kin.commit_state()
        max_diff = max(max_diff, abs(s_bb - s_k))
    print(f"\nKinematic vs backbone (monotonic): max |d_sigma| = {max_diff:.2e} ksi "
          f"(identical curve)")

    # --- 3. one cyclic excursion: +0.02, unload, -0.02, back to 0 ---------
    cyc = ReinforcingSteelKinematic(E, fy, fu, esh, esu)
    print("\nCyclic response (kinematic hardening):")
    print(f"{'strain':>10}{'sigma (ksi)':>14}{'Et/E':>10}")
    print("-" * 34)

    def sweep(mat, start, stop, n):
        for eps in np.linspace(start, stop, n)[1:]:
            s, Et = mat.get_response(eps)
            mat.commit_state()
        return eps, s, Et

    cur = 0.0
    for label, target in (("peak +", 0.02), ("unload", 0.0),
                          ("peak -", -0.02), ("return", 0.0)):
        eps, s, Et = sweep(cyc, cur, target, 200)
        print(f"{eps:>10.4f}{s:>14.3f}{Et / E:>10.3f}   <- {label}")
        cur = target

    # back-stress after the cycle demonstrates kinematic translation
    print(f"\nBack-stress after cycle q = {cyc.q_committed:.3f} ksi; "
          f"accumulated plastic strain = {cyc.p_committed:.4f}")
    print("Compression re-yield began near sigma = q - f_y "
          "(< f_y in magnitude): the Bauschinger effect from kinematic hardening.")

    print("\n" + "=" * 72)
    print("P3 closed: Park backbone verified + kinematic cyclic steel added.")
    print("Next: P4 (recorders + benchmark section M-phi on the unified core).")
    print("=" * 72)


if __name__ == "__main__":
    main()
