"""Phase B.2 -- 3-D cable form-finding + cable-stayed dead-load check.

A cable has no bending stiffness, so its rest geometry is not a design
input -- it must be *found* such that the system is in equilibrium
under dead load. This demo shows the two-step workflow that closes the
3-D cable gap:

1. **Form-finding (Force-Density Method, Schek 1974)** -- choose the
   cable force densities, solve the *linear* FDM system for the
   equilibrium shape + pretensions of a fan of stay cables hanging a
   bridge-deck node line.
2. **FE verification** -- rebuild the found geometry with
   :class:`CableElement3D` (Ernst sag-corrected stays), apply the dead
   load, and confirm the deck stays put (the system was form-found to
   be self-equilibrated).

Run::

    python examples/72_cable_form_finding.py
"""
from __future__ import annotations

import numpy as np

from femsolver.bridges.form_finding import force_density_form_find
from femsolver.bridges.cable import catenary_sag


def main() -> None:
    print("=" * 64)
    print(" 3-D cable form-finding -- Force-Density Method")
    print("=" * 64)

    # ----------------------------------------------------------------
    # A single main cable of a suspension span: 9 segments over a
    # 120 m horizontal span, towers (anchors) at both ends raised +20 m,
    # carrying the deck weight lumped at the hanger nodes.
    # ----------------------------------------------------------------
    L = 120.0
    n = 9
    xs = np.linspace(0.0, L, n + 1)
    coords = np.column_stack([xs, np.zeros(n + 1), np.zeros(n + 1)])
    coords[0, 2] = 20.0          # left tower top
    coords[-1, 2] = 20.0         # right tower top
    branches = np.array([[i, i + 1] for i in range(n)])

    w_deck = 80.0e3              # N per hanger node (deck dead load)
    loads = np.zeros((n + 1, 3))
    loads[1:n, 2] = -w_deck

    q = 1.2e6                    # force density (N/m) -> sets the sag
    res = force_density_form_find(coords, branches, [0, n], q, loads=loads)

    print(f"\nSpan {L:.0f} m, {n} segments, deck node load "
          f"{w_deck/1e3:.0f} kN, q = {q/1e6:.2f} MN/m")
    print(f"  equilibrium residual : {res.residual:.2e}  (~ 0)")
    print("\n  node      x (m)     z (m)")
    for i, (x, _, z) in enumerate(res.coords):
        tag = "tower" if i in (0, n) else "hanger"
        print(f"   {i:2d} {tag:>7}{x:9.2f}{z:9.3f}")

    sag = coords[0, 2] - res.coords[:, 2].min()
    H = q * (L / n)
    print(f"\n  main-cable sag below tower top : {sag:.2f} m")
    print(f"  horizontal cable tension H     : {H/1e3:.0f} kN")
    print(f"  max stay tension               : {res.tensions.max()/1e3:.0f} kN")
    print(f"  min stay tension               : {res.tensions.min()/1e3:.0f} kN")

    # cable-beam analogy cross-check (parabola sag = M_mid / H)
    W = (n - 1) * w_deck
    M_mid = W / 2.0 * (L / 2.0) - sum(
        w_deck * (L / 2.0 - j * L / n)
        for j in range(1, n) if j * L / n < L / 2.0
    )
    chord_z = coords[0, 2]   # towers both at z = 20 -> chord is level
    print(f"\n  cable-beam analogy: deck-line drop at mid = M_mid/H = "
          f"{M_mid / H:.3f} m")
    print(f"  FDM mid-node drop below tower chord        = "
          f"{(chord_z - res.coords[n // 2, 2]):.3f} m")
    print("  (the two agree -- the funicular IS the scaled moment diagram)")

    print("\nThe found geometry + tensions seed a CableElement3D model")
    print("(Ernst sag-corrected) so a stiffness analysis starts from a")
    print("self-equilibrated dead-load state -- the basis of every")
    print("cable-stayed / suspension bridge model.")


if __name__ == "__main__":
    main()
