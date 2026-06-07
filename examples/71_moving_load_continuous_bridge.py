"""Phase B.1 -- General moving-load / influence-line engine.

Demonstrates the capability that the closed-form simple-span helpers
could never provide: **influence lines built directly from the
finite-element model of an arbitrary structure**, here a 3-span
continuous bridge girder.

Pipeline:

1. Build a 3-span continuous girder (60 m: 20 + 20 + 20) as a meshed
   beam-column model.
2. Define the traffic lane (the ordered run of girder nodes).
3. In a SINGLE lane traversal, extract influence lines for:
   * sagging moment at the centre of span 1,
   * hogging moment over the first interior pier,
   * the interior pier reaction.
4. Run the AASHTO HL-93 live load (truck / tandem + lane, with the
   33 % dynamic load allowance) over each influence line to get the
   governing maximum and minimum design effects.

The hogging-moment and pier-reaction influence lines span all three
spans with the characteristic sign reversals of a continuous
structure -- impossible to capture with a single-span closed form.

Run::

    python examples/71_moving_load_continuous_bridge.py
"""
from __future__ import annotations

import numpy as np

from femsolver.core.model import Model
from femsolver.elements.beam import BeamColumn2D
from femsolver.materials.elastic import ElasticIsotropic
from femsolver.bridges.moving_load import (
    BeamForce,
    InfluenceLineEngine,
    Lane,
    Reaction,
    aashto_hl93_envelope,
)


def build_three_span(span=20.0, n_per_span=20, *, E=34.0e9):
    """3 equal continuous spans on 4 supports (pier reactions vertical)."""
    A, I = 0.8, 0.20            # ~deep box-girder strip
    mat = ElasticIsotropic(1, E=E, nu=0.2, rho=2500.0)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)

    n_nodes = 3 * n_per_span + 1
    dx = span / n_per_span
    for i in range(n_nodes):
        m.add_node(i + 1, i * dx, 0.0)
    for i in range(n_nodes - 1):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, I))

    # supports at x = 0, span, 2*span, 3*span
    support_nodes = [1, n_per_span + 1, 2 * n_per_span + 1, 3 * n_per_span + 1]
    m.fix(support_nodes[0], [1, 1, 0])     # pin
    for s in support_nodes[1:]:
        m.fix(s, [0, 1, 0])                # rollers
    return m, n_nodes, n_per_span, support_nodes


def main() -> None:
    span = 20.0
    nps = 20
    m, n_nodes, nps, supports = build_three_span(span=span, n_per_span=nps)

    print("=" * 66)
    print(" General moving-load engine -- 3-span continuous girder")
    print(f"   spans: 3 x {span:.0f} m = {3*span:.0f} m, "
          f"{n_nodes} nodes, {n_nodes-1} elements")
    print("=" * 66)

    engine = InfluenceLineEngine(m)
    lane = Lane(node_tags=list(range(1, n_nodes + 1)), load_dof=1,
                name="traffic lane")

    # response locations
    mid_span1_elem = nps // 2                 # element whose node-j is at span-1 centre
    pier1_elem = nps                          # element ending at first interior pier
    pier1_node = nps + 1

    ils = engine.influence_lines(lane, {
        "M_span1":  BeamForce(element_tag=mid_span1_elem, component="M", end="j"),
        "M_pier1":  BeamForce(element_tag=pier1_elem, component="M", end="j"),
        "R_pier1":  Reaction(node_tag=pier1_node, dof=1),
    })

    # ---- influence-line character -------------------------------------
    il_mp = ils["M_pier1"]
    il_ms = ils["M_span1"]
    print("\nInfluence-line character (continuous action):")
    print(f"  span-1 sagging-moment IL : peak +{il_ms.max_value:8.3f} "
          f"(x={il_ms.max_station:5.1f} m),  dip {il_ms.min_value:8.3f} "
          f"(x={il_ms.min_station:5.1f} m)")
    print(f"  pier-1 hogging-moment IL : peak {il_mp.min_value:8.3f} "
          f"(x={il_mp.min_station:5.1f} m)   <- negative over the pier")
    print(f"  pier-1 reaction IL       : peak  {ils['R_pier1'].max_value:7.3f} "
          f"(x={ils['R_pier1'].max_station:5.1f} m)")

    # ---- HL-93 envelopes ----------------------------------------------
    print("\nAASHTO HL-93 design live-load envelope (IM = 33 %):")
    print(f"  {'response':<22}{'max':>16}{'min':>16}  governing")
    for name, unit, scale in [
        ("M_span1", "kN.m", 1e-3),
        ("M_pier1", "kN.m", 1e-3),
        ("R_pier1", "kN", 1e-3),
    ]:
        env = aashto_hl93_envelope(ils[name])
        print(f"  {name:<22}{env['max']*scale:>12.1f} {unit:<4}"
              f"{env['min']*scale:>12.1f} {unit:<4}"
              f"  {env['governing_max']}")

    print("\nNote: the pier-moment and pier-reaction envelopes draw load")
    print("from all three spans with sign reversals -- a genuinely")
    print("continuous-structure result, not a single-span approximation.")


if __name__ == "__main__":
    main()
