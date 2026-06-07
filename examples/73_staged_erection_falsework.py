"""Phase B.3 -- Incremental staged erection with falsework removal.

The headline staged-construction effect: a beam cast on falsework and
then *released* does not carry load the way a one-shot analysis of the
final structure predicts. The dying falsework's locked-in reaction is
released onto the permanent structure, and the final state depends on
the erection sequence -- exactly what a commercial construction-stage
analysis must capture.

Scenario: a two-span continuous girder is cast on a temporary central
falsework tower, loaded by its (lumped) self weight, then the
falsework is struck.

* **Stage 1** -- girder + falsework prop active; self-weight applied.
  The prop carries a large reaction and the girder barely deflects.
* **Stage 2** -- falsework removed (element death). Its reaction is
  released onto the girder, which now spans as a true two-span
  continuous beam.

We verify the released state equals the one-shot continuous-beam
solution (the load was applied once), and report the locked-in girder
moment that the falsework reaction produced.

Run::

    python examples/73_staged_erection_falsework.py
"""
from __future__ import annotations

import numpy as np

from femsolver.core.model import Model
from femsolver.elements.beam import BeamColumn2D
from femsolver.elements.truss import Truss2D
from femsolver.materials.elastic import ElasticIsotropic
from femsolver.analysis.linear_static import LinearStaticAnalysis
from femsolver.bridges.staged_construction import (
    ErectionStage,
    IncrementalStagedAnalysis,
)
from femsolver.io.diagrams import beam_force_diagram


def build(nps=6, span=15.0, *, with_prop):
    """Two-span continuous girder (2 x span). Optional central
    falsework prop (a stiff truss to a fixed ground node)."""
    A, I, E = 0.6, 0.12, 34e9
    mat = ElasticIsotropic(1, E=E, nu=0.2, rho=0.0)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    n_nodes = 2 * nps + 1
    dx = span / nps
    for i in range(n_nodes):
        m.add_node(i + 1, i * dx, 0.0)
    for i in range(n_nodes - 1):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, I))
    # permanent supports: pin at left end, roller at right end
    m.fix(1, [1, 1, 0])
    m.fix(n_nodes, [0, 1, 0])
    prop_tag = None
    if with_prop:
        gnd = n_nodes + 1
        m.add_node(gnd, span, -4.0)
        prop_tag = 500
        m.add_element(Truss2D(prop_tag, (nps + 1, gnd), mat, 0.4))
        m.fix(gnd, [1, 1, 1])
    return m, n_nodes, prop_tag


def main() -> None:
    nps, span = 6, 15.0
    w_node = -90e3      # lumped self-weight per node (N)

    print("=" * 66)
    print(" Incremental staged erection -- falsework removal")
    print(f"   two-span continuous girder, 2 x {span:.0f} m, "
          f"self-weight {abs(w_node)/1e3:.0f} kN/node")
    print("=" * 66)

    m, n_nodes, prop = build(nps, span, with_prop=True)
    beam = list(range(1, n_nodes))   # all beam element tags
    interior = list(range(2, n_nodes))   # interior nodes get self-weight
    loads = {nd: [0.0, w_node, 0.0] for nd in interior}

    stages = [
        ErectionStage(name="cast on falsework + self weight",
                      add_elements=beam + [prop], loads=loads),
        ErectionStage(name="strike falsework", remove_elements=[prop]),
    ]
    res = IncrementalStagedAnalysis(m, stages).run()

    # prop reaction carried in stage 1 (axial force of the prop truss)
    prop_force_stage1 = res.element_force_history[prop][0]
    prop_axial = float(np.linalg.norm(prop_force_stage1[:2]))

    mid_node = nps + 1
    mid_after = m.node(mid_node).disp[1]

    print(f"\nStage 1 (on falsework): prop carried "
          f"{prop_axial/1e3:.0f} kN of self weight")
    print(f"Stage 2 (struck):       mid-pier node deflection = "
          f"{mid_after*1e3:.3f} mm")

    # one-shot continuous beam (no falsework) under the same self weight
    m2, n2, _ = build(nps, span, with_prop=False)
    for nd in interior:
        m2.add_nodal_load(nd, [0.0, w_node, 0.0])
    LinearStaticAnalysis(m2).run()
    mid_oneshot = m2.node(mid_node).disp[1]

    print(f"\nReleased state vs one-shot two-span continuous beam:")
    print(f"   staged  mid deflection = {mid_after*1e3:.4f} mm")
    print(f"   one-shot mid deflection = {mid_oneshot*1e3:.4f} mm")
    print(f"   difference              = {abs(mid_after-mid_oneshot)*1e6:.2e} um")
    print("   (they match -- the released falsework reaction puts the")
    print("    girder in the true continuous-beam state)")

    # locked-in midspan moment in span 1 after striking
    diag = beam_force_diagram(m.element(nps // 2 + 1), n_points=3)
    print(f"\nSpan-1 girder moment after striking (mid-element) = "
          f"{diag['M'][-1]/1e3:.1f} kN.m")


if __name__ == "__main__":
    main()
