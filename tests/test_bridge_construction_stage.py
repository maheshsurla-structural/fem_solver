"""Construction-stage bridge workflows (bridge plan T1.4).

Camber / geometry control, tendon stressing sequence, and composite
(wet → hardened) staging, on the incremental staged-erection core.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from femsolver import (BeamColumn2D, ElasticIsotropic,  # noqa: E402
                       LinearStaticAnalysis, Model)
from femsolver.bridges import (ErectionStage,  # noqa: E402
                               IncrementalStagedAnalysis, Tendon,
                               merge_stage_loads, staged_camber,
                               tendon_stage_loads)

E, A = 3.0e10, 0.5


def _ss_beam(n, L=20.0, Iz=0.04):
    m = Model(ndm=2, ndf=3)
    mat = ElasticIsotropic(1, E=E, nu=0.2)
    m.add_material(mat)
    for i in range(n + 1):
        m.add_node(i + 1, i * L / n, 0.0)
    for i in range(n):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, Iz))
    m.fix(1, [1, 1, 0])
    m.fix(n + 1, [0, 1, 0])
    return m


# ------------------------------------------------------------ load helpers
def test_merge_stage_loads_sums_overlapping_nodes():
    merged = merge_stage_loads({1: [1.0, 2.0, 0.0]},
                               {1: [0.0, 3.0, 0.0], 2: [5.0, 0.0, 0.0]})
    assert merged[1] == [1.0, 5.0, 0.0]
    assert merged[2] == [5.0, 0.0, 0.0]


def test_tendon_stage_loads_leaves_model_untouched():
    n = 8
    m = _ss_beam(n)
    m.number_dofs()
    before = {t: nd._load.copy() for t, nd in m.nodes.items()}
    tend = Tendon(nodes=list(range(1, n + 2)),
                  eccentricity=[-0.2] * (n + 1), area=2e-3,
                  jacking_force=2e6)
    loads = tendon_stage_loads(tend, m)
    assert loads                                       # non-empty
    for t, nd in m.nodes.items():
        assert np.allclose(nd._load, before[t])        # model unchanged


def test_tendon_at_stage_equals_complete_structure():
    n = 8
    nodes = list(range(1, n + 2))
    tend = Tendon(nodes=nodes, eccentricity=[-0.2] * (n + 1),
                  area=2e-3, jacking_force=2e6)
    # reference: tendon applied to the complete structure
    m = _ss_beam(n)
    m.number_dofs()
    m.clear_loads()
    tend.apply_to(m)
    LinearStaticAnalysis(m).run()
    u_ref = np.array([m.node(t).disp[1] for t in nodes])
    # staged: tendon stressed at the (single) stage
    m2 = _ss_beam(n)
    m2.number_dofs()
    res = IncrementalStagedAnalysis(
        m2, [ErectionStage("stress", add_elements=list(range(1, n + 1)),
                           loads=tendon_stage_loads(tend, m2))]).run()
    u_stg = np.array([m2.node(t).disp[1] for t in nodes])
    assert np.max(np.abs(u_ref - u_stg)) < 1e-12


# ------------------------------------------------------------ camber
def test_camber_single_stage_matches_direct_solve():
    n = 8
    mid = n // 2 + 1
    # direct
    m = _ss_beam(n)
    m.number_dofs()
    m.add_nodal_load(mid, [0, -1e5, 0])
    LinearStaticAnalysis(m).run()
    d_direct = m.node(mid).disp[1]
    # staged
    m2 = _ss_beam(n)
    stages = [ErectionStage("all", add_elements=list(range(1, n + 1)),
                            loads={mid: [0, -1e5, 0]})]
    res = IncrementalStagedAnalysis(m2, stages).run()
    cam = staged_camber(res, m2, stages, dof=1)
    i = cam.nodes.index(mid)
    assert cam.final_deflection[i] == pytest.approx(d_direct, rel=1e-9)
    assert cam.final_camber[i] == pytest.approx(-d_direct, rel=1e-9)


def test_camber_two_segment_history_and_birth():
    n, L = 8, 16.0
    m = Model(ndm=2, ndf=3)
    mat = ElasticIsotropic(1, E=E, nu=0.2)
    m.add_material(mat)
    for i in range(n + 1):
        m.add_node(i + 1, i * L / n, 0.0)
    for i in range(n):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, 0.04))
    m.fix(1, [1, 1, 1])                                # cantilever
    half = n // 2
    stages = [
        ErectionStage("segA", add_elements=list(range(1, half + 1)),
                      loads={half + 1: [0, -5e4, 0]}),
        ErectionStage("segB", add_elements=list(range(half + 1, n + 1)),
                      loads={n + 1: [0, -5e4, 0]}),
    ]
    res = IncrementalStagedAnalysis(m, stages).run()
    cam = staged_camber(res, m, stages, dof=1)
    tip = cam.nodes.index(n + 1)
    assert cam.birth_stage[tip] == 1                   # born in the 2nd stage
    assert cam.stage_deflection[0, tip] == 0.0         # 0 before it is cast
    assert cam.stage_deflection[1, tip] != 0.0         # deflects once cast
    assert cam.final_camber[tip] == pytest.approx(-cam.final_deflection[tip])


# ------------------------------------------------------------ composite
def test_composite_wet_then_hardened_staging():
    """Wet-concrete load acts on the bare girder; superimposed dead load acts
    on the composite section (girder + deck born at hardening)."""
    n, L, mid, P = 8, 20.0, 5, 1e5
    Ig, Id = 0.04, 0.06
    m = Model(ndm=2, ndf=3)
    mat = ElasticIsotropic(1, E=E, nu=0.2)
    m.add_material(mat)
    for i in range(n + 1):
        m.add_node(i + 1, i * L / n, 0.0)
    for i in range(n):                                 # bare girder
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, Ig))
    for i in range(n):                                 # deck overlay (composite)
        m.add_element(BeamColumn2D(100 + i + 1, (i + 1, i + 2), mat, A, Id))
    m.fix(1, [1, 1, 0])
    m.fix(n + 1, [0, 1, 0])
    stages = [
        ErectionStage("wet", add_elements=list(range(1, n + 1)),
                      loads={mid: [0, -P, 0]}),
        ErectionStage("sdl", add_elements=list(range(101, 101 + n)),
                      loads={mid: [0, -P, 0]}),
    ]
    res = IncrementalStagedAnalysis(m, stages).run()
    eq = int(m.node(mid).eqn[1])
    wet = res.u_increments[0][eq]
    sdl = res.u_increments[1][eq]
    assert wet == pytest.approx(-P * L ** 3 / (48 * E * Ig), rel=1e-6)
    assert sdl == pytest.approx(-P * L ** 3 / (48 * E * (Ig + Id)), rel=1e-6)
    assert abs(sdl) < abs(wet)                          # composite is stiffer
