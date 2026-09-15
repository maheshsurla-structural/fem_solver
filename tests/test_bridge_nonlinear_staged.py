"""Nonlinear staged cable-stayed erection (bridge plan T2.2, second half).

Validates the corotational active-set Newton driver against closed forms:
axial elongation, taut-string transverse deflection (geometric stiffness +
pretension), Ernst sag reduction, stress-free birth in the deformed geometry,
removal redistribution, and the linear small-displacement limit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from femsolver.bridges import ernst_equivalent_modulus  # noqa: E402
from femsolver.bridges.nonlinear_staged import (  # noqa: E402
    CableSegment, ErectionStage, NonlinearStagedErection)

E = 2.0e11
A = 0.01


# --------------------------------------------------------------- axial (linear)
def test_axial_elongation_matches_closed_form():
    """A horizontal member pulled axially: u = F L /(E A), tension = F."""
    nodes = {1: (0.0, 0.0), 2: (10.0, 0.0)}
    seg = [CableSegment(1, 1, 2, E, A)]
    stages = [ErectionStage("pull", add=[1], loads={2: (1.0e6, 0.0)})]
    # node 1 pinned; node 2 is a horizontal roller (y fixed → no mechanism)
    supports = {1: (True, True), 2: (False, True)}
    res = NonlinearStagedErection(nodes, seg, stages, supports=supports,
                                  initial_active=[]).run()
    assert all(res.converged)
    ux = res.displacements[2][0]
    assert ux == pytest.approx(1.0e6 * 10.0 / (E * A), rel=1e-6)   # 5e-3
    assert res.tensions[1] == pytest.approx(1.0e6, rel=1e-6)


def test_two_bar_truss_matches_linear_hand_calc():
    """Two bars from two pinned supports to a loaded apex — small load, so the
    nonlinear solve reproduces the linear pin-jointed result."""
    #   supports at (-3,4) and (3,4); apex at (0,0); vertical load down.
    nodes = {1: (-3.0, 4.0), 2: (3.0, 4.0), 3: (0.0, 0.0)}
    seg = [CableSegment(1, 1, 3, E, A), CableSegment(2, 2, 3, E, A)]
    stages = [ErectionStage("load", add=[1, 2], loads={3: (0.0, -1.0e3)})]
    supports = {1: (True, True), 2: (True, True)}
    res = NonlinearStagedErection(nodes, seg, stages, supports=supports).run()
    assert all(res.converged)
    # symmetric → both bars equal; vertical equilibrium: 2 N cosθ = P,
    # cosθ = 4/5 (bar length 5, vertical component 4). N = P/(2·0.8) = 625 N
    assert res.tensions[1] == pytest.approx(625.0, rel=1e-3)
    assert res.tensions[2] == pytest.approx(625.0, rel=1e-3)


# ------------------------------------------------- taut string (geometric K)
def test_taut_string_transverse_deflection():
    """A pretensioned straight cable with a central transverse load deflects
    δ = P L /(4 T) — pure geometric (string) stiffness."""
    L, T0, P = 10.0, 1.0e5, 100.0
    nodes = {1: (0.0, 0.0), 2: (L / 2, 0.0), 3: (L, 0.0)}
    seg = [CableSegment(1, 1, 2, E, A), CableSegment(2, 2, 3, E, A)]
    stages = [ErectionStage("stress", add=[1, 2],
                            pretension={1: T0, 2: T0}),
              ErectionStage("load", loads={2: (0.0, -P)})]
    supports = {1: (True, True), 3: (True, True)}
    res = NonlinearStagedErection(nodes, seg, stages, supports=supports).run()
    assert all(res.converged)
    # after stressing (stage 1) the string is straight and taut
    assert abs(res.stage_displacements[0][2][1]) < 1e-9
    assert res.tension_history[1][0] == pytest.approx(T0, rel=1e-6)
    # after the load: central deflection of a taut string
    delta = -res.displacements[2][1]
    assert delta == pytest.approx(P * L / (4.0 * T0), rel=2e-2)   # 2.5e-3


# ------------------------------------------------------------- Ernst sag
def test_ernst_sag_softens_axial_stiffness():
    """A heavy horizontal cable under axial tension elongates more than the
    bare rod, by exactly the Ernst equivalent-modulus factor."""
    L, F, gamma = 10.0, 1.0e6, 3464.0
    nodes = {1: (0.0, 0.0), 2: (L, 0.0)}
    supports = {1: (True, True), 2: (False, True)}
    # no sag
    bare = NonlinearStagedErection(
        nodes, [CableSegment(1, 1, 2, E, A)],
        [ErectionStage("pull", add=[1], loads={2: (F, 0.0)})],
        supports=supports).run()
    # with sag
    sag = NonlinearStagedErection(
        nodes, [CableSegment(1, 1, 2, E, A, gamma_eff=gamma)],
        [ErectionStage("pull", add=[1], loads={2: (F, 0.0)})],
        supports=supports).run()
    E_eff = ernst_equivalent_modulus(E=E, A=A, L_h=L, gamma_eff=gamma, T=F)
    assert E_eff < E
    assert sag.displacements[2][0] > bare.displacements[2][0]
    assert sag.displacements[2][0] == pytest.approx(
        F * L / (E_eff * A), rel=5e-3)
    assert sag.tensions[1] == pytest.approx(F, rel=1e-6)          # axial


# --------------------------------------------- stress-free birth in deformed geom
def _staged_stay_system():
    #   node 1 (0,0) pinned deck anchor; node 2 (10,0) free deck tip;
    #   node 3 (10,8) pinned pylon top over the tip; node 4 (4,8) pinned pylon.
    nodes = {1: (0.0, 0.0), 2: (10.0, 0.0), 3: (10.0, 8.0), 4: (4.0, 8.0)}
    seg = [CableSegment(1, 1, 2, E, A),          # deck chord (horizontal)
           CableSegment(2, 3, 2, E, A),          # stay A (vertical) — stage 1
           CableSegment(3, 4, 2, E, A)]          # stay B (raking) — stage 2
    supports = {1: (True, True), 3: (True, True), 4: (True, True)}
    return nodes, seg, supports


def test_born_segment_is_stress_free_at_birth():
    nodes, seg, supports = _staged_stay_system()
    stages = [
        ErectionStage("erect + load", add=[1, 2], loads={2: (0.0, -5.0e4)}),
        ErectionStage("add stay B", add=[3]),               # no new load
        ErectionStage("more load", loads={2: (0.0, -5.0e4)}),
    ]
    res = NonlinearStagedErection(nodes, seg, stages, supports=supports).run()
    assert all(res.converged)
    N_A_s1 = res.tension_history[2][0]
    N_B_birth = res.tension_history[3][1]                    # stay B at birth
    N_B_final = res.tension_history[3][2]
    assert N_A_s1 > 0.0                                      # stay A carries load
    # stay B is installed stress-free in the deflected geometry
    assert abs(N_B_birth) < 1e-6 * N_A_s1
    # once further load is applied, stay B picks up tension
    assert N_B_final > 0.0


# ------------------------------------------------------------ removal redistributes
def test_removal_redistributes_force():
    """A temporary stay carries load, then is removed — the permanent stay
    picks up the released force and equilibrium is maintained."""
    #   tip (2) held by two stays to two pylon anchors; remove one.
    nodes = {1: (0.0, 0.0), 2: (10.0, 0.0),
             3: (10.0, 8.0), 4: (6.0, 8.0)}
    seg = [CableSegment(1, 1, 2, E, A),          # deck chord
           CableSegment(2, 3, 2, E, A),          # permanent stay
           CableSegment(3, 4, 2, E, A)]          # temporary stay
    supports = {1: (True, True), 3: (True, True), 4: (True, True)}
    stages = [
        ErectionStage("both stays", add=[1, 2, 3], loads={2: (0.0, -8.0e4)}),
        ErectionStage("remove temp", remove=[3]),
    ]
    res = NonlinearStagedErection(nodes, seg, stages, supports=supports).run()
    assert all(res.converged)
    N_perm_before = res.tension_history[2][0]
    N_perm_after = res.tension_history[2][1]
    assert res.tension_history[3][0] > 0.0                   # temp carried load
    assert res.tension_history[3][1] is None                 # temp gone
    assert N_perm_after > N_perm_before                      # picked up the slack


# ------------------------------------------------------------ pretension held
def test_pretension_between_fixed_anchors():
    """Pretension a cable strung between two fixed anchors → tension = the
    pretension, no displacement (self-equilibrated)."""
    nodes = {1: (0.0, 0.0), 2: (10.0, 0.0)}
    seg = [CableSegment(1, 1, 2, E, A)]
    stages = [ErectionStage("stress", add=[1], pretension={1: 2.5e5})]
    supports = {1: (True, True), 2: (True, True)}
    res = NonlinearStagedErection(nodes, seg, stages, supports=supports).run()
    assert all(res.converged)
    assert res.tensions[1] == pytest.approx(2.5e5, rel=1e-9)
    assert abs(res.displacements[2][0]) < 1e-12


def test_tension_only_cable_goes_slack():
    """A tension-only cable carries no compression: in parallel with a regular
    (compression-capable) bar, a compressive push is taken entirely by the bar
    while the cable goes slack (N = 0)."""
    nodes = {1: (0.0, 0.0), 2: (10.0, 0.0)}
    seg = [CableSegment(1, 1, 2, E, A),                      # regular bar
           CableSegment(2, 1, 2, E, A, tension_only=True)]   # parallel cable
    # push node 2 toward node 1 (compression); the bar provides the stiffness
    stages = [ErectionStage("push", add=[1, 2], loads={2: (-1.0e5, 0.0)})]
    supports = {1: (True, True), 2: (False, True)}
    res = NonlinearStagedErection(nodes, seg, stages, supports=supports).run()
    assert all(res.converged)
    assert res.tensions[2] == pytest.approx(0.0, abs=1e-3)   # cable slack
    assert res.tensions[1] == pytest.approx(-1.0e5, rel=1e-3)  # bar compressed
    # a tension-only bar (seg 1 as regular) proves the bar can carry compression
    assert res.tensions[1] < 0.0
