"""Nonlinear dynamic time-history on the fiber model (plan §16 C3).

``nonlinear.run_time_history`` applies rigid-base ground acceleration to the
fiber column (``-M·ι·ü_g`` inertia load), with Rayleigh damping from the modal
frequencies and Newmark + Newton integration, and returns the monitored DOF's
response history. Qt-free (compute layer), so it runs in the main env.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "desktop"))

import nonlinear as NL                                         # noqa: E402
from project import Material, Member, Node, Project, Section   # noqa: E402


def _column(D=0.6, L=3.0):
    import section_gui_core as core
    spec = core.Spec(kind="Circular", D=D, fc=35e6, fy=500e6)
    p = Project()
    p.nodes = [Node(1, 0.0, 0.0, supports=(1, 1, 1)), Node(2, 0.0, L)]
    p.sections = [Section(id=1, name="col", A=math.pi * D * D / 4,
                          Iz=math.pi * D**4 / 64,
                          gsd_spec=dataclasses.asdict(spec))]
    p.materials = [Material(1, "conc", E=30e9, nu=0.2)]
    p.members = [Member(1, 1, 2, 1, 1)]
    return p


def _sine(g, freq, dur=2.0, dt=0.02):
    t = np.arange(0.0, dur, dt)
    return g * 9.81 * np.sin(2.0 * math.pi * freq * t), dt


def test_time_history_runs_and_returns_histories():
    p = _column()
    ag, dt = _sine(0.3, 2.0)
    r = NL.run_time_history(p, ag, dt, control_node=2, control_dof=0,
                            direction="x", density=2400.0)
    assert r["protocol"] == "time_history"
    assert r["dt"] == dt
    assert len(r["times"]) == len(ag)             # n = len(ag) - 1 -> n+1 points
    for key in ("disp", "velocity", "acceleration"):
        assert len(r[key]) == len(ag)
        assert all(math.isfinite(x) for x in r[key])
    assert r["peak_disp"] > 0.0                    # base motion drives response
    assert r["times"][-1] == pytest.approx(dt * (len(ag) - 1), rel=1e-9)


def test_time_history_scales_with_excitation():
    p = _column()
    ag, dt = _sine(0.2, 2.0)
    small = NL.run_time_history(p, ag, dt, control_node=2, control_dof=0,
                               direction="x", density=2400.0)
    big = NL.run_time_history(p, 2.0 * ag, dt, control_node=2, control_dof=0,
                              direction="x", density=2400.0)
    assert big["peak_disp"] > small["peak_disp"]   # stronger motion -> more drift


def test_time_history_rejects_short_record():
    p = _column()
    with pytest.raises(ValueError):
        NL.run_time_history(p, [0.1], 0.02, control_node=2, control_dof=0)
