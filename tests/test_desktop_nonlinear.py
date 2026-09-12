"""Nonlinear-run subsystem — desktop model plumbing + pushover (plan §14 GUI-5).

The "background": compile a project's GSD column section into an inelastic
FiberSection2D + displacement-based fiber beam-column and run a
displacement-controlled pushover (optionally holding an axial preload). Pure
engine + section_gui_core reuse; headless (no Qt).
"""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))                 # section_gui_core (repo root)
sys.path.insert(0, str(_ROOT / "desktop"))     # flat desktop imports

import nonlinear as NL                          # noqa: E402
from project import Material, Member, Node, Project, Section  # noqa: E402


def _circular_col_project(*, D=0.6, L=3.0, fc=35e6, fy=500e6):
    import section_gui_core as core
    spec = core.Spec(kind="Circular", D=D, fc=fc, fy=fy)
    gsd = dataclasses.asdict(spec)
    A = math.pi * D * D / 4.0
    Iz = math.pi * D**4 / 64.0
    p = Project()
    p.nodes = [Node(1, 0.0, 0.0, supports=(1, 1, 1)), Node(2, L, 0.0)]
    p.sections = [Section(id=1, name="col", A=A, Iz=Iz, gsd_spec=gsd)]
    p.materials = [Material(1, "conc", E=30e9, nu=0.2)]
    p.members = [Member(1, 1, 2, 1, 1)]
    return p


# ------------------------------------------------------ fiber_section_from_spec

def test_fiber_section_from_spec_circular():
    import section_gui_core as core
    spec = core.Spec(kind="Circular", D=0.6, fc=35e6, fy=500e6)
    fs = NL.fiber_section_from_spec(spec)
    # concrete cells (polar mesh) sum to ~pi R^2; plus discrete rebar fibers
    conc_area = sum(f.area for f in fs.fibers if f.area > 1e-6)
    assert fs.gross_area > 0
    # steel fibers have their own material objects (independent state)
    mats = {id(f.material) for f in fs.fibers}
    assert len(mats) == len(fs.fibers)               # every fiber cloned
    assert len(fs.fibers) > 50


# ------------------------------------------------------ build_nonlinear_model

def test_build_nonlinear_model_uses_fiber_element():
    from femsolver.elements.beam_corot import BeamColumn2DCorotational
    p = _circular_col_project()
    m = NL.build_nonlinear_model(p)
    els = list(m.elements.values()) if isinstance(m.elements, dict) else list(m.elements)
    assert len(els) == 1
    assert isinstance(els[0], BeamColumn2DCorotational)
    assert len(els[0].sections[0].fibers) > 50


def test_build_nonlinear_model_requires_a_fiber_section():
    p = Project()
    p.nodes = [Node(1, 0, 0, supports=(1, 1, 1)), Node(2, 1, 0)]
    p.sections = [Section(id=1, name="s", A=1.0, Iz=0.08)]     # no gsd_spec
    p.materials = [Material(1, "s", E=2e11, nu=0.3)]
    p.members = [Member(1, 1, 2, 1, 1)]
    with pytest.raises(ValueError):
        NL.build_nonlinear_model(p)


# ------------------------------------------------------ run_pushover

def test_pushover_yields():
    p = _circular_col_project()
    res = NL.run_pushover(p, control_node=2, control_dof=1, target=0.05,
                          n_steps=25)
    d = np.array(res["disp"])
    V = np.array(res["shear"])
    assert len(d) > 10
    assert d[-1] == pytest.approx(0.05, rel=0.05)
    assert V.max() > 0
    assert np.all(np.diff(d) >= -1e-9)               # monotonic push
    # post-yield softening: late tangent well below the initial
    k0 = (V[2] - V[1]) / (d[2] - d[1])
    k1 = (V[-1] - V[-3]) / (d[-1] - d[-3])
    assert k1 < 0.6 * k0


def test_pushover_capture_fibers():
    """GUI-6: per-step fiber snapshots at the base section — stresses grow to
    steel yield and the frame count matches the curve."""
    p = _circular_col_project()
    res = NL.run_pushover(p, control_node=2, control_dof=1, target=0.05,
                          n_steps=15, capture_fibers=True)
    fr = res["fiber_frames"]
    assert len(fr) == len(res["disp"])
    assert len(fr[0]) > 50                          # (y, z, sigma, strain) tuples
    s0 = max(abs(t[2]) for t in fr[0])
    sL = max(abs(t[2]) for t in fr[-1])
    assert sL > s0                                  # response grows (yielding)
    assert sL > 4.0e8                               # steel reaches ~yield stress
    # strain field is the plane-section gradient: both signs present at the end
    strains = [t[3] for t in fr[-1]]
    assert min(strains) < 0 < max(strains)


def test_pushover_with_axial_preload_runs():
    p = _circular_col_project()
    res = NL.run_pushover(p, control_node=2, control_dof=1, target=0.04,
                          n_steps=20, axial=2.0e6, axial_node=2, axial_dof=0)
    assert len(res["disp"]) > 5
    assert max(res["shear"]) > 0


def test_pushover_on_step_progress_and_cancel():
    """on_step fires per push step; should_cancel stops early (keeps the
    curve so far) — the hooks the threaded GUI-5 runner uses."""
    p = _circular_col_project()
    seen = []

    def on_step(info):
        seen.append(info["step"])
        assert info["disp"] >= 0.0 and info["shear"] >= 0.0

    res = NL.run_pushover(p, control_node=2, control_dof=1, target=0.06,
                          n_steps=30, on_step=on_step,
                          should_cancel=lambda: len(seen) >= 6)
    assert seen[:3] == [1, 2, 3]                 # progress fired in order
    assert len(res["disp"]) == 6                 # cancelled after 6 steps


# ------------------------------------------------------ engine step_callback

def test_engine_step_callback_fires_and_cancels():
    from femsolver import (BeamColumn2D, ElasticIsotropic,
                           NonlinearStaticAnalysis, Model)
    m = Model(ndm=2, ndf=3)
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m.add_material(mat)
    m.add_node(1, 0, 0)
    m.add_node(2, 1, 0)
    m.fix(1, [1, 1, 1])
    m.add_element(BeamColumn2D(1, (1, 2), mat, 0.01, 1e-4))
    m.add_nodal_load(2, [0.0, -1000.0, 0.0])
    calls = []
    NonlinearStaticAnalysis(
        m, num_steps=10, dlambda=0.1, integrator="load_control",
        step_callback=lambda info: (calls.append(info["step"]),
                                    len(calls) < 4)[1]).run()
    assert calls == [1, 2, 3, 4]                 # returned False at step 4 -> stop
