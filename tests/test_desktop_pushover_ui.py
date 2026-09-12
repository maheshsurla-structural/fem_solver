"""GUI-5 UI — nonlinear pushover dialog + threaded worker (plan §14).

Headless (offscreen). The worker's logic is exercised by calling ``run()``
directly (synchronous, no thread); Qt delivers same-thread signals inline, so
progress/cancel are observable without real threading.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

from project import Material, Member, Node, Project, Section  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    pytest.importorskip("matplotlib")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _gsd_column_project(*, D=0.6, L=3.0):
    import section_gui_core as core
    spec = core.Spec(kind="Circular", D=D, fc=35e6, fy=500e6)
    gsd = dataclasses.asdict(spec)
    A = math.pi * D * D / 4.0
    Iz = math.pi * D**4 / 64.0
    p = Project()
    p.nodes = [Node(1, 0.0, 0.0, supports=(1, 1, 1)), Node(2, L, 0.0)]
    p.sections = [Section(id=1, name="col", A=A, Iz=Iz, gsd_spec=gsd)]
    p.materials = [Material(1, "conc", E=30e9, nu=0.2)]
    p.members = [Member(1, 1, 2, 1, 1)]
    return p


def test_main_window_wires_pushover(qapp):
    from demo_model import demo_project
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(demo_project())
    assert hasattr(w, "act_pushover")
    assert callable(w.run_pushover_dialog)
    # demo has no fiber section -> guarded (no dialog, just a status message)
    w.run_pushover_dialog()


def test_worker_runs_and_streams(qapp):
    from pushover_dialog import PushoverWorker
    p = _gsd_column_project()
    steps, results = [], []
    wk = PushoverWorker(p, dict(control_node=2, control_dof=1, target=0.05,
                                n_steps=20))
    wk.progress.connect(lambda info: steps.append(info["step"]))
    wk.done.connect(lambda res: results.append(res))
    wk.run()                                    # synchronous (no thread)
    assert steps[:3] == [1, 2, 3]
    assert results and len(results[0]["disp"]) > 10
    assert max(results[0]["shear"]) > 0


def test_worker_cancel(qapp):
    from pushover_dialog import PushoverWorker
    p = _gsd_column_project()
    results = []
    wk = PushoverWorker(p, dict(control_node=2, control_dof=1, target=0.06,
                                n_steps=30))

    def _maybe_cancel(info):
        if info["step"] >= 5:
            wk.cancel()
    wk.progress.connect(_maybe_cancel)
    wk.done.connect(lambda res: results.append(res))
    wk.run()
    assert results and len(results[0]["disp"]) <= 7      # stopped early


def test_dialog_constructs_and_reads(qapp):
    from pushover_dialog import PushoverDialog
    dlg = PushoverDialog(None, _gsd_column_project())
    kw = dlg._kwargs()
    assert kw["control_node"] == 2                       # the free node
    assert kw["control_dof"] in (0, 1, 2)
    assert kw["n_steps"] >= 2
    assert kw["capture_fibers"] is True                  # on by default
    dlg._draw_curve()                                    # no crash
    dlg._draw_fibers()                                   # placeholder, no crash


def test_dialog_fiber_contour(qapp):
    """GUI-6: after a captured run the step slider is enabled and the fiber
    contour draws in both stress and strain modes."""
    from pushover_dialog import PushoverDialog, PushoverWorker
    p = _gsd_column_project()
    dlg = PushoverDialog(None, p)
    dlg.n_steps.setValue(12)
    wk = PushoverWorker(p, dlg._kwargs())               # capture on by default
    wk.progress.connect(dlg._on_progress)
    wk.done.connect(dlg._on_done)
    wk.run()
    assert dlg._results.has_fibers and dlg.step_slider.isEnabled()
    assert dlg.step_slider.maximum() == dlg._results.n_steps - 1
    for mode in ("strain", "stress"):
        dlg.fiber_mode.setCurrentText(mode)
        dlg.step_slider.setValue(0)
        dlg._draw_fibers()
        dlg.step_slider.setValue(dlg.step_slider.maximum())
        dlg._draw_fibers()


def test_dialog_deformed_shape(qapp):
    """GUI-6 extras: after a captured run the deformed-shape tab has per-step
    frames + hinge-state damage, the play button is enabled, and the shape draws
    at both ends of the slider and at different scale factors — no crash."""
    from pushover_dialog import PushoverDialog, PushoverWorker
    p = _gsd_column_project()
    dlg = PushoverDialog(None, p)
    dlg.n_steps.setValue(12)
    wk = PushoverWorker(p, dlg._kwargs())               # capture on by default
    wk.progress.connect(dlg._on_progress)
    wk.done.connect(dlg._on_done)
    wk.run()
    assert dlg._results.has_shape
    assert dlg._results.step(0).member_damage is not None
    assert dlg.step_slider.isEnabled() and dlg.play_btn.isEnabled()
    for step in (0, dlg.step_slider.maximum()):
        dlg.step_slider.setValue(step)
        dlg._draw_shape()
    for scale in (0.0, 50.0):
        dlg.shape_scale.setValue(scale)
        dlg._draw_shape()


def test_dialog_animation_advances_and_loops(qapp):
    """The play button drives the shared step slider: _advance_step steps
    forward and wraps back to 0 past the end (the animation loop)."""
    from pushover_dialog import PushoverDialog, PushoverWorker
    p = _gsd_column_project()
    dlg = PushoverDialog(None, p)
    dlg.n_steps.setValue(10)
    wk = PushoverWorker(p, dlg._kwargs())
    wk.progress.connect(dlg._on_progress)
    wk.done.connect(dlg._on_done)
    wk.run()
    n = dlg.step_slider.maximum()
    dlg.step_slider.setValue(0)
    dlg._advance_step()
    assert dlg.step_slider.value() == 1
    dlg.step_slider.setValue(n)
    dlg._advance_step()
    assert dlg.step_slider.value() == 0                 # wrapped

    dlg.play_btn.setChecked(True)                       # toggles the timer on
    assert dlg._timer.isActive()
    dlg.play_btn.setChecked(False)
    assert not dlg._timer.isActive()
