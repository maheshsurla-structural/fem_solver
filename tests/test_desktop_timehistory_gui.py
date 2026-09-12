"""Nonlinear dynamic time-history GUI (plan §16 C3 — GUI half).

Record import, the threaded worker (progress + cancel via the engine transient
step_callback), and the response-history dialog. Headless (offscreen); the
worker logic runs by calling ``run()`` directly.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

import nonlinear as NL                                 # noqa: E402
from project import Material, Member, Node, Project, Section  # noqa: E402


def _col_project():
    import section_gui_core as core
    spec = core.Spec(kind="Circular", D=0.6, fc=35e6, fy=500e6)
    gsd = dataclasses.asdict(spec)
    A = math.pi * 0.6**2 / 4.0
    p = Project()
    p.nodes = [Node(1, 0, 0, supports=(1, 1, 1)), Node(2, 3.0, 0.0)]
    p.sections = [Section(id=1, name="c", A=A, Iz=math.pi * 0.6**4 / 64,
                          gsd_spec=gsd)]
    p.materials = [Material(1, "c", E=30e9, nu=0.2)]
    p.members = [Member(1, 1, 2, 1, 1)]
    return p


def _accel(n=100, dt=0.02):
    t = np.arange(n) * dt
    return 2.0 * np.sin(2.0 * np.pi * 2.0 * t)


# ------------------------------------------------------ record parsing

def test_load_accel_record_skips_headers_and_columns():
    from timehistory_dialog import load_accel_record
    fn = os.path.join(tempfile.gettempdir(), "rec_test.txt")
    with open(fn, "w") as f:
        f.write("PEER NGA record\nNPTS= 4, DT= .02\n0.1 0.2\n-0.3\n0.4\n")
    arr = load_accel_record(fn)
    assert arr.tolist() == [0.1, 0.2, -0.3, 0.4]


# ------------------------------------------------------ engine hook via run_time_history

def test_run_time_history_on_step_and_cancel():
    p = _col_project()
    seen = []
    r = NL.run_time_history(
        p, _accel(80), 0.02, control_node=2, control_dof=1, direction="y",
        zeta=0.05, density=2400.0,
        on_step=lambda i: seen.append(i["step"]),
        should_cancel=lambda: len(seen) >= 10)
    assert seen[:3] == [1, 2, 3]
    # cancelled early -> fewer recorded steps than the full record
    assert len(r["disp"]) < 80


# ------------------------------------------------------ Qt worker + dialog

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    pytest.importorskip("matplotlib")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_worker_streams_and_produces_history(qapp):
    from timehistory_dialog import TimeHistoryWorker
    p = _col_project()
    steps, results = [], []
    wk = TimeHistoryWorker(p, _accel(100), 0.02,
                           dict(control_node=2, control_dof=1, direction="y",
                                zeta=0.05, density=2400.0))
    wk.progress.connect(lambda i: steps.append(i["step"]))
    wk.done.connect(lambda r: results.append(r))
    wk.run()
    assert steps and results
    r = results[0]
    assert len(r["times"]) == len(r["disp"]) > 10
    assert all(k in r for k in ("disp", "velocity", "acceleration"))


def test_dialog_constructs_and_g_scaling(qapp):
    from timehistory_dialog import TimeHistoryDialog
    dlg = TimeHistoryDialog(None, _col_project())
    dlg._accel = np.array([1.0, 2.0, 3.0])
    dlg.scale.setValue(2.0)
    dlg.in_g.setCurrentIndex(1)                         # record in g
    scaled = dlg._scaled_accel()
    assert scaled[1] == pytest.approx(2.0 * 2.0 * 9.80665, rel=1e-6)
    kw = dlg._kwargs()
    assert kw["control_dof"] == 1 and kw["direction"] == "y"
    dlg._draw()                                         # placeholder, no crash
