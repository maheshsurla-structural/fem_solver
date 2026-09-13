"""Nonlinear case manager (plan §14 GUI-4).

Covers the data model (``NonlinearCase`` + serialization + continue-from chain),
the case runner (``run_case``: monotonic / cyclic / staged continuation), and
the Qt case-manager + pushover-dialog case selector (headless offscreen).
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

import nonlinear as NL                                     # noqa: E402
from project import (Material, Member, Node, NonlinearCase,  # noqa: E402
                     Project, Section)


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


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


# ------------------------------------------------------------- data model

def test_nonlinear_case_serialization_roundtrip():
    p = _gsd_column_project()
    p.nonlinear_cases = [
        NonlinearCase(id=1, name="mono", control_node=2, target=0.05,
                      n_steps=20),
        NonlinearCase(id=2, name="cyc", control_node=2, protocol="cyclic",
                      amplitudes=[0.5, 1.0], cycles=2, pts_per_cycle=16,
                      continue_from=1, tol=1e-5)]
    q = Project.from_json(p.to_json())
    assert len(q.nonlinear_cases) == 2
    c2 = q.nonlinear_case(2)
    assert c2.protocol == "cyclic" and c2.amplitudes == [0.5, 1.0]
    assert c2.continue_from == 1 and c2.tol == 1e-5


def test_case_chain_follows_continue_from():
    p = _gsd_column_project()
    a = NonlinearCase(id=1, name="A", control_node=2)
    b = NonlinearCase(id=2, name="B", control_node=2, continue_from=1)
    c = NonlinearCase(id=3, name="C", control_node=2, continue_from=2)
    p.nonlinear_cases = [a, b, c]
    assert [x.id for x in NL._case_chain(p, c)] == [1, 2, 3]      # root first
    assert [x.id for x in NL._case_chain(p, a)] == [1]


def test_case_total_steps_cyclic():
    p = _gsd_column_project()
    c = NonlinearCase(id=1, name="cyc", control_node=2, protocol="cyclic",
                      amplitudes=[0.5, 1.0], cycles=1, pts_per_cycle=16)
    # 2 amplitudes x 1 cycle x 16 pts = 32 increments
    assert NL.case_total_steps(p, c) == 32


# ------------------------------------------------------------- run_case

def test_run_case_monotonic_matches_pushover():
    p = _gsd_column_project()
    c = NonlinearCase(id=1, name="m", control_node=2, control_dof=1,
                      target=0.05, n_steps=20)
    p.nonlinear_cases = [c]
    rc = NL.run_case(p, c)
    rp = NL.run_pushover(p, control_node=2, control_dof=1, target=0.05,
                         n_steps=20)
    assert rc["protocol"] == "monotonic"
    assert len(rc["disp"]) == len(rp["disp"])
    assert max(rc["shear"]) == pytest.approx(max(rp["shear"]), rel=1e-6)


def test_run_case_cyclic_traces_hysteresis():
    p = _gsd_column_project()
    c = NonlinearCase(id=1, name="cyc", control_node=2, control_dof=1,
                      target=0.02, protocol="cyclic", amplitudes=[0.5, 1.0],
                      cycles=1, pts_per_cycle=16, tol=1e-5)
    p.nonlinear_cases = [c]
    r = NL.run_case(p, c, capture_shape=True)
    d, s = np.array(r["disp"]), np.array(r["shear"])
    assert r["protocol"] == "cyclic"
    assert d.min() < 0 < d.max()                     # both push directions
    assert s.min() < 0 < s.max()                     # reversed base shear
    assert d.max() == pytest.approx(0.02, rel=0.1)
    assert len(r["shape_frames"]) == len(d)


def test_run_case_continue_from_is_continuous():
    p = _gsd_column_project()
    a = NonlinearCase(id=1, name="A", control_node=2, control_dof=1,
                      target=0.02, n_steps=10)
    b = NonlinearCase(id=2, name="B", control_node=2, control_dof=1,
                      target=0.04, n_steps=10, continue_from=1)
    p.nonlinear_cases = [a, b]
    r = NL.run_case(p, b)
    d = np.array(r["disp"])
    assert len(d) == 20                              # 10 (A) + 10 (B)
    assert np.all(np.diff(d) >= -1e-9)              # continuous, non-decreasing
    assert d[-1] == pytest.approx(0.06, rel=0.05)   # 0.02 (A) + 0.04 (B)


# ------------------------------------------------------------- Qt dialogs

def test_case_dialog_monotonic_and_cyclic(qapp):
    from nonlinear_cases import NonlinearCaseDialog
    p = _gsd_column_project()
    dlg = NonlinearCaseDialog(None, p)
    # monotonic by default -> mono host visible, cyclic hidden
    assert dlg.mono_host.isVisibleTo(dlg) or not dlg.cyc_host.isVisibleTo(dlg)
    c = dlg.data()
    assert c.protocol == "monotonic"
    # switch to cyclic
    dlg.protocol.setCurrentIndex(dlg.protocol.findData("cyclic"))
    dlg.amplitudes.setText("0.5, 1.0, 1.5")
    dlg.cycles.setValue(2)
    c = dlg.data()
    assert c.protocol == "cyclic" and c.amplitudes == [0.5, 1.0, 1.5]
    assert c.cycles == 2


def test_case_dialog_uses_scaffold_panels(qapp):
    """A2: the dialog is built from the L1 scaffold — a CaseHeader plus a
    two-column grid of GroupCards — not a flat form."""
    import analysis_ui as ui
    from nonlinear_cases import NonlinearCaseDialog
    p = _gsd_column_project()
    dlg = NonlinearCaseDialog(None, p)
    assert isinstance(dlg.header, ui.CaseHeader)
    assert dlg.header.type_label() == "Nonlinear Static"
    # at least one GroupCard and a two-column grid are present
    from PySide6.QtWidgets import QGridLayout
    assert dlg.findChild(ui.GroupCard) is not None
    assert dlg.findChild(QGridLayout) is not None


def test_case_dialog_notes_roundtrip(qapp):
    """Notes typed in the header are carried on the NonlinearCase and survive a
    JSON round-trip (the notes field added for A2)."""
    from nonlinear_cases import NonlinearCaseDialog
    p = _gsd_column_project()
    src = NonlinearCase(id=1, name="N", control_node=2, notes="hold P then push")
    p.nonlinear_cases = [src]
    dlg = NonlinearCaseDialog(None, p, src)
    assert dlg.header.notes() == "hold P then push"
    assert dlg.data().notes == "hold P then push"
    q = Project.from_json(p.to_json())
    assert q.nonlinear_case(1).notes == "hold P then push"


def test_case_dialog_edit_seeds_existing(qapp):
    from nonlinear_cases import NonlinearCaseDialog
    p = _gsd_column_project()
    src = NonlinearCase(id=3, name="X", control_node=2, control_dof=1,
                        target=0.03, protocol="cyclic", amplitudes=[1.0],
                        tol=1e-5)
    p.nonlinear_cases = [src]
    dlg = NonlinearCaseDialog(None, p, src)
    c = dlg.data()
    assert c.id == 3 and c.name == "X" and c.protocol == "cyclic"
    assert c.target == pytest.approx(0.03) and c.tol == 1e-5


def test_case_manager_returns_cases(qapp):
    from nonlinear_cases import NonlinearCaseManagerDialog
    p = _gsd_column_project()
    p.nonlinear_cases = [NonlinearCase(id=1, name="m", control_node=2)]
    dlg = NonlinearCaseManagerDialog(None, p)
    assert dlg.table.rowCount() == 1
    out = dlg.result_cases()
    assert len(out) == 1 and out is not p.nonlinear_cases


def test_pushover_dialog_case_selector(qapp):
    from pushover_dialog import PushoverDialog, PushoverWorker
    p = _gsd_column_project()
    p.nonlinear_cases = [NonlinearCase(id=1, name="m", control_node=2,
                                       control_dof=1, target=0.05, n_steps=15)]
    dlg = PushoverDialog(None, p)
    # selecting the case populates + disables manual inputs
    dlg.case_combo.setCurrentIndex(dlg.case_combo.findData(1))
    assert dlg._selected_case().id == 1
    assert not dlg.node.isEnabled()
    # run the selected case through the worker (synchronous)
    wk = PushoverWorker(p, dlg._kwargs(), case=dlg._selected_case())
    wk.progress.connect(dlg._on_progress)
    wk.done.connect(dlg._on_done)
    wk.run()
    assert dlg._protocol == "monotonic"
    assert len(dlg._disp) > 5 and max(dlg._shear) > 0
    # back to manual re-enables inputs
    dlg.case_combo.setCurrentIndex(dlg.case_combo.findData(None))
    assert dlg.node.isEnabled() and dlg._selected_case() is None


def test_main_window_wires_nonlinear_cases(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_gsd_column_project())
    assert hasattr(w, "act_nlcases")
    assert callable(w.manage_nonlinear_cases)
