"""Cable-stayed tuning (ULF) — desktop wiring (bridge GUI increment G6).

Headless coverage of the setup dialog, the Analysis-cases row, and the
``MainWindow.run_cable_tuning`` runner.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "desktop"))

from project import (Load, LoadCase, Material, Member,  # noqa: E402
                     Node, Project, Section)


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _cable_stayed(with_load=True):
    """Symmetric cable-stayed model: deck 1-7, ground-anchored pylon (9→8),
    two stays (members 8, 9), dead load on the deck."""
    p = Project(ndm=2, ndf=3)
    for i, x in enumerate([0, 5, 10, 15, 20, 25, 30]):
        p.nodes.append(Node(id=i + 1, x=float(x), y=0.0))
    p.nodes.append(Node(id=8, x=15.0, y=12.0))         # pylon top
    p.nodes.append(Node(id=9, x=15.0, y=0.0))          # pylon base
    p.nodes[0].supports = (1, 1, 0)
    p.nodes[6].supports = (1, 1, 0)
    next(n for n in p.nodes if n.id == 9).supports = (1, 1, 1)
    p.sections = [Section(id=1, name="deck", A=0.5, Iz=0.05)]
    p.materials = [Material(1, "steel", E=2e11, nu=0.3, rho=7850.0)]
    mid = 1
    for i in range(6):
        p.members.append(Member(mid, i + 1, i + 2, 1, 1)); mid += 1
    p.members.append(Member(7, 9, 8, 1, 1))            # pylon
    p.members.append(Member(8, 8, 3, 1, 1))            # cable L
    p.members.append(Member(9, 8, 5, 1, 1))            # cable R
    p.load_cases = [LoadCase(1, "Dead", "dead")]
    if with_load:
        for nd in (2, 3, 4, 5, 6):
            p.loads.append(Load(node=nd, values=(0, -50e3, 0), case=1))
    return p


# ---------------------------------------------------------------- dialog
def test_dialog_lists_members_and_nodes(qapp):
    from cable_tuning_dialog import CableTuningDialog
    p = _cable_stayed()
    d = CableTuningDialog(None, p)
    assert d.cables.count() == len(p.members)
    assert d.targets.count() == len(p.nodes)
    assert d.result() == {"cables": [], "targets": []}   # nothing pre-selected


# ---------------------------------------------------- analysis-cases row
def test_analysis_cases_lists_cable_tuning(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog
    dlg = AnalysisCasesDialog(None, _cable_stayed())
    kinds = [m["kind"] for m in dlg._row_meta]
    assert "cabletuning" in kinds
    dlg.table.setCurrentCell(kinds.index("cabletuning"), 0)
    dlg._run()
    assert dlg._run_request == ("cabletuning",)


# --------------------------------------------------------- run_cable_tuning
def test_run_cable_tuning_meets_targets(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_cable_stayed())
    res = w.run_cable_tuning(config={"cables": [8, 9], "targets": [3, 5]})
    assert res is not None
    import numpy as np
    assert np.all(res["tensions"] > 0)                 # stays in tension
    assert np.max(np.abs(res["residual"])) < 1e-9      # targets met
    assert hasattr(w, "_cable_tuning_results_dlg")


def test_run_guards(qapp, monkeypatch):
    import main_window as MW
    monkeypatch.setattr(MW.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    # no dead load
    w = MW.MainWindow()
    w.load_project(_cable_stayed(with_load=False))
    assert w.run_cable_tuning(config={"cables": [8, 9],
                                      "targets": [3, 5]}) is None
    # no cables / targets selected
    w.load_project(_cable_stayed())
    assert w.run_cable_tuning(config={"cables": [], "targets": [3]}) is None
    # 3-D gated
    p3 = Project(ndm=3, ndf=6)
    p3.nodes = [Node(1, 0, 0, z=0), Node(2, 3, 0, z=0)]
    p3.sections = [Section(id=1, name="s", A=0.5, Iz=0.05, Iy=0.05, J=1e-3)]
    p3.materials = [Material(1, "s", E=2e11, nu=0.3, rho=7850.0)]
    p3.members = [Member(1, 1, 2, 1, 1)]
    p3.load_cases = [LoadCase(1, "Dead", "dead")]
    p3.loads = [Load(node=2, values=(0, -1e3, 0, 0, 0, 0), case=1)]
    w.load_project(p3)
    assert w.run_cable_tuning(config={"cables": [1], "targets": [2]}) is None
