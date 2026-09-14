"""Temperature-gradient load — desktop wiring (bridge GUI increment G2).

Headless coverage of the setup dialog, the Analysis-cases row, and the
``MainWindow.run_temperature_gradient`` runner.
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

from project import Material, Member, Node, Project, Section  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _two_span(n=8, dx=3.0):
    """A 2-span continuous girder (so continuity moments appear)."""
    p = Project(ndm=2, ndf=3)
    p.nodes = [Node(i + 1, i * dx, 0.0) for i in range(n + 1)]
    p.nodes[0].supports = (1, 1, 0)
    p.nodes[n // 2].supports = (0, 1, 0)
    p.nodes[-1].supports = (0, 1, 0)
    p.sections = [Section(id=1, name="deck", A=0.6, Iz=0.08)]
    p.materials = [Material(1, "conc", E=3e10, nu=0.2, rho=2500.0)]
    p.members = [Member(i + 1, i + 1, i + 2, 1, 1) for i in range(n)]
    return p


# ---------------------------------------------------------------- dialog
def test_dialog_defaults_and_source_switch(qapp):
    from temperature_gradient_dialog import TemperatureGradientDialog
    d = TemperatureGradientDialog(None, _two_span())
    assert len(d.member_ids()) == 8                    # all selected by default
    cfg = d.result()
    assert cfg["source"] == "aashto" and cfg["alpha"] == pytest.approx(1e-5)
    d.source.setCurrentIndex(1)                        # linear
    assert d.stack.currentIndex() == 1
    assert d.result()["source"] == "linear"


# ---------------------------------------------------- analysis-cases row
def test_analysis_cases_lists_temperature_gradient(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog
    dlg = AnalysisCasesDialog(None, _two_span())
    kinds = [m["kind"] for m in dlg._row_meta]
    assert "tempgradient" in kinds
    dlg.table.setCurrentCell(kinds.index("tempgradient"), 0)
    assert dlg._run_btn.isEnabled()
    dlg._run()
    assert dlg._run_request == ("tempgradient",)


# ------------------------------------------------- run_temperature_gradient
def test_run_aashto_gradient_self_stress_and_continuity(qapp):
    from main_window import MainWindow
    w = MainWindow()
    p = _two_span()
    w.load_project(p)
    res = w.run_temperature_gradient(config={
        "source": "aashto", "zone": 3, "dt_top": 20, "dt_bot": 0,
        "alpha": 1e-5, "members": [m.id for m in p.members]})
    assert res is not None
    a = res["actions"]
    assert a.self_stress_top < 0.0                     # top fibre compression
    assert a.dT_uniform > 0.0
    # continuous span → a non-zero continuity (secondary) moment develops
    assert res["max_moment"] > 0.0
    assert hasattr(w, "_temp_gradient_results_dlg")
    assert "Temperature gradient" in w.statusBar().currentMessage()


def test_run_linear_gradient(qapp):
    from main_window import MainWindow
    w = MainWindow()
    p = _two_span()
    w.load_project(p)
    res = w.run_temperature_gradient(config={
        "source": "linear", "zone": 1, "dt_top": 15.0, "dt_bot": 0.0,
        "alpha": 1e-5, "members": [m.id for m in p.members]})
    assert res is not None
    # a linear profile is fully accommodated → ~zero self-stress
    assert abs(res["actions"].self_stress_top) < 1.0   # ~0 Pa


def test_run_guards(qapp, monkeypatch):
    import main_window as MW
    monkeypatch.setattr(MW.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    w = MW.MainWindow()
    w.load_project(_two_span())
    # no members selected
    assert w.run_temperature_gradient(config={
        "source": "aashto", "zone": 3, "dt_top": 0, "dt_bot": 0,
        "alpha": 1e-5, "members": []}) is None
    # 3-D gated
    p3 = Project(ndm=3, ndf=6)
    p3.nodes = [Node(1, 0, 0, z=0), Node(2, 3, 0, z=0)]
    p3.sections = [Section(id=1, name="s", A=0.6, Iz=0.08, Iy=0.08, J=1e-3)]
    p3.materials = [Material(1, "c", E=3e10, nu=0.2, rho=2500.0)]
    p3.members = [Member(1, 1, 2, 1, 1)]
    w.load_project(p3)
    assert w.run_temperature_gradient(config={
        "source": "aashto", "zone": 3, "dt_top": 0, "dt_bot": 0,
        "alpha": 1e-5, "members": [1]}) is None
