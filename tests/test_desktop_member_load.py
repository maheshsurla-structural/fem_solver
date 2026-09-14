"""Member (line) load dialog + wiring (plan L5).

Headless coverage of ``member_load_dialog.MemberLoadDialog`` (2-D / 3-D, data
round-trip) and its main-window wiring (action, handler, tree group).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

from project import (LoadCase, Material, Member, MemberLoad,  # noqa: E402
                     Node, Project, Section)


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _isolate_settings(qapp, tmp_path, monkeypatch):
    """Point the window's QSettings at a throwaway ini so persisted nav flags
    (summary / expanded) can't leak in from the real store. Without this the
    tree can come up in summary mode (counts only), where individual leaf refs
    like ("member_load", 0) intentionally don't exist. Mirrors the fixture in
    test_desktop_nav.py."""
    import main_window
    from PySide6.QtCore import QSettings
    ini = str(tmp_path / "settings.ini")
    monkeypatch.setattr(
        main_window, "QSettings",
        lambda *a, **k: QSettings(ini, QSettings.Format.IniFormat))
    yield


def _project(ndm=2, ndf=3):
    p = Project(ndm=ndm, ndf=ndf)
    p.nodes = [Node(1, 0, 0, supports=(1, 1, 1)), Node(2, 4, 0)]
    p.sections = [Section(id=1, name="s", A=1e-3, Iz=1e-5)]
    p.materials = [Material(1, "m", E=2e11, nu=0.3)]
    p.members = [Member(1, 1, 2, 1, 1)]
    p.load_cases = [LoadCase(1, "Dead", "dead"), LoadCase(2, "Live", "live")]
    return p


def test_dialog_reads_back_2d(qapp):
    from member_load_dialog import MemberLoadDialog
    dlg = MemberLoadDialog(None, _project())
    assert dlg.wz is None                       # 2-D: no local-z UDL
    dlg.wy.setValue(-3000.0)
    dlg.case.setCurrentIndex(dlg.case.findData(2))
    ml = dlg.data()
    assert isinstance(ml, MemberLoad)
    assert ml.member == 1 and ml.wy == pytest.approx(-3000.0)
    assert ml.wz == 0.0 and ml.case == 2


def test_dialog_3d_has_wz(qapp):
    from member_load_dialog import MemberLoadDialog
    dlg = MemberLoadDialog(None, _project(ndm=3, ndf=6))
    assert dlg.wz is not None
    dlg.wy.setValue(100.0)
    dlg.wz.setValue(200.0)
    ml = dlg.data()
    assert ml.wy == pytest.approx(100.0) and ml.wz == pytest.approx(200.0)


def test_dialog_seeds_existing(qapp):
    from member_load_dialog import MemberLoadDialog
    p = _project()
    src = MemberLoad(member=1, wy=-1234.0, case=2)
    dlg = MemberLoadDialog(None, p, src)
    assert dlg.member.currentData() == 1
    assert dlg.case.currentData() == 2
    assert dlg.wy.value() == pytest.approx(-1234.0)


def test_main_window_wires_line_loads(qapp):
    from main_window import MainWindow
    w = MainWindow()
    p = _project()
    p.member_loads = [MemberLoad(member=1, wy=-500.0, case=1)]
    w.load_project(p)
    assert hasattr(w, "act_add_lineload")
    assert callable(w.add_line_load)
    # the tree carries a Line loads group with a ("member_load", 0) ref
    found = w._find_item(("member_load", 0))
    assert found is not None
