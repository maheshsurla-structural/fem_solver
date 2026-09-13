"""Nodal load dialog redesign (plan L4) — ``editing.LoadDialog``.

Headless (offscreen) coverage that the grouped rebuild keeps the data contract
(node / case / component values) and reads back a correct ``Load``.
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

from project import Load, LoadCase, Node, Project        # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _project(ndm=2, ndf=3):
    p = Project(ndm=ndm, ndf=ndf)
    p.nodes = [Node(1, 0, 0, supports=(1, 1, 1)), Node(2, 0, 3)]
    p.load_cases = [LoadCase(1, "Dead", "dead"), LoadCase(2, "Live", "live")]
    return p


def test_load_dialog_uses_scaffold_cards(qapp):
    import analysis_ui as ui
    from editing import LoadDialog
    dlg = LoadDialog(None, _project())
    cards = dlg.findChildren(ui.GroupCard)
    titles = {c._box.title() for c in cards}
    assert "Applied to" in titles
    assert any(t.startswith("Components") for t in titles)
    assert len(dlg.vals) == 3                       # one spin per 2-D DOF


def test_load_dialog_reads_back_values(qapp):
    from editing import LoadDialog
    p = _project()
    dlg = LoadDialog(None, p)
    dlg.vals[0].setValue(1000.0)                    # Fx
    dlg.vals[1].setValue(-2500.0)                   # Fy
    dlg.case.setCurrentIndex(dlg.case.findData(2))  # Live
    ld = dlg.data()
    assert isinstance(ld, Load)
    assert ld.values[0] == pytest.approx(1000.0)
    assert ld.values[1] == pytest.approx(-2500.0)
    assert ld.case == 2


def test_load_dialog_seeds_existing(qapp):
    from editing import LoadDialog
    p = _project()
    src = Load(node=2, values=(10.0, 20.0, 5.0), case=2)
    dlg = LoadDialog(None, p, src)
    assert dlg.node.currentData() == 2
    assert dlg.case.currentData() == 2
    assert [s.value() for s in dlg.vals] == pytest.approx([10.0, 20.0, 5.0])


def test_load_dialog_3d_has_six_components(qapp):
    from editing import LoadDialog
    dlg = LoadDialog(None, _project(ndm=3, ndf=6))
    assert len(dlg.vals) == 6
