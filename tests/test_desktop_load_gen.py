"""Parametric load-generator dialog redesign (plan L6) — ``editing.LoadGenDialog``.

Headless coverage that the grouped rebuild keeps ``.get()``/``params()`` and
shows an accurate live preview of what will be generated.
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

from project import Material, Member, Node, Project, Section    # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _frame():
    # base node + two elevated nodes -> gravity hits 2 nodes
    p = Project(ndm=2, ndf=3)
    p.nodes = [Node(1, 0, 0, supports=(1, 1, 1)), Node(2, 0, 3), Node(3, 4, 3)]
    p.sections = [Section(id=1, name="s", A=1e-3, Iz=1e-5)]
    p.materials = [Material(1, "m", E=2e11, nu=0.3)]
    p.members = [Member(1, 1, 2, 1, 1), Member(2, 1, 3, 1, 1)]
    return p


def test_uses_card_and_params_contract(qapp):
    import analysis_ui as ui
    from editing import LoadGenDialog
    dlg = LoadGenDialog(None, _frame())
    assert dlg.findChild(ui.GroupCard) is not None
    dlg.kind.setCurrentIndex(dlg.kind.findData("gravity"))
    dlg.mag.setValue(1000.0)
    assert dlg.params() == ("gravity", 1000.0)


def test_gravity_preview_counts_elevated_nodes(qapp):
    from editing import LoadGenDialog
    dlg = LoadGenDialog(None, _frame())
    dlg.kind.setCurrentIndex(dlg.kind.findData("gravity"))
    dlg.mag.setValue(1000.0)
    text = dlg._preview.text()
    assert "2 nodal load" in text            # 2 nodes above the base
    assert "2,000" in text                   # total = 2 x 1000


def test_label_switches_for_lateral(qapp):
    from editing import LoadGenDialog
    dlg = LoadGenDialog(None, _frame())
    dlg.kind.setCurrentIndex(dlg.kind.findData("lateral_x"))
    assert "Base shear" in dlg._label.text()
    assert "base shear" in dlg._preview.text().lower()


def test_empty_when_no_elevated_nodes(qapp):
    from editing import LoadGenDialog
    p = Project(ndm=2, ndf=3)
    p.nodes = [Node(1, 0, 0, supports=(1, 1, 1)), Node(2, 4, 0)]   # all at base
    dlg = LoadGenDialog(None, p)
    dlg.kind.setCurrentIndex(dlg.kind.findData("gravity"))
    dlg.mag.setValue(5000.0)
    assert "no loads" in dlg._preview.text()
