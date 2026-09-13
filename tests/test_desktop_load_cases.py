"""Load-cases dialog redesign (plan L3) — ``editing.LoadCaseDialog``.

Headless coverage that the grouped rebuild keeps the data contract (id
preservation, add / remove, nature edit) and shows a per-nature badge.
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

from project import LoadCase, Project           # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _project():
    p = Project(ndm=2, ndf=3)
    p.load_cases = [LoadCase(1, "Dead", "dead"), LoadCase(2, "Live", "live")]
    return p


def test_uses_scaffold_card_and_badges(qapp):
    import analysis_ui as ui
    from editing import LoadCaseDialog
    dlg = LoadCaseDialog(None, _project())
    assert dlg.findChild(ui.GroupCard) is not None
    assert dlg.tbl.rowCount() == 2
    assert not dlg.tbl.item(0, 0).icon().isNull()      # nature badge present


def test_add_and_accept_preserves_ids(qapp):
    from editing import LoadCaseDialog
    p = _project()
    dlg = LoadCaseDialog(None, p)
    dlg._add()                                          # a new row (id 0)
    assert dlg.tbl.rowCount() == 3
    dlg.accept()
    cases = dlg.result_cases
    assert cases is not None and len(cases) == 3
    assert [c.id for c in cases[:2]] == [1, 2]          # existing ids kept
    assert cases[2].id == 3                             # fresh id assigned


def test_nature_change_updates_result(qapp):
    from editing import LoadCaseDialog
    p = _project()
    dlg = LoadCaseDialog(None, p)
    combo = dlg.tbl.cellWidget(1, 1)                    # row 2 = "Live"
    combo.setCurrentIndex(combo.findData("wind"))
    dlg.accept()
    assert dlg.result_cases[1].nature == "wind"


def test_remove_keeps_at_least_one(qapp):
    from editing import LoadCaseDialog
    p = _project()
    p.load_cases = [LoadCase(1, "Dead", "dead")]
    dlg = LoadCaseDialog(None, p)
    dlg.tbl.setCurrentCell(0, 0)
    dlg._remove()                                       # refuse: never empty
    assert dlg.tbl.rowCount() == 1
