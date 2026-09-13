"""Shared Loads/Analysis UI scaffold (plan L1) — ``desktop/analysis_ui.py``.

Headless (offscreen) coverage of the frozen scaffold contract: GroupCard,
CaseHeader, LoadsAppliedTable, two_column, dialog_buttons, direction_glyph, and
the demo dialog used for the screenshot record.
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


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


# ------------------------------------------------------------------ GroupCard
def test_groupcard_form_rows(qapp):
    from PySide6.QtWidgets import QLineEdit
    import analysis_ui as ui
    g = ui.GroupCard("Control")
    g.add_row("Node", QLineEdit())
    g.add_row("DOF", QLineEdit())
    from PySide6.QtWidgets import QFormLayout
    assert isinstance(g.body_layout(), QFormLayout)
    assert g.body_layout().rowCount() == 2


def test_groupcard_custom_mode(qapp):
    from PySide6.QtWidgets import QVBoxLayout, QLabel
    import analysis_ui as ui
    g = ui.GroupCard("Loads Applied", form=False)
    assert isinstance(g.body_layout(), QVBoxLayout)
    g.body_layout().addWidget(QLabel("x"))
    with pytest.raises(RuntimeError):
        g.add_row("nope", QLabel())          # form-only API on a custom card


# ----------------------------------------------------------------- CaseHeader
def test_caseheader_getters(qapp):
    import analysis_ui as ui
    h = ui.CaseHeader(name="PUSHOVER-X", type_label="Nonlinear Static",
                      notes="hello")
    assert h.name() == "PUSHOVER-X"
    assert h.type_label() == "Nonlinear Static"
    assert h.notes() == "hello"


# ----------------------------------------------------------- LoadsAppliedTable
def test_loads_applied_set_and_get_rows(qapp):
    import analysis_ui as ui
    t = ui.LoadsAppliedTable(["Load Type", "Load Name", "Scale"])
    t.set_rows([{"Load Type": "Accel", "Load Name": "UX", "Scale": 1.0,
                 "hidden": 42}])
    assert t.table.rowCount() == 1
    assert t.table.item(0, 0).text() == "Accel"
    # round-trips including keys not shown as columns
    assert t.rows()[0]["hidden"] == 42


def test_loads_applied_add_edit_delete(qapp):
    import analysis_ui as ui
    t = ui.LoadsAppliedTable(
        ["Load Type", "Scale"],
        on_add=lambda: {"Load Type": "Accel", "Scale": 1.0},
        on_edit=lambda row: {**row, "Scale": 2.0})
    t._add()
    assert len(t.rows()) == 1 and t.rows()[0]["Scale"] == 1.0
    t.table.setCurrentCell(0, 0)
    t._edit()
    assert t.rows()[0]["Scale"] == 2.0       # editor callback applied
    t.table.setCurrentCell(0, 0)
    t._delete()
    assert t.rows() == []


def test_loads_applied_buttons_gated(qapp):
    import analysis_ui as ui
    # no on_add -> Add disabled; no selection -> Modify/Delete disabled
    t = ui.LoadsAppliedTable(["A"], on_edit=lambda r: r)
    assert not t._add_btn.isEnabled()
    assert not t._mod_btn.isEnabled() and not t._del_btn.isEnabled()
    t.set_rows([{"A": "x"}])
    t.table.setCurrentCell(0, 0)
    assert t._mod_btn.isEnabled() and t._del_btn.isEnabled()


# ------------------------------------------------------------------ two_column
def test_two_column_grid_and_full_width_span(qapp):
    from PySide6.QtWidgets import QGridLayout
    import analysis_ui as ui
    a, b, c = (ui.GroupCard("A"), ui.GroupCard("B"),
               ui.GroupCard("Wide", form=False))
    c.full_width = True
    host = ui.two_column(a, b, c)
    grid = host.layout()
    assert isinstance(grid, QGridLayout)
    # A and B share a row (cols 0,1); C spans both columns on its own row
    ra, ca, *_ = grid.getItemPosition(grid.indexOf(a))
    rb, cb, *_ = grid.getItemPosition(grid.indexOf(b))
    rc, cc, rs, cs = grid.getItemPosition(grid.indexOf(c))
    assert (ra, ca) == (0, 0) and (rb, cb) == (0, 1)
    assert cc == 0 and cs == 2 and rc == 1


# --------------------------------------------------------- buttons / glyph / demo
def test_dialog_buttons_accept(qapp):
    from PySide6.QtWidgets import QDialog
    import analysis_ui as ui
    dlg = QDialog()
    bb = ui.dialog_buttons(dlg)
    accepted = []
    dlg.accepted.connect(lambda: accepted.append(True))
    bb.button(bb.StandardButton.Ok).click()
    assert accepted == [True]


def test_direction_glyph_maps_and_falls_back(qapp):
    import analysis_ui as ui
    assert "+Y" in ui.direction_glyph("Dy").text()
    assert "+X" in ui.direction_glyph("Ux").text()
    assert ui.direction_glyph("Q7").text() == "Q7"     # unknown -> label itself


def test_demo_dialog_constructs(qapp):
    import analysis_ui as ui
    dlg = ui._demo_dialog()
    assert dlg.windowTitle().startswith("Analysis case")
