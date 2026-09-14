"""Units preferences + live switch (plan U4).

Covers the picker dialog and the whole-app switch driven from the clickable
status-bar chip: the model stays SI, every readout re-renders, the choice is
remembered as the app default, and the change is an undoable (dirty) edit.
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
    pytest.importorskip("pyvistaqt")
    from PySide6.QtCore import QSettings
    QSettings("MidasStructural", "Desktop").setValue("theme", "light")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


# ------------------------------------------------------------------ dialog
def test_dialog_lists_units_and_previews(qapp):
    from units_dialog import UnitsDialog
    dlg = UnitsDialog(None, "kN", "m")
    forces = [dlg.force.itemData(i) for i in range(dlg.force.count())]
    lengths = [dlg.length.itemData(i) for i in range(dlg.length.count())]
    assert forces == ["N", "kN", "kgf", "tonf", "kip", "lbf"]   # + imperial (U5)
    assert lengths == ["m", "cm", "mm", "in", "ft"]
    assert dlg.result_units() == ("kN", "m")
    assert "Moment: kN·m" in dlg._preview.text()


def test_dialog_preview_updates_live(qapp):
    from units_dialog import UnitsDialog
    dlg = UnitsDialog(None, "N", "mm")
    assert "Stress: N/mm²" in dlg._preview.text()
    i = [dlg.length.itemData(k) for k in range(dlg.length.count())].index("m")
    dlg.length.setCurrentIndex(i)
    assert "Length: m" in dlg._preview.text()
    assert "Stress: N/m²" in dlg._preview.text()


# ------------------------------------------------------------ live switch
@pytest.fixture(scope="module")
def win(qapp):
    from demo_model import demo_project
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(demo_project())
    return w


def test_status_chip_is_clickable(win):
    from main_window import _ClickableLabel
    assert isinstance(win._st_units, _ClickableLabel)


def test_change_units_switches_live_and_persists(win, monkeypatch):
    from PySide6.QtCore import QSettings
    import units_dialog
    s = QSettings("MidasStructural", "Desktop")
    try:
        p = win._project
        p.force_unit, p.length_unit = "N", "m"
        win._undo_stack.clear()
        win._refresh_status()

        monkeypatch.setattr(units_dialog.UnitsDialog, "get",
                            staticmethod(lambda *a, **k: ("kN", "mm")))
        win.change_units()

        # model untouched (still SI), display switched
        assert (p.force_unit, p.length_unit) == ("kN", "mm")
        assert win._st_units.text() == "kN · mm"
        # remembered as the app default
        assert s.value("units/force") == "kN"
        assert s.value("units/length") == "mm"
        # a real, dirty-making, undoable change
        assert not win._undo_stack.isClean()
        win._undo_stack.undo()
        assert (win._project.force_unit, win._project.length_unit) == ("N", "m")
    finally:                                  # keep the session default at SI
        s.setValue("units/force", "N")
        s.setValue("units/length", "m")


def test_change_units_noop_when_unchanged(win, monkeypatch):
    import units_dialog
    win._project.force_unit, win._project.length_unit = "N", "m"
    win._undo_stack.clear()
    monkeypatch.setattr(units_dialog.UnitsDialog, "get",
                        staticmethod(lambda *a, **k: ("N", "m")))
    win.change_units()
    assert win._undo_stack.isClean()          # nothing pushed


def test_new_model_adopts_app_default(win):
    from PySide6.QtCore import QSettings
    from project import Project
    s = QSettings("MidasStructural", "Desktop")
    s.setValue("units/force", "kN")
    s.setValue("units/length", "cm")
    try:
        win.load_project(Project(), None)     # path=None → adopt app default
        assert win._project.force_unit == "kN"
        assert win._project.length_unit == "cm"
    finally:
        s.setValue("units/force", "N")
        s.setValue("units/length", "m")


def test_opened_file_keeps_its_own_units(win):
    from PySide6.QtCore import QSettings
    from project import Project
    s = QSettings("MidasStructural", "Desktop")
    s.setValue("units/force", "kN")
    try:
        p = Project(force_unit="tonf", length_unit="mm")
        win.load_project(p, path="C:/some/model.fsproj")   # path set → keep
        assert win._project.force_unit == "tonf"
        assert win._project.length_unit == "mm"
    finally:
        s.setValue("units/force", "N")
        s.setValue("units/length", "m")
