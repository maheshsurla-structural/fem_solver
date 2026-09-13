"""Customizable viewport toolbar — ``desktop/nav_toolbar.py`` +
``desktop/toolbar_commands.py``.

Covers the command registry, the default layout, the add/remove/reorder/reset
edits of the Customize dialog, and that a chosen layout persists and reloads.
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

_KEY = "viewport_toolbar_layout"


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvistaqt")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _clean_settings(qapp):
    from PySide6.QtCore import QSettings
    QSettings("MidasStructural", "Desktop").remove(_KEY)
    yield
    QSettings("MidasStructural", "Desktop").remove(_KEY)


def _view():
    import model_view as mv
    from demo_model import demo_project
    v = mv.ModelView()
    v.resize(900, 600)
    v.set_model(demo_project().build_model())
    return v


# --- registry & default layout ---------------------------------------------
def test_view_registers_builtin_commands(qapp):
    v = _view()
    reg = v._nav_bar._registry
    for cid in ("select", "orbit", "pan", "zoomwin", "fit", "fitsel",
                "zoomin", "zoomout", "iso", "top", "front", "lock"):
        assert cid in reg


def test_default_layout_builds_expected_buttons(qapp):
    import nav_toolbar as nt
    v = _view()
    ids = list(v._nav_bar._buttons_by_id)
    assert ids == [c for c in nt._DEFAULT_LAYOUT if c != nt.SEPARATOR_ID]


def test_rebuild_skips_unknown_ids(qapp):
    v = _view()
    bar = v._nav_bar
    bar._layout = ["select", "does_not_exist", "fit"]
    bar.rebuild()
    assert list(bar._buttons_by_id) == ["select", "fit"]


def test_customize_button_present_with_menu(qapp):
    v = _view()
    bar = v._nav_bar
    assert bar._customize_btn is not None
    labels = [a.text().replace("…", "").strip() for a in
              bar._customize_btn.menu().actions()]
    assert "Customize Toolbar" in labels and "Reset Toolbar" in labels
    # the affordance is fixed, never part of the editable layout
    assert "customize" not in bar._buttons_by_id
    assert "customize" not in bar._registry


# --- customize dialog edits -------------------------------------------------
def _dialog(bar):
    from toolbar_commands import CustomizeToolbarDialog
    import nav_toolbar as nt
    return CustomizeToolbarDialog(bar._registry, bar._layout,
                                  nt._DEFAULT_LAYOUT)


def test_dialog_add_and_remove(qapp):
    from PySide6.QtCore import Qt
    v = _view()
    dlg = _dialog(v._nav_bar)
    # add the first available command after the <Separator> row
    dlg._avail.setCurrentRow(1)
    added = dlg._avail.currentItem().data(Qt.ItemDataRole.UserRole)
    before = len(dlg._current)
    dlg._add()
    assert dlg._current.count(added) == 1
    assert len(dlg._current) == before + 1
    # remove it again
    dlg._cur.setCurrentRow(dlg._current.index(added))
    dlg._remove()
    assert added not in dlg._current


def test_dialog_move_reorders(qapp):
    v = _view()
    dlg = _dialog(v._nav_bar)
    dlg._current = ["select", "orbit", "pan"]
    dlg._refresh_current()
    dlg._cur.setCurrentRow(2)
    dlg._move(-1)
    assert dlg._current == ["select", "pan", "orbit"]


def test_dialog_reset_restores_default(qapp):
    import nav_toolbar as nt
    v = _view()
    dlg = _dialog(v._nav_bar)
    dlg._current = ["select"]
    dlg._reset()
    assert dlg.result_layout() == list(nt._DEFAULT_LAYOUT)


def test_dialog_available_excludes_current(qapp):
    from PySide6.QtCore import Qt
    v = _view()
    dlg = _dialog(v._nav_bar)
    avail_ids = {dlg._avail.item(i).data(Qt.ItemDataRole.UserRole)
                 for i in range(dlg._avail.count())}
    assert "select" not in avail_ids            # already on the bar
    assert "iso" in avail_ids                    # not on the default bar


# --- persistence ------------------------------------------------------------
def test_layout_persists_and_reloads(qapp):
    from PySide6.QtCore import QSettings
    v = _view()
    v._nav_bar._layout = ["select", "pan", "|", "fit", "lock"]
    v._nav_bar._save_layout()
    assert QSettings("MidasStructural", "Desktop").value(_KEY)
    v2 = _view()                                 # a fresh view reads the setting
    assert v2._nav_bar._layout == ["select", "pan", "|", "fit", "lock"]
    assert list(v2._nav_bar._buttons_by_id) == ["select", "pan", "fit", "lock"]


# --- shell registration -----------------------------------------------------
def test_shell_registers_model_edit_commands(qapp):
    from main_window import MainWindow
    w = MainWindow()
    reg = w.view._nav_bar._registry
    for cid in ("cmd_node", "cmd_delete", "cmd_undo", "cmd_run"):
        assert cid in reg
    assert reg["cmd_delete"].group == "Edit"
