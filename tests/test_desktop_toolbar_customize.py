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
    bar = v._nav_bar
    ids = list(bar._buttons_by_id)
    # a bare view knows the built-in commands + the selection container but not
    # the shell's ``grp_active`` container, and ``rebuild`` skips unknown ids —
    # so compare against the default layout filtered to what this view registers.
    assert ids == [c for c in nt._DEFAULT_LAYOUT
                   if c != nt.SEPARATOR_ID and c in bar._registry]
    # the selection tools live inside the container, not as loose buttons
    assert "grp_select" in ids
    grp = bar._registry["grp_select"]
    assert grp.kind == "group"
    assert grp.members == ["select", "window", "polygon"]


def test_selection_container_flyout_and_state(qapp):
    v = _view()
    bar = v._nav_bar
    btn = bar._buttons_by_id["grp_select"]
    # the flyout menu offers all three selection tools
    labels = [a.text() for a in btn.menu().actions()]
    assert labels == ["Select", "Window select", "Polygon select"]
    # choosing Window from the flyout switches the view mode and the container
    # adopts it as its current tool
    win_action = next(a for a in btn.menu().actions()
                      if a.data() == "window")
    win_action.trigger()
    assert v.current_mode() == "window"
    assert bar._group_state["grp_select"] == "window"
    assert btn.isChecked()


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
    assert "grp_select" not in avail_ids        # container already on the bar
    assert "select" in avail_ids                 # its member can still be pinned
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


def test_shell_registers_activation_container_on_bar(qapp):
    from main_window import MainWindow
    w = MainWindow()
    bar = w.view._nav_bar
    for cid in ("cmd_inactivate", "cmd_activate_only", "cmd_activate_all",
                "cmd_invert_active"):
        assert cid in bar._registry
        assert bar._registry[cid].group == "Active"
    # they are gathered under the activation container on the default bar
    assert "grp_active" in bar._buttons_by_id
    grp = bar._registry["grp_active"]
    assert grp.kind == "group"
    assert grp.members == ["cmd_inactivate", "cmd_activate_only",
                           "cmd_activate_all", "cmd_invert_active"]
    # its flyout offers each action (an action group, so no popup-mode tool face)
    btn = bar._buttons_by_id["grp_active"]
    assert len(btn.menu().actions()) == 4
