"""Ribbon toolbar groups (plan S2) — ``desktop/main_window.py``.

Headless (offscreen) coverage of the S2 toolbar reorg: dedicated **Loads** and
**Analysis** toolbars built from captioned, text-under-icon ribbon groups, with
the load actions pulled off the Edit/Generate toolbars. Asserts the groups, the
captions, the text-under-icon button style (menus keep the full action text),
and that Ctrl+R linear-static still rides a toolbar button.
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


@pytest.fixture(scope="module")
def win(qapp):
    from main_window import MainWindow
    return MainWindow()


def _toolbar(win, title):
    from PySide6.QtWidgets import QToolBar
    for tb in win.findChildren(QToolBar):
        if tb.windowTitle() == title:
            return tb
    return None


def _ribbon_buttons(tb):
    from PySide6.QtWidgets import QToolButton
    return [b for b in tb.findChildren(QToolButton)
            if b.objectName() == "ribbonBtn"]


def _ribbon_actions(tb):
    return [b.defaultAction() for b in _ribbon_buttons(tb)]


def _captions(tb):
    from PySide6.QtWidgets import QLabel
    return [lbl.text() for lbl in tb.findChildren(QLabel)
            if lbl.objectName() == "ribbonCap"]


def test_loads_toolbar_ribbon_groups(win):
    tb = _toolbar(win, "Loads")
    assert tb is not None
    assert tb.objectName() == "ribbonBar"
    acts = set(_ribbon_actions(tb))
    for attr in ("act_loadcases", "act_add_load", "act_add_lineload",
                 "act_genloads", "act_editcombos", "act_gencombos"):
        assert getattr(win, attr) in acts, attr
    caps = _captions(tb)
    assert "LOADS" in caps and "COMBINATIONS" in caps, caps


def test_analysis_toolbar_ribbon_groups(win):
    tb = _toolbar(win, "Analysis")
    assert tb is not None
    acts = set(_ribbon_actions(tb))
    for attr in ("act_analysiscases", "act_runanalysis", "act_run",
                 "act_undef", "act_diag_n", "act_diag_v", "act_diag_m",
                 "act_design", "act_checkmodel"):
        assert getattr(win, attr) in acts, attr
    caps = _captions(tb)
    for c in ("ANALYSE", "RESULTS", "DESIGN"):
        assert c in caps, (c, caps)


def test_ribbon_buttons_text_under_icon_but_menu_text_kept(win):
    from PySide6.QtCore import Qt
    tb = _toolbar(win, "Loads")
    btns = _ribbon_buttons(tb)
    assert btns
    for b in btns:
        assert b.toolButtonStyle() == \
            Qt.ToolButtonStyle.ToolButtonTextUnderIcon
    # The ribbon shows a short iconText; the menu still shows the full text.
    assert win.act_loadcases.iconText() == "Cases"
    assert win.act_loadcases.text() == "Load &cases…"


def test_load_actions_gathered_only_in_loads_ribbon(win):
    # The load actions live in the Loads ribbon group and are not duplicated in
    # the Home band (S2 pulled them off the Edit/Generate toolbars; S3 then
    # folded those bars into the ribbon — see test_desktop_ribbon.py).
    loads = set(_ribbon_actions(_toolbar(win, "Loads")))
    home = set(_ribbon_actions(_toolbar(win, "Home")))
    for attr in ("act_add_load", "act_add_lineload", "act_genloads"):
        assert getattr(win, attr) in loads, attr
        assert getattr(win, attr) not in home, attr


def test_ctrl_r_rides_a_ribbon_button(win):
    tb = _toolbar(win, "Analysis")
    assert win.act_run in _ribbon_actions(tb)
    assert win.act_run.shortcut().toString() == "Ctrl+R"
