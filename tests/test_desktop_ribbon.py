"""Unified ribbon band (plan S3) — ``desktop/main_window.py``.

Headless (offscreen) coverage of the S3 consolidation: the scattered icon
toolbars (File/Edit/Generate/View/Tools) are folded into one captioned ribbon
band — a **Home** row over the S2 **Loads** and **Analysis** rows — while the
modal draw/select tool palettes stay as icon-only toolbars and the menus keep
the full action sets (nothing is lost).
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


def _ribbon_actions(tb):
    from PySide6.QtWidgets import QToolButton
    return [b.defaultAction() for b in tb.findChildren(QToolButton)
            if b.objectName() == "ribbonBtn"]


def _captions(tb):
    from PySide6.QtWidgets import QLabel
    return [lbl.text() for lbl in tb.findChildren(QLabel)
            if lbl.objectName() == "ribbonCap"]


def _menu(win, title):
    for a in win.menuBar().actions():
        if a.menu() is not None and a.text().replace("&", "") == title:
            return a.menu()
    return None


def _menu_actions(menu):
    out = []
    for a in menu.actions():
        if a.isSeparator():
            continue
        sub = a.menu()
        out.extend(_menu_actions(sub) if sub is not None else [a])
    return out


def test_home_ribbon_row_groups(win):
    tb = _toolbar(win, "Home")
    assert tb is not None and tb.objectName() == "ribbonBar"
    acts = set(_ribbon_actions(tb))
    for attr in ("act_new", "act_open", "act_save", "act_add_node",
                 "act_add_member", "act_add_section", "act_undo", "act_redo",
                 "act_fit", "act_v_iso", "act_sectiondesigner"):
        assert getattr(win, attr) in acts, attr
    caps = _captions(tb)
    for c in ("FILE", "MODEL", "EDIT", "VIEW", "TOOLS"):
        assert c in caps, (c, caps)


def test_band_is_three_ribbon_rows(win):
    from PySide6.QtWidgets import QToolBar
    ribbons = [tb for tb in win.findChildren(QToolBar)
               if tb.objectName() == "ribbonBar"]
    titles = {tb.windowTitle() for tb in ribbons}
    assert {"Home", "Loads", "Analysis"} <= titles, titles


def test_old_icon_toolbars_folded_in(win):
    # The scattered per-area icon toolbars are gone (their common actions now
    # ride the ribbon; the full sets stay in the menus).
    for title in ("File", "Edit", "Generate", "View", "Tools"):
        assert _toolbar(win, title) is None, title


def test_modal_tool_palettes_remain_icon_only(win):
    from PySide6.QtWidgets import QToolButton
    for title, members in (("Select", ("act_select", "act_sel_window",
                                       "act_sel_poly", "act_deselect")),
                           ("Draw", ("act_draw_node", "act_draw_member",
                                     "act_snap"))):
        tb = _toolbar(win, title)
        assert tb is not None, title
        acts = tb.actions()
        for attr in members:
            assert getattr(win, attr) in acts, (title, attr)
        # icon-only palette: no captioned ribbon buttons here
        assert not any(b.objectName() == "ribbonBtn"
                       for b in tb.findChildren(QToolButton)), title


def test_menus_keep_the_full_action_sets(win):
    # Actions the ribbon curates away are still reachable in the menus.
    edit = set(_menu_actions(_menu(win, "Edit")))
    for attr in ("act_materials", "act_move", "act_copy", "act_mirror",
                 "act_rotate", "act_extrude", "act_delete"):
        assert getattr(win, attr) in edit, attr
    assert win.act_gen in set(_menu_actions(_menu(win, "Generate")))
    view = set(_menu_actions(_menu(win, "View")))
    for attr in ("act_v_top", "act_v_front", "act_drawings"):
        assert getattr(win, attr) in view, attr
