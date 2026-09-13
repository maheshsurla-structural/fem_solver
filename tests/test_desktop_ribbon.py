"""Tabbed ribbon — CSiBridge-style top chrome (plan ribbon R1).

Headless (offscreen) coverage of the R1 consolidation: the classic menu bar and
the old three stacked ribbon rows (S1/S2/S3) are replaced by ONE compact strip —
a File "backstage" button plus a tab strip (Home · Draw · Loads · Analysis ·
Results · View) whose active tab swaps a single row of captioned tool-groups.

The load-bearing guarantee is *nothing is lost*: every ``act_*`` on the window
is reachable from exactly one ribbon button or the File menu (the two launched
only from the Analysis-cases dialog are the sole exceptions), Ctrl+R still rides
a ribbon button, and the ribbon buttons stay text-under-icon while the actions
keep their full text.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

# Actions that are deliberately not on the ribbon: they are launched from the
# Analysis-cases home dialog, not a top-level button.
_DIALOG_ONLY = {"act_pushover", "act_timehistory"}

TABS = ["Home", "Draw", "Loads", "Analysis", "Results", "View"]


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def win(qapp):
    from main_window import MainWindow
    return MainWindow()


def _page_buttons(win, title):
    from PySide6.QtWidgets import QToolButton
    page = win._ribbon.page(title)
    return [b for b in page.findChildren(QToolButton)
            if b.objectName() == "ribbonBtn"]


def _page_actions(win, title):
    return {b.defaultAction() for b in _page_buttons(win, title)}


def _all_ribbon_buttons(win):
    from PySide6.QtWidgets import QToolButton
    return [b for b in win.findChildren(QToolButton)
            if b.objectName() == "ribbonBtn"]


def _captions(win, title):
    from PySide6.QtWidgets import QLabel
    page = win._ribbon.page(title)
    return [lbl.text() for lbl in page.findChildren(QLabel)
            if lbl.objectName() == "ribbonCap"]


# ---- the strip replaces the menu bar --------------------------------------
def test_single_ribbon_strip_and_no_menu_bar(win):
    from main_window import RibbonBar
    assert isinstance(win._ribbon, RibbonBar)
    # The classic menu bar is gone from view — tabs are the top-level nav.
    assert not win.menuBar().isVisible()


def test_tabs_in_expected_order(win):
    rb = win._ribbon
    assert [rb.tabs.tabText(i) for i in range(rb.tabs.count())] == TABS
    assert rb.stack.count() == len(TABS)


def test_file_backstage_carries_file_actions(win):
    file_acts = set(win._ribbon.file_menu.actions())
    for attr in ("act_new", "act_new3d", "act_open", "act_save", "act_saveas"):
        assert getattr(win, attr) in file_acts, attr


# ---- per-tab placement (spot checks) --------------------------------------
def test_home_tab_groups(win):
    acts = _page_actions(win, "Home")
    for attr in ("act_add_node", "act_add_member", "act_add_section",
                 "act_materials", "act_undo", "act_redo", "act_delete",
                 "act_move", "act_copy", "act_mirror", "act_rotate",
                 "act_extrude", "act_gen", "act_sectiondesigner"):
        assert getattr(win, attr) in acts, attr
    for c in ("MODEL", "EDIT", "MODIFY", "GENERATE", "TOOLS"):
        assert c in _captions(win, "Home"), c


def test_draw_tab_folds_the_modal_palettes(win):
    acts = _page_actions(win, "Draw")
    for attr in ("act_draw_node", "act_draw_member", "act_snap", "act_select",
                 "act_sel_window", "act_sel_poly", "act_deselect",
                 "act_sel_all_nodes", "act_sel_all_members", "act_sel_all",
                 "act_sel_by_section"):
        assert getattr(win, attr) in acts, attr
    # the grid-snap spin box rides the Draw tab too
    assert win.snap_spin in win._ribbon.page("Draw").findChildren(type(win.snap_spin))


def test_loads_tab_groups(win):
    acts = _page_actions(win, "Loads")
    for attr in ("act_loadcases", "act_add_load", "act_add_lineload",
                 "act_genloads", "act_editcombos", "act_gencombos"):
        assert getattr(win, attr) in acts, attr
    assert "LOADS" in _captions(win, "Loads")
    assert "COMBINATIONS" in _captions(win, "Loads")


def test_analysis_tab_groups(win):
    acts = _page_actions(win, "Analysis")
    for attr in ("act_analysiscases", "act_runanalysis", "act_run",
                 "act_hinges", "act_assign_hinges"):
        assert getattr(win, attr) in acts, attr


def test_results_tab_groups(win):
    acts = _page_actions(win, "Results")
    for attr in ("act_undef", "act_diag_n", "act_diag_v", "act_diag_m",
                 "act_runhistory", "act_design", "act_checkmodel"):
        assert getattr(win, attr) in acts, attr


def test_view_tab_holds_orientation_and_appearance(win):
    acts = _page_actions(win, "View")
    for attr in ("act_fit", "act_v_iso", "act_v_top", "act_v_front",
                 "act_v_back", "act_v_left", "act_v_right", "act_v_bottom",
                 "act_drawings", "act_theme", "act_density"):
        assert getattr(win, attr) in acts, attr


# ---- the "nothing is lost" guarantee --------------------------------------
def test_every_action_is_homed_exactly_once(win):
    counts = Counter(b.defaultAction() for b in _all_ribbon_buttons(win))
    # no action is duplicated across ribbon buttons (keeps iconText unambiguous)
    assert [a.text() for a, c in counts.items() if c > 1] == []
    homed = set(counts) | set(win._ribbon.file_menu.actions())
    orphans = [n for n in dir(win)
               if n.startswith("act_") and n not in _DIALOG_ONLY
               and getattr(win, n) not in homed]
    assert orphans == [], orphans


def test_ribbon_buttons_text_under_icon_but_action_text_kept(win):
    from PySide6.QtCore import Qt
    for b in _all_ribbon_buttons(win):
        assert b.toolButtonStyle() == \
            Qt.ToolButtonStyle.ToolButtonTextUnderIcon
    # short label under the icon; the action keeps its full text
    assert win.act_loadcases.iconText() == "Cases"
    assert win.act_loadcases.text() == "Load &cases…"


def test_ctrl_r_rides_a_ribbon_button(win):
    assert win.act_run in _page_actions(win, "Analysis")
    assert win.act_run.shortcut().toString() == "Ctrl+R"


def test_switching_tabs_swaps_the_group_row(win):
    rb = win._ribbon
    rb.set_current("Home")
    assert rb.stack.currentIndex() == 0
    rb.tabs.setCurrentIndex(TABS.index("Loads"))
    assert rb.stack.currentIndex() == TABS.index("Loads")
    rb.set_current("Home")
