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


def test_every_ribbon_button_has_an_icon(win):
    # R6: no text-only ribbon button — every one carries a (themed) glyph.
    # File is a backstage menu (not a tab); every tab button carries a glyph.
    naked = [(t, b.defaultAction().iconText())
             for t in TABS
             for b in _page_buttons(win, t)
             if b.defaultAction().icon().isNull()]
    assert naked == [], naked


def test_switching_tabs_swaps_the_group_row(win):
    rb = win._ribbon
    rb.set_current("Home")
    assert rb.stack.currentIndex() == 0
    rb.tabs.setCurrentIndex(TABS.index("Loads"))
    assert rb.stack.currentIndex() == TABS.index("Loads")
    rb.set_current("Home")


# ---- R2: persist / restore the active tab ---------------------------------
def test_active_tab_persists_and_restores(qapp):
    from PySide6.QtCore import QSettings
    from main_window import MainWindow
    s = QSettings("MidasStructural", "Desktop")
    try:
        s.setValue("ribbon/tab", "Loads")
        w = MainWindow()                              # reopens where left off
        assert w._ribbon.tabs.tabText(w._ribbon.tabs.currentIndex()) == "Loads"
        w._ribbon.set_current("Results")              # a switch re-persists
        assert str(s.value("ribbon/tab")) == "Results"
    finally:
        s.setValue("ribbon/tab", "Home")             # keep the shared store clean


def test_unknown_saved_tab_falls_back_to_home(qapp):
    from PySide6.QtCore import QSettings
    from main_window import MainWindow
    s = QSettings("MidasStructural", "Desktop")
    try:
        s.setValue("ribbon/tab", "Nonexistent")
        w = MainWindow()
        assert w._ribbon.tabs.tabText(w._ribbon.tabs.currentIndex()) == "Home"
    finally:
        s.setValue("ribbon/tab", "Home")


# ---- R3: keyboard access ---------------------------------------------------
def test_keyboard_tab_accelerators_exist(win):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QShortcut
    seqs = {sc.key().toString() for sc in win.findChildren(QShortcut)}
    for k in ("Alt+H", "Alt+D", "Alt+L", "Alt+A", "Alt+R", "Alt+V",
              "Ctrl+Tab", "Ctrl+Shift+Tab"):
        assert k in seqs, k
    assert win._ribbon.tabs.focusPolicy() == Qt.FocusPolicy.TabFocus


def test_cycle_ribbon_tab_wraps(win):
    rb = win._ribbon
    rb.set_current("View")                            # last tab
    win._cycle_ribbon_tab(1)
    assert rb.tabs.currentIndex() == 0                # wraps to first
    win._cycle_ribbon_tab(-1)
    assert rb.tabs.currentIndex() == len(TABS) - 1    # wraps back to last
    rb.set_current("Home")


# ---- R4: contextual tab raising -------------------------------------------
def _tab(win):
    return win._ribbon.tabs.tabText(win._ribbon.tabs.currentIndex())


def test_run_raises_results_tab(qapp):
    from main_window import MainWindow
    from demo_model import demo_project
    w = MainWindow()
    w.load_project(demo_project())
    w._ribbon.set_collapsed(False)
    w._ribbon.set_current("Analysis")
    assert w.run_linear_static() is not None           # solvable demo frame
    assert _tab(w) == "Results"


def test_draw_tool_raises_draw_tab(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w._ribbon.set_collapsed(False)
    w._ribbon.set_current("Home")
    w._set_mode("draw_node")
    assert _tab(w) == "Draw"


def test_contextual_raise_is_suppressed_when_collapsed(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w._ribbon.set_current("Home")
    w._ribbon.set_collapsed(True)
    w._show_results_tab()                              # deliberately hidden → no jump
    assert _tab(w) == "Home"
    w._ribbon.set_collapsed(False)


# ---- R5: collapse / expand -------------------------------------------------
def test_collapse_hides_group_row(win):
    rb = win._ribbon
    rb.set_collapsed(False)
    assert not rb.is_collapsed() and not rb.stack.isHidden()
    rb.set_collapsed(True)
    assert rb.is_collapsed() and rb.stack.isHidden()   # just the tab strip left
    rb.toggle_collapsed()
    assert not rb.is_collapsed() and not rb.stack.isHidden()


def test_collapsed_state_persists(qapp):
    from PySide6.QtCore import QSettings
    from main_window import MainWindow
    s = QSettings("MidasStructural", "Desktop")
    try:
        s.setValue("ribbon/collapsed", True)
        w = MainWindow()
        assert w._ribbon.is_collapsed()               # reopened collapsed
    finally:
        s.setValue("ribbon/collapsed", False)          # keep the shared store clean


def test_transient_reveal_then_recollapse(win):
    rb = win._ribbon
    rb.set_collapsed(True)
    rb._on_tab_clicked(1)                              # a click peeks the row
    assert not rb.stack.isHidden()
    rb._end_transient()                               # click-away re-collapses
    assert rb.stack.isHidden()
    rb.set_collapsed(False)


def test_ctrl_f1_collapse_shortcut(win):
    from PySide6.QtGui import QShortcut
    seqs = {sc.key().toString() for sc in win.findChildren(QShortcut)}
    assert "Ctrl+F1" in seqs


# ---- R7: width overflow ----------------------------------------------------
def test_narrow_width_collapses_groups_into_overflow(win):
    from main_window import RibbonPage
    pg = win._ribbon.page("View")                     # 4 groups, the widest tab
    assert isinstance(pg, RibbonPage)
    nat = sum(pg._group_width(g) for g in pg._groups) + 64
    pg.resize(nat + 200, pg.height())
    pg._relayout()
    assert pg.hidden_captions() == [] and pg._more.isHidden()   # all fit
    pg.resize(360, pg.height())
    pg._relayout()
    assert pg.hidden_captions()                        # something spilled
    assert not pg._more.isHidden()                     # the '»' button appears
    assert pg._shown >= 1                              # one group always stays
    assert "Appearance" in pg.hidden_captions()        # rightmost drops first
    pg.resize(nat + 200, pg.height())
    pg._relayout()
    assert pg.hidden_captions() == [] and pg._more.isHidden()   # reverses


def test_overflow_popup_lists_hidden_group_actions(win):
    pg = win._ribbon.page("View")
    pg.resize(360, pg.height())
    pg._relayout()
    pg._fill_overflow()
    menu_acts = {a for a in pg._more_menu.actions() if not a.isSeparator()}
    assert win.act_theme in menu_acts or win.act_drawings in menu_acts
    pg.resize(1600, pg.height())                       # restore for other tests
    pg._relayout()


def test_overflow_button_is_not_a_ribbon_button(win):
    # the '»' control is not counted among the tab's command buttons
    assert win._ribbon.page("View")._more.objectName() == "ribbonMore"
