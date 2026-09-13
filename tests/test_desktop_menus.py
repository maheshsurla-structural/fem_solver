"""Menu information architecture (plan S1) — ``desktop/main_window.py``.

Headless (offscreen) coverage of the S1 menu reorg: a dedicated **Loads** menu
gathering every load-definition surface, and a slimmed **Analysis** menu
(cases -> run -> results -> design/check). Asserts the new placement, that the
scattered load actions left Edit/Generate, and that no action object or the
Ctrl+R linear-static shortcut was lost in the move.
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


def _title(action) -> str:
    return action.text().replace("&", "")


def _menubar_titles(w) -> list[str]:
    return [_title(a) for a in w.menuBar().actions() if a.menu() is not None]


def _menu(w, title):
    for a in w.menuBar().actions():
        m = a.menu()
        if m is not None and _title(a) == title:
            return m
    return None


def _flatten(menu) -> list:
    """All leaf actions of a menu, descending into submenus."""
    acts = []
    for a in menu.actions():
        if a.isSeparator():
            continue
        sub = a.menu()
        if sub is not None:
            acts.extend(_flatten(sub))
        else:
            acts.append(a)
    return acts


@pytest.fixture(scope="module")
def win(qapp):
    from main_window import MainWindow
    return MainWindow()


def test_loads_menu_exists_before_analysis(win):
    titles = _menubar_titles(win)
    assert "Loads" in titles, titles
    assert "Analysis" in titles, titles
    # Reads Generate -> Loads -> Analysis: define geometry, then loads, then run.
    assert titles.index("Generate") < titles.index("Loads") < titles.index(
        "Analysis")


def test_loads_menu_gathers_every_load_surface(win):
    loads = set(_flatten(_menu(win, "Loads")))
    for attr in ("act_loadcases", "act_add_load", "act_add_lineload",
                 "act_genloads", "act_editcombos", "act_gencombos"):
        assert getattr(win, attr) in loads, attr


def test_analysis_menu_is_slim(win):
    analysis = _menu(win, "Analysis")
    flat = set(_flatten(analysis))
    # The define -> run -> view -> design/check spine.
    for attr in ("act_analysiscases", "act_runanalysis", "act_undef",
                 "act_diag_n", "act_diag_v", "act_diag_m", "act_runhistory",
                 "act_design", "act_checkmodel"):
        assert getattr(win, attr) in flat, attr
    # A dedicated "Results & diagrams" submenu holds the view actions.
    submenus = [a.menu() for a in analysis.actions() if a.menu() is not None]
    assert any("Results" in m.title() for m in submenus), \
        [m.title() for m in submenus]
    # The load-definition actions moved out to the Loads home.
    for attr in ("act_loadcases", "act_editcombos", "act_gencombos"):
        assert getattr(win, attr) not in flat, attr
    # Direct nonlinear launchers now live in the Analysis-cases home, and the
    # bare linear-static action is reached via Run / the toolbar, not the menu.
    for attr in ("act_pushover", "act_timehistory", "act_run"):
        assert getattr(win, attr) not in flat, attr


def test_load_actions_left_edit_and_generate(win):
    edit = set(_flatten(_menu(win, "Edit")))
    assert win.act_add_load not in edit
    assert win.act_add_lineload not in edit
    gen = set(_flatten(_menu(win, "Generate")))
    assert win.act_genloads not in gen
    assert win.act_gen in gen           # geometry generation stays


def test_no_action_lost_and_ctrl_r_survives(win):
    from PySide6.QtWidgets import QToolButton
    # Every action the reorg touched still exists on the window.
    for attr in ("act_run", "act_pushover", "act_timehistory", "act_loadcases",
                 "act_add_load", "act_add_lineload", "act_genloads",
                 "act_editcombos", "act_gencombos"):
        assert hasattr(win, attr), attr
    # act_run left the menu but a toolbar button still carries it (a ribbon
    # button's default action after S2), so Ctrl+R stays live window-wide.
    carried = any(b.defaultAction() is win.act_run
                  for b in win.findChildren(QToolButton))
    assert carried
    assert win.act_run.shortcut().toString() == "Ctrl+R"
