"""The pre/post-processing mode switch — ``desktop/mode_switch.py`` and its
wiring into the shell (it replaced the retired viewport rotation lock).

Covers that the switch is pinned as the toolbar's fixed trailing widget, that a
segment click drives the ribbon workspace, that the ribbon drives the switch
back, and that the switch survives a toolbar rebuild (Customize / Reset).
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
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _current_tab(w) -> str:
    tabs = w._ribbon.tabs
    return tabs.tabText(tabs.currentIndex())


# --- placement --------------------------------------------------------------
def test_switch_is_the_toolbar_trailing_widget(qapp):
    from main_window import MainWindow
    from mode_switch import ModeSwitch
    w = MainWindow()
    bar = w.view._nav_bar
    assert isinstance(w._mode_switch, ModeSwitch)
    assert bar._trailing is w._mode_switch
    # never part of the editable layout or the command registry
    assert "modeSwitch" not in bar._registry
    assert w._mode_switch not in bar._buttons_by_id.values()


def test_switch_defaults_to_model(qapp):
    from main_window import MainWindow
    import mode_switch as ms
    w = MainWindow()
    assert w._mode_switch.mode() == ms.MODEL


# --- switch drives the ribbon ----------------------------------------------
def test_clicking_results_raises_results_tab(qapp):
    from main_window import MainWindow
    import mode_switch as ms
    w = MainWindow()
    w._ribbon.set_current("Home")
    w._mode_switch._buttons[ms.RESULTS].click()
    assert _current_tab(w) == "Results"
    assert w._mode_switch.mode() == ms.RESULTS


def test_clicking_model_from_results_returns_to_home(qapp):
    from main_window import MainWindow
    import mode_switch as ms
    w = MainWindow()
    w._ribbon.set_current("Results")
    w._mode_switch._buttons[ms.MODEL].click()
    assert _current_tab(w) == "Home"
    assert w._mode_switch.mode() == ms.MODEL


def test_model_click_keeps_a_modelling_tab_put(qapp):
    # if the user is already on a modelling tab, picking Model must not yank them
    # to Home — only a jump *out of* Results does that.
    from main_window import MainWindow
    import mode_switch as ms
    w = MainWindow()
    w._ribbon.set_current("Loads")
    w._mode_switch._buttons[ms.MODEL].click()
    assert _current_tab(w) == "Loads"


# --- ribbon drives the switch back -----------------------------------------
def test_raising_results_tab_reflects_on_switch(qapp):
    from main_window import MainWindow
    import mode_switch as ms
    w = MainWindow()
    w._ribbon.set_current("Results")           # any route (run auto-raise, click)
    assert w._mode_switch.mode() == ms.RESULTS
    w._ribbon.set_current("Draw")
    assert w._mode_switch.mode() == ms.MODEL


def test_set_mode_does_not_echo_a_signal(qapp):
    from mode_switch import ModeSwitch, RESULTS
    sw = ModeSwitch()
    seen = []
    sw.modeChanged.connect(seen.append)
    sw.set_mode(RESULTS)                        # programmatic reflect → no emit
    assert sw.mode() == RESULTS
    assert seen == []


# --- survives a toolbar rebuild --------------------------------------------
def test_trailing_widget_survives_rebuild(qapp):
    from main_window import MainWindow
    w = MainWindow()
    sw = w._mode_switch
    w.view._nav_bar.rebuild()                   # e.g. Customize / Reset
    assert w.view._nav_bar._trailing is sw
    assert sw.parent() is w.view._nav_bar
    assert sw.isVisibleTo(w.view._nav_bar)
