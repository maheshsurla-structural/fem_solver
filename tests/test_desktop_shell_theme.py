"""Shell theming (plan gui-polish **T1–T3**) — ``desktop/main_window.py``.

T1: the shell's icons were the last theme-blind ones (bare ``icons.icon(name)``
at the hard-coded ``#3a3a3a``). They now render in ``style.ICON`` and remember
their icon name so ``_retheme_icons()`` can recolour them on a theme switch.
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


def _store():
    from PySide6.QtCore import QSettings
    return QSettings("MidasStructural", "Desktop")


@pytest.fixture(autouse=True)
def _reset_theme():
    import style
    yield
    style.set_theme("light")
    style.set_density("comfortable")
    _store().setValue("theme", "light")        # keep the shared store clean
    _store().setValue("density", "comfortable")


@pytest.fixture(scope="module")
def win(qapp):
    import style
    _store().setValue("theme", "light")        # deterministic light startup
    _store().setValue("density", "comfortable")
    style.set_theme("light")
    from main_window import MainWindow
    return MainWindow()


# ---- T1 -------------------------------------------------------------------
def test_shell_icon_actions_remember_their_name(win):
    from PySide6.QtGui import QAction
    named = {a.property("iconName") for a in win.findChildren(QAction)
             if a.property("iconName")}
    # a representative spread across File / Model / Analysis / Draw
    for expected in ("new", "open", "save", "run", "single", "drawnode",
                     "moment", "design", "snap"):
        assert expected in named, f"{expected} not themed via _set_icon"


def test_no_theme_blind_icon_calls_in_shell():
    src = (_ROOT / "desktop" / "main_window.py").read_text(encoding="utf-8")
    # every icons.icon(...) must pass an ink argument (style.ICON) — no bare
    # single-arg call left at the hard-coded default.
    import re
    bare = re.findall(r'icons\.icon\(\s*("[a-z0-9_]+"|[a-z_]+)\s*\)', src)
    assert not bare, f"theme-blind icons.icon() calls: {bare}"


def test_retheme_icons_reinks_on_theme_change(win):
    import style
    style.set_theme("light")
    win._retheme_icons()
    before = win.act_run.icon().cacheKey()
    style.set_theme("dark")
    win._retheme_icons()
    after = win.act_run.icon().cacheKey()
    assert before != after                     # icon re-inked for the dark theme


# ---- T2 -------------------------------------------------------------------
def test_theme_toggle_flips_and_persists(win):
    import style
    style.set_theme("light")
    win._apply_theme_density()
    win.toggle_theme()
    assert style.current_theme() == "dark"
    assert str(_store().value("theme")) == "dark"      # choice persisted
    win.toggle_theme()
    assert style.current_theme() == "light"


def test_density_toggle_flips_and_syncs_action(win):
    import style
    style.set_density("comfortable")
    win.toggle_density()
    assert style.current_density() == "compact"
    assert win.act_density.isChecked()
    assert str(_store().value("density")) == "compact"
    win.toggle_density()
    assert style.current_density() == "comfortable"
    assert not win.act_density.isChecked()


def test_theme_controls_are_reachable(win):
    from PySide6.QtWidgets import QToolButton

    # theme + density live on the ribbon's View tab (Appearance group)
    view_acts = {b.defaultAction() for b in
                 win._ribbon.page("View").findChildren(QToolButton)
                 if b.objectName() == "ribbonBtn"}
    assert win.act_theme in view_acts
    assert win.act_density in view_acts
    # the always-visible status-bar chip
    assert isinstance(win._theme_btn, QToolButton)
    assert win._theme_btn in win.statusBar().findChildren(QToolButton)


def test_persisted_theme_restored_on_construction(qapp):
    import style
    from main_window import MainWindow
    _store().setValue("theme", "dark")
    try:
        w = MainWindow()
        assert style.current_theme() == "dark"     # came up in the saved theme
        del w
    finally:
        _store().setValue("theme", "light")
        style.set_theme("light")


# ---- T3 -------------------------------------------------------------------
def test_window_has_product_icon(win):
    from demo_model import demo_project
    assert not win.windowIcon().isNull()           # branded titlebar / taskbar
    win.load_project(demo_project())
    assert win.windowTitle().endswith("(preview)")  # title convention intact
