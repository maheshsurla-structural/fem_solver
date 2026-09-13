"""Empty states (plan gui-polish **T5**) — the viewport no longer shows a blank
canvas when there is nothing to draw.

``ModelView`` carries a centred ``#canvasHint`` overlay that appears whenever the
model is absent or has no nodes (e.g. right after File ▸ New, which creates a
project with materials/sections but no geometry) and hides once a model with
nodes is drawn. (The Properties panel already has its own "select…" empty state.)
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


def test_hint_shows_when_no_model(qapp):
    from main_window import MainWindow
    w = MainWindow()
    # isHidden() reflects the explicit show/hide independent of window visibility
    assert not w.view._hint.isHidden()             # blank shell → hint shown
    assert "draw a node" in w.view._hint.text()


def test_hint_hides_with_a_model_then_returns_on_new(qapp):
    from demo_model import demo_project
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(demo_project())
    assert w.view._hint.isHidden()                 # model with nodes → no hint
    w.new_project()                                # empty project (no nodes)
    assert not w.view._hint.isHidden()             # empty again → hint returns


def test_hint_is_click_through(qapp):
    from PySide6.QtCore import Qt
    from main_window import MainWindow
    w = MainWindow()
    # the overlay must not swallow clicks meant for the canvas beneath it
    assert w.view._hint.testAttribute(
        Qt.WidgetAttribute.WA_TransparentForMouseEvents)
