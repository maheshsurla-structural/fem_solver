"""Status-bar context (plan gui-polish **T4**) — ``desktop/main_window.py``.

The status bar used to only ``showMessage`` a transient model count that any
other message overwrote. T4 gives it persistent right-side context — model
summary, selection count, live cursor coords, and units — as permanent widgets,
leaving ``showMessage`` for transient hints.
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


@pytest.fixture(scope="module")
def win(qapp):
    from demo_model import demo_project
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(demo_project())
    return w


def test_model_summary_is_persistent_widget(win):
    from PySide6.QtWidgets import QLabel
    assert isinstance(win._st_model, QLabel)
    assert win._st_model in win.statusBar().findChildren(QLabel)
    txt = win._st_model.text()
    assert "9 nodes" in txt and "10 members" in txt and "2 loads" in txt


def test_units_readout(win):
    # demo project stores SI base units
    assert win._st_units.text() == "N · m"


def test_selection_readout_updates(win):
    win._update_sel_status(4)
    assert win._st_sel.text() == "4 selected"
    win._update_sel_status(0)
    assert win._st_sel.text() == ""


def test_cursor_coords_readout(win):
    win._on_cursor_coords(4.25, -1.5)
    assert win._st_coord.text() == "X 4.25  Y -1.50 m"


def test_units_readout_converts_when_not_si(win):
    """Plan U2 — the readouts must reflect the *chosen* units, not always SI."""
    p = win._project
    try:
        p.force_unit, p.length_unit = "kN", "mm"
        win._refresh_status()
        assert win._st_units.text() == "kN · mm"
        # 1.0 m of cursor travel shows as 1000 mm
        win._on_cursor_coords(1.0, -2.0)
        assert win._st_coord.text() == "X 1000.00  Y -2000.00 mm"
    finally:                                   # restore SI for the other tests
        p.force_unit, p.length_unit = "N", "m"
        win._refresh_status()


def test_viewport_coord_callback_wired(win):
    assert win.view._coord_cb is not None


def test_world_on_ground_is_safe_offscreen(win):
    from PySide6.QtCore import QPointF
    # offscreen the projection may be degenerate; it must never raise
    out = win.view._world_on_ground(QPointF(100.0, 100.0))
    assert out is None or (isinstance(out, tuple) and len(out) == 2)
