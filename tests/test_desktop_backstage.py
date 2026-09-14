"""File backstage overlay — plan ribbon R8 (``desktop/backstage.py``).

Headless (offscreen) coverage: the ribbon's File button opens a full-window
backstage (not a dropdown menu) that renders the file commands, lists recent
projects (a persisted MRU, filtered to files that still exist), and opens one.

Visibility is asserted with ``isHidden()`` (the explicit hide state) rather than
``isVisible()``, which is always False while the top-level window is never shown.
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


def _cmd_buttons(bs):
    from PySide6.QtWidgets import QToolButton
    return [b for b in bs.findChildren(QToolButton)
            if b.objectName() == "backstageCmd"]


def test_file_button_opens_backstage_not_a_menu(win):
    # R8: the File button no longer pops a menu — it opens the backstage overlay.
    assert win._ribbon.file_btn.menu() is None
    win._open_backstage()
    assert win._backstage is not None
    assert not win._backstage.isHidden()
    win._backstage.close_panel()
    assert win._backstage.isHidden()


def test_backstage_renders_the_file_commands(win):
    win._open_backstage()
    labels = [b.text() for b in _cmd_buttons(win._backstage)]
    assert len(labels) == 5
    assert labels[0] == "New" and labels[1] == "New 3-D frame"
    assert labels[2].startswith("Open") and labels[3] == "Save"
    assert labels[4].startswith("Save As")
    win._backstage.close_panel()


def test_backstage_covers_the_window(win):
    win.resize(1000, 600)
    win._open_backstage()
    assert win._backstage.geometry().size() == win.rect().size()
    win._backstage.close_panel()


def test_recent_files_are_tracked_listed_and_opened(win, tmp_path):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QListWidget
    from demo_model import demo_project
    s = QSettings("MidasStructural", "Desktop")
    try:
        s.setValue("recent/files", [])
        win.load_project(demo_project())
        a = str(tmp_path / "a.fsproj")
        b = str(tmp_path / "b.fsproj")
        win._write(a)
        win._write(b)                              # newest first
        recent = win._recent_files()
        assert [Path(p).name for p in recent] == ["b.fsproj", "a.fsproj"]
        win._open_backstage()
        assert win._backstage.findChild(QListWidget).count() == 2
        win._backstage.close_panel()
        win._open_recent(a)                        # open one from the list
        assert Path(win._path).name == "a.fsproj"
    finally:
        s.setValue("recent/files", [])


def test_missing_recent_files_are_filtered_out(win, tmp_path):
    from PySide6.QtCore import QSettings
    s = QSettings("MidasStructural", "Desktop")
    try:
        s.setValue("recent/files", [str(tmp_path / "gone.fsproj")])
        assert win._recent_files() == []           # a deleted file drops off
    finally:
        s.setValue("recent/files", [])
