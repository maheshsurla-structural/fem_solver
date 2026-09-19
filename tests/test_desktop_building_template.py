"""Wall-modeling W8c — building quick template (uniform grid + simple stories)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

import building_template as bt  # noqa: E402
from project import Project  # noqa: E402


# ------------------------------------------------------------- label sequence

def test_label_seq_letters_and_numbers():
    assert bt._label_seq("A", 4) == ["A", "B", "C", "D"]
    assert bt._label_seq("1", 4) == ["1", "2", "3", "4"]
    assert bt._label_seq("Z", 2) == ["Z", "AA"]
    assert bt._label_seq("C", 2) == ["C", "D"]


# ------------------------------------------------------------- generator

def test_build_grid_stories_grid():
    stories, grids = bt.build_grid_stories(
        nx=4, ny=3, sx=6.0, sy=5.0, n_stories=3, typ_h=3.5, bot_h=4.5)
    xg = [g for g in grids if g.axis == "x"]
    yg = [g for g in grids if g.axis == "y"]
    assert [g.name for g in xg] == ["A", "B", "C", "D"]
    assert [g.coord for g in xg] == [0.0, 6.0, 12.0, 18.0]
    assert [g.name for g in yg] == ["1", "2", "3"]
    assert [g.coord for g in yg] == [0.0, 5.0, 10.0]


def test_build_grid_stories_story_stack():
    stories, _ = bt.build_grid_stories(
        nx=2, ny=2, sx=6, sy=6, n_stories=3, typ_h=3.5, bot_h=4.5)
    assert [s.name for s in stories] == ["Base", "Story 1", "Story 2", "Story 3"]
    # elevations: 0, 4.5 (bottom), 8.0, 11.5 (typical)
    assert [s.elev for s in stories] == pytest.approx([0.0, 4.5, 8.0, 11.5])
    assert [s.height for s in stories] == pytest.approx([0.0, 4.5, 3.5, 3.5])


def test_bottom_height_defaults_to_typical():
    stories, _ = bt.build_grid_stories(2, 2, 6, 6, 2, 3.0)
    assert [s.elev for s in stories] == pytest.approx([0.0, 3.0, 6.0])


def test_ids_unique():
    stories, grids = bt.build_grid_stories(3, 3, 6, 6, 2, 3.0)
    assert len({s.id for s in stories}) == len(stories)
    assert len({g.id for g in grids}) == len(grids)


# ------------------------------------------------------------- dialog / action

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    pytest.importorskip("matplotlib")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_template_dialog_params_and_preview(qapp):
    from building_template_dialog import GridStoryTemplateDialog
    dlg = GridStoryTemplateDialog(None)
    dlg.nx.setValue(4)
    dlg.ny.setValue(4)
    dlg.n_st.setValue(4)
    dlg._redraw()                                   # exercises the live preview
    dlg._accept()
    stories, grids = dlg.result
    assert len(stories) == 5                         # Base + 4
    assert len(grids) == 8                           # 4 X + 4 Y


@pytest.fixture(scope="module")
def qapp_vtk():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvistaqt")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_new_building_template_action(qapp_vtk, monkeypatch):
    from main_window import MainWindow
    import building_template_dialog as btd
    import building_template as _bt
    w = MainWindow()
    w.load_project(Project(ndm=3, ndf=6))
    monkeypatch.setattr(
        btd.GridStoryTemplateDialog, "get",
        classmethod(lambda cls, *a, **k: _bt.build_grid_stories(
            3, 3, 6, 6, 4, 3.5, 4.5)))
    w.new_building_template()
    assert len(w._project.stories) == 5             # Base + 4
    assert len(w._project.grid_lines) == 6          # 3 X + 3 Y
    # active-story combo now populated (drawable stories)
    assert w.story_combo.count() == 4
    w._undo_stack.undo()
    assert w._project.stories == []
