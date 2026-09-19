"""Wall-modeling W4 — the Story / Grid data model.

Covers project.Story / project.GridLine, the Project story/grid accessors
(sorting, elevation lookup, grid-by-axis) and the JSON round-trip + migration.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

from project import GridLine, Project, Story  # noqa: E402


def _proj() -> Project:
    p = Project(ndm=3, ndf=6)
    p.stories.extend([
        Story(id=1, name="Base", elev=0.0, height=0.0),
        Story(id=2, name="L1", elev=3.0, height=3.0),
        Story(id=3, name="Roof", elev=6.0, height=3.0),
    ])
    p.grid_lines.extend([
        GridLine(id=1, name="A", axis="x", coord=0.0),
        GridLine(id=2, name="B", axis="x", coord=6.0),
        GridLine(id=3, name="1", axis="y", coord=0.0),
    ])
    return p


# ---------------------------------------------------------------- dataclasses

def test_story_coercion():
    s = Story(id=1, name="L1", elev="3.0", height="3")
    assert s.elev == 3.0 and s.height == 3.0
    assert isinstance(s.elev, float)


def test_grid_axis_validated():
    assert GridLine(id=1, name="A", axis="z").axis == "x"   # bad → default x
    assert GridLine(id=2, name="B", axis="y").axis == "y"


# ------------------------------------------------------------- accessors

def test_stories_sorted_and_elevations():
    p = _proj()
    p.stories.reverse()                       # unsorted input
    assert [s.name for s in p.stories_sorted()] == ["Base", "L1", "Roof"]
    assert p.story_elevations() == [0.0, 3.0, 6.0]
    assert p.next_story_id() == 4
    assert p.next_grid_id() == 4


def test_story_at_elevation_band():
    p = _proj()
    assert p.story_at(0.0).name == "Base"
    assert p.story_at(1.5).name == "L1"        # within (0, 3] band
    assert p.story_at(3.0).name == "L1"
    assert p.story_at(4.5).name == "Roof"
    assert p.story_at(9.0) is None             # above the top level


def test_grid_lines_by_axis():
    p = _proj()
    assert [g.name for g in p.grid_lines_on("x")] == ["A", "B"]
    assert [g.name for g in p.grid_lines_on("y")] == ["1"]


# ------------------------------------------------------------- round-trip

def test_round_trip_preserves_stories_and_grid():
    p = _proj()
    q = Project.from_json(p.to_json())
    assert [s.name for s in q.stories_sorted()] == ["Base", "L1", "Roof"]
    assert q.story_at(4.5).name == "Roof"
    assert [g.coord for g in q.grid_lines_on("x")] == [0.0, 6.0]


def test_legacy_load_without_stories():
    legacy = {"schema": "femsolver-project/1", "name": "old", "ndm": 3,
              "ndf": 6, "nodes": []}
    p = Project.from_dict(legacy)
    assert p.stories == [] and p.grid_lines == []
    assert p.story_at(1.0) is None


# ------------------------------------------------------------- manager dialog

import os  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_story_grid_dialog_loads_and_parses(qapp):
    from story_grid_dialog import StoryGridDialog
    dlg = StoryGridDialog(None, _proj())
    assert dlg.st_tbl.rowCount() == 3
    assert dlg.gr_tbl.rowCount() == 3
    dlg._accept()
    stories, grids = dlg.result
    assert [s.name for s in stories] == ["Base", "L1", "Roof"]
    assert stories[2].elev == pytest.approx(6.0)
    assert {g.axis for g in grids} == {"x", "y"}


def test_story_grid_dialog_generate_stack(qapp):
    from story_grid_dialog import StoryGridDialog
    dlg = StoryGridDialog(None, Project(ndm=3, ndf=6))
    dlg.stories_from_stack(0.0, [3.0, 3.5, 3.5])
    assert [s.name for s in dlg._project.stories_sorted()] == \
        ["Base", "Story 1", "Story 2", "Story 3"]
    assert dlg._project.story_elevations() == pytest.approx([0.0, 3.0, 6.5, 10.0])


def test_story_grid_dialog_drops_unnamed_rows(qapp):
    from PySide6.QtWidgets import QTableWidgetItem
    from story_grid_dialog import StoryGridDialog
    dlg = StoryGridDialog(None, Project(ndm=3, ndf=6))
    dlg._add_story()
    dlg.st_tbl.setItem(0, 0, QTableWidgetItem("   "))   # blank name → dropped
    dlg._accept()
    stories, _ = dlg.result
    assert stories == []


def test_pier_forces_dialog_shows_story_column(qapp):
    """The pier-forces table gains a Story label from story_at (wall plan W4)."""
    import sys as _sys
    _sys.path.insert(0, str(_ROOT / "src"))
    from femsolver.analysis.linear_static import LinearStaticAnalysis
    import walls
    from pier_forces_dialog import PierForcesDialog
    from project import Material, Node, ShellSection

    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C", E=30e9, nu=0.2, rho=2400.0))
    p.shell_sections.append(ShellSection(id=1, name="W", thickness=0.25))
    p.nodes.extend([Node(id=1, x=0.0, y=0.0, z=0.0),
                    Node(id=2, x=3.0, y=0.0, z=0.0)])
    walls.build_wall_line(p, [1, 2], height=4.0, shell_section=1, material=1,
                          mesh=(2, 4), pier="P1")
    p.stories.extend([Story(id=1, name="Base", elev=0.0),
                      Story(id=2, name="L1", elev=2.0, height=2.0),
                      Story(id=3, name="Roof", elev=4.0, height=2.0)])
    m = p.build_model(with_loads=False)
    for tag, nd in m.nodes.items():
        if abs(nd.coords[2]) < 1e-9:
            m.fix(tag, [1, 1, 1, 1, 1, 1])
    tops = [t for t, nd in m.nodes.items() if abs(nd.coords[2] - 4.0) < 1e-9]
    for t in tops:
        m.add_nodal_load(t, [50_000.0 / len(tops), 0, 0, 0, 0, 0])
    LinearStaticAnalysis(m).run()

    dlg = PierForcesDialog(None, p, m)
    assert dlg.tbl.horizontalHeaderItem(0).text() == "Story"
    labels = {dlg.tbl.item(r, 0).text() for r in range(dlg.tbl.rowCount())}
    assert labels & {"Base", "L1", "Roof"}      # cuts labelled by story


# ------------------------------------------------- W8a: story parity + linking

def test_story_color_master_round_trip():
    p = Project(ndm=3, ndf=6)
    p.stories.extend([
        Story(id=1, name="Base", elev=0.0),
        Story(id=2, name="L1", elev=3.0, height=3.0, color="#ff8800"),
        Story(id=3, name="L2", elev=6.0, height=3.0, master="L1"),
    ])
    q = Project.from_json(p.to_json())
    assert q.story(2).color == "#ff8800"
    assert q.story(3).master == "L1"
    assert {s.name for s in q.similar_stories(q.story(2))} == {"L1", "L2"}


def _dlg(qapp):
    from story_grid_dialog import StoryGridDialog
    p = Project(ndm=3, ndf=6)
    p.stories.extend([Story(id=1, name="Base", elev=0.0, height=0.0),
                      Story(id=2, name="L1", elev=3.0, height=3.0),
                      Story(id=3, name="L2", elev=6.0, height=3.0)])
    return StoryGridDialog(None, p)


def test_height_edit_keeps_heights(qapp):
    d = _dlg(qapp)
    d.st_tbl.item(2, 2).setText("4")            # L2 height 3 → 4
    elevs = [d.st_tbl.item(r, 1).text() for r in range(3)]
    assert elevs == ["0", "3", "7"]             # L2 elevation recomputed 6 → 7


def test_elevation_edit_keeps_elevations(qapp):
    d = _dlg(qapp)
    d.st_tbl.item(1, 1).setText("4")            # L1 elevation 3 → 4
    # L1 height becomes 4 (from base), L2 height becomes 2 (6-4); elevations kept
    assert d.st_tbl.item(1, 2).text() == "4"
    assert d.st_tbl.item(2, 2).text() == "2"
    assert [d.st_tbl.item(r, 1).text() for r in range(3)] == ["0", "4", "6"]


def test_base_move_shifts_stack(qapp):
    d = _dlg(qapp)
    d.st_tbl.item(0, 1).setText("1.5")          # move datum up
    assert [d.st_tbl.item(r, 1).text() for r in range(3)] == ["1.5", "4.5", "7.5"]


def test_similar_to_combo_sets_master(qapp):
    from PySide6.QtWidgets import QComboBox
    d = _dlg(qapp)
    combo = d.st_tbl.cellWidget(2, 3)           # L2 'similar to'
    assert isinstance(combo, QComboBox)
    combo.setCurrentIndex(combo.findData("L1"))
    d._accept()
    stories, _ = d.result
    by = {s.name: s for s in stories}
    assert by["L2"].master == "L1"
    # a lone story stays independent
    assert by["L1"].master is None


def test_story_color_pick(qapp, monkeypatch):
    import story_grid_dialog as sgd
    from PySide6.QtGui import QColor
    d = _dlg(qapp)
    monkeypatch.setattr(sgd.QColorDialog, "getColor",
                        staticmethod(lambda *a, **k: QColor("#123456")))
    d._on_story_double_click(1, 4)              # pick a colour for L1
    d._accept()
    stories, _ = d.result
    assert {s.name: s.color for s in stories}["L1"] == "#123456"
