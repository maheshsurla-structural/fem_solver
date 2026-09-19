"""Wall-modeling W7 — story-based plan wall drawing + story scope.

Covers Project.story_below / similar_stories, the plan wall builders
(build_wall_between / build_wall_stack), the MainWindow scope resolution
(One / Similar / All) and the plan-draw callback.
"""
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

import walls  # noqa: E402
from project import Material, Node, Project, ShellSection, Story  # noqa: E402


def _tower():
    """Base + 3 stories; L2/L3 are 'similar to' L1 via master='L1'."""
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C", E=30e9, nu=0.2, rho=2400.0))
    p.shell_sections.append(ShellSection(id=1, name="W", thickness=0.25))
    p.stories.extend([
        Story(id=1, name="Base", elev=0.0, height=0.0),
        Story(id=2, name="L1", elev=3.0, height=3.0),
        Story(id=3, name="L2", elev=6.0, height=3.0, master="L1"),
        Story(id=4, name="L3", elev=9.0, height=3.0, master="L1"),
    ])
    return p


# ------------------------------------------------------------- story helpers

def test_story_below():
    p = _tower()
    assert p.story_below(p.story(3)).name == "L1"       # below L2 is L1
    assert p.story_below(p.story(2)).name == "Base"
    assert p.story_below(p.story(1)) is None            # base has none


def test_similar_stories_keyed_off_master():
    p = _tower()
    grp = {s.name for s in p.similar_stories(p.story(2))}   # L1 is the master
    assert grp == {"L1", "L2", "L3"}
    # a lone story (Base, its own key) groups only with itself
    assert [s.name for s in p.similar_stories(p.story(1))] == ["Base"]


# ------------------------------------------------------------- plan builders

def test_build_wall_between_spans_story():
    p = _tower()
    aid = walls.build_wall_between(p, (0, 0), (4, 0), top_elev=3.0,
                                   bottom_elev=0.0, shell_section=1, material=1,
                                   pier="P1")
    a = p.area(aid)
    assert a.role == "wall" and a.pier == "P1"
    zs = sorted(next(n for n in p.nodes if n.id == nid).z for nid in a.nodes)
    assert zs == [0.0, 0.0, 3.0, 3.0]                  # bottom edge + top edge
    # first two corners are the bottom edge (build_wall_line convention)
    bottom = [next(n for n in p.nodes if n.id == nid) for nid in a.nodes[:2]]
    assert all(abs(n.z) < 1e-9 for n in bottom)


def test_build_wall_between_rejects_degenerate():
    p = _tower()
    assert walls.build_wall_between(p, (0, 0), (0, 0), 3.0, 0.0, 1, 1) is None
    assert walls.build_wall_between(p, (0, 0), (4, 0), 3.0, 3.0, 1, 1) is None


def test_build_wall_stack_is_continuous():
    p = _tower()
    levels = [(3.0, 0.0), (6.0, 3.0), (9.0, 6.0)]
    ids = walls.build_wall_stack(p, (0, 0), (4, 0), levels, 1, 1, pier="P1")
    assert len(ids) == 3
    # 4 elevations × 2 plan points = 8 unique nodes (shared between panels)
    zs = sorted({n.z for n in p.nodes})
    assert zs == [0.0, 3.0, 6.0, 9.0]
    assert len(p.nodes) == 8
    assert all(p.area(i).pier == "P1" for i in ids)     # pier runs up the stack
    # every panel builds as shells
    m = p.build_model(with_loads=False)
    from femsolver.elements.shell import ShellMITC4
    assert sum(isinstance(e, ShellMITC4) for e in m.elements.values()) == 3 * 4


# ------------------------------------------------------------- MainWindow scope

@pytest.fixture(scope="module")
def qapp_vtk():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvistaqt")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _win(qapp_vtk, project):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(project)
    return w


def test_story_combo_lists_only_drawable(qapp_vtk):
    w = _win(qapp_vtk, _tower())
    labels = [w.story_combo.itemText(i) for i in range(w.story_combo.count())]
    # Base excluded (no story below); L1/L2/L3 present
    assert w.story_combo.count() == 3
    assert not any("Base" in t for t in labels)


def _set_active(w, name):
    for i in range(w.story_combo.count()):
        if name in w.story_combo.itemText(i):
            w.story_combo.setCurrentIndex(i)
            return


def test_scope_levels_one_similar_all(qapp_vtk):
    w = _win(qapp_vtk, _tower())
    _set_active(w, "L1")
    w.scope_combo.setCurrentIndex(w.scope_combo.findData("one"))
    assert w._wall_plan_levels() == [(3.0, 0.0)]
    w.scope_combo.setCurrentIndex(w.scope_combo.findData("similar"))
    assert sorted(w._wall_plan_levels()) == [(3.0, 0.0), (6.0, 3.0), (9.0, 6.0)]
    w.scope_combo.setCurrentIndex(w.scope_combo.findData("all"))
    assert sorted(w._wall_plan_levels()) == [(3.0, 0.0), (6.0, 3.0), (9.0, 6.0)]


def test_active_story_sets_work_plane(qapp_vtk):
    w = _win(qapp_vtk, _tower())
    _set_active(w, "L2")
    assert w.view._work_plane == "xy"
    assert w.view._work_offset == pytest.approx(6.0)


def test_draw_wall_plan_points_builds_stack_and_undoes(qapp_vtk):
    w = _win(qapp_vtk, _tower())
    _set_active(w, "L1")
    w.scope_combo.setCurrentIndex(w.scope_combo.findData("all"))
    n0 = len(w._project.areas)
    w._draw_wall_plan_points((0.0, 0.0), (4.0, 0.0))
    assert len(w._project.walls()) == 3            # one per drawable story
    assert len(w._project.areas) == n0 + 3
    w._undo_stack.undo()
    assert len(w._project.areas) == n0


def test_enter_wall_plan_mode_guarded_without_stories(qapp_vtk, monkeypatch):
    import main_window as mw
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C", E=30e9, nu=0.2))
    p.shell_sections.append(ShellSection(id=1, name="W", thickness=0.25))
    w = _win(qapp_vtk, p)
    seen = {}
    monkeypatch.setattr(mw.QMessageBox, "information",
                        lambda *a, **k: seen.setdefault("info", a))
    w._enter_wall_plan_mode()
    assert "info" in seen
    assert w.view._mode != "draw_wall_plan"
