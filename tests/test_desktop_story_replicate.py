"""Wall-modeling W4c — similar-story replication.

Covers story_replicate.replicate_story (band selection, translated copy, node
merging, pier labels running up the building) headless, then the dialog + the
MainWindow action wiring.
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

import story_replicate as sr  # noqa: E402
import walls  # noqa: E402
from project import (Area, Material, Member, Node, Project, Section,  # noqa: E402
                     ShellSection, Story)


def _one_story_building():
    """Ground story (0–3 m): one wall panel + one column, with two stories above
    it defined but empty."""
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C", E=30e9, nu=0.2, rho=2400.0))
    p.sections.append(Section(id=1, name="COL", A=0.16, Iz=2e-3, Iy=2e-3, J=1e-3))
    p.shell_sections.append(ShellSection(id=1, name="W", thickness=0.25))
    # column at (0,0) from z=0 to 3
    p.nodes.extend([Node(id=1, x=0.0, y=0.0, z=0.0),
                    Node(id=2, x=0.0, y=0.0, z=3.0)])
    p.members.append(Member(id=1, n1=1, n2=2, section=1, material=1))
    # wall panel base line (3,0,0)-(6,0,0) extruded up 3 m, pier P1
    p.nodes.extend([Node(id=3, x=3.0, y=0.0, z=0.0),
                    Node(id=4, x=6.0, y=0.0, z=0.0)])
    walls.build_wall_line(p, [3, 4], height=3.0, shell_section=1, material=1,
                          mesh=(2, 2), pier="P1")
    p.stories.extend([Story(id=1, name="S1", elev=3.0, height=3.0),
                      Story(id=2, name="S2", elev=6.0, height=3.0),
                      Story(id=3, name="S3", elev=9.0, height=3.0)])
    return p


# ------------------------------------------------------------ band selection

def test_objects_in_band():
    p = _one_story_building()
    nodes, members, areas = sr.objects_in_band(p, 0.0, 3.0)
    assert 1 in nodes and 2 in nodes            # the column's ends
    assert members == {1}
    assert len(areas) == 1                      # the wall panel


# ------------------------------------------------------------ replication

def test_replicate_to_two_stories():
    p = _one_story_building()
    n_area0 = len(p.areas)
    made = sr.replicate_story(p, source_id=1, target_ids=[2, 3])
    assert made["areas"] == 2                   # wall copied to S2 and S3
    assert made["members"] == 2                 # column copied to S2 and S3
    assert len(p.areas) == n_area0 + 2
    # every copied wall keeps the pier label → the pier runs up the building
    assert all(a.pier == "P1" for a in p.walls())
    # a copied column now reaches z = 9
    assert any(abs(n.z - 9.0) < 1e-9 for n in p.nodes)


def test_replication_merges_shared_column_line():
    """The column top of S1 (z=3) is the base of the S2 copy — they must share
    one node, not stack two."""
    p = _one_story_building()
    sr.replicate_story(p, source_id=1, target_ids=[2])
    zs = sorted(n.z for n in p.nodes if abs(n.x) < 1e-9 and abs(n.y) < 1e-9)
    # column line nodes at z = 0, 3, 6 — exactly three, no duplicate at z=3
    assert zs == [0.0, 3.0, 6.0]


def test_replicate_unknown_source_is_noop():
    p = _one_story_building()
    before = (len(p.nodes), len(p.members), len(p.areas))
    made = sr.replicate_story(p, source_id=999, target_ids=[2])
    assert made == {"nodes": 0, "members": 0, "areas": 0}
    assert (len(p.nodes), len(p.members), len(p.areas)) == before


# ------------------------------------------------------------ dialog / action

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_replicate_dialog_defaults_to_above(qapp):
    from story_replicate_dialog import StoryReplicateDialog
    from PySide6.QtCore import Qt
    p = _one_story_building()
    dlg = StoryReplicateDialog(None, p)          # source defaults to S1 (lowest)
    dlg._accept()
    src, targets = dlg.result
    assert src == 1
    assert set(targets) == {2, 3}                # both stories above, checked


@pytest.fixture(scope="module")
def qapp_vtk():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvistaqt")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_replicate_action_applies_and_undoes(qapp_vtk, monkeypatch):
    import main_window as mw
    from main_window import MainWindow
    import story_replicate_dialog as srd
    p = _one_story_building()
    w = MainWindow()
    w.load_project(p)
    monkeypatch.setattr(srd.StoryReplicateDialog, "get",
                        classmethod(lambda cls, *a, **k: (1, [2, 3])))
    n_before = len(w._project.areas)
    w.replicate_story()
    assert len(w._project.areas) == n_before + 2
    w._undo_stack.undo()
    assert len(w._project.areas) == n_before


def test_replicate_action_guarded_without_stories(qapp_vtk, monkeypatch):
    import main_window as mw
    from main_window import MainWindow
    p = Project(ndm=3, ndf=6)                     # no stories
    w = MainWindow()
    w.load_project(p)
    seen = {}
    monkeypatch.setattr(mw.QMessageBox, "information",
                        lambda *a, **k: seen.setdefault("info", a))
    w.replicate_story()
    assert "info" in seen
