"""Model navigation tree — summary outline + Show Table… (nav plan N1/N2).

Headless coverage: the tree groups every category (incl. empty ones) under the
Properties/Structures/Loads/Analysis super-groups, counts show in column 1,
elements group by type, selection refs stay locatable by ``_find_item``, and the
per-category tables build.
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

from project import (Load, LoadCase, Material, Member, MemberLoad,  # noqa: E402
                     Node, Project, Section)


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _project():
    p = Project(ndm=2, ndf=3)
    p.nodes = [Node(1, 0, 0, supports=(1, 1, 1)), Node(2, 4, 0),
               Node(3, 8, 0)]
    p.sections = [Section(id=1, name="W12x65", A=1e-2, Iz=2e-4, shape="W12x65")]
    p.materials = [Material(1, "A992", E=2e11, nu=0.3)]
    p.members = [Member(1, 1, 2, 1, 1), Member(2, 2, 3, 1, 1)]
    p.load_cases = [LoadCase(1, "Dead", "dead")]
    p.loads = [Load(node=2, values=(1e3, 0, 0), case=1)]
    p.member_loads = [MemberLoad(member=1, wy=-500.0, case=1)]
    return p


def _titles(w):
    out = {}
    for i in range(w.tree.topLevelItemCount()):
        top = w.tree.topLevelItem(i)
        out[top.text(0)] = [top.child(j).text(0)
                            for j in range(top.childCount())]
    return out


def test_super_groups_present(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_project())
    groups = _titles(w)
    for name in ("Properties", "Structures", "Loads", "Analysis"):
        assert name in groups


def test_all_categories_present_even_empty(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_project())        # no hinges / combinations → still listed
    groups = _titles(w)
    assert "Hinge properties" in groups["Properties"]
    assert "Load combinations" in groups["Loads"]
    assert "Materials" in groups["Properties"]
    assert "Sections" in groups["Properties"]
    assert "Supports" in groups["Structures"]
    assert "Results" in groups["Analysis"]


def test_counts_in_second_column(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_project())
    struct = next(w.tree.topLevelItem(i)
                  for i in range(w.tree.topLevelItemCount())
                  if w.tree.topLevelItem(i).text(0) == "Structures")
    nodes = next(struct.child(j) for j in range(struct.childCount())
                 if struct.child(j).text(0) == "Nodes")
    assert nodes.text(1) == "3"


def test_elements_grouped_by_type(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_project())
    struct = next(w.tree.topLevelItem(i)
                  for i in range(w.tree.topLevelItemCount())
                  if w.tree.topLevelItem(i).text(0) == "Structures")
    elements = next(struct.child(j) for j in range(struct.childCount())
                    if struct.child(j).text(0) == "Elements")
    assert elements.text(1) == "2"
    type_rows = [elements.child(k).text(0)
                 for k in range(elements.childCount())]
    assert "Beam" in type_rows


def test_selection_refs_still_findable(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_project())
    for ref in [("node", 1), ("member", 2), ("section", 1),
                ("load", 0), ("member_load", 0)]:
        assert w._find_item(ref) is not None, ref
    # and selecting a leaf under a collapsed category expands its ancestors
    w._select(("node", 3))
    assert w._find_item(("node", 3)).isSelected()


def test_tables_build_for_every_category(qapp):
    from model_tables import ModelTableDialog, _BUILDERS
    p = _project()
    for cat, (builder, _title) in _BUILDERS.items():
        headers, rows = builder(p)
        assert isinstance(headers, list) and headers
        for cells, _ref in rows:
            assert len(cells) == len(headers), cat


def _cat(w, group, name):
    top = next(w.tree.topLevelItem(i)
               for i in range(w.tree.topLevelItemCount())
               if w.tree.topLevelItem(i).text(0) == group)
    return next(top.child(j) for j in range(top.childCount())
                if top.child(j).text(0) == name)


def _temp_settings(w, tmp_path):
    """Redirect the window's persisted-expansion writes to a throwaway file so
    the tests don't touch the user's real QSettings."""
    from PySide6.QtCore import QSettings
    w._settings = QSettings(str(tmp_path / "nav.ini"),
                            QSettings.Format.IniFormat)


def test_expand_and_collapse_all(qapp, tmp_path):
    from main_window import MainWindow
    w = MainWindow()
    _temp_settings(w, tmp_path)
    w.load_project(_project())
    w.expand_all_tree()
    assert _cat(w, "Structures", "Nodes").isExpanded()
    w.collapse_all_tree()
    # super-groups stay open (category list visible); categories collapse
    assert not _cat(w, "Structures", "Nodes").isExpanded()
    struct = next(w.tree.topLevelItem(i)
                  for i in range(w.tree.topLevelItemCount())
                  if w.tree.topLevelItem(i).text(0) == "Structures")
    assert struct.isExpanded()


def test_expansion_survives_rebuild(qapp, tmp_path):
    from main_window import MainWindow
    w = MainWindow()
    _temp_settings(w, tmp_path)
    w.load_project(_project())
    _cat(w, "Structures", "Nodes").setExpanded(True)   # user expands Nodes
    assert "cat:nodes" in w._expanded
    w._populate_tree()                                 # an edit rebuilds the tree
    assert _cat(w, "Structures", "Nodes").isExpanded()
