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


def _node_leaf(w, nid):
    from main_window import CAT_ROLE  # noqa: F401
    return w._find_item(("node", nid))


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _isolate_settings(qapp, tmp_path, monkeypatch):
    """Point the window's QSettings at a throwaway ini so persisted nav flags
    (summary / expanded) can't leak in from the real store or between tests."""
    import main_window
    from PySide6.QtCore import QSettings
    ini = str(tmp_path / "settings.ini")
    monkeypatch.setattr(
        main_window, "QSettings",
        lambda *a, **k: QSettings(ini, QSettings.Format.IniFormat))
    yield


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


def _top(w, name):
    return next(w.tree.topLevelItem(i)
               for i in range(w.tree.topLevelItemCount())
               if w.tree.topLevelItem(i).text(0) == name)


def test_filter_hides_nonmatching(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_project())
    w._apply_filter("beam")
    elements = _cat(w, "Structures", "Elements")
    beam = next(elements.child(k) for k in range(elements.childCount())
                if elements.child(k).text(0) == "Beam")
    assert not elements.isHidden() and not beam.isHidden()
    assert elements.isExpanded()                       # match reveals children
    assert _top(w, "Properties").isHidden()            # no match → hidden


def test_filter_matches_container_shows_descendants(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_project())
    w._apply_filter("nodes")                           # a category name
    nodes = _cat(w, "Structures", "Nodes")
    assert not nodes.isHidden() and nodes.childCount() > 0
    assert not nodes.child(0).isHidden()               # descendants forced on


def test_filter_clear_restores(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_project())
    w._apply_filter("beam")
    assert _top(w, "Properties").isHidden()
    w._apply_filter("")                                # clear
    assert not _top(w, "Properties").isHidden()
    assert not _cat(w, "Properties", "Materials").isHidden()


def test_value_edit_refreshes_label_in_place(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_project())
    node_item = _node_leaf(w, 2)
    id_before = id(node_item)
    w._project.nodes[1].x = 9.0                        # move node 2, no structure change
    w._refresh_tree()                                  # the rebuild entry point
    same_item = _node_leaf(w, 2)
    assert id(same_item) == id_before                  # item reused, not rebuilt
    assert "9" in same_item.text(0)                    # label updated in place


def test_structural_edit_updates_counts(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_project())
    assert _cat(w, "Structures", "Nodes").text(1) == "3"
    w._project.nodes.append(Node(4, 12, 0))            # add a node → structure change
    w._refresh_tree()
    assert _cat(w, "Structures", "Nodes").text(1) == "4"
    assert w._find_item(("node", 4)) is not None


def test_refresh_preserves_selection_and_scroll(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_project())
    w._select(("member", 1))
    assert w._selected_refs() == [("member", 1)]
    w._project.sections[0].name = "W14x90"             # value-only edit
    w._refresh_tree()
    assert w._selected_refs() == [("member", 1)]       # selection kept


# --------------------------------------------------------------- N6: summary mode
def test_summary_mode_drops_leaves_keeps_counts(qapp, tmp_path):
    from main_window import MainWindow
    w = MainWindow()
    _temp_settings(w, tmp_path)
    w.load_project(_project())
    w.toggle_summary_mode(True)
    nodes = _cat(w, "Structures", "Nodes")
    assert nodes.text(1) == "3"                         # count still shown
    assert nodes.childCount() == 0                      # but no individual leaves
    elements = _cat(w, "Structures", "Elements")
    assert elements.text(1) == "2"
    beam = next(elements.child(k) for k in range(elements.childCount())
                if elements.child(k).text(0) == "Beam")
    assert beam.text(1) == "2" and beam.childCount() == 0  # type row kept, no leaves


def test_summary_mode_selection_via_viewport(qapp, tmp_path):
    from main_window import MainWindow
    w = MainWindow()
    _temp_settings(w, tmp_path)
    w.load_project(_project())
    w.toggle_summary_mode(True)
    assert w._find_item(("member", 1)) is None          # no leaf to click
    w._on_pick("member", 1)                             # viewport drives selection
    assert w._selected_refs() == [("member", 1)]
    w._on_pick("node", 2)                               # non-additive replaces
    assert w._selected_refs() == [("node", 2)]


def test_leaf_mode_pick_and_additive(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_project())
    w._on_pick("member", 1)
    assert w._selected_refs() == [("member", 1)]
    assert w._find_item(("member", 1)).isSelected()     # tree mirrors it
    w._set_selection([("member", 1), ("member", 2)])    # multi selection
    assert set(w._selected_refs()) == {("member", 1), ("member", 2)}
    assert w._find_item(("member", 2)).isSelected()


def test_delete_multiple_and_deselect(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_project())
    w._set_selection([("load", 0), ("member_load", 0)])
    w.delete_selected()
    assert w._project.loads == [] and w._project.member_loads == []
    assert w._selected_refs() == []


def test_summary_mode_persists(qapp, tmp_path):
    from main_window import MainWindow
    w = MainWindow()
    _temp_settings(w, tmp_path)
    w.toggle_summary_mode(True)
    assert w._settings.value("nav/summary", False, type=bool) is True


# --------------------------------------------------------------- N7: in-table edit
def test_table_commit_edits_and_undoes(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_project())
    assert w._table_commit(("node", 2), "x", 7.0) is True
    assert w._project.nodes[1].x == 7.0
    w._undo_stack.undo()
    assert w._project.nodes[1].x == 4.0
    assert w._table_commit(("material", 1), "E", 2.1e11) is True
    assert w._project.materials[0].E == 2.1e11
    assert w._table_commit(("load", 0), "v0", 5000.0) is True
    assert w._project.loads[0].values[0] == 5000.0


def test_table_commit_rejects_unknown_field(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_project())
    assert w._table_commit(("member", 1), "n1", 5) is False   # not editable
    assert w._table_commit(("node", 2), "bogus", 1) is False


def test_table_commit_member_reassign_validates(qapp, monkeypatch):
    from main_window import MainWindow
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    w = MainWindow()
    p = _project()
    p.sections.append(Section(id=2, name="W8x31", A=5e-3, Iz=1e-4))
    w.load_project(p)
    assert w._table_commit(("member", 1), "section", 999) is False   # no such id
    assert w._table_commit(("member", 1), "section", 2) is True
    assert next(m for m in w._project.members if m.id == 1).section == 2


def test_gsd_section_geometry_not_editable(qapp):
    from model_tables import _sections, Editable
    p = _project()
    p.sections[0].gsd_spec = {"dummy": 1}
    _headers, rows = _sections(p)
    cells, ref = rows[0]
    assert ref == ("section", 1)
    assert isinstance(cells[1], Editable)          # name stays editable
    assert not isinstance(cells[3], Editable)      # A is GSD-driven → locked


def test_dialog_cell_edit_calls_commit_and_reverts(qapp):
    from model_tables import ModelTableDialog, _nodes
    p = _project()
    calls = []
    headers, rows = _nodes(p)
    dlg = ModelTableDialog(None, "Nodes", headers, rows,
                           on_commit=lambda ref, f, v: (calls.append((ref, f, v))
                                                        or True))
    dlg.table.item(0, 1).setText("12")             # edit node 1's X
    assert calls == [(("node", 1), "x", 12.0)]
    # a bad parse reverts to the original text and does not commit
    dlg.table.item(0, 2).setText("abc")            # Y column
    assert dlg.table.item(0, 2).text() == "0"
    assert len(calls) == 1
