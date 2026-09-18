"""Active / inactive working set (MIDAS-style activation) — ``MainWindow``.

Deactivating hides nodes/members/areas from the viewport's *display project*
while the full analysis model (``self._model``) is untouched, so results never
change. Covers inactivate / isolate / show-all / invert and the stale-ref prune.
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


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvistaqt")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _frame_project():
    from project import Material, Member, Node, Project, Section
    p = Project(ndm=2, ndf=3)
    p.materials.append(Material(id=1, name="A992", E=200e9, nu=0.3))
    p.sections.append(Section(id=1, name="W12", A=1e-2, Iz=1e-4))
    # a 2-bay chain: n1 - m1 - n2 - m2 - n3
    p.nodes.extend([Node(id=1, x=0.0, y=0.0), Node(id=2, x=3.0, y=0.0),
                    Node(id=3, x=6.0, y=0.0)])
    p.members.extend([Member(id=1, n1=1, n2=2, section=1, material=1),
                      Member(id=2, n1=2, n2=3, section=1, material=1)])
    return p


def _window(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_frame_project())
    return w


def _disp_member_ids(w):
    return {m.id for m in w._display_project().members}


def _disp_node_ids(w):
    return {n.id for n in w._display_project().nodes}


def test_no_inactive_returns_project_itself(qapp):
    w = _window(qapp)
    assert w._display_project() is w._project        # fast, allocation-free path


def test_inactivate_selected_hides_member_only_in_view(qapp):
    w = _window(qapp)
    w._set_selection([("member", 1)])
    w.inactivate_selected()
    assert ("member", 1) in w._inactive
    assert _disp_member_ids(w) == {2}                 # hidden from the view
    assert 1 in w._model.elements                     # full analysis model intact
    # its now-orphaned end node (1) drops out; the shared node (2) stays
    assert _disp_node_ids(w) == {2, 3}
    assert w._selection == []                          # selection cleared


def test_activate_all_restores_full_model(qapp):
    w = _window(qapp)
    w._set_selection([("member", 1)])
    w.inactivate_selected()
    w.activate_all()
    assert w._inactive == set()
    assert w._display_project() is w._project


def test_isolate_keeps_selected_and_its_nodes(qapp):
    w = _window(qapp)
    w._set_selection([("member", 2)])
    w.activate_selected_only()
    assert _disp_member_ids(w) == {2}
    assert _disp_node_ids(w) == {2, 3}                # endpoints of member 2
    assert ("member", 1) in w._inactive


def test_invert_active_swaps_the_set(qapp):
    w = _window(qapp)
    w._set_selection([("member", 1)])
    w.inactivate_selected()                            # inactive = {member 1}
    before = set(w._inactive)
    w.invert_active()
    after = set(w._inactive)
    assert before != after
    assert ("member", 1) not in after                 # member 1 now active again
    assert ("member", 2) in after                     # everything else inactive


def test_inactive_node_hides_its_members(qapp):
    w = _window(qapp)
    w._set_selection([("node", 2)])
    w.inactivate_selected()
    # node 2 is shared by both members, so hiding it hides both
    assert _disp_member_ids(w) == set()
    assert 2 not in _disp_node_ids(w)


def test_prune_drops_stale_inactive_refs(qapp):
    w = _window(qapp)
    w._inactive = {("member", 999), ("member", 1)}    # 999 no longer exists
    w._prune_inactive()
    assert w._inactive == {("member", 1)}
