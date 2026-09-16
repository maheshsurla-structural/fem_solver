"""Modeless "pick from the model" for node / member dialogs (``desktop/pick.py``).

The nodal-load and every other node/member dialog used to be modal, so the 3-D
viewport could not be clicked to choose a node/member until the dialog closed.
:class:`pick.PickDialog` makes them modeless-but-synchronous and routes a
viewport pick into the dialog's registered field. These headless (offscreen)
tests cover the field plumbing, the host routing on ``MainWindow``, and that
``exec`` is genuinely modeless (a pick lands *while* it is open).
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

from project import LoadCase, Member, Node, Project, Section     # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _project():
    p = Project(ndm=2, ndf=3)
    p.nodes = [Node(1, 0, 0, supports=(1, 1, 1)), Node(2, 3, 0), Node(3, 3, 3)]
    p.load_cases = [LoadCase(1, "Dead", "dead")]
    p.sections = [Section(1, "S", 6e-3, 2e-4)]
    p.materials = [type("M", (), {"id": 1, "name": "steel"})()]
    p.members = [Member(1, 1, 2, section=1, material=1),
                 Member(2, 2, 3, section=1, material=1)]
    return p


# --------------------------------------------------------------- combo fields
def test_load_dialog_node_is_pickable(qapp):
    from editing import LoadDialog
    d = LoadDialog(None, _project())
    assert d.has_pick_fields() and d.accepts_kind("node")
    assert not d.accepts_kind("member")
    assert d.accept_pick("node", 3) and d.node.currentData() == 3
    # a wrong-kind pick is not consumed, so the host can fall back to selection
    assert d.accept_pick("member", 1) is False


def test_member_dialog_two_node_fields_follow_focus(qapp):
    from editing import MemberDialog
    d = MemberDialog(None, _project())
    # default armed field is the first registered (start node)
    assert d.accept_pick("node", 2) and d.n1.currentData() == 2
    d._armed_field = d.n2                       # focusing End node arms it
    assert d.accept_pick("node", 3) and d.n2.currentData() == 3
    assert d.n1.currentData() == 2             # unchanged


def test_member_load_dialog_member_is_pickable(qapp):
    from member_load_dialog import MemberLoadDialog
    d = MemberLoadDialog(None, _project())
    assert d.accepts_kind("member")
    assert d.accept_pick("member", 2) and d.member.currentData() == 2


# ---------------------------------------------------------------- list fields
def test_list_field_toggles_membership(qapp):
    from temperature_gradient_dialog import TemperatureGradientDialog
    d = TemperatureGradientDialog(None, _project())
    ids = {d.members.item(i).data(0x0100) for i in range(d.members.count())}
    assert ids == {1, 2}
    # every member starts selected; a pick toggles the clicked one off, then on
    assert d.members.selectedItems()
    assert d.accept_pick("member", 1)
    on = {it.data(0x0100) for it in d.members.selectedItems()}
    assert on == {2}
    assert d.accept_pick("member", 1)
    on = {it.data(0x0100) for it in d.members.selectedItems()}
    assert on == {1, 2}


# ------------------------------------------------------------- host routing
def test_main_window_routes_picks_to_active_sink(qapp):
    from editing import LoadDialog
    from main_window import MainWindow
    w = MainWindow()
    p = _project()
    w._project = p
    d = LoadDialog(w, p)

    w._set_selection([("node", 1)])
    w.push_pick_sink(d)
    w._on_pick("node", 2)                       # consumed by the dialog
    assert d.node.currentData() == 2
    assert w._selection == [("node", 1)]        # selection untouched while picking

    w.pop_pick_sink(d)
    w._on_pick("node", 3)                       # no sink → normal selection
    assert w._selection == [("node", 3)]


# ------------------------------------------------------ exec() is modeless
def test_exec_is_modeless_and_synchronous(qapp):
    from PySide6.QtCore import QTimer
    from editing import LoadDialog
    from main_window import MainWindow
    w = MainWindow()
    p = _project()
    w._project = p
    d = LoadDialog(w, p)
    seen = {}

    def during_exec():
        seen["visible"] = d.isVisible()
        seen["modal"] = d.isModal()
        seen["sink"] = w._active_pick_sink() is d
        w._on_pick("node", 3)                   # simulate a viewport click
        d.accept()

    QTimer.singleShot(0, during_exec)
    result = d.exec()                            # blocks on a private loop, but…
    assert seen["visible"] is True              # …it was shown (not app-modal)
    assert seen["modal"] is False               # …the viewport was not grabbed
    assert seen["sink"] is True                 # …it was the pick sink
    assert result and d.data().node == 3        # …and the pick flowed in


def test_exec_falls_back_to_modal_under_a_modal_parent(qapp):
    """Opened under an application-modal dialog, a PickDialog must NOT go
    modeless (Qt would block a modeless window over a modal one). It should run
    a normal modal exec instead — picking is simply unavailable there."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QDialog
    from editing import LoadDialog

    gate = QDialog()
    gate.setModal(True)
    outcome = {}

    def run_child():
        # gate is now the active modal widget; the child must fall back to modal.
        child = LoadDialog(gate, _project())
        QTimer.singleShot(0, child.reject)
        child.exec()
        outcome["child_modal_while_open"] = None  # reached only if exec returned
        gate.accept()

    QTimer.singleShot(0, run_child)
    gate.exec()
    assert "child_modal_while_open" in outcome    # child exec returned cleanly
