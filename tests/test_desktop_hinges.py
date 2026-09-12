"""Fiber plastic-hinge property + assignment (plan §14 GUI-3).

Covers the data model (``Hinge`` + ``Member.hinge`` + length resolution +
serialization), the nonlinear-model wiring (``FiberHingeBeamColumn2D`` for
hinged members), a hinge pushover, and the Qt editor/assignment dialogs +
model-view marking (headless offscreen).
"""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import pytest

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))                 # section_gui_core (repo root)
sys.path.insert(0, str(_ROOT / "desktop"))     # flat desktop imports

import nonlinear as NL                          # noqa: E402
from project import Hinge, Material, Member, Node, Project, Section  # noqa: E402


def _gsd_column_project(*, D=0.6, L=3.0, hinge=False):
    import section_gui_core as core
    spec = core.Spec(kind="Circular", D=D, fc=35e6, fy=500e6)
    gsd = dataclasses.asdict(spec)
    A = math.pi * D * D / 4.0
    Iz = math.pi * D**4 / 64.0
    p = Project()
    p.nodes = [Node(1, 0.0, 0.0, supports=(1, 1, 1)), Node(2, L, 0.0)]
    p.sections = [Section(id=1, name="col", A=A, Iz=Iz, gsd_spec=gsd)]
    p.materials = [Material(1, "conc", E=30e9, nu=0.2)]
    if hinge:
        p.hinges = [Hinge(id=1, name="H1", lp=0.1, relative=True)]
    p.members = [Member(1, 1, 2, 1, 1, hinge=(1 if hinge else None))]
    return p


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


# ------------------------------------------------------------- data model

def test_resolve_hinge_lengths_relative_and_absolute():
    p = _gsd_column_project(L=3.0)
    # relative: 0.1 of a 3 m member -> 0.3 m each end (symmetric)
    lp_i, lp_j = p.resolve_hinge_lengths(
        p.members[0], Hinge(id=1, name="h", lp=0.1, relative=True))
    assert lp_i == pytest.approx(0.3) and lp_j == pytest.approx(0.3)
    # absolute, asymmetric
    lp_i, lp_j = p.resolve_hinge_lengths(
        p.members[0], Hinge(id=1, name="h", lp=0.4, lp_j=0.2, relative=False))
    assert lp_i == pytest.approx(0.4) and lp_j == pytest.approx(0.2)


def test_resolve_hinge_lengths_clamped_to_member():
    """Over-long hinges are scaled so lp_i + lp_j < member length."""
    p = _gsd_column_project(L=3.0)
    lp_i, lp_j = p.resolve_hinge_lengths(
        p.members[0], Hinge(id=1, name="h", lp=0.9, relative=True))  # 2.7 each
    assert lp_i + lp_j < 3.0
    assert lp_i + lp_j == pytest.approx(0.98 * 3.0)


def test_hinge_serialization_roundtrip():
    p = _gsd_column_project(hinge=True)
    q = Project.from_json(p.to_json())
    assert len(q.hinges) == 1
    h = q.hinges[0]
    assert (h.id, h.name, h.lp, h.lp_j, h.relative) == (1, "H1", 0.1, None, True)
    assert q.members[0].hinge == 1


def test_member_defaults_to_no_hinge():
    m = Member(1, 1, 2, 1, 1)
    assert m.hinge is None
    # legacy project dicts (no hinge key / no hinges list) still load
    q = Project.from_dict({"members": [{"id": 1, "n1": 1, "n2": 2,
                                        "section": 1, "material": 1}]})
    assert q.members[0].hinge is None
    assert q.hinges == []


# ------------------------------------------------------------- model wiring

def test_build_uses_fiber_hinge_element_when_assigned():
    from femsolver.elements.beam_corot import BeamColumn2DCorotational
    from femsolver.elements.beam_fiber_hinge import FiberHingeBeamColumn2D
    m = NL.build_nonlinear_model(_gsd_column_project(hinge=True))
    el = next(iter(m.elements.values()))
    assert isinstance(el, FiberHingeBeamColumn2D)
    assert el.lp_i == pytest.approx(0.3) and el.lp_j == pytest.approx(0.3)
    # without a hinge -> distributed corotational fiber element
    m0 = NL.build_nonlinear_model(_gsd_column_project(hinge=False))
    assert isinstance(next(iter(m0.elements.values())),
                      BeamColumn2DCorotational)


def test_hinge_pushover_runs_and_captures():
    """A hinged-column pushover yields and the GUI-6 capture works for the
    force-based hinge element (fiber frames from ``_e_committed``). The
    force-based hinge element needs a looser tol than the distributed default
    (plan §8)."""
    p = _gsd_column_project(hinge=True)
    res = NL.run_pushover(p, control_node=2, control_dof=1, target=0.02,
                          n_steps=20, tol=1e-5, capture_fibers=True,
                          capture_shape=True)
    assert len(res["disp"]) > 10
    assert max(res["shear"]) > 0
    assert len(res["fiber_frames"]) == len(res["disp"])
    assert len(res["fiber_frames"][-1]) > 50            # fibers captured
    assert res["damage_frames"][-1][1] > res["damage_frames"][0][1]  # hinge forms


# ------------------------------------------------------------- Qt dialogs

def test_hinge_dialog_symmetric_and_asymmetric(qapp):
    from hinge_editor import HingeDialog
    p = _gsd_column_project()
    dlg = HingeDialog(None, p)
    dlg.relative.setChecked(True)
    dlg.lp_i.setValue(0.12)
    dlg.symmetric.setChecked(True)
    h = dlg.data()
    assert h.lp == pytest.approx(0.12) and h.lp_j is None and h.relative is True
    # asymmetric + absolute
    dlg.relative.setChecked(False)
    dlg.symmetric.setChecked(False)
    dlg.lp_i.setValue(0.4)
    dlg.lp_j.setValue(0.25)
    h = dlg.data()
    assert h.relative is False
    assert h.lp == pytest.approx(0.4) and h.lp_j == pytest.approx(0.25)


def test_hinge_dialog_edit_seeds_existing(qapp):
    from hinge_editor import HingeDialog
    p = _gsd_column_project(hinge=True)
    dlg = HingeDialog(None, p, p.hinges[0])
    h = dlg.data()
    assert h.id == 1 and h.name == "H1" and h.lp == pytest.approx(0.1)
    assert h.lp_j is None and h.relative is True


def test_hinge_manager_returns_hinges(qapp):
    from hinge_editor import HingeManagerDialog
    p = _gsd_column_project(hinge=True)
    dlg = HingeManagerDialog(None, p)
    assert dlg.table.rowCount() == 1
    out = dlg.result_hinges()
    assert len(out) == 1 and out[0].id == 1
    assert out is not p.hinges                          # works on a copy


def test_hinge_assignment_dialog(qapp):
    from hinge_editor import HingeAssignmentDialog
    p = _gsd_column_project(hinge=True)
    # start unassigned so the combo default is "none"
    p.members[0].hinge = None
    dlg = HingeAssignmentDialog(None, p)
    assert dlg.table.rowCount() == 1                    # one fiber member
    assert dlg.result_assignments() == {1: None}
    dlg._all_combo.setCurrentIndex(dlg._all_combo.findData(1))
    dlg._set_all()
    assert dlg.result_assignments() == {1: 1}


def test_member_dialog_carries_hinge(qapp):
    from editing import MemberDialog
    p = _gsd_column_project(hinge=True)
    dlg = MemberDialog(None, p, p.members[0])
    assert dlg.data().hinge == 1                        # preserved on edit
    dlg.hinge.setCurrentIndex(dlg.hinge.findData(None))
    assert dlg.data().hinge is None                     # can clear


def test_pushover_dialog_exposes_tol_and_maxiter(qapp):
    from pushover_dialog import PushoverDialog
    dlg = PushoverDialog(None, _gsd_column_project(hinge=True))
    kw = dlg._kwargs()
    assert kw["tol"] == pytest.approx(1e-6)             # default
    assert kw["max_iter"] == 60


def test_main_window_marks_hinges(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_gsd_column_project(hinge=True))
    assert callable(w.view.mark_hinges)
    assert "hinges" in w.view.actors                    # marker glyph added
