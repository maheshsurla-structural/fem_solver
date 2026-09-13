"""In-app model checks & non-convergence guidance (plan §16 G-S5).

The pure checks (topology, restraint, unit-scale plausibility, section/fiber,
nonlinear-case wiring) and the post-run advice; plus the presentation dialog and
the pushover dialog's Check button / advice. Headless (offscreen) for the GUI.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))
sys.path.insert(0, str(_ROOT / "src"))

import model_checks as MC                            # noqa: E402
from project import (Material, Member, Node, NonlinearCase, Project,  # noqa: E402
                     Section)


def _clean_fiber_column():
    import section_gui_core as core
    spec = core.Spec(kind="Circular", D=0.6, fc=35e6, fy=500e6)
    p = Project()
    p.nodes = [Node(1, 0.0, 0.0, supports=(1, 1, 1)), Node(2, 3.0, 0.0)]
    p.sections = [Section(id=1, name="col", A=math.pi * 0.09,
                          Iz=math.pi * 0.6**4 / 64.0,
                          gsd_spec=dataclasses.asdict(spec))]
    p.materials = [Material(1, "conc", E=30e9, nu=0.2)]
    p.members = [Member(1, 1, 2, 1, 1)]
    return p


def _levels(checks):
    return [(c.level, c.message) for c in checks]


# ------------------------------------------------------------- clean model
def test_clean_model_has_no_errors_or_warnings():
    checks = MC.check_project(_clean_fiber_column())
    n_err, n_warn, _ = MC.summarize(checks)
    assert n_err == 0 and n_warn == 0


# ------------------------------------------------------------- topology
def test_no_supports_is_an_error():
    p = _clean_fiber_column()
    p.nodes[0] = Node(1, 0.0, 0.0)                  # remove the fixity
    msgs = " ".join(m for _l, m in _levels(MC.check_project(p)))
    assert "No supports" in msgs
    assert any(c.level == "error" for c in MC.check_project(p))


def test_zero_length_and_missing_refs():
    p = _clean_fiber_column()
    p.nodes[1] = Node(2, 0.0, 0.0)                  # coincides with node 1
    p.members.append(Member(2, 1, 99, 1, 1))        # missing node 99
    p.members.append(Member(3, 1, 2, 7, 1))         # missing section 7
    errs = [c.message for c in MC.check_project(p) if c.level == "error"]
    assert any("zero length" in m for m in errs)
    assert any("missing node 99" in m for m in errs)
    assert any("missing section 7" in m for m in errs)


def test_duplicate_node_id():
    p = _clean_fiber_column()
    p.nodes.append(Node(1, 5.0, 0.0))              # duplicate id 1
    assert any("Duplicate node id 1" in c.message
               for c in MC.check_project(p) if c.level == "error")


# ------------------------------------------------------------- units
def test_modulus_in_wrong_unit_is_flagged():
    p = _clean_fiber_column()
    p.materials[0].E = 200.0                        # 200 Pa (meant 200 GPa)
    warns = [c.message for c in MC.check_project(p) if c.level == "warning"]
    assert any("too small" in m and "MPa or GPa" in m for m in warns)


def test_fc_and_strain_scale_flags():
    p = _clean_fiber_column()
    p.materials[0].params = {"fc": 35.0, "eps_cu": 3.0}   # MPa + ‰-as-3
    warns = " ".join(c.message for c in MC.check_project(p)
                     if c.level == "warning")
    assert "f'c" in warns and "below 1 MPa" in warns
    assert "eps_cu" in warns and "dimensionless" in warns


# ------------------------------------------------------------- fiber readiness
def test_no_fiber_section_is_info():
    p = Project()
    p.nodes = [Node(1, 0, 0, supports=(1, 1, 1)), Node(2, 3, 0)]
    p.sections = [Section(id=1, name="w", A=0.01, Iz=1e-4)]
    p.materials = [Material(1, "s", E=200e9, nu=0.3)]
    p.members = [Member(1, 1, 2, 1, 1)]
    assert any(c.level == "info" and "nonlinear" in c.message.lower()
               for c in MC.check_project(p))


# ------------------------------------------------------------- NL cases
def test_nonlinear_case_wiring_errors():
    p = _clean_fiber_column()
    p.nonlinear_cases = [
        NonlinearCase(id=1, name="bad", control_node=99, control_dof=9,
                      target=0.0, continue_from=42)]
    errs = " ".join(c.message for c in MC.check_project(p) if c.level == "error")
    assert "control node 99" in errs
    assert "control DOF 9" in errs
    assert "missing case 42" in errs


# ------------------------------------------------------------- advice
def test_convergence_advice_error_and_short():
    tips = MC.convergence_advice(error="singular tangent (mechanism)")
    joined = " ".join(tips).lower()
    assert "singular" in joined and "tolerance" in joined
    short = MC.convergence_advice(got_steps=6, requested_steps=40)
    assert any("stopped at step 6 of 40" in t for t in short)
    assert MC.convergence_advice() == []           # nothing to say on success


# ------------------------------------------------------------- GUI
@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_checks_dialog_lists_findings(qapp):
    from model_checks_dialog import ModelChecksDialog
    p = _clean_fiber_column()
    p.nodes[0] = Node(1, 0.0, 0.0)                  # -> a "No supports" error
    checks = MC.check_project(p)
    dlg = ModelChecksDialog(None, checks)
    assert dlg.list.count() == len(checks)
    assert "error" in dlg.summary.text().lower()


def test_checks_dialog_colours_by_severity(qapp):
    """D3: error rows use the themed BAD ink (not a raw hex / default)."""
    from PySide6.QtGui import QColor

    import style
    from model_checks_dialog import ModelChecksDialog
    p = _clean_fiber_column()
    p.nodes[0] = Node(1, 0.0, 0.0)                  # forces an error finding
    checks = MC.check_project(p)
    dlg = ModelChecksDialog(None, checks)
    err = next(i for i in range(dlg.list.count())
               if dlg.list.item(i).data(0x0100) == "error")   # UserRole
    assert dlg.list.item(err).foreground().color() == QColor(style.BAD)


def test_pushover_dialog_check_and_advice(qapp):
    pytest.importorskip("matplotlib")
    from pushover_dialog import PushoverDialog
    dlg = PushoverDialog(None, _clean_fiber_column())
    dlg._advise(error="singular tangent")
    dlg._advise(got=3)                              # requested defaults > 3
    text = dlg.log.toPlainText().lower()
    assert "singular" in text
    assert "stopped at step 3" in text
