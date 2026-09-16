"""Initial conditions / state chaining (E2) — data model + seeding helper.

E2a: the ``initial_condition`` field on ``NonlinearCase`` / ``AnalysisCase``
(default, coercion, serialization round-trip, old-project migration) and the
Analysis-cases manager's "· from ‹source›" detail suffix.

E2b: ``nonlinear.seed_to_committed_state`` — runs a nonlinear case to its
committed end state without recording a pushover curve, returning the seeded
model + the held constant-force vector, seeding to exactly the state the case's
own pushover reaches.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

import nonlinear as NL                                      # noqa: E402
from project import (AnalysisCase, Material, Member, Node,  # noqa: E402
                     NonlinearCase, Project, Section, _coerce_ic)


def _gsd_column_project(*, D=0.6, L=3.0):
    import section_gui_core as core
    spec = core.Spec(kind="Circular", D=D, fc=35e6, fy=500e6)
    gsd = dataclasses.asdict(spec)
    A = math.pi * D * D / 4.0
    Iz = math.pi * D**4 / 64.0
    p = Project()
    p.nodes = [Node(1, 0.0, 0.0, supports=(1, 1, 1)), Node(2, L, 0.0)]
    p.sections = [Section(id=1, name="col", A=A, Iz=Iz, gsd_spec=gsd)]
    p.materials = [Material(1, "conc", E=30e9, nu=0.2)]
    p.members = [Member(1, 1, 2, 1, 1)]
    return p


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


# ------------------------------------------------------------- E2a data model

def test_initial_condition_defaults_to_zero():
    assert NonlinearCase(id=1, name="n").initial_condition == ("zero",)
    assert AnalysisCase(id=1, name="a", type="modal").initial_condition \
        == ("zero",)


def test_coerce_ic_normalizes():
    assert _coerce_ic(None) == ("zero",)
    assert _coerce_ic([]) == ("zero",)
    assert _coerce_ic(("zero",)) == ("zero",)
    assert _coerce_ic(["state", 3]) == ("state", 3)     # JSON list → tuple
    assert _coerce_ic(("state", "5")) == ("state", 5)   # id coerced to int
    assert _coerce_ic(["state", None]) == ("zero",)     # dangling → zero
    assert _coerce_ic(["bogus", 1]) == ("zero",)


def test_state_ic_survives_serialization_roundtrip():
    p = _gsd_column_project()
    p.nonlinear_cases = [NonlinearCase(id=1, name="preload", control_node=2)]
    p.analysis_cases = [
        AnalysisCase(id=1, name="modal-after", type="modal",
                     params={"num_modes": 4, "lumped": False},
                     initial_condition=("state", 1))]
    p.nonlinear_cases.append(
        NonlinearCase(id=2, name="th-after", control_node=2,
                      initial_condition=("state", 1)))
    q = Project.from_json(p.to_json())
    assert q.analysis_cases[0].initial_condition == ("state", 1)
    assert q.nonlinear_cases[1].initial_condition == ("state", 1)
    # canonical tuple, not a JSON list
    assert isinstance(q.analysis_cases[0].initial_condition, tuple)


def test_old_project_without_field_defaults_to_zero():
    # a project dict predating E2 (no initial_condition key anywhere)
    p = _gsd_column_project()
    p.nonlinear_cases = [NonlinearCase(id=1, name="n", control_node=2)]
    p.analysis_cases = [AnalysisCase(id=1, name="a", type="modal")]
    d = p.to_dict()
    for c in d["nonlinear_cases"]:
        c.pop("initial_condition", None)
    for c in d["analysis_cases"]:
        c.pop("initial_condition", None)
    q = Project.from_dict(d)
    assert q.nonlinear_cases[0].initial_condition == ("zero",)
    assert q.analysis_cases[0].initial_condition == ("zero",)


def test_manager_detail_suffix_names_source(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog
    p = _gsd_column_project()
    p.nonlinear_cases = [NonlinearCase(id=1, name="PRELOAD", control_node=2)]
    p.analysis_cases = [
        AnalysisCase(id=1, name="modal-after", type="modal",
                     params={"num_modes": 6, "lumped": False},
                     initial_condition=("state", 1)),
        AnalysisCase(id=2, name="dangling", type="modal",
                     params={"num_modes": 6, "lumped": False},
                     initial_condition=("state", 99))]
    dlg = AnalysisCasesDialog(None, p)
    details = {m["name"]: m["detail"] for m in dlg._row_meta
               if m["kind"] == "analysis"}
    assert details["modal-after"].endswith("· from PRELOAD")
    assert details["dangling"].endswith("· from (missing case)")
    # a zero-IC case gets no suffix
    plain = next(m for m in dlg._row_meta if m["kind"] == "nonlinear")
    assert "from" not in plain["detail"]


# ------------------------------------------------------- E2b seeding helper

def test_seed_matches_run_case_end_state():
    """The seeded model is left at exactly the committed control displacement
    the case's own pushover reaches (same physics, no curve recorded)."""
    p = _gsd_column_project()
    c = NonlinearCase(id=1, name="mono", control_node=2, control_dof=1,
                      target=0.03, n_steps=15)
    p.nonlinear_cases = [c]
    rc = NL.run_case(p, c)
    model, f_const = NL.seed_to_committed_state(p, c)
    seeded = float(model.nodes[2].disp[1])
    assert seeded == pytest.approx(rc["disp"][-1], rel=1e-6, abs=1e-9)
    assert f_const is not None
    assert np.any(np.asarray(f_const, dtype=float) != 0.0)   # a held force


def test_seed_with_axial_preload_commits_state_and_holds_force():
    p = _gsd_column_project()
    c = NonlinearCase(id=1, name="preload+push", control_node=2, control_dof=1,
                      target=0.02, n_steps=12, axial=2.0e5, axial_node=2,
                      axial_dof=0)
    p.nonlinear_cases = [c]
    model, f_const = NL.seed_to_committed_state(p, c)
    assert abs(float(model.nodes[2].disp[1])) > 0.0     # lateral state present
    assert f_const is not None
    assert np.any(np.asarray(f_const, dtype=float) != 0.0)


def test_seed_is_nondestructive_to_load_pattern():
    """The held load lives only in the returned vector; the seeded model's own
    load pattern is cleared (so a downstream analysis starts clean)."""
    p = _gsd_column_project()
    c = NonlinearCase(id=1, name="mono", control_node=2, control_dof=1,
                      target=0.02, n_steps=10)
    p.nonlinear_cases = [c]
    model, _ = NL.seed_to_committed_state(p, c)
    assert all(not np.any(n._load) for n in model.nodes.values())
