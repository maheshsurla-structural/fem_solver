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
                     NonlinearCase, Project, Section,
                     TimeHistoryFunction, _coerce_ic)


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


# ------------------------------------------------------- E2c Time History

def test_time_history_from_state_gravity_hold_stays_at_rest():
    """The E2 equilibrium check: seed a time history from a nonlinear case's
    committed state, hold the source loads, and excite with ZERO ground motion —
    the preloaded structure must stay put (velocity ~ 0 at its committed
    displacement). Without the hold the preload is unbalanced and it moves."""
    p = _gsd_column_project()
    preload = NonlinearCase(id=1, name="push", control_node=2, control_dof=1,
                            target=0.005, n_steps=5)
    p.nonlinear_cases = [preload]
    accel = np.zeros(20)                       # no ground motion
    seeded, _ = NL.seed_to_committed_state(p, preload, density=2400.0)
    delta = float(seeded.nodes[2].disp[1])     # committed lateral displacement

    held = NL.run_time_history(p, accel, 0.02, control_node=2, control_dof=1,
                               direction="y", density=2400.0,
                               initial_case=1, hold_source_loads=True)
    assert held["disp"][0] == pytest.approx(delta, abs=5e-4)   # stays deformed
    assert max(abs(v) for v in held["velocity"]) < 5e-3        # at rest

    free = NL.run_time_history(p, accel, 0.02, control_node=2, control_dof=1,
                               direction="y", density=2400.0,
                               initial_case=1, hold_source_loads=False)
    # unbalanced preload -> the structure springs back and moves
    assert max(abs(v) for v in free["velocity"]) > \
        20 * max(abs(v) for v in held["velocity"]) + 1e-3


def test_time_history_missing_source_raises():
    p = _gsd_column_project()
    with pytest.raises(ValueError, match="deleted"):
        NL.run_time_history(p, np.zeros(10), 0.02, control_node=2,
                            control_dof=1, initial_case=999,
                            hold_source_loads=True)


def test_th_build_config_threads_initial_condition():
    import case_types
    p = _gsd_column_project()
    p.th_functions = [TimeHistoryFunction(id=1, name="rec", dt=0.01,
                                          values=[0.0, 1.0, 0.0, -1.0, 0.0])]
    ct = case_types.get("timehistory")
    params = {"function_id": 1, "control_node": 2, "direction": "y",
              "scale": 1.0, "zeta": 0.05, "density": 2400.0,
              "hold_source_loads": True}
    cfg = ct.build_config(p, params, initial_condition=("state", 7))
    assert cfg["initial_condition"] == ("state", 7)
    assert cfg["hold_source_loads"] is True
    # default (zero) still works and defaults the hold off
    cfg0 = ct.build_config(p, {"function_id": 1})
    assert cfg0["initial_condition"] == ("zero",)
    assert cfg0["hold_source_loads"] is False


def test_initial_condition_card_roundtrip(qapp):
    from analysis_ui import InitialConditionCard
    card = InitialConditionCard([(1, "PRELOAD"), (2, "OTHER")])
    assert card.value() == ("zero",)                   # default unstressed
    card.set_value(("state", 2), hold=False)
    assert card.value() == ("state", 2)
    assert card.hold() is False
    card.set_value(("zero",))
    assert card.value() == ("zero",)
    # a source that no longer exists falls back to zero, not a crash
    card.set_value(("state", 404))
    assert card.value() == ("zero",)


def test_initial_condition_card_empty_disables_state(qapp):
    from analysis_ui import InitialConditionCard
    card = InitialConditionCard([])                    # no nonlinear cases
    card.set_value(("state", 1))
    assert card.value() == ("zero",)                   # cannot select state


def test_th_case_dialog_carries_ic_and_hold(qapp):
    from timehistory_dialog import TimeHistoryCaseDialog
    p = _gsd_column_project()
    p.nonlinear_cases = [NonlinearCase(id=5, name="PRELOAD", control_node=2)]
    p.th_functions = [TimeHistoryFunction(id=1, name="rec", dt=0.01,
                                          values=[0.0, 1.0, 0.0])]
    dlg = TimeHistoryCaseDialog(
        None, p, initial={"function_id": 1, "control_node": 2,
                          "hold_source_loads": True},
        initial_ic=("state", 5))
    assert dlg.initial_condition() == ("state", 5)
    assert dlg.params()["hold_source_loads"] is True


def test_th_edit_sets_case_initial_condition(qapp, monkeypatch):
    """The adapter round-trips the IC onto the saved AnalysisCase (E1: Time
    History is edited through the unified CaseEditorDialog)."""
    import case_editor
    import case_types
    p = _gsd_column_project()
    p.nonlinear_cases = [NonlinearCase(id=3, name="PRELOAD", control_node=2)]
    p.th_functions = [TimeHistoryFunction(id=1, name="rec", dt=0.01,
                                          values=[0.0, 1.0, 0.0])]
    # drive the editor headlessly: seed a state IC + hold, then accept
    monkeypatch.setattr(case_editor.CaseEditorDialog, "exec", lambda self: True)
    orig_init = case_editor.CaseEditorDialog.__init__

    def _init(self, *a, **k):
        orig_init(self, *a, **k)
        self.initial.set_value(("state", 3), hold=True)
    monkeypatch.setattr(case_editor.CaseEditorDialog, "__init__", _init)

    case = case_types.get("timehistory").edit(None, p)
    assert case is not None and case.type == "timehistory"
    assert case.initial_condition == ("state", 3)
    assert case.params["hold_source_loads"] is True
    assert case.params["hold_source_loads"] is True


# ------------------------------------------------- E2d Modal / P-Δ from state

def test_modal_build_config_threads_initial_condition():
    import case_types
    p = _gsd_column_project()
    ct = case_types.get("modal")
    cfg = ct.build_config(p, {"num_modes": 4, "lumped": True},
                          initial_condition=("state", 2))
    assert cfg == (4, True, ("state", 2))
    assert ct.build_config(p, {"num_modes": 6}) == (6, False, ("zero",))


def test_modal_dialog_carries_ic(qapp):
    from modal_dialog import ModalDialog
    dlg = ModalDialog(None, max_modes=10, default_modes=4,
                      sources=[(1, "PRELOAD")], initial_ic=("state", 1))
    assert dlg.initial_condition() == ("state", 1)
    dlg2 = ModalDialog(None, max_modes=10, default_modes=4)   # no sources
    assert dlg2.initial_condition() == ("zero",)


def _slender_fiber_column(rho=2400.0, D=0.3, L=6.0):
    import section_gui_core as core
    spec = core.Spec(kind="Circular", D=D, fc=35e6, fy=500e6)
    gsd = dataclasses.asdict(spec)
    A = math.pi * D * D / 4.0
    Iz = math.pi * D**4 / 64.0
    p = Project()
    p.nodes = [Node(1, 0.0, 0.0, supports=(1, 1, 1)), Node(2, L, 0.0)]
    p.sections = [Section(id=1, name="col", A=A, Iz=Iz, gsd_spec=gsd)]
    p.materials = [Material(1, "conc", E=30e9, nu=0.2, rho=rho)]
    p.members = [Member(1, 1, 2, 1, 1)]
    return p


def test_modal_from_state_softens_frequency_on_fiber_path():
    """P-Δ modal (E2d): tangent modal on the seeded fiber model gives a lower
    fundamental frequency at an axial-preloaded state than at zero preload — the
    geometric-stiffness softening a nonlinear-case initial condition captures."""
    from femsolver.analysis.eigen import EigenAnalysis
    p = _slender_fiber_column()

    def f_tangent(axial):
        c = NonlinearCase(id=1, name="ax", control_node=2, control_dof=1,
                          target=1e-6, n_steps=2, axial=axial, axial_node=2,
                          axial_dof=0)
        p.nonlinear_cases = [c]
        m, _ = NL.seed_to_committed_state(p, c, density=2400.0)
        m.number_dofs()
        return EigenAnalysis(m, num_modes=1,
                             stiffness="tangent").run()["frequencies_hz"][0]

    f0 = f_tangent(0.0)
    fP = f_tangent(4.0e5)              # axial compression
    assert fP < 0.95 * f0             # preload softens the fundamental mode


# ------------------------------------------ E2d Response Spectrum from state

def test_rs_build_config_threads_initial_condition():
    import case_types
    p = _gsd_column_project()
    ct = case_types.get("responsespectrum")
    params = {"source": "asce7", "num_modes": 4, "direction": "x",
              "combination": "srss", "asce7": {"SDS": 1.0, "SD1": 0.6,
                                               "TL": 8.0}}
    cfg = ct.build_config(p, params, initial_condition=("state", 9))
    assert len(cfg) == 5 and cfg[4] == ("state", 9)
    assert ct.build_config(p, params)[4] == ("zero",)


def test_rs_dialog_carries_ic(qapp):
    from response_spectrum_dialog import ResponseSpectrumDialog
    dlg = ResponseSpectrumDialog(None, ndm=2, max_modes=10,
                                 sources=[(1, "PRELOAD")],
                                 initial_ic=("state", 1))
    assert dlg.initial_condition() == ("state", 1)
    dlg2 = ResponseSpectrumDialog(None, ndm=2, max_modes=10)   # no sources
    assert dlg2.initial_condition() == ("zero",)


def test_rs_from_state_lengthens_period_on_fiber_path(qapp):
    """RS from a preloaded state uses the tangent modal basis (and preserves the
    committed state — no reset), so the fundamental period lengthens (E2d)."""
    from femsolver import ResponseSpectrumAnalysis
    from response_spectrum_dialog import spectrum_from_params
    spec = spectrum_from_params({"source": "asce7", "damping": 0.05,
                                 "asce7": {"SDS": 1.0, "SD1": 0.6, "TL": 8.0}})

    def period(axial):
        p = _slender_fiber_column()
        c = NonlinearCase(id=1, name="ax", control_node=2, control_dof=1,
                          target=1e-6, n_steps=2, axial=axial, axial_node=2,
                          axial_dof=0)
        p.nonlinear_cases = [c]
        m, _ = NL.seed_to_committed_state(p, c, density=2400.0)
        m.number_dofs()
        info = ResponseSpectrumAnalysis(m, spec, num_modes=1, direction="y",
                                        combination="srss",
                                        stiffness="tangent").run()
        return info["modal_results"][0]["period"]

    assert period(4.0e5) > 1.02 * period(0.0)


# --------------------------------------------------- E2e Buckling from state

def test_buckling_build_config_threads_initial_condition():
    import case_types
    p = _gsd_column_project()
    ct = case_types.get("buckling")
    cfg = ct.build_config(p, {"selection": ["all", None], "num_modes": 3,
                              "subdivisions": 6}, initial_condition=("state", 2))
    assert cfg == (("all", None), 3, 6, ("state", 2))
    assert ct.build_config(p, {})[3] == ("zero",)


def test_buckling_dialog_carries_ic(qapp):
    from buckling_dialog import BucklingDialog
    p = _gsd_column_project()
    p.nonlinear_cases = [NonlinearCase(id=1, name="PRELOAD", control_node=2)]
    dlg = BucklingDialog(None, p, sources=[(1, "PRELOAD")],
                         initial_ic=("state", 1))
    assert dlg.initial_condition() == ("state", 1)
    dlg2 = BucklingDialog(None, p)                    # no sources -> zero
    assert dlg2.initial_condition() == ("zero",)


# ---------------------------------------------- E2-ui referential-integrity

def test_delete_guard_blocks_nonlinear_case_referenced_by_ic(qapp, monkeypatch):
    import analysis_cases_dialog as acd
    from analysis_cases_dialog import AnalysisCasesDialog
    warned = []
    monkeypatch.setattr(acd.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warned.append(a)))
    p = _gsd_column_project()
    p.nonlinear_cases = [NonlinearCase(id=1, name="PRELOAD", control_node=2)]
    p.analysis_cases = [AnalysisCase(id=1, name="modal-after", type="modal",
                                     params={"num_modes": 4, "lumped": False},
                                     initial_condition=("state", 1))]
    dlg = AnalysisCasesDialog(None, p)
    r = next(i for i, m in enumerate(dlg._row_meta) if m["kind"] == "nonlinear")
    dlg.table.setCurrentCell(r, 0)
    dlg._delete()
    assert len(dlg._cases) == 1                # not deleted — still referenced
    assert warned and "modal-after" in warned[0][2]


def test_delete_guard_allows_unreferenced_nonlinear_case(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog
    p = _gsd_column_project()
    p.nonlinear_cases = [NonlinearCase(id=1, name="free", control_node=2)]
    dlg = AnalysisCasesDialog(None, p)
    r = next(i for i, m in enumerate(dlg._row_meta) if m["kind"] == "nonlinear")
    dlg.table.setCurrentCell(r, 0)
    dlg._delete()
    assert dlg._cases == []                    # unreferenced -> deleted


def test_model_checks_flags_dangling_initial_condition():
    import model_checks
    p = _gsd_column_project()
    p.analysis_cases = [AnalysisCase(id=1, name="rs-after",
                                     type="responsespectrum", params={},
                                     initial_condition=("state", 99))]
    msgs = [c.message for c in model_checks.check_project(p)]
    assert any("missing nonlinear case 99" in msg for msg in msgs)
