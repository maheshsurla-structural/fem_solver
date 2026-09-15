"""Modal (free-vibration) analysis — desktop wiring.

Headless coverage of the material-density prerequisite, the Modal setup /
results dialogs, the Analysis-cases row, and the ``MainWindow.run_modal``
runner (including the empty-mass and too-small guards).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "desktop"))

from project import (LoadCase, Material, Member, Node,  # noqa: E402
                     Project, Section)


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _frame(rho: float = 7850.0) -> Project:
    """A tiny cantilever with mass, enough for a couple of modes."""
    p = Project(ndm=2, ndf=3)
    p.nodes = [Node(1, 0, 0, supports=(1, 1, 1)),
               Node(2, 0, 3), Node(3, 0, 6)]
    p.sections = [Section(id=1, name="s", A=1e-2, Iz=1e-4)]
    p.materials = [Material(1, "steel", E=2e11, nu=0.3, rho=rho)]
    p.members = [Member(1, 1, 2, 1, 1), Member(2, 2, 3, 1, 1)]
    p.load_cases = [LoadCase(1, "Dead", "dead")]
    return p


# ------------------------------------------------------------ density → mass
def test_material_density_field_defaults_zero():
    m = Material(1, "m", E=2e11, nu=0.3)
    assert m.rho == 0.0


def test_build_model_carries_density_into_element_mass():
    from femsolver.analysis.assembler import assemble_mass
    model = _frame(rho=7850.0).build_model(with_loads=False)
    model.number_dofs()
    assert abs(assemble_mass(model)).max() > 0.0
    # a zero-density project produces a zero (singular) mass matrix
    model0 = _frame(rho=0.0).build_model(with_loads=False)
    model0.number_dofs()
    assert abs(assemble_mass(model0)).max() == 0.0


def test_material_editor_roundtrips_density(qapp):
    from material_editor import MaterialDialog
    p = _frame()
    dlg = MaterialDialog(None, p, Material(1, "C", E=25e9, nu=0.2, rho=2400.0))
    assert dlg.rho.value() == 2400.0
    assert dlg.data().rho == 2400.0
    # a brand-new material defaults to steel density
    assert MaterialDialog(None, p, None).data().rho == 7850.0


# ------------------------------------------------------------------- dialogs
def test_modal_dialog_result_and_clamp(qapp):
    from modal_dialog import ModalDialog
    d = ModalDialog(None, max_modes=10, default_modes=6)
    assert d.result() == (6, False)
    d.mass.setCurrentIndex(1)                     # lumped
    assert d.result()[1] is True
    # default is clamped to the available modes
    d2 = ModalDialog(None, max_modes=3, default_modes=6)
    assert d2.modes.value() == 3 and d2.modes.maximum() == 3


def test_modal_results_dialog_rows_and_preview(qapp):
    from modal_results_dialog import ModalResultsDialog
    info = {"num_modes": 3, "periods_s": [0.5, 0.2, 0.1],
            "frequencies_hz": [2.0, 5.0, 10.0]}
    seen = []
    dlg = ModalResultsDialog(None, info, on_show_mode=seen.append)
    assert dlg.table.rowCount() == 3
    assert seen == [0]                            # mode 1 auto-previewed
    dlg.table.setCurrentCell(2, 0)
    assert seen[-1] == 2


# ---------------------------------------------------------- analysis-cases row
def test_modal_is_a_saveable_case_type(qapp):
    # Modal is now a saved, multi-instance AnalysisCase type (Add ▾ menu), not a
    # fixed launcher row — see the analysis-cases-manager plan.
    import case_types
    ct = case_types.get("modal")
    assert ct is not None and ct.type_label == "Modal"
    assert ct.build_config(_frame(), {"num_modes": 5, "lumped": True}) == (5, True)


def test_analysis_cases_lists_saved_modal_case(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog
    from project import AnalysisCase
    p = _frame()
    p.analysis_cases = [AnalysisCase(id=1, name="Modal-6",
                                     type="modal",
                                     params={"num_modes": 6, "lumped": False})]
    dlg = AnalysisCasesDialog(None, p)
    kinds = [m["kind"] for m in dlg._row_meta]
    assert "analysis" in kinds
    r = kinds.index("analysis")
    dlg.table.setCurrentCell(r, 0)
    assert dlg._run_btn.isEnabled()          # runnable
    assert dlg._mod_btn.isEnabled()          # and now editable
    assert dlg._del_btn.isEnabled()
    dlg._run()
    assert dlg._run_request == ("case", 1)


# --------------------------------------------------------------- run_modal
def test_run_modal_extracts_modes_and_opens_results(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_frame())
    info = w.run_modal(num_modes=3, lumped=False)
    assert info is not None and info["num_modes"] == 3
    assert len(info["periods_s"]) == 3
    assert hasattr(w, "_modal_results_dlg")
    w._modal_results_dlg.table.setCurrentCell(1, 0)
    assert "Mode 2" in w.statusBar().currentMessage()


def test_run_modal_guards_zero_mass(qapp, monkeypatch):
    import main_window as MW
    monkeypatch.setattr(MW.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    w = MW.MainWindow()
    w.load_project(_frame(rho=0.0))
    assert w.run_modal(num_modes=1) is None


def test_run_modal_guards_no_members(qapp):
    from main_window import MainWindow
    w = MainWindow()
    p = Project(ndm=2, ndf=3)
    p.nodes = [Node(1, 0, 0, supports=(1, 1, 1))]
    p.materials = [Material(1, "m", E=2e11, nu=0.3, rho=7850.0)]
    w.load_project(p)
    assert w.run_modal(num_modes=1) is None
