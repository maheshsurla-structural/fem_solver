"""Response-spectrum analysis — desktop wiring.

Headless coverage of the code Sa(T) forms, the setup dialog (code spectra +
custom table + validation), the Analysis-cases row, and the
``MainWindow.run_response_spectrum`` runner.
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
    p = Project(ndm=2, ndf=3)
    p.nodes = [Node(1, 0, 0, supports=(1, 1, 1)),
               Node(2, 0, 3), Node(3, 0, 6)]
    p.sections = [Section(id=1, name="s", A=1e-2, Iz=1e-4)]
    p.materials = [Material(1, "steel", E=2e11, nu=0.3, rho=rho)]
    p.members = [Member(1, 1, 2, 1, 1), Member(2, 2, 3, 1, 1)]
    p.load_cases = [LoadCase(1, "Dead", "dead")]
    return p


# ---------------------------------------------------------------- Sa(T) forms
def test_asce7_spectrum_branches():
    from response_spectrum_dialog import _G, asce7_Sa
    SDS, SD1, TL = 1.0, 0.6, 8.0                 # T0=0.12, TS=0.6
    assert asce7_Sa(0.3, SDS=SDS, SD1=SD1, TL=TL) == pytest.approx(_G * SDS)
    # descending 1/T branch at T=1.2 (> TS, < TL): Sa = SD1/T
    assert asce7_Sa(1.2, SDS=SDS, SD1=SD1, TL=TL) == pytest.approx(
        _G * SD1 / 1.2)
    # long-period 1/T² branch beyond TL
    assert asce7_Sa(10.0, SDS=SDS, SD1=SD1, TL=TL) == pytest.approx(
        _G * SD1 * TL / 100.0)


# -------------------------------------------------------------------- dialog
def test_dialog_builds_every_source(qapp):
    from response_spectrum_dialog import ResponseSpectrumDialog
    d = ResponseSpectrumDialog(None, ndm=2, max_modes=18, default_modes=6)
    for src in ("asce7", "ec8", "is1893", "custom"):
        d.source.setCurrentIndex(d.source.findData(src))
        spec = d.build_spectrum()
        assert len(spec.periods) >= 2
        assert spec.Sa(0.3) > 0.0
    spectrum, num_modes, direction, combination = d.result()
    assert num_modes == 6 and direction == "x" and combination == "cqc"


def test_dialog_direction_options_2d_vs_3d(qapp):
    from response_spectrum_dialog import ResponseSpectrumDialog
    d2 = ResponseSpectrumDialog(None, ndm=2)
    assert [d2.direction.itemData(i) for i in range(d2.direction.count())] \
        == ["x", "y"]
    d3 = ResponseSpectrumDialog(None, ndm=3)
    assert "z" in [d3.direction.itemData(i)
                   for i in range(d3.direction.count())]


def test_custom_table_needs_two_points(qapp):
    from response_spectrum_dialog import ResponseSpectrumDialog
    d = ResponseSpectrumDialog(None, ndm=2)
    d.source.setCurrentIndex(d.source.findData("custom"))
    while d.custom.rowCount():
        d.custom.removeRow(0)
    d._add_custom_row(0.5, 5.0)                   # only one point
    with pytest.raises(ValueError):
        d.build_spectrum()


def test_results_dialog_mass_percentages(qapp):
    from response_spectrum_results_dialog import ResponseSpectrumResultsDialog
    info = {"direction": "x", "combination": "cqc", "damping_ratio": 0.05,
            "total_participating_mass": 90.0,
            "modal_results": [
                {"mode": 1, "period": 0.5, "Sa": 5.0, "Gamma": 9.0,
                 "modal_mass_eff": 81.0},
                {"mode": 2, "period": 0.2, "Sa": 6.0, "Gamma": 3.0,
                 "modal_mass_eff": 9.0}]}
    dlg = ResponseSpectrumResultsDialog(None, info, total_dir_mass=100.0)
    assert dlg.table.rowCount() == 2
    assert dlg.table.item(0, 4).text() == "81.0"     # mode-1 mass %
    assert dlg.table.item(1, 5).text() == "90.0"     # cumulative %


# ---------------------------------------------------------- analysis-cases row
def test_response_spectrum_is_a_saveable_case_type(qapp):
    # RS is the "object" case: params are the spectrum inputs; build_config
    # rebuilds the derived ResponseSpectrum headlessly.
    import case_types
    from femsolver import ResponseSpectrum
    ct = case_types.get("responsespectrum")
    assert ct is not None and ct.type_label == "Response Spectrum"
    params = {"source": "asce7", "damping": 0.05, "num_modes": 4,
              "direction": "x", "combination": "srss",
              "asce7": {"SDS": 1.0, "SD1": 0.6, "TL": 8.0}}
    spec, n, direction, comb = ct.build_config(_frame(), params)
    assert isinstance(spec, ResponseSpectrum)
    assert (n, direction, comb) == (4, "x", "srss")
    assert "ASCE 7" in ct.detail(_frame(), params)


def test_response_spectrum_params_round_trip(qapp):
    # seeding a dialog from params, then reading params() back, is stable
    from response_spectrum_dialog import ResponseSpectrumDialog
    params = {"source": "ec8", "damping": 0.03, "num_modes": 5,
              "direction": "y", "combination": "cqc",
              "asce7": {"SDS": 1.2, "SD1": 0.5, "TL": 6.0},
              "ec8": {"ag": 3.0, "ground": "D", "q": 2.0, "type": 2},
              "is1893": {"zone": 5, "I": 1.5, "R": 4.0, "soil": 3},
              "custom": []}
    dlg = ResponseSpectrumDialog(None, ndm=2, max_modes=20, initial=params,
                                 name="RS-Y")
    got = dlg.params()
    assert got["source"] == "ec8" and got["num_modes"] == 5
    assert got["direction"] == "y" and got["combination"] == "cqc"
    assert abs(got["damping"] - 0.03) < 1e-9
    assert got["ec8"] == {"ag": 3.0, "ground": "D", "q": 2.0, "type": 2}
    assert got["is1893"]["zone"] == 5 and got["is1893"]["soil"] == 3
    assert dlg.header.name() == "RS-Y"


def test_analysis_cases_lists_saved_response_spectrum_case(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog
    from project import AnalysisCase
    p = _frame()
    p.analysis_cases = [AnalysisCase(id=5, name="RS-X", type="responsespectrum",
                                     params={"source": "asce7", "num_modes": 6,
                                             "direction": "x",
                                             "combination": "cqc",
                                             "asce7": {"SDS": 1.0, "SD1": 0.6,
                                                       "TL": 8.0}})]
    dlg = AnalysisCasesDialog(None, p)
    kinds = [m["kind"] for m in dlg._row_meta]
    assert "analysis" in kinds
    dlg.table.setCurrentCell(kinds.index("analysis"), 0)
    assert dlg._mod_btn.isEnabled() and dlg._del_btn.isEnabled()
    dlg._run()
    assert dlg._run_request == ("case", 5)


# ----------------------------------------------------------- run_response_spectrum
def test_run_response_spectrum_end_to_end(qapp):
    from femsolver import ResponseSpectrum
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_frame())
    spec = ResponseSpectrum.from_function(lambda T: 5.0, T_min=0.01,
                                          T_max=10.0, damping_ratio=0.05)
    info = w.run_response_spectrum(config=(spec, 4, "x", "cqc"))
    assert info is not None and info["num_modes"] == 4
    assert info["direction"] == "x" and info["combination"] == "cqc"
    assert hasattr(w, "_rs_results_dlg")
    assert w._rs_results_dlg.table.rowCount() == 4
    assert "participating mass" in w.statusBar().currentMessage()


def test_run_response_spectrum_guards_zero_mass(qapp, monkeypatch):
    import main_window as MW
    from femsolver import ResponseSpectrum
    monkeypatch.setattr(MW.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    w = MW.MainWindow()
    w.load_project(_frame(rho=0.0))
    spec = ResponseSpectrum.from_function(lambda T: 5.0, T_min=0.01,
                                          T_max=10.0)
    assert w.run_response_spectrum(config=(spec, 2, "x", "cqc")) is None
