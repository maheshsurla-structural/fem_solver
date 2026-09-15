"""Load rating (AASHTO LRFR) — desktop wiring (bridge GUI increment G7).

Headless coverage of the setup dialog (incl. display-unit conversion), the
Analysis-cases row, and the ``MainWindow.run_load_rating`` runner (2-D + 3-D).
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

from project import Material, Member, Node, Project, Section  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _girder(nspan=8, dx=3.0):
    p = Project(ndm=2, ndf=3)
    p.nodes = [Node(i + 1, i * dx, 0.0) for i in range(nspan + 1)]
    p.nodes[0].supports = (1, 1, 0)
    p.nodes[-1].supports = (0, 1, 0)
    p.sections = [Section(id=1, name="g", A=0.5, Iz=0.2)]
    p.materials = [Material(1, "steel", E=2e11, nu=0.3, rho=7850.0)]
    p.members = [Member(i + 1, i + 1, i + 2, 1, 1) for i in range(nspan)]
    return p


# a config that bypasses the dialog (SI capacity / dead-load effects)
def _cfg(nel=8, **over):
    cfg = {"lane": list(range(1, nel + 2)),
           "response": ("M", nel // 2, "j"),
           "Rn": 8.0e6, "DC": 1.0e6, "DW": 0.3e6, "P": 0.0,
           "phi": 1.0, "phi_c": 1.0, "phi_s": 1.0, "im": 0.33,
           "adtt": 5000, "permit_gamma_LL": None}
    cfg.update(over)
    return cfg


# --------------------------------------------------------------- dialog
def test_dialog_defaults(qapp):
    from load_rating_dialog import LoadRatingDialog
    d = LoadRatingDialog(None, _girder())
    assert d.lane_nodes() == [1, 2, 3, 4, 5, 6, 7, 8, 9]     # all, X-ordered
    assert d.effect.count() == 2                             # moment / shear
    cfg = d.result()
    for k in ("lane", "response", "Rn", "DC", "DW", "phi",
              "phi_c", "phi_s", "im", "adtt"):
        assert k in cfg
    assert cfg["response"][0] == "M"
    assert cfg["adtt"] == 5000                               # legal on by default
    assert cfg["permit_gamma_LL"] is None                    # permit off


def test_dialog_converts_display_units_to_si(qapp):
    """Rn/DC/DW are entered in the project's units and returned in SI."""
    from load_rating_dialog import LoadRatingDialog
    p = _girder()
    p.force_unit = "kN"                                      # display in kN·m
    d = LoadRatingDialog(None, p)
    d.Rn.setValue(1000.0)                                    # 1000 kN·m
    cfg = d.result()
    assert cfg["Rn"] == pytest.approx(1.0e6)                 # → 1e6 N·m (SI)


def test_dialog_permit_toggle(qapp):
    from load_rating_dialog import LoadRatingDialog
    d = LoadRatingDialog(None, _girder())
    d.permit.setChecked(True)
    d.permit_gamma.setValue(1.20)
    d.legal.setChecked(False)
    cfg = d.result()
    assert cfg["permit_gamma_LL"] == pytest.approx(1.20)
    assert cfg["adtt"] is None                               # legal switched off


# ---------------------------------------------------- analysis-cases row
def test_load_rating_is_a_saveable_case_type(qapp):
    import case_types
    ct = case_types.get("loadrating")
    assert ct is not None and ct.type_label == "Load Rating"
    params = {"response": ["M", 2, "i"], "adtt": 5000, "permit_gamma_LL": None}
    d = ct.detail(_girder(), params)
    assert "moment @ member 2" in d and "legal" in d and "permit" not in d


def test_analysis_cases_lists_saved_load_rating_case(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog
    from project import AnalysisCase
    p = _girder()
    p.analysis_cases = [AnalysisCase(id=9, name="Girder RF", type="loadrating",
                                     params={"lane": [1, 2],
                                             "response": ["M", 1, "i"],
                                             "Rn": 1000.0, "DC": 200.0,
                                             "DW": 50.0, "P": 0.0, "phi": 1.0,
                                             "phi_c": 1.0, "phi_s": 1.0,
                                             "im": 0.33, "adtt": None,
                                             "permit_gamma_LL": None})]
    dlg = AnalysisCasesDialog(None, p)
    kinds = [m["kind"] for m in dlg._row_meta]
    assert "analysis" in kinds
    dlg.table.setCurrentCell(kinds.index("analysis"), 0)
    assert dlg._mod_btn.isEnabled() and dlg._del_btn.isEnabled()
    dlg._run()
    assert dlg._run_request == ("case", 9)


# --------------------------------------------------------- run_load_rating
def test_run_load_rating(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_girder())
    # ADTT 100 → legal gamma_LL 1.40 (< inventory 1.75), so inventory controls
    res = w.run_load_rating(config=_cfg(adtt=100))
    assert res is not None
    assert res["ll_im"] > 0.0
    rating = res["rating"]
    assert set(rating.results) == {"inventory", "operating", "legal"}
    # operating (gamma_LL 1.35) rates higher than inventory (1.75)
    assert rating["operating"].rf > rating["inventory"].rf
    assert rating["legal"].rf > rating["inventory"].rf      # 1.40 < 1.75
    # controlling = lowest RF = inventory
    assert rating.controlling.level == "inventory"
    assert hasattr(w, "_load_rating_results_dlg")
    assert "Load rating" in w.statusBar().currentMessage()


def test_run_load_rating_busy_route_legal_controls(qapp):
    """At ADTT 5000 the legal factor 1.80 exceeds inventory 1.75, so the
    legal rating controls (more demanding)."""
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_girder())
    res = w.run_load_rating(config=_cfg(adtt=5000))
    rating = res["rating"]
    assert rating.controlling.level == "legal"
    assert rating["legal"].rf < rating["inventory"].rf


def test_run_load_rating_permit_level(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_girder())
    res = w.run_load_rating(config=_cfg(adtt=None, permit_gamma_LL=1.15))
    rating = res["rating"]
    assert set(rating.results) == {"inventory", "operating", "permit"}
    assert rating["permit"].rf > rating["operating"].rf     # gamma_LL 1.15


def test_run_load_rating_deficient(qapp):
    """A small capacity → RF < 1 (deficient)."""
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_girder())
    res = w.run_load_rating(config=_cfg(Rn=1.0e6))
    assert not res["rating"].controlling.adequate
    assert "deficient" in w.statusBar().currentMessage().lower()


def test_run_load_rating_guards(qapp, monkeypatch):
    import main_window as MW
    monkeypatch.setattr(MW.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    w = MW.MainWindow()
    w.load_project(_girder())
    # too few lane nodes
    assert w.run_load_rating(config=_cfg(lane=[1])) is None


def test_run_load_rating_3d(qapp):
    """The rating works on a 3-D girder line via the vertical (uz) DOF."""
    from main_window import MainWindow
    n, L = 8, 24.0
    p = Project(ndm=3, ndf=6)
    p.nodes = [Node(i + 1, i * L / n, 0.0, z=0.0) for i in range(n + 1)]
    p.nodes[0].supports = (1, 1, 1, 1, 0, 0)
    p.nodes[-1].supports = (0, 1, 1, 1, 0, 0)
    p.sections = [Section(id=1, name="g", A=0.5, Iz=0.2, Iy=0.2, J=0.05)]
    p.materials = [Material(1, "steel", E=2e11, nu=0.3, rho=7850.0)]
    p.members = [Member(i + 1, i + 1, i + 2, 1, 1) for i in range(n)]
    w = MainWindow()
    w.load_project(p)
    res = w.run_load_rating(config=_cfg(nel=n, lane=list(range(1, n + 2))))
    assert res is not None and res["ll_im"] > 0.0
    assert res["rating"].controlling.rf > 0.0
