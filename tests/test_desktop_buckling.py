"""Linear buckling — desktop wiring.

Headless coverage of the member-sub-dividing buckling model builder, the
setup dialog, the Analysis-cases row, and the ``MainWindow.run_buckling``
runner (Euler validation, 3-D gate, no-compression guard).
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "desktop"))

from project import (Load, LoadCase, Material, Member,  # noqa: E402
                     Node, Project, Section)


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


# fixed-base column: gravity tip load → compression → cantilever buckling
_E, _A, _Iz, _L = 2.0e11, 1.0e-3, 1.0e-7, 5.0
_P_EULER = math.pi ** 2 * _E * _Iz / (2.0 * _L) ** 2      # effective length 2L


def _column(load=(0.0, -1.0, 0.0)) -> Project:
    p = Project(ndm=2, ndf=3)
    p.nodes = [Node(1, 0, 0, supports=(1, 1, 1)), Node(2, 0, _L)]
    p.sections = [Section(id=1, name="s", A=_A, Iz=_Iz)]
    p.materials = [Material(1, "steel", E=_E, nu=0.3, rho=7850.0)]
    p.members = [Member(1, 1, 2, 1, 1)]
    p.load_cases = [LoadCase(1, "Grav", "dead")]
    p.loads = [Load(node=2, values=load, case=1)]
    return p


# ---------------------------------------------------------- model builder
def test_build_buckling_model_subdivides_and_hits_euler():
    from femsolver import LinearBucklingAnalysis
    model, subs = _column().build_buckling_model(subdivisions=8)
    assert len(subs[1]) == 8                       # member meshed into 8
    assert len(model.nodes) == 9                   # 2 ends + 7 internal
    res = LinearBucklingAnalysis(model, num_modes=1).run()
    assert res["critical_load_factor"] == pytest.approx(_P_EULER, rel=1.0e-3)


def test_build_buckling_model_3d_uses_beamcolumn3d():
    from femsolver import BeamColumn3D
    p = Project(ndm=3, ndf=6)
    p.nodes = [Node(1, 0, 0, z=0, supports=(1, 1, 1, 1, 1, 1)),
               Node(2, _L, 0, z=0)]
    p.sections = [Section(id=1, name="s", A=_A, Iz=_Iz, Iy=2 * _Iz, J=1e-8)]
    p.materials = [Material(1, "steel", E=_E, nu=0.3, rho=7850.0)]
    p.members = [Member(1, 1, 2, 1, 1)]
    model, subs = p.build_buckling_model(subdivisions=4)
    assert len(subs[1]) == 4
    assert isinstance(model.element(subs[1][0]), BeamColumn3D)


# ------------------------------------------------------------------ dialog
def test_buckling_dialog_result(qapp):
    from buckling_dialog import BucklingDialog
    d = BucklingDialog(None, _column(), max_modes=10, default_modes=4)
    sel, modes, subdiv = d.result()
    assert sel == ("all", None) and modes == 4 and subdiv == 6
    # the reference combo offers the project's load cases too
    labels = [d.reference.itemText(i) for i in range(d.reference.count())]
    assert any("Grav" in t for t in labels)


def test_buckling_results_dialog_preview(qapp):
    from buckling_results_dialog import BucklingResultsDialog
    info = {"load_factors": [1973.9, 17768.0, 49408.0],
            "critical_load_factor": 1973.9, "num_modes": 3}
    seen = []
    dlg = BucklingResultsDialog(None, info, "all load cases", seen.append)
    assert dlg.table.rowCount() == 3
    assert seen == [0]
    dlg.table.setCurrentCell(2, 0)
    assert seen[-1] == 2


# ---------------------------------------------------------- analysis-cases row
def test_analysis_cases_lists_buckling(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog, _PLANNED
    assert "Buckling" not in _PLANNED
    dlg = AnalysisCasesDialog(None, _column())
    kinds = [m["kind"] for m in dlg._row_meta]
    assert "buckling" in kinds
    dlg.table.setCurrentCell(kinds.index("buckling"), 0)
    dlg._run()
    assert dlg._run_request == ("buckling",)


# --------------------------------------------------------------- run_buckling
def test_run_buckling_end_to_end(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_column())
    info = w.run_buckling(config=(("all", None), 3, 8))
    assert info is not None
    assert info["critical_load_factor"] == pytest.approx(_P_EULER, rel=1.0e-3)
    assert hasattr(w, "_buckling_results_dlg")
    assert w._buckling_results_dlg.table.rowCount() == 3


def test_run_buckling_3d_hits_euler(qapp):
    """3-D coverage (A3.1): a pinned 3-D column buckles about its weak axis at
    the Euler load, via the desktop runner."""
    import math

    from main_window import MainWindow
    from project import Load, LoadCase
    E, A, Iz, Iy, J, L, n = 2.0e11, 1.0e-3, 1.0e-6, 2.0e-6, 1.0e-6, 5.0, 10
    p = Project(ndm=3, ndf=6)
    for i in range(n + 1):
        p.nodes.append(Node(id=i + 1, x=i * L / n, y=0.0, z=0.0))
    p.nodes[0].supports = (1, 1, 1, 1, 0, 0)
    p.nodes[-1].supports = (0, 1, 1, 1, 0, 0)
    p.sections = [Section(id=1, name="col", A=A, Iz=Iz, Iy=Iy, J=J)]
    p.materials = [Material(1, "steel", E=E, nu=0.3, rho=7850.0)]
    p.members = [Member(i + 1, i + 1, i + 2, 1, 1) for i in range(n)]
    p.load_cases = [LoadCase(1, "Axial", "dead")]
    p.loads = [Load(node=n + 1, values=(-1.0, 0, 0, 0, 0, 0), case=1)]
    w = MainWindow()
    w.load_project(p)
    info = w.run_buckling(config=(("all", None), 2, 16))
    assert info is not None
    p_weak = math.pi ** 2 * E * min(Iy, Iz) / L ** 2
    assert info["critical_load_factor"] == pytest.approx(p_weak, rel=1e-3)


def test_run_buckling_guards_no_compression(qapp, monkeypatch):
    import main_window as MW
    monkeypatch.setattr(MW.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: None))
    w = MW.MainWindow()
    w.load_project(_column(load=(0.0, +1.0, 0.0)))     # tensile → no buckling
    assert w.run_buckling(config=(("all", None), 2, 8)) is None
