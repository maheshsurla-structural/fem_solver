"""Nonlinear result export + report (plan §14 GUI-7).

Pure export/report functions (headless, no Qt) plus the pushover dialog's
export buttons wired through a monkeypatched save dialog.
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

import nl_report as R                                     # noqa: E402


# ------------------------------------------------------------- pure functions

def test_curve_csv_header_and_rows():
    txt = R.curve_csv([0.0, 0.01, 0.02], [0.0, 100.0, 150.0],
                      length_unit="m", force_unit="kN")
    lines = txt.strip().splitlines()
    assert lines[0] == "step,displacement [m],base_shear [kN]"
    assert lines[1].startswith("1,0,0")
    assert lines[3].split(",")[0] == "3"
    assert len(lines) == 4


def test_fibers_csv():
    txt = R.fibers_csv([(0.1, -0.2, 5.0e8, 0.003)], length_unit="m")
    lines = txt.strip().splitlines()
    assert lines[0] == "y [m],z [m],stress [Pa],strain"
    assert lines[1] == "0.1,-0.2,500000000,0.003"


def test_member_peak_strain_max_over_steps():
    res = {"damage_frames": [{1: 0.001, 2: 0.0005},
                             {1: 0.003, 2: 0.002},
                             {1: 0.0025, 2: 0.004}]}
    assert R.member_peak_strain(res) == {1: 0.003, 2: 0.004}


def test_run_summary_monotonic():
    res = {"protocol": "monotonic", "disp": [0.0, 0.01, 0.02],
           "shear": [0.0, 100.0, 150.0]}
    s = R.run_summary(res)
    assert s["steps"] == 3
    assert s["peak_shear"] == 150.0
    assert s["disp_at_peak_shear"] == 0.02
    assert s["max_abs_disp"] == 0.02
    assert s["initial_stiffness"] == pytest.approx(10000.0)
    assert "dissipated_energy" not in s


def test_run_summary_cyclic_and_strain():
    res = {"protocol": "cyclic", "disp": [0.0, 0.01, 0.0, -0.01, 0.0],
           "shear": [0.0, 100.0, 20.0, -100.0, -20.0],
           "damage_frames": [{1: 0.001}, {1: 0.004}]}
    s = R.run_summary(res)
    assert "dissipated_energy" in s and s["dissipated_energy"] >= 0.0
    assert s["peak_fiber_strain"] == 0.004


def test_report_html_contains_sections():
    res = {"protocol": "cyclic", "disp": [0.0, 0.01, -0.01],
           "shear": [0.0, 120.0, -110.0],
           "damage_frames": [{1: 0.006, 2: 0.001}]}
    meta = {"Project": "Demo", "Case": "1: Cyclic", "Control": "node 2 Uy"}
    html = R.report_html(res, meta, length_unit="m", force_unit="kN",
                         curve_png_b64="QUJD")           # dummy base64
    assert "<html" in html and "</html>" in html
    assert "Demo" in html and "1: Cyclic" in html         # meta rendered
    assert "Peak base shear" in html and "kN" in html     # metrics + units
    assert "Dissipated energy" in html                    # cyclic metric
    assert "Member hinge state" in html                   # hinge table
    assert "yielded" in html                              # member 1 > yield
    assert "data:image/png;base64,QUJD" in html           # embedded curve


def test_report_html_without_image_or_strain():
    res = {"protocol": "monotonic", "disp": [0.0, 0.01], "shear": [0.0, 90.0]}
    html = R.report_html(res, {"Project": "P"})
    assert "Member hinge state" not in html               # no damage frames
    assert "data:image" not in html                       # no image passed


# ------------------------------------------------------------- dialog wiring

def _gsd_column_project():
    import section_gui_core as core
    from project import Material, Member, Node, Project, Section
    spec = core.Spec(kind="Circular", D=0.6, fc=35e6, fy=500e6)
    gsd = dataclasses.asdict(spec)
    A = math.pi * 0.6**2 / 4.0
    Iz = math.pi * 0.6**4 / 64.0
    p = Project()
    p.nodes = [Node(1, 0.0, 0.0, supports=(1, 1, 1)), Node(2, 3.0, 0.0)]
    p.sections = [Section(id=1, name="col", A=A, Iz=Iz, gsd_spec=gsd)]
    p.materials = [Material(1, "conc", E=30e9, nu=0.2)]
    p.members = [Member(1, 1, 2, 1, 1)]
    return p


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _run_dialog(project):
    from pushover_dialog import PushoverDialog, PushoverWorker
    dlg = PushoverDialog(None, project)
    dlg.n_steps.setValue(12)
    wk = PushoverWorker(project, dlg._kwargs())
    wk.progress.connect(dlg._on_progress)
    wk.done.connect(dlg._on_done)
    wk.run()
    return dlg


def test_export_buttons_enable_after_run(qapp):
    dlg = _run_dialog(_gsd_column_project())
    assert dlg._disp
    assert all(b.isEnabled() for b in dlg._export_btns)


def test_dialog_exports_curve_and_report(qapp, tmp_path, monkeypatch):
    import pushover_dialog as PD
    dlg = _run_dialog(_gsd_column_project())

    curve = tmp_path / "curve.csv"
    monkeypatch.setattr(PD.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(curve), "CSV")))
    dlg._export_curve_csv()
    assert curve.exists()
    assert curve.read_text().splitlines()[0].startswith("step,displacement")

    report = tmp_path / "report.html"
    monkeypatch.setattr(PD.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(report), "HTML")))
    dlg._export_report()
    assert report.exists()
    body = report.read_text()
    assert "<html" in body and "Peak base shear" in body

    fibers = tmp_path / "fibers.csv"
    monkeypatch.setattr(PD.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(fibers), "CSV")))
    dlg._export_fibers_csv()
    assert fibers.exists()
    assert fibers.read_text().splitlines()[0].startswith("y [")


def test_dialog_export_cancel_writes_nothing(qapp, tmp_path, monkeypatch):
    import pushover_dialog as PD
    dlg = _run_dialog(_gsd_column_project())
    monkeypatch.setattr(PD.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))   # cancelled
    dlg._export_curve_csv()                                       # no crash
    assert not list(tmp_path.iterdir())
