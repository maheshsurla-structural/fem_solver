"""nl_report value conversion (plan U6) — the export must convert SI values to
match the unit labels it is given, so a report reads consistently with the
on-screen (converted) pushover plots.

Pure functions, no Qt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

import nl_report  # noqa: E402


def test_curve_csv_converts_to_labelled_units():
    # disp in m, shear in N → report in mm and kN
    csv = nl_report.curve_csv([0.0, 0.002], [0.0, 5000.0],
                              length_unit="mm", force_unit="kN")
    lines = csv.strip().splitlines()
    assert lines[0] == "step,displacement [mm],base_shear [kN]"
    # 0.002 m → 2 mm ; 5000 N → 5 kN
    assert lines[2].split(",")[1:] == ["2", "5"]


def test_curve_csv_si_is_identity():
    csv = nl_report.curve_csv([0.0, 0.002], [0.0, 5000.0])   # default m, N
    assert csv.strip().splitlines()[2].split(",")[1:] == ["0.002", "5000"]


def test_report_html_converts_metrics():
    result = {"protocol": "monotonic",
              "disp": [0.0, 0.001, 0.002],
              "shear": [0.0, 3000.0, 5000.0]}
    html = nl_report.report_html(result, {}, length_unit="mm", force_unit="kN")
    assert "5 kN" in html                     # peak base shear 5000 N → 5 kN
    assert "2 mm" in html                     # max |disp| 0.002 m → 2 mm
    # initial stiffness 3000 N / 0.001 m = 3e6 N/m → 3 kN/mm
    assert "3 kN/mm" in html


def test_fibers_csv_converts_geometry_keeps_stress_pa():
    csv = nl_report.fibers_csv([(0.05, -0.02, 2.5e7, 1.2e-3)], length_unit="mm")
    row = csv.strip().splitlines()[1].split(",")
    assert row[0] == "50" and row[1] == "-20"      # m → mm
    assert row[2] == "25000000"                    # stress stays Pa
