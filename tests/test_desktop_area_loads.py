"""Slab-modeling S5b — area (surface / pressure) loads in the desktop model.

Covers the AreaLoad data round-trip, that apply_case compiles gravity / pressure
area loads onto the area's shell element, an end-to-end solve, and the dialog's
data contract.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

from femsolver.analysis.linear_static import LinearStaticAnalysis  # noqa: E402

from project import (Area, AreaLoad, Material, Node,  # noqa: E402
                     Project, ShellSection, area_element_tag)


def _slab_project(kind="gravity", w=5000.0):
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C30", E=30e9, nu=0.2, rho=2400.0))
    p.shell_sections.append(ShellSection(id=1, name="SLAB200", thickness=0.20))
    p.nodes.extend([
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=4.0, y=0.0, z=0.0),
        Node(id=3, x=4.0, y=3.0, z=0.0),
        Node(id=4, x=0.0, y=3.0, z=0.0),
    ])
    p.areas.append(Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=1))
    p.area_loads.append(AreaLoad(area=1, w=w, kind=kind, case=1))
    return p


# --------------------------------------------------------- round-trip

def test_area_load_round_trip():
    p = _slab_project(kind="pressure", w=1234.0)
    q = Project.from_json(p.to_json())
    assert len(q.area_loads) == 1
    al = q.area_loads[0]
    assert al.area == 1
    assert al.w == pytest.approx(1234.0)
    assert al.kind == "pressure"


def test_backward_compat_no_area_loads_key():
    d = {"schema": "femsolver-project/1", "ndm": 3, "ndf": 6}
    assert Project.from_dict(d).area_loads == []


# ------------------------------------------------------- apply_case

def test_gravity_area_load_compiles_to_element():
    p = _slab_project(kind="gravity", w=5000.0)
    m = p.build_model(with_loads=True)           # sums all cases (case 1)
    el = m.element(area_element_tag(1, 0))
    A = 4.0 * 3.0
    total = el.f_eq_global().reshape(4, 6)[:, :3].sum(axis=0)
    assert total[2] == pytest.approx(-5000.0 * A)     # global −Z gravity
    assert total[0] == pytest.approx(0.0, abs=1e-6)


def test_pressure_area_load_compiles_to_element():
    p = _slab_project(kind="pressure", w=800.0)
    m = p.build_model(with_loads=True)
    el = m.element(area_element_tag(1, 0))
    # flat XY slab → normal is ±Z, so pressure resolves onto Z
    A = 4.0 * 3.0
    total = el.f_eq_global().reshape(4, 6)[:, :3].sum(axis=0)
    assert abs(total[2]) == pytest.approx(800.0 * A)


def test_area_load_scales_with_combination_factor():
    p = _slab_project(kind="gravity", w=1000.0)
    m = p.build_model(with_loads=False)
    p.apply_case(m, case_id=1, factor=1.6)       # LRFR-style factor
    el = m.element(area_element_tag(1, 0))
    A = 4.0 * 3.0
    total = el.f_eq_global().reshape(4, 6)[:, 2].sum()
    assert total == pytest.approx(-1.6 * 1000.0 * A)


def test_area_load_solve_deflects_down():
    p = _slab_project(kind="gravity", w=5000.0)
    m = p.build_model(with_loads=True)
    for tag in (1, 2, 3):                         # clamp 3 corners
        m.fix(tag, [1, 1, 1, 1, 1, 1])
    LinearStaticAnalysis(m).run()
    assert m.node(4).disp[2] < 0.0               # free corner sags


# ------------------------------------------------------------ dialog

def test_area_load_dialog_data():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from area_load_dialog import AreaLoadDialog, _select
    dlg = AreaLoadDialog(None, _slab_project())
    _select(dlg.kind, "pressure")
    dlg.w.set_si(2000.0)
    al = dlg.data()
    assert al.area == 1
    assert al.kind == "pressure"
    assert al.w == pytest.approx(2000.0)
