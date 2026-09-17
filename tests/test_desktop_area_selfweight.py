"""Slab-modeling — automatic slab self-weight area load (ρ·t·g).

Covers _area_selfweight, the apply path (self-weight total = ρ·t·g·A on the
built model), that ``w`` is ignored for self-weight, and the dialog contract.
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

from project import (AreaLoad, GRAVITY, Area, Material, Node,  # noqa: E402
                     Project, ShellSection, area_element_tag)


def _slab(rho=2400.0, t=0.20):
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C", E=30e9, nu=0.2, rho=rho))
    p.shell_sections.append(ShellSection(id=1, name="T", thickness=t))
    p.nodes.extend([
        Node(id=1, x=0.0, y=0.0, z=0.0), Node(id=2, x=4.0, y=0.0, z=0.0),
        Node(id=3, x=4.0, y=3.0, z=0.0), Node(id=4, x=0.0, y=3.0, z=0.0),
    ])
    p.areas.append(Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=1,
                        mesh=(1, 1)))
    return p


def test_area_selfweight_value():
    p = _slab(rho=2500.0, t=0.25)
    assert p._area_selfweight(p.areas[0]) == pytest.approx(2500.0 * 0.25 * GRAVITY)


def test_selfweight_total_on_model():
    p = _slab(rho=2400.0, t=0.20)
    p.area_loads.append(AreaLoad(area=1, kind="selfweight", case=1))
    m = p.build_model(with_loads=True)
    A = 4.0 * 3.0
    total = m.element(area_element_tag(1, 0)).f_eq_global().reshape(4, 6)[:, 2].sum()
    assert total == pytest.approx(-2400.0 * 0.20 * GRAVITY * A, rel=1e-9)


def test_selfweight_ignores_w():
    p = _slab()
    # a stray w must not affect a self-weight load
    p.area_loads.append(AreaLoad(area=1, w=99999.0, kind="selfweight", case=1))
    m = p.build_model(with_loads=True)
    A = 4.0 * 3.0
    total = m.element(area_element_tag(1, 0)).f_eq_global().reshape(4, 6)[:, 2].sum()
    assert total == pytest.approx(-2400.0 * 0.20 * GRAVITY * A, rel=1e-9)


def test_selfweight_zero_without_density():
    p = _slab(rho=0.0)
    assert p._area_selfweight(p.areas[0]) == 0.0


def test_area_load_dialog_selfweight_disables_w():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from area_load_dialog import AreaLoadDialog, _select
    dlg = AreaLoadDialog(None, _slab())
    _select(dlg.kind, "selfweight")
    dlg._on_kind()
    assert dlg.w.isEnabled() is False
    al = dlg.data()
    assert al.kind == "selfweight" and al.w == 0.0
