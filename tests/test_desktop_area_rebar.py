"""Slab-modeling S9 slice 2 — required-reinforcement (As) contour.

Covers model_geometry.areas_reinforcement_mesh (Wood-Armer moment → ACI As per
width, mm²/m, nodal-averaged), validated against the engine sizing for a one-way
slab, plus the design-inputs dialog contract.
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
from femsolver.design.wood_armer import required_reinforcement  # noqa: E402

import model_geometry as mg  # noqa: E402
from project import (Area, AreaLoad, Material, Node, Project,  # noqa: E402
                     ShellSection)


def _one_way(L=5.0, B=1.0, t=0.20, w=10_000.0):
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C", E=30e9, nu=0.2, rho=2400.0))
    p.shell_sections.append(ShellSection(id=1, name="T", thickness=t))
    p.nodes.extend([
        Node(id=1, x=0.0, y=0.0, z=0.0), Node(id=2, x=L, y=0.0, z=0.0),
        Node(id=3, x=L, y=B, z=0.0), Node(id=4, x=0.0, y=B, z=0.0),
    ])
    p.areas.append(Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=1,
                        mesh=(12, 2)))
    p.area_loads.append(AreaLoad(area=1, w=w, kind="gravity", case=1))
    m = p.build_model(with_loads=True)
    for tag, nd in m.nodes.items():
        x, _y, _z = nd.coords
        if abs(x) < 1e-9 or abs(x - L) < 1e-9:
            m.fix(tag, [1, 1, 1, 0, 0, 0])
    LinearStaticAnalysis(m).run()
    return m


# ------------------------------------------------------------ mesh

def test_reinforcement_none_without_areas():
    p = Project(ndm=3, ndf=6)
    p.nodes.append(Node(id=1, x=0.0, y=0.0, z=0.0))
    assert mg.areas_reinforcement_mesh(p.build_model(with_loads=False)) is None


def test_bottom_As_matches_engine_sizing():
    t, cover, fy, fc = 0.20, 0.025, 420e6, 30e6
    m = _one_way(t=t)
    poly = mg.areas_reinforcement_mesh(m, "As_x_bot", cover=cover, fy=fy, fc=fc)
    as_peak = float(np.nanmax(poly.point_data["value"]))          # mm²/m
    m11_peak = float(np.max(np.abs(mg.areas_result_mesh(m, "M11").point_data["value"])))
    expect = required_reinforcement(m11_peak, d=t - cover, fy=fy, fc=fc) * 1e6
    assert as_peak == pytest.approx(expect, rel=0.05)
    assert 400.0 < as_peak < 4000.0                              # sane mm²/m


def test_top_As_zero_for_sagging_one_way():
    # a simply-supported one-way slab has no hogging → ~no top steel
    m = _one_way()
    poly = mg.areas_reinforcement_mesh(m, "As_x_top")
    assert float(np.nanmax(poly.point_data["value"])) == pytest.approx(0.0, abs=1.0)


def test_more_cover_needs_more_steel():
    m = _one_way()
    a1 = np.nanmax(mg.areas_reinforcement_mesh(
        m, "As_x_bot", cover=0.02).point_data["value"])
    a2 = np.nanmax(mg.areas_reinforcement_mesh(
        m, "As_x_bot", cover=0.06).point_data["value"])
    assert a2 > a1                              # smaller d → more steel


# ------------------------------------------------------------ dialog

def test_reinforcement_dialog_data():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from slab_design_dialog import ReinforcementDialog
    dlg = ReinforcementDialog(None, Project(ndm=3, ndf=6))
    cfg = dlg.data()
    assert cfg["quantity"] in mg.AREA_REBAR_QUANTITIES
    assert cfg["cover"] == pytest.approx(0.025)
    assert cfg["fy"] == pytest.approx(420e6)
    assert cfg["fc"] == pytest.approx(30e6)
