"""Slab-modeling S7 — section cuts (design strips).

Validates model_geometry.section_cut integrating shell resultants along a line
against one-way-slab totals (midspan moment = wL²/8·B, support shear ≈ wLB/2),
plus the dialog contract.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

from femsolver.analysis.linear_static import LinearStaticAnalysis  # noqa: E402

import model_geometry as mg  # noqa: E402
from project import (Area, AreaLoad, Material, Node, Project,  # noqa: E402
                     ShellSection)


def _one_way(L=5.0, B=2.0, t=0.20, w=10_000.0):
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C", E=30e9, nu=0.2))
    p.shell_sections.append(ShellSection(id=1, name="T", thickness=t))
    p.nodes.extend([
        Node(id=1, x=0.0, y=0.0, z=0.0), Node(id=2, x=L, y=0.0, z=0.0),
        Node(id=3, x=L, y=B, z=0.0), Node(id=4, x=0.0, y=B, z=0.0),
    ])
    p.areas.append(Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=1,
                        mesh=(16, 6)))
    p.area_loads.append(AreaLoad(area=1, w=w, kind="gravity", case=1))
    m = p.build_model(with_loads=True)
    for tag, nd in m.nodes.items():
        x, _y, _z = nd.coords
        if abs(x) < 1e-9 or abs(x - L) < 1e-9:
            m.fix(tag, [1, 1, 1, 0, 0, 0])
    LinearStaticAnalysis(m).run()
    return m, dict(L=L, B=B, w=w)


# --------------------------------------------------------- integration

def test_midspan_moment_cut_matches_wL2_over_8_times_width():
    m, d = _one_way()
    L, B, w = d["L"], d["B"], d["w"]
    total = mg.section_cut(m, (L / 2, 0.0), (L / 2, B), "M", 64)
    assert abs(total) == pytest.approx(w * L ** 2 / 8.0 * B, rel=0.05)


def test_support_shear_cut_matches_half_load():
    m, d = _one_way()
    L, B, w = d["L"], d["B"], d["w"]
    total = mg.section_cut(m, (0.05, 0.0), (0.05, B), "V", 64)
    assert abs(total) == pytest.approx(w * L * B / 2.0, rel=0.12)


def test_zero_length_cut_is_zero():
    m, _d = _one_way()
    assert mg.section_cut(m, (1.0, 1.0), (1.0, 1.0), "M") == 0.0


def test_cut_off_the_slab_is_zero():
    m, d = _one_way()
    assert mg.section_cut(m, (100.0, 0.0), (100.0, d["B"]), "M") == 0.0


# ------------------------------------------------------------- dialog

def test_section_cut_dialog_computes():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    m, d = _one_way()
    L, B = d["L"], d["B"]
    # a project whose node ids/coords mirror the model's four corners
    p = Project(ndm=3, ndf=6)
    p.nodes.extend([Node(id=1, x=0.0, y=0.0, z=0.0),
                    Node(id=2, x=L, y=0.0, z=0.0),
                    Node(id=3, x=L, y=B, z=0.0),
                    Node(id=4, x=0.0, y=B, z=0.0)])
    from section_cut_dialog import SectionCutDialog
    from editing import _select
    dlg = SectionCutDialog(None, p, m)
    _select(dlg.n1, 2)                 # (L,0) → (L,B): a mid-support cut line
    _select(dlg.n2, 3)
    _select(dlg.quantity, "M")
    dlg._compute()
    assert dlg.result_value() is not None
