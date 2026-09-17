"""Slab-modeling S9 slice 3 — punching-shear check at a slab column.

Covers the ACI check calc (slab_punching.aci_punching_check), the demand read
from a node's vertical reaction after a solve, and the dialog wiring.
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

from project import (Area, AreaLoad, Material, Node, Project,  # noqa: E402
                     ShellSection)
from slab_punching import (aci_punching_check,  # noqa: E402
                           punching_demand_from_reaction)


def _corner_supported_slab(L=6.0, t=0.25, w=12_000.0):
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C", E=30e9, nu=0.2))
    p.shell_sections.append(ShellSection(id=1, name="T", thickness=t))
    fix = (1, 1, 1, 1, 1, 1)
    p.nodes.extend([
        Node(id=1, x=0.0, y=0.0, z=0.0, supports=fix),
        Node(id=2, x=L, y=0.0, z=0.0, supports=fix),
        Node(id=3, x=L, y=L, z=0.0, supports=fix),
        Node(id=4, x=0.0, y=L, z=0.0, supports=fix),
    ])
    p.areas.append(Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=1,
                        mesh=(4, 4)))
    p.area_loads.append(AreaLoad(area=1, w=w, kind="gravity", case=1))
    m = p.build_model(with_loads=True)
    LinearStaticAnalysis(m).run()
    return m, L, w


# ------------------------------------------------------------- calc

def test_aci_check_safe_column_ok():
    r = aci_punching_check(200e3, c_x=0.4, c_y=0.4, d=0.2, f_c=30e6)
    assert r["ok"] is True and r["dcr"] < 1.0
    assert r["code"] == "ACI 318-19"
    assert r["b_0"] == pytest.approx(2 * (0.4 + 0.2) + 2 * (0.4 + 0.2))


def test_aci_check_overloaded_needs_reinforcement():
    r = aci_punching_check(2000e3, c_x=0.3, c_y=0.3, d=0.18, f_c=25e6)
    assert r["ok"] is False and r["needs_reinforcement"] is True
    assert r["dcr"] > 1.0


def test_dcr_scales_with_demand():
    a = aci_punching_check(300e3, c_x=0.4, c_y=0.4, d=0.2, f_c=30e6)["dcr"]
    b = aci_punching_check(600e3, c_x=0.4, c_y=0.4, d=0.2, f_c=30e6)["dcr"]
    assert b == pytest.approx(2 * a, rel=1e-9)


# ----------------------------------------------- demand from reaction

def test_demand_from_corner_reaction():
    m, L, w = _corner_supported_slab()
    total = w * L * L
    v = punching_demand_from_reaction(m, 1)         # a corner support
    assert v == pytest.approx(total / 4.0, rel=1e-6)   # symmetry → total/4


def test_demand_zero_at_free_node():
    m, _L, _w = _corner_supported_slab()
    # an interior mesh node carries no reaction
    interior = max(m.nodes)                          # a generated mesh node
    assert punching_demand_from_reaction(m, interior) == pytest.approx(0.0, abs=1.0)


# ------------------------------------------------------------- dialog

def test_punching_dialog_runs_check():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    m, _L, _w = _corner_supported_slab()
    p = Project(ndm=3, ndf=6)
    p.nodes.extend([Node(id=t, x=0.0, y=0.0, z=0.0) for t in (1, 2, 3, 4)])
    from slab_punching_dialog import SlabPunchingDialog
    dlg = SlabPunchingDialog(None, p, m)
    from editing import _select
    _select(dlg.node, 1)
    dlg._run_check()
    res = dlg.result_data()
    assert res is not None and "dcr" in res and res["V_u"] > 0.0
