"""Slab-modeling S2 + S6-lite — the Area dialog and area face rendering.

Headless coverage of ``editing.AreaDialog`` (the create/edit contract, triangle
vs quad, validation) and ``model_geometry.areas_mesh`` (the filled-face
PolyData the viewport draws for slabs / shells).
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

from project import Area, Material, Node, Project, ShellSection  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _project():
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C30", E=30e9, nu=0.2, rho=2400.0))
    p.shell_sections.append(ShellSection(id=1, name="SLAB200", thickness=0.20))
    p.nodes.extend([
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=4.0, y=0.0, z=0.0),
        Node(id=3, x=4.0, y=3.0, z=0.0),
        Node(id=4, x=0.0, y=3.0, z=0.0),
    ])
    return p


# ------------------------------------------------------------ AreaDialog

def test_area_dialog_quad_data(qapp):
    from editing import AreaDialog
    from editing import _select
    dlg = AreaDialog(None, _project())
    for i, nid in enumerate([1, 2, 3, 4]):
        _select(dlg.corners[i], nid)
    a = dlg.data()
    assert a.nodes == [1, 2, 3, 4]
    assert a.shell_section == 1
    assert a.material == 1


def test_area_dialog_triangle_drops_none_corner(qapp):
    from editing import AreaDialog, _select
    dlg = AreaDialog(None, _project())
    for i, nid in enumerate([1, 2, 3]):
        _select(dlg.corners[i], nid)
    _select(dlg.corners[3], None)          # 4th corner = triangle
    a = dlg.data()
    assert a.nodes == [1, 2, 3]


def test_area_dialog_seed_nodes(qapp):
    from editing import AreaDialog
    dlg = AreaDialog(None, _project(), seed_nodes=[2, 3, 4])
    assert dlg._node_list()[:3] == [2, 3, 4]


def test_area_dialog_rejects_duplicate_corners(qapp, monkeypatch):
    import editing
    from PySide6.QtWidgets import QDialog
    warned = []
    monkeypatch.setattr(editing.QMessageBox, "warning",
                        lambda *a, **k: warned.append(a))
    from editing import AreaDialog, _select
    dlg = AreaDialog(None, _project())
    _select(dlg.corners[0], 1)
    _select(dlg.corners[1], 1)             # duplicate corner -> invalid
    _select(dlg.corners[2], 2)
    _select(dlg.corners[3], None)
    dlg.accept()
    assert warned                          # a validation warning fired
    assert dlg.result() != QDialog.DialogCode.Accepted

    _select(dlg.corners[1], 3)             # fix the duplicate
    dlg.accept()
    assert dlg.result() == QDialog.DialogCode.Accepted


# ------------------------------------------------------- areas_mesh (S6)

def test_areas_mesh_none_without_areas():
    import model_geometry as mg
    m = _project().build_model(with_loads=False)   # no areas yet
    assert mg.areas_mesh(m) is None


def test_areas_mesh_one_quad_face():
    import model_geometry as mg
    p = _project()
    p.areas.append(Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=1))
    m = p.build_model(with_loads=False)
    poly = mg.areas_mesh(m)
    assert poly is not None
    assert poly.n_cells == 1
    assert poly.n_points == 4


def test_areas_mesh_triangle_face():
    import model_geometry as mg
    p = _project()
    p.areas.append(Area(id=1, nodes=[1, 2, 3], shell_section=1, material=1))
    m = p.build_model(with_loads=False)
    poly = mg.areas_mesh(m)
    assert poly is not None
    assert poly.n_cells == 1
