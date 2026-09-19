"""Wall-modeling W5 — openings in a wall panel (opening-aware meshing).

Covers Area.openings normalization, the kept-cell / needed-node helpers, that
build_model drops the opening's cells without orphaning nodes, that
_area_element_tags stays consistent, and the JSON round-trip.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

import project as proj  # noqa: E402
import walls  # noqa: E402
from project import Area, Material, Node, Project, ShellSection  # noqa: E402


def _wall(mesh=(6, 6), openings=None):
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C", E=30e9, nu=0.2, rho=2400.0))
    p.shell_sections.append(ShellSection(id=1, name="W", thickness=0.25))
    p.nodes.extend([Node(id=1, x=0.0, y=0.0, z=0.0),
                    Node(id=2, x=6.0, y=0.0, z=0.0)])
    ids = walls.build_wall_line(p, [1, 2], height=6.0, shell_section=1,
                                material=1, mesh=mesh, pier="P1")
    if openings is not None:
        p.area(ids[0]).openings = openings
    return p, ids[0]


# ------------------------------------------------------------- data model

def test_opening_normalization():
    a = Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=1,
             openings=[(0.7, 0.2, 0.3, 0.6),     # unsorted → sorted
                       (0.4, 0.4, 0.4, 0.8),     # zero width → dropped
                       "bad"])                    # junk → dropped
    assert a.openings == [(0.3, 0.2, 0.7, 0.6)]


def test_opening_clamped_to_unit_square():
    a = Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=1,
             openings=[(-0.2, 0.1, 1.5, 0.9)])
    assert a.openings == [(0.0, 0.1, 1.0, 0.9)]


# ------------------------------------------------------------- kept cells

def test_quad_cells_drop_opening():
    p, aid = _wall(mesh=(6, 6),
                   openings=[(1 / 3, 1 / 3, 2 / 3, 2 / 3)])   # centre 2×2 block
    cells = proj.area_quad_cells(p.area(aid))
    assert len(cells) == 36 - 4                  # 4 central cells removed
    # tag index stays k = j*n1 + i (stable); the removed ks are the centre block
    ks = {k for (k, _i, _j) in cells}
    removed = {j * 6 + i for j in (2, 3) for i in (2, 3)}
    assert ks.isdisjoint(removed)
    assert len(ks) == 32


def test_no_openings_is_unchanged():
    p, aid = _wall(mesh=(4, 5))
    assert len(proj.area_quad_cells(p.area(aid))) == 20   # full grid


# --------------------------------------------------------- build / no orphans

def test_build_drops_elements_and_leaves_no_orphans():
    p, aid = _wall(mesh=(6, 6), openings=[(1 / 3, 1 / 3, 2 / 3, 2 / 3)])
    m = p.build_model(with_loads=False)
    assert len(m.elements) == 32                 # 36 − 4 opening cells
    # every model node is used by at least one element → no singular free node
    used = set()
    for e in m.elements.values():
        used.update(e.node_tags)
    assert set(m.nodes.keys()) == used


def test_element_tags_match_built_elements():
    p, aid = _wall(mesh=(5, 5), openings=[(0.0, 0.0, 0.4, 0.4)])  # corner door
    m = p.build_model(with_loads=False)
    tags = set(p._area_element_tags(p.area(aid)))
    assert tags == set(m.elements.keys())        # tag list == what was emitted


def test_full_wall_still_builds_n1_by_n2():
    p, aid = _wall(mesh=(4, 6))
    m = p.build_model(with_loads=False)
    assert len(m.elements) == 24


# ------------------------------------------------------------- round-trip

def test_round_trip_preserves_openings():
    p, aid = _wall(mesh=(4, 4), openings=[(0.25, 0.5, 0.75, 1.0)])
    q = Project.from_json(p.to_json())
    assert q.area(aid).openings == [(0.25, 0.5, 0.75, 1.0)]
    assert len(q.build_model(with_loads=False).elements) == \
        len(p.build_model(with_loads=False).elements)


# ------------------------------------------------------------- GUI dialog

import os  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_opening_dialog_physical_to_fraction(qapp):
    from wall_opening_dialog import WallOpeningDialog
    from PySide6.QtWidgets import QTableWidgetItem
    p, aid = _wall(mesh=(4, 4))            # panel 6 m wide × 6 m tall
    dlg = WallOpeningDialog(None, p, p.area(aid))
    assert dlg._L == pytest.approx(6.0)
    assert dlg._H == pytest.approx(6.0)
    dlg.tbl.insertRow(0)
    for c, v in enumerate(["1.5", "3.0", "3.0", "3.0"]):   # X,Z,W,H metres
        dlg.tbl.setItem(0, c, QTableWidgetItem(v))
    dlg._accept()
    # X/L=0.25, Z/H=0.5, (X+W)/L=0.75, (Z+H)/H=1.0
    assert dlg.result == [pytest.approx((0.25, 0.5, 0.75, 1.0))]


def test_opening_dialog_loads_existing(qapp):
    from wall_opening_dialog import WallOpeningDialog
    p, aid = _wall(mesh=(4, 4), openings=[(0.25, 0.5, 0.75, 1.0)])
    dlg = WallOpeningDialog(None, p, p.area(aid))
    assert dlg.tbl.rowCount() == 1
    assert float(dlg.tbl.item(0, 2).text()) == pytest.approx(3.0)   # width 3 m


@pytest.fixture(scope="module")
def qapp_vtk():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvistaqt")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_edit_wall_openings_action(qapp_vtk, monkeypatch):
    from main_window import MainWindow
    import wall_opening_dialog as wod
    p, aid = _wall(mesh=(6, 6))
    w = MainWindow()
    w.load_project(p)
    w._set_selection([("area", aid)])
    monkeypatch.setattr(wod.WallOpeningDialog, "edit",
                        classmethod(lambda cls, *a, **k: [(1/3, 1/3, 2/3, 2/3)]))
    w.edit_wall_openings()
    assert w._project.area(aid).openings == [(1/3, 1/3, 2/3, 2/3)]
    # rebuild honours it
    assert len(w._project.build_model(with_loads=False).elements) == 32
    w._undo_stack.undo()
    assert w._project.area(aid).openings == []


def test_edit_wall_openings_guarded_without_selection(qapp_vtk, monkeypatch):
    from main_window import MainWindow
    import main_window as mw
    p, _aid = _wall(mesh=(2, 2))
    w = MainWindow()
    w.load_project(p)
    w._set_selection([])
    seen = {}
    monkeypatch.setattr(mw.QMessageBox, "information",
                        lambda *a, **k: seen.setdefault("info", a))
    w.edit_wall_openings()
    assert "info" in seen


# ------------------------------------------ non-rectangular (polygon) openings

def test_polygon_opening_drops_cells_no_orphans():
    # an L-shaped opening (6 verts) over the lower-left quadrant of an 8×8 wall
    poly = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.25),
            (0.25, 0.25), (0.25, 0.5), (0.0, 0.5)]
    p, aid = _wall(mesh=(8, 8), openings=[poly])
    cells = proj.area_quad_cells(p.area(aid))
    assert len(cells) < 64                       # some cells removed
    m = p.build_model(with_loads=False)
    used = set()
    for e in m.elements.values():
        used.update(e.node_tags)
    assert set(m.nodes.keys()) == used           # no orphaned nodes


def test_triangle_opening_point_in_poly():
    tri = [(0.25, 0.25), (0.75, 0.25), (0.5, 0.75)]
    p, aid = _wall(mesh=(8, 8), openings=[tri])
    # a cell centre inside the triangle is dropped; one clearly outside is kept
    assert proj._cell_in_opening([tri], 0.5, 0.35)      # inside
    assert not proj._cell_in_opening([tri], 0.05, 0.9)  # outside
    m = p.build_model(with_loads=False)
    assert 0 < len(m.elements) < 64


def test_polygon_opening_round_trip():
    tri = [(0.25, 0.25), (0.75, 0.25), (0.5, 0.75)]
    p, aid = _wall(mesh=(4, 4), openings=[tri])
    q = Project.from_json(p.to_json())
    assert q.area(aid).openings == [((0.25, 0.25), (0.75, 0.25), (0.5, 0.75))]
    assert len(q.build_model(with_loads=False).elements) == \
        len(p.build_model(with_loads=False).elements)


def test_rect_and_polygon_openings_mixed():
    rect = (0.0, 0.0, 0.25, 0.25)
    tri = [(0.6, 0.6), (0.9, 0.6), (0.75, 0.9)]
    p, aid = _wall(mesh=(8, 8), openings=[rect, tri])
    a = p.area(aid)
    assert len(a.openings) == 2
    assert proj._opening_is_rect(a.openings[0])
    assert not proj._opening_is_rect(a.openings[1])


def test_opening_dialog_preserves_polygon(qapp):
    from wall_opening_dialog import WallOpeningDialog
    tri = ((0.6, 0.6), (0.9, 0.6), (0.75, 0.9))
    p, aid = _wall(mesh=(4, 4), openings=[(0.0, 0.0, 0.25, 0.25), tri])
    dlg = WallOpeningDialog(None, p, p.area(aid))
    assert dlg.tbl.rowCount() == 1               # only the rect shows in the table
    dlg._accept()                                # editing rects must keep the poly
    assert tri in dlg.result
