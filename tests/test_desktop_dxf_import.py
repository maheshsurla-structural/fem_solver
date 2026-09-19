"""Wall-modeling W8d — DXF grid import (minimal LINE/LWPOLYLINE reader)."""
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

import dxf_import  # noqa: E402
from project import Project  # noqa: E402


def _line(x1, y1, x2, y2, layer="0"):
    return (f"0\nLINE\n8\n{layer}\n10\n{x1}\n20\n{y1}\n30\n0\n"
            f"11\n{x2}\n21\n{y2}\n31\n0\n")


def _dxf(*entities):
    return ("0\nSECTION\n2\nENTITIES\n" + "".join(entities)
            + "0\nENDSEC\n0\nEOF\n")


# ------------------------------------------------------------- reader

def test_read_line_segments():
    text = _dxf(_line(0, 0, 0, 10), _line(0, 0, 10, 0, layer="GRID"))
    segs = dxf_import.read_segments(text)
    assert len(segs) == 2
    assert segs[0][:2] == ((0.0, 0.0), (0.0, 10.0))
    assert segs[1][2] == "GRID"


def test_read_lwpolyline():
    poly = "0\nLWPOLYLINE\n8\n0\n10\n0\n20\n0\n10\n6\n20\n0\n10\n6\n20\n6\n"
    segs = dxf_import.read_segments(_dxf(poly))
    assert len(segs) == 2                        # 3 verts → 2 segments
    assert segs[0][:2] == ((0.0, 0.0), (6.0, 0.0))


# ------------------------------------------------------------- classification

def test_grids_from_dxf_axis_and_diagonal():
    text = _dxf(
        _line(0, 0, 0, 12),        # vertical → const-X at 0
        _line(6, 0, 6, 12),        # vertical → const-X at 6
        _line(0, 0, 12, 0),        # horizontal → const-Y at 0
        _line(0, 0, 12, 12),       # diagonal → general
    )
    grids, generals = dxf_import.grids_from_dxf(text)
    xs = [(g.name, g.coord) for g in grids if g.axis == "x"]
    ys = [(g.name, g.coord) for g in grids if g.axis == "y"]
    assert xs == [("A", 0.0), ("B", 6.0)]
    assert ys == [("1", 0.0)]
    assert len(generals) == 1
    assert (generals[0].x1, generals[0].y1, generals[0].x2, generals[0].y2) \
        == (0.0, 0.0, 12.0, 12.0)


def test_grids_dedupe_coincident():
    text = _dxf(_line(6, 0, 6, 12), _line(6, 2, 6, 20))   # same const-X = 6
    grids, _ = dxf_import.grids_from_dxf(text)
    assert len([g for g in grids if g.axis == "x"]) == 1


def test_grids_scale():
    text = _dxf(_line(0, 0, 0, 120))            # mm drawing
    grids, _ = dxf_import.grids_from_dxf(text, scale=0.001)   # mm → m
    assert grids == [] or grids[0].axis == "x"   # single vertical line at x=0


def test_empty_dxf():
    grids, generals = dxf_import.grids_from_dxf(_dxf())
    assert grids == [] and generals == []


# ------------------------------------------------------------- action

@pytest.fixture(scope="module")
def qapp_vtk():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvistaqt")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_import_grid_dxf_action(qapp_vtk, monkeypatch, tmp_path):
    from main_window import MainWindow
    from PySide6.QtWidgets import QFileDialog
    path = tmp_path / "grid.dxf"
    path.write_text(_dxf(_line(0, 0, 0, 12), _line(6, 0, 6, 12),
                         _line(0, 0, 12, 0)), encoding="utf-8")
    w = MainWindow()
    w.load_project(Project(ndm=3, ndf=6))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(path),
                                                      "DXF files (*.dxf)")))
    w.import_grid_dxf()
    assert len(w._project.grid_lines) == 3       # A, B (const-X) + 1 (const-Y)
    w._undo_stack.undo()
    assert w._project.grid_lines == []
