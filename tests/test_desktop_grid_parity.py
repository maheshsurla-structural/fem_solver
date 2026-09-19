"""Wall-modeling W8b — grid-system parity (visibility, bubbles, general grids)."""
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

import model_geometry as mg  # noqa: E402
from project import (GeneralGrid, GridLine, Material, Node, Project,  # noqa: E402
                     Section)


def _proj_model():
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="S", E=200e9, nu=0.3))
    p.sections.append(Section(id=1, name="W", A=1e-2, Iz=1e-4, Iy=1e-4, J=1e-5))
    p.nodes.extend([Node(id=1, x=0.0, y=0.0, z=0.0),
                    Node(id=2, x=6.0, y=0.0, z=0.0),
                    Node(id=3, x=6.0, y=6.0, z=0.0)])
    return p


# ------------------------------------------------------------- data model

def test_gridline_visible_bubble_round_trip():
    p = _proj_model()
    p.grid_lines.append(GridLine(id=1, name="A", axis="x", coord=0.0,
                                 visible=False, bubble="start"))
    p.general_grids.append(GeneralGrid(id=1, name="D1", x1=0, y1=0, x2=6, y2=6))
    q = Project.from_json(p.to_json())
    g = q.grid_lines[0]
    assert g.visible is False and g.bubble == "start"
    assert q.general_grids[0].x2 == 6 and q.general_grids[0].name == "D1"


def test_bubble_defaults_and_validation():
    g = GridLine(id=1, name="A", bubble="bogus")
    assert g.bubble == "end" and g.visible is True


# ------------------------------------------------------------- rendering

def test_mesh_skips_invisible_and_draws_general():
    p = _proj_model()
    m = p.build_model(with_loads=False)
    p.grid_lines.append(GridLine(id=1, name="A", axis="x", coord=0.0))
    p.grid_lines.append(GridLine(id=2, name="B", axis="x", coord=6.0,
                                 visible=False))
    p.general_grids.append(GeneralGrid(id=1, name="D1", x1=0, y1=0, x2=6, y2=6))
    poly = mg.grid_story_mesh(p, m)
    # 1 visible grid line + 1 general grid = 2 segments (B is hidden)
    assert poly.n_lines == 2


def test_general_grid_alone_draws():
    p = _proj_model()
    m = p.build_model(with_loads=False)
    p.general_grids.append(GeneralGrid(id=1, name="D1", x1=0, y1=0, x2=6, y2=6))
    assert mg.grid_story_mesh(p, m).n_lines == 1


def test_bubble_labels():
    p = _proj_model()
    m = p.build_model(with_loads=False)
    p.grid_lines.append(GridLine(id=1, name="A", axis="x", coord=0.0,
                                 bubble="end"))
    p.grid_lines.append(GridLine(id=2, name="B", axis="y", coord=3.0,
                                 bubble="none"))          # no bubble
    p.general_grids.append(GeneralGrid(id=1, name="D1", x1=0, y1=0, x2=6, y2=6))
    pts, labels = mg.grid_bubble_labels(p, m)
    assert set(labels) == {"A", "D1"}                     # B suppressed
    assert len(pts) == len(labels) == 2


# ------------------------------------------------------------- dialog

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_dialog_round_trips_grid_flags_and_generals(qapp):
    from PySide6.QtWidgets import QTableWidgetItem
    from story_grid_dialog import StoryGridDialog
    p = _proj_model()
    p.grid_lines.append(GridLine(id=1, name="A", axis="x", coord=0.0,
                                 visible=False, bubble="start"))
    p.general_grids.append(GeneralGrid(id=1, name="D1", x1=0, y1=0, x2=6, y2=6))
    dlg = StoryGridDialog(None, p)
    assert dlg.gr_tbl.rowCount() == 1
    assert dlg.gg_tbl.rowCount() == 1
    dlg._accept()
    _st, grids, generals, _sys = dlg.result
    assert grids[0].visible is False and grids[0].bubble == "start"
    assert generals[0].name == "D1" and generals[0].x2 == pytest.approx(6.0)


def test_dialog_add_general_grid(qapp):
    from story_grid_dialog import StoryGridDialog
    dlg = StoryGridDialog(None, _proj_model())
    dlg._add_general()
    assert dlg.gg_tbl.rowCount() == 1
    dlg._accept()
    _st, _g, generals, _sys = dlg.result
    assert len(generals) == 1


# ------------------------------------------ W8b-2: grid systems (origin + rotation)

def test_grid_system_round_trip():
    from project import GridSystem
    p = _proj_model()
    p.grid_systems.append(GridSystem(id=1, name="G2", origin_x=2.0,
                                     origin_y=3.0, rotation=30.0))
    p.grid_lines.append(GridLine(id=1, name="A", axis="x", coord=1.0, system=1))
    q = Project.from_json(p.to_json())
    assert q.grid_system(1).rotation == 30.0
    assert q.grid_lines[0].system == 1


def test_grid_line_endpoints_origin_offset():
    from project import GridSystem
    p = _proj_model()
    m = p.build_model(with_loads=False)
    lo, hi = mg._model_bbox(m)
    p.grid_systems.append(GridSystem(id=1, name="G", origin_x=5.0, origin_y=0.0))
    g = GridLine(id=1, name="A", axis="x", coord=0.0, system=1)
    (x1, _y1), (x2, _y2) = mg.grid_line_endpoints(p, g, lo, hi)
    assert x1 == pytest.approx(5.0) and x2 == pytest.approx(5.0)  # shifted to x=5


def test_grid_line_endpoints_rotation():
    import math
    from project import GridSystem
    p = _proj_model()
    m = p.build_model(with_loads=False)
    lo, hi = mg._model_bbox(m)
    p.grid_systems.append(GridSystem(id=1, name="G", rotation=90.0))
    g = GridLine(id=1, name="A", axis="x", coord=2.0, system=1)
    # const-X at local x=2 rotated 90° → a horizontal line at global y=2
    (x1, y1), (x2, y2) = mg.grid_line_endpoints(p, g, lo, hi)
    assert y1 == pytest.approx(2.0) and y2 == pytest.approx(2.0)
    assert abs(x1 - x2) > 1.0                       # runs in X now


def test_snap_targets_origin_and_skip_rotated():
    from project import GridSystem
    p = _proj_model()
    p.grid_systems.extend([GridSystem(id=1, name="G", origin_x=5.0),
                           GridSystem(id=2, name="R", rotation=45.0)])
    p.grid_lines.extend([
        GridLine(id=1, name="A", axis="x", coord=1.0, system=1),   # → x=6
        GridLine(id=2, name="B", axis="x", coord=2.0, system=2),   # rotated skip
        GridLine(id=3, name="C", axis="x", coord=3.0)])            # global → x=3
    xs, _ys, _zs = mg.snap_targets(p)
    assert sorted(xs) == [3.0, 6.0]                 # rotated line B excluded


def test_dialog_grid_system_round_trip(qapp):
    from project import GridSystem
    from story_grid_dialog import StoryGridDialog
    p = _proj_model()
    p.grid_systems.append(GridSystem(id=1, name="G2", origin_x=2.0,
                                     rotation=15.0))
    p.grid_lines.append(GridLine(id=1, name="A", axis="x", coord=0.0, system=1))
    dlg = StoryGridDialog(None, p)
    assert dlg.gs_tbl.rowCount() == 1
    dlg._accept()
    _st, grids, _gg, systems = dlg.result
    assert systems[0].name == "G2" and systems[0].rotation == pytest.approx(15.0)
    assert grids[0].system == 1
