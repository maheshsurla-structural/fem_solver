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
    _st, grids, generals = dlg.result
    assert grids[0].visible is False and grids[0].bubble == "start"
    assert generals[0].name == "D1" and generals[0].x2 == pytest.approx(6.0)


def test_dialog_add_general_grid(qapp):
    from story_grid_dialog import StoryGridDialog
    dlg = StoryGridDialog(None, _proj_model())
    dlg._add_general()
    assert dlg.gg_tbl.rowCount() == 1
    dlg._accept()
    _st, _g, generals = dlg.result
    assert len(generals) == 1
