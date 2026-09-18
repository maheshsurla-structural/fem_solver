"""Wall-modeling W1 — draw a wall by extruding a base line upward.

Covers the pure geometry builder (``walls.build_wall_line`` +
``wall_baseline_from_nodes``) headless, then the ``WallDialog`` create contract
under a QApplication.
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

from project import Material, Node, Project, ShellSection  # noqa: E402


def _project() -> Project:
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C30", E=30e9, nu=0.2, rho=2400.0))
    p.shell_sections.append(ShellSection(id=1, name="WALL250", thickness=0.25))
    # a base line of three nodes along X at z = 0
    p.nodes.extend([
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=4.0, y=0.0, z=0.0),
        Node(id=3, x=8.0, y=0.0, z=0.0),
    ])
    return p


# ------------------------------------------------------------ build_wall_line

def test_extrudes_two_node_base_into_one_wall_panel():
    import walls
    p = _project()
    del p.nodes[2]                      # keep just nodes 1, 2
    ids = walls.build_wall_line(p, [1, 2], height=3.0, shell_section=1,
                                material=1, mesh=(4, 3), pier="P1")
    assert len(ids) == 1
    a = p.area(ids[0])
    assert a.role == "wall" and a.pier == "P1"
    assert a.mesh == (4, 3)
    # two new top nodes appended, 3 m above the base
    tops = [n for n in p.nodes if n.z == 3.0]
    assert len(tops) == 2
    # quad is CCW bottom A->B, top B'->A' (the four corners are distinct)
    assert len(set(a.nodes)) == 4
    assert a.nodes[0] == 1 and a.nodes[1] == 2   # bottom edge first


def test_multi_segment_base_shares_top_nodes():
    import walls
    p = _project()
    ids = walls.build_wall_line(p, [1, 2, 3], height=3.0, shell_section=1,
                                material=1)
    assert len(ids) == 2                # one panel per base segment
    # only three new top nodes (the shared middle top is not duplicated)
    assert sum(1 for n in p.nodes if n.z == 3.0) == 3
    a0, a1 = p.area(ids[0]), p.area(ids[1])
    # the two panels share their common vertical (top-of-node-2) edge node
    assert set(a0.nodes) & set(a1.nodes)


def test_height_can_be_negative_downstand():
    import walls
    p = _project()
    ids = walls.build_wall_line(p, [1, 2], height=-2.5, shell_section=1,
                                material=1)
    assert p.area(ids[0]).role == "wall"
    assert any(n.z == -2.5 for n in p.nodes)


def test_zero_height_and_bad_baseline_raise():
    import walls
    p = _project()
    with pytest.raises(ValueError):
        walls.build_wall_line(p, [1, 2], height=0.0, shell_section=1, material=1)
    with pytest.raises(ValueError):
        walls.build_wall_line(p, [1], height=3.0, shell_section=1, material=1)
    with pytest.raises(ValueError):
        walls.build_wall_line(p, [1, 999], height=3.0, shell_section=1,
                              material=1)


def test_baseline_ordering_two_nodes_passthrough():
    import walls
    p = _project()
    assert walls.wall_baseline_from_nodes(p, [2, 1]) == [2, 1]


def test_baseline_ordering_chains_by_proximity():
    import walls
    p = _project()
    # give them out of order; expect a left-to-right chain 1,2,3
    assert walls.wall_baseline_from_nodes(p, [3, 1, 2]) == [1, 2, 3]


def test_vertical_wall_meshes_and_builds_as_shells():
    """A wall is a vertical quad — it must mesh n1×n2 and compile through the
    existing (slab) area pipeline as shell elements."""
    import walls
    from femsolver.elements.shell import ShellMITC4
    p = _project()
    del p.nodes[2]                      # nodes 1, 2 only
    walls.build_wall_line(p, [1, 2], height=3.0, shell_section=1, material=1,
                          mesh=(4, 3))
    m = p.build_model(with_loads=False)
    shells = [e for e in m.elements.values() if isinstance(e, ShellMITC4)]
    assert len(shells) == 4 * 3         # n1 × n2 elements up the vertical panel


# ----------------------------------------------------------------- WallDialog

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_wall_dialog_params(qapp):
    from editing import WallDialog
    p = _project()
    dlg = WallDialog(None, p, [1, 2])
    dlg.height.set_si(3.5)
    dlg.n1.setValue(6)
    dlg.n2.setValue(4)
    dlg.pier.setText("  W1 ")
    params = dlg.params()
    assert params["base_nodes"] == [1, 2]
    assert params["height"] == pytest.approx(3.5)
    assert params["mesh"] == (6, 4)
    assert params["pier"] == "W1"
    # the params feed the builder cleanly
    import walls
    ids = walls.build_wall_line(p, **params)
    assert p.area(ids[0]).pier == "W1"


def test_wall_dialog_blank_pier_is_none(qapp):
    from editing import WallDialog
    dlg = WallDialog(None, _project(), [1, 2])
    dlg.pier.setText("   ")
    assert dlg.params()["pier"] is None


# ------------------------------------------------ MainWindow action wiring

@pytest.fixture(scope="module")
def qapp_vtk():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvistaqt")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_draw_wall_action_creates_and_selects_wall(qapp_vtk, monkeypatch):
    import editing
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_project())
    # select the base line (two nodes)
    w._set_selection([("node", 1), ("node", 2)])
    # patch the modal dialog to return params headlessly (QMessageBox gotcha)
    monkeypatch.setattr(
        editing.WallDialog, "get",
        classmethod(lambda cls, *a, **k: dict(
            base_nodes=[1, 2], height=3.0, shell_section=1, material=1,
            mesh=(2, 2), pier="P1")))
    n_before = len(w._project.areas)
    w.draw_wall()
    assert len(w._project.areas) == n_before + 1
    wall = w._project.walls()[0]
    assert wall.role == "wall" and wall.pier == "P1"
    # the new wall area is selected
    assert ("area", wall.id) in w._selection
    # and it survives undo/redo
    w._undo_stack.undo()
    assert len(w._project.areas) == n_before
    w._undo_stack.redo()
    assert len(w._project.areas) == n_before + 1


def test_draw_wall_needs_two_base_nodes(qapp_vtk, monkeypatch):
    from main_window import MainWindow
    import main_window as mw
    w = MainWindow()
    w.load_project(_project())
    w._set_selection([("node", 1)])            # only one node
    seen = {}
    monkeypatch.setattr(mw.QMessageBox, "information",
                        lambda *a, **k: seen.setdefault("info", a))
    n_before = len(w._project.areas)
    w.draw_wall()
    assert len(w._project.areas) == n_before   # nothing created
    assert "info" in seen                       # user was told to select a base
