"""Wall-modeling W6 — coupling beams between shear walls (desktop shell model).

Covers coupling.add_coupling_beam (inner-edge pick, level snap, beam creation,
tie-in to the shell by coincident-node merge at build) and the dialog + action.
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

import coupling  # noqa: E402
import walls  # noqa: E402
from project import Material, Node, Project, Section, ShellSection  # noqa: E402


def _two_walls(gap=2.0, H=4.0, mesh=(2, 4)):
    """Two coplanar wall panels in the X-Z plane separated by a gap, pier P1/P2.
    Wall A: x in [0,3]; wall B: x in [3+gap, 6+gap]."""
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C", E=30e9, nu=0.2, rho=2400.0))
    p.sections.append(Section(id=1, name="CB", A=0.09, Iz=6.75e-4, Iy=6.75e-4,
                              J=1e-3))
    p.shell_sections.append(ShellSection(id=1, name="W", thickness=0.25))
    p.nodes.extend([Node(id=1, x=0.0, y=0.0, z=0.0),
                    Node(id=2, x=3.0, y=0.0, z=0.0),
                    Node(id=3, x=3.0 + gap, y=0.0, z=0.0),
                    Node(id=4, x=6.0 + gap, y=0.0, z=0.0)])
    a = walls.build_wall_line(p, [1, 2], height=H, shell_section=1, material=1,
                              mesh=mesh, pier="P1")[0]
    b = walls.build_wall_line(p, [3, 4], height=H, shell_section=1, material=1,
                              mesh=mesh, pier="P2")[0]
    return p, a, b


# ------------------------------------------------------------- geometry

def test_coupling_beam_connects_inner_faces():
    p, a, b = _two_walls(gap=2.0, H=4.0, mesh=(2, 4))
    na, nb, mid = coupling.add_coupling_beam(p, a, b, elev=4.0, section=1,
                                             material=1)
    ax = next(n for n in p.nodes if n.id == na)
    bx = next(n for n in p.nodes if n.id == nb)
    # inner faces: wall A right edge x=3, wall B left edge x=5 (=3+gap)
    assert ax.x == pytest.approx(3.0)
    assert bx.x == pytest.approx(5.0)
    assert ax.z == pytest.approx(4.0) and bx.z == pytest.approx(4.0)
    # a beam member now spans them
    m = next(mm for mm in p.members if mm.id == mid)
    assert {m.n1, m.n2} == {na, nb}


def test_elevation_snaps_to_mesh_row():
    p, a, b = _two_walls(H=4.0, mesh=(2, 4))       # rows at z = 0,1,2,3,4
    na, _nb, _mid = coupling.add_coupling_beam(p, a, b, elev=2.4, section=1,
                                               material=1)
    za = next(n for n in p.nodes if n.id == na).z
    assert za == pytest.approx(2.0)                # 2.4 → nearest row 2.0


def test_end_reuses_existing_coincident_node():
    p, a, b = _two_walls(H=4.0, mesh=(2, 4))
    n_before = len(p.nodes)
    # top-of-wall corners already exist (created by build_wall_line) at z=4
    coupling.add_coupling_beam(p, a, b, elev=4.0, section=1, material=1)
    # both ends land on existing top corners → no new nodes, just the member
    assert len(p.nodes) == n_before


def test_rejects_non_quad_or_missing():
    p, a, b = _two_walls()
    with pytest.raises(ValueError):
        coupling.add_coupling_beam(p, a, 999, elev=4.0, section=1, material=1)


# ------------------------------------------------------- tie-in at build

def test_coupling_beam_ties_into_shell_on_build():
    p, a, b = _two_walls(H=4.0, mesh=(2, 4))
    coupling.add_coupling_beam(p, a, b, elev=2.0, section=1, material=1)
    m = p.build_model(with_loads=False)
    from femsolver.elements.beam import BeamColumn3D
    beams = [e for e in m.elements.values() if isinstance(e, BeamColumn3D)]
    assert len(beams) == 1
    beam = beams[0]
    # each beam end shares a node with a shell element (merged), i.e. the end
    # nodes are among the shell mesh nodes
    shell_nodes = set()
    for e in m.elements.values():
        if len(getattr(e, "node_tags", ())) in (3, 4):
            shell_nodes.update(e.node_tags)
    assert set(beam.node_tags) <= shell_nodes


# ------------------------------------------------------------- dialog / action

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_coupling_dialog_params(qapp):
    from coupling_beam_dialog import CouplingBeamDialog
    p, a, b = _two_walls()
    dlg = CouplingBeamDialog(None, p, seed=[a, b])
    dlg.elev.set_si(3.0)
    prm = dlg.params()
    assert prm["area_a_id"] == a and prm["area_b_id"] == b
    assert prm["elev"] == pytest.approx(3.0)
    assert prm["section"] == 1 and prm["material"] == 1


@pytest.fixture(scope="module")
def qapp_vtk():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvistaqt")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_coupling_action_adds_member(qapp_vtk, monkeypatch):
    from main_window import MainWindow
    import coupling_beam_dialog as cbd
    p, a, b = _two_walls()
    w = MainWindow()
    w.load_project(p)
    monkeypatch.setattr(cbd.CouplingBeamDialog, "get",
                        classmethod(lambda cls, *ar, **k: dict(
                            area_a_id=a, area_b_id=b, elev=4.0, section=1,
                            material=1)))
    n_before = len(w._project.members)
    w.add_coupling_beam()
    assert len(w._project.members) == n_before + 1
    w._undo_stack.undo()
    assert len(w._project.members) == n_before


def test_coupling_action_guarded_one_wall(qapp_vtk, monkeypatch):
    from main_window import MainWindow
    import main_window as mw
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C", E=30e9, nu=0.2))
    p.sections.append(Section(id=1, name="B", A=0.09, Iz=6.75e-4))
    p.shell_sections.append(ShellSection(id=1, name="W", thickness=0.25))
    p.nodes.extend([Node(id=1, x=0.0, y=0.0, z=0.0), Node(id=2, x=3.0, y=0.0, z=0.0)])
    walls.build_wall_line(p, [1, 2], height=4.0, shell_section=1, material=1,
                          mesh=(2, 2), pier="P1")
    w = MainWindow()
    w.load_project(p)
    seen = {}
    monkeypatch.setattr(mw.QMessageBox, "information",
                        lambda *a, **k: seen.setdefault("info", a))
    w.add_coupling_beam()
    assert "info" in seen
