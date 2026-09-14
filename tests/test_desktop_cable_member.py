"""Pin-ended cable / truss member type (bridge GUI track A, item A1).

The desktop ``Member.kind == "cable"`` compiles to a pin-ended truss element
(axial only) instead of a beam-column — in build_model, the buckling builder,
the member editor, and serialization.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "desktop"))

from femsolver import (BeamColumn2D, BeamColumn3D, Truss2D,  # noqa: E402
                       Truss3D)
from project import (Load, LoadCase, Material, Member,  # noqa: E402
                     Node, Project, Section)


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _stayed(kind_cable="cable"):
    p = Project(ndm=2, ndf=3)
    for i, x in enumerate([0, 5, 10, 15, 20, 25, 30]):
        p.nodes.append(Node(id=i + 1, x=float(x), y=0.0))
    p.nodes.append(Node(id=8, x=15.0, y=12.0))
    p.nodes.append(Node(id=9, x=15.0, y=0.0))
    p.nodes[0].supports = (1, 1, 0)
    p.nodes[6].supports = (1, 1, 0)
    next(n for n in p.nodes if n.id == 9).supports = (1, 1, 1)
    p.sections = [Section(id=1, name="deck", A=0.5, Iz=0.05),
                  Section(id=2, name="strand", A=0.01, Iz=1e-8)]
    p.materials = [Material(1, "steel", E=2e11, nu=0.3, rho=7850.0)]
    mid = 1
    for i in range(6):
        p.members.append(Member(mid, i + 1, i + 2, 1, 1)); mid += 1
    p.members.append(Member(7, 9, 8, 1, 1))
    p.members.append(Member(8, 8, 3, 2, 1, kind=kind_cable))
    p.members.append(Member(9, 8, 5, 2, 1, kind=kind_cable))
    p.load_cases = [LoadCase(1, "Dead", "dead")]
    for nd in (2, 3, 4, 5, 6):
        p.loads.append(Load(node=nd, values=(0, -50e3, 0), case=1))
    return p


# --------------------------------------------------------- build_model
def test_cable_member_builds_truss():
    m = _stayed().build_model(with_loads=True)
    assert isinstance(m.element(8), Truss2D)            # cable → truss
    assert isinstance(m.element(9), Truss2D)
    assert isinstance(m.element(1), BeamColumn2D)       # deck → beam


def test_cable_member_builds_truss3d():
    p = Project(ndm=3, ndf=6)
    p.nodes = [Node(1, 0, 0, z=0, supports=(1, 1, 1, 1, 1, 1)),
               Node(2, 5, 0, z=0), Node(3, 2.5, 0, z=4)]
    p.sections = [Section(id=1, name="s", A=0.5, Iz=0.05, Iy=0.05, J=1e-3)]
    p.materials = [Material(1, "steel", E=2e11, nu=0.3, rho=7850.0)]
    p.members = [Member(1, 1, 2, 1, 1),
                 Member(2, 3, 2, 1, 1, kind="cable")]
    m = p.build_model(with_loads=False)
    assert isinstance(m.element(1), BeamColumn3D)
    assert isinstance(m.element(2), Truss3D)


def test_buckling_builder_cable_is_single_truss():
    model, subs = _stayed().build_buckling_model(subdivisions=6)
    assert isinstance(model.element(subs[8][0]), Truss2D)
    assert len(subs[8]) == 1                            # cables not sub-divided


# --------------------------------------------------------- serialization
def test_kind_round_trips():
    p2 = Project.from_dict(_stayed().to_dict())
    kinds = {mb.id: mb.kind for mb in p2.members}
    assert kinds[8] == "cable" and kinds[9] == "cable"
    assert kinds[1] == "beamcolumn2d"


def test_old_member_without_kind_is_beam():
    p = _stayed()
    d = p.to_dict()
    for mb in d["members"]:
        mb.pop("kind", None)                            # pre-cable project
    p2 = Project.from_dict(d)
    assert all(mb.kind == "beamcolumn2d" for mb in p2.members)


# --------------------------------------------------------- member editor
def test_member_dialog_roundtrips_kind(qapp):
    from editing import MemberDialog
    p = _stayed()
    cable = next(mb for mb in p.members if mb.id == 8)
    assert MemberDialog(None, p, cable).data().kind == "cable"
    beam = next(mb for mb in p.members if mb.id == 1)
    assert MemberDialog(None, p, beam).data().kind == "beamcolumn2d"


# ------------------------------------------------- tuning with real cables
def test_cable_tuning_with_truss_cables(qapp):
    from main_window import MainWindow
    import numpy as np
    w = MainWindow()
    w.load_project(_stayed())
    res = w.run_cable_tuning(config={"cables": [8, 9], "targets": [3, 5]})
    assert res is not None
    assert np.all(res["tensions"] > 0)                 # stays in tension
    assert np.max(np.abs(res["residual"])) < 1e-9
