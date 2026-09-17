"""Slab-modeling S8 — rigid floor diaphragms.

Covers the Diaphragm data model + round-trip, build_model compiling a
RigidDiaphragm MP-constraint (auto master node at the centroid, out-of-plane
DOFs pinned), the 3-D gate, an end-to-end 4-column rigid-floor solve where the
tied joints share the master's in-plane translation, and the dialog contract.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

from femsolver import RigidDiaphragm  # noqa: E402
from femsolver.analysis.linear_static import LinearStaticAnalysis  # noqa: E402

from project import (DIAPHRAGM_MASTER_NODE_BASE, Diaphragm,  # noqa: E402
                     Load, Material, Node, Project, Section)


def _frame(L=4.0, H=3.0):
    """4 columns from a fixed base (z=0) to a free floor (z=H)."""
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="S", E=2.0e11, nu=0.3))
    p.sections.append(Section(id=1, name="C", A=0.05, Iz=4.17e-4, Iy=4.17e-4,
                              J=8.33e-4))
    base = [(0, 0), (L, 0), (L, L), (0, L)]
    for i, (x, y) in enumerate(base):
        p.nodes.append(Node(id=i + 1, x=float(x), y=float(y), z=0.0,
                            supports=(1, 1, 1, 1, 1, 1)))
    for i, (x, y) in enumerate(base):
        p.nodes.append(Node(id=i + 5, x=float(x), y=float(y), z=H))
    for i in range(4):
        from project import Member
        p.members.append(Member(id=i + 1, n1=i + 1, n2=i + 5, section=1,
                                material=1, kind="beamcolumn2d"))
    return p


# ------------------------------------------------------------ data model

def test_diaphragm_round_trip():
    p = _frame()
    p.diaphragms.append(Diaphragm(id=1, name="L1", nodes=[5, 6, 7, 8],
                                  perp_dir=2))
    q = Project.from_json(p.to_json())
    assert len(q.diaphragms) == 1
    d = q.diaphragm(1)
    assert d.name == "L1"
    assert d.nodes == [5, 6, 7, 8]
    assert d.perp_dir == 2
    assert d.master is None


def test_backward_compat_no_diaphragms_key():
    d = {"schema": "femsolver-project/1", "ndm": 3, "ndf": 6}
    assert Project.from_dict(d).diaphragms == []


def test_next_diaphragm_id():
    p = _frame()
    assert p.next_diaphragm_id() == 1
    p.diaphragms.append(Diaphragm(id=3, name="x", nodes=[5, 6]))
    assert p.next_diaphragm_id() == 4


# ------------------------------------------------------- build_model

def test_build_adds_rigid_diaphragm_and_master():
    p = _frame()
    p.diaphragms.append(Diaphragm(id=1, name="L1", nodes=[5, 6, 7, 8]))
    m = p.build_model(with_loads=False)
    cons = [c for c in m.mp_constraints if isinstance(c, RigidDiaphragm)]
    assert len(cons) == 1
    master = DIAPHRAGM_MASTER_NODE_BASE + 1
    assert master in m.nodes
    # centroid of the 4 top corners of a 4×4 bay at z=3
    assert np.allclose(m.node(master).coords, [2.0, 2.0, 3.0])
    # out-of-plane DOFs (Uz, θx, θy) pinned; in-plane (Ux, Uy, θz) free
    assert list(m.node(master).fixity) == [0, 0, 1, 1, 1, 0]


def test_two_d_project_skips_diaphragms():
    p = Project(ndm=2, ndf=3)
    p.nodes.extend([Node(id=1, x=0.0, y=0.0), Node(id=2, x=1.0, y=0.0)])
    p.diaphragms.append(Diaphragm(id=1, name="x", nodes=[1, 2]))
    m = p.build_model(with_loads=False)
    assert not getattr(m, "mp_constraints", [])


def test_fewer_than_two_slaves_skipped():
    p = _frame()
    p.diaphragms.append(Diaphragm(id=1, name="x", nodes=[5]))
    m = p.build_model(with_loads=False)
    assert [c for c in m.mp_constraints if isinstance(c, RigidDiaphragm)] == []


# ------------------------------------------------------- end-to-end solve

def test_rigid_floor_ties_joint_translations():
    p = _frame()
    p.diaphragms.append(Diaphragm(id=1, name="L1", nodes=[5, 6, 7, 8]))
    # push the floor in +X with a torsion-free resultant (nodes 5 & 8 are
    # symmetric about the floor's y-centroid) so the rigid floor translates
    # without rotating — every tied joint should share the same Ux.
    p.loads.append(Load(node=5, values=(2.5e4, 0, 0, 0, 0, 0), case=1))
    p.loads.append(Load(node=8, values=(2.5e4, 0, 0, 0, 0, 0), case=1))
    m = p.build_model(with_loads=True)
    LinearStaticAnalysis(m).run()
    ux = [m.node(t).disp[0] for t in (5, 6, 7, 8)]
    assert ux[0] > 0.0                          # it moved
    assert np.allclose(ux, ux[0], rtol=1e-6)    # all four share the translation


def test_rigid_floor_corner_load_moves_as_rigid_body():
    """An eccentric (corner) push twists the floor: joints at the same y still
    share Ux (tied), while different-y joints differ by the rigid rotation."""
    p = _frame()
    p.diaphragms.append(Diaphragm(id=1, name="L1", nodes=[5, 6, 7, 8]))
    p.loads.append(Load(node=5, values=(5.0e4, 0, 0, 0, 0, 0), case=1))
    m = p.build_model(with_loads=True)
    LinearStaticAnalysis(m).run()
    ux = {t: m.node(t).disp[0] for t in (5, 6, 7, 8)}
    assert ux[5] == pytest.approx(ux[6], rel=1e-6)   # both y=0 → tied
    assert ux[7] == pytest.approx(ux[8], rel=1e-6)   # both y=L → tied
    assert abs(ux[5] - ux[7]) > 1e-9                 # rotation → y-levels differ


# ------------------------------------------------------------ dialog

def test_diaphragm_dialog_parses_nodes():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from editing import DiaphragmDialog
    dlg = DiaphragmDialog(None, _frame(), seed_nodes=[5, 6, 7, 8])
    d = dlg.data()
    assert d.nodes == [5, 6, 7, 8]
    assert d.perp_dir == 2
    # free-text parsing tolerates spaces / dupes
    dlg.nodes.setText("5, 6 6, 9")
    assert dlg._node_list() == [5, 6, 9]
