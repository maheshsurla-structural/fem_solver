"""3-D nonlinear (P-M2-M3) in the app — plan §16 C2.

The desktop nonlinear-run subsystem now compiles 3-D (ndm=3) GSD-column models
into FiberSection3D + displacement-based 3-D fiber elements and runs a
displacement-controlled pushover in any of the 6 DOFs, with 3-D-aware capture
(shape / peak strain / ASCE 41 state). Headless.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

import nonlinear as NL                                 # noqa: E402
from project import Material, Member, Node, Project, Section  # noqa: E402


def _col_project(ndm=3, *, D=0.6, L=3.0):
    import section_gui_core as core
    spec = core.Spec(kind="Circular", D=D, fc=35e6, fy=500e6)
    gsd = dataclasses.asdict(spec)
    A = math.pi * D * D / 4.0
    Iz = math.pi * D**4 / 64.0
    p = Project()
    p.ndm, p.ndf = ndm, (6 if ndm == 3 else 3)
    if ndm == 3:
        p.nodes = [Node(1, 0, 0, 0, supports=(1, 1, 1, 1, 1, 1)),
                   Node(2, L, 0.0, 0.0)]
    else:
        p.nodes = [Node(1, 0, 0, supports=(1, 1, 1)), Node(2, L, 0.0)]
    p.sections = [Section(id=1, name="col", A=A, Iz=Iz, gsd_spec=gsd)]
    p.materials = [Material(1, "conc", E=30e9, nu=0.2)]
    p.members = [Member(1, 1, 2, 1, 1)]
    return p


def test_build_3d_uses_corotational_3d_fiber_element():
    from femsolver.elements.beam_corot_3d import BeamColumn3DCorotational
    from femsolver.sections.response.fiber import FiberSection3D
    m = NL.build_nonlinear_model(_col_project(3))
    assert m.ndm == 3 and m.ndf == 6
    el = list(m.elements.values())[0]
    assert isinstance(el, BeamColumn3DCorotational)
    assert isinstance(el.sections[0], FiberSection3D)
    assert el.sections[0].GJ > 0


def test_pushover_3d_uniaxial_matches_2d():
    r3 = NL.run_pushover(_col_project(3), control_node=2, control_dof=1,
                         target=0.05, n_steps=20)
    r2 = NL.run_pushover(_col_project(2), control_node=2, control_dof=1,
                         target=0.05, n_steps=20)
    assert max(r3["shear"]) == pytest.approx(max(r2["shear"]), rel=0.02)


def test_pushover_3d_uz_direction_runs():
    r = NL.run_pushover(_col_project(3), control_node=2, control_dof=2,
                        target=0.04, n_steps=15)
    assert len(r["disp"]) > 5 and max(r["shear"]) > 0


def test_pushover_3d_capture_shape_and_acceptance():
    r = NL.run_pushover(_col_project(3), control_node=2, control_dof=1,
                        target=0.08, n_steps=30, capture_shape=True)
    sf = r["shape_frames"]
    assert len(sf[0][2]) == 3                          # 3-D node disp (dx,dy,dz)
    worst = max(max(fr.values()) for fr in r["accept_frames"])
    assert worst >= 1                                  # reaches at least IO
    assert "IO" in r["accept_milestones"]


# ---------------------------------------------------------------- Qt (headless)

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    pytest.importorskip("matplotlib")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_dialog_offers_six_dofs_in_3d(qapp):
    from pushover_dialog import PushoverDialog
    dlg = PushoverDialog(None, _col_project(3))
    assert dlg.dof.count() == 6                        # Ux..Rz
    assert dlg.dof.itemData(2) == 2                    # Uz present
    dlg2 = PushoverDialog(None, _col_project(2))
    assert dlg2.dof.count() == 3


def test_main_window_scrubs_3d(qapp):
    from main_window import MainWindow
    from nl_results import NonlinearResults
    p = _col_project(3)
    r = NL.run_pushover(p, control_node=2, control_dof=1, target=0.06,
                        n_steps=20, capture_shape=True)
    w = MainWindow()
    w.load_project(p)
    w.set_nl_results(NonlinearResults.from_run(r))
    w._nl_color.setCurrentText("acceptance")
    w._on_nl_step()                                    # 3-D show_nl_step path
    w._nl_color.setCurrentText("peak strain")
    w._on_nl_step()
