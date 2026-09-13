"""Loads & Analysis redesign — offscreen construction sweep (plan Q1).

A single place that constructs every Loads/Analysis dialog the L/A phases built
or redesigned, under ``QT_QPA_PLATFORM=offscreen``, as one regression guard: if
any dialog's ``__init__`` breaks, this catches it and names the surface. Each
dialog's behaviour is covered in depth by its own ``test_desktop_*.py``; the L1
scaffold (``test_desktop_analysis_ui.py``) and the S1/S2 menu + toolbar shell
(``test_desktop_menus.py`` / ``test_desktop_toolbars.py``) have their own tests.
"""
from __future__ import annotations

import dataclasses
import importlib
import math
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

from project import (LoadCase, LoadCombination, Material, Member,  # noqa: E402
                     NonlinearCase, Node, Project, Section)


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _project():
    """A representative fiber-column model — load cases, a combination and a
    nonlinear case — rich enough to construct every redesigned surface."""
    import section_gui_core as core
    D, L = 0.6, 3.0
    gsd = dataclasses.asdict(core.Spec(kind="Circular", D=D, fc=35e6, fy=500e6))
    p = Project()
    p.nodes = [Node(1, 0.0, 0.0, supports=(1, 1, 1)), Node(2, L, 0.0)]
    p.sections = [Section(id=1, name="col", A=math.pi * D * D / 4.0,
                          Iz=math.pi * D ** 4 / 64.0, gsd_spec=gsd)]
    p.materials = [Material(1, "conc", E=30e9, nu=0.2)]
    p.members = [Member(1, 1, 2, 1, 1)]
    p.load_cases = [LoadCase(1, "Dead", "dead"), LoadCase(2, "Live", "live")]
    p.combinations = [LoadCombination(id=1, name="1.4D", factors={1: 1.4})]
    p.nonlinear_cases = [NonlinearCase(id=1, name="Pushover-X", control_node=2,
                                       control_dof=1, target=0.05)]
    return p


# label -> (module, class). Constructed as ``cls(None, project)`` — the shared
# signature every redesigned dialog keeps.
_SURFACES = {
    "L2 combinations": ("combinations_dialog", "CombinationsDialog"),
    "L3 load cases": ("editing", "LoadCaseDialog"),
    "L4 nodal load": ("editing", "LoadDialog"),
    "L5 member load": ("member_load_dialog", "MemberLoadDialog"),
    "L6 load gen": ("editing", "LoadGenDialog"),
    "A1 analysis cases": ("analysis_cases_dialog", "AnalysisCasesDialog"),
    "A2 nonlinear case": ("nonlinear_cases", "NonlinearCaseDialog"),
    "A3 pushover": ("pushover_dialog", "PushoverDialog"),
    "A3 time history": ("timehistory_dialog", "TimeHistoryDialog"),
    "A4 run analysis": ("run_analysis_dialog", "RunAnalysisDialog"),
}


@pytest.mark.parametrize("label", list(_SURFACES))
def test_surface_constructs_offscreen(qapp, label):
    modname, clsname = _SURFACES[label]
    cls = getattr(importlib.import_module(modname), clsname)
    dlg = cls(None, _project())
    assert dlg is not None
    assert dlg.windowTitle()            # every dialog names itself
    dlg.deleteLater()


def test_sweep_covers_every_LA_dialog():
    # Guards the inventory itself: one row per L/A dialog phase that ships a
    # dialog (L1 = scaffold, A5 = a fold, S/Q = shell/quality — no new dialog).
    phases = {label.split()[0] for label in _SURFACES}
    assert phases == {"L2", "L3", "L4", "L5", "L6",
                      "A1", "A2", "A3", "A4"}
