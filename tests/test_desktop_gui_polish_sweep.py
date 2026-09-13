"""Charter sweep (plan gui-polish **G1**) — every surface the GUI-polish stream
touched must build *and* theme in all four light/dark × comfortable/compact
combinations (charter §A2: "if a surface can't be shown in dark mode, it isn't
done"). Mirrors the loads-ux Q1 sweep; a companion inventory guard keeps the
surface list from silently drifting.

Headless (offscreen). Per-surface behaviour still lives in each dialog's own
``test_desktop_*.py``; this is the cross-cutting theme/density guarantee.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

COMBOS = [("light", "comfortable"), ("light", "compact"),
          ("dark", "comfortable"), ("dark", "compact")]

# every dialog the D-phase rebuilt/polished, by name → constructor
SURFACES = ["MaterialDialog", "MaterialManagerDialog", "HingeDialog",
            "HingeManagerDialog", "HingeAssignmentDialog", "ModelChecksDialog",
            "RunHistoryDialog"]


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvistaqt")
    pytest.importorskip("matplotlib")
    from PySide6.QtCore import QSettings
    QSettings("MidasStructural", "Desktop").setValue("theme", "light")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _reset():
    import style
    yield
    style.set_theme("light")
    style.set_density("comfortable")


def _project():
    from project import Hinge, Material, Member, Node, Project, Section
    p = Project()
    p.materials = [Material(id=1, name="A992", E=200e9, nu=0.3)]
    p.sections = [Section(id=1, name="W12x65", A=0.012, Iz=2.2e-4,
                          shape="W12x65")]
    p.nodes = [Node(1, 0.0, 0.0, supports=(1, 1, 1)), Node(2, 4.0, 0.0)]
    p.members = [Member(id=1, n1=1, n2=2, section=1, material=1)]
    p.hinges = [Hinge(id=1, name="H1", lp=0.1)]
    return p


def _constructors(p):
    import model_checks as MC
    from hinge_editor import (HingeAssignmentDialog, HingeDialog,
                              HingeManagerDialog)
    from material_editor import MaterialDialog, MaterialManagerDialog
    from model_checks_dialog import ModelChecksDialog
    from run_history_dialog import RunHistoryDialog
    return {
        "MaterialDialog": lambda: MaterialDialog(None, p),
        "MaterialManagerDialog": lambda: MaterialManagerDialog(None, p),
        "HingeDialog": lambda: HingeDialog(None, p),
        "HingeManagerDialog": lambda: HingeManagerDialog(None, p),
        "HingeAssignmentDialog": lambda: HingeAssignmentDialog(None, p),
        "ModelChecksDialog": lambda: ModelChecksDialog(None, MC.check_project(p)),
        "RunHistoryDialog": lambda: RunHistoryDialog(None, p),
    }


def test_surface_inventory_is_stable(qapp):
    assert set(_constructors(_project())) == set(SURFACES)


@pytest.mark.parametrize("theme,density", COMBOS)
def test_dialogs_build_in_every_theme_density(qapp, theme, density):
    import style
    style.set_theme(theme)
    style.set_density(density)
    ctors = _constructors(_project())
    for name in SURFACES:
        w = ctors[name]()
        assert w is not None, f"{name} failed in {theme}/{density}"
        w.deleteLater()


@pytest.mark.parametrize("theme,density", COMBOS)
def test_shell_and_viewport_theme_in_every_combo(qapp, theme, density):
    import style
    from PySide6.QtCore import QSettings

    from demo_model import demo_project
    from main_window import MainWindow
    QSettings("MidasStructural", "Desktop").setValue("theme", theme)
    style.set_theme(theme)
    style.set_density(density)
    w = MainWindow()
    w.load_project(demo_project())
    w._apply_theme_density()               # QSS cascade + icons + viewport repaint
    assert w.view._model is not None
    assert w.view._hint.isHidden()         # a model with nodes → empty-state gone
    QSettings("MidasStructural", "Desktop").setValue("theme", "light")
    w.deleteLater()
