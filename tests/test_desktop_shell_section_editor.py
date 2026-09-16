"""Slab-modeling S4 — the Thickness / shell-section editor + manager.

Headless (offscreen) coverage of ``shell_section_editor``: the editor's
``data()`` contract (id / name / kind / SI thickness / only-non-unit
modifiers) and the manager's add / edit / delete / in-use-guard behaviour.
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

from project import Area, Project, ShellSection  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _project():
    p = Project(ndm=3, ndf=6)
    p.shell_sections.append(ShellSection(id=1, name="SLAB200", thickness=0.20))
    return p


# ------------------------------------------------------------- editor

def test_editor_data_defaults(qapp):
    from shell_section_editor import ShellSectionDialog
    dlg = ShellSectionDialog(None, Project(ndm=3, ndf=6))
    s = dlg.data()
    assert s.id == 1                       # next free id
    assert s.kind == "shell-thin"          # first SHELL_KIND
    assert s.thickness == pytest.approx(0.20)
    assert s.modifiers == {}               # all factors unity -> nothing stored


def test_editor_round_trips_existing(qapp):
    from shell_section_editor import ShellSectionDialog
    src = ShellSection(id=3, name="WALL", thickness=0.30, kind="plate-thin",
                       modifiers={"m11": 0.5})
    dlg = ShellSectionDialog(None, _project(), src)
    s = dlg.data()
    assert s.id == 3
    assert s.name == "WALL"
    assert s.kind == "plate-thin"
    assert s.thickness == pytest.approx(0.30)
    assert s.modifiers.get("m11") == pytest.approx(0.5)


def test_editor_stores_only_non_unit_modifiers(qapp):
    from shell_section_editor import ShellSectionDialog
    dlg = ShellSectionDialog(None, _project())
    dlg._mod_spins["f11"].setValue(0.25)
    dlg._mod_spins["m22"].setValue(1.0)     # left at unity -> dropped
    s = dlg.data()
    assert s.modifiers == {"f11": 0.25}


# ------------------------------------------------------------- manager

def test_manager_lists_existing(qapp):
    from shell_section_editor import ShellSectionManagerDialog
    mgr = ShellSectionManagerDialog(None, _project())
    assert mgr.table.rowCount() == 1
    assert mgr.table.item(0, 1).text() == "SLAB200"


def test_manager_delete_in_use_blocked(qapp, monkeypatch):
    import shell_section_editor as sse
    # the in-use guard pops a modal warning; stub it so the headless test
    # does not block waiting for a click.
    monkeypatch.setattr(sse.QMessageBox, "warning",
                        lambda *a, **k: sse.QMessageBox.StandardButton.Ok)
    p = _project()
    p.areas.append(Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=1))
    mgr = sse.ShellSectionManagerDialog(None, p)
    mgr.table.selectRow(0)
    mgr._delete()
    # still present — deletion refused because area 1 references it
    assert len(mgr.result_sections()) == 1


def test_manager_delete_unused_ok(qapp):
    from shell_section_editor import ShellSectionManagerDialog
    mgr = ShellSectionManagerDialog(None, _project())
    mgr.table.selectRow(0)
    mgr._delete()
    assert mgr.result_sections() == []
