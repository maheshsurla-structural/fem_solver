"""Load-combination editor (plan L2) — ``desktop/combinations_dialog.py``.

Headless (offscreen) coverage of the factor-grid editor: add / delete, the
per-case factor round-trip, the folded-in ASCE 7-22 generator, and the
main-window wiring.
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

from project import (LoadCase, LoadCombination, Material, Member,  # noqa: E402
                     Node, Project, Section)


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _project():
    p = Project(ndm=2, ndf=3)
    p.nodes = [Node(1, 0, 0, supports=(1, 1, 1)), Node(2, 0, 3)]
    p.sections = [Section(id=1, name="s", A=1e-3, Iz=1e-5)]
    p.materials = [Material(1, "m", E=2e11, nu=0.3)]
    p.members = [Member(1, 1, 2, 1, 1)]
    p.load_cases = [LoadCase(id=1, name="Dead", nature="dead"),
                    LoadCase(id=2, name="Live", nature="live"),
                    LoadCase(id=3, name="Wind", nature="wind")]
    return p


def test_grid_has_one_row_per_case(qapp):
    from combinations_dialog import CombinationsDialog
    p = _project()
    dlg = CombinationsDialog(None, p)
    assert dlg.grid.rowCount() == 3
    assert len(dlg._spins) == 3


def test_factor_roundtrip_and_zero_drop(qapp):
    from combinations_dialog import CombinationsDialog
    p = _project()
    p.combinations = [LoadCombination(id=1, name="C1",
                                      factors={1: 1.2, 2: 1.6})]
    dlg = CombinationsDialog(None, p)
    dlg.listw.setCurrentRow(0)
    # seeded factors show up in the spins (case order 1,2,3)
    assert dlg._spins[0].value() == pytest.approx(1.2)
    assert dlg._spins[1].value() == pytest.approx(1.6)
    assert dlg._spins[2].value() == pytest.approx(0.0)
    # edit: add wind 0.9, drop live to 0
    dlg._spins[2].setValue(0.9)
    dlg._spins[1].setValue(0.0)
    dlg.accept()
    out = dlg.result_combos
    assert out is not None and len(out) == 1
    assert out[0].factors == {1: pytest.approx(1.2), 3: pytest.approx(0.9)}


def test_add_and_delete(qapp):
    from combinations_dialog import CombinationsDialog
    p = _project()
    dlg = CombinationsDialog(None, p)
    assert dlg.listw.count() == 0
    dlg._add()
    dlg._add()
    assert dlg.listw.count() == 2
    assert dlg._combos[0].id != dlg._combos[1].id       # unique ids
    dlg.listw.setCurrentRow(0)
    dlg._delete()
    assert dlg.listw.count() == 1


def test_name_edit_updates_list_and_persists(qapp):
    from combinations_dialog import CombinationsDialog
    p = _project()
    dlg = CombinationsDialog(None, p)
    dlg._add()
    dlg.listw.setCurrentRow(0)
    dlg.name.setText("ULS-1")
    dlg._on_name_edited("ULS-1")
    assert dlg.listw.item(0).text() == "ULS-1"
    dlg.accept()
    assert dlg.result_combos[0].name == "ULS-1"


def test_generate_asce7_folds_in(qapp):
    from combinations_dialog import CombinationsDialog
    p = _project()                                      # D + L + W natures
    dlg = CombinationsDialog(None, p)
    n_added, n_gen = dlg._append_generated()   # UI-free; avoids the modal
    assert n_gen > 0 and n_added == n_gen
    assert len(dlg._combos) > 0
    # every generated combo references only real case ids, with a factor
    ids = {c.id for c in p.load_cases}
    for combo in dlg._combos:
        assert combo.factors and set(combo.factors).issubset(ids)


def test_main_window_wires_combinations(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_project())
    assert hasattr(w, "act_editcombos")
    assert callable(w.manage_combinations)
