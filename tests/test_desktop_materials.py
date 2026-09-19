"""GUI-1 — desktop inelastic-material bridge + editor (plan §14).

Two layers:
* the engine-backed bridge `desktop.materials` (uniaxial_law / stress_strain_curve)
  — pure logic, runs anywhere femsolver imports;
* the Qt editor dialogs (`material_editor`) — headless-constructed under
  QT_QPA_PLATFORM=offscreen; skipped where PySide6/matplotlib are absent.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "desktop"))

import materials as M                       # noqa: E402  (desktop bridge)
from project import Material, Member, Node, Project, Section  # noqa: E402


# ============================================================ bridge (logic)

def test_kinds_and_defaults_consistent():
    assert set(M.KIND_DEFAULTS) == set(M.MATERIAL_KINDS)
    assert not M.is_inelastic("elastic_isotropic")
    assert M.is_inelastic("concrete_mander")


def test_uniaxial_law_builds_every_kind():
    for kind in M.MATERIAL_KINDS:
        mat = Material(id=1, name="m", E=200e9, kind=kind,
                       params=dict(M.KIND_DEFAULTS[kind]))
        assert M.uniaxial_law(mat) is not None


def test_concrete_curve_peaks_at_fc_and_no_tension():
    mat = Material(id=1, name="c", E=25e9, kind="concrete_kentpark",
                   params={"fc": 30e6, "eps_c0": 0.002, "fpcu_ratio": 0.2,
                           "eps_cu": 0.0035})
    eps, sig = M.stress_strain_curve(mat, n=150)
    assert sig.min() == pytest.approx(-30e6, rel=0.05)     # compression ~ -f'c
    assert sig.max() <= 1.0                                # no tension
    assert (eps <= 1e-12).all()                            # swept in compression


def test_steel_curve_peaks_at_fu():
    mat = Material(id=1, name="s", E=200e9, kind="reinforcing_steel",
                   params={"E": 200e9, "fy": 460e6, "fu": 650e6,
                           "eps_sh": 0.008, "eps_su": 0.09})
    eps, sig = M.stress_strain_curve(mat, n=150)
    assert sig.max() == pytest.approx(650e6, rel=0.03)
    assert sig.min() >= -1.0                               # swept in tension


def test_mander_uses_Ec_not_si_autoformula():
    # Ec explicit (SI); law must accept it and yield compression stress
    mat = Material(id=1, name="c", E=24.9e9, kind="concrete_mander",
                   params={"fc": 34.5e6, "eps_c0": 0.002219, "Ec": 24.9e9,
                           "eps_cu": 0.02})
    _eps, sig = M.stress_strain_curve(mat, n=80)
    assert sig.min() == pytest.approx(-34.5e6, rel=0.05)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        M.uniaxial_law(Material(id=1, name="x", E=1.0, kind="bogus"))


def test_build_model_tolerates_inelastic_materials():
    """The linear compile path must still work with inelastic materials
    assigned (they contribute a representative elastic modulus for now)."""
    p = Project()
    p.nodes = [Node(id=1, x=0, y=0, supports=(1, 1, 1)), Node(id=2, x=1, y=0)]
    p.materials = [Material(id=1, name="core", E=25e9, nu=0.2,
                            kind="concrete_mander",
                            params={"fc": 34.5e6, "eps_c0": 0.002219,
                                    "Ec": 25e9, "eps_cu": 0.02})]
    p.sections = [Section(id=1, name="s", A=1.0, Iz=0.08)]
    p.members = [Member(id=1, n1=1, n2=2, section=1, material=1)]
    m = p.build_model(with_loads=False)
    assert m is not None


# ============================================================ Qt editor (headless)

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    pytest.importorskip("matplotlib")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _project():
    p = Project()
    p.materials = [Material(id=1, name="A992", E=200e9, nu=0.3)]
    return p


def test_dialog_constructs_and_switches_kind(qapp):
    from material_editor import MaterialDialog
    dlg = MaterialDialog(None, _project())
    i = dlg.kind.findData("concrete_mander")
    dlg.kind.setCurrentIndex(i)               # triggers param-row rebuild + redraw
    mat = dlg.data()
    assert mat.kind == "concrete_mander"
    assert mat.params.get("fc", 0) > 0
    assert mat.E > 0                          # representative modulus set


def test_dialog_uses_scaffold_cards(qapp):
    """D1: the flat form was rebuilt onto GroupCard panels (Identity / Parameters)."""
    from PySide6.QtWidgets import QGroupBox

    from material_editor import MaterialDialog
    dlg = MaterialDialog(None, _project())
    titles = {b.title() for b in dlg.findChildren(QGroupBox)}
    assert {"Identity", "Parameters"} <= titles


def test_dialog_edit_roundtrips_steel(qapp):
    from material_editor import MaterialDialog
    steel = Material(id=5, name="rebar", E=200e9, kind="cyclic_steel",
                     params={"E": 200e9, "fy": 460e6, "fu": 650e6,
                             "eps_sh": 0.008, "eps_su": 0.09})
    dlg = MaterialDialog(None, _project(), steel)
    mat = dlg.data()
    assert mat.id == 5 and mat.kind == "cyclic_steel"
    assert mat.params["fy"] == pytest.approx(460e6, rel=1e-6)
    assert mat.fy == pytest.approx(460e6, rel=1e-6)   # design field kept in sync


def test_creep_card_present_and_writes_props(qapp):
    """C5: the material editor has a creep card that writes structured creep
    props (f_cm / RH / h_0 / chi / enabled) onto the material."""
    from PySide6.QtWidgets import QGroupBox

    from material_editor import MaterialDialog
    dlg = MaterialDialog(None, _project())
    titles = {b.title() for b in dlg.findChildren(QGroupBox)}
    assert "Time-dependent (creep / shrinkage)" in titles

    dlg.creep_on.setChecked(True)
    dlg.cr_fcm.setValue(48.0)
    dlg.cr_rh.setValue(60.0)
    dlg.cr_h0.setValue(0.30)
    dlg.cr_chi.setValue(0.8)
    c = dlg.data().creep
    assert c["enabled"] is True
    assert c["f_cm"] == pytest.approx(48.0e6, rel=1e-9)
    assert c["RH"] == 60.0 and c["h_0"] == 0.30 and c["chi"] == 0.8


def test_creep_props_round_trip_and_edit(qapp):
    """A material's creep props survive save/load and repopulate the editor."""
    from material_editor import MaterialDialog

    m = Material(id=2, name="C40", E=34e9, nu=0.2, kind="elastic_isotropic",
                 creep={"enabled": True, "f_cm": 48e6, "RH": 65.0,
                        "h_0": 0.25, "chi": 0.85})
    # JSON round-trip through the project
    p = _project()
    p.materials = [m]
    m2 = Project.from_dict(p.to_dict()).materials[0]
    assert m2.creep == m.creep

    # editor repopulates from the seeded material
    dlg = MaterialDialog(None, _project(), m)
    assert dlg.creep_on.isChecked()
    assert dlg.cr_fcm.value() == pytest.approx(48.0, rel=1e-9)
    assert dlg.cr_chi.value() == 0.85


def test_manager_delete_guard_when_in_use(qapp, monkeypatch):
    import material_editor
    monkeypatch.setattr(material_editor.QMessageBox, "warning",
                        lambda *a, **k: None)          # don't block offscreen
    p = _project()
    p.members = [Member(id=1, n1=1, n2=2, section=1, material=1)]
    dlg = material_editor.MaterialManagerDialog(None, p)
    dlg.table.setCurrentCell(0, 0)
    dlg._delete()
    assert len(dlg.result_materials()) == 1            # in-use -> not deleted


def test_manager_delete_unused(qapp):
    from material_editor import MaterialManagerDialog
    p = _project()
    dlg = MaterialManagerDialog(None, p)               # no members
    dlg.table.setCurrentCell(0, 0)
    dlg._delete()
    assert len(dlg.result_materials()) == 0
