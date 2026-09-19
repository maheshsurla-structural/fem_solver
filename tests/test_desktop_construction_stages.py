"""Construction stages + camber — desktop wiring (bridge GUI increment G3).

Covers the persisted Stage model, the stage-manager dialog (auto-sequence,
add/delete, exclusivity), the Analysis-cases row, and the
``MainWindow.run_construction_stages`` runner.
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

from project import (Material, Member, Node, Project,  # noqa: E402
                     Section, Stage)


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _cantilever(n=8, dx=3.0):
    p = Project(ndm=2, ndf=3)
    p.nodes = [Node(i + 1, i * dx, 0.0) for i in range(n + 1)]
    p.nodes[0].supports = (1, 1, 1)                    # fixed base
    p.sections = [Section(id=1, name="deck", A=0.6, Iz=0.08)]
    p.materials = [Material(1, "conc", E=3e10, nu=0.2, rho=2500.0)]
    p.members = [Member(i + 1, i + 1, i + 2, 1, 1) for i in range(n)]
    return p


# ---------------------------------------------------------------- model
def test_stage_model_serializes():
    p = _cantilever()
    p.stages = [Stage(id=1, name="S1", add_members=[1, 2]),
                Stage(id=2, name="S2", add_members=[3, 4])]
    p2 = Project.from_dict(p.to_dict())
    assert [(s.id, s.name, s.add_members) for s in p2.stages] == \
        [(1, "S1", [1, 2]), (2, "S2", [3, 4])]


def test_stage_time_dependent_fields_round_trip():
    """C0: death + time-dependent stage attributes survive save/load."""
    p = _cantilever()
    p.stages = [Stage(id=1, name="cast", add_members=[1, 2],
                      remove_members=[9], duration_days=30.0,
                      age_at_activation_days=7.0, creep=True)]
    s = Project.from_dict(p.to_dict()).stages[0]
    assert s.remove_members == [9]
    assert s.duration_days == 30.0
    assert s.age_at_activation_days == 7.0
    assert s.creep is True


def test_old_stage_migrates_new_fields():
    """A pre-C0 stage dict (only id/name/add_members) loads with defaults."""
    p = _cantilever()
    d = p.to_dict()
    d["stages"] = [{"id": 1, "name": "S1", "add_members": [1]}]
    s = Project.from_dict(d).stages[0]
    assert s.remove_members == [] and s.duration_days == 0.0
    assert s.age_at_activation_days == 28.0 and s.creep is False


def test_old_project_without_stages_loads():
    p = _cantilever()
    d = p.to_dict()
    d.pop("stages", None)                             # pre-stage project
    assert Project.from_dict(d).stages == []


# ---------------------------------------------------------------- dialog
def test_auto_sequence_partitions_members(qapp):
    from stage_dialog import StageManagerDialog
    d = StageManagerDialog(None, _cantilever())
    d.nseg.setValue(4)
    d._auto_sequence()
    stages = d.result()
    assert len(stages) == 4
    # every member assigned exactly once, left → right
    seen = [m for s in stages for m in s.add_members]
    assert sorted(seen) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert stages[0].add_members == [1, 2]


def test_add_delete_stage(qapp):
    from stage_dialog import StageManagerDialog
    d = StageManagerDialog(None, _cantilever())
    d._add()
    assert len(d.result()) == 1
    d._add()
    d.stage_list.setCurrentRow(0)
    d._delete()
    assert len(d.result()) == 1


def test_dialog_edits_time_dependent_fields(qapp):
    """C4: duration / age / creep editors write onto the selected stage."""
    from stage_dialog import StageManagerDialog
    p = _cantilever()
    p.stages = [Stage(id=1, name="cast", add_members=[1, 2])]
    d = StageManagerDialog(None, p)
    d.stage_list.setCurrentRow(0)
    d.duration.setValue(45.0)
    d.age.setValue(7.0)
    d.creep.setChecked(True)
    s = d.result()[0]
    assert s.duration_days == 45.0
    assert s.age_at_activation_days == 7.0
    assert s.creep is True


def test_dialog_reselect_repopulates_time_fields(qapp):
    """Switching stages loads each stage's own duration/age/creep + removals."""
    from stage_dialog import StageManagerDialog
    p = _cantilever()
    p.stages = [Stage(id=1, name="s1", add_members=[1, 2], duration_days=10.0,
                      creep=True),
                Stage(id=2, name="s2", add_members=[3, 4], remove_members=[1],
                      age_at_activation_days=3.0)]
    d = StageManagerDialog(None, p)
    d.stage_list.setCurrentRow(0)
    assert d.duration.value() == 10.0 and d.creep.isChecked()
    d.stage_list.setCurrentRow(1)
    assert d.age.value() == 3.0 and not d.creep.isChecked()
    gone = [d.removed.item(i).data(0x0100) for i in range(d.removed.count())
            if d.removed.item(i).isSelected()]
    assert gone == [1]


def test_dialog_built_removed_exclusive(qapp):
    """A member can't be both built and removed in the same stage."""
    from stage_dialog import StageManagerDialog
    p = _cantilever()
    p.stages = [Stage(id=1, name="s1", add_members=[1, 2, 3])]
    d = StageManagerDialog(None, p)
    d.stage_list.setCurrentRow(0)
    # mark member 2 for removal -> it drops out of "built"
    for i in range(d.removed.count()):
        if d.removed.item(i).data(0x0100) == 2:
            d.removed.item(i).setSelected(True)
    s = d.result()[0]
    assert 2 in s.remove_members
    assert 2 not in s.add_members


# ---------------------------------------------------- analysis-cases row
def test_analysis_cases_lists_stages(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog
    dlg = AnalysisCasesDialog(None, _cantilever())
    kinds = [m["kind"] for m in dlg._row_meta]
    assert "stages" in kinds
    dlg.table.setCurrentCell(kinds.index("stages"), 0)
    dlg._run()
    assert dlg._run_request == ("stages",)


# ------------------------------------------------ run_construction_stages
def test_run_stages_camber(qapp):
    from main_window import MainWindow
    w = MainWindow()
    p = _cantilever()
    w.load_project(p)
    stages = [Stage(id=i + 1, name=f"S{i + 1}",
                    add_members=[2 * i + 1, 2 * i + 2]) for i in range(4)]
    res = w.run_construction_stages(config=stages)
    assert res is not None and res["stages"] == 4
    cam = res["camber"]
    # cantilever built under self-weight → tip droops, needs build-high camber
    assert cam.final_camber[-1] > 0.0
    assert abs(cam.final_deflection[-1]) > 0.0
    assert hasattr(w, "_stage_results_dlg")


def test_run_stages_creep_amplifies_deflection(qapp):
    """C0→C1a end to end: a stage that opts into creep (with a concrete f_cm)
    droops more than the elastic build under the same self-weight."""
    from main_window import MainWindow

    p0 = _cantilever()
    w0 = MainWindow()
    w0.load_project(p0)
    r0 = w0.run_construction_stages(config=[
        Stage(id=1, name="S1", add_members=[m.id for m in p0.members])])
    d_elastic = abs(r0["camber"].final_deflection[-1])

    p1 = _cantilever()
    p1.materials[0].params = {"fc": 30.0e6}          # concrete mean strength
    w1 = MainWindow()
    w1.load_project(p1)
    r1 = w1.run_construction_stages(config=[
        Stage(id=1, name="S1", add_members=[m.id for m in p1.members],
              duration_days=18250.0, age_at_activation_days=28.0,
              creep=True)])
    d_creep = abs(r1["camber"].final_deflection[-1])

    assert d_creep > 1.2 * d_elastic                 # creep amplifies droop


def test_run_stages_uses_material_creep_props(qapp):
    """C5: structured per-material creep props (incl. ageing χ) drive the run.
    A lower χ (age-adjusted) gives less long-term droop than χ=1 (plain EMM)."""
    from main_window import MainWindow

    def _run(chi):
        p = _cantilever()
        p.materials[0].creep = {"enabled": True, "f_cm": 38.0e6, "RH": 70.0,
                                "h_0": 0.20, "chi": chi}
        w = MainWindow()
        w.load_project(p)
        r = w.run_construction_stages(config=[
            Stage(id=1, name="S1", add_members=[m.id for m in p.members],
                  duration_days=18250.0, age_at_activation_days=28.0,
                  creep=True)])
        return abs(r["camber"].final_deflection[-1])

    d_emm = _run(1.0)          # plain EMM
    d_aaem = _run(0.8)         # age-adjusted (stiffer -> less droop)
    assert d_emm > d_aaem > 0.0


def test_stage_forces_tab_populates(qapp):
    """C7: the results dialog exposes a per-stage member-force table that
    reflects birth (elements are '—' before they exist, forces once active)."""
    from construction_stage_results_dialog import (
        ConstructionStageResultsDialog, _local_axial_moment)
    from femsolver.bridges import (ErectionStage, IncrementalStagedAnalysis,
                                   staged_camber)

    p = _cantilever(n=4, dx=3.0)
    model = p.build_model(with_loads=False)

    def _sw(ids):
        loads = {}
        for tag in ids:
            el = model.element(tag)
            L = float(((el.node_coords()[1] - el.node_coords()[0]) ** 2)
                      .sum() ** 0.5)
            w = el.material.rho * el.area * L * 9.80665
            for nd in el.node_tags:
                loads.setdefault(nd, [0.0, 0.0, 0.0])[1] += -w / 2.0
        return loads

    erection = [ErectionStage("S1", add_elements=[1, 2], loads=_sw([1, 2])),
                ErectionStage("S2", add_elements=[3, 4], loads=_sw([3, 4]))]
    res = IncrementalStagedAnalysis(model, erection).run()
    cam = staged_camber(res, model, erection, dof=1)

    dlg = ConstructionStageResultsDialog(
        None, cam, n_stages=2, result=res, model=model,
        stage_names=["S1", "S2"])
    # stage 0: elements 3,4 are not yet born
    dlg.stage_combo.setCurrentIndex(0)
    rows = {int(dlg.table.item(r, 0).text()): dlg.table.item(r, 1).text()
            for r in range(dlg.table.rowCount())}
    assert rows[1] == "active" and rows[3] == "—"
    # stage 1: all four active
    dlg.stage_combo.setCurrentIndex(1)
    rows = {int(dlg.table.item(r, 0).text()): dlg.table.item(r, 1).text()
            for r in range(dlg.table.rowCount())}
    assert rows[3] == "active" and rows[4] == "active"

    # the reducer returns a finite (N, |M|) for a born 2-D beam element
    nm = _local_axial_moment(model.element(1), res.element_force_history[1][0])
    assert nm is not None and len(nm) == 2


def test_run_stages_guards(qapp, monkeypatch):
    import main_window as MW
    monkeypatch.setattr(MW.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(MW.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: None))
    # zero density → no self-weight
    w = MW.MainWindow()
    p = _cantilever()
    p.materials[0].rho = 0.0
    w.load_project(p)
    stages = [Stage(id=1, name="S1", add_members=[m.id for m in p.members])]
    assert w.run_construction_stages(config=stages) is None
    # unstable build order → the tip segment born alone is a mechanism
    w2 = MW.MainWindow()
    w2.load_project(_cantilever())
    bad = [Stage(id=1, name="S1", add_members=[8]),
           Stage(id=2, name="S2", add_members=[1, 2, 3, 4, 5, 6, 7])]
    assert w2.run_construction_stages(config=bad) is None


def test_run_stages_3d_camber(qapp):
    """A3.3a: construction-stage camber works in 3-D (vertical DOF = uz)."""
    from main_window import MainWindow
    n, L = 8, 24.0
    p = Project(ndm=3, ndf=6)
    for i in range(n + 1):
        p.nodes.append(Node(id=i + 1, x=i * L / n, y=0.0, z=0.0))
    p.nodes[0].supports = (1, 1, 1, 1, 1, 1)          # fixed base cantilever
    p.sections = [Section(id=1, name="deck", A=0.6, Iz=0.08, Iy=0.05, J=0.03)]
    p.materials = [Material(1, "conc", E=3e10, nu=0.2, rho=2500.0)]
    p.members = [Member(i + 1, i + 1, i + 2, 1, 1) for i in range(n)]
    w = MainWindow()
    w.load_project(p)
    stages = [Stage(id=i + 1, name=f"S{i + 1}",
                    add_members=[2 * i + 1, 2 * i + 2]) for i in range(4)]
    res = w.run_construction_stages(config=stages)
    assert res is not None and res["stages"] == 4
    cam = res["camber"]
    assert cam.final_camber[-1] > 0.0                  # tip droops → build high
    assert abs(cam.final_deflection[-1]) > 0.0
