"""Saveable analysis-case type registry — ``desktop/case_types.py``.

Headless coverage of the :class:`case_types.CaseType` adapter contract for the
migrated built-in types (Modal, Buckling): the list-row helpers
(``detail`` / ``default_params``), the ``params → runtime config``
(``build_config``, incl. the JSON list→tuple normalisation), and that
``dispatch`` invokes the matching ``MainWindow.run_*``. The dialog-driven
``edit`` path is exercised via the analysis-cases home tests.
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

from project import Material, Member, Node, Project, Section  # noqa: E402


def _project():
    p = Project(ndm=2, ndf=3)
    p.nodes = [Node(1, 0, 0, supports=(1, 1, 1)), Node(2, 0, 3)]
    p.sections = [Section(id=1, name="s", A=1e-3, Iz=1e-5)]
    p.materials = [Material(1, "m", E=2e11, nu=0.3)]
    p.members = [Member(1, 1, 2, 1, 1)]
    return p


class _FakeWin:
    """Records the run_* call a dispatch makes, without running anything."""

    def __init__(self):
        self.calls = []

    def run_modal(self, num_modes=None, lumped=None,
                  initial_condition=("zero",)):
        self.calls.append(("run_modal", num_modes, lumped, initial_condition))

    def run_buckling(self, config=None):
        self.calls.append(("run_buckling", config))

    def run_moving_load(self, config=None):
        self.calls.append(("run_moving_load", config))

    def run_temperature_gradient(self, config=None):
        self.calls.append(("run_temperature_gradient", config))

    def run_load_rating(self, config=None):
        self.calls.append(("run_load_rating", config))

    def run_timehistory_dialog(self, seed=None):
        self.calls.append(("run_timehistory_dialog", seed))

    def run_linear_static(self, loads_applied=None):
        self.calls.append(("run_linear_static", loads_applied))


def test_registry_has_migrated_types():
    import case_types
    assert set(case_types.TYPES) >= {"linstatic", "modal", "buckling",
                                     "movingload", "tempgradient", "loadrating",
                                     "responsespectrum", "vehicledynamics",
                                     "influencesurface", "cabletuning",
                                     "timehistory"}
    # every registered type carries a label + icon and lands in the ordered list
    for ct in case_types._ORDER:
        assert ct.type_id and ct.type_label and ct.icon
    assert case_types.get("does-not-exist") is None


def test_linstatic_adapter_contract():
    import case_types
    ct = case_types.get("linstatic")
    p = _project()                                   # auto-seeds a Dead pattern
    # default: the first pattern ×1
    assert ct.default_params(p) == {"loads_applied": [[1, 1.0]]}
    assert "Dead" in ct.detail(p, {"loads_applied": [[1, 1.2]]})
    assert ct.detail(p, {"loads_applied": []}) == "no loads applied"
    # build_config normalises the JSON rows (drops a zero) to (int, float) pairs
    cfg = ct.build_config(p, {"loads_applied": [[1, "1.4"], [1, 0]]})
    assert cfg == [(1, 1.4)]
    win = _FakeWin()
    ct.dispatch(win, cfg)
    assert win.calls == [("run_linear_static", [(1, 1.4)])]


def test_modal_adapter_contract():
    import case_types
    ct = case_types.get("modal")
    p = _project()
    assert ct.default_params(p) == {"num_modes": 6, "lumped": False}
    assert "6 modes" in ct.detail(p, ct.default_params(p))
    # config now carries the E2 initial condition (default unstressed)
    assert ct.build_config(p, {"num_modes": 8, "lumped": True}) \
        == (8, True, ("zero",))
    win = _FakeWin()
    ct.dispatch(win, (8, True, ("zero",)))
    assert win.calls == [("run_modal", 8, True, ("zero",))]


def test_buckling_adapter_contract():
    import case_types
    ct = case_types.get("buckling")
    p = _project()
    dp = ct.default_params(p)
    assert dp["selection"] == ["all", None] and dp["num_modes"] == 4
    assert "all load patterns" in ct.detail(p, dp)
    assert "pattern 2" in ct.detail(p, {"selection": ["case", 2], "num_modes": 3,
                                        "subdivisions": 5})
    # JSON stores the selection as a list; build_config restores the tuple the
    # runner expects
    cfg = ct.build_config(p, {"selection": ["case", 2], "num_modes": 3,
                              "subdivisions": 5})
    assert cfg == (("case", 2), 3, 5, ("zero",))    # E2e initial condition
    win = _FakeWin()
    ct.dispatch(win, cfg)
    assert win.calls == [("run_buckling", (("case", 2), 3, 5, ("zero",)))]


def test_edit_cap_is_a_positive_bound():
    import case_types
    ct = case_types.get("modal")
    assert ct._edit_cap(_project()) >= 1


def test_timehistory_dispatch_seeds_the_runner():
    import case_types
    from project import TimeHistoryFunction
    p = _project()
    p.th_functions = [TimeHistoryFunction(id=1, name="EC", dt=0.02,
                                          values=[0.1, 0.2, 0.3])]
    ct = case_types.get("timehistory")
    cfg = ct.build_config(p, {"function_id": 1, "control_node": 2,
                              "direction": "y", "scale": 1.0})
    win = _FakeWin()
    ct.dispatch(win, cfg)
    assert win.calls == [("run_timehistory_dialog", cfg)]   # opens seeded


def test_dict_config_types_dispatch_by_identity():
    # moving load / temp gradient / load rating persist their dialog dict as-is;
    # build_config is identity and dispatch forwards config= to the runner
    import case_types
    p = _project()
    cases = {
        "movingload": ({"lane": [1, 2], "vehicle": "hl93",
                        "response": ["M", 1, "i"]}, "run_moving_load"),
        "tempgradient": ({"source": "aashto", "zone": 3, "alpha": 1e-5,
                          "members": [1]}, "run_temperature_gradient"),
        "loadrating": ({"lane": [1, 2], "response": ["M", 1, "i"], "Rn": 1.0,
                        "adtt": None, "permit_gamma_LL": None},
                       "run_load_rating"),
    }
    for type_id, (params, run_name) in cases.items():
        ct = case_types.get(type_id)
        cfg = ct.build_config(p, params)
        assert cfg == params                              # identity
        win = _FakeWin()
        ct.dispatch(win, cfg)
        assert win.calls == [(run_name, params)]
