"""U6b — the remaining unit-conversion surfaces: moving-load results, the
hinge editor (mode-aware), and the frame generator.

The model stays SI; these dialogs display / parse in the project's units.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))
sys.path.insert(0, str(_ROOT / "src"))

from project import Hinge, Project  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


# --------------------------------------------------- moving-load results (#3)
def test_moving_load_results_convert_envelope_and_axes(qapp):
    from PySide6.QtWidgets import QLabel
    from moving_load_results_dialog import MovingLoadResultsDialog
    from units import Quantity, UnitSystem
    il = types.SimpleNamespace(stations=[0.0, 1.0, 2.0], values=[0.0, 0.5, 0.0])
    env = {"max": 1000.0, "min": -200.0}                # SI N·m
    dlg = MovingLoadResultsDialog(None, il, env, response_label="moment",
                                  unitsys=UnitSystem("kN", "m"),
                                  quantity=Quantity.MOMENT, vehicle="HL-93")
    sub = next(l for l in dlg.findChildren(QLabel) if "envelope" in l.text())
    # 1000 N·m -> 1 kN·m ; -200 -> -0.2 kN·m
    assert "max = 1 kN·m" in sub.text()
    assert "min = -0.2 kN·m" in sub.text()


# ---------------------------------------------------- hinge editor (#2)
def _proj_mm():
    p = Project(force_unit="kN", length_unit="mm")
    p.hinges = []
    return p


def test_hinge_absolute_length_converts(qapp):
    from hinge_editor import HingeDialog
    p = _proj_mm()
    h = Hinge(id=1, name="H", lp=2.0, lp_j=None, relative=False)   # 2 m stored
    dlg = HingeDialog(None, p, hinge=h)
    assert dlg.lp_i.value() == pytest.approx(2000.0)              # shown as mm
    dlg.lp_i.setValue(3000.0)                                     # type 3000 mm
    out = dlg.data()
    assert not out.relative
    assert out.lp == pytest.approx(3.0)                          # stored 3 m


def test_hinge_relative_ratio_not_converted(qapp):
    from hinge_editor import HingeDialog
    p = _proj_mm()
    h = Hinge(id=1, name="H", lp=0.1, lp_j=None, relative=True)   # ratio
    dlg = HingeDialog(None, p, hinge=h)
    assert dlg.lp_i.value() == pytest.approx(0.1)                # ratio unchanged
    out = dlg.data()
    assert out.relative and out.lp == pytest.approx(0.1)        # still a ratio


# ------------------------------------------------------- frame generator (#1)
def test_frame_dialog_lengths_convert(qapp):
    from editing import FrameDialog
    from units import UnitSystem
    dlg = FrameDialog(None, UnitSystem("kN", "mm"))
    assert dlg.bay_x.value() == pytest.approx(6000.0)            # 6 m default -> mm
    dlg.storey_h.setValue(4000.0)                                # 4000 mm
    params = dlg.params()
    assert params["storey_h"] == pytest.approx(4.0)             # stored 4 m
    assert params["bay_x"] == pytest.approx(6.0)


def test_frame_dialog_default_is_metres(qapp):
    from editing import FrameDialog
    dlg = FrameDialog(None)                                       # no units
    assert dlg.bay_x.value() == pytest.approx(6.0)              # metres, identity
    assert dlg.params()["bay_x"] == pytest.approx(6.0)
