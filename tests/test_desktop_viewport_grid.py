"""Viewport CAD grid (plan gui-polish **V2**) — ``desktop/model_view.py``.

V2 replaces PyVista's plot-style bounds box (the old 'X Axis / Y Axis' frame)
with a **CAD ground grid** in the model plane plus a corner orientation triad.
Headless coverage of the grid builder (real geometry, no GL) + a live rebuild.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvistaqt")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_nice_step_snaps_to_1_2_5():
    import model_view as mv
    assert mv._nice_step(80.0) in (5.0, 10.0)     # ~8 → 10
    assert mv._nice_step(20.0) == 2.0             # ~2
    assert mv._nice_step(0.0) == 1.0              # degenerate guard
    for span in (0.3, 3.0, 12.0, 45.0, 100.0):
        step = mv._nice_step(span)
        assert step > 0
        assert 4 <= span / step <= 30            # a sane number of lines


def test_ground_grid_spans_the_model():
    import model_view as mv
    from demo_model import demo_project
    model = demo_project().build_model()          # 8 m × 6 m frame
    grid = mv._build_ground_grid(model)
    assert grid is not None
    assert grid.n_lines > 0                        # real line segments
    xs, ys = grid.points[:, 0], grid.points[:, 1]
    assert xs.min() <= 0.0 and xs.max() >= 8.0     # brackets the frame in x
    assert ys.min() <= 0.0 and ys.max() >= 6.0     # …and in y
    # the ground grid sits behind the model plane (z ≲ 0), not coplanar
    assert grid.points[:, 2].max() <= 0.0


def test_empty_model_has_no_grid():
    import model_view as mv
    assert mv._build_ground_grid(None) is None


def test_draw_grid_rebuilds_live(qapp):
    import model_view as mv
    from demo_model import demo_project
    v = mv.ModelView()
    v.set_model(demo_project().build_model())      # calls _draw_grid internally
    # the ground grid actor is present under its stable name
    assert "groundgrid" in dict(v.renderer.actors)
    v._draw_grid()                                 # idempotent rebuild, no raise
    assert "groundgrid" in dict(v.renderer.actors)
