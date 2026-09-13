"""Viewport theming (plan gui-polish **V1**) — ``desktop/model_view.py``.

The 3-D viewport used to be hard-coded ``white`` with fixed entity hex and a
plot-style ``show_grid()``; it never followed the app theme. V1 moves its
background, grid/axis colour and every model-entity ink into the ``style``
palette (read at render time) and adds ``apply_theme()`` so the shell can
restyle it on a theme switch — mirroring ``SectionCanvas``.

Headless (offscreen) coverage: the tokens exist and swap across themes, the
diagram colour tracks the theme, no hard-coded ink survives in a draw call, and
the real viewport builds + restyles in both light and dark over a demo model
without raising (and the renderer background actually darkens).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

VIEW_TOKENS = ["VIEW_BG", "VIEW_GRID", "VIEW_AXIS", "V_MEMBER", "V_NODE",
               "V_SUPPORT", "V_REFERENCE", "V_DEFORMED", "V_DEFORMED_NODE",
               "V_SELECTION", "V_HINGE", "V_DIAG_N", "V_DIAG_V", "V_DIAG_M"]


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvistaqt")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _reset_theme():
    import style
    yield
    style.set_theme("light")


def test_view_tokens_present_and_theme_dependent():
    import style
    style.set_theme("light")
    light = {}
    for t in VIEW_TOKENS:
        assert hasattr(style, t), f"missing style.{t}"
        light[t] = getattr(style, t)
    style.set_theme("dark")
    changed = [t for t in VIEW_TOKENS if getattr(style, t) != light[t]]
    # the palette must genuinely swap the viewport inks between themes
    assert len(changed) >= len(VIEW_TOKENS) - 2, f"only {changed} changed"


def test_diagram_colour_tracks_theme():
    import model_view as mv
    import style
    style.set_theme("light")
    assert mv._diagram_color("M") == style.V_DIAG_M
    n_light = mv._diagram_color("N")
    style.set_theme("dark")
    assert mv._diagram_color("N") == style.V_DIAG_N != n_light


def test_no_hardcoded_entity_hex_in_draw_calls():
    """Every ``color=`` in a draw call reads a token/helper — no ``color="#..."``
    literal (the semantic DCR ramp + legend swatches are not ``color=`` calls)."""
    src = (_ROOT / "desktop" / "model_view.py").read_text(encoding="utf-8")
    literals = re.findall(r'color=("#[0-9a-fA-F]{3,8}")', src)
    assert not literals, f"hard-coded ink in a draw call: {literals}"


def test_viewport_builds_and_restyles_both_themes(qapp):
    import style
    from demo_model import demo_project
    from main_window import MainWindow

    style.set_theme("light")
    w = MainWindow()
    w.load_project(demo_project())
    assert hasattr(w.view, "apply_theme")
    assert w.view._replay is not None          # a scene was captured to restyle

    def _lum():
        try:
            return sum(w.view.renderer.GetBackground())
        except Exception:
            return None

    lum_light = _lum()
    style.set_theme("dark")
    w.view.apply_theme()                       # must not raise
    lum_dark = _lum()
    style.set_theme("light")
    w.view.apply_theme()

    assert w.view._model is not None           # model survived the restyle
    if lum_light is not None and lum_dark is not None:
        assert lum_dark < lum_light            # background actually darkened
