"""Core-vs-cover fibre preview in the Section Designer Fibres tab (plan §16 G-S4).

The preview must show exactly the two zones the confined analysis integrates —
a confined Mander **core** (the outline inset by the cover) and an unconfined
**cover** ring. The engine helper ``confined_core_polygon`` is the single source
of that geometry (shared by ``_confined_fiber_section`` and the preview), so
"what you see is what you run". Headless (offscreen) for the GUI half.
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
sys.path.insert(0, str(_ROOT / "src"))

import section_gui_core as core                    # noqa: E402


def _manual_conf(shape, **kw):
    conf = {"conf_shape": shape, "conf_fyh": 400e6, "conf_Asp": 1e-4,
            "conf_s": 0.10, "conf_sp": 0.09, "conf_rho_cc": 0.02,
            "conf_hooptype": "Hoop", **kw}
    return tuple(conf.items())


def _confined_circular():
    return core.Spec(kind="Circular", D=0.9, cover=0.05, fc=35e6, fy=500e6,
                     conc_model="Mander", conf_override=True,
                     conf_manual=_manual_conf("Circular", conf_ds=0.8),
                     rebar_groups=(("Perimeter", "12B25", "", "S500"),))


# ------------------------------------------------ engine helper (shared source)

def test_confined_core_polygon_area_matches_inset():
    poly = core.confined_core_polygon(_confined_circular())
    assert poly is not None
    # a circle of radius R inset by the cover -> radius (R - cover)
    assert poly.area == pytest.approx(math.pi * (0.45 - 0.05) ** 2, rel=0.01)


def test_confined_core_polygon_none_when_cover_too_large():
    spec = core.Spec(kind="Circular", D=0.6, cover=0.35, n_perim=0,
                     rebar_groups=())          # no rebar -> section still builds
    assert core.confined_core_polygon(spec) is None


def test_preview_zoning_matches_the_run():
    """The key G-S4 guarantee: a fibre's *material* zone in the section the
    analysis builds (raised f'cc = core, base f'c = cover) agrees with the
    *polygon* the preview uses to colour it."""
    import nonlinear as NL
    import numpy as np
    from shapely.geometry import Point
    from femsolver.materials.uniaxial import ConcreteTensionStiffening

    spec = _confined_circular()
    fs = NL.fiber_section_from_spec(spec)
    poly = core.confined_core_polygon(spec)

    def peak_MPa(mat):
        return -min(mat.get_response(-e)[0]
                    for e in np.linspace(1e-4, 8e-3, 40)) / 1e6

    conc = [f for f in fs.fibers
            if isinstance(f.material, ConcreteTensionStiffening)]
    base = 35.0
    for f in conc:
        raised = peak_MPa(f.material) > 1.05 * base      # confined law?
        inside = poly.contains(Point(f.z, f.y))          # preview says core?
        assert raised == inside                          # they must agree


# ------------------------------------------------------ GUI (Fibres tab)

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    pytest.importorskip("matplotlib")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _legend(ax):
    lg = ax.get_legend()
    return [t.get_text() for t in lg.get_texts()] if lg else []


def test_fibres_tab_has_core_cover_mode(qapp):
    from section_designer import SectionDesignerWindow
    w = SectionDesignerWindow(spec=_confined_circular())
    modes = [w.fib_mode.itemText(i) for i in range(w.fib_mode.count())]
    assert "Core / cover" in modes


def test_core_cover_render_confined(qapp):
    from section_designer import SectionDesignerWindow
    w = SectionDesignerWindow(spec=_confined_circular())
    w.fib_mode.setCurrentText("Core / cover")
    w._draw_fibers()
    ax = w.fib_fig.axes[0]
    labels = " ".join(_legend(ax))
    assert "Confined core" in labels
    assert "Cover" in labels
    assert "Rebar" in labels
    assert "core" in ax.get_title().lower()


def test_core_cover_render_unconfined_labels_honestly(qapp):
    """A section with no tie group still shows the geometric core/cover, but is
    labelled unconfined (no false 'confined' claim)."""
    from section_designer import SectionDesignerWindow
    spec = core.Spec(kind="Circular", D=0.9, cover=0.05, fc=35e6, fy=500e6,
                     conc_model="Kent-Park",
                     rebar_groups=(("Perimeter", "12B25", "", "S500"),))
    w = SectionDesignerWindow(spec=spec)
    w.fib_mode.setCurrentText("Core / cover")
    w._draw_fibers()
    ax = w.fib_fig.axes[0]
    labels = " ".join(_legend(ax))
    assert "unconfined" in labels.lower() or "unconfined" in ax.get_title().lower()
    assert "Confined core" not in labels
