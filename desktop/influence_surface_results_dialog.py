"""Influence-surface results — the plan contour of the influence surface plus
the governing multi-lane vehicle envelope (bridge GUI plan G4).

Fed the engine's :class:`femsolver.bridges.InfluenceSurface` and the
:func:`femsolver.bridges.multi_lane_envelope` result. Non-modal. Pure Qt +
matplotlib (Agg) — headless-constructible under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

import style
from units import Quantity, UnitSystem


class InfluenceSurfaceResultsDialog(QDialog):
    """Influence-surface contour + governing multi-lane envelope."""

    def __init__(self, parent, surface, envelope, *, response_label="response",
                 quantity=Quantity.LENGTH, vehicle="", n_lanes=0, unitsys=None):
        super().__init__(parent)
        self.setWindowTitle("Influence-surface results")
        self.resize(620, 500)
        us = unitsys or UnitSystem()
        u = f" {us.label(quantity)}"
        l_u = us.label(Quantity.LENGTH)

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_SM)

        head = QLabel(f"Influence surface — {response_label}")
        head.setObjectName("h2")
        root.addWidget(head)
        emax = us.to_display(envelope.get('max', 0.0), quantity)
        emin = us.to_display(envelope.get('min', 0.0), quantity)
        sub = QLabel(
            f"{vehicle} · {n_lanes} design lane(s) · governing "
            f"max = {emax:.4g}{u} "
            f"({envelope.get('max_num_lanes', 0)} lane(s), "
            f"m={envelope.get('max_factor', 1.0):.2f})   ·   "
            f"min = {emin:.4g}{u}")
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        root.addWidget(sub)

        fig = Figure(figsize=(5.6, 3.8), layout="constrained")
        ax = fig.add_subplot(111)
        pts = np.asarray(surface.points)
        px = [us.to_display(v, Quantity.LENGTH) for v in pts[:, 0]]
        py = [us.to_display(v, Quantity.LENGTH) for v in pts[:, 1]]
        vals = [us.to_display(v, quantity) for v in surface.values]
        try:
            tc = ax.tricontourf(px, py, vals, levels=16, cmap="viridis")
            fig.colorbar(tc, ax=ax, shrink=0.9).set_label(
                f"influence ordinate{u}")
            ax.plot(px, py, "k.", ms=2, alpha=0.3)
        except Exception:                              # noqa: BLE001
            ax.scatter(px, py, c=vals, cmap="viridis")
        ax.set_aspect("equal")
        ax.set_xlabel(f"longitudinal x ({l_u})")
        ax.set_ylabel(f"transverse y ({l_u})")
        ax.set_title(f"per unit load — {response_label}")
        canvas = Canvas(fig)
        canvas.setMinimumHeight(300)
        root.addWidget(canvas, 1)

        hint = QLabel("Ordinate = response per unit load at that plan position; "
                      "vehicles are placed lane-by-lane with multiple presence "
                      "to find the governing effect.")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        root.addWidget(hint)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        root.addWidget(btns)
        style.apply(self)

    @classmethod
    def show_results(cls, parent, surface, envelope, **kw):
        dlg = cls(parent, surface, envelope, **kw)
        dlg.show()
        return dlg
