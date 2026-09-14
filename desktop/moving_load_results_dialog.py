"""Moving-load results — the influence-line plot and the vehicle envelope.

``MovingLoadResultsDialog`` draws the influence line for the selected response
and reports the governing vehicle envelope (max / min effect). Non-modal, so it
stays open beside the model. Fed by the engine's ``InfluenceLine`` and the
envelope dict from ``aashto_hl93_envelope`` / ``moving_load_envelope``. Pure Qt
+ matplotlib (Agg) — headless-constructible under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QLabel, QVBoxLayout)

import style


class MovingLoadResultsDialog(QDialog):
    """Influence-line plot + vehicle envelope."""

    def __init__(self, parent, il, env, *, response_label="response",
                 units="", vehicle=""):
        super().__init__(parent)
        self.setWindowTitle("Moving-load results")
        self.resize(560, 460)

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_SM)

        head = QLabel(f"Influence line — {response_label}")
        head.setObjectName("h2")
        root.addWidget(head)
        u = f" {units}" if units else ""
        sub = QLabel(f"{vehicle} envelope:  max = {env.get('max', 0.0):.4g}{u}"
                     f"   ·   min = {env.get('min', 0.0):.4g}{u}")
        sub.setObjectName("sub")
        root.addWidget(sub)

        fig = Figure(figsize=(5.2, 3.4), layout="constrained")
        ax = fig.add_subplot(111)
        st = list(il.stations)
        vals = list(il.values)
        ax.fill_between(st, vals, 0.0, color=style.C_PRIMARY, alpha=0.20)
        ax.plot(st, vals, "-", lw=1.8, color=style.C_PRIMARY)
        ax.axhline(0.0, color=style.AX_SPINE, lw=0.8)
        ax.set_xlabel("station along lane (m)")
        ax.set_ylabel(f"influence ordinate{u}")
        ax.set_title(f"per unit load — {response_label}")
        try:
            style.beautify_axes(ax)
        except Exception:                              # noqa: BLE001
            pass
        canvas = Canvas(fig)
        canvas.setMinimumHeight(260)
        root.addWidget(canvas, 1)

        hint = QLabel("Ordinate = response per unit downward load at that "
                      "station; the envelope convolves the vehicle over it.")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        root.addWidget(hint)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        root.addWidget(btns)
        style.apply(self)

    @classmethod
    def show_results(cls, parent, il, env, **kw):
        dlg = cls(parent, il, env, **kw)
        dlg.show()
        return dlg
