"""Construction-stage results — the camber / geometry-control diagram
(bridge GUI plan G3).

Fed a :class:`femsolver.bridges.StagedCamber`, it draws the deflection building
up stage by stage and the required build-high **camber** (= −final deflection),
so the erected bridge lands on the target profile. Non-modal. Pure Qt +
matplotlib (Agg) — headless-constructible under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

import style
from units import Quantity, UnitSystem


class ConstructionStageResultsDialog(QDialog):
    """Camber diagram + per-stage deflection history."""

    def __init__(self, parent, camber, *, n_stages=0, unitsys=None):
        super().__init__(parent)
        self.setWindowTitle("Construction-stage results")
        self.resize(600, 460)
        us = unitsys or UnitSystem()
        L = Quantity.LENGTH
        l_u = us.label(L)

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_SM)

        head = QLabel("Camber / geometry control")
        head.setObjectName("h2")
        root.addWidget(head)
        dmax = float(np.max(np.abs(camber.final_deflection))) if len(
            camber.final_deflection) else 0.0
        dmax_d = us.to_display(dmax, L)
        sub = QLabel(f"{n_stages} stages · max final deflection = "
                     f"{dmax_d:.4g} {l_u} · required camber = build "
                     f"{dmax_d:.4g} {l_u} high")
        sub.setObjectName("sub")
        root.addWidget(sub)

        def _disp(arr):
            return [us.to_display(v, L) for v in np.asarray(arr)]

        fig = Figure(figsize=(5.6, 3.6), layout="constrained")
        ax = fig.add_subplot(111)
        x = _disp(camber.x)
        sd = np.asarray(camber.stage_deflection)
        if sd.ndim == 2 and sd.shape[0] > 1:
            cmap = __import__("matplotlib").colormaps["viridis"]
            for k in range(sd.shape[0]):
                ax.plot(x, _disp(sd[k]), lw=1.0,
                        color=cmap(k / max(1, sd.shape[0] - 1)), alpha=0.7)
        ax.plot(x, _disp(camber.final_deflection), "-o",
                color=style.C_SECONDARY, lw=2, ms=3, label="final deflection")
        ax.plot(x, _disp(camber.final_camber), "-o",
                color=style.C_PRIMARY, lw=2.4, ms=3,
                label="required camber (build high)")
        ax.axhline(0.0, color=style.AX_SPINE, lw=0.8)
        ax.set_xlabel(f"x ({l_u})")
        ax.set_ylabel(f"vertical ({l_u})")
        ax.set_title("faint = stage-by-stage deflection")
        ax.legend(fontsize=8)
        try:
            style.beautify_axes(ax)
        except Exception:                              # noqa: BLE001
            pass
        canvas = Canvas(fig)
        canvas.setMinimumHeight(280)
        root.addWidget(canvas, 1)

        hint = QLabel("Build each node the 'camber' amount high so the finished "
                      "structure settles onto the target profile.")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        root.addWidget(hint)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        root.addWidget(btns)
        style.apply(self)

    @classmethod
    def show_results(cls, parent, camber, **kw):
        dlg = cls(parent, camber, **kw)
        dlg.show()
        return dlg
