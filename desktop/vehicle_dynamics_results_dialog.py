"""Vehicle-dynamics results — the response time-history and DAF, plus the
wheel–deck contact force for a sprung-mass run (bridge GUI plan G5).

Fed the times / dynamic / quasi-static histories and the DAF from
:class:`femsolver.bridges.MovingForceAnalysis` /
:class:`femsolver.bridges.VBIAnalysis`. Non-modal. Pure Qt + matplotlib (Agg) —
headless-constructible under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

import style


class VehicleDynamicsResultsDialog(QDialog):
    """Deflection time-history + DAF (+ contact force for VBI)."""

    def __init__(self, parent, times, dynamic, static, daf, *,
                 contact_force=None, weight=None, response_label="response",
                 speed_kmh=0.0):
        super().__init__(parent)
        self.setWindowTitle("Vehicle-dynamics results")
        self.resize(600, 500)

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_SM)

        head = QLabel(f"Vehicle dynamics — {response_label}")
        head.setObjectName("h2")
        root.addWidget(head)
        sub = QLabel(f"Speed {speed_kmh:.0f} km/h   ·   "
                     f"dynamic amplification factor DAF = {daf:.3f}")
        sub.setObjectName("sub")
        root.addWidget(sub)

        t = np.asarray(times)
        has_cf = contact_force is not None
        fig = Figure(figsize=(5.6, 4.0 if has_cf else 3.4),
                     layout="constrained")
        ax = fig.add_subplot(2, 1, 1) if has_cf else fig.add_subplot(111)
        ax.plot(t, -np.asarray(static) * 1e3, color="#94a3b8", lw=2,
                label="quasi-static")
        ax.plot(t, -np.asarray(dynamic) * 1e3, color=style.C_PRIMARY, lw=1.8,
                label="dynamic")
        ax.axhline(0.0, color=style.AX_SPINE, lw=0.7)
        ax.set_ylabel("deflection (mm, ↓)")
        ax.set_xlabel("time (s)")
        ax.legend(fontsize=8)
        try:
            style.beautify_axes(ax)
        except Exception:                              # noqa: BLE001
            pass

        if has_cf:
            ax2 = fig.add_subplot(2, 1, 2)
            cf = np.asarray(contact_force)
            if cf.ndim == 2:
                cf = cf[:, 0]
            ax2.plot(t, cf / 1e3, color=style.C_SECONDARY, lw=1.6,
                     label="contact force")
            if weight:
                ax2.axhline(weight / 1e3, color=style.AX_SPINE, ls="--",
                            lw=1.2, label=f"static weight {weight / 1e3:.0f} kN")
            ax2.set_xlabel("time (s)")
            ax2.set_ylabel("contact force (kN)")
            ax2.legend(fontsize=8)
            try:
                style.beautify_axes(ax2)
            except Exception:                          # noqa: BLE001
                pass

        canvas = Canvas(fig)
        canvas.setMinimumHeight(300)
        root.addWidget(canvas, 1)

        hint = QLabel("DAF = peak dynamic / peak static response — the impact "
                      "the static moving-load case cannot capture.")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        root.addWidget(hint)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        root.addWidget(btns)
        style.apply(self)

    @classmethod
    def show_results(cls, parent, times, dynamic, static, daf, **kw):
        dlg = cls(parent, times, dynamic, static, daf, **kw)
        dlg.show()
        return dlg
