"""Temperature-gradient results — the section self-equilibrated stress diagram
plus the structure-level summary (deflection, continuity moment).

Fed the governing section's :class:`TemperatureGradient` +
:class:`SectionThermalActions` (from the engine reduction) and a summary of the
solved frame effects. Non-modal. Pure Qt + matplotlib (Agg) —
headless-constructible under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

import style


class TemperatureGradientResultsDialog(QDialog):
    """Self-stress diagram + gradient summary."""

    def __init__(self, parent, gradient, actions, height, *,
                 max_deflection=0.0, max_moment=0.0, section_label=""):
        super().__init__(parent)
        self.setWindowTitle("Temperature-gradient results")
        self.resize(560, 480)

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_SM)

        head = QLabel("Temperature-gradient response")
        head.setObjectName("h2")
        root.addWidget(head)
        sub = QLabel(
            f"Equivalent ΔT_uniform = {actions.dT_uniform:.2f} °C   ·   "
            f"gradient ≈ {actions.dT_gradient_equiv:.2f} °C   ·   "
            f"self-stress top/bottom = {actions.self_stress_top / 1e6:.2f} / "
            f"{actions.self_stress_bottom / 1e6:.2f} MPa")
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        root.addWidget(sub)
        sub2 = QLabel(
            f"Max deflection = {max_deflection * 1e3:.2f} mm   ·   "
            f"max continuity moment = {max_moment / 1e3:.2f} kN·m"
            + (f"   ·   {section_label}" if section_label else ""))
        sub2.setObjectName("sub")
        root.addWidget(sub2)

        fig = Figure(figsize=(5.4, 3.6), layout="constrained")
        ax = fig.add_subplot(111)
        d = np.linspace(0.0, height, 200)
        ax.plot(gradient.T(d), -d, color=style.C_SECONDARY, lw=2,
                label="temperature (°C)")
        ax.set_xlabel("temperature (°C)", color=style.C_SECONDARY)
        ax.tick_params(axis="x", labelcolor=style.C_SECONDARY)
        ax.set_ylabel("depth below top (m)")
        ax2 = ax.twiny()
        ax2.plot(actions.self_stress(d) / 1e6, -d, color=style.C_PRIMARY, lw=2,
                 label="self-stress (MPa)")
        ax2.axvline(0.0, color=style.AX_SPINE, lw=0.8)
        ax2.set_xlabel("self-equilibrated stress (MPa)", color=style.C_PRIMARY)
        ax2.tick_params(axis="x", labelcolor=style.C_PRIMARY)
        try:
            style.beautify_axes(ax)
        except Exception:                              # noqa: BLE001
            pass
        canvas = Canvas(fig)
        canvas.setMinimumHeight(280)
        root.addWidget(canvas, 1)

        hint = QLabel("Self-stress is present even in a simple span (it "
                      "integrates to zero axial force and moment); continuity "
                      "moments arise only in continuous spans.")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        root.addWidget(hint)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        root.addWidget(btns)
        style.apply(self)

    @classmethod
    def show_results(cls, parent, gradient, actions, height, **kw):
        dlg = cls(parent, gradient, actions, height, **kw)
        dlg.show()
        return dlg
