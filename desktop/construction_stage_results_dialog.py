"""Construction-stage results — camber / geometry control **and** per-stage
member forces (bridge GUI plan G3; construction-stage parity C7).

Fed a :class:`femsolver.bridges.StagedCamber` it draws the deflection building
up stage by stage and the required build-high **camber**. When also given the
:class:`femsolver.bridges.IncrementalStagedResult` and the model, it adds a
**Stage forces** tab: a stage selector over a table of each element's status,
axial force, peak moment, and applied stiffness (creep) factor at the end of the
selected stage — the stage-by-stage force table an engineer checks against a
commercial tool. Non-modal. Pure Qt + matplotlib (Agg) — headless-constructible
under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog,
                               QDialogButtonBox, QHBoxLayout, QLabel,
                               QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

import style
from units import Quantity, UnitSystem


def _local_axial_moment(el, f_global):
    """Return ``(N, |M|max)`` (SI) from an element's global end-force vector,
    or ``None`` when the element type can't be reduced (no transform)."""
    T = getattr(el, "transform_matrix", None)
    if T is None:
        return None
    fl = np.asarray(T()) @ np.asarray(f_global, dtype=float)
    if fl.size == 4:                     # 2-D truss: [u1,v1,u2,v2]
        return float(fl[2]), 0.0
    if fl.size == 6:                     # 2-D beam: [u,v,θ]×2, N=+x@node j
        return float(fl[3]), float(max(abs(fl[2]), abs(fl[5])))
    if fl.size == 12:                    # 3-D beam: axial@6, My/Mz@4,5,10,11
        return float(fl[6]), float(max(abs(fl[4]), abs(fl[5]),
                                       abs(fl[10]), abs(fl[11])))
    return None


class ConstructionStageResultsDialog(QDialog):
    """Camber diagram + (optionally) a per-stage member-force table."""

    def __init__(self, parent, camber, *, n_stages=0, unitsys=None,
                 result=None, model=None, stage_names=None):
        super().__init__(parent)
        self.setWindowTitle("Construction-stage results")
        self.resize(660, 520)
        self._us = unitsys or UnitSystem()
        self._camber = camber
        self._result = result
        self._model = model
        self._stage_names = list(stage_names) if stage_names else (
            list(result.stage_names) if result is not None else [])

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_SM)

        tabs = QTabWidget()
        tabs.addTab(self._build_camber_tab(n_stages), "Camber")
        if result is not None and model is not None:
            tabs.addTab(self._build_stage_forces_tab(), "Stage forces")
        root.addWidget(tabs, 1)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        root.addWidget(btns)
        style.apply(self)

    # ------------------------------------------------------------- camber tab
    def _build_camber_tab(self, n_stages) -> QWidget:
        us, L = self._us, Quantity.LENGTH
        l_u = us.label(L)
        camber = self._camber
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(style.SP_SM)

        head = QLabel("Camber / geometry control")
        head.setObjectName("h2")
        v.addWidget(head)
        dmax = float(np.max(np.abs(camber.final_deflection))) if len(
            camber.final_deflection) else 0.0
        dmax_d = us.to_display(dmax, L)
        sub = QLabel(f"{n_stages} stages · max final deflection = "
                     f"{dmax_d:.4g} {l_u} · required camber = build "
                     f"{dmax_d:.4g} {l_u} high")
        sub.setObjectName("sub")
        v.addWidget(sub)

        def _disp(arr):
            return [us.to_display(val, L) for val in np.asarray(arr)]

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
        canvas.setMinimumHeight(260)
        v.addWidget(canvas, 1)

        hint = QLabel("Build each node the 'camber' amount high so the finished "
                      "structure settles onto the target profile.")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        v.addWidget(hint)
        return w

    # -------------------------------------------------------- stage-forces tab
    def _build_stage_forces_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(style.SP_SM)

        head = QLabel("Member forces by construction stage")
        head.setObjectName("h2")
        v.addWidget(head)

        sel = QHBoxLayout()
        sel.addWidget(QLabel("Stage:"))
        self.stage_combo = QComboBox()
        for i, nm in enumerate(self._stage_names):
            self.stage_combo.addItem(f"{i + 1}. {nm}", i)
        self.stage_combo.currentIndexChanged.connect(self._fill_stage_table)
        sel.addWidget(self.stage_combo, 1)
        v.addLayout(sel)

        # creep factors present and non-trivial → show the E-factor column
        self._show_creep = bool(getattr(self._result, "creep_factors", None)) \
            and any(abs(f - 1.0) > 1e-9
                    for st in self._result.creep_factors for f in st.values())
        us = self._us
        cols = ["Element", "Status",
                f"Axial N ({us.label(Quantity.FORCE)})",
                f"|M|max ({us.label(Quantity.MOMENT)})"]
        if self._show_creep:
            cols.append("E-factor")
        self.table = QTableWidget(0, len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.table, 1)

        self.stage_summary = QLabel("")
        self.stage_summary.setObjectName("hintLabel")
        self.stage_summary.setWordWrap(True)
        v.addWidget(self.stage_summary)

        if self._stage_names:
            self.stage_combo.setCurrentIndex(len(self._stage_names) - 1)
        self._fill_stage_table()
        return w

    def _fill_stage_table(self, *_):
        res, m, us = self._result, self._model, self._us
        k = self.stage_combo.currentData()
        if k is None:
            return
        hist = res.element_force_history
        tags = sorted(hist)
        self.table.setRowCount(len(tags))
        peakN = peakM = 0.0
        n_active = 0
        for r, tag in enumerate(tags):
            fh = hist[tag][k] if k < len(hist[tag]) else None
            active = fh is not None
            self.table.setItem(r, 0, QTableWidgetItem(str(tag)))
            self.table.setItem(r, 1, QTableWidgetItem(
                "active" if active else "—"))
            nm = None
            if active:
                n_active += 1
                try:
                    nm = _local_axial_moment(m.element(tag), fh)
                except Exception:                      # noqa: BLE001
                    nm = None
            if nm is not None:
                N, Mmax = nm
                peakN = max(peakN, abs(N))
                peakM = max(peakM, Mmax)
                self.table.setItem(r, 2, QTableWidgetItem(
                    f"{us.to_display(N, Quantity.FORCE):.4g}"))
                self.table.setItem(r, 3, QTableWidgetItem(
                    f"{us.to_display(Mmax, Quantity.MOMENT):.4g}"))
            else:
                self.table.setItem(r, 2, QTableWidgetItem("—"))
                self.table.setItem(r, 3, QTableWidgetItem("—"))
            if self._show_creep:
                f = res.creep_factors[k].get(tag) if k < len(
                    res.creep_factors) else None
                self.table.setItem(r, 4, QTableWidgetItem(
                    f"{f:.3f}" if f is not None else "—"))
        self.stage_summary.setText(
            f"{n_active} of {len(tags)} elements active · peak axial "
            f"{us.to_display(peakN, Quantity.FORCE):.4g} "
            f"{us.label(Quantity.FORCE)} · peak moment "
            f"{us.to_display(peakM, Quantity.MOMENT):.4g} "
            f"{us.label(Quantity.MOMENT)}")

    @classmethod
    def show_results(cls, parent, camber, **kw):
        dlg = cls(parent, camber, **kw)
        dlg.show()
        return dlg
