"""Wall design dialog (wall plan W3).

Runs the ACI 318-19 §18.10 special-wall check (``femsolver.design.walls``) for
each pier, using the integrated pier demand P/V/M from the shell membrane
stresses (wall plan W2) and a small reinforcement/material input form. Reports
the governing demand/capacity ratio, the boundary-element trigger and minimum-
reinforcement status per cut, per pier.

Strengths are entered in MPa and reinforcement areas in mm² — the idiom wall
engineers use — and converted to base SI for the engine.
"""
from __future__ import annotations

from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QComboBox, QHBoxLayout, QHeaderView,
                               QLabel, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

import piers
import style
from femsolver.design.walls import (WallDemand, WallGeometry, WallMaterial,
                                     WallReinforcement, design_wall_pier)


def _spin(value, lo, hi, dec=3, step=0.1, suffix=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(value)
    if suffix:
        s.setSuffix(suffix)
    return s


class WallDesignDialog(QDialog):
    """Per-pier ACI 318-19 §18.10 wall design against the integrated demand."""

    def __init__(self, parent, project, model):
        super().__init__(parent)
        self.setWindowTitle("Wall design (ACI 318-19 §18.10)")
        self.resize(680, 640)
        self._project = project
        self._model = model
        self._piers = project.pier_names()

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_SM)

        head = QLabel("Wall design — ACI 318-19 §18.10")
        head.setObjectName("h2")
        root.addWidget(head)

        # --- pier + material/reinforcement inputs ---
        top = QHBoxLayout()
        top.addWidget(QLabel("Pier"))
        self.pier = QComboBox()
        for name in self._piers:
            self.pier.addItem(name, name)
        self.pier.currentIndexChanged.connect(self._on_pier_changed)
        top.addWidget(self.pier)
        top.addStretch(1)
        root.addLayout(top)

        form = QFormLayout()
        self.fc = _spin(30.0, 10.0, 150.0, 1, 5.0, " MPa")
        self.fy = _spin(420.0, 200.0, 700.0, 0, 10.0, " MPa")
        self.rho_l = _spin(0.0025, 0.0, 0.05, 4, 0.0005)
        self.rho_t = _spin(0.0025, 0.0, 0.05, 4, 0.0005)
        self.as_be = _spin(2000.0, 0.0, 100000.0, 0, 500.0, " mm²")
        self.d_be = _spin(200.0, 0.0, 2000.0, 0, 50.0, " mm")
        form.addRow("f'c", self.fc)
        form.addRow("fy (rebar)", self.fy)
        form.addRow("ρℓ (vertical web)", self.rho_l)
        form.addRow("ρt (horizontal web)", self.rho_t)
        form.addRow("Boundary bars As (each end)", self.as_be)
        form.addRow("Boundary bar depth from end", self.d_be)
        fw = QWidget()
        fw.setLayout(form)
        root.addWidget(fw)

        run = QPushButton("Check")
        run.clicked.connect(self._compute)
        root.addWidget(run)

        self.summary = QLabel("")
        self.summary.setObjectName("sub")
        self.summary.setWordWrap(True)
        root.addWidget(self.summary)

        self.tbl = QTableWidget(0, 7)
        self.tbl.setHorizontalHeaderLabels(
            ["Elev (m)", "P (kN, +C)", "M (kN·m)", "V (kN)", "DCR P-M",
             "DCR V", "Boundary"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setEditTriggers(self.tbl.EditTrigger.NoEditTriggers)
        self.tbl.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.tbl, 1)

        self.notes = QLabel("")
        self.notes.setObjectName("hintLabel")
        self.notes.setWordWrap(True)
        root.addWidget(self.notes)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        root.addWidget(btns)
        style.apply(self)

        self._prefill_material()
        if self._piers:
            self._compute()
        else:
            self.summary.setText("No piers defined — draw a wall with a pier "
                                 "label first.")

    def _on_pier_changed(self) -> None:
        self._prefill_material()
        self._compute()

    def _prefill_material(self) -> None:
        """Seed f'c from the first wall area's material params if it carries a
        concrete ``fc`` (Pa → MPa)."""
        name = self.pier.currentData()
        if name is None:
            return
        ids = piers.pier_area_ids(self._project, name)
        for a in self._project.areas:
            if a.id in ids:
                mat = next((m for m in self._project.materials
                            if m.id == a.material), None)
                if mat and mat.params.get("fc"):
                    self.fc.setValue(float(mat.params["fc"]) / 1e6)
                break

    def _inputs(self):
        mat = WallMaterial(fc=self.fc.value() * 1e6, fy=self.fy.value() * 1e6)
        reinf = WallReinforcement(
            rho_l=self.rho_l.value(), rho_t=self.rho_t.value(),
            As_boundary=self.as_be.value() * 1e-6,     # mm² → m²
            d_boundary=self.d_be.value() * 1e-3)       # mm → m
        return mat, reinf

    def _compute(self) -> None:
        name = self.pier.currentData()
        if name is None:
            return
        geom_t = piers.pier_geometry(self._project, self._model, name)
        forces = piers.pier_forces(self._project, self._model, name)
        if geom_t is None or not forces:
            self.summary.setText("No wall elements for this pier.")
            self.tbl.setRowCount(0)
            return
        lw, t, hw = geom_t
        mat, reinf = self._inputs()

        rows = sorted(forces, key=lambda f: f.z, reverse=True)  # top → base
        self.tbl.setRowCount(len(rows))
        worst = None
        for r, f in enumerate(rows):
            geom = WallGeometry(lw=lw, t=t, hw=hw)
            demand = WallDemand(Pu=-f.axial, Mu=f.moment, Vu=f.shear)  # +C
            res = design_wall_pier(geom, mat, reinf, demand)
            if worst is None or res.dcr > worst[1].dcr:
                worst = (f, res)
            cells = [f"{f.z:.3g}", f"{-f.axial / 1e3:.1f}",
                     f"{f.moment / 1e3:.1f}", f"{f.shear / 1e3:.1f}",
                     f"{res.pm.dcr:.2f}", f"{res.shear.dcr:.2f}",
                     "required" if res.boundary.required else "—"]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(text)
                if c in (4, 5) and float(text) > 1.0:
                    it.setForeground(QBrush(QColor("#c0392b")))
                self.tbl.setItem(r, c, it)

        f, res = worst
        verdict = "PASS" if res.ok else "FAIL"
        color = style.C_PRIMARY if res.ok else "#c0392b"
        self.summary.setText(
            f"<b>Pier {name}</b> — ℓw = {lw:.2f} m, t = {t*1e3:.0f} mm, "
            f"hw = {hw:.2f} m · governing DCR = <b>{res.dcr:.2f}</b> at "
            f"elev {f.z:.2f} m · <b style='color:{color}'>{verdict}</b>")
        notes = list(res.notes)
        if res.shear.two_curtains_required and \
                "Two curtains of reinforcement required." not in notes:
            notes.append("Two curtains of reinforcement required.")
        self.notes.setText("  ·  ".join(notes) if notes
                           else "No special detailing flags at the governing cut.")

    @classmethod
    def show_results(cls, parent, project, model):
        dlg = cls(parent, project, model)
        dlg.show()
        return dlg
