"""Post-tensioning **tendon** manager + editor (construction-stage parity C6).

``TendonManagerDialog`` lists / adds / edits / deletes the project's tendons;
``TendonDialog`` edits one: its node path with per-node eccentricity, strand
area, jacking force, type and post-tension loss parameters. A tendon is
*stressed* at a construction stage that lists its id (the stage manager's
"Tendons stressed this stage" selector); the runner lowers it to equivalent
nodal loads via :class:`femsolver.bridges.Tendon`.

Headless-constructible under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

import copy

from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog,
                               QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

import style
from analysis_ui import GroupCard, dialog_buttons
from project import Tendon


def _next_id(ids) -> int:
    return (max(ids) + 1) if ids else 1


class TendonDialog(QDialog):
    """Edit one :class:`project.Tendon`."""

    def __init__(self, parent, project, tendon=None):
        super().__init__(parent)
        self.setWindowTitle("Edit tendon" if tendon else "Add tendon")
        self._project = project
        self.resize(520, 520)
        v = QVBoxLayout(self)
        v.setContentsMargins(style.SP_LG, style.SP_LG, style.SP_LG, style.SP_LG)
        v.setSpacing(style.SP_MD)

        ident = GroupCard("Identity")
        self.id_spin = QSpinBox()
        self.id_spin.setRange(1, 10_000_000)
        self.id_spin.setValue(tendon.id if tendon
                              else _next_id([t.id for t in project.tendons]))
        self.id_spin.setEnabled(tendon is None)
        ident.add_row("Tendon id", self.id_spin)
        self.name = QLineEdit(tendon.name if tendon else "Tendon")
        ident.add_row("Name", self.name)
        self.kind = QComboBox()
        self.kind.addItems(["post-tension", "pre-tension"])
        if tendon:
            i = self.kind.findText(tendon.tendon_type)
            if i >= 0:
                self.kind.setCurrentIndex(i)
        ident.add_row("Type", self.kind)
        v.addWidget(ident)

        # ---- node path + eccentricity table ----
        path = GroupCard("Path (node → eccentricity)", form=False)
        hint = QLabel("Node ids the tendon passes through (joined by members) "
                      "and its eccentricity at each (m; − = below centroid).")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        path.body_layout().addWidget(hint)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["node id", "ecc [m]"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        path.body_layout().addWidget(self.table)
        row = QHBoxLayout()
        for label, cb in (("Add node", self._add_row),
                          ("Remove node", self._del_row)):
            b = QPushButton(label)
            b.clicked.connect(cb)
            row.addWidget(b)
        row.addStretch(1)
        path.body_layout().addLayout(row)
        v.addWidget(path, 1)
        if tendon and tendon.nodes:
            for nd, e in zip(tendon.nodes, tendon.ecc):
                self._add_row(int(nd), float(e))
        else:
            self._add_row(); self._add_row()

        # ---- force + losses ----
        fc = GroupCard("Force & losses")
        self.area = QDoubleSpinBox()
        self.area.setRange(1e-6, 1.0)
        self.area.setDecimals(6)
        self.area.setValue(tendon.area if tendon else 1.4e-3)
        fc.add_row("Strand area [m²]", self.area)
        self.pj = QDoubleSpinBox()
        self.pj.setRange(1.0, 1e12)
        self.pj.setDecimals(0)
        self.pj.setSingleStep(1e5)
        self.pj.setValue(tendon.jacking_force if tendon else 1.0e6)
        fc.add_row("Jacking force [N]", self.pj)
        self.mu = QDoubleSpinBox()
        self.mu.setRange(0.0, 1.0)
        self.mu.setDecimals(3)
        self.mu.setValue(tendon.mu if tendon else 0.20)
        fc.add_row("Friction μ", self.mu)
        self.wk = QDoubleSpinBox()
        self.wk.setRange(0.0, 1.0)
        self.wk.setDecimals(5)
        self.wk.setValue(tendon.wobble_k if tendon else 0.0066)
        fc.add_row("Wobble k [1/m]", self.wk)
        self.slip = QDoubleSpinBox()
        self.slip.setRange(0.0, 0.1)
        self.slip.setDecimals(4)
        self.slip.setValue(tendon.anchor_slip if tendon else 0.0)
        fc.add_row("Anchor slip [m]", self.slip)
        v.addWidget(fc)

        v.addWidget(dialog_buttons(self))
        style.apply(self)

    def _add_row(self, node=None, ecc=0.0):
        r = self.table.rowCount()
        self.table.insertRow(r)
        ns = QSpinBox()
        ns.setRange(1, 10_000_000)
        if node:
            ns.setValue(int(node))
        self.table.setCellWidget(r, 0, ns)
        es = QDoubleSpinBox()
        es.setRange(-100.0, 100.0)
        es.setDecimals(4)
        es.setSingleStep(0.05)
        es.setValue(float(ecc))
        self.table.setCellWidget(r, 1, es)

    def _del_row(self):
        r = self.table.currentRow()
        if r < 0:
            r = self.table.rowCount() - 1
        if r >= 0:
            self.table.removeRow(r)

    def data(self) -> Tendon:
        nodes, ecc = [], []
        for r in range(self.table.rowCount()):
            nodes.append(int(self.table.cellWidget(r, 0).value()))
            ecc.append(float(self.table.cellWidget(r, 1).value()))
        return Tendon(
            id=self.id_spin.value(), name=self.name.text(),
            nodes=nodes, ecc=ecc, area=float(self.area.value()),
            jacking_force=float(self.pj.value()),
            tendon_type=self.kind.currentText(), mu=float(self.mu.value()),
            wobble_k=float(self.wk.value()), anchor_slip=float(self.slip.value()))

    @classmethod
    def edit(cls, parent, project, tendon=None):
        dlg = cls(parent, project, tendon)
        return dlg.data() if dlg.exec() else None


class TendonManagerDialog(QDialog):
    """List / add / edit / delete the project's post-tensioning tendons."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Tendons")
        self._project = project
        self._tendons = copy.deepcopy(project.tendons)
        v = QVBoxLayout(self)
        v.setContentsMargins(style.SP_LG, style.SP_LG, style.SP_LG, style.SP_LG)
        v.setSpacing(style.SP_SM)
        head = QLabel("Post-tensioning tendons")
        head.setObjectName("h2")
        v.addWidget(head)
        sub = QLabel("Define tendons here, then stress each one at a "
                     "construction stage (Construction stages ▸ Tendons "
                     "stressed this stage).")
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        v.addWidget(sub)

        body = QHBoxLayout()
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["id", "name", "nodes", "Pj [kN]"])
        self.table.setMinimumSize(440, 240)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.doubleClicked.connect(lambda *_: self._edit())
        body.addWidget(self.table, 1)
        col = QVBoxLayout()
        for label, cb in (("Add…", self._add), ("Edit…", self._edit),
                          ("Delete", self._delete)):
            b = QPushButton(label)
            b.clicked.connect(cb)
            col.addWidget(b)
        col.addStretch(1)
        body.addLayout(col)
        v.addLayout(body)

        v.addWidget(dialog_buttons(self))
        style.apply(self)
        self._refresh()

    def _refresh(self):
        self.table.setRowCount(len(self._tendons))
        for r, t in enumerate(self._tendons):
            self.table.setItem(r, 0, QTableWidgetItem(str(t.id)))
            self.table.setItem(r, 1, QTableWidgetItem(t.name))
            self.table.setItem(r, 2, QTableWidgetItem(
                "→".join(str(n) for n in t.nodes)))
            self.table.setItem(r, 3, QTableWidgetItem(
                f"{t.jacking_force / 1e3:.0f}"))

    def _proxy(self):
        proxy = copy.copy(self._project)
        proxy.tendons = self._tendons
        return proxy

    def _add(self):
        t = TendonDialog.edit(self, self._proxy())
        if t is None:
            return
        if any(x.id == t.id for x in self._tendons):
            QMessageBox.warning(self, "Duplicate",
                                f"Tendon {t.id} already exists.")
            return
        self._tendons.append(t)
        self._refresh()

    def _edit(self):
        r = self.table.currentRow()
        if r < 0:
            return
        t = TendonDialog.edit(self, self._proxy(), self._tendons[r])
        if t is not None:
            self._tendons[r] = t
            self._refresh()

    def _delete(self):
        r = self.table.currentRow()
        if 0 <= r < len(self._tendons):
            del self._tendons[r]
            self._refresh()

    def result_tendons(self) -> list:
        return self._tendons

    @classmethod
    def manage(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.result_tendons() if dlg.exec() else None
