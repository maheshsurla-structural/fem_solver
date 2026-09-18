"""Load-combination editor (plan L2) — the surface commercial tools always have
and ours was missing.

A :class:`project.LoadCombination` is a weighted sum of load patterns
(``factors`` maps pattern id → factor). Before this dialog, combinations could
only be produced wholesale by ``Project.generate_asce7_combinations`` and edited
by deletion. Here the user gets the familiar CSiBridge / ETABS *Define
Combinations* layout: a list of combinations on the left, and for the selected
one a **factor grid over every load pattern** on the right — plus a one-click
"Generate ASCE 7-22 LRFD" that folds in the existing generator.

Built from the L1 scaffold (:mod:`analysis_ui`). Pure Qt, headless-constructible
under ``QT_QPA_PLATFORM=offscreen``; :meth:`manage` runs it modally and returns
the edited ``list[LoadCombination]`` (or ``None`` if cancelled).
"""
from __future__ import annotations

import copy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDoubleSpinBox, QFormLayout,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QListWidget, QMessageBox, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

import analysis_ui as ui
import style
from project import NATURE_LABELS, LoadCombination


class CombinationsDialog(QDialog):
    """Add / rename / delete load combinations and set each pattern's factor."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Load combinations")
        self._project = project
        self._combos = copy.deepcopy(project.combinations)
        self._cases = list(project.load_cases)
        self._case_ids = [c.id for c in self._cases]
        self._cur = -1                         # index of the combo being edited
        self.result_combos = None
        self.resize(760, 480)

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_MD)
        body = QHBoxLayout()
        body.setSpacing(style.SP_MD)

        # ---- left: the list of combinations + actions ----
        left = ui.GroupCard("Combinations", form=False)
        self.listw = QListWidget()
        self.listw.currentRowChanged.connect(self._on_select)
        left.body_layout().addWidget(self.listw, 1)
        act = QHBoxLayout()
        add = QPushButton("＋ Combo")
        add.clicked.connect(self._add)
        self._del_btn = QPushButton("Delete")
        self._del_btn.clicked.connect(self._delete)
        act.addWidget(add)
        act.addWidget(self._del_btn)
        left.body_layout().addLayout(act)
        gen = QPushButton("Generate ASCE 7-22 LRFD…")
        gen.setIcon(_icon("loadsgen"))
        gen.clicked.connect(self._generate)
        left.body_layout().addWidget(gen)
        body.addWidget(left, 2)

        # ---- right: the selected combination's definition ----
        right = ui.GroupCard("Definition", form=False)
        name_host = QWidget()
        nf = QFormLayout(name_host)
        nf.setContentsMargins(0, 0, 0, 0)
        self.name = QLineEdit()
        self.name.textEdited.connect(self._on_name_edited)
        nf.addRow("Name", self.name)
        right.body_layout().addWidget(name_host)

        self.grid = QTableWidget(0, 3)
        self.grid.setHorizontalHeaderLabels(["Load pattern", "Nature",
                                             "Scale factor"])
        self.grid.verticalHeader().setVisible(False)
        hdr = self.grid.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._spins: list[QDoubleSpinBox] = []
        self._build_grid()
        right.body_layout().addWidget(self.grid, 1)
        hint = QLabel("A pattern with factor 0 is not part of the combination. "
                      "Run Design to envelope all combinations.")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        right.body_layout().addWidget(hint)
        body.addWidget(right, 3)

        root.addLayout(body, 1)
        root.addWidget(ui.dialog_buttons(self))
        style.apply(self)

        self._reload_list()
        if self._combos:
            self.listw.setCurrentRow(0)
        else:
            self._set_editor_enabled(False)

    # ---- grid (one row per load pattern, a factor spin in the last column) ----
    def _build_grid(self) -> None:
        self.grid.setRowCount(len(self._cases))
        readonly = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        for r, c in enumerate(self._cases):
            name_it = QTableWidgetItem(c.name)
            name_it.setFlags(readonly)                      # not editable
            self.grid.setItem(r, 0, name_it)
            nat_it = QTableWidgetItem(NATURE_LABELS.get(c.nature, c.nature))
            nat_it.setFlags(readonly)
            self.grid.setItem(r, 1, nat_it)
            spin = QDoubleSpinBox()
            spin.setRange(-100.0, 100.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.05)
            self.grid.setCellWidget(r, 2, spin)
            self._spins.append(spin)

    # ---- list <-> editor plumbing ----
    def _reload_list(self) -> None:
        self.listw.blockSignals(True)
        self.listw.clear()
        for c in self._combos:
            self.listw.addItem(c.name)
        self.listw.blockSignals(False)

    def _on_select(self, row: int) -> None:
        self._sync_current()                    # commit the combo we're leaving
        self._cur = row
        if not (0 <= row < len(self._combos)):
            self._set_editor_enabled(False)
            return
        self._set_editor_enabled(True)
        combo = self._combos[row]
        self.name.setText(combo.name)
        for cid, spin in zip(self._case_ids, self._spins):
            spin.blockSignals(True)
            spin.setValue(float(combo.factors.get(cid, 0.0)))
            spin.blockSignals(False)

    def _sync_current(self) -> None:
        if not (0 <= self._cur < len(self._combos)):
            return
        combo = self._combos[self._cur]
        combo.name = self.name.text().strip() or combo.name
        factors = {}
        for cid, spin in zip(self._case_ids, self._spins):
            v = float(spin.value())
            if abs(v) > 1e-12:
                factors[cid] = v
        combo.factors = factors

    def _set_editor_enabled(self, on: bool) -> None:
        self.name.setEnabled(on)
        self.grid.setEnabled(on)
        self._del_btn.setEnabled(on)
        if not on:
            self.name.blockSignals(True)
            self.name.clear()
            self.name.blockSignals(False)
            for spin in self._spins:
                spin.blockSignals(True)
                spin.setValue(0.0)
                spin.blockSignals(False)

    def _on_name_edited(self, text: str) -> None:
        if 0 <= self._cur < self.listw.count():
            self.listw.item(self._cur).setText(text)

    # ---- actions ----
    def _add(self) -> None:
        self._sync_current()
        nid = max((c.id for c in self._combos), default=0) + 1
        combo = LoadCombination(id=nid, name=f"Combo {len(self._combos) + 1}",
                                factors={})
        self._combos.append(combo)
        self._reload_list()
        self.listw.setCurrentRow(len(self._combos) - 1)

    def _delete(self) -> None:
        r = self.listw.currentRow()
        if not (0 <= r < len(self._combos)):
            return
        del self._combos[r]
        self._cur = -1                          # the row we edited is gone
        self._reload_list()
        if self._combos:
            self.listw.setCurrentRow(min(r, len(self._combos) - 1))
        else:
            self._set_editor_enabled(False)

    def _append_generated(self) -> tuple[int, int]:
        """Append the ASCE 7-22 LRFD set (deduped by name) using a proxy project
        so generated ids continue from the working list, not the saved one.
        UI-free (no modal) so it is unit-testable; returns
        ``(n_added, n_generated)``."""
        self._sync_current()
        proxy = copy.copy(self._project)
        proxy.combinations = self._combos
        generated = proxy.generate_asce7_combinations()
        have = {c.name for c in self._combos}
        added = [c for c in generated if c.name not in have]
        self._combos.extend(added)
        self._reload_list()
        if added:
            self.listw.setCurrentRow(len(self._combos) - 1)
        return len(added), len(generated)

    def _generate(self) -> None:
        n_added, n_gen = self._append_generated()
        if n_gen == 0:
            QMessageBox.information(
                self, "Generate combinations",
                "No combinations generated — give your load patterns natures "
                "(Dead / Live / Wind …) in Analysis ▸ Load patterns first.")
        else:
            QMessageBox.information(
                self, "Generate combinations",
                f"Added {n_added} ASCE 7-22 LRFD combination(s)."
                + ("" if n_added else " (all were already present.)"))

    def accept(self) -> None:
        self._sync_current()
        self.result_combos = self._combos
        super().accept()

    @classmethod
    def manage(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.result_combos if dlg.exec() else None


def _icon(name: str):
    try:
        import icons
        return icons.icon(name, style.ICON)
    except Exception:                                          # noqa: BLE001
        from PySide6.QtGui import QIcon
        return QIcon()
