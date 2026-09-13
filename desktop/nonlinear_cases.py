"""Nonlinear (fiber) analysis case manager (GUI-4, plan §14 stage 4).

A :class:`project.NonlinearCase` is a saved nonlinear static load case — the
desktop counterpart of a Midas/CSI nonlinear case: control node/DOF, target,
protocol (monotonic ramp or reversed-cyclic), a held axial preload, an optional
``continue_from`` (staged continuation from another case's committed state), and
the Newton controls (tolerance, max iterations).

* :class:`NonlinearCaseDialog` edits one case (protocol-dependent fields toggle).
* :class:`NonlinearCaseManagerDialog` is the list / add / edit / delete manager.

Cases are run by ``nonlinear.run_case`` (wired into the pushover dialog's Case
selector). Pure Qt — headless-constructible under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

import copy

from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QSpinBox,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

import analysis_ui as ui
import style
from project import NonlinearCase

_DOFS = [("Ux", 0), ("Uy", 1), ("Rz", 2)]
_PROTOCOLS = [("Monotonic", "monotonic"), ("Cyclic", "cyclic")]


def _next_id(ids) -> int:
    return (max(ids) + 1) if ids else 1


def _combo(entries, default=None) -> QComboBox:
    c = QComboBox()
    for label, value in entries:
        c.addItem(label, value)
    if default is not None:
        i = c.findData(default)
        if i >= 0:
            c.setCurrentIndex(i)
    return c


def _parse_amplitudes(text: str, fallback) -> list:
    out = []
    for tok in text.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out or list(fallback)


class NonlinearCaseDialog(QDialog):
    """Edit one nonlinear (fiber) analysis case, laid out as CSiBridge-style
    *Load Case Data* panels (plan A2): a Name / Notes / Type header over a
    two-column grid of **Control**, **Protocol**, **Initial conditions** and
    **Solver** cards. The monotonic vs cyclic fields swap in place inside the
    Protocol card (``mono_host`` / ``cyc_host``). Built entirely from the L1
    scaffold (:mod:`analysis_ui`); pure Qt, headless-constructible.
    """

    def __init__(self, parent, project, case=None):
        super().__init__(parent)
        self.setWindowTitle("Edit nonlinear case" if case else
                            "Add nonlinear case")
        self._project = project
        node_ids = [n.id for n in project.nodes]
        free = [n.id for n in project.nodes
                if not (n.supports and any(n.supports))]
        default_node = (free[-1] if free else
                        (node_ids[-1] if node_ids else 0))
        lu, fu = project.length_unit, project.force_unit

        # ---- header: Name / Notes / Type ----
        self.header = ui.CaseHeader(
            name=(case.name if case else "Pushover"),
            type_label="Nonlinear Static",
            notes=(getattr(case, "notes", "") if case else ""))

        # ---- Control ----
        control = ui.GroupCard("Control")
        self.id_spin = QSpinBox()
        self.id_spin.setRange(1, 10_000_000)
        self.id_spin.setValue(case.id if case
                              else _next_id([c.id for c in
                                             project.nonlinear_cases]))
        self.id_spin.setEnabled(case is None)
        self.node = _combo([(str(i), i) for i in node_ids],
                           default=(case.control_node if case else default_node))
        self.dof = _combo(_DOFS, default=(case.control_dof if case else 1))
        self.dof.currentIndexChanged.connect(self._on_dof)
        self.target = self._spin(case.target if case else 0.05, decimals=4)
        control.add_row("Case id", self.id_spin)
        control.add_row("Control node", self.node)
        control.add_row("Control DOF", self.dof)
        control.add_row(f"Target [{lu}]", self.target)
        self._dir_hint = ui.direction_glyph(self.dof.currentText())
        control.add_full_row(self._dir_hint)

        # ---- Protocol (monotonic <-> cyclic fields swap in place) ----
        protocol = ui.GroupCard("Protocol")
        self.protocol = _combo(_PROTOCOLS,
                               default=(case.protocol if case else "monotonic"))
        self.protocol.currentIndexChanged.connect(self._on_protocol)
        protocol.add_row("Protocol", self.protocol)

        self.mono_host = QWidget()
        mono = QFormLayout(self.mono_host)
        mono.setContentsMargins(0, 0, 0, 0)
        self.n_steps = QSpinBox()
        self.n_steps.setRange(2, 5000)
        self.n_steps.setValue(case.n_steps if case else 40)
        mono.addRow("Steps", self.n_steps)
        protocol.add_full_row(self.mono_host)

        self.cyc_host = QWidget()
        cyc = QFormLayout(self.cyc_host)
        cyc.setContentsMargins(0, 0, 0, 0)
        amps = case.amplitudes if case else [0.25, 0.5, 0.75, 1.0]
        self.amplitudes = QLineEdit(", ".join(f"{a:g}" for a in amps))
        self.cycles = QSpinBox()
        self.cycles.setRange(1, 100)
        self.cycles.setValue(case.cycles if case else 1)
        self.pts = QSpinBox()
        self.pts.setRange(4, 2000)
        self.pts.setValue(case.pts_per_cycle if case else 40)
        cyc.addRow("Amplitudes (× target)", self.amplitudes)
        cyc.addRow("Cycles / amplitude", self.cycles)
        cyc.addRow("Points / cycle", self.pts)
        protocol.add_full_row(self.cyc_host)

        # ---- Initial conditions (staged continuation + held axial preload) ----
        init = ui.GroupCard("Initial conditions")
        others = [(f"{c.id}: {c.name}", c.id) for c in project.nonlinear_cases
                  if not case or c.id != case.id]
        self.continue_from = _combo([("— none —", None)] + others,
                                    default=(case.continue_from if case
                                             else None))
        self.axial = self._spin(case.axial if case else 0.0, decimals=1,
                                big=True)
        self.axial_node = _combo([(str(i), i) for i in node_ids],
                                 default=(case.axial_node if
                                          (case and case.axial_node) else
                                          default_node))
        self.axial_dof = _combo(_DOFS,
                                default=(case.axial_dof if case else 0))
        init.add_row("Continue from", self.continue_from)
        init.add_row(f"Axial preload [{fu}]", self.axial)
        init.add_row("Axial node", self.axial_node)
        init.add_row("Axial DOF", self.axial_dof)

        # ---- Solver (Newton controls) ----
        solver = ui.GroupCard("Solver")
        self.tol = _combo([(f"{t:.0e}", t) for t in
                           (1e-4, 1e-5, 1e-6, 1e-7, 1e-8)],
                          default=(case.tol if case else 1e-6))
        self.max_iter = QSpinBox()
        self.max_iter.setRange(10, 1000)
        self.max_iter.setValue(case.max_iter if case else 60)
        solver.add_row("Convergence tol", self.tol)
        solver.add_row("Max iterations", self.max_iter)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(style.SP_LG, style.SP_LG,
                                 style.SP_LG, style.SP_LG)
        outer.setSpacing(style.SP_MD)
        outer.addWidget(self.header)
        outer.addWidget(ui.two_column(control, protocol, init, solver))
        outer.addWidget(ui.dialog_buttons(self))
        self._on_protocol()
        style.apply(self)

    @staticmethod
    def _spin(value, *, decimals=4, big=False) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(-1e15 if big else -1e6, 1e15 if big else 1e6)
        s.setDecimals(decimals)
        s.setValue(value)
        return s

    def _on_protocol(self, *_) -> None:
        cyclic = self.protocol.currentData() == "cyclic"
        self.mono_host.setVisible(not cyclic)
        self.cyc_host.setVisible(cyclic)

    def _on_dof(self, *_) -> None:
        self._dir_hint.setText(ui.direction_glyph(self.dof.currentText()).text())

    def data(self) -> NonlinearCase:
        axial = float(self.axial.value())
        return NonlinearCase(
            id=self.id_spin.value(),
            name=self.header.name() or f"Case {self.id_spin.value()}",
            control_node=self.node.currentData(),
            control_dof=self.dof.currentData(),
            target=float(self.target.value()),
            n_steps=int(self.n_steps.value()),
            protocol=self.protocol.currentData(),
            amplitudes=_parse_amplitudes(self.amplitudes.text(),
                                         [0.25, 0.5, 0.75, 1.0]),
            cycles=int(self.cycles.value()),
            pts_per_cycle=int(self.pts.value()),
            axial=axial,
            axial_node=(self.axial_node.currentData() if axial else None),
            axial_dof=self.axial_dof.currentData(),
            continue_from=self.continue_from.currentData(),
            tol=float(self.tol.currentData()),
            max_iter=int(self.max_iter.value()),
            notes=self.header.notes())

    @classmethod
    def edit(cls, parent, project, case=None):
        dlg = cls(parent, project, case)
        return dlg.data() if dlg.exec() else None


class NonlinearCaseManagerDialog(QDialog):
    """List / add / edit / delete nonlinear cases. Returns the edited list via
    :meth:`result_cases` after a successful close."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Nonlinear cases")
        self._project = project
        self._cases = copy.deepcopy(project.nonlinear_cases)
        v = QVBoxLayout(self)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["id", "name", "protocol", "control"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumSize(480, 240)
        v.addWidget(self.table)

        row = QHBoxLayout()
        for label, cb in (("Add…", self._add), ("Edit…", self._edit),
                          ("Delete", self._delete)):
            b = QPushButton(label)
            b.clicked.connect(cb)
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)
        self._refresh()

    def _refresh(self) -> None:
        dof = {0: "Ux", 1: "Uy", 2: "Rz"}
        self.table.setRowCount(len(self._cases))
        for r, c in enumerate(self._cases):
            self.table.setItem(r, 0, QTableWidgetItem(str(c.id)))
            self.table.setItem(r, 1, QTableWidgetItem(c.name))
            proto = c.protocol + (" ↩" if c.continue_from else "")
            self.table.setItem(r, 2, QTableWidgetItem(proto))
            self.table.setItem(r, 3, QTableWidgetItem(
                f"node {c.control_node} {dof.get(c.control_dof, '?')}"))

    def _proxy_project(self):
        proxy = copy.copy(self._project)
        proxy.nonlinear_cases = self._cases
        return proxy

    def _add(self) -> None:
        c = NonlinearCaseDialog.edit(self, self._proxy_project())
        if c is None:
            return
        if any(x.id == c.id for x in self._cases):
            QMessageBox.warning(self, "Duplicate",
                                f"Case {c.id} already exists.")
            return
        self._cases.append(c)
        self._refresh()

    def _edit(self) -> None:
        r = self.table.currentRow()
        if r < 0:
            return
        c = NonlinearCaseDialog.edit(self, self._proxy_project(),
                                     self._cases[r])
        if c is not None:
            self._cases[r] = c
            self._refresh()

    def _delete(self) -> None:
        r = self.table.currentRow()
        if r < 0:
            return
        cid = self._cases[r].id
        used = [c.id for c in self._cases if c.continue_from == cid]
        if used:
            QMessageBox.warning(self, "In use",
                                f"Case {cid} is continued-from by case(s) "
                                f"{', '.join(map(str, used))}.")
            return
        del self._cases[r]
        self._refresh()

    def result_cases(self) -> list:
        return self._cases

    @classmethod
    def manage(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.result_cases() if dlg.exec() else None
