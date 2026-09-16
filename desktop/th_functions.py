"""Time-history function library (analysis-cases-manager plan, TH-1) — the
desktop *Define ▸ Functions ▸ Time History* of CSiBridge / MIDAS.

A :class:`project.TimeHistoryFunction` is a named ground-motion acceleration
record (imported from a plain-text file), defined **once** here and referenced
by id from a saved Time-History :class:`project.AnalysisCase` (TH-2).

* :class:`TimeHistoryFunctionDialog` edits one function — name, time step,
  units, and an imported record (with a live preview of the ordinates).
* :class:`TimeHistoryFunctionManagerDialog` is the list / add / edit / delete
  manager; :meth:`manage` returns the edited list (or ``None`` if cancelled).

Pure Qt (+ a matplotlib Agg preview); headless-constructible under
``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

import copy

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog,
                               QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

import style
from analysis_ui import GroupCard, dialog_buttons
from project import TimeHistoryFunction


class TimeHistoryFunctionDialog(QDialog):
    """Edit one :class:`project.TimeHistoryFunction`."""

    def __init__(self, parent, project, func=None):
        super().__init__(parent)
        self.setWindowTitle("Edit time-history function" if func else
                            "Add time-history function")
        self._project = project
        self._edit_id = func.id if func else None
        self._values = list(func.values) if func else []
        self._source = func.source if func else ""

        outer = QHBoxLayout(self)
        outer.setContentsMargins(style.SP_LG, style.SP_LG,
                                 style.SP_LG, style.SP_LG)
        outer.setSpacing(style.SP_MD)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(style.SP_MD)
        left.setMaximumWidth(320)

        card = GroupCard("Function")
        self.name = QLineEdit(func.name if func else "")
        self.name.setPlaceholderText("e.g. El Centro NS")
        card.add_row("Name", self.name)
        self.dt = QDoubleSpinBox()
        self.dt.setRange(1.0e-5, 10.0)
        self.dt.setDecimals(5)
        self.dt.setSingleStep(0.001)
        self.dt.setValue(func.dt if func else 0.01)
        self.dt.valueChanged.connect(self._refresh_summary)
        card.add_row("Time step dt [s]", self.dt)
        self.units = QComboBox()
        self.units.addItem("m/s²", False)
        self.units.addItem("g", True)
        if func and func.in_g:
            self.units.setCurrentIndex(1)
        card.add_row("Units", self.units)

        imp = QWidget()
        ih = QHBoxLayout(imp)
        ih.setContentsMargins(0, 0, 0, 0)
        self._imp_btn = QPushButton("Import from file…")
        self._imp_btn.clicked.connect(self._import)
        ih.addWidget(self._imp_btn)
        ih.addStretch(1)
        card.add_row("Record", imp)
        self.summary = QLabel()
        self.summary.setObjectName("hintLabel")
        self.summary.setWordWrap(True)
        card.add_full_row(self.summary)
        lv.addWidget(card)
        lv.addStretch(1)
        lv.addWidget(dialog_buttons(self))
        outer.addWidget(left, 0)

        # ---- preview ----
        self._fig = Figure(figsize=(3.6, 2.4), layout="constrained")
        self._ax = self._fig.add_subplot(111)
        self._canvas = Canvas(self._fig)
        self._canvas.setMinimumWidth(340)
        outer.addWidget(self._canvas, 1)

        self._refresh_summary()
        style.apply(self)

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import acceleration record", "",
            "Records (*.txt *.csv *.at2 *.acc *.dat);;All files (*)")
        if not path:
            return
        try:
            from timehistory_dialog import load_accel_record
            vals = load_accel_record(path)
        except Exception as exc:                              # noqa: BLE001
            QMessageBox.warning(self, "Import",
                                f"Could not read the record:\n\n{exc}")
            return
        self._values = [float(v) for v in vals]
        self._source = path.replace("\\", "/").rsplit("/", 1)[-1]
        if not self.name.text().strip():                      # default the name
            self.name.setText(self._source.rsplit(".", 1)[0])
        self._refresh_summary()

    def _refresh_summary(self, *_) -> None:
        n = len(self._values)
        dur = max(0, n - 1) * float(self.dt.value())
        self.summary.setText(f"{self._source or '(no file imported)'} — "
                             f"{n} points · {dur:.3g} s")
        self._draw()

    def _draw(self) -> None:
        self._ax.clear()
        if len(self._values) >= 2:
            dt = float(self.dt.value())
            t = [i * dt for i in range(len(self._values))]
            self._ax.plot(t, self._values, "-", lw=0.9, color=style.C_PRIMARY)
            self._ax.set_xlabel("time (s)")
            self._ax.set_ylabel("accel")
            self._ax.set_title("Ground-motion record", fontsize=9)
        else:
            self._ax.text(0.5, 0.5, "(import a record)", ha="center",
                          va="center", transform=self._ax.transAxes,
                          fontsize=8, color=style.MUTED)
        self._ax.grid(True, alpha=0.25)
        self._canvas.draw_idle()

    def accept(self) -> None:
        if len(self._values) < 2:
            QMessageBox.warning(
                self, "Time-history function",
                "Import a record with at least two points first.")
            return
        super().accept()

    def data(self) -> TimeHistoryFunction:
        return TimeHistoryFunction(
            id=(self._edit_id or 0),
            name=self.name.text().strip() or "Function",
            dt=float(self.dt.value()), values=list(self._values),
            in_g=bool(self.units.currentData()), source=self._source)

    @classmethod
    def edit(cls, parent, project, func=None):
        dlg = cls(parent, project, func)
        return dlg.data() if dlg.exec() else None


class TimeHistoryFunctionManagerDialog(QDialog):
    """List / add / edit / delete time-history functions."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Time-history functions")
        self._project = project
        self._funcs = copy.deepcopy(project.th_functions)
        v = QVBoxLayout(self)
        v.setContentsMargins(style.SP_LG, style.SP_LG, style.SP_LG, style.SP_LG)
        v.setSpacing(style.SP_SM)

        head = QLabel("Time-history functions")
        head.setObjectName("h2")
        v.addWidget(head)
        sub = QLabel("Named ground-motion records — define once, then reference "
                     "them from Time-History analysis cases.")
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        v.addWidget(sub)

        body = QHBoxLayout()
        body.setSpacing(style.SP_MD)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["id", "name", "points", "dt [s]", "duration [s]"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumSize(480, 240)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.doubleClicked.connect(lambda *_: self._edit())
        body.addWidget(self.table, 1)

        col = QVBoxLayout()
        col.setSpacing(style.SP_SM)
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

    def _refresh(self) -> None:
        self.table.setRowCount(len(self._funcs))
        for r, f in enumerate(self._funcs):
            for c, txt in enumerate((str(f.id), f.name, str(f.npts),
                                     f"{f.dt:g}", f"{f.duration:.3g}")):
                self.table.setItem(r, c, QTableWidgetItem(txt))

    def _proxy_project(self):
        proxy = copy.copy(self._project)
        proxy.th_functions = self._funcs
        return proxy

    def _next_id(self) -> int:
        return max((f.id for f in self._funcs), default=0) + 1

    def _add(self) -> None:
        f = TimeHistoryFunctionDialog.edit(self, self._proxy_project())
        if f is None:
            return
        f.id = self._next_id()
        self._funcs.append(f)
        self._refresh()
        self.table.setCurrentCell(len(self._funcs) - 1, 0)

    def _edit(self) -> None:
        r = self.table.currentRow()
        if r < 0:
            return
        f = TimeHistoryFunctionDialog.edit(self, self._proxy_project(),
                                           self._funcs[r])
        if f is not None:
            f.id = self._funcs[r].id
            self._funcs[r] = f
            self._refresh()

    def _delete(self) -> None:
        r = self.table.currentRow()
        if r < 0:
            return
        fid = self._funcs[r].id
        used = [c.name for c in getattr(self._project, "analysis_cases", [])
                if c.type == "timehistory"
                and c.params.get("function_id") == fid]
        if used:
            QMessageBox.warning(
                self, "In use", f"Function {fid} is used by case(s) "
                f"{', '.join(used)}.")
            return
        del self._funcs[r]
        self._refresh()

    def result_functions(self) -> list:
        return self._funcs

    @classmethod
    def manage(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.result_functions() if dlg.exec() else None
