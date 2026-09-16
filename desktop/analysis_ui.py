"""Shared UI scaffold for the Loads & Analysis redesign (plan L1).

See ``docs/source/loads_analysis_ux_plan.md`` §5 — this module freezes the small
set of reusable pieces that make every Loads/Analysis dialog read like a serious
commercial FEA tool (CSiBridge / Midas *Load Case Data*): titled group panels,
a two-column layout, a "Loads Applied" table with Add / Modify / Delete, and a
standard Name / Notes / Type header.

Everything here is pure Qt and headless-constructible under
``QT_QPA_PLATFORM=offscreen`` — no solver, no OpenGL. Colours, spacing, radii and
type come only from :mod:`style`; the widgets inherit the app's light/dark theme
through ``style.apply(top_level_widget)`` like the rest of the GUI.

Contract (do not rename / resignature — later phases import these)::

    GroupCard(title, *, form=True)     .add_row(label, widget) / .body_layout()
    CaseHeader(*, name, type_label, notes="")   .name() / .notes() / .type_label()
    InitialConditionCard(sources)   .value() / .hold() / .set_value(ic, hold)
    LoadsAppliedTable(columns, *, on_add=None, on_edit=None)  .rows() / .set_rows(rows)
    two_column(*cards) -> QWidget          # responsive 2-col grid of GroupCards
    dialog_buttons(dialog) -> QDialogButtonBox   # Ok|Cancel, wired
    direction_glyph(dof_label) -> QWidget        # sign / axis hint
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox,
                               QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPlainTextEdit, QPushButton, QRadioButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

import style


# --------------------------------------------------------------------- GroupCard
class GroupCard(QWidget):
    """A titled card panel — the building block of a "Load Case Data" dialog.

    Rendered as a card by the app QSS (``QGroupBox`` styling); use it instead of
    a bare stack of form rows so a dialog groups into scannable panels.

    Form mode (default) hosts a right-aligned :class:`QFormLayout`::

        g = GroupCard("Control")
        g.add_row("Control node", node_combo)
        g.add_row("Control DOF", dof_combo)

    Custom mode gives you the body layout to fill yourself::

        g = GroupCard("Loads Applied", form=False)
        g.body_layout().addWidget(my_table)

    Set ``card.full_width = True`` before handing a card to :func:`two_column` to
    make it span both columns (e.g. a Loads-Applied table).
    """

    def __init__(self, title: str, parent=None, *, form: bool = True):
        super().__init__(parent)
        # An inner QGroupBox carries the QSS card look + title; the widget itself
        # is a thin transparent wrapper so callers can toggle ``full_width``.
        from PySide6.QtWidgets import QGroupBox
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._box = QGroupBox(title)
        outer.addWidget(self._box)
        self.full_width = False
        if form:
            self._form: QFormLayout | None = QFormLayout(self._box)
            self._form.setLabelAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._form.setFormAlignment(Qt.AlignmentFlag.AlignLeft
                                        | Qt.AlignmentFlag.AlignTop)
            self._form.setFieldGrowthPolicy(
                QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            self._form.setHorizontalSpacing(style.SP_MD)
            self._form.setVerticalSpacing(style.SP_SM)
            self._layout = self._form
        else:
            self._form = None
            self._layout = QVBoxLayout(self._box)
            self._layout.setSpacing(style.SP_SM)

    def add_row(self, label: str, widget) -> None:
        """Add a labelled row (form mode only)."""
        if self._form is None:
            raise RuntimeError("GroupCard(form=False) has no rows — "
                               "use body_layout() instead.")
        self._form.addRow(label, widget)

    def add_full_row(self, widget) -> None:
        """Add a widget that spans the full width (no label), form mode only."""
        if self._form is None:
            raise RuntimeError("GroupCard(form=False) has no rows.")
        self._form.addRow(widget)

    def body_layout(self):
        """The card's layout — a QFormLayout in form mode, else a QVBoxLayout."""
        return self._layout


# -------------------------------------------------------------------- CaseHeader
class CaseHeader(QWidget):
    """The standard top row of a case dialog: **Name · Notes · Type**.

    Mirrors the CSiBridge *Load Case Name / Notes / Load Case Type* header. The
    Notes affordance is a "Modify/Show…" button that opens a small multi-line
    editor on demand (never at construction — safe headless). Read the values on
    accept via :meth:`name`, :meth:`notes`, :meth:`type_label`.
    """

    def __init__(self, *, name: str, type_label: str, notes: str = "",
                 parent=None):
        super().__init__(parent)
        self._notes = notes
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(style.SP_MD)

        name_card = GroupCard("Load case name")
        self._name = QLineEdit(name)
        name_card.add_row("", self._name)
        row.addWidget(name_card, 3)

        notes_card = GroupCard("Notes")
        self._notes_btn = QPushButton("Modify / Show…")
        self._notes_btn.clicked.connect(self._edit_notes)
        notes_card.add_row("", self._notes_btn)
        row.addWidget(notes_card, 2)

        type_card = GroupCard("Load case type")
        self._type = QLabel(type_label)
        self._type.setObjectName("h3")
        type_card.add_row("", self._type)
        row.addWidget(type_card, 2)
        self._refresh_notes_btn()

    def _refresh_notes_btn(self) -> None:
        self._notes_btn.setText("Notes ✎" if self._notes.strip()
                                else "Modify / Show…")

    def _edit_notes(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Notes")
        v = QVBoxLayout(dlg)
        edit = QPlainTextEdit(self._notes)
        v.addWidget(edit)
        v.addWidget(dialog_buttons(dlg))
        if dlg.exec():
            self._notes = edit.toPlainText()
            self._refresh_notes_btn()

    def name(self) -> str:
        return self._name.text().strip()

    def notes(self) -> str:
        return self._notes

    def type_label(self) -> str:
        return self._type.text()


# -------------------------------------------------------- InitialConditionCard
class InitialConditionCard(GroupCard):
    """The *Stiffness to Use* group of a Load-Case-Data dialog (E2) — the desktop
    counterpart of SAP2000's initial-condition radio.

    Choose an **unstressed** start, or continue from the committed **state +
    stiffness** at the end of a nonlinear case (``sources`` = eligible
    ``(id, name)`` pairs). As in SAP the source case's *loads* are not otherwise
    carried; the **Hold source loads constant** checkbox re-applies them (interim
    until an explicit Loads-Applied list exists), which a dynamic run needs so the
    preload stays in equilibrium.

    ``value()`` → ``("zero",)`` or ``("state", id)``; ``hold()`` → the checkbox;
    ``set_value(ic, hold)`` seeds both. Pure Qt, headless-constructible.
    """

    def __init__(self, sources, parent=None):
        super().__init__("Stiffness to use", parent)
        self._zero = QRadioButton("Zero initial conditions — unstressed state")
        self._state = QRadioButton("State at end of nonlinear case")
        self._grp = QButtonGroup(self)
        self._grp.addButton(self._zero, 0)
        self._grp.addButton(self._state, 1)
        self._zero.setChecked(True)

        self._src = QComboBox()
        for cid, nm in sources:
            self._src.addItem(nm, cid)
        self._hold = QCheckBox("Hold source loads constant")
        self._hold.setChecked(True)
        note = QLabel("Loads from the nonlinear case are not otherwise included "
                      "in this case.")
        note.setObjectName("hintLabel")
        note.setWordWrap(True)

        self.add_full_row(self._zero)
        self.add_full_row(self._state)
        self.add_row("From case", self._src)
        self.add_full_row(self._hold)
        self.add_full_row(note)
        if not sources:                          # nothing to continue from
            self._state.setEnabled(False)
            empty = QLabel("No nonlinear cases defined to continue from.")
            empty.setObjectName("hintLabel")
            empty.setWordWrap(True)
            self.add_full_row(empty)
        self._state.toggled.connect(lambda *_: self._sync())
        self._sync()

    def _sync(self) -> None:
        on = self._state.isChecked()
        self._src.setEnabled(on and self._src.count() > 0)
        self._hold.setEnabled(on)

    def value(self) -> tuple:
        if self._state.isChecked() and self._src.currentData() is not None:
            return ("state", int(self._src.currentData()))
        return ("zero",)

    def hold(self) -> bool:
        return bool(self._hold.isChecked())

    def set_value(self, ic, hold: bool = True) -> None:
        if ic and ic[0] == "state" and self._src.findData(ic[1]) >= 0:
            self._src.setCurrentIndex(self._src.findData(ic[1]))
            self._state.setChecked(True)
        else:
            self._zero.setChecked(True)
        self._hold.setChecked(bool(hold))
        self._sync()


# ------------------------------------------------------------- LoadsAppliedTable
class LoadsAppliedTable(QWidget):
    """A columns-configurable table with a right-side Add / Modify / Delete
    column — the CSiBridge *Loads Applied* box, reused for combination factor
    grids, response-spectrum and time-history load lists.

    Rows are plain ``dict`` payloads; the owner supplies editor callbacks:

    * ``on_add()`` → return a new row ``dict`` (or ``None`` to cancel).
    * ``on_edit(row)`` → return the edited row ``dict`` (or ``None`` to cancel).

    Only the keys named in ``columns`` are shown; a payload may carry extra keys
    the owner cares about. Double-clicking a row triggers *Modify*.
    """

    def __init__(self, columns: list[str], *,
                 on_add: Callable[[], dict | None] | None = None,
                 on_edit: Callable[[dict], dict | None] | None = None,
                 parent=None):
        super().__init__(parent)
        self._cols = list(columns)
        self._rows: list[dict] = []
        self._on_add = on_add
        self._on_edit = on_edit

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(style.SP_MD)

        self.table = QTableWidget(0, len(self._cols))
        self.table.setHorizontalHeaderLabels(self._cols)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.doubleClicked.connect(lambda *_: self._edit())
        self.table.itemSelectionChanged.connect(self._sync_buttons)
        row.addWidget(self.table, 1)

        col = QVBoxLayout()
        col.setSpacing(style.SP_SM)
        self._add_btn = QPushButton("Add…")
        self._add_btn.clicked.connect(self._add)
        self._mod_btn = QPushButton("Modify…")
        self._mod_btn.clicked.connect(self._edit)
        self._del_btn = QPushButton("Delete")
        self._del_btn.clicked.connect(self._delete)
        for b in (self._add_btn, self._mod_btn, self._del_btn):
            col.addWidget(b)
        col.addStretch(1)
        row.addLayout(col)

        self._add_btn.setEnabled(self._on_add is not None)
        self._sync_buttons()

    # -- data -----------------------------------------------------------------
    def rows(self) -> list[dict]:
        """A copy of the current row payloads, in display order."""
        return [dict(r) for r in self._rows]

    def set_rows(self, rows: list[dict]) -> None:
        self._rows = [dict(r) for r in rows]
        self._refresh()

    # -- internals ------------------------------------------------------------
    def _refresh(self) -> None:
        self.table.setRowCount(len(self._rows))
        for r, payload in enumerate(self._rows):
            for c, key in enumerate(self._cols):
                val = payload.get(key, "")
                it = QTableWidgetItem("" if val is None else str(val))
                self.table.setItem(r, c, it)
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        has_sel = self.table.currentRow() >= 0 and bool(self._rows)
        self._mod_btn.setEnabled(has_sel and self._on_edit is not None)
        self._del_btn.setEnabled(has_sel)

    def _add(self) -> None:
        if self._on_add is None:
            return
        payload = self._on_add()
        if payload:
            self._rows.append(dict(payload))
            self._refresh()
            self.table.setCurrentCell(len(self._rows) - 1, 0)

    def _edit(self) -> None:
        r = self.table.currentRow()
        if self._on_edit is None or not (0 <= r < len(self._rows)):
            return
        payload = self._on_edit(dict(self._rows[r]))
        if payload:
            self._rows[r] = dict(payload)
            self._refresh()
            self.table.setCurrentCell(r, 0)

    def _delete(self) -> None:
        r = self.table.currentRow()
        if 0 <= r < len(self._rows):
            del self._rows[r]
            self._refresh()


# --------------------------------------------------------------------- layout
def two_column(*cards: QWidget) -> QWidget:
    """Lay :class:`GroupCard` (or any widget) into a responsive two-column grid,
    filling left-to-right then top-to-bottom. A card with a truthy
    ``full_width`` attribute takes a whole row on its own. The caller adds its
    own :func:`dialog_buttons` below the returned widget.
    """
    host = QWidget()
    grid = QGridLayout(host)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(style.SP_MD)
    grid.setVerticalSpacing(style.SP_MD)
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)
    r = c = 0
    for card in cards:
        if getattr(card, "full_width", False):
            if c != 0:                       # close a half-filled row first
                r += 1
                c = 0
            grid.addWidget(card, r, 0, 1, 2)
            r += 1
            c = 0
        else:
            grid.addWidget(card, r, c)
            c += 1
            if c == 2:
                r += 1
                c = 0
    return host


def dialog_buttons(dialog: QDialog) -> QDialogButtonBox:
    """An OK / Cancel button box wired to the dialog's accept / reject."""
    bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                          | QDialogButtonBox.StandardButton.Cancel)
    bb.accepted.connect(dialog.accept)
    bb.rejected.connect(dialog.reject)
    return bb


# ----------------------------------------------------------------- sign hints
# Positive convention = acting along / about the +axis. One small hint per DOF.
_DIRECTION_HINT = {
    "dx": "→  +X", "ux": "→  +X",
    "dy": "↑  +Y", "uy": "↑  +Y",
    "dz": "⊙  +Z (toward viewer)", "uz": "⊙  +Z (toward viewer)",
    "rx": "↻  about +X", "ry": "↻  about +Y", "rz": "↻  about +Z",
}


def direction_glyph(dof_label: str) -> QWidget:
    """A muted one-line sign/axis hint for a DOF label (e.g. ``"Dy"`` →
    ``"↑ +Y"``). Reads as help text; falls back to the label itself."""
    txt = _DIRECTION_HINT.get((dof_label or "").strip().lower(), dof_label)
    lbl = QLabel(txt)
    lbl.setObjectName("hintLabel")
    return lbl


# --------------------------------------------------------------------- demo
def _demo_dialog(parent=None) -> QDialog:
    """A throwaway dialog assembled purely from the scaffold — used by the L1
    screenshot script to prove the grouped, two-column, boxed look in both
    themes. Not part of the app; safe to construct headless.
    """
    from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox

    dlg = QDialog(parent)
    dlg.setWindowTitle("Analysis case (scaffold demo)")
    outer = QVBoxLayout(dlg)
    outer.setContentsMargins(style.SP_LG, style.SP_LG, style.SP_LG, style.SP_LG)
    outer.setSpacing(style.SP_MD)

    header = CaseHeader(name="PUSHOVER-X", type_label="Nonlinear Static")
    outer.addWidget(header)

    control = GroupCard("Control")
    node = QComboBox(); node.addItems(["9", "8", "7"])
    dof = QComboBox(); dof.addItems(["Ux", "Uy", "Rz"]); dof.setCurrentText("Uy")
    target = QDoubleSpinBox(); target.setDecimals(4); target.setValue(0.05)
    control.add_row("Control node", node)
    control.add_row("Control DOF", dof)
    control.add_row("Target [m]", target)
    control.add_full_row(direction_glyph("Uy"))

    protocol = GroupCard("Protocol")
    proto = QComboBox(); proto.addItems(["Monotonic", "Cyclic"])
    steps = QSpinBox(); steps.setRange(2, 5000); steps.setValue(40)
    protocol.add_row("Protocol", proto)
    protocol.add_row("Steps", steps)

    solver = GroupCard("Solver")
    tol = QComboBox(); tol.addItems(["1e-4", "1e-5", "1e-6"]); tol.setCurrentText("1e-6")
    mi = QSpinBox(); mi.setRange(10, 1000); mi.setValue(60)
    solver.add_row("Convergence tol", tol)
    solver.add_row("Max iterations", mi)

    applied = GroupCard("Loads Applied", form=False)
    applied.full_width = True
    table = LoadsAppliedTable(["Load Type", "Load Name", "Scale"],
                              on_add=lambda: {"Load Type": "Accel",
                                              "Load Name": "UX", "Scale": 1.0})
    table.set_rows([{"Load Type": "Load Pattern", "Load Name": "Dead",
                     "Scale": 1.0},
                    {"Load Type": "Accel", "Load Name": "UX", "Scale": 1.0}])
    applied.body_layout().addWidget(table)

    outer.addWidget(two_column(control, protocol, solver, applied))
    outer.addWidget(dialog_buttons(dlg))
    style.apply(dlg)
    return dlg
