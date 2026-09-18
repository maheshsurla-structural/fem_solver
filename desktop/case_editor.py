"""Unified *Load Case Data* editor (E1) — one dialog whose **Type ▾** reshapes
the body, the desktop counterpart of SAP2000 / CSiBridge's *Load Case Data*.

The shared chrome — Name, Notes, the **Type** dropdown, and the E2 *Stiffness to
use* card — stays put while the type-specific body (:mod:`case_bodies`) is
swapped into a scroll area. Switching the type on an existing case changes its
type (its body's defaults take over), which is how one edits a case from, say,
Modal to Buckling.

Only the current type's page occupies the scroll area (the others wait
off-screen in ``self._pages``), so the dialog sizes to the current type and
tall bodies (Load Rating) scroll rather than pushing the buttons off-screen.
:meth:`edit` returns the edited :class:`project.AnalysisCase` (id 0 for a new
one — the caller assigns a fresh id), or ``None`` if cancelled. Pure Qt,
headless-constructible.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QFrame, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPlainTextEdit,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)

import case_bodies
import style
from analysis_ui import GroupCard, InitialConditionCard, dialog_buttons
from project import AnalysisCase


class CaseEditorDialog(QDialog):
    """Edit an analysis case with a type dropdown that swaps the body."""

    def __init__(self, parent, project, type_id, case=None):
        super().__init__(parent)
        self.setWindowTitle("Load case data")
        self._project = project
        self._notes = case.notes if case else ""
        self.resize(600, 660)
        self.setMinimumSize(480, 420)

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_MD)

        # ---- header: Name · Notes · Type ▾ ----
        hrow = QHBoxLayout()
        hrow.setSpacing(style.SP_MD)
        name_card = GroupCard("Load case name")
        self._name = QLineEdit(case.name if case else "")
        name_card.add_row("", self._name)
        hrow.addWidget(name_card, 3)
        notes_card = GroupCard("Notes")
        self._notes_btn = QPushButton()
        self._notes_btn.clicked.connect(self._edit_notes)
        notes_card.add_row("", self._notes_btn)
        hrow.addWidget(notes_card, 2)
        type_card = GroupCard("Load case type")
        self.type_combo = QComboBox()
        for tid, label, _ in case_bodies.REGISTRY:
            self.type_combo.addItem(label, tid)
        type_card.add_row("", self.type_combo)
        hrow.addWidget(type_card, 2)
        root.addLayout(hrow)
        self._refresh_notes_btn()

        # ---- type-specific body ----
        # Each type's body lives in its own page; only the *current* page sits
        # in the scroll area (swapped in _on_type). Keeping the pages in a dict
        # rather than a QStackedWidget means the scroll — and so the dialog —
        # sizes to the current type, not the tallest one (a QStackedWidget
        # reserves the tallest page's minimum height, which pushed the OK/Cancel
        # buttons off-screen for short types like Modal).
        cap = max(1, int(project.ndf) * max(1, len(project.nodes)))
        self._bodies: dict = {}
        self._pages: dict = {}
        for tid, _label, bcls in case_bodies.REGISTRY:
            init = dict(case.params) if (case and case.type == tid) else None
            body = bcls(project, initial=init, max_modes=cap)
            self._bodies[tid] = body
            page = QWidget()
            pl = QVBoxLayout(page)
            pl.setContentsMargins(0, 0, 0, 0)
            pl.setSpacing(style.SP_SM)
            title = QLabel(body.TITLE)
            title.setObjectName("h2")
            subl = QLabel(body.SUB)
            subl.setObjectName("sub")
            subl.setWordWrap(True)
            pl.addWidget(title)
            pl.addWidget(subl)
            pl.addWidget(body)
            pl.addStretch(1)          # keep the body compact at the top
            # Leave the page parentless (kept alive by self._pages) until it is
            # placed in the scroll area — a parented-but-unlaid-out page would
            # otherwise paint at (0, 0) on top of the header.
            self._pages[tid] = page
        # Types range from a couple of fields (Modal) to tall forms with node
        # lists (Load Rating). Scroll the body so the dialog stays a sane size
        # instead of growing past the screen and clipping the buttons.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll = scroll
        root.addWidget(scroll, 1)

        # ---- stiffness to use (shared across the eigen bodies) ----
        sources = [(c.id, c.name) for c in project.nonlinear_cases]
        self.initial = InitialConditionCard(sources, show_hold=False)
        hold0 = (bool((case.params or {}).get("hold_source_loads", True))
                 if case else True)
        self.initial.set_value(case.initial_condition if case else ("zero",),
                               hold0)
        root.addWidget(self.initial)
        root.addWidget(dialog_buttons(self))

        i = self.type_combo.findData(type_id)
        if i >= 0:
            self.type_combo.setCurrentIndex(i)
        if not (case and case.name):        # default name = the type label
            self._name.setText(self.type_combo.currentText())
        self.type_combo.currentIndexChanged.connect(lambda *_: self._on_type())
        self._on_type()
        style.apply(self)

    def _on_type(self) -> None:
        tid = self.type_combo.currentData()
        body = self._bodies[tid]
        # Swap the current page into the scroll area without deleting the others
        # (takeWidget detaches the previous page and reparents it to None — kept
        # alive by self._pages and no longer painted — so the scroll, and the
        # dialog, sizes to this type alone).
        self._scroll.takeWidget()
        self._scroll.setWidget(self._pages[tid])
        # only stiffness-based analyses can continue from a nonlinear state; the
        # hold-loads checkbox only applies to load-applying ones (time history)
        self.initial.setVisible(body.SUPPORTS_IC)
        self.initial.set_hold_visible(getattr(body, "SHOW_HOLD", False))

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

    def _refresh_notes_btn(self) -> None:
        self._notes_btn.setText("Notes ✎" if self._notes.strip()
                                else "Modify / Show…")

    def name(self) -> str:
        return self._name.text().strip()

    def notes(self) -> str:
        return self._notes

    def _result_case(self, case_id: int) -> AnalysisCase:
        tid = self.type_combo.currentData()
        body = self._bodies[tid]
        ic = self.initial.value() if body.SUPPORTS_IC else ("zero",)
        params = body.case_params()
        if getattr(body, "HOLD_IN_PARAMS", False):     # time history (E2c)
            params["hold_source_loads"] = self.initial.hold()
        return AnalysisCase(
            id=case_id, name=self.name() or self.type_combo.currentText(),
            type=tid, params=params, notes=self.notes(), initial_condition=ic)

    @classmethod
    def edit(cls, parent, project, type_id, case=None):
        """Open the editor seeded from ``case`` (or a fresh ``type_id``) and
        return the edited :class:`AnalysisCase`, or ``None`` if cancelled."""
        dlg = cls(parent, project, type_id, case)
        while dlg.exec():
            tid = dlg.type_combo.currentData()
            err = dlg._bodies[tid].validate()
            if err:
                QMessageBox.warning(dlg, "Load case", err)
                continue
            return dlg._result_case(case.id if case else 0)
        return None
