"""Load Case Tree (E4) — the desktop counterpart of CSiBridge's *Show Load Case
Tree*.

Renders the analysis-case dependency graph (:mod:`case_graph`) as a tree: each
case is nested under the case it continues from / starts its stiffness from, so
a staged chain (``PRELOAD → MONO → TH-0.25 …``) reads top-down. Dangling
references (source deleted) and dependency cycles are flagged inline. Read-only —
it opens from the Analysis-cases home and closes without changing anything. Pure
Qt, headless-constructible.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHeaderView, QLabel,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout)

import case_graph
import style


class CaseTreeDialog(QDialog):
    """Show the analysis cases as a dependency tree (source → dependents)."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Load case tree")
        self.resize(520, 420)
        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_MD)

        sub = QLabel("Each case is nested under the nonlinear case it continues "
                     "from or takes its initial state from.")
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        root.addWidget(sub)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Case", "Type"])
        self.tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents)
        root.addWidget(self.tree, 1)

        forest, cycles = case_graph.case_forest(project)
        cycle_set = set(cycles)
        for node in forest:
            self.tree.addTopLevelItem(self._item(node, cycle_set))
        self.tree.expandAll()

        self._empty = QLabel("No analysis cases yet — add some in the "
                             "Analysis-cases home.")
        self._empty.setObjectName("hintLabel")
        self._empty.setWordWrap(True)
        self._empty.setVisible(not forest)
        root.addWidget(self._empty)

        note = self._summary(forest, cycles)
        if note:
            lbl = QLabel(note)
            lbl.setObjectName("hintLabel")
            lbl.setWordWrap(True)
            root.addWidget(lbl)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        btns.button(QDialogButtonBox.StandardButton.Close).clicked.connect(
            self.accept)
        root.addWidget(btns)
        style.apply(self)

    def _item(self, node: dict, cycle_set: set) -> QTreeWidgetItem:
        name = node["name"]
        flags = []
        if node["dangling"]:
            flags.append("source missing")
        if node["key"] in cycle_set:
            flags.append("cycle")
        if flags:
            name = f"{name}  ⚠ {', '.join(flags)}"
        it = QTreeWidgetItem([name, node["type"]])
        for child in node.get("children", []):
            it.addChild(self._item(child, cycle_set))
        return it

    @staticmethod
    def _summary(forest, cycles) -> str:
        n = _count(forest) if forest else 0
        parts = [f"{n} case{'s' if n != 1 else ''}"]
        dangling = _count(forest, key=lambda x: x["dangling"])
        if dangling:
            parts.append(f"{dangling} with a missing source")
        if cycles:
            parts.append(f"{len(cycles)} in a dependency cycle")
        return " · ".join(parts) if (dangling or cycles) else ""

    @classmethod
    def show_tree(cls, parent, project):
        dlg = cls(parent, project)
        dlg.exec()
        return dlg


def _count(forest, key=None) -> int:
    total = 0
    for node in forest:
        if key is None or key(node):
            total += 1
        total += _count(node.get("children", []), key)
    return total
