"""Similar-story replication dialog (wall plan W4c).

Pick a source story and the target stories to copy its plan (walls, columns,
beams) up to. Returns ``(source_id, [target_id, …])`` for
``story_replicate.replicate_story``.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem,
                               QPushButton, QVBoxLayout)

import style


class StoryReplicateDialog(QDialog):
    """Choose a source story and target stories for replication."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Replicate story")
        self.resize(380, 420)
        self._project = project
        self._stories = project.stories_sorted()
        self.result = None

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_SM)

        head = QLabel("Replicate story")
        head.setObjectName("h2")
        root.addWidget(head)

        row = QHBoxLayout()
        row.addWidget(QLabel("Source"))
        self.source = QComboBox()
        for s in self._stories:
            self.source.addItem(f"{s.name}  (elev {s.elev:g})", s.id)
        self.source.currentIndexChanged.connect(self._reload_targets)
        row.addWidget(self.source, 1)
        root.addLayout(row)

        root.addWidget(QLabel("Copy to"))
        self.targets = QListWidget()
        root.addWidget(self.targets, 1)
        btns = QHBoxLayout()
        allb = QPushButton("All above")
        allb.clicked.connect(lambda: self._check(above_only=True))
        noneb = QPushButton("None")
        noneb.clicked.connect(lambda: self._check(state=False))
        btns.addWidget(allb)
        btns.addWidget(noneb)
        btns.addStretch(1)
        root.addLayout(btns)

        hint = QLabel("The source story's walls, columns and beams are copied "
                      "to each checked target (translated by the elevation "
                      "difference); shared joints merge, so pier labels run up "
                      "the building.")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        root.addWidget(hint)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)
        style.apply(self)

        self._reload_targets()

    def _reload_targets(self) -> None:
        src_id = self.source.currentData()
        src = next((s for s in self._stories if s.id == src_id), None)
        self.targets.clear()
        for s in self._stories:
            if s.id == src_id:
                continue
            it = QListWidgetItem(f"{s.name}  (elev {s.elev:g})")
            it.setData(Qt.ItemDataRole.UserRole, s.id)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            # default: check stories above the source (the usual case)
            above = src is not None and s.elev > src.elev
            it.setCheckState(Qt.CheckState.Checked if above
                             else Qt.CheckState.Unchecked)
            self.targets.addItem(it)

    def _check(self, state=True, above_only=False) -> None:
        src_id = self.source.currentData()
        src = next((s for s in self._stories if s.id == src_id), None)
        for i in range(self.targets.count()):
            it = self.targets.item(i)
            on = state
            if above_only:
                sid = it.data(Qt.ItemDataRole.UserRole)
                s = next((x for x in self._stories if x.id == sid), None)
                on = bool(src and s and s.elev > src.elev)
            it.setCheckState(Qt.CheckState.Checked if on
                             else Qt.CheckState.Unchecked)

    def _accept(self) -> None:
        targets = [self.targets.item(i).data(Qt.ItemDataRole.UserRole)
                   for i in range(self.targets.count())
                   if self.targets.item(i).checkState() == Qt.CheckState.Checked]
        self.result = (self.source.currentData(), targets)
        self.accept()

    @classmethod
    def get(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.result if dlg.exec() else None
