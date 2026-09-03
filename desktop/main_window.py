"""The application shell: a docked-panel main window over the 3-D viewport.

This is the native version of the AdSec-style layout — a central viewport,
a left model tree, and a bottom output log — the frame every future view
(analysis, design, drawings) will dock into.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QDockWidget, QMainWindow, QPlainTextEdit,
                               QTreeWidget, QTreeWidgetItem)

import model_geometry as mg
from model_view import ModelView


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("femsolver — desktop (preview)")
        self.resize(1280, 820)
        self._model = None

        self.view = ModelView(self)
        self.setCentralWidget(self.view)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Model"])
        dock_tree = QDockWidget("Model", self)
        dock_tree.setWidget(self.tree)
        dock_tree.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                                  | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock_tree)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        dock_log = QDockWidget("Output", self)
        dock_log.setWidget(self.log)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock_log)

        self._build_menu()
        self.statusBar().showMessage("Ready")

    def _build_menu(self) -> None:
        self.act_run = QAction("&Run (linear static)", self)
        self.act_run.setShortcut("Ctrl+R")
        self.act_run.triggered.connect(self.run_linear_static)

        self.act_undef = QAction("&Undeformed", self)
        self.act_undef.triggered.connect(self._show_undeformed)

        self.act_fit = QAction("&Fit", self)
        self.act_fit.setShortcut("F")
        self.act_fit.triggered.connect(self.view.fit)

        analysis_menu = self.menuBar().addMenu("&Analysis")
        analysis_menu.addAction(self.act_run)
        analysis_menu.addAction(self.act_undef)
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.act_fit)

        toolbar = self.addToolBar("Main")
        toolbar.addAction(self.act_run)
        toolbar.addAction(self.act_undef)
        toolbar.addSeparator()
        toolbar.addAction(self.act_fit)

    def run_linear_static(self) -> None:
        if self._model is None:
            return
        from femsolver import LinearStaticAnalysis
        info = LinearStaticAnalysis(self._model).run()
        dmax = mg.max_translation(self._model)
        span = mg.model_span(self._model)
        scale = (0.08 * span / dmax) if dmax > 0 else 1.0
        self.view.show_deformed(self._model, scale)
        self.log.appendPlainText(
            f"Linear static solved: neq={info.get('neq', '?')}, "
            f"max|u| = {dmax:.4e} m, deformation ×{scale:.0f}")
        self.statusBar().showMessage(
            f"Solved · max|u| {dmax:.3e} m · deformation ×{scale:.0f}")

    def _show_undeformed(self) -> None:
        if self._model is not None:
            self.view.set_model(self._model)

    # ---------------------------------------------------------------- loading
    def load_model(self, model) -> None:
        self._model = model
        self.view.set_model(model)
        self._populate_tree(model)
        self.log.appendPlainText(repr(model))
        self.statusBar().showMessage(
            f"{len(model.nodes)} nodes · {len(model.elements)} elements · "
            f"ndm {model.ndm} · ndf {model.ndf}")

    def _populate_tree(self, model) -> None:
        self.tree.clear()
        nodes = QTreeWidgetItem(self.tree, [f"Nodes ({len(model.nodes)})"])
        for t, n in model.nodes.items():
            coords = ", ".join(f"{c:g}" for c in n.coords)
            fixed = "  [fixed]" if bool(np.any(n.fixity)) else ""
            QTreeWidgetItem(nodes, [f"{t}:  ({coords}){fixed}"])
        elems = QTreeWidgetItem(self.tree, [f"Elements ({len(model.elements)})"])
        for t, e in model.elements.items():
            QTreeWidgetItem(elems, [f"{t}:  {type(e).__name__} {tuple(e.node_tags)}"])
        nodes.setExpanded(True)
        elems.setExpanded(True)
