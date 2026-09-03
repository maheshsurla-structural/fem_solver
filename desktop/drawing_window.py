"""Drawing-sheet window — the matplotlib GA drawing embedded in Qt, with
PDF / DXF export. Opened from the main window's Drawings action.
"""
from __future__ import annotations

from matplotlib.backends.backend_qtagg import (FigureCanvasQTAgg as Canvas,
                                               NavigationToolbar2QT as NavBar)
from matplotlib.figure import Figure
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox

import drawing


class DrawingWindow(QMainWindow):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self._project = project
        self.setWindowTitle(f"Drawing — {project.name}")
        self.resize(1000, 780)

        self.fig = Figure(figsize=(8.5, 7.2))
        self.canvas = Canvas(self.fig)
        self.setCentralWidget(self.canvas)
        self.addToolBar(NavBar(self.canvas, self))

        tb = self.addToolBar("Export")
        act_pdf = QAction("Export &PDF…", self)
        act_pdf.triggered.connect(self._export_pdf)
        act_dxf = QAction("Export &DXF…", self)
        act_dxf.triggered.connect(self._export_dxf)
        tb.addAction(act_pdf)
        tb.addAction(act_dxf)

        self._redraw()

    def _redraw(self) -> None:
        drawing.draw_sheet(self._project, self.fig)
        self.canvas.draw_idle()

    def _export_pdf(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export drawing", f"{self._project.name}.pdf",
            "PDF (*.pdf);;PNG (*.png);;SVG (*.svg)")
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=200, bbox_inches="tight")
            self.statusBar().showMessage(f"Saved {path}")
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(exc))

    def _export_dxf(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export DXF", f"{self._project.name}.dxf", "DXF (*.dxf)")
        if not path:
            return
        try:
            drawing.export_dxf(self._project, path)
            self.statusBar().showMessage(f"Saved {path}")
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(exc))
