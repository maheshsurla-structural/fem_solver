"""femsolver desktop — application entry point (preview).

The first real view over the engine: it builds a genuine ``femsolver.Model``
and renders it in a docked Qt/PyVista shell. Everything downstream
(analysis, design, drawings) will be further views on the same model.

Run (from the repo root, using the GUI virtual-env):

    .\.venv-gui\Scripts\python.exe desktop\app.py
"""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from demo_model import demo_project
from main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.load_project(demo_project())
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
