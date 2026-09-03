"""Toolchain spike: prove PySide6 + PyVista/VTK open an embedded 3-D viewport.

This is NOT the app -- it is the smallest program that exercises the whole
desktop stack end to end: a Qt main window with a PyVista/VTK viewport
embedded via ``pyvistaqt.QtInteractor``, rendering a little portal frame
(nodes as spheres, members as tubes) the way the real FEM model view will.

If this window opens and you can orbit/zoom the frame, the stack works and
everything after it is "just" building views over the femsolver engine.

Run (from the GUI virtual-env that has the stack installed):

    python desktop/spike_viewport.py

Headless import check only (no visible window), useful in CI / over SSH:

    QT_QPA_PLATFORM=offscreen python desktop/spike_viewport.py --check
"""
from __future__ import annotations

import sys

try:
    import numpy as np
    import pyvista as pv
    from pyvistaqt import QtInteractor
    from PySide6.QtWidgets import QApplication, QMainWindow
except Exception as exc:  # pragma: no cover - the whole point is to see this
    print("Desktop stack import FAILED:", exc)
    print("Install it into a virtual-env with:")
    print("    python -m pip install PySide6 pyvista pyvistaqt pyqtgraph")
    raise SystemExit(1)


def _portal_frame():
    """A 2-bay, 2-storey portal frame as PyVista point + line geometry."""
    xs, zs = (0.0, 4.0, 8.0), (0.0, 3.0, 6.0)
    idx, pts = {}, []
    for j, z in enumerate(zs):
        for i, x in enumerate(xs):
            idx[(i, j)] = len(pts)
            pts.append((x, 0.0, z))
    pts = np.asarray(pts, float)

    segs = []
    for i in range(len(xs)):                       # columns
        for j in range(len(zs) - 1):
            segs.append((idx[(i, j)], idx[(i, j + 1)]))
    for j in range(1, len(zs)):                    # beams
        for i in range(len(xs) - 1):
            segs.append((idx[(i, j)], idx[(i + 1, j)]))

    lines = np.hstack([[2, a, b] for a, b in segs])
    return pts, pv.PolyData(pts, lines=lines)


class SpikeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("femsolver desktop — viewport spike")
        self.resize(1100, 720)
        self.plotter = QtInteractor(self)
        self.setCentralWidget(self.plotter)

        pts, frame = _portal_frame()
        self.plotter.add_mesh(frame.tube(radius=0.06), color="#3b6d11")
        self.plotter.add_points(pts, color="#185fa5",
                                render_points_as_spheres=True, point_size=18)
        self.plotter.show_grid()
        self.plotter.set_background("white")
        self.plotter.view_isometric()

    def closeEvent(self, event):                   # tidy VTK teardown
        self.plotter.close()
        super().closeEvent(event)


def main() -> int:
    check_only = "--check" in sys.argv
    app = QApplication(sys.argv)
    win = SpikeWindow()
    if check_only:
        print("Desktop stack import + construct OK "
              f"(PySide6 / pyvista {pv.__version__} / VTK).")
        return 0
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
