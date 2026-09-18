"""The pre/post-processing mode switch — a small segmented **Model | Results**
control that lives on the viewport tool strip (where the retired 2-D rotation
lock used to sit).

Commercial packages (SAP2000, Midas) put a clear switch between *building* the
model and *reviewing* results; engineers reach for it far more often than a
rotation lock. Picking **Model** returns to the modelling workspace (the Home
ribbon tab) and clears any deformed / contour overlay back to the plain model;
picking **Results** raises the Results ribbon tab where the diagrams live.

The widget is purely a view control — it emits :attr:`modeChanged` and the shell
does the wiring. :meth:`set_mode` lets the shell reflect a programmatic switch
(e.g. the auto-raise to Results after a run) without echoing the signal back.
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QToolButton, QWidget

import icons
import style

MODEL = "model"
RESULTS = "results"


class ModeSwitch(QWidget):
    """A two-segment Model/Results toggle emitting :attr:`modeChanged`."""

    modeChanged = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("modeSwitch")
        self.setCursor(Qt.CursorShape.ArrowCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QToolButton] = {}
        for mode, label, icon, tip in (
            (MODEL, "Model", "undeformed",
             "Pre-processing — build and edit the model"),
            (RESULTS, "Results", "contour",
             "Post-processing — review analysis results"),
        ):
            b = QToolButton(self)
            b.setObjectName("modeSeg")
            b.setProperty("_seg", mode)
            b.setText(label)
            b.setToolTip(tip)
            b.setCheckable(True)
            b.setAutoRaise(False)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            b.setIcon(icons.icon(icon, style.ICON))
            b.setIconSize(QSize(15, 15))
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _c=False, m=mode: self._on_click(m))
            lay.addWidget(b)
            self._group.addButton(b)
            self._buttons[mode] = b
        self._buttons[MODEL].setChecked(True)
        self._group.buttonToggled.connect(lambda *_: self._reink())
        self.apply_theme()

    # -- state ---------------------------------------------------------------
    def _on_click(self, mode: str) -> None:
        # a click always lands on the button; keep it checked and announce it
        self._buttons[mode].setChecked(True)
        self.modeChanged.emit(mode)

    def _reink(self) -> None:
        """The active segment is accent-filled, so its icon must read white; the
        inactive one uses the normal icon colour."""
        for mode, b in self._buttons.items():
            icon = "undeformed" if mode == MODEL else "contour"
            b.setIcon(icons.icon(icon, "#ffffff" if b.isChecked() else style.ICON))

    def mode(self) -> str:
        return RESULTS if self._buttons[RESULTS].isChecked() else MODEL

    def set_mode(self, mode: str) -> None:
        """Reflect ``mode`` without emitting :attr:`modeChanged` (the shell calls
        this when *it* drives the workspace, e.g. auto-raising Results)."""
        mode = RESULTS if mode == RESULTS else MODEL
        b = self._buttons[mode]
        if not b.isChecked():
            b.blockSignals(True)
            b.setChecked(True)
            b.blockSignals(False)

    # -- theme ---------------------------------------------------------------
    def apply_theme(self) -> None:
        self._reink()
        self.setStyleSheet(_QSS.format(
            PANEL=style.PANEL, BORDER=style.BORDER_STRONG, TEXT=style.TEXT,
            MUTED=style.MUTED, ACCENT=style.ACCENT, SOFT=style.ACCENT_SOFT))


# a joined segmented control: rounded ends, a shared middle edge, the active
# segment filled with the accent (its icon re-inked white in ``apply_theme``).
_QSS = """
QWidget#modeSwitch {{ background: transparent; }}
QToolButton#modeSeg {{
    background: {PANEL};
    color: {MUTED};
    border: 1px solid {BORDER};
    padding: 3px 12px;
    font-weight: 600;
}}
QToolButton#modeSeg:hover {{ color: {TEXT}; background: {SOFT}; }}
QToolButton#modeSeg:checked {{
    background: {ACCENT};
    color: #ffffff;
    border-color: {ACCENT};
}}
QToolButton#modeSeg[_seg="model"] {{
    border-top-left-radius: 5px; border-bottom-left-radius: 5px;
}}
QToolButton#modeSeg[_seg="results"] {{
    border-top-right-radius: 5px; border-bottom-right-radius: 5px;
    border-left: none;
}}
"""
