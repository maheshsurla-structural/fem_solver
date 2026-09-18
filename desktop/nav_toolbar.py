"""The viewport tool strip — a customizable toolbar of the interaction and
camera commands, docked as its own band *below the ribbon* (not floating over
the 3-D canvas, so a click on a button never bleeds through to the model).

Like Midas Civil's toolbars, related tools sit together as individual buttons
with a clear **divider between groups** — selection ┃ camera ┃ activate/inactive
┃ view — rather than hidden behind one dropdown. **Right-click → Customize
Toolbar** lets the user choose which command icons live on the bar and in what
order; the chosen layout persists in ``QSettings``.

Built-in view commands (tools, orient, zoom) are registered here; the shell adds
model/edit + activation commands via :meth:`register_many`. The bar drives the
host :class:`ModelView` and stays in step with it through the view's
``mode_changed`` / ``rotation_lock_changed`` signals, so the ribbon and this bar
never disagree.
"""
from __future__ import annotations

from PySide6.QtCore import QSettings, QSize, Qt
from PySide6.QtWidgets import (QButtonGroup, QFrame, QHBoxLayout, QMenu,
                               QToolButton, QWidget)

import icons
import style
from toolbar_commands import SEPARATOR_ID, ToolCommand

_SETTINGS_KEY = "viewport_toolbar_layout"

# the out-of-the-box bar — individual buttons in Midas-style groups divided by a
# rule: selection (single / window / polygon) ┃ camera (orbit / pan / zoom-win /
# fit / fit-sel / zoom ±) ┃ activate/inactive (inactivate / isolate / show-all /
# invert) ┃ the 2-D lock. The ``cmd_*`` ids are registered by the shell
# (MainWindow), so a bare view skips them until the shell adds them (``rebuild``
# ignores unknown ids).
_DEFAULT_LAYOUT = ["select", "window", "polygon", SEPARATOR_ID,
                   "orbit", "pan", "zoomwin", "fit", "fitsel",
                   "zoomin", "zoomout", SEPARATOR_ID,
                   "cmd_inactivate", "cmd_activate_only", "cmd_activate_all",
                   "cmd_invert_active", SEPARATOR_ID, "lock"]

# Earlier default bars — a saved layout that still matches one of these is an
# *untouched* default, so it is silently upgraded to ``_DEFAULT_LAYOUT`` while a
# genuinely customized bar is left exactly as saved.
_SUPERSEDED_DEFAULTS = [
    ["select", "orbit", "pan", "zoomwin", SEPARATOR_ID,
     "fit", "fitsel", "zoomin", "zoomout", SEPARATOR_ID, "lock"],
    ["select", "window", "polygon", SEPARATOR_ID,
     "orbit", "pan", "zoomwin", SEPARATOR_ID,
     "fit", "fitsel", "zoomin", "zoomout", SEPARATOR_ID,
     "cmd_inactivate", "cmd_activate_only", "cmd_activate_all",
     "cmd_invert_active", SEPARATOR_ID, "lock"],
    ["grp_select", SEPARATOR_ID, "orbit", "pan", "zoomwin", SEPARATOR_ID,
     "fit", "fitsel", "zoomin", "zoomout", SEPARATOR_ID,
     "grp_active", SEPARATOR_ID, "lock"],
]


def _view_commands(view) -> list:
    """The commands every viewport carries — interaction tools, orient views,
    zoom + the 2-D lock. (Model/edit commands are added by the shell.)"""
    def tool(cid, label, icon):
        return ToolCommand(cid, label, icon, group="Tools", kind="tool",
                           activate=lambda c=cid: view.set_mode(c))

    def orient(cid, label, icon):
        return ToolCommand(cid, label, icon, group="Orient", kind="action",
                           activate=lambda n=cid: view.set_view(n))

    return [
        tool("select", "Select", "single"),
        tool("orbit", "Orbit", "orbit"),
        tool("pan", "Pan", "pan"),
        tool("zoomwin", "Zoom window", "zoomwin"),
        tool("window", "Window select", "window"),
        tool("polygon", "Polygon select", "polygon"),
        tool("draw_node", "Draw node", "drawnode"),
        tool("draw_member", "Draw member", "drawmember"),
        ToolCommand("fit", "Fit all", "fit", group="Navigate",
                    activate=view.fit),
        ToolCommand("fitsel", "Fit selected", "fitsel", group="Navigate",
                    activate=view.fit_selected),
        ToolCommand("zoomin", "Zoom in", "zoomin", group="Navigate",
                    activate=view.zoom_in),
        ToolCommand("zoomout", "Zoom out", "zoomout", group="Navigate",
                    activate=view.zoom_out),
        orient("iso", "Isometric", "iso"),
        orient("top", "Top", "top"),
        orient("bottom", "Bottom", "top"),
        orient("front", "Front", "front"),
        orient("back", "Back", "front"),
        orient("left", "Left", "front"),
        orient("right", "Right", "front"),
        ToolCommand("lock", "Rotation lock", "lock2d", group="Appearance",
                    kind="toggle", activate=view.set_rotation_locked),
    ]


class NavToolbar(QFrame):
    """A flat, dockable strip of viewport tool buttons (hosted in a toolbar row
    by the shell, so it never overlaps the canvas)."""

    def __init__(self, view, parent=None):
        super().__init__(parent)
        self._view = view
        self.setObjectName("viewportTools")
        self.setCursor(Qt.CursorShape.ArrowCursor)
        # a stable band height so the hosting toolbar row sizes correctly even
        # though the strip is built while hidden (before the shell docks it)
        self.setMinimumHeight(34)
        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(6, 3, 6, 3)
        self._lay.setSpacing(2)

        self._registry: dict[str, ToolCommand] = {}
        self._buttons_by_id: dict[str, QToolButton] = {}
        self._separators: list[QFrame] = []      # inner divider lines (re-inked)
        self._group: QButtonGroup | None = None
        self.register_many(_view_commands(view))
        self._layout = self._load_layout()
        self.rebuild()

        view.mode_changed.connect(self.set_active_tool)
        view.rotation_lock_changed.connect(self._reflect_lock)

    # -- command registry ----------------------------------------------------
    def register_many(self, cmds) -> None:
        for c in cmds:
            self._registry[c.id] = c

    # -- layout persistence --------------------------------------------------
    def _load_layout(self) -> list:
        v = QSettings("MidasStructural", "Desktop").value(_SETTINGS_KEY)
        if isinstance(v, str) and v:
            layout = v.split(",")
        elif isinstance(v, (list, tuple)) and v:
            layout = [str(x) for x in v]
        else:
            return list(_DEFAULT_LAYOUT)
        if layout in _SUPERSEDED_DEFAULTS:         # auto-upgrade an untouched bar
            return list(_DEFAULT_LAYOUT)
        return layout

    def _save_layout(self) -> None:
        QSettings("MidasStructural", "Desktop").setValue(
            _SETTINGS_KEY, ",".join(self._layout))

    # -- (re)build the visible bar ------------------------------------------
    def rebuild(self) -> None:
        while self._lay.count():
            it = self._lay.takeAt(0)
            w = it.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        if self._group is not None:
            self._group.deleteLater()
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons_by_id = {}
        self._separators = []
        for cid in self._layout:
            if cid == SEPARATOR_ID:
                self._lay.addWidget(self._separator())
                continue
            cmd = self._registry.get(cid)
            if cmd is None:
                continue                       # unknown id (not yet registered)
            b = self._make_button(cmd)
            self._lay.addWidget(b)
            self._buttons_by_id[cid] = b
            if cmd.kind == "tool":
                self._group.addButton(b)
        self._lay.addWidget(self._separator())
        self._lay.addWidget(self._make_customize_button())
        self.adjustSize()
        self._sync_all()

    def _make_button(self, cmd: ToolCommand) -> QToolButton:
        b = QToolButton(self)
        b.setProperty("_cmdId", cmd.id)
        b.setIcon(icons.icon(cmd.icon, style.ICON))
        b.setIconSize(QSize(18, 18))
        b.setAutoRaise(True)
        b.setCheckable(cmd.kind in ("tool", "toggle"))
        b.setToolTip(cmd.label)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        if cmd.kind == "toggle":
            b.toggled.connect(lambda on, c=cmd: c.activate(on))
        else:
            b.clicked.connect(lambda _c=False, c=cmd: c.activate())
        return b

    def _separator(self) -> QWidget:
        """A vertical divider with breathing room on each side, so tool groups
        read as distinct clusters (the Midas-style rule between families)."""
        box = QWidget(self)
        box.setFixedWidth(13)
        lay = QHBoxLayout(box)
        lay.setContentsMargins(6, 3, 6, 3)
        lay.setSpacing(0)
        line = QFrame(box)
        line.setObjectName("navSep")
        line.setFixedWidth(1)
        line.setStyleSheet(f"#navSep{{background:{style.BORDER_STRONG};}}")
        lay.addWidget(line)
        self._separators.append(line)
        return box

    def _make_customize_button(self) -> QToolButton:
        """A persistent trailing affordance so Customize is discoverable without
        knowing to right-click. Clicking drops a small menu (Customize / Reset).
        Not part of the customizable layout, so it can never be removed."""
        b = QToolButton(self)
        b.setObjectName("navCustomize")
        b.setIcon(icons.icon("customize", style.ICON))
        b.setIconSize(QSize(18, 18))
        b.setAutoRaise(True)
        b.setToolTip("Customize toolbar — add, remove or reorder buttons")
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        b.setMenu(self._toolbar_menu(b))
        self._customize_btn = b
        return b

    # -- state sync ----------------------------------------------------------
    def _sync_all(self) -> None:
        mode = self._view.current_mode() if hasattr(self._view,
                                                    "current_mode") else "select"
        self.set_active_tool(mode)
        self._reflect_lock(self._view.rotation_locked())

    def set_active_tool(self, mode: str) -> None:
        """Reflect the view's active tool; a non-tool mode (draw / a ribbon-only
        tool) leaves none of our tool buttons pressed."""
        if self._group is None:
            return
        self._group.setExclusive(False)
        for cid, b in self._buttons_by_id.items():
            cmd = self._registry.get(cid)
            if cmd and cmd.kind == "tool":
                b.setChecked(cid == mode)
        self._group.setExclusive(True)

    def _reflect_lock(self, locked: bool) -> None:
        b = self._buttons_by_id.get("lock")
        if b is not None:
            b.blockSignals(True)
            b.setChecked(locked)
            b.blockSignals(False)
            b.setIcon(icons.icon("lock2d" if locked else "unlock2d", style.ICON))
            b.setToolTip(
                "Rotation locked (planar model) — click to allow 3-D orbit"
                if locked else "Rotation free — click to lock to the plane")
        orb = self._buttons_by_id.get("orbit")
        if orb is not None:
            orb.setEnabled(not locked)
            if locked and orb.isChecked():
                sel = self._buttons_by_id.get("select")
                if sel is not None:
                    sel.setChecked(True)

    # -- customize -----------------------------------------------------------
    def _toolbar_menu(self, parent=None) -> QMenu:
        menu = QMenu(parent or self)
        menu.addAction("Customize Toolbar…").triggered.connect(
            self._open_customize)
        menu.addAction("Reset Toolbar").triggered.connect(self._reset)
        return menu

    def contextMenuEvent(self, ev):
        self._toolbar_menu().exec(ev.globalPos())

    def _open_customize(self) -> None:
        from toolbar_commands import CustomizeToolbarDialog
        dlg = CustomizeToolbarDialog(self._registry, self._layout,
                                     _DEFAULT_LAYOUT, self.window())
        if dlg.exec():
            self._layout = dlg.result_layout()
            self._save_layout()
            self.rebuild()

    def _reset(self) -> None:
        self._layout = list(_DEFAULT_LAYOUT)
        self._save_layout()
        self.rebuild()

    # -- theme ---------------------------------------------------------------
    def apply_theme(self) -> None:
        """Re-ink every button + divider for the current palette (called on a
        theme switch, mirroring the shell's ``_retheme_icons``)."""
        for cid, b in self._buttons_by_id.items():
            if cid == "lock":                  # icon depends on lock state
                continue
            cmd = self._registry.get(cid)
            if cmd:
                b.setIcon(icons.icon(cmd.icon, style.ICON))
        if getattr(self, "_customize_btn", None) is not None:
            self._customize_btn.setIcon(icons.icon("customize", style.ICON))
        for line in self._separators:
            line.setStyleSheet(f"#navSep{{background:{style.BORDER_STRONG};}}")
        self._reflect_lock(self._view.rotation_locked())
        self.update()
