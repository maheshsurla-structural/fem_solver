"""The on-viewport navigation toolbar — a customizable tool strip pinned to the
top-left of the 3-D viewport (companion to the orientation cube top-right).

It surfaces the camera controls that were otherwise invisible mouse gestures
(**Orbit / Pan / Zoom-window**, **Fit / Fit-selected**, **zoom ±**, a **2-D
lock**), and — like Midas Civil's toolbar — **right-click → Customize Toolbar**
lets the user choose which command icons live on the bar and in what order.
The chosen layout persists in ``QSettings``.

Built-in view commands (tools, orient, zoom) are registered here; the shell adds
model/edit commands via :meth:`register_many`. The bar drives the host
:class:`ModelView` and stays in step with it through the view's ``mode_changed``
/ ``rotation_lock_changed`` signals, so the ribbon and this bar never disagree.
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, QSettings, QSize, Qt
from PySide6.QtGui import QAction, QActionGroup, QColor, QPainter, QPen
from PySide6.QtWidgets import (QButtonGroup, QFrame, QHBoxLayout, QMenu,
                               QToolButton)

import icons
import style
from toolbar_commands import SEPARATOR_ID, ToolCommand

_SETTINGS_KEY = "viewport_toolbar_layout"

# the out-of-the-box bar — the *selection* container (single / window /
# polygon), the camera tools, then the MIDAS-style *activation* container
# (inactivate / isolate / show-all / invert) and the 2-D lock. ``grp_select`` is
# a view-level container; ``grp_active`` is registered by the shell (MainWindow),
# so a bare view skips it until the shell adds it (``rebuild`` ignores unknowns).
_DEFAULT_LAYOUT = ["grp_select", SEPARATOR_ID,
                   "orbit", "pan", "zoomwin", SEPARATOR_ID,
                   "fit", "fitsel", "zoomin", "zoomout", SEPARATOR_ID,
                   "grp_active", SEPARATOR_ID, "lock"]

# Earlier default bars — a saved layout that still matches one of these is an
# *untouched* default, so it is silently upgraded to ``_DEFAULT_LAYOUT`` (the
# new containers appear) while a genuinely customized bar is left as saved.
_SUPERSEDED_DEFAULTS = [
    ["select", "orbit", "pan", "zoomwin", SEPARATOR_ID,
     "fit", "fitsel", "zoomin", "zoomout", SEPARATOR_ID, "lock"],
    ["select", "window", "polygon", SEPARATOR_ID,
     "orbit", "pan", "zoomwin", SEPARATOR_ID,
     "fit", "fitsel", "zoomin", "zoomout", SEPARATOR_ID,
     "cmd_inactivate", "cmd_activate_only", "cmd_activate_all",
     "cmd_invert_active", SEPARATOR_ID, "lock"],
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
        # a container flyout gathering the three selection tools under one
        # button (its members are registered above, so their modes still fire)
        ToolCommand("grp_select", "Selection tools", "single", group="Tools",
                    kind="group", members=["select", "window", "polygon"]),
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
    _PAD = 7                                   # room around the panel for shadow

    def __init__(self, view, parent=None):
        super().__init__(parent or view)
        self._view = view
        self.setObjectName("canvasBar")        # reuse the button QSS (hover…)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        # panel + shadow are custom-painted (a QSS frame background does not
        # paint reliably over the native GL viewport); keep the frame itself
        # transparent and draw everything in paintEvent.
        self.setStyleSheet(
            "QFrame#canvasBar { background: transparent; border: none; }")
        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(self._PAD + 5, self._PAD + 3,
                                     self._PAD + 5, self._PAD + 3)
        self._lay.setSpacing(2)

        self._registry: dict[str, ToolCommand] = {}
        self._buttons_by_id: dict[str, QToolButton] = {}
        self._group: QButtonGroup | None = None
        # for each container flyout, the member last chosen (drives its face
        # icon + the main-click action); survives a rebuild.
        self._group_state: dict[str, str] = {}
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
            if cmd.kind == "tool" or (cmd.kind == "group"
                                      and self._is_tool_group(cmd)):
                self._group.addButton(b)       # share the exclusive tool group
        self._lay.addWidget(self._separator())
        self._lay.addWidget(self._make_customize_button())
        self.adjustSize()
        self._sync_all()
        parent = self.parent()
        if hasattr(parent, "_position_nav_bar"):
            parent._position_nav_bar()

    def _make_button(self, cmd: ToolCommand) -> QToolButton:
        if cmd.kind == "group":
            return self._make_group_button(cmd)
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

    # -- container flyouts ---------------------------------------------------
    def _members(self, cmd: ToolCommand) -> list:
        """The registered member commands of a group, in order (skips any not
        yet registered)."""
        out = []
        for mid in (cmd.members or []):
            m = self._registry.get(mid)
            if m is not None:
                out.append(m)
        return out

    def _is_tool_group(self, cmd: ToolCommand) -> bool:
        """A group of interaction *tools* (exclusive, checkable, mode-driven) —
        as opposed to a group of momentary *actions* (e.g. activate/inactivate).
        Empty groups count as action groups (nothing to reflect)."""
        members = self._members(cmd)
        return bool(members) and all(m.kind == "tool" for m in members)

    def _current_member(self, cmd: ToolCommand):
        """The member a tool-group's face currently shows / its main-click runs;
        seeded to the first member."""
        members = self._members(cmd)
        if not members:
            return None
        cur = self._group_state.get(cmd.id)
        by_id = {m.id: m for m in members}
        if cur in by_id:
            return by_id[cur]
        self._group_state[cmd.id] = members[0].id
        return members[0]

    def _make_group_button(self, cmd: ToolCommand) -> QToolButton:
        b = QToolButton(self)
        b.setProperty("_cmdId", cmd.id)
        b.setIconSize(QSize(18, 18))
        b.setAutoRaise(True)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        tool_group = self._is_tool_group(cmd)
        menu = QMenu(b)
        member_actions: dict[str, QAction] = {}
        agroup = QActionGroup(b) if tool_group else None
        if agroup is not None:
            agroup.setExclusive(True)
        for m in self._members(cmd):
            act = QAction(icons.icon(m.icon, style.ICON), m.label, b)
            act.setData(m.id)
            if tool_group:
                act.setCheckable(True)
                agroup.addAction(act)
            act.triggered.connect(
                lambda _c=False, c=cmd, mm=m: self._activate_member(c, mm))
            menu.addAction(act)
            member_actions[m.id] = act
        b.setMenu(menu)
        b._memberActions = member_actions        # for state + retheme
        if tool_group:
            b.setCheckable(True)
            b.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
            b.clicked.connect(
                lambda _c=False, c=cmd: self._activate_member(
                    c, self._current_member(c)))
        else:
            b.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._paint_group_button(b, cmd)
        return b

    def _activate_member(self, cmd: ToolCommand, member) -> None:
        """Run a flyout member and (for a tool group) adopt it as the container's
        current tool. A tool's ``activate`` re-enters ``set_active_tool`` via the
        view's ``mode_changed``, which repaints the face; action members just
        run and leave the face as the group glyph."""
        if member is None:
            return
        if member.kind == "tool":
            self._group_state[cmd.id] = member.id
        if member.activate is not None:
            member.activate()
        b = self._buttons_by_id.get(cmd.id)
        if b is not None:
            self._paint_group_button(b, cmd)

    def _paint_group_button(self, b: QToolButton, cmd: ToolCommand) -> None:
        """Refresh a flyout's face: a tool group shows its current member's icon
        and ticks that member in the menu; an action group shows the group glyph.
        """
        if self._is_tool_group(cmd):
            cur = self._current_member(cmd)
            icon_name = cur.icon if cur else cmd.icon
            b.setIcon(icons.icon(icon_name, style.ICON))
            b.setToolTip(
                f"{cur.label} — {cmd.label} (▾ for more)" if cur else cmd.label)
            actions = getattr(b, "_memberActions", {})
            for mid, act in actions.items():
                act.setChecked(cur is not None and mid == cur.id)
        else:
            b.setIcon(icons.icon(cmd.icon, style.ICON))
            b.setToolTip(f"{cmd.label} (▾)")

    def _separator(self) -> QFrame:
        s = QFrame(self)
        s.setFixedWidth(9)
        return s

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
        """Reflect the view's active tool; a non-tool mode (ribbon draw/window)
        leaves none of our tool buttons pressed."""
        if self._group is None:
            return
        self._group.setExclusive(False)
        for cid, b in self._buttons_by_id.items():
            cmd = self._registry.get(cid)
            if not cmd:
                continue
            if cmd.kind == "tool":
                b.setChecked(cid == mode)
            elif cmd.kind == "group" and self._is_tool_group(cmd):
                members = cmd.members or []
                active = mode in members
                b.setChecked(active)
                if active:
                    self._group_state[cid] = mode
                self._paint_group_button(b, cmd)
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
                orb.setChecked(False)          # the view falls back to Select

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

    # -- painting ------------------------------------------------------------
    def paintEvent(self, ev):
        """Opaque rounded panel + soft drop shadow so the bar reads as a card
        floating above the viewport (independent of QSS-over-GL)."""
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        panel = QRectF(self.rect()).adjusted(self._PAD, self._PAD,
                                             -self._PAD, -self._PAD)
        p.setPen(Qt.PenStyle.NoPen)
        for grow, alpha in ((6, 18), (4, 26), (2, 34)):
            p.setBrush(QColor(0, 0, 0, alpha))
            p.drawRoundedRect(panel.adjusted(-grow, -grow + 2, grow, grow + 2),
                              10, 10)
        p.setBrush(QColor(style.PANEL))
        p.setPen(QPen(QColor(style.BORDER_STRONG), 1))
        p.drawRoundedRect(panel, 9, 9)
        p.end()
        super().paintEvent(ev)

    def apply_theme(self) -> None:
        """Re-ink every button for the current palette (called on a theme
        switch, mirroring the shell's ``_retheme_icons``)."""
        for cid, b in self._buttons_by_id.items():
            if cid == "lock":                  # icon depends on lock state
                continue
            cmd = self._registry.get(cid)
            if not cmd:
                continue
            if cmd.kind == "group":            # face + every flyout member icon
                for mid, act in getattr(b, "_memberActions", {}).items():
                    m = self._registry.get(mid)
                    if m is not None:
                        act.setIcon(icons.icon(m.icon, style.ICON))
                self._paint_group_button(b, cmd)
            else:
                b.setIcon(icons.icon(cmd.icon, style.ICON))
        if getattr(self, "_customize_btn", None) is not None:
            self._customize_btn.setIcon(icons.icon("customize", style.ICON))
        self._reflect_lock(self._view.rotation_locked())
        self.update()
