"""Modeless "pick from the model" support for node / member dialogs.

Commercial FE tools (SAP2000 / CSiBridge / MIDAS) let you click a joint or a
frame *in the model* while an assignment dialog is open, instead of hunting for
its id in a dropdown. Our dialogs were all **modal** (``QDialog.exec``), which
grabs the whole application — so the 3-D viewport could not receive a single
click until the dialog was closed ("it is asking me to close the window to
select").

This module makes such a dialog **modeless but still synchronous**:
:class:`PickDialog` reimplements ``exec`` so the dialog is shown *without* an
input grab and a private event loop is spun instead. Every existing caller keeps
its return contract unchanged — ``LoadDialog.edit`` still returns a ``Load``,
``if dlg.exec():`` still branches on OK/Cancel — while the viewport stays live
underneath and a click on a node/member flows into the dialog.

A dialog opts in by:

* subclassing :class:`PickDialog` instead of ``QDialog``; and
* declaring its pickable combos with :meth:`PickDialog.register_pick_field`
  (``"node"`` / ``"member"`` + the combo whose ``itemData`` is the id, i.e. one
  built with ``addItem(label, id)``).

While it is open the dialog registers itself as the *pick sink* on the host
window — any object up the parent chain that exposes ``push_pick_sink`` /
``pop_pick_sink`` (the :class:`main_window.MainWindow`). A viewport pick of a
matching kind then fills the *armed* field (the pick-combo that last had focus,
else the first one of that kind) rather than changing the global selection.

Nothing here needs OpenGL, so dialogs stay headless-constructible under
``QT_QPA_PLATFORM=offscreen``; when there is no host (e.g. ``parent=None`` in a
unit test, or a dialog opened from inside another modal dialog) picking is simply
inert and the dialog behaves like the modal one it replaced.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QEventLoop, Qt
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QListWidget)


class PickDialog(QDialog):
    """A :class:`QDialog` whose ``exec`` is modeless, so the viewport behind it
    can be clicked to pick node/member fields registered with
    :meth:`register_pick_field`."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Each entry is (kind, widget); widget is a QComboBox (single value) or a
        # QListWidget (multi-select membership toggled by each pick).
        self._pick_fields: list[tuple[str, object]] = []
        self._armed_field: object | None = None
        self._pick_loop: QEventLoop | None = None
        self._pick_host = None
        # Float above the viewport so the model stays visible while picking; a
        # modal dialog did not need this because nothing behind it was usable.
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.finished.connect(self._on_pick_finished)

    # -- field registration --------------------------------------------------
    def register_pick_field(self, kind: str, widget) -> None:
        """Declare *widget* as pickable: a viewport click on a *kind* item
        (``"node"`` / ``"member"``) feeds it.

        * a :class:`QComboBox` has its current item set to the pick (its
          ``itemData`` must be the id — build it with ``addItem(label, id)``);
        * a :class:`QListWidget` (multi-select set, e.g. "apply to members")
          toggles the picked item's membership, so clicking items in the model
          builds the set.

        Call after the widget exists; the first one registered is armed by
        default. The armed one (the pick-widget that last had focus) receives
        picks, so a dialog with several fields of the same kind still works."""
        if kind not in ("node", "member") or widget is None:
            return
        if not isinstance(widget, (QComboBox, QListWidget)):
            return
        self._pick_fields.append((kind, widget))
        if self._armed_field is None:
            self._armed_field = widget
        widget.installEventFilter(self)         # arm this field when it gets focus
        vp = getattr(widget, "viewport", None)  # QListWidget focus lands on viewport
        if callable(vp):
            vp().installEventFilter(self)

    def eventFilter(self, obj, ev):             # noqa: N802 (Qt override)
        if ev.type() == QEvent.Type.FocusIn:
            for _kind, widget in self._pick_fields:
                vp = getattr(widget, "viewport", None)
                if obj is widget or (callable(vp) and obj is vp()):
                    self._armed_field = widget
                    break
        return super().eventFilter(obj, ev)

    def has_pick_fields(self) -> bool:
        return bool(self._pick_fields)

    # -- pick sink protocol (the host calls this on a viewport pick) ----------
    def accepts_kind(self, kind) -> bool:
        return any(k == kind for k, _ in self._pick_fields)

    def accept_pick(self, kind, ident) -> bool:
        """Feed *ident* to the armed (or first matching) field. Returns True if
        it was consumed, so the host leaves the global selection untouched."""
        fields = [w for (k, w) in self._pick_fields if k == kind]
        if not fields:
            return False
        widget = self._armed_field if self._armed_field in fields else fields[0]
        if isinstance(widget, QComboBox):
            return self._fill_combo(widget, ident)
        if isinstance(widget, QListWidget):
            return self._toggle_list(widget, ident)
        return False

    @staticmethod
    def _match_data(get, count, ident) -> int:
        """Index of the item whose data equals *ident* (int-tolerant), else -1."""
        cands = [ident]
        try:
            cands.append(int(ident))
        except (TypeError, ValueError):
            pass
        for i in range(count):
            if get(i) in cands:
                return i
        return -1

    def _fill_combo(self, combo: QComboBox, ident) -> bool:
        i = combo.findData(ident)
        if i < 0:
            i = self._match_data(combo.itemData, combo.count(), ident)
        if i < 0:                               # not a selectable option (filtered)
            return False
        combo.setCurrentIndex(i)
        return True

    def _toggle_list(self, lst: QListWidget, ident) -> bool:
        role = Qt.ItemDataRole.UserRole
        i = self._match_data(lambda k: lst.item(k).data(role), lst.count(), ident)
        if i < 0:
            return False
        item = lst.item(i)
        item.setSelected(not item.isSelected())  # click in model = add / remove
        lst.scrollToItem(item)                    # reveal it; don't touch current
        return True                               # (setCurrentItem would reselect)

    # -- modeless-but-synchronous exec --------------------------------------
    def _find_pick_host(self):
        w = self.parent()
        while w is not None:
            if hasattr(w, "push_pick_sink") and hasattr(w, "pop_pick_sink"):
                return w
            w = w.parent() if callable(getattr(w, "parent", None)) else None
        return None

    def exec(self):                             # noqa: A003 (Qt name)
        """Show the dialog modeless and block on a private event loop, so the
        viewport keeps receiving clicks while we wait for OK/Cancel. Returns the
        dialog result code exactly like ``QDialog.exec``.

        If another **modal** dialog is already up (e.g. this dialog was opened
        from the modal Analysis-cases manager), fall back to a normal modal
        ``exec``: Qt would otherwise *block* a modeless window shown over an
        application-modal one, leaving it unusable. Picking is then simply
        unavailable in that nested context — exactly the previous behaviour."""
        if self.isVisible():                    # re-entrant / already up
            return super().exec()
        blocking = QApplication.activeModalWidget()
        if blocking is not None and blocking is not self:
            return super().exec()
        self.setModal(False)
        self._pick_host = self._find_pick_host()
        if self._pick_host is not None:
            self._pick_host.push_pick_sink(self)
        self.show()
        self.raise_()
        self.activateWindow()
        loop = QEventLoop()
        self._pick_loop = loop
        try:
            loop.exec()
        finally:
            self._pick_loop = None
            host, self._pick_host = self._pick_host, None
            if host is not None:
                host.pop_pick_sink(self)
        return self.result()

    exec_ = exec                                # Qt5-style alias, defensively

    def _on_pick_finished(self, _result) -> None:
        if self._pick_loop is not None:
            self._pick_loop.quit()
