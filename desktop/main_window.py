"""The application shell: a docked-panel main window over the 3-D viewport.

The native version of the AdSec-style layout — a central viewport, a left
model tree, a bottom output log — the frame every future view (analysis,
design, drawings) docks into. Edits act on the ``Project`` (the source of
truth); the solver ``Model`` is recompiled and re-rendered after each change.
"""
from __future__ import annotations

import copy

from PySide6.QtCore import (QEvent, QItemSelectionModel, QSettings, QSize, Qt,
                            Signal)
from PySide6.QtGui import (QAction, QActionGroup, QBrush, QColor, QFont,
                           QKeySequence, QShortcut, QUndoStack)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox,
                               QDockWidget, QDoubleSpinBox, QFileDialog,
                               QFrame, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QMainWindow,
                               QMenu, QMessageBox, QPlainTextEdit, QSizePolicy,
                               QSlider, QStackedWidget, QTabBar, QToolButton,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                               QWidget)

import icons
import model_geometry as mg
import style
from commands import EditCommand
from editing import (AreaDialog, DiaphragmDialog, LoadDialog, MemberDialog,
                     NodeDialog, SectionDialog, dof_labels)
from hinge_editor import HingeAssignmentDialog, HingeManagerDialog
from material_editor import MaterialManagerDialog
from model_view import ModelView
from project import Material, Member, Node, Project, Section
from properties import PropertiesPanel
from units import Quantity, UnitSystem

PRODUCT_MONO = "FS"              # femsolver desktop — titlebar / taskbar mark
PRODUCT_ACCENT = "#2563eb"       # stable brand blue (independent of theme)

# Category rows in the model tree tag themselves with ("cat", key) in this role
# so the context menu / double-click know which table or manager to open,
# without carrying a selection ref (which lives in Qt.UserRole on the leaves).
CAT_ROLE = Qt.ItemDataRole.UserRole + 1
# Stable per-branch key (nav plan N3) for persisting expand/collapse state
# across rebuilds and sessions — set on every expandable branch item.
KEY_ROLE = Qt.ItemDataRole.UserRole + 2
# Branches expanded on first run (before the user has chosen): the four
# super-groups + Elements (so its by-type rows show). Everything else collapsed.
_DEFAULT_EXPANDED = {"grp:Properties", "grp:Structures", "grp:Loads",
                     "grp:Analysis", "cat:elements"}


def _cat_key_str(cat_key) -> str:
    """Flatten a category key (str or ``("elements", "Beam")``) into a stable
    string for the persisted expand/collapse set (nav N3)."""
    if isinstance(cat_key, tuple):
        return ":".join(str(x) for x in cat_key)
    return str(cat_key)


def _element_type(m) -> str:
    """Friendly element-type label for the tree's by-type grouping (nav B3)."""
    if getattr(m, "hinge", None) is not None:
        return "Fiber hinge"
    kind = (getattr(m, "kind", "") or "").lower()
    if kind.startswith("truss"):
        return "Truss"
    if kind.startswith("beam"):
        return "Beam"
    return (kind.replace("2d", "").replace("3d", "").title() or "Element")


class _ClickableLabel(QLabel):
    """A status-bar label that emits ``clicked`` — used for the units chip so
    the whole app's display units are one click away (plan U4)."""
    clicked = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, ev) -> None:            # noqa: N802 (Qt override)
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(ev)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1280, 820)
        # restore the persisted theme + density BEFORE any widget (the viewport
        # reads style.VIEW_BG in its own __init__), so the app comes up right.
        self._settings = QSettings("MidasStructural", "Desktop")
        style.set_theme(str(self._settings.value("theme", "light")))
        style.set_density(str(self._settings.value("density", "comfortable")))
        self.setWindowIcon(icons.monogram_icon(PRODUCT_MONO, "#ffffff",
                                               PRODUCT_ACCENT))
        self._model = None
        self._project = None
        self._path = None
        self._undo_stack = QUndoStack(self)

        self.view = ModelView(self)
        # Central area = a thin viewport tool strip *above* the 3-D canvas. The
        # strip is created by the view but re-homed here so it docks in its own
        # band off the canvas — a click on a tool never bleeds through to the
        # model (which a floating overlay did).
        central = QWidget(self)
        central.setObjectName("viewportCentral")
        cl = QVBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        self.view._nav_bar.show()
        cl.addWidget(self.view._nav_bar)
        cl.addWidget(self.view, 1)
        self.setCentralWidget(central)

        # Model tree — a compact *table of contents* (nav plan N1): category
        # rows with a count in a narrow second column, individual items as
        # collapsed leaves. Header hidden; the count column hugs its content.
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderHidden(True)
        self.tree.setUniformRowHeights(True)
        hdr = self.tree.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        self.tree.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_tree_menu)
        # Persist which branches the user expands (nav plan N3): a full rebuild
        # would otherwise reset the outline on every edit. Keyed per branch
        # (super-group / category / element-type); seeded on first run.
        self._building_tree = False
        self._last_tree_sig = None       # structure signature for the N5 fast path
        # Authoritative selection (nav N6): the tree is only one *driver* of it
        # (in leaf mode); in summary mode the viewport / tables drive it. Every
        # consumer (move/copy/delete/Properties/highlight) reads _selected_refs.
        self._selection: list = []
        # View/edit *working set* (MIDAS-style activation): refs the user has
        # made inactive. Inactive entities are hidden from the viewport and
        # excluded from picking/selection, but the analysis model is always
        # built from the full project — deactivating never changes results.
        # A pure view/session concept, so it lives here (not on the project) and
        # is not part of the undo stack.
        self._inactive: set = set()
        # Open node/member dialogs register here (as a stack, so a nested one
        # wins) so a viewport click flows into the dialog instead of changing the
        # selection — the modeless "pick from the model" path (see pick.py).
        self._pick_sinks: list = []
        self._summary_mode = self._settings.value(
            "nav/summary", False, type=bool)
        saved = self._settings.value("nav/expanded", None)
        self._expanded = (set(saved) if saved is not None
                          else set(_DEFAULT_EXPANDED))
        self.tree.itemExpanded.connect(self._on_branch_expanded)
        self.tree.itemCollapsed.connect(self._on_branch_collapsed)

        dock_tree = QDockWidget("Model", self)
        dock_tree.setWidget(self._build_nav_panel())
        dock_tree.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                                  | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock_tree)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        dock_log = QDockWidget("Output", self)
        dock_log.setWidget(self.log)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock_log)

        # Nonlinear-results step scrubber (plan G-S1/G-S2) — hidden until a
        # pushover/case run hands results back; scrubs the whole model view
        # through the run's steps.
        self._nl_results = None
        self._nl_scale = 1.0
        self._nl_dock = self._build_nl_step_dock()
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._nl_dock)
        self._nl_dock.hide()

        self.props = PropertiesPanel(self._apply_from_inspector,
                                     self._apply_bulk, self)
        dock_props = QDockWidget("Properties", self)
        dock_props.setWidget(self.props)
        dock_props.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                                   | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_props)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.view.set_pick_callback(self._on_pick)
        self.view.set_add_node_callback(self._draw_add_node)
        self.view.set_add_member_callback(self._draw_add_member)
        self.view.set_add_area_callback(self._draw_add_area)
        self.view.set_region_callback(self._on_region_select)
        # keep the ribbon's selection group in step with the viewport's own
        # navigation toolbar (orbit / pan / zoom-window have no ribbon entry).
        self.view.mode_changed.connect(self._sync_ribbon_mode)

        self._build_menu()
        self._register_toolbar_commands()    # model/edit cmds for Customize…
        self._build_status_items()           # persistent context (right side)
        # an always-visible theme switch in the status-bar corner (the "chip")
        self._theme_btn = QToolButton()
        self._theme_btn.setAutoRaise(True)
        self._theme_btn.setIconSize(QSize(18, 18))
        self._theme_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._theme_btn.clicked.connect(self.toggle_theme)
        self.statusBar().addPermanentWidget(self._theme_btn)
        self.view.set_coord_callback(self._on_cursor_coords)
        style.apply(self)                    # unified light/blue theme
        self._sync_theme_ui()
        self._refresh_status()
        self.statusBar().showMessage("Ready")

    def toggle_theme(self) -> None:
        style.toggle_theme()
        self._settings.setValue("theme", style.current_theme())
        self._apply_theme_density()

    def toggle_density(self) -> None:
        new = "comfortable" if style.current_density() == "compact" else "compact"
        style.set_density(new)
        self._settings.setValue("density", new)
        self.act_density.setChecked(new == "compact")
        self._apply_theme_density()

    def _apply_theme_density(self) -> None:
        """Re-skin the whole shell after a theme/density change: QSS cascade,
        re-inked icons, and a viewport repaint — so nothing is left behind."""
        style.apply(self)
        self._retheme_icons()
        self.view.apply_theme()
        self._sync_theme_ui()

    def _build_status_items(self) -> None:
        """Persistent context on the right of the status bar (model summary ·
        selection · cursor coords · units), separate from the transient
        ``showMessage`` hints on the left."""
        sb = self.statusBar()

        def _lbl():
            w = QLabel("")
            w.setObjectName("sub")
            return w

        self._st_model = _lbl()
        self._st_sel = _lbl()
        self._st_coord = _lbl()
        self._st_units = _ClickableLabel("")           # click → Units dialog (U4)
        self._st_units.setObjectName("sub")
        self._st_units.setToolTip("Display units — click to change")
        self._st_units.clicked.connect(self.change_units)
        for w in (self._st_model, self._st_sel, self._st_coord, self._st_units):
            sb.addPermanentWidget(w)

    def _units(self) -> UnitSystem:
        """The active display unit system (plan U2), derived from the project's
        stored force/length preferences. SI base is always the source of truth;
        this is the presentation layer everything formats through."""
        if self._project is None:
            return UnitSystem()
        return UnitSystem.from_project(self._project)

    def change_units(self) -> None:
        """Open the Units dialog and, on accept, switch the whole app's display
        units live (plan U4). The model is untouched (it is SI base); only how
        values read and parse changes. The choice is saved on the project and
        remembered as the app default for new models."""
        if self._project is None:
            return
        from units_dialog import UnitsDialog
        us = self._units()
        chosen = UnitsDialog.get(self, us.force, us.length)
        if chosen is None:
            return
        force, length = chosen
        if (force, length) == (self._project.force_unit,
                               self._project.length_unit):
            return
        self._settings.setValue("units/force", force)   # remember app default
        self._settings.setValue("units/length", length)

        def mutate():
            self._project.force_unit = force
            self._project.length_unit = length
        # Units live on the project (saved in the file), so this is a real,
        # undoable document change — it marks the model dirty and refreshes the
        # status readouts via _rebuild. The Properties inspector re-renders too.
        self._apply_edit(f"Set units → {force} · {length}", mutate)
        self._apply_selection_effects()
        self.log.appendPlainText(f"Display units → {force} · {length}")

    def _refresh_status(self) -> None:
        """Update the persistent model-summary + units readouts from the project."""
        p = self._project
        if p is None:
            self._st_model.setText("")
            self._st_units.setText("")
            return
        self._st_model.setText(
            f"{len(p.nodes)} nodes · {len(p.members)} members · "
            f"{len(p.loads)} loads")
        self._st_units.setText(self._units().pair_label)
        self._update_sel_status(len(self._selected_refs()))   # stay consistent

    def _update_sel_status(self, n: int) -> None:
        self._st_sel.setText(f"{n} selected" if n else "")

    def _on_cursor_coords(self, x, y) -> None:
        # Cursor coords arrive in SI base (m); show them in the chosen length unit.
        us = self._units()
        xd = us.to_display(x, Quantity.LENGTH)
        yd = us.to_display(y, Quantity.LENGTH)
        self._st_coord.setText(f"X {xd:.2f}  Y {yd:.2f} {us.label(Quantity.LENGTH)}")

    def _sync_theme_ui(self) -> None:
        """Point the theme controls at the theme they switch TO."""
        dark = style.current_theme() == "dark"
        nxt = "theme_light" if dark else "theme_dark"
        tip = "Switch to light theme" if dark else "Switch to dark theme"
        self._theme_btn.setIcon(icons.icon(nxt, style.ICON))
        self._theme_btn.setToolTip(tip)
        self.act_theme.setIcon(icons.icon(nxt, style.ICON))
        self.act_theme.setToolTip(tip)

    def _retheme_icons(self) -> None:
        """Recolour every icon-bearing action to the current theme's ink
        (``style.ICON``). Actions remember their icon name via ``_set_icon``, so
        the ribbon buttons that mirror them follow automatically."""
        for act in self.findChildren(QAction):
            name = act.property("iconName")
            if name:
                act.setIcon(icons.icon(name, style.ICON))

    # --------------------------------------------------------------- ribbon nav
    def _persist_ribbon_tab(self, index: int) -> None:
        """R2 — remember the active ribbon tab so the next session reopens it."""
        self._settings.setValue("ribbon/tab", self._ribbon.tabs.tabText(index))

    def _cycle_ribbon_tab(self, delta: int) -> None:
        """Move the active ribbon tab by ``delta``, wrapping around (Ctrl+Tab)."""
        tabs = self._ribbon.tabs
        if tabs.count():
            tabs.setCurrentIndex((tabs.currentIndex() + delta) % tabs.count())

    def _install_ribbon_shortcuts(self) -> None:
        """R3 — keyboard access to the ribbon: Alt+<initial> raises a tab by its
        first letter (H/D/L/A/R/V), and Ctrl+Tab / Ctrl+Shift+Tab cycle. The
        tab strip is also focusable so the arrow keys walk it."""
        self._ribbon.tabs.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        titles = [self._ribbon.tabs.tabText(i)
                  for i in range(self._ribbon.tabs.count())]
        seen = set()
        for title in titles:
            initial = title[0].upper()
            if initial in seen:                       # keep each accelerator unique
                continue
            seen.add(initial)
            sc = QShortcut(QKeySequence(f"Alt+{initial}"), self)
            sc.activated.connect(lambda t=title: self._ribbon.set_current(t))
        for seq, delta in (("Ctrl+Tab", 1), ("Ctrl+Shift+Tab", -1)):
            sc = QShortcut(QKeySequence(seq), self)
            sc.activated.connect(lambda d=delta: self._cycle_ribbon_tab(d))
        sc = QShortcut(QKeySequence("Ctrl+F1"), self)   # R5: collapse toggle
        sc.activated.connect(self._ribbon.toggle_collapsed)

    def _show_results_tab(self) -> None:
        """R4 — after a successful run, surface the Results tab so the diagrams
        are one click away. A no-op when the ribbon is collapsed (the user hid
        it deliberately) so nothing pops up unbidden."""
        rb = getattr(self, "_ribbon", None)
        if rb is not None and not rb.is_collapsed():
            rb.set_current("Results")

    # ------------------------------------------------------------- backstage (R8)
    def _open_backstage(self) -> None:
        """Show the File backstage overlay, built lazily from the ribbon's file
        actions and refreshed with the current recent-files list."""
        if getattr(self, "_backstage", None) is None:
            from backstage import Backstage
            self._backstage = Backstage(
                self, list(self._ribbon.file_menu.actions()),
                product="femsolver desktop",
                monogram=icons.monogram_icon(PRODUCT_MONO, "#ffffff",
                                             PRODUCT_ACCENT))
            self._backstage.openRecentRequested.connect(self._open_recent)
        self._backstage.set_recent(self._recent_files())
        self._backstage.setGeometry(self.rect())
        self._backstage.show()
        self._backstage.raise_()
        self._backstage.setFocus()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        bs = getattr(self, "_backstage", None)
        if bs is not None and bs.isVisible():
            bs.setGeometry(self.rect())

    # --------------------------------------------------------------- menu / UI
    def _build_menu(self) -> None:
        self.act_undo = self._undo_stack.createUndoAction(self, "&Undo")
        self.act_undo.setShortcut("Ctrl+Z")
        _set_icon(self.act_undo, "undo")
        self.act_redo = self._undo_stack.createRedoAction(self, "&Redo")
        self.act_redo.setShortcut("Ctrl+Y")
        _set_icon(self.act_redo, "redo")

        self.act_new = _action(self, "&New", "Ctrl+N", self.new_project, "new")
        self.act_new3d = _action(self, "New &3-D frame", None,
                                 self.new_project_3d, "new3d")
        self.act_gen = _action(self, "&Generate frame…", "Ctrl+G",
                               self.generate_frame, "frame")
        self.act_open = _action(self, "&Open…", "Ctrl+O", self.open_project,
                                "open")
        self.act_save = _action(self, "&Save", "Ctrl+S", self.save_project,
                                "save")
        self.act_saveas = _action(self, "Save &As…", "Ctrl+Shift+S",
                                  self.save_project_as, "save")

        self.act_add_node = _action(self, "Add &node…", "Ctrl+Shift+N",
                                    self.add_node, "node")
        self.act_add_member = _action(self, "Add &member…", "Ctrl+Shift+M",
                                      self.add_member, "member")
        self.act_add_area = _action(self, "Add &area…", None,
                                    self.add_area, "slab")
        self.act_add_diaphragm = _action(self, "Add &diaphragm…", None,
                                         self.add_diaphragm, "diaphragm")
        self.act_add_load = _action(self, "Add &load…", None, self.add_load,
                                    "load")
        self.act_add_lineload = _action(self, "Add l&ine load…", None,
                                        self.add_line_load, "load")
        self.act_add_areaload = _action(self, "Add a&rea load…", None,
                                        self.add_area_load, "load")
        self.act_add_section = _action(self, "Add &section…", None,
                                       self.add_section, "section")
        self.act_shell_sections = _action(self, "&Thickness…", None,
                                          self.manage_shell_sections, "slab")
        self.act_materials = _action(self, "&Materials…", None,
                                     self.manage_materials, "materials")
        self.act_stories = _action(self, "Stories && grid…", None,
                                   self.manage_stories, "grid")
        self.act_replicate_story = _action(self, "&Replicate story…", None,
                                           self.replicate_story, "copy")
        self.act_hinges = _action(self, "&Hinges…", None, self.manage_hinges,
                                  "hinge")
        self.act_th_functions = _action(self, "Time-history &functions…", None,
                                        self.manage_th_functions, "function")
        self.act_assign_hinges = _action(self, "Assign &hinges…", None,
                                         self.assign_hinges, "assignhinge")
        self.act_genloads = _action(self, "Generate &loads…", None,
                                    self.generate_loads, "loadsgen")
        self.act_delete = _action(self, "&Delete", "Del", self.delete_selected,
                                  "delete")
        self.act_move = _action(self, "&Move…", None, self.move_selected, "move")
        self.act_copy = _action(self, "Cop&y / array…", None, self.copy_selected,
                                "copy")
        self.act_mirror = _action(self, "Mirro&r…", None, self.mirror_selected,
                                  "mirror")
        self.act_rotate = _action(self, "Ro&tate…", None, self.rotate_selected,
                                  "rotate")
        self.act_extrude = _action(self, "E&xtrude…", None, self.extrude_selected,
                                   "extrude")

        self.act_analysiscases = _action(self, "Analysis &cases…", None,
                                         self.manage_analysis_cases, "run")
        # Ctrl+R opens the Run-analysis chooser (E3b: Linear Static is now a
        # saved case listed there, not a one-click launcher).
        self.act_run = _action(self, "&Run analysis…", "Ctrl+R",
                               self.run_analysis, "run")
        self.act_pushover = _action(self, "Nonlinear &pushover…", None,
                                    self.run_pushover_dialog, "run")
        self.act_timehistory = _action(self, "Nonlinear &time history…", None,
                                       self.run_timehistory_dialog, "run")
        self.act_runhistory = _action(self, "Run &history…", None,
                                      self.show_run_history, "history")
        self.act_export_slab = _action(self, "&Export slab results…", None,
                                       self.export_slab_results, "export")
        self.act_checkmodel = _action(self, "&Check model…", None,
                                      self.check_model, "checkmodel")
        self.act_undef = _action(self, "&Undeformed", None,
                                 self._show_undeformed, "undeformed")
        self.act_fit = _action(self, "&Fit", "F", self.view.fit, "fit")

        def _v(text, name, icon_name):
            return _action(self, text, None,
                           lambda *_a, n=name: self.view.set_view(n), icon_name)
        self.act_v_iso = _v("&Isometric", "iso", "iso")
        self.act_v_top = _v("&Top", "top", "top")
        self.act_v_bottom = _v("&Bottom", "bottom", "top")
        self.act_v_front = _v("Fro&nt", "front", "front")
        self.act_v_back = _v("Bac&k", "back", "front")
        self.act_v_left = _v("&Left", "left", "front")
        self.act_v_right = _v("&Right", "right", "front")
        self.act_deselect = _action(self, "Deselect &all", "Escape",
                                    self.deselect_all, "deselect")

        # Active / inactive working set (MIDAS-style activation) — a view/edit
        # aid that hides part of the model so you can work on the rest; the
        # analysis model is always the full project (see ``_inactive``).
        self.act_inactivate = _action(self, "&Inactivate selected", None,
                                      self.inactivate_selected, "inactivate")
        self.act_activate_only = _action(self, "Activate selected &only", None,
                                         self.activate_selected_only,
                                         "activate_only")
        self.act_activate_all = _action(self, "&Activate all", None,
                                        self.activate_all, "activate_all")
        self.act_invert_active = _action(self, "In&vert active", None,
                                         self.invert_active, "invert_active")

        self.act_diag_n = _set_icon(QAction("Axial &N", self), "axial")
        self.act_diag_n.triggered.connect(lambda *_: self.show_diagram("N"))
        self.act_diag_v = _set_icon(QAction("Shear &V", self), "shear")
        self.act_diag_v.triggered.connect(lambda *_: self.show_diagram("V"))
        self.act_diag_m = _set_icon(QAction("Moment &M", self), "moment")
        self.act_diag_m.triggered.connect(lambda *_: self.show_diagram("M"))
        self.act_area_contour = _set_icon(
            QAction("De&flection contour", self), "contour")
        self.act_area_contour.triggered.connect(
            lambda *_: self.show_deflection_contour("Umag"))
        self.act_area_forces = _set_icon(
            QAction("Shell &forces / moments", self), "shellforce")
        self.act_area_forces.triggered.connect(self.pick_and_show_area_force)
        self.act_area_rebar = _set_icon(
            QAction("Slab &reinforcement…", self), "rebar")
        self.act_area_rebar.triggered.connect(self.show_slab_reinforcement)
        self.act_punching = _set_icon(
            QAction("&Punching check…", self), "punching")
        self.act_punching.triggered.connect(self.show_punching_check)
        self.act_section_cut = _set_icon(
            QAction("Section &cut…", self), "sectioncut")
        self.act_section_cut.triggered.connect(self.show_section_cut)
        self.act_pier_forces = _set_icon(
            QAction("&Pier forces…", self), "wall")
        self.act_pier_forces.setStatusTip(
            "Pier forces — integrate the shell membrane stress into P/V/M up "
            "each wall pier (needs a wall with a pier label)")
        self.act_pier_forces.triggered.connect(self.show_pier_forces)
        self.act_wall_design = _set_icon(
            QAction("&Wall design…", self), "design")
        self.act_wall_design.setStatusTip(
            "Wall design — ACI 318-19 §18.10 check of each pier against its "
            "integrated P/V/M demand")
        self.act_wall_design.triggered.connect(self.show_wall_design)
        self.act_design = _set_icon(QAction("&Design (DCR)", self), "design")
        self.act_design.triggered.connect(self.show_design)
        self.act_loadcases = _action(self, "Load &patterns…", None,
                                     self.manage_load_cases, "load")
        self.act_editcombos = _action(self, "Load com&binations…", None,
                                      self.manage_combinations, "loadsgen")
        self.act_gencombos = _action(self, "Generate ASCE-7 &combinations", None,
                                     self.generate_combinations, "loadsgen")
        self.act_drawings = _action(self, "&Drawings…", None, self.open_drawings,
                                    "drawings")
        self.act_area_axes = _set_icon(QAction("Local a&xes", self), "axes")
        self.act_area_axes.setCheckable(True)
        self.act_area_axes.toggled.connect(
            lambda on: self.view.set_area_axes(on))
        self.act_story_grid = _set_icon(QAction("Stories && &grid", self), "grid")
        self.act_story_grid.setCheckable(True)
        self.act_story_grid.setChecked(True)
        self.act_story_grid.setStatusTip(
            "Show the named grid lines and story levels overlay")
        self.act_story_grid.toggled.connect(
            lambda on: self.view.show_story_grid(on))
        self.act_node_labels = _set_icon(QAction("&Node numbers", self), "node")
        self.act_node_labels.setCheckable(True)
        self.act_node_labels.toggled.connect(
            lambda on: self.view.set_node_labels(on))
        self.act_elem_labels = _set_icon(QAction("&Element numbers", self),
                                         "member")
        self.act_elem_labels.setCheckable(True)
        self.act_elem_labels.toggled.connect(
            lambda on: self.view.set_element_labels(on))
        self.act_sectiondesigner = _action(
            self, "&Section Designer…", None, self.open_section_designer,
            "sectiondesigner")
        self.act_theme = _action(self, "Toggle &theme (light / dark)", None,
                                 self.toggle_theme)
        self.act_density = _action(self, "Compact &density", None,
                                   self.toggle_density, "density")
        self.act_density.setCheckable(True)
        self.act_density.setChecked(style.current_density() == "compact")

        self.act_select = _set_icon(QAction("&Single select", self), "single")
        self.act_select.setCheckable(True)
        self.act_select.setChecked(True)
        self.act_select.triggered.connect(lambda: self._set_mode("select"))
        self.act_sel_window = _set_icon(QAction("&Window select", self),
                                        "window")
        self.act_sel_window.setCheckable(True)
        self.act_sel_window.triggered.connect(lambda: self._set_mode("window"))
        self.act_sel_poly = _set_icon(QAction("&Polygon select", self),
                                      "polygon")
        self.act_sel_poly.setCheckable(True)
        self.act_sel_poly.triggered.connect(lambda: self._set_mode("polygon"))
        self.act_draw_node = _set_icon(QAction("Draw n&ode", self), "drawnode")
        self.act_draw_node.setCheckable(True)
        self.act_draw_node.triggered.connect(lambda: self._set_mode("draw_node"))
        self.act_draw_member = _set_icon(QAction("Draw m&ember", self),
                                         "drawmember")
        self.act_draw_member.setCheckable(True)
        self.act_draw_member.triggered.connect(
            lambda: self._set_mode("draw_member"))
        self.act_draw_area = _set_icon(QAction("Draw a&rea", self), "slab")
        self.act_draw_area.setCheckable(True)
        self.act_draw_area.triggered.connect(
            lambda: self._set_mode("draw_area"))
        # Draw wall is a selection-driven command (extrude the selected base
        # line), not a viewport mode — so it is not part of the mode group.
        self.act_draw_wall = _set_icon(QAction("Draw &wall", self), "wall")
        self.act_draw_wall.setStatusTip(
            "Draw wall — select the base line (2+ nodes), then extrude it "
            "upward by a height into vertical wall panels")
        self.act_draw_wall.triggered.connect(self.draw_wall)
        self._mode_group = QActionGroup(self)
        for a in (self.act_select, self.act_sel_window, self.act_sel_poly,
                  self.act_draw_node, self.act_draw_member, self.act_draw_area):
            self._mode_group.addAction(a)
        self.act_snap = _set_icon(QAction("&Snap to grid", self), "snap")
        self.act_snap.setCheckable(True)
        self.act_snap.setChecked(True)
        self.act_snap.toggled.connect(self._update_snap)
        self.act_sel_all_nodes = _action(self, "Select all &nodes", None,
                                         self.select_all_nodes, "selnodes")
        self.act_sel_all_members = _action(self, "Select all &members", None,
                                           self.select_all_members, "selmembers")
        self.act_sel_all = _action(self, "Select &all", "Ctrl+A",
                                   self.select_all, "selall")
        self.act_sel_by_section = _action(self, "Select by &section…", None,
                                          self.select_by_section, "selsection")
        self.snap_spin = QDoubleSpinBox()
        self.snap_spin.setRange(0.05, 10.0)
        self.snap_spin.setSingleStep(0.05)
        self.snap_spin.setDecimals(2)
        self.snap_spin.setValue(0.5)
        self.snap_spin.setPrefix("grid ")
        self.snap_spin.setSuffix(" m")
        self.snap_spin.valueChanged.connect(lambda _v: self._update_snap())
        # work-plane selector (wall plan W1b): draw nodes on XY / XZ / YZ at an
        # offset, so walls can be started off the ground plane (e.g. an elevation)
        self.plane_combo = QComboBox()
        self.plane_combo.addItem("Plane XY", "xy")
        self.plane_combo.addItem("Plane XZ", "xz")
        self.plane_combo.addItem("Plane YZ", "yz")
        self.plane_combo.currentIndexChanged.connect(self._update_work_plane)
        self.plane_offset = QDoubleSpinBox()
        self.plane_offset.setRange(-1e6, 1e6)
        self.plane_offset.setDecimals(2)
        self.plane_offset.setSingleStep(0.5)
        self.plane_offset.setValue(0.0)
        self.plane_offset.setPrefix("@ ")
        self.plane_offset.setSuffix(" m")
        self.plane_offset.valueChanged.connect(lambda _v: self._update_work_plane())

        # ---- CSiBridge-style tabbed ribbon (plan ribbon R1) ---------------
        # One compact strip replaces BOTH the classic menu bar and the old
        # three stacked ribbon rows. A File "backstage" button plus a tab strip
        # (Home · Draw · Loads · Analysis · Results · View) whose active tab
        # swaps a single row of captioned tool-groups. Every command that used
        # to be menu-only is re-homed onto a tab here, so nothing is lost — the
        # whole top chrome drops from ~four rows to one. Buttons mirror their
        # QAction (setDefaultAction), so enabled state, re-inked icons and the
        # Ctrl-shortcuts (e.g. act_run's Ctrl+R) all stay live window-wide.
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.menuBar().hide()          # the ribbon tabs are the top-level nav

        rb = self._ribbon = RibbonBar(self)
        # File backstage — the classic File menu, reached from the File button.
        for a in (self.act_new, self.act_new3d, self.act_open, self.act_save,
                  self.act_saveas):
            rb.file_menu.addAction(a)

        rb.add_tab("Home", (
            ("Model", ((self.act_add_node, "Node"),
                       (self.act_add_member, "Member"),
                       (self.act_add_section, "Section"),
                       (self.act_shell_sections, "Thickness"),
                       (self.act_materials, "Materials"))),
            ("Constraints", ((self.act_add_diaphragm, "Diaphragm"),)),
            ("Levels", ((self.act_stories, "Stories & grid"),
                        (self.act_replicate_story, "Replicate"))),
            ("Edit", ((self.act_undo, "Undo"), (self.act_redo, "Redo"),
                      (self.act_delete, "Delete"))),
            ("Modify", ((self.act_move, "Move"), (self.act_copy, "Copy"),
                        (self.act_mirror, "Mirror"),
                        (self.act_rotate, "Rotate"),
                        (self.act_extrude, "Extrude"))),
            ("Generate", ((self.act_gen, "Frame"),)),
            ("Tools", ((self.act_sectiondesigner, "Designer"),)),
        ))
        rb.add_tab("Draw", (
            ("Draw", ((self.act_draw_node, "Node"),
                      (self.act_draw_member, "Member"),
                      (self.act_draw_area, "Area"),
                      (self.act_add_area, "Area…"),
                      (self.act_draw_wall, "Wall"),
                      (self.act_snap, "Snap"), self.snap_spin,
                      self.plane_combo, self.plane_offset)),
            ("Select", ((self.act_select, "Select"),
                        (self.act_sel_window, "Window"),
                        (self.act_sel_poly, "Poly"),
                        (self.act_deselect, "Deselect"))),
            ("Select by", ((self.act_sel_all_nodes, "Nodes"),
                           (self.act_sel_all_members, "Members"),
                           (self.act_sel_all, "All"),
                           (self.act_sel_by_section, "Section"))),
            ("Active", ((self.act_inactivate, "Inactivate"),
                        (self.act_activate_only, "Isolate"),
                        (self.act_activate_all, "Show all"),
                        (self.act_invert_active, "Invert"))),
        ))
        rb.add_tab("Loads", (
            ("Loads", ((self.act_loadcases, "Patterns"),
                       (self.act_add_load, "Nodal"),
                       (self.act_add_lineload, "Line"),
                       (self.act_add_areaload, "Area"),
                       (self.act_genloads, "Generate"))),
            ("Combinations", ((self.act_editcombos, "Combos"),
                              (self.act_gencombos, "ASCE-7"))),
        ))
        rb.add_tab("Analysis", (
            ("Analyse", ((self.act_analysiscases, "Cases"),
                         (self.act_run, "Run"))),
            ("Hinges", ((self.act_hinges, "Define"),
                        (self.act_assign_hinges, "Assign"))),
            ("Functions", ((self.act_th_functions, "Time History"),)),
        ))
        rb.add_tab("Results", (
            ("Diagrams", ((self.act_undef, "Undeformed"),
                          (self.act_diag_n, "Axial"),
                          (self.act_diag_v, "Shear"),
                          (self.act_diag_m, "Moment"),
                          (self.act_area_contour, "Deflection"),
                          (self.act_area_forces, "Shell F/M"))),
            ("Reports", ((self.act_runhistory, "History"),
                         (self.act_export_slab, "Export slab"))),
            ("Design", ((self.act_design, "Design"),
                        (self.act_area_rebar, "Slab rebar"),
                        (self.act_punching, "Punching"),
                        (self.act_section_cut, "Section cut"),
                        (self.act_checkmodel, "Check"))),
            ("Wall", ((self.act_pier_forces, "Pier forces"),
                      (self.act_wall_design, "Wall design"))),
        ))
        rb.add_tab("View", (
            ("Navigate", ((self.act_fit, "Fit"),)),
            ("Orient", ((self.act_v_iso, "Iso"), (self.act_v_top, "Top"),
                        (self.act_v_front, "Front"),
                        (self.act_v_right, "Right"), (self.act_v_left, "Left"),
                        (self.act_v_back, "Back"),
                        (self.act_v_bottom, "Bottom"))),
            ("Display", ((self.act_drawings, "Drawings"),
                         (self.act_area_axes, "Local axes"),
                         (self.act_story_grid, "Grid"))),
            ("Labels", ((self.act_node_labels, "Nodes"),
                        (self.act_elem_labels, "Elements"))),
            ("Appearance", ((self.act_theme, "Theme"),
                            (self.act_density, "Compact"))),
        ))
        # R2: reopen the ribbon on the tab the user left it on (persisted on
        # every switch below); fall back to Home if the saved name is unknown.
        titles = [rb.tabs.tabText(i) for i in range(rb.tabs.count())]
        saved = str(self._settings.value("ribbon/tab", "Home"))
        rb.set_current(saved if saved in titles else "Home")
        rb.tabs.currentChanged.connect(self._persist_ribbon_tab)
        # R5: restore the collapsed state and keep it persisted.
        rb.set_collapsed(self._settings.value("ribbon/collapsed", False,
                                              type=bool))
        rb.collapsedChanged.connect(
            lambda c: self._settings.setValue("ribbon/collapsed", c))

        host = self.addToolBar("Ribbon")
        host.setObjectName("ribbonHost")
        host.setMovable(False)
        host.setFloatable(False)
        host.addWidget(rb)

        self._install_ribbon_shortcuts()      # R3: keyboard tab access
        rb.file_btn.clicked.connect(self._open_backstage)   # R8: File backstage
        rb.set_quick_actions([self.act_save, self.act_undo,   # R11: quick access
                              self.act_redo, self.act_run])

    # ---------------------------------------------------------------- analysis
    def _solve(self):
        if self._model is None or not self._model.elements:
            self.statusBar().showMessage("Nothing to solve — add members first.")
            return None
        from femsolver import LinearStaticAnalysis
        return LinearStaticAnalysis(self._model).run()

    def run_linear_static(self, loads_applied=None):
        """Solve a linear static case.

        With ``loads_applied`` — a list of ``(pattern_id, scale)`` rows (E3b) —
        build the model under exactly those scaled load patterns (SAP's *Loads
        Applied* spec); this is how a saved :class:`LinearStaticType` case runs.
        Without it, solve the current model as built (every pattern ×1) — the
        fallback used by internal callers."""
        if loads_applied is not None:
            p = self._project
            if p is None or not p.members:
                self.statusBar().showMessage(
                    "Nothing to solve — add members first.")
                return None
            from project import normalize_loads_applied
            m = p.build_model(with_loads=False)
            p.apply_loads(m, ("applied", normalize_loads_applied(loads_applied)))
            self._model = m
        info = self._solve()
        if info is None:
            return None
        dmax = mg.max_translation(self._model)
        span = mg.model_span(self._model)
        scale = (0.08 * span / dmax) if dmax > 0 else 1.0
        self.view.show_deformed(self._visible_model(self._model), scale)
        self.log.appendPlainText(
            f"Linear static solved: neq={info.get('neq', '?')}, "
            f"max|u| = {dmax:.4e} m, deformation ×{scale:.0f}")
        self.statusBar().showMessage(
            f"Solved · max|u| {dmax:.3e} m · deformation ×{scale:.0f}")
        self._show_results_tab()               # R4: jump to Results after a run
        return info

    def run_modal(self, num_modes=None, lumped=None,
                  initial_condition=("zero",)):
        """Free-vibration modal analysis (Analysis-cases ▸ Modal).

        Builds the model (geometry + supports; loads are irrelevant to an
        eigen solve), extracts the lowest modes via
        :class:`femsolver.analysis.eigen.EigenAnalysis`, and opens the
        results table whose selection previews each mode shape on the view.
        Mass comes from material density — a zero-mass model is reported
        with a pointer to the Material editor rather than a solver error.

        With ``initial_condition = ("state", nl_case_id)`` (E2d) the modes are
        taken at the committed state of that Nonlinear Static case using the
        **tangent** stiffness ``K + K_g`` — i.e. P-Δ modal on a preloaded
        structure — instead of the unstressed elastic stiffness.
        """
        import numpy as np

        from femsolver import EigenAnalysis

        ic = initial_condition or ("zero",)
        if ic[0] == "state":
            ready = self._seed_state_model(ic[1])
            stiffness = "tangent"
        else:
            ready = self._modal_ready_model()
            stiffness = "elastic"
        if ready is None:
            return None
        model, neq = ready

        max_modes = max(1, neq - 1)
        if num_modes is None:
            from modal_dialog import ModalDialog
            cfg = ModalDialog.configure(self, max_modes=max_modes,
                                        default_modes=min(6, max_modes))
            if cfg is None:
                return None
            num_modes, lumped = cfg
        num_modes = max(1, min(int(num_modes), max_modes))
        lumped = bool(lumped)

        try:
            info = EigenAnalysis(model, num_modes=num_modes, lumped=lumped,
                                 stiffness=stiffness).run()
        except Exception as exc:                           # noqa: BLE001
            QMessageBox.warning(
                self, "Modal analysis",
                f"The eigen solve failed:\n\n{exc}")
            return None

        def _show_mode(k: int) -> None:
            for node in model.nodes.values():
                md = getattr(node, "mode_disp", None)
                if md is None:
                    continue
                node.disp = np.asarray(md, dtype=float)[:, k].copy()
            dmax = mg.max_translation(model)
            span = mg.model_span(model)
            scale = (0.08 * span / dmax) if dmax > 0 else 1.0
            self.view.show_deformed(self._visible_model(model), scale)
            self.statusBar().showMessage(
                f"Mode {k + 1}: T = {info['periods_s'][k]:.4g} s, "
                f"f = {info['frequencies_hz'][k]:.4g} Hz")

        from modal_results_dialog import ModalResultsDialog
        # hold a reference — the dialog is non-modal so the view updates live
        self._modal_results_dlg = ModalResultsDialog.show_results(
            self, info, _show_mode)

        periods = info.get("periods_s", [])
        t1 = next((t for t in periods if t and t not in (float("inf"),)), None)
        self.log.appendPlainText(
            f"Modal solved: {info.get('num_modes', '?')} modes, "
            + (f"T₁ = {t1:.4g} s" if t1 else "no finite modes"))
        return info

    def _modal_ready_model(self):
        """Build the model for a modal-based analysis and validate it carries
        mass. Returns ``(model, neq)`` or ``None`` (after showing why)."""
        import numpy as np

        from femsolver.analysis.assembler import assemble_mass

        p = self._project
        if p is None or not p.members:
            self.statusBar().showMessage("Add members first.")
            return None
        model = p.build_model(with_loads=False)
        if not model.elements:
            self.statusBar().showMessage("Nothing to solve — add members first.")
            return None
        model.number_dofs()
        if model.neq < 2:
            QMessageBox.information(
                self, "Analysis",
                "The model has too few free DOFs — release some supports "
                "or add members.")
            return None
        if abs(assemble_mass(model)).max() <= 0.0:
            QMessageBox.information(
                self, "Analysis",
                "The model has no mass — every material's density is zero.\n\n"
                "Open the Material editor and set a density (ρ, kg/m³) on the "
                "materials your members use, then run again.")
            return None
        return model, model.neq

    def _seed_state_model(self, nl_case_id, *, require_mass=True):
        """Build a fiber model at the committed state of nonlinear case
        ``nl_case_id`` for an analysis *from that state* (E2 — modal / response
        spectrum / buckling).

        The seeded model carries the deformed geometry + committed element state
        (so its tangent + geometric stiffness reflect the preload), plus mass
        from the materials' density (a representative ρ — exact for a
        single-material column, the usual fiber case). ``require_mass`` gates the
        density check: modal / response spectrum need mass, buckling does not.
        Returns ``(model, neq)`` or ``None`` (with a reason shown), mirroring
        :meth:`_modal_ready_model`."""
        import nonlinear as NL
        from femsolver.analysis.assembler import assemble_mass

        p = self._project
        src = p.nonlinear_case(nl_case_id)
        if src is None:
            QMessageBox.warning(
                self, "Analysis",
                f"The initial-condition source case {nl_case_id} was deleted — "
                "pick another in the case's Modify dialog.")
            return None
        density = max((float(getattr(m, "rho", 0.0)) for m in p.materials),
                      default=0.0)
        try:
            model, _f = NL.seed_to_committed_state(p, src, density=density)
        except Exception as exc:                           # noqa: BLE001
            QMessageBox.warning(
                self, "Analysis",
                f"Could not establish the initial state from '{src.name}':"
                f"\n\n{exc}")
            return None
        model.number_dofs()
        if model.neq < 2:
            QMessageBox.information(
                self, "Analysis", "The model has too few free DOFs.")
            return None
        if require_mass and abs(assemble_mass(model)).max() <= 0.0:
            QMessageBox.information(
                self, "Analysis",
                "The model has no mass — set a density (ρ, kg/m³) on the "
                "materials your members use, then run again.")
            return None
        return model, model.neq

    def run_response_spectrum(self, config=None):
        """Response-spectrum (modal-superposition) seismic analysis
        (Analysis-cases ▸ Response Spectrum).

        Extracts modes, samples the design spectrum, combines the modal peaks
        (SRSS / CQC) into a single peak response drawn on the view, and reports
        per-mode participation. ``config`` = ``(spectrum, num_modes, direction,
        combination[, initial_condition])`` bypasses the setup dialog (for tests
        / scripting). With ``initial_condition = ("state", nl_case_id)`` (E2d) the
        modal basis is taken at that nonlinear case's committed state on the
        tangent stiffness — a response spectrum of a preloaded structure.
        """
        import numpy as np

        from femsolver import ResponseSpectrumAnalysis
        from femsolver.analysis.assembler import assemble_mass

        if config is None:                       # legacy direct-run via dialog
            ready = self._modal_ready_model()
            if ready is None:
                return None
            model, neq = ready
            from response_spectrum_dialog import ResponseSpectrumDialog
            mm = max(1, neq - 1)
            config = ResponseSpectrumDialog.configure(
                self, ndm=self._project.ndm, max_modes=mm,
                default_modes=min(6, mm))
            if config is None:
                return None
            ic = ("zero",)
        else:                                    # saved AnalysisCase dispatch
            ic = config[4] if len(config) > 4 else ("zero",)
            ready = (self._seed_state_model(ic[1]) if ic[0] == "state"
                     else self._modal_ready_model())
            if ready is None:
                return None
            model, neq = ready

        spectrum, num_modes, direction, combination = config[:4]
        num_modes = max(1, min(int(num_modes), max(1, neq - 1)))
        stiffness = "tangent" if ic[0] == "state" else "elastic"

        try:
            info = ResponseSpectrumAnalysis(
                model, spectrum, num_modes=num_modes, direction=direction,
                combination=combination, stiffness=stiffness).run()
        except Exception as exc:                           # noqa: BLE001
            QMessageBox.warning(
                self, "Response spectrum",
                f"The response-spectrum solve failed:\n\n{exc}")
            return None

        # total in-direction mass ιᵀMι → turns effective modal masses into %
        M = assemble_mass(model)
        idx = {"x": 0, "y": 1, "z": 2}.get(direction, 0)
        iota = np.zeros(model.neq)
        for node in model.nodes.values():
            if idx < node.ndf:
                eq = int(node.eqn[idx])
                if eq >= 0:
                    iota[eq] = 1.0
        total_dir_mass = float(iota @ (M @ iota))

        # the engine scattered the combined peak into Node.disp — draw it
        dmax = mg.max_translation(model)
        span = mg.model_span(model)
        scale = (0.08 * span / dmax) if dmax > 0 else 1.0
        self.view.show_deformed(self._visible_model(model), scale)

        from response_spectrum_results_dialog import \
            ResponseSpectrumResultsDialog
        self._rs_results_dlg = ResponseSpectrumResultsDialog.show_results(
            self, info, total_dir_mass)

        pct = (100.0 * info.get("total_participating_mass", 0.0)
               / total_dir_mass) if total_dir_mass > 0 else 0.0
        self.log.appendPlainText(
            f"Response spectrum solved: {info.get('num_modes', '?')} modes, "
            f"dir {direction.upper()}, {combination.upper()}, "
            f"participating mass {pct:.1f}%, max|u| = {dmax:.4e} m")
        self.statusBar().showMessage(
            f"Response spectrum · {combination.upper()} {direction.upper()} · "
            f"participating mass {pct:.1f}% · max|u| {dmax:.3e} m")
        return info

    def _reference_load_label(self, selection) -> str:
        kind = selection[0] if selection else "all"
        if kind in ("case", "pattern"):
            c = self._project.load_case(selection[1])
            return f"pattern {c.name}" if c else f"pattern {selection[1]}"
        if kind == "combination":
            c = self._project.combination(selection[1])
            return f"combo {c.name}" if c else f"combo {selection[1]}"
        return "all load patterns"

    def run_buckling(self, config=None):
        """Linear (eigenvalue) buckling analysis (Analysis-cases ▸ Buckling).

        Builds a member-sub-divided model under a chosen reference load, forms
        the dedicated geometric stiffness on the elastic beams, and solves
        ``(K + λ·K_g)·φ = 0`` via
        :class:`femsolver.analysis.buckling.LinearBucklingAnalysis`. The
        critical factor λ multiplies the reference load. ``config`` =
        ``(selection, num_modes, subdivisions)`` bypasses the setup dialog.
        """
        import numpy as np

        from femsolver import LinearBucklingAnalysis

        p = self._project
        if p is None or not p.members:
            self.statusBar().showMessage("Add members first.")
            return None

        if config is None:
            from buckling_dialog import BucklingDialog
            config = BucklingDialog.configure(self, p)
            if config is None:
                return None
        selection, num_modes, subdivisions = config[0], config[1], config[2]
        ic = config[3] if len(config) > 3 else ("zero",)

        if ic[0] == "state":            # buckle from a nonlinear preload (E2e)
            ready = self._seed_state_model(ic[1], require_mass=False)
            if ready is None:
                return None
            model, _neq = ready
            prestress = "current_state"
            src = p.nonlinear_case(ic[1])
            ref_label = f"state of {src.name}" if src else "committed state"
        else:
            model, _subs = p.build_buckling_model(selection=selection,
                                                  subdivisions=subdivisions)
            model.number_dofs()
            prestress = "reference"
            ref_label = self._reference_load_label(selection)
        max_modes = max(1, model.neq - 2)
        num_modes = max(1, min(int(num_modes), max_modes))

        try:
            info = LinearBucklingAnalysis(model, num_modes=num_modes,
                                          prestress=prestress).run()
        except Exception as exc:                           # noqa: BLE001
            QMessageBox.warning(
                self, "Buckling",
                f"The buckling solve did not find a critical load:\n\n{exc}")
            return None

        def _show_mode(k: int) -> None:
            for node in model.nodes.values():
                md = getattr(node, "mode_disp", None)
                if md is None or md.shape[1] <= k:
                    continue
                node.disp = np.asarray(md, dtype=float)[:, k].copy()
            dmax = mg.max_translation(model)
            span = mg.model_span(model)
            scale = (0.08 * span / dmax) if dmax > 0 else 1.0
            self.view.show_deformed(self._visible_model(model), scale)
            self.statusBar().showMessage(
                f"Buckling mode {k + 1}: λ = {info['load_factors'][k]:.4g}")

        from buckling_results_dialog import BucklingResultsDialog
        self._buckling_results_dlg = BucklingResultsDialog.show_results(
            self, info, ref_label, _show_mode)

        self.log.appendPlainText(
            f"Buckling solved: {info.get('num_modes', '?')} modes, "
            f"critical λ = {info.get('critical_load_factor', float('nan')):.4g} "
            f"× ({ref_label})")
        return info

    def run_moving_load(self, config=None):
        """Moving-load / influence-line analysis (Analysis-cases ▸ Moving Load).

        Builds the influence line for a chosen response as a unit load travels
        a lane of girder nodes, then convolves the selected vehicle (AASHTO
        HL-93 / IRC) to get the design envelope. ``config`` bypasses the setup
        dialog (for tests): ``dict(lane, vehicle, response=(kind, id, end))``.
        2-D girder-line models only.
        """
        from femsolver.bridges import (BeamForce, Displacement,
                                        InfluenceLineEngine, Lane, MovingLoad,
                                        Reaction, aashto_hl93_envelope,
                                        moving_load_envelope)

        p = self._project
        if p is None or not p.members:
            self.statusBar().showMessage("Add members first.")
            return None
        if config is None:
            from moving_load_dialog import MovingLoadDialog
            config = MovingLoadDialog.configure(self, p)
            if config is None:
                return None
        lane_nodes = config["lane"]
        if len(lane_nodes) < 2:
            QMessageBox.information(self, "Moving load",
                                   "Select at least two lane nodes.")
            return None

        vdof = 2 if p.ndm == 3 else 1                   # vertical translation DOF
        kind, target, end = config["response"]
        _RESP = {
            "M": lambda: BeamForce(element_tag=target, component="M", end=end),
            "V": lambda: BeamForce(element_tag=target, component="V", end=end),
            "disp": lambda: Displacement(node_tag=target, dof=vdof),
            "reaction": lambda: Reaction(node_tag=target, dof=vdof),
        }
        _LABEL = {
            "M": f"moment at member {target} ({end})",
            "V": f"shear at member {target} ({end})",
            "disp": f"vertical displacement at node {target}",
            "reaction": f"vertical reaction at node {target}",
        }
        # the response's physical quantity, for unit conversion (plan U6b)
        _QTY = {"M": Quantity.MOMENT, "V": Quantity.FORCE,
                "disp": Quantity.LENGTH, "reaction": Quantity.FORCE}
        response = _RESP[kind]()
        label = _LABEL[kind]
        us = self._units()
        qty = _QTY[kind]
        units = us.label(qty)

        model = p.build_model(with_loads=False)
        lane = Lane(node_tags=lane_nodes, load_dof=vdof, gravity_sign=-1.0)
        try:
            engine = InfluenceLineEngine(model)
            il = engine.influence_line(lane, response)
        except Exception as exc:                           # noqa: BLE001
            QMessageBox.warning(self, "Moving load",
                                f"Could not build the influence line:\n\n{exc}")
            return None

        veh_key = config["vehicle"]
        veh_label = {"hl93": "AASHTO HL-93", "hl93_truck": "HL-93 truck",
                     "hl93_tandem": "HL-93 tandem", "irc_class_a": "IRC Class A",
                     "irc_70r": "IRC 70R"}.get(veh_key, veh_key)
        try:
            if veh_key == "hl93":
                env = aashto_hl93_envelope(il)
            else:
                env = moving_load_envelope(il, MovingLoad.preset(veh_key))
        except Exception as exc:                           # noqa: BLE001
            QMessageBox.warning(self, "Moving load",
                                f"Envelope failed:\n\n{exc}")
            return None

        from moving_load_results_dialog import MovingLoadResultsDialog
        self._moving_load_results_dlg = MovingLoadResultsDialog.show_results(
            self, il, env, response_label=label, unitsys=us, quantity=qty,
            vehicle=veh_label)
        emax = us.to_display(env.get('max', 0.0), qty)
        emin = us.to_display(env.get('min', 0.0), qty)
        self.log.appendPlainText(
            f"Moving load solved: {label}, {veh_label} envelope "
            f"max={emax:.4g} min={emin:.4g} {units}")
        self.statusBar().showMessage(
            f"Moving load · {veh_label} · max {emax:.3g} {units}")
        return {"il": il, "env": env}

    def run_temperature_gradient(self, config=None):
        """Temperature-gradient load case (Analysis-cases ▸ Temperature
        Gradient).

        Reduces a vertical temperature gradient on each selected member's
        section (rectangular-equivalent depth/width from A, Iz) to an axial
        strain + curvature, applies the equivalent beam actions, and solves —
        giving the self-equilibrated section stress plus the frame deflection
        and continuity moments. ``config`` bypasses the setup dialog (tests).
        2-D girder-line models only.
        """
        import numpy as np

        from femsolver import LinearStaticAnalysis
        from femsolver.bridges import (aashto_gradient,
                                       apply_beam_thermal_actions,
                                       equivalent_thermal_actions,
                                       linear_gradient)

        p = self._project
        if p is None or not p.members:
            self.statusBar().showMessage("Add members first.")
            return None
        if p.ndm != 2:
            QMessageBox.information(
                self, "Temperature gradient",
                "Temperature-gradient analysis is currently 2-D only.")
            return None

        if config is None:
            from temperature_gradient_dialog import TemperatureGradientDialog
            config = TemperatureGradientDialog.configure(self, p)
            if config is None:
                return None
        member_ids = config["members"]
        if not member_ids:
            self.statusBar().showMessage("Select at least one member.")
            return None
        alpha = config["alpha"]

        model = p.build_model(with_loads=False)

        def _gradient(depth):
            if config["source"] == "aashto":
                return aashto_gradient(config["zone"], depth, alpha=alpha)
            return linear_gradient(config["dt_top"], config["dt_bot"], depth,
                                   alpha=alpha)

        governing = None                                # (|σ|, gradient, act, h, label)
        applied = 0
        for tag in member_ids:
            try:
                el = model.element(tag)
            except KeyError:
                continue
            A, Iz, E = el.area, el.Iz, el.material.E
            if A <= 0 or Iz <= 0:
                continue
            h = float(np.sqrt(12.0 * Iz / A))           # rectangular-equivalent
            b = A / h
            grad = _gradient(h)
            act = equivalent_thermal_actions(grad, height=h, width=b, E=E)
            apply_beam_thermal_actions(model, eps0=act.eps0,
                                       kappa=act.curvature, elements=[tag])
            applied += 1
            peak = max(abs(act.self_stress_top), abs(act.self_stress_bottom))
            if governing is None or peak > governing[0]:
                governing = (peak, grad, act, h,
                             f"member {tag} (h≈{h:.2f} m)")
        if governing is None:
            self.statusBar().showMessage("No valid members to load.")
            return None

        LinearStaticAnalysis(model).run()

        dmax = mg.max_translation(model)
        span = mg.model_span(model)
        scale = (0.08 * span / dmax) if dmax > 0 else 1.0
        self.view.show_deformed(self._visible_model(model), scale)

        max_moment = 0.0
        for tag in member_ids:
            try:
                efl = model.element(tag).end_forces_local
            except KeyError:
                continue
            if efl is not None and len(efl) >= 6:
                max_moment = max(max_moment, abs(efl[2]), abs(efl[5]))

        _peak, grad, act, h, label = governing
        from temperature_gradient_results_dialog import \
            TemperatureGradientResultsDialog
        self._temp_gradient_results_dlg = \
            TemperatureGradientResultsDialog.show_results(
                self, grad, act, h, max_deflection=dmax, max_moment=max_moment,
                section_label=label, unitsys=self._units())
        self.log.appendPlainText(
            f"Temperature gradient solved: {applied} member(s), "
            f"ΔT_uniform={act.dT_uniform:.2f}°C, self-stress "
            f"{act.self_stress_top / 1e6:.2f}/{act.self_stress_bottom / 1e6:.2f} "
            f"MPa, max|u|={dmax:.4e} m, max M={max_moment / 1e3:.2f} kN·m")
        self.statusBar().showMessage(
            f"Temperature gradient · self-stress "
            f"{act.self_stress_top / 1e6:.2f} MPa · max|u| {dmax:.3e} m")
        return {"actions": act, "gradient": grad, "max_deflection": dmax,
                "max_moment": max_moment}

    def run_construction_stages(self, config=None):
        """Construction-stage (incremental erection) analysis + camber
        (Analysis-cases ▸ Construction Stages).

        Opens the stage manager (unless ``config`` — a list of
        :class:`project.Stage` — is given), builds each stage's members under
        their self-weight via
        :class:`femsolver.bridges.IncrementalStagedAnalysis`, and reports the
        camber (:func:`femsolver.bridges.staged_camber`). 2-D only.
        """
        G = 9.80665

        from femsolver.bridges import (ErectionStage,
                                       IncrementalStagedAnalysis, staged_camber)

        import numpy as np

        p = self._project
        if p is None or not p.members:
            self.statusBar().showMessage("Add members first.")
            return None
        vdof = 2 if p.ndm == 3 else 1                   # vertical translation DOF
        ndf = p.ndf

        if config is not None:
            stages = config
        else:
            from stage_dialog import StageManagerDialog
            stages = StageManagerDialog.manage(self, p)
            if stages is None:
                return None
            if stages != p.stages:
                self._apply_edit("Edit construction stages",
                                 lambda: setattr(p, "stages", stages))

        model = p.build_model(with_loads=False)
        all_ids = [m.id for m in p.members]
        assigned = set()
        for s in stages:
            assigned.update(s.add_members)
        unassigned = [i for i in all_ids if i not in assigned]

        def _self_weight(member_ids):
            loads: dict = {}
            for tag in member_ids:
                try:
                    el = model.element(tag)
                except KeyError:
                    continue
                X = el.node_coords()
                L = float(np.linalg.norm(X[1] - X[0]))
                w = getattr(el.material, "rho", 0.0) * el.area * L * G
                if w <= 0.0:
                    continue
                for nd in el.node_tags:
                    loads.setdefault(nd, [0.0] * ndf)[vdof] += -w / 2.0
            return loads

        erection = []
        if not stages:
            erection = [ErectionStage("All", add_elements=all_ids,
                                      loads=_self_weight(all_ids))]
        else:
            for i, s in enumerate(stages):
                ids = list(s.add_members)
                if i == 0 and unassigned:
                    ids = unassigned + ids
                erection.append(ErectionStage(
                    s.name or f"Stage {i + 1}", add_elements=ids,
                    loads=_self_weight(ids)))

        total_w = sum(abs(v[vdof]) for st in erection
                      for v in st.loads.values())
        if total_w <= 0.0:
            QMessageBox.information(
                self, "Construction stages",
                "No self-weight — set a material density (ρ) so the staged "
                "self-weight (and camber) is non-zero.")
            return None

        try:
            res = IncrementalStagedAnalysis(model, erection).run()
        except Exception as exc:                           # noqa: BLE001
            QMessageBox.warning(
                self, "Construction stages",
                f"A stage could not be solved (an intermediate structure may "
                f"be unstable — check the build order):\n\n{exc}")
            return None

        camber = staged_camber(res, model, erection, dof=vdof)
        dmax = mg.max_translation(model)
        span = mg.model_span(model)
        scale = (0.08 * span / dmax) if dmax > 0 else 1.0
        self.view.show_deformed(self._visible_model(model), scale)

        from construction_stage_results_dialog import \
            ConstructionStageResultsDialog
        self._stage_results_dlg = ConstructionStageResultsDialog.show_results(
            self, camber, n_stages=len(erection), unitsys=self._units())
        self.log.appendPlainText(
            f"Construction stages solved: {len(erection)} stages, "
            f"max final deflection {dmax * 1e3:.2f} mm → camber {dmax * 1e3:.2f} "
            f"mm high")
        self.statusBar().showMessage(
            f"Construction stages · {len(erection)} stages · camber "
            f"{dmax * 1e3:.2f} mm")
        return {"camber": camber, "stages": len(erection)}

    def run_vehicle_dynamics(self, config=None):
        """Vehicle-dynamics / moving-load time-history (Analysis-cases ▸ Vehicle
        Dynamics).

        A vehicle crosses the lane at speed and the transient response gives the
        dynamic amplification factor. ``kind='force'`` uses constant moving
        axle forces (`MovingForceAnalysis`); ``kind='vbi'`` a coupled
        sprung-mass vehicle (`VBIAnalysis`, with a contact-force history).
        ``config`` bypasses the setup dialog (tests). 2-D girder-line models,
        mass from material density.
        """
        import numpy as np

        from femsolver import EigenAnalysis
        from femsolver.analysis.assembler import assemble_mass
        from femsolver.analysis.damping import RayleighDamping
        from femsolver.bridges import Lane

        p = self._project
        if p is None or not p.members:
            self.statusBar().showMessage("Add members first.")
            return None
        if p.ndm != 2:
            QMessageBox.information(self, "Vehicle dynamics",
                                   "Vehicle-dynamics analysis is currently "
                                   "2-D only.")
            return None

        if config is None:
            from vehicle_dynamics_dialog import VehicleDynamicsDialog
            config = VehicleDynamicsDialog.configure(self, p)
            if config is None:
                return None
        lane_nodes = config["lane"]
        if len(lane_nodes) < 2:
            QMessageBox.information(self, "Vehicle dynamics",
                                   "Select at least two lane nodes.")
            return None

        model = p.build_model(with_loads=False)
        model.number_dofs()
        if abs(assemble_mass(model)).max() <= 0.0:
            QMessageBox.information(
                self, "Vehicle dynamics",
                "The model has no mass — set a material density (ρ) so the "
                "dynamics (and DAF) are meaningful.")
            return None

        lane = Lane(node_tags=lane_nodes, load_dof=1, gravity_sign=-1.0)
        track = (config["node"], 1)
        speed = config["speed"]
        zeta = config["zeta"]

        damping = None                                  # Rayleigh from modes 1,3
        if zeta > 0.0:
            try:
                w1 = 2.0 * np.pi * EigenAnalysis(
                    model, num_modes=2).run()["frequencies_hz"][0]
                damping = RayleighDamping.from_modes(w1, zeta, 3.0 * w1, zeta)
            except Exception:                           # noqa: BLE001
                damping = None

        try:
            if config["kind"] == "vbi":
                from femsolver.bridges import SprungMassVehicle, VBIAnalysis
                m_s = config["mass"]
                k = m_s * (2.0 * np.pi * config["bounce"]) ** 2
                c = 2.0 * config["susp_damp"] * float(np.sqrt(k * m_s))
                veh = SprungMassVehicle(mass=m_s, stiffness=k, damping=c)
                res = VBIAnalysis(model, lane, veh, speed, track=track,
                                  bridge_damping=damping,
                                  free_vibration_time=0.5).run()
                dynamic = res["bridge_disp"]
                contact = res["contact_force"]
                weight = m_s * 9.80665
            else:
                from femsolver.bridges import (MovingForceAnalysis, MovingLoad,
                                               VehicleAxles)
                preset = MovingLoad.preset(config["vehicle"])
                veh = VehicleAxles(axle_loads=preset.axle_loads,
                                   axle_offsets=preset.axle_offsets)
                res = MovingForceAnalysis(model, lane, veh, speed, track=track,
                                          damping=damping,
                                          free_vibration_time=0.5).run()
                dynamic = res["dynamic_disp"]
                contact = None
                weight = None
        except Exception as exc:                           # noqa: BLE001
            QMessageBox.warning(self, "Vehicle dynamics",
                                f"The dynamic analysis failed:\n\n{exc}")
            return None

        from vehicle_dynamics_results_dialog import VehicleDynamicsResultsDialog
        self._vehicle_dyn_results_dlg = \
            VehicleDynamicsResultsDialog.show_results(
                self, res["times"], dynamic, res["static_disp"], res["DAF"],
                contact_force=contact, weight=weight,
                response_label=f"node {config['node']} vertical",
                speed_kmh=speed * 3.6, unitsys=self._units())
        kind_label = "sprung-mass VBI" if config["kind"] == "vbi" \
            else "moving force"
        self.log.appendPlainText(
            f"Vehicle dynamics solved ({kind_label}): {speed * 3.6:.0f} km/h, "
            f"DAF = {res['DAF']:.3f}, peak dynamic {res['peak_dynamic']:.4e} m")
        self.statusBar().showMessage(
            f"Vehicle dynamics · {kind_label} · DAF {res['DAF']:.3f}")
        return res

    def run_influence_surface(self, config=None):
        """Influence-surface / multi-lane moving-load analysis (Analysis-cases
        ▸ Influence Surface).

        A unit load traverses a 2-D deck / grillage to build the influence
        surface for a chosen response; vehicles are then placed in the AASHTO
        design lanes with multiple-presence factors for the governing effect.
        ``config`` bypasses the setup dialog (tests). Needs a 3-D deck model.
        """
        import numpy as np

        from femsolver.bridges import (DeckSurface, DesignLane, Displacement,
                                       InfluenceLineEngine, MovingLoad,
                                       Reaction, Vehicle2D,
                                       generate_design_lanes,
                                       multi_lane_envelope)

        p = self._project
        if p is None or not p.members:
            self.statusBar().showMessage("Add members first.")
            return None
        if p.ndm != 3:
            QMessageBox.information(
                self, "Influence surface",
                "Influence surfaces need a 3-D deck / grillage model (a plan "
                "spread of nodes). Build the deck in 3-D and try again.")
            return None

        if config is None:
            from influence_surface_dialog import InfluenceSurfaceDialog
            config = InfluenceSurfaceDialog.configure(self, p)
            if config is None:
                return None
        deck_nodes = config["deck"]
        if len(deck_nodes) < 3:
            QMessageBox.information(self, "Influence surface",
                                   "Select at least three deck nodes.")
            return None

        kind, node = config["response"]
        response = (Displacement(node_tag=node, dof=2) if kind == "disp"
                    else Reaction(node_tag=node, dof=2))
        qty = Quantity.LENGTH if kind == "disp" else Quantity.FORCE
        label = (f"vertical {'displacement' if kind == 'disp' else 'reaction'} "
                 f"at node {node}")

        model = p.build_model(with_loads=False)
        deck = DeckSurface(node_tags=deck_nodes, load_dof=2, plan_axes=(0, 1))
        try:
            IS = InfluenceLineEngine(model).influence_surface(deck, response)
        except Exception as exc:                           # noqa: BLE001
            QMessageBox.warning(self, "Influence surface",
                                f"Could not build the influence surface:\n\n{exc}")
            return None

        veh_key = config["vehicle"]
        veh_label = {"hl93_truck": "HL-93 truck", "hl93_tandem": "HL-93 tandem",
                     "irc_class_a": "IRC Class A", "irc_70r": "IRC 70R"}.get(
                         veh_key, veh_key)
        vehicle = Vehicle2D.from_axle_train(MovingLoad.preset(veh_key))

        ys = IS.points[:, 1]
        lanes = generate_design_lanes(float(ys.min()), float(ys.max()))
        if not lanes:                                      # deck narrower than a lane
            lanes = [DesignLane(y_center=float(0.5 * (ys.min() + ys.max())),
                                width=max(float(ys.max() - ys.min()), 3.0))]
        try:
            env = multi_lane_envelope(
                IS, vehicle, lanes, multi_presence=config["multi_presence"])
        except Exception as exc:                           # noqa: BLE001
            QMessageBox.warning(self, "Influence surface",
                                f"Multi-lane envelope failed:\n\n{exc}")
            return None

        us = self._units()
        units = us.label(qty)
        from influence_surface_results_dialog import \
            InfluenceSurfaceResultsDialog
        self._influence_surface_results_dlg = \
            InfluenceSurfaceResultsDialog.show_results(
                self, IS, env, response_label=label, quantity=qty,
                vehicle=veh_label, n_lanes=len(lanes), unitsys=us)
        emax = us.to_display(env.get('max', 0.0), qty)
        self.log.appendPlainText(
            f"Influence surface solved: {label}, {veh_label}, {len(lanes)} "
            f"lane(s), governing max={emax:.4g} "
            f"(m={env.get('max_factor', 1.0):.2f}, "
            f"{env.get('max_num_lanes', 0)} lane(s)) {units}")
        self.statusBar().showMessage(
            f"Influence surface · {veh_label} · max {emax:.3g} "
            f"{units} ({env.get('max_num_lanes', 0)} lane(s))")
        return {"surface": IS, "envelope": env, "lanes": len(lanes)}

    def run_cable_tuning(self, config=None):
        """Cable-stayed tuning — unknown-load-factor (Analysis-cases ▸ Cable
        Tuning).

        Solves the stay pretensions so the target deck nodes reach zero
        vertical deflection under dead load
        (:func:`femsolver.bridges.unknown_load_factors`), then shows the tuned
        tensions and the before/after deck profile. ``config`` bypasses the
        setup dialog (tests). 2-D only; needs dead load defined; stays are
        designated among the model's members.
        """
        import numpy as np

        from femsolver import LinearStaticAnalysis
        from femsolver.bridges import (Cable, Displacement,
                                       apply_cable_tensions,
                                       unknown_load_factors)

        p = self._project
        if p is None or not p.members:
            self.statusBar().showMessage("Add members first.")
            return None
        if p.ndm != 2:
            QMessageBox.information(self, "Cable tuning",
                                   "Cable tuning is currently 2-D only.")
            return None
        if not p.loads and not p.member_loads:
            QMessageBox.information(
                self, "Cable tuning",
                "Define the dead load first — the stays are tuned to cancel "
                "its deck deflection.")
            return None

        if config is None:
            from cable_tuning_dialog import CableTuningDialog
            config = CableTuningDialog.configure(self, p)
            if config is None:
                return None
        cab_ids = config["cables"]
        tgt_ids = config["targets"]
        if not cab_ids or not tgt_ids:
            QMessageBox.information(self, "Cable tuning",
                                   "Select at least one stay and one target "
                                   "node.")
            return None

        by_id = {mb.id: mb for mb in p.members}
        cables = [Cable(by_id[i].n1, by_id[i].n2, name=f"member {i}")
                  for i in cab_ids if i in by_id]
        targets = [(Displacement(node_tag=n, dof=1), 0.0) for n in tgt_ids]

        model = p.build_model(with_loads=True)             # includes dead load
        try:
            res = unknown_load_factors(model, cables, targets)
        except Exception as exc:                           # noqa: BLE001
            QMessageBox.warning(self, "Cable tuning",
                                f"The tuning solve failed:\n\n{exc}")
            return None

        def _profile(with_tensions):
            m = p.build_model(with_loads=True)
            if with_tensions:
                apply_cable_tensions(m, cables, res.tensions)
            LinearStaticAnalysis(m).run()
            order = sorted(m.nodes.values(),
                           key=lambda nd: nd.coords[0])
            xs = [float(nd.coords[0]) for nd in order]
            uy = [float(nd.disp[1]) for nd in order]
            return xs, uy, m

        xs, before, _ = _profile(False)
        _, after, m_after = _profile(True)
        # show the tuned (after) deflected shape on the view
        dmax = mg.max_translation(m_after)
        span = mg.model_span(m_after)
        scale = (0.08 * span / dmax) if dmax > 0 else 1.0
        self.view.show_deformed(self._visible_model(m_after), scale)

        from cable_tuning_results_dialog import CableTuningResultsDialog
        self._cable_tuning_results_dlg = CableTuningResultsDialog.show_results(
            self, [c.name for c in cables], res.tensions, xs, before, after,
            max_residual=float(np.max(np.abs(res.residual))) if len(
                res.residual) else 0.0, unitsys=self._units())
        self.log.appendPlainText(
            f"Cable tuning solved: {len(cables)} stay(s), tensions "
            + ", ".join(f"{t / 1e3:.0f}" for t in res.tensions)
            + f" kN; target residual {np.max(np.abs(res.residual)) * 1e3:.3g} mm")
        self.statusBar().showMessage(
            f"Cable tuning · {len(cables)} stay(s) · "
            f"max tension {max(abs(t) for t in res.tensions) / 1e3:.0f} kN")
        return {"tensions": res.tensions, "residual": res.residual}

    def run_load_rating(self, config=None):
        """AASHTO LRFR load rating (Analysis-cases ▸ Load Rating).

        Builds the influence line for a rated member effect (moment or shear),
        pulls the HL-93 live-load effect (incl. IM + lane) from it, then
        computes the LRFR rating factors — design inventory + operating, and
        optionally legal (by ADTT) and permit — via
        :func:`femsolver.bridges.rate_member`. ``config`` bypasses the setup
        dialog (tests) and carries SI capacity / dead-load effects; see
        :class:`load_rating_dialog.LoadRatingDialog`. 2-D or 3-D girder lines.
        """
        from femsolver.bridges import (BeamForce, InfluenceLineEngine, Lane,
                                       live_load_effect, rate_member)

        p = self._project
        if p is None or not p.members:
            self.statusBar().showMessage("Add members first.")
            return None
        if config is None:
            from load_rating_dialog import LoadRatingDialog
            config = LoadRatingDialog.configure(self, p)
            if config is None:
                return None
        lane_nodes = config["lane"]
        if len(lane_nodes) < 2:
            QMessageBox.information(self, "Load rating",
                                   "Select at least two lane nodes.")
            return None

        vdof = 2 if p.ndm == 3 else 1                   # vertical translation DOF
        comp, target, end = config["response"]
        qty = Quantity.MOMENT if comp == "M" else Quantity.FORCE
        label = {"M": f"moment at member {target} ({end})",
                 "V": f"shear at member {target} ({end})"}[comp]

        model = p.build_model(with_loads=False)
        lane = Lane(node_tags=lane_nodes, load_dof=vdof, gravity_sign=-1.0)
        try:
            engine = InfluenceLineEngine(model)
            il = engine.influence_line(
                lane, BeamForce(element_tag=target, component=comp, end=end))
            ll_im = live_load_effect(il, im=config.get("im", 0.33))
        except Exception as exc:                           # noqa: BLE001
            QMessageBox.warning(self, "Load rating",
                                f"Could not build the live-load effect:\n\n{exc}")
            return None

        if ll_im <= 0.0:
            QMessageBox.information(
                self, "Load rating",
                "The HL-93 live-load effect is zero for this location — pick a "
                "member/end that the lane actually loads.")
            return None

        rating = rate_member(
            Rn=config["Rn"], DC=config["DC"], DW=config["DW"], LL_IM=ll_im,
            phi=config.get("phi", 1.0), phi_c=config.get("phi_c", 1.0),
            phi_s=config.get("phi_s", 1.0), P=config.get("P", 0.0),
            adtt=config.get("adtt"),
            permit_gamma_LL=config.get("permit_gamma_LL"))

        us = self._units()
        from load_rating_results_dialog import LoadRatingResultsDialog
        self._load_rating_results_dlg = LoadRatingResultsDialog.show_results(
            self, rating, effect_label=label, ll_im=ll_im, unitsys=us,
            quantity=qty)
        ctrl = rating.controlling
        self.log.appendPlainText(
            f"Load rating solved: {label}; controlling {ctrl.level} "
            f"RF = {ctrl.rf:.3f} "
            f"({'adequate' if ctrl.adequate else 'DEFICIENT'})")
        self.statusBar().showMessage(
            f"Load rating · controlling RF {ctrl.rf:.2f} "
            f"({'OK' if ctrl.adequate else 'deficient'})")
        return {"rating": rating, "ll_im": ll_im, "il": il}

    def run_pushover_dialog(self, preselect_case=None) -> None:
        from pushover_dialog import PushoverDialog
        p = self._project
        if not p.members:
            self.statusBar().showMessage("Add members first.")
            return
        has_fiber = any(getattr(s, "gsd_spec", None) for s in p.sections)
        if not has_fiber:
            self.statusBar().showMessage(
                "Nonlinear pushover needs a Section Designer (fiber) section "
                "on a member.")
            return
        dlg = PushoverDialog(self, p)
        if preselect_case is not None:                 # from the Analysis-cases home
            i = dlg.case_combo.findData(preselect_case)
            if i >= 0:
                dlg.case_combo.setCurrentIndex(i)
        dlg.exec()
        res = getattr(dlg, "_results", None)
        if res is not None and res.n_curve:
            self._record_run(dlg.run_descriptor(), res)   # save to history (G-S2)
        if res is not None and res.has_shape:
            self.set_nl_results(res)

    def run_timehistory_dialog(self, seed=None) -> None:
        from timehistory_dialog import TimeHistoryDialog
        p = self._project
        if not p.members:
            self.statusBar().showMessage("Add members first.")
            return
        if not any(getattr(s, "gsd_spec", None) for s in p.sections):
            self.statusBar().showMessage(
                "Nonlinear time history needs a Section Designer (fiber) "
                "section on a member.")
            return
        dlg = TimeHistoryDialog(self, p, seed=seed)
        dlg.exec()
        res = getattr(dlg, "_result", None)
        if res and res.get("disp"):
            from nl_runs import RunRecord, next_run_id
            direction, _ = dlg.direction.currentData()
            rec = RunRecord.from_time_history(
                res, id=next_run_id(p.runs), name="Time history",
                meta={"Project": p.name, "Monitor":
                      f"node {dlg.node.currentData()} {direction.upper()}",
                      "Damping": f"{dlg.zeta.value():g}"})
            self._apply_edit("Record time-history run",
                             lambda: self._project.runs.append(rec))

    # ------------------------------------------------ nonlinear results (G-S1/2)
    def _build_nl_step_dock(self) -> QDockWidget:
        dock = QDockWidget("Analysis steps", self)
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(8, 4, 8, 4)
        self._nl_slider = QSlider(Qt.Orientation.Horizontal)
        self._nl_slider.setEnabled(False)
        self._nl_slider.valueChanged.connect(self._on_nl_step)
        self._nl_step_lbl = QLabel("—")
        self._nl_scale_spin = QDoubleSpinBox()
        self._nl_scale_spin.setRange(0.0, 1.0e6)
        self._nl_scale_spin.setDecimals(1)
        self._nl_scale_spin.valueChanged.connect(self._on_nl_scale)
        self._nl_color = QComboBox()
        self._nl_color.addItems(["acceptance", "peak strain"])
        self._nl_color.currentIndexChanged.connect(self._on_nl_step)
        row.addWidget(QLabel("Step"))
        row.addWidget(self._nl_slider, 1)
        row.addWidget(self._nl_step_lbl)
        row.addWidget(QLabel("scale ×"))
        row.addWidget(self._nl_scale_spin)
        row.addWidget(QLabel("colour"))
        row.addWidget(self._nl_color)
        dock.setWidget(w)
        return dock

    def set_nl_results(self, results) -> None:
        """Hold a completed run's :class:`NonlinearResults` (survives the
        dialog close — plan G-S2) and enable scrubbing the whole model view
        through its steps on the main view (plan G-S1)."""
        self._nl_results = results
        n = results.n_steps
        if n == 0 or self._model is None:
            self._nl_dock.hide()
            return
        # auto displacement scale from the largest nodal translation over the run
        dmax = 0.0
        for k in range(n):
            for d in (results.step(k).node_disp or {}).values():
                dmax = max(dmax, sum(v * v for v in d) ** 0.5)
        span = mg.model_span(self._model)
        self._nl_scale = (0.08 * span / dmax) if dmax > 0 else 1.0
        self._nl_scale_spin.blockSignals(True)
        self._nl_scale_spin.setValue(self._nl_scale)
        self._nl_scale_spin.blockSignals(False)
        self._nl_slider.blockSignals(True)
        self._nl_slider.setRange(0, n - 1)
        self._nl_slider.setValue(n - 1)
        self._nl_slider.setEnabled(True)
        self._nl_slider.blockSignals(False)
        self._nl_dock.show()
        self._on_nl_step()

    def _record_run(self, descriptor, results) -> None:
        """Append a completed pushover/cyclic/case run to the project history
        (plan §16 G-S2) as one undoable edit, so it persists across save/load."""
        from nl_runs import RunRecord, next_run_id
        name, kind, meta = descriptor
        rec = RunRecord.from_results(
            results, id=next_run_id(self._project.runs), name=name, kind=kind,
            meta=meta)
        self._apply_edit(f"Record run '{name}'",
                         lambda: self._project.runs.append(rec))

    def show_run_history(self) -> None:
        from run_history_dialog import RunHistoryDialog
        if not self._project.runs:
            self.statusBar().showMessage(
                "No saved runs yet — run a nonlinear pushover or time history.")
            return
        result = RunHistoryDialog.manage(self, self._project)
        if result is not None:
            self._apply_edit("Edit run history",
                             lambda: setattr(self._project, "runs", result))

    def check_model(self) -> None:
        """Run the static model checks (plan §16 G-S5) and show them; also log a
        one-line summary to the Output dock."""
        import model_checks as MC
        from model_checks_dialog import ModelChecksDialog
        checks = MC.check_project(self._project)
        n_err, n_warn, n_info = MC.summarize(checks)
        self.log.appendPlainText(
            f"Model checks: {n_err} error(s), {n_warn} warning(s), "
            f"{n_info} note(s).")
        ModelChecksDialog(self, checks).exec()
        self.statusBar().showMessage(
            "Model OK" if n_err == 0 and n_warn == 0
            else f"Model checks: {n_err} error(s), {n_warn} warning(s)")

    def _on_nl_scale(self, val: float) -> None:
        self._nl_scale = float(val)
        self._on_nl_step()

    def _on_nl_step(self, *_a) -> None:
        if self._nl_results is None or self._model is None:
            return
        k = self._nl_slider.value()
        st = self._nl_results.step(k)
        lvl = ""
        if st.member_state:
            from femsolver.performance.acceptance import level_name
            lvl = f"  [{level_name(max(st.member_state.values()))}]"
        self._nl_step_lbl.setText(
            f"{k + 1}/{self._nl_results.n_steps}  (d={st.disp:.4g}){lvl}")
        mode = ("acceptance" if self._nl_color.currentText() == "acceptance"
                else "strain")
        self.view.show_nl_step(self._visible_model(self._model), st.node_disp, st.member_damage,
                               self._nl_scale, member_state=st.member_state,
                               color_mode=mode)

    def show_diagram(self, kind: str) -> None:
        if self._solve() is None:
            return
        vmax = self.view.show_diagram(self._visible_model(self._model), kind)
        names = {"N": "Axial N", "V": "Shear V", "M": "Moment M"}
        # vmax is SI (N for axial/shear, N·m for moment); show in chosen units.
        us = self._units()
        qty = Quantity.MOMENT if kind == "M" else Quantity.FORCE
        vd, unit = us.to_display(vmax, qty), us.label(qty)
        self.log.appendPlainText(
            f"{names[kind]} diagram — max |{kind}| = {vd:.4e} {unit}")
        self.statusBar().showMessage(
            f"{names[kind]} · max |{kind}| {vd:.3e} {unit}")

    def show_deflection_contour(self, quantity: str = "Umag") -> None:
        """Colour-map a nodal displacement quantity over the slab/shell areas
        (slab plan S7). Solves first, like the member diagrams."""
        if not getattr(self._project, "areas", None):
            QMessageBox.information(
                self, "Deflection contour",
                "Add an area (Draw ▸ Area) to contour a surface.")
            return
        if self._solve() is None:
            return
        vmax = self.view.show_area_contour(self._visible_model(self._model), quantity)
        us = self._units()
        vd, unit = us.to_display(vmax, Quantity.DISP), us.label(Quantity.DISP)
        self.log.appendPlainText(
            f"Deflection contour — max |U| = {vd:.4e} {unit}")
        self.statusBar().showMessage(f"Contour · max |U| {vd:.3e} {unit}")
        self._show_results_tab()

    def pick_and_show_area_force(self) -> None:
        """Ask which shell resultant to contour, then show it (slab plan S7)."""
        import model_geometry as mg
        if not getattr(self._project, "areas", None):
            QMessageBox.information(
                self, "Shell forces",
                "Add an area (Draw ▸ Area) to contour shell forces / moments.")
            return
        from PySide6.QtWidgets import QInputDialog
        quantities = list(mg.AREA_RESULT_QUANTITIES.keys())
        labels = [mg.AREA_RESULT_QUANTITIES[q][0] for q in quantities]
        label, ok = QInputDialog.getItem(self, "Shell forces / moments",
                                         "Quantity:", labels, 0, False)
        if not ok:
            return
        self._show_area_force(quantities[labels.index(label)])

    def _show_area_force(self, quantity: str) -> None:
        import model_geometry as mg
        if self._solve() is None:
            return
        vmax = self.view.show_area_result(self._visible_model(self._model), quantity)
        label, unit, _signed = mg.AREA_RESULT_QUANTITIES.get(
            quantity, (quantity, "", True))
        self.log.appendPlainText(
            f"{label} contour — max |{quantity}| = {vmax:.4e} {unit}")
        self.statusBar().showMessage(
            f"{label} · max {vmax:.3e} {unit}")
        self._show_results_tab()

    def show_slab_reinforcement(self) -> None:
        """Required-reinforcement (As per width) contour from Wood-Armer design
        moments (slab plan S9). Prompts for cover / fy / f'c, then solves."""
        if not getattr(self._project, "areas", None):
            QMessageBox.information(
                self, "Slab reinforcement",
                "Add an area (Draw ▸ Area) to design slab reinforcement.")
            return
        from slab_design_dialog import ReinforcementDialog
        cfg = ReinforcementDialog.configure(
            self, self._project, last=getattr(self, "_rebar_cfg", None))
        if cfg is None:
            return
        self._rebar_cfg = cfg
        if self._solve() is None:
            return
        import model_geometry as mg
        vmax = self.view.show_area_reinforcement(
            self._visible_model(self._model), cfg["quantity"], cover=cfg["cover"], fy=cfg["fy"],
            fc=cfg["fc"])
        label = mg.AREA_REBAR_QUANTITIES.get(cfg["quantity"],
                                             (cfg["quantity"], ""))[0]
        self.log.appendPlainText(
            f"Required reinforcement ({label}) — peak {vmax:.0f} mm²/m")
        self.statusBar().showMessage(
            f"{label} · peak As {vmax:.0f} mm²/m")
        self._show_results_tab()

    def show_punching_check(self) -> None:
        """Punching-shear check at a slab column (slab plan S9): build + solve,
        then read the demand from a chosen column node's reaction and check the
        ACI 318-19 capacity."""
        p = self._project
        if p.ndm != 3 or not getattr(p, "areas", None):
            QMessageBox.information(
                self, "Punching check",
                "Punching needs a 3-D slab model with at least one area.")
            return
        self._model = p.build_model()
        if self._solve() is None:
            return
        from slab_punching_dialog import SlabPunchingDialog
        SlabPunchingDialog.run(self, p, self._model)
        self._show_results_tab()

    def export_slab_results(self) -> None:
        """Export per-node slab results (displacement + M/N/V) to CSV (slab S9).
        Builds + solves, then writes a table for every area node."""
        p = self._project
        if p.ndm != 3 or not getattr(p, "areas", None):
            QMessageBox.information(
                self, "Export slab results",
                "Need a 3-D slab model with at least one area.")
            return
        self._model = p.build_model()
        if self._solve() is None:
            return
        import model_geometry as mg
        table = mg.slab_results_table(self._model)
        if table is None:
            QMessageBox.information(self, "Export slab results",
                                   "No slab results to export.")
            return
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Export slab results", "slab_results.csv",
            "CSV files (*.csv)")
        if not path:
            return
        _write_slab_csv(path, table)
        self.statusBar().showMessage(
            f"Exported {len(table[1])} slab node rows → {path}")

    def show_section_cut(self) -> None:
        """Design-strip section cut: integrate a shell result along a two-node
        cut line (slab plan S7). Builds + solves, then opens the dialog."""
        p = self._project
        if p.ndm != 3 or not getattr(p, "areas", None):
            QMessageBox.information(
                self, "Section cut",
                "Section cuts need a 3-D slab model with at least one area.")
            return
        self._model = p.build_model()
        if self._solve() is None:
            return
        from section_cut_dialog import SectionCutDialog
        SectionCutDialog.run(self, p, self._model)

    def show_pier_forces(self) -> None:
        """Wall pier forces (wall plan W2): integrate the shell membrane stress
        into P/V/M up each pier. Builds + solves, then opens the results dialog.
        Needs at least one wall area carrying a pier label (wall plan W0)."""
        p = self._project
        if p.ndm != 3 or not getattr(p, "areas", None):
            QMessageBox.information(
                self, "Pier forces",
                "Pier forces need a 3-D wall model with at least one area.")
            return
        if not p.pier_names():
            QMessageBox.information(
                self, "Pier forces",
                "No piers defined. Draw a wall (Draw ▸ Wall) with a pier label "
                "— or set the Pier field on a wall area — then try again.")
            return
        self._model = p.build_model()
        if self._solve() is None:
            return
        from pier_forces_dialog import PierForcesDialog
        self._pier_forces_dlg = PierForcesDialog.show_results(
            self, p, self._model, unitsys=self._units())

    def show_wall_design(self) -> None:
        """Wall design (wall plan W3): ACI 318-19 §18.10 special-wall check of
        each pier against its integrated P/V/M demand. Builds + solves, then
        opens the design dialog."""
        p = self._project
        if p.ndm != 3 or not getattr(p, "areas", None):
            QMessageBox.information(
                self, "Wall design",
                "Wall design needs a 3-D wall model with at least one area.")
            return
        if not p.pier_names():
            QMessageBox.information(
                self, "Wall design",
                "No piers defined. Draw a wall (Draw ▸ Wall) with a pier label "
                "— or set the Pier field on a wall area — then try again.")
            return
        self._model = p.build_model()
        if self._solve() is None:
            return
        from wall_design_dialog import WallDesignDialog
        self._wall_design_dlg = WallDesignDialog.show_results(
            self, p, self._model)
        self._show_results_tab()

    def show_design(self) -> None:
        import design
        governing = None
        if getattr(self._project, "combinations", None):
            # envelope every member's DCR over all load combinations
            self._model = self._project.build_model()   # for the view coloring
            if not self._model.elements:
                self.statusBar().showMessage("Nothing to design.")
                return
            dcrs, governing = design.design_envelope(self._project)
        else:
            if self._solve() is None:
                return
            dcrs = design.design_all(self._model, self._project)
        self.view.show_design(self._visible_model(self._model), dcrs)
        vals = {t: d for t, d in dcrs.items() if d is not None}
        if not vals:
            self.log.appendPlainText(
                "Design: no members have a checkable section — set a W-shape "
                "(AISC §H1) or apply a Section-Designer section (P-M-M) to run "
                "the checks.")
            self.statusBar().showMessage("Design: assign a section first.")
            return
        worst = max(vals, key=vals.get)
        mx = vals[worst]
        verdict = "PASS" if mx <= 1.0 else "FAIL"
        scope = (f"envelope of {len(self._project.combinations)} combinations"
                 if governing else "current loads")
        gov = (f", governed by {governing.get(worst)}"
               if governing and governing.get(worst) else "")
        self.log.appendPlainText(
            f"Design (AISC §H1 steel + P-M-M concrete, {scope}): "
            f"{len(vals)} members checked, max DCR = {mx:.2f} at member "
            f"{worst}{gov} — {verdict}")
        self.statusBar().showMessage(f"Design · max DCR {mx:.2f} · {verdict}")

    def manage_load_cases(self) -> None:
        """Add / rename / remove load patterns. Loads whose pattern is deleted
        fall back to the first pattern, and combinations drop factors for
        removed patterns."""
        if self._project is None:
            return
        from editing import LoadCaseDialog
        cases = LoadCaseDialog.edit(self, self._project)
        if cases is None:
            return
        valid = {c.id for c in cases}
        default = cases[0].id

        def _mut():
            self._project.load_cases = cases
            for ld in self._project.loads:
                if ld.case not in valid:
                    ld.case = default
            for ml in self._project.member_loads:
                if ml.case not in valid:
                    ml.case = default
            for combo in self._project.combinations:
                combo.factors = {cid: f for cid, f in combo.factors.items()
                                 if cid in valid}
        self._apply_edit("Edit load patterns", _mut)
        self.log.appendPlainText(
            f"Load patterns: {', '.join(c.name for c in cases)}")

    def manage_combinations(self) -> None:
        """Open the load-combination editor (plan L2): add / rename / delete
        combinations and set each pattern's factor, with a one-click ASCE 7-22
        generator folded in. Replaces the project's combinations on OK."""
        if self._project is None:
            return
        from combinations_dialog import CombinationsDialog
        result = CombinationsDialog.manage(self, self._project)
        if result is None:
            return

        def _mut():
            self._project.combinations = result
        self._apply_edit("Edit load combinations", _mut)
        self.log.appendPlainText(
            f"Load combinations: {len(result)} defined"
            + (f" — {', '.join(c.name for c in result)}" if result else ""))
        self.statusBar().showMessage(
            f"{len(result)} load combination(s) · run Design to envelope them")

    def generate_combinations(self) -> None:
        """Replace the project's combinations with the ASCE 7-22 LRFD strength
        set generated from the load patterns' natures."""
        if self._project is None:
            return
        combos = self._project.generate_asce7_combinations()
        if not combos:
            QMessageBox.information(
                self, "Combinations",
                "No combinations generated — add load patterns with natures "
                "(Dead / Live / Wind …) in Analysis ▸ Load patterns first.")
            return

        def _mut():
            self._project.combinations = combos
        self._apply_edit("Generate combinations", _mut)
        self.log.appendPlainText(
            f"Generated {len(combos)} ASCE 7-22 LRFD combinations:")
        for c in combos:
            terms = " + ".join(f"{f:g}·{self._project.case(cid).name}"
                               for cid, f in c.factors.items()
                               if self._project.case(cid))
            self.log.appendPlainText(f"  {c.name}:  {terms}")
        self.statusBar().showMessage(
            f"{len(combos)} load combinations · run Design to envelope them")

    def open_section_designer(self) -> None:
        """Open the General Section Designer (concrete/PSC/composite) over
        section_gui_core, bridged to this FEM project. When a selected section
        carries a GSD spec it is loaded for editing; the designer can push its
        active section back into the model (``apply_gsd_section``)."""
        try:
            from section_designer import SectionDesignerWindow
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.critical(
                self, "Section Designer unavailable",
                f"Could not load the Section Designer:\n{exc}")
            return
        code = getattr(self._project, "design_code", None) if self._project \
            else None
        # seed from a selected section that already carries a GSD spec
        spec, name = None, None
        for kind, sid in self._selected_refs():
            if kind == "section":
                sec = _find(self._project.sections, sid)
                if sec is not None and getattr(sec, "gsd_spec", None):
                    try:
                        import section_gui_core as core
                        flds = set(core.Spec.__dataclass_fields__)
                        spec = core.Spec(**{k: v for k, v in sec.gsd_spec.items()
                                            if k in flds})
                        name = sec.name
                        code = sec.gsd_code or code
                    except Exception:                  # noqa: BLE001
                        spec = None
                break
        self._sd_win = SectionDesignerWindow(self, code=code, spec=spec,
                                             fem_window=self, section_name=name)
        self._sd_win.show()

    def apply_gsd_section(self, name: str, spec_dict: dict, code: str) -> None:
        """Create or update a FEM ``Section`` from a General-Section-Designer
        section: its gross A / Iz / Iy / J drive the frame analysis and the
        ``gsd_spec`` is stored for re-editing (and future P-M-M member design).
        Called by the Section Designer's 'Apply to FEM model' action."""
        if self._project is None:
            return
        import project as _pmod
        try:
            A, Iz, Iy, J = _pmod._gsd_section_props(spec_dict)
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.critical(self, "Apply failed",
                                 f"Could not build the section:\n{exc}")
            return
        existing = next((s for s in self._project.sections if s.name == name),
                        None)
        sid = existing.id if existing else (
            max((s.id for s in self._project.sections), default=0) + 1)
        new = Section(id=sid, name=name, A=A, Iz=Iz, Iy=Iy, J=J,
                      gsd_spec=dict(spec_dict), gsd_code=code)

        def _mut():
            if existing is not None:
                _replace(self._project.sections, sid, new)
            else:
                self._project.sections.append(new)
        self._apply_edit(f"Apply GSD section '{name}'", _mut, ("section", sid))
        verb = "Updated" if existing else "Added"
        self.log.appendPlainText(
            f"{verb} section '{name}' from Section Designer — "
            f"A={A * 1e4:.1f} cm², Iz={Iz * 1e8:.0f} cm⁴")
        self.statusBar().showMessage(f"{verb} FEM section '{name}'")

    def open_drawings(self) -> None:
        if self._project is None:
            return
        if self._project.ndm == 3:
            QMessageBox.information(
                self, "Drawings",
                "The GA drawing is 2-D only for now — 3-D drawings are a "
                "future step.")
            return
        try:
            from drawing_window import DrawingWindow
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.critical(self, "Drawings unavailable",
                                 f"matplotlib Qt backend failed to load:\n{exc}")
            return
        self._drawing_win = DrawingWindow(self._project, self)
        self._drawing_win.show()

    def _show_undeformed(self) -> None:
        if self._model is not None:
            self._render_model()

    # ------------------------------------------------------------- project I/O
    def load_project(self, project, path=None) -> None:
        self._project = project
        self._path = path
        self._inactive = set()               # a fresh document starts fully active
        if path:
            self._remember_recent(path)          # R8: feed the backstage MRU
        else:
            # A new / generated / demo model adopts the remembered app-default
            # display units (plan U4); an opened file keeps whatever units it
            # was saved with. The app baseline is SI base (N · m) — matching the
            # stored model — until the user picks otherwise.
            from units import FORCE_UNITS, LENGTH_UNITS
            f = str(self._settings.value("units/force", "N"))
            l = str(self._settings.value("units/length", "m"))
            project.force_unit = f if f in FORCE_UNITS else "N"
            project.length_unit = l if l in LENGTH_UNITS else "m"
        self._undo_stack.clear()
        self._last_tree_sig = None       # a new document → always a full rebuild
        self._rebuild()
        self._update_title()
        self.log.appendPlainText(
            f"Loaded project '{project.name}' — {self._model!r}")

    def new_project(self) -> None:
        p = Project()
        p.materials.append(Material(id=1, name="A992", E=200e9, nu=0.3,
                                    rho=7850.0))
        p.sections.append(Section(id=1, name="W12x65", A=0.012323, Iz=2.2185e-4,
                                  shape="W12x65"))
        self.load_project(p, None)

    def new_project_3d(self) -> None:
        from demo_model import demo_project_3d
        self.load_project(demo_project_3d(), None)

    def generate_frame(self) -> None:
        import generators
        from editing import FrameDialog
        params = FrameDialog.get(self, self._units())
        if params is not None:
            self.load_project(generators.frame(**params), None)

    def generate_loads(self) -> None:
        if not self._project.nodes:
            QMessageBox.information(self, "Generate loads",
                                   "Add or generate a structure first.")
            return
        import generators
        from editing import LoadGenDialog
        res = LoadGenDialog.get(self, self._project)
        if res is None:
            return
        kind, mag = res
        if kind == "gravity":
            loads = generators.gravity_loads(self._project, mag)
        else:
            direction = "Y" if kind == "lateral_y" else "X"
            loads = generators.lateral_loads(self._project, mag, direction)
        if not loads:
            QMessageBox.information(self, "Generate loads",
                                   "No loads generated (need nodes above the "
                                   "base level).")
            return
        self._apply_edit(f"Generate loads ({kind})",
                         lambda: self._project.loads.extend(loads))

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open project", "", "femsolver project (*.fsproj *.json)")
        if not path:
            return
        try:
            self.load_project(Project.load(path), path)
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.critical(self, "Open failed", str(exc))

    def save_project(self) -> None:
        self._write(self._path) if self._path else self.save_project_as()

    def save_project_as(self) -> None:
        suggested = self._path or f"{self._project.name}.fsproj"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save project", suggested, "femsolver project (*.fsproj *.json)")
        if path:
            self._path = path
            self._write(path)

    def _write(self, path) -> None:
        try:
            self._project.save(path)
            self._undo_stack.setClean()
            self._remember_recent(path)            # R8: feed the backstage MRU
            self._update_title()
            self.statusBar().showMessage(f"Saved {path}")
            self.log.appendPlainText(f"Saved project to {path}")
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(exc))

    # ---------------------------------------------------------- recent files (R8)
    _RECENT_MAX = 8

    def _recent_files(self) -> list[str]:
        """Persisted most-recently-used project paths (newest first), filtered to
        those that still exist on disk."""
        import os
        raw = self._settings.value("recent/files", [])
        if isinstance(raw, str):                   # a lone entry returns as str
            raw = [raw]
        return [p for p in (raw or []) if p and os.path.exists(p)]

    def _remember_recent(self, path) -> None:
        """Push ``path`` to the front of the MRU list (deduped, capped)."""
        if not path:
            return
        import os
        path = os.path.abspath(path)
        rest = [p for p in self._recent_files() if os.path.abspath(p) != path]
        self._settings.setValue("recent/files", [path, *rest][:self._RECENT_MAX])

    def _open_recent(self, path) -> None:
        """Open a project chosen from the backstage's Recent list."""
        import os
        if not os.path.exists(path):
            QMessageBox.warning(self, "Open failed",
                                f"That file no longer exists:\n{path}")
            return
        try:
            self.load_project(Project.load(path), path)
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.critical(self, "Open failed", str(exc))

    # ------------------------------------------------------------------ edits
    def add_node(self) -> None:
        node = NodeDialog.edit(self, self._project)
        if node is None:
            return
        if _find(self._project.nodes, node.id) is not None:
            QMessageBox.warning(self, "Duplicate", f"Node {node.id} already exists.")
            return
        self._apply_edit("Add node",
                         lambda: self._project.nodes.append(node),
                         ("node", node.id))

    def add_member(self) -> None:
        p = self._project
        if len(p.nodes) < 2 or not p.sections or not p.materials:
            QMessageBox.information(
                self, "Add member",
                "Need at least two nodes and one section + material first.")
            return
        member = MemberDialog.edit(self, p)
        if member is None:
            return
        if _find(p.members, member.id) is not None:
            QMessageBox.warning(self, "Duplicate",
                                f"Member {member.id} already exists.")
            return
        self._apply_edit("Add member",
                         lambda: self._project.members.append(member),
                         ("member", member.id))

    def add_area(self) -> None:
        """Add a surface (shell / plate) area object (slab plan S2). Areas are a
        3-D feature — shells carry 6 DOF/node — so this requires a 3-D model and
        at least one thickness (shell section) + material."""
        p = self._project
        if p.ndm != 3:
            QMessageBox.information(
                self, "Add area",
                "Areas (slabs / walls / shells) are a 3-D feature. Start a 3-D "
                "model (File ▸ New 3-D) to model surfaces.")
            return
        if len(p.nodes) < 3 or not p.shell_sections or not p.materials:
            QMessageBox.information(
                self, "Add area",
                "Need at least three nodes, one thickness (Home ▸ Thickness) "
                "and one material first.")
            return
        seed = [key for kind, key in self._selected_refs() if kind == "node"]
        area = AreaDialog.edit(self, p, seed_nodes=seed[:4])
        if area is None:
            return
        if _find(p.areas, area.id) is not None:
            QMessageBox.warning(self, "Duplicate",
                                f"Area {area.id} already exists.")
            return
        self._apply_edit("Add area",
                         lambda: self._project.areas.append(area),
                         ("area", area.id))

    def draw_wall(self) -> None:
        """Draw a wall by extruding the selected base line upward (wall plan W1).

        Select 2+ base nodes (the wall's bottom edge) — nodes, or members whose
        end nodes form the base — then this extrudes them by a height into
        vertical wall ``Area``(s) with ``role="wall"`` and an optional pier
        label. A 3-D feature, like all areas."""
        import walls
        from editing import WallDialog
        p = self._project
        if p.ndm != 3:
            QMessageBox.information(
                self, "Draw wall",
                "Walls are a 3-D feature. Start a 3-D model (File ▸ New 3-D).")
            return
        if not p.shell_sections or not p.materials:
            QMessageBox.information(
                self, "Draw wall",
                "Add a thickness (Home ▸ Thickness) and a material first.")
            return
        # base nodes = selected nodes + the end nodes of selected members
        node_ids, member_ids = self._sel_nodes_members()
        if len(node_ids) < 2:
            QMessageBox.information(
                self, "Draw wall",
                "Select the wall's base line first: 2+ nodes (or members whose "
                "ends form the base), then Draw ▸ Wall.")
            return
        try:
            base = walls.wall_baseline_from_nodes(p, node_ids)
        except ValueError as e:
            QMessageBox.warning(self, "Draw wall", str(e))
            return
        params = WallDialog.get(self, p, base, self._units())
        if params is None:
            return
        # the areas build_wall_line will create get consecutive ids from the
        # next free one — precompute them so undo/redo restores the selection
        first = p.next_area_id()
        segments = max(len(params["base_nodes"]) - 1, 0)
        refs = [("area", first + k) for k in range(segments)]
        self._apply_edit("Draw wall",
                         lambda: walls.build_wall_line(p, **params),
                         refs[0] if len(refs) == 1 else None)
        if refs:
            self._set_selection(refs)

    def add_diaphragm(self) -> None:
        """Add a rigid floor diaphragm tying the selected joints (slab plan S8).
        A 3-D feature (needs ndf=6); seeds its node list from the selection."""
        p = self._project
        if p.ndm != 3:
            QMessageBox.information(
                self, "Add diaphragm",
                "Rigid diaphragms are a 3-D feature. Start a 3-D model "
                "(File ▸ New 3-D) to add one.")
            return
        seed = [key for kind, key in self._selected_refs() if kind == "node"]
        dia = DiaphragmDialog.edit(self, p, seed_nodes=seed)
        if dia is None:
            return
        if _find(p.diaphragms, dia.id) is not None:
            QMessageBox.warning(self, "Duplicate",
                                f"Diaphragm {dia.id} already exists.")
            return
        self._apply_edit("Add diaphragm",
                         lambda: self._project.diaphragms.append(dia),
                         ("diaphragm", dia.id))

    def add_load(self) -> None:
        if not self._project.nodes:
            QMessageBox.information(self, "Add load", "Add a node first.")
            return
        sel = [ident for kind, ident in self._selection if kind == "node"]
        loads = LoadDialog.create(self, self._project, nodes=sel)
        if not loads:
            return
        start = len(self._project.loads)
        label = f"Add load ({len(loads)} nodes)" if len(loads) > 1 else "Add load"
        self._apply_edit(label,
                         lambda: self._project.loads.extend(loads),
                         ("load", start))

    def add_line_load(self) -> None:
        if not self._project.members:
            QMessageBox.information(self, "Add line load", "Add a member first.")
            return
        from member_load_dialog import MemberLoadDialog
        ml = MemberLoadDialog.edit(self, self._project)
        if ml is None:
            return
        idx = len(self._project.member_loads)
        self._apply_edit("Add line load",
                         lambda: self._project.member_loads.append(ml),
                         ("member_load", idx))

    def add_area_load(self) -> None:
        """Add a uniform area (surface / pressure) load (slab plan S5b)."""
        if not getattr(self._project, "areas", None):
            QMessageBox.information(self, "Add area load",
                                   "Add an area (Draw ▸ Area) first.")
            return
        from area_load_dialog import AreaLoadDialog
        al = AreaLoadDialog.edit(self, self._project)
        if al is None:
            return
        idx = len(self._project.area_loads)
        self._apply_edit("Add area load",
                         lambda: self._project.area_loads.append(al),
                         ("area_load", idx))

    def add_section(self) -> None:
        section = SectionDialog.edit(self, self._project)
        if section is None:
            return
        if _find(self._project.sections, section.id) is not None:
            QMessageBox.warning(self, "Duplicate",
                                f"Section {section.id} already exists.")
            return
        self._apply_edit("Add section",
                         lambda: self._project.sections.append(section),
                         ("section", section.id))

    def manage_materials(self) -> None:
        result = MaterialManagerDialog.manage(self, self._project)
        if result is None:
            return
        self._apply_edit(
            "Edit materials",
            lambda: setattr(self._project, "materials", result))

    def manage_shell_sections(self) -> None:
        """Thickness / shell-section manager (slab plan S4): define the
        through-thickness properties area (slab / wall / shell) objects use."""
        if self._project is None:
            return
        from shell_section_editor import ShellSectionManagerDialog
        result = ShellSectionManagerDialog.manage(self, self._project)
        if result is None:
            return
        self._apply_edit(
            "Edit thickness sections",
            lambda: setattr(self._project, "shell_sections", result))

    def manage_hinges(self) -> None:
        result = HingeManagerDialog.manage(self, self._project)
        if result is None:
            return
        self._apply_edit(
            "Edit hinges",
            lambda: setattr(self._project, "hinges", result))

    def manage_stories(self) -> None:
        """Define the building stories and grid lines (wall plan W4)."""
        if self._project is None:
            return
        from story_grid_dialog import StoryGridDialog
        result = StoryGridDialog.manage(self, self._project, self._units())
        if result is None:
            return
        stories, grids = result

        def _mut():
            self._project.stories = stories
            self._project.grid_lines = grids
        self._apply_edit("Edit stories & grid", _mut)
        self.statusBar().showMessage(
            f"{len(stories)} stor{'y' if len(stories) == 1 else 'ies'}, "
            f"{len(grids)} grid line(s)")

    def replicate_story(self) -> None:
        """Copy a source story's walls/columns/beams up to similar stories
        (wall plan W4c)."""
        p = self._project
        if p is None:
            return
        if len(p.stories) < 2:
            QMessageBox.information(
                self, "Replicate story",
                "Define at least two stories (Home ▸ Stories & grid) first.")
            return
        from story_replicate_dialog import StoryReplicateDialog
        import story_replicate
        picked = StoryReplicateDialog.get(self, p)
        if picked is None:
            return
        source_id, target_ids = picked
        if not target_ids:
            return
        made = {}

        def _mut():
            made.update(story_replicate.replicate_story(p, source_id, target_ids))
        self._apply_edit("Replicate story", _mut)
        self.statusBar().showMessage(
            f"Replicated: +{made.get('areas', 0)} areas, "
            f"+{made.get('members', 0)} members, +{made.get('nodes', 0)} nodes")

    def manage_th_functions(self) -> None:
        """Open the time-history function library (analysis-cases-manager TH-1):
        named ground-motion records referenced by Time-History cases."""
        if self._project is None:
            return
        from th_functions import TimeHistoryFunctionManagerDialog
        result = TimeHistoryFunctionManagerDialog.manage(self, self._project)
        if result is None:
            return
        self._apply_edit(
            "Edit time-history functions",
            lambda: setattr(self._project, "th_functions", result))

    def assign_hinges(self) -> None:
        result = HingeAssignmentDialog.assign(self, self._project)
        if result is None:
            return

        def _apply():
            by_id = {mb.id: mb for mb in self._project.members}
            for mid, hid in result.items():
                if mid in by_id:
                    by_id[mid].hinge = hid
        self._apply_edit("Assign hinges", _apply)

    def run_analysis(self) -> None:
        """Open the Run-analysis control (plan A4): choose which cases to run,
        then dispatch each — saved analysis cases (incl. Linear Static, E3b)
        run from their stored params, while nonlinear / time-history cases
        open their interactive dialogs."""
        if self._project is None:
            return
        from run_analysis_dialog import RunAnalysisDialog
        requests = RunAnalysisDialog.run(self, self._project)
        for req in requests:
            if req[0] == "nonlinear":
                self.run_pushover_dialog(preselect_case=req[1])
            elif req[0] == "case":            # a saved AnalysisCase (E5a)
                self._run_saved_case(req[1])

    def manage_analysis_cases(self) -> None:
        """Open the unified analysis-cases home (plan A1): one list of every
        analysis case (linear static, nonlinear, time history, + roadmap
        placeholders). Commits any nonlinear-case edits, then dispatches a Run
        request to the matching runner."""
        if self._project is None:
            return
        from analysis_cases_dialog import AnalysisCasesDialog
        res = AnalysisCasesDialog.manage(self, self._project)
        if res is None:
            return
        cases, acases, run = res
        if (cases != self._project.nonlinear_cases
                or acases != self._project.analysis_cases):
            def _commit():
                self._project.nonlinear_cases = cases
                self._project.analysis_cases = acases
            self._apply_edit("Edit analysis cases", _commit)
        if run is None:
            return
        kind = run[0]
        if kind == "nonlinear":
            self.run_pushover_dialog(preselect_case=run[1])
        elif kind == "case":
            self._run_saved_case(run[1])
        elif kind == "stages":
            self.run_construction_stages()
        # R10: broaden R4's contextual raise — after *any* analysis dispatched
        # from the cases home, surface the Results tab (History/diagrams/design
        # all live there). A no-op when the ribbon is collapsed.
        self._show_results_tab()

    def _run_saved_case(self, case_id):
        """Run a saved :class:`project.AnalysisCase` from its stored params —
        the type's :mod:`case_types` adapter rebuilds the runtime config
        (``build_config``) and invokes the matching ``run_*`` (``dispatch``).
        No re-prompt."""
        import case_types
        c = self._project.analysis_case(case_id)
        if c is None:
            return None
        ct = case_types.get(c.type)
        if ct is None:
            QMessageBox.warning(self, "Analysis case",
                                f"Unknown analysis type '{c.type}'.")
            return None
        self.log.appendPlainText(
            f"Analysis case '{c.name}' ({ct.type_label}) — running…")
        try:
            config = ct.build_config(self._project, c.params,
                                     initial_condition=c.initial_condition)
        except Exception as exc:                           # noqa: BLE001
            self._project.set_case_status(("analysis", case_id),
                                          "Could not start")
            QMessageBox.warning(self, "Analysis case",
                                f"Cannot run '{c.name}':\n\n{exc}")
            return None
        out = ct.dispatch(self, config)
        self._project.set_case_status(("analysis", case_id), "Finished")
        return out

    def _on_double_click(self, item, _col) -> None:
        ref = item.data(0, Qt.ItemDataRole.UserRole)
        if ref:
            kind, key = ref
            handler = {"node": self._edit_node, "member": self._edit_member,
                       "section": self._edit_section, "load": self._edit_load,
                       "member_load": self._edit_member_load}.get(kind)
            if handler:
                handler(key)
            return
        cat = item.data(0, CAT_ROLE)          # a category header → primary action
        if cat:
            self._category_primary(cat[1])

    # Category "primary" action (double-click / context-menu default): open the
    # category's manager where one exists, else its read-only table.
    def _category_primary(self, key) -> None:
        base = key[0] if isinstance(key, tuple) else key
        managers = {
            "materials": self.manage_materials,
            "hinges": self.manage_hinges,
            "load_cases": self.manage_load_cases,
            "combinations": self.manage_combinations,
            "analysis_cases": self.manage_analysis_cases,
        }
        if base in managers:
            managers[base]()
        else:
            self._open_category_table(key)

    def _on_tree_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        if item is None:
            return
        cat = item.data(0, CAT_ROLE)
        if not cat:
            return
        key = cat[1]
        base = key[0] if isinstance(key, tuple) else key
        menu = QMenu(self)
        menu.addAction(_action(self, "Show Table…", None,
                               lambda: self._open_category_table(key)))
        extra = {
            "materials": ("Manage materials…", self.manage_materials),
            "sections": ("New section…", self.add_section),
            "hinges": ("Manage hinges…", self.manage_hinges),
            "nodes": ("New node…", self.add_node),
            "loads": ("New load…", self.add_load),
            "member_loads": ("New line load…", self.add_line_load),
            "load_cases": ("Manage load patterns…", self.manage_load_cases),
            "combinations": ("Manage combinations…", self.manage_combinations),
            "analysis_cases": ("Manage analysis cases…",
                               self.manage_analysis_cases),
        }.get(base)
        if extra:
            menu.addSeparator()
            menu.addAction(_action(self, extra[0], None, extra[1]))
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _open_category_table(self, key) -> None:
        from model_tables import ModelTableDialog
        base = key[0] if isinstance(key, tuple) else key
        efilter = key[1] if isinstance(key, tuple) and len(key) > 1 else None
        ModelTableDialog.show_category(self, self._project, base,
                                       on_activate=self._table_activate,
                                       on_commit=self._table_commit,
                                       type_filter=efilter)

    def _table_activate(self, ref) -> None:
        """A read-only row was double-clicked in a table → select it (and, for
        editable kinds, open its editor)."""
        if not ref:
            return
        kind, key = ref
        handler = {"node": self._edit_node, "member": self._edit_member,
                   "section": self._edit_section, "load": self._edit_load,
                   "member_load": self._edit_member_load}.get(kind)
        if handler is None:            # kinds with no drill-in editor (yet)
            return
        self._select(ref)
        handler(key)

    def _table_commit(self, ref, field, value) -> bool:
        """Apply an in-place table edit (nav N7) through ``_apply_edit`` so it
        is undoable. Returns True when applied, False (reverting the cell) when
        the field/value is rejected. Each mutate re-resolves the target against
        the live project (``_apply_edit`` swaps in a fresh copy)."""
        kind, key = ref
        p = self._project

        def commit(text, mutate) -> bool:
            self._apply_edit(text, mutate)
            return True

        if kind == "node" and field in ("x", "y", "z"):
            if not any(n.id == key for n in p.nodes):
                return False
            return commit(f"Edit node {key}", lambda: setattr(
                next(n for n in self._project.nodes if n.id == key),
                field, float(value)))

        if kind == "member" and field in ("section", "material"):
            if not any(m.id == key for m in p.members):
                return False
            vid = int(value)
            pool = p.sections if field == "section" else p.materials
            if not any(x.id == vid for x in pool):
                QMessageBox.information(
                    self, "Reassign", f"No {field} with id {vid}.")
                return False
            return commit(f"Set member {key} {field}", lambda: setattr(
                next(m for m in self._project.members if m.id == key),
                field, vid))

        if kind == "section":
            s = next((s for s in p.sections if s.id == key), None)
            if s is None:
                return False
            if field == "name":
                return commit(f"Rename section {key}", lambda: setattr(
                    next(x for x in self._project.sections if x.id == key),
                    "name", str(value)))
            if field in ("A", "Iz", "Iy", "J"):
                if getattr(s, "gsd_spec", None):
                    return False       # geometry is Section-Designer-driven
                return commit(f"Edit section {key} {field}", lambda: setattr(
                    next(x for x in self._project.sections if x.id == key),
                    field, float(value)))
            return False

        if kind == "material" and field in ("name", "E", "nu", "rho",
                                            "fy", "fu"):
            if not any(m.id == key for m in p.materials):
                return False
            cast = str if field == "name" else float
            return commit(f"Edit material {key}", lambda: setattr(
                next(m for m in self._project.materials if m.id == key),
                field, cast(value)))

        if kind == "hinge" and field in ("name", "lp"):
            if not any(h.id == key for h in p.hinges):
                return False
            cast = str if field == "name" else float
            return commit(f"Edit hinge {key}", lambda: setattr(
                next(h for h in self._project.hinges if h.id == key),
                field, cast(value)))

        if kind == "load_case" and field == "name":
            if not any(c.id == key for c in p.load_cases):
                return False
            return commit(f"Rename case {key}", lambda: setattr(
                next(c for c in self._project.load_cases if c.id == key),
                "name", str(value)))

        if kind == "load" and field.startswith("v"):
            comp = int(field[1:])
            if not (0 <= key < len(p.loads)) or comp >= len(p.loads[key].values):
                return False

            def mutate():
                ld = self._project.loads[key]
                vals = list(ld.values)
                vals[comp] = float(value)
                ld.values = tuple(vals)
            return commit(f"Edit load {key + 1}", mutate)

        if kind == "member_load" and field in ("wy", "wz"):
            if not (0 <= key < len(p.member_loads)):
                return False
            return commit(f"Edit line load {key + 1}", lambda: setattr(
                self._project.member_loads[key], field, float(value)))

        return False

    def _edit_node(self, nid) -> None:
        new = NodeDialog.edit(self, self._project, _find(self._project.nodes, nid))
        if new is not None:
            self._apply_edit("Edit node",
                             lambda: _replace(self._project.nodes, nid, new),
                             ("node", nid))

    def _edit_member(self, mid) -> None:
        new = MemberDialog.edit(self, self._project,
                                _find(self._project.members, mid))
        if new is not None:
            self._apply_edit("Edit member",
                             lambda: _replace(self._project.members, mid, new),
                             ("member", mid))

    def _edit_section(self, sid) -> None:
        new = SectionDialog.edit(self, self._project,
                                 _find(self._project.sections, sid))
        if new is not None:
            self._apply_edit("Edit section",
                             lambda: _replace(self._project.sections, sid, new),
                             ("section", sid))

    def _edit_load(self, index) -> None:
        if not 0 <= index < len(self._project.loads):
            return
        new = LoadDialog.edit(self, self._project, self._project.loads[index])
        if new is not None:
            self._apply_edit(
                "Edit load",
                lambda: self._project.loads.__setitem__(index, new),
                ("load", index))

    def _edit_member_load(self, index) -> None:
        if not 0 <= index < len(self._project.member_loads):
            return
        from member_load_dialog import MemberLoadDialog
        new = MemberLoadDialog.edit(self, self._project,
                                    self._project.member_loads[index])
        if new is not None:
            self._apply_edit(
                "Edit line load",
                lambda: self._project.member_loads.__setitem__(index, new),
                ("member_load", index))

    def delete_selected(self) -> None:
        refs = self._selected_refs()
        if not refs:
            return
        p = self._project
        node_ids = {k for (t, k) in refs if t == "node"}
        member_ids = {k for (t, k) in refs if t == "member"}
        area_ids = {k for (t, k) in refs if t == "area"}
        section_ids = {k for (t, k) in refs if t == "section"}
        # Delete loads / line loads by identity (indices shift as we filter).
        loads_del = {id(p.loads[k]) for (t, k) in refs
                     if t == "load" and 0 <= k < len(p.loads)}
        mloads_del = {id(p.member_loads[k]) for (t, k) in refs
                      if t == "member_load" and 0 <= k < len(p.member_loads)}
        # A section is only blocked when a *surviving* member still uses it.
        surviving = [m for m in p.members if m.id not in member_ids
                     and m.n1 not in node_ids and m.n2 not in node_ids]
        blocked = section_ids & {m.section for m in surviving}
        if blocked:
            QMessageBox.information(
                self, "Delete section",
                "Section is used by a member — reassign it first.")
            return

        def mutate():
            if node_ids:
                p.nodes = [n for n in p.nodes if n.id not in node_ids]
            if node_ids or member_ids:
                p.members = [m for m in p.members if m.id not in member_ids
                             and m.n1 not in node_ids and m.n2 not in node_ids]
            if section_ids:
                p.sections = [s for s in p.sections if s.id not in section_ids]
            # areas: delete selected ones, or any that lose a corner node
            if (area_ids or node_ids) and getattr(p, "areas", None):
                p.areas = [a for a in p.areas if a.id not in area_ids
                           and not (node_ids & set(a.nodes))]
                live = {a.id for a in p.areas}
                p.area_loads = [al for al in getattr(p, "area_loads", [])
                                if al.area in live]
            if node_ids or loads_del:
                p.loads = [ld for ld in p.loads if ld.node not in node_ids
                           and id(ld) not in loads_del]
            if mloads_del:
                p.member_loads = [ml for ml in p.member_loads
                                  if id(ml) not in mloads_del]
        n = len(refs)
        self._apply_edit(f"Delete {n} item{'s' if n != 1 else ''}", mutate)
        self._set_selection([])

    # ---------------------------------------------------- selection model (N6)
    # ``self._selection`` is the single source of truth. In leaf mode the tree
    # mirrors it (and user clicks on leaves flow back in via
    # ``_on_selection_changed``); in summary mode the viewport / tables are the
    # only drivers. Every consumer reads ``_selected_refs``.
    def _selected_refs(self):
        return list(self._selection)

    def _ref_exists(self, ref) -> bool:
        kind, key = ref
        p = self._project
        if kind == "node":
            return any(n.id == key for n in p.nodes)
        if kind == "member":
            return any(m.id == key for m in p.members)
        if kind == "area":
            return any(a.id == key for a in getattr(p, "areas", []))
        if kind == "diaphragm":
            return any(d.id == key for d in getattr(p, "diaphragms", []))
        if kind == "section":
            return any(s.id == key for s in p.sections)
        if kind == "load":
            return 0 <= key < len(p.loads)
        if kind == "member_load":
            return 0 <= key < len(p.member_loads)
        if kind == "area_load":
            return 0 <= key < len(getattr(p, "area_loads", []))
        return False

    def _apply_selection_effects(self) -> None:
        # Drop refs an edit removed so a rebuild can't show a deleted item.
        self._selection = [r for r in self._selection if self._ref_exists(r)]
        refs = self._selection
        self._update_sel_status(len(refs))
        geom = [r for r in refs if r[0] in ("node", "member", "area")]
        if geom:
            self.view.highlight(geom)
        else:
            self.view.clear_highlight()
        if not refs:
            self.props.clear_selection()
        elif len(refs) == 1:
            self.props.show_item(self._project, refs[0][0], refs[0][1])
        else:
            self.props.show_multi(self._project, refs)

    def _set_selection(self, refs) -> None:
        """Set the authoritative selection, mirror it onto the tree (leaf mode),
        and refresh the viewport highlight / Properties / status."""
        seen, norm = set(), []
        for r in refs:
            t = tuple(r)
            if t not in seen:
                seen.add(t)
                norm.append(t)
        self._selection = norm
        if not self._summary_mode:
            self._sync_tree_selection(norm)
        self._apply_selection_effects()

    def _sync_tree_selection(self, refs) -> None:
        """Reflect ``refs`` onto the tree's leaves without re-entering the
        selection-changed signal (leaf mode only; a no-op when no leaf matches,
        e.g. in summary mode)."""
        targets = {tuple(r) for r in refs}
        single = len(targets) == 1        # reveal + scroll only a lone pick
        self.tree.blockSignals(True)
        self.tree.clearSelection()
        first = None
        for it in self._iter_tree_items():
            r = it.data(0, Qt.ItemDataRole.UserRole)
            if r is not None and tuple(r) in targets:
                it.setSelected(True)
                if single:
                    parent = it.parent()
                    while parent is not None:
                        parent.setExpanded(True)
                        parent = parent.parent()
                first = first or it
        if first is not None:
            self.tree.setCurrentItem(
                first, 0, QItemSelectionModel.SelectionFlag.NoUpdate)
            if single:
                self.tree.scrollToItem(first)
        self.tree.blockSignals(False)

    def _on_selection_changed(self) -> None:
        """The tree's own selection changed (user clicked leaves). In summary
        mode the tree has no selectable leaves, so it never drives selection."""
        if self._summary_mode:
            return
        seen, refs = set(), []
        for it in self.tree.selectedItems():
            r = it.data(0, Qt.ItemDataRole.UserRole)
            if r and tuple(r) not in seen:
                seen.add(tuple(r))
                refs.append(tuple(r))
        self._selection = refs
        self._apply_selection_effects()

    # ---- modeless "pick from the model" for node/member dialogs -----------
    # An open PickDialog (pick.py) pushes itself here; while one is active a
    # viewport click fills its armed field instead of driving the selection.
    def push_pick_sink(self, sink) -> None:
        self._pick_sinks.append(sink)

    def pop_pick_sink(self, sink) -> None:
        if sink in self._pick_sinks:
            self._pick_sinks.remove(sink)
        # Restore the real selection highlight the transient pick feedback hid.
        self._apply_selection_effects()

    def _active_pick_sink(self):
        return self._pick_sinks[-1] if self._pick_sinks else None

    def _on_pick(self, kind, ident) -> None:
        sink = self._active_pick_sink()
        if sink is not None:
            try:
                if sink.accept_pick(kind, ident):
                    self.view.highlight([(kind, ident)])   # transient feedback
                    return
            except Exception:                              # never break a click
                pass
        additive = bool(QApplication.keyboardModifiers() & (
            Qt.KeyboardModifier.ShiftModifier
            | Qt.KeyboardModifier.ControlModifier))
        ref = (kind, ident)
        if not additive:
            self._set_selection([ref])
        elif ref in self._selection:
            self._set_selection([r for r in self._selection if r != ref])
        else:
            self._set_selection(self._selection + [ref])

    def move_selected(self) -> None:
        refs = self._selected_refs()
        node_ids = {rid for (k, rid) in refs if k == "node"}
        mem = {m.id: m for m in self._project.members}
        for k, rid in refs:
            if k == "member" and rid in mem:
                node_ids.update((mem[rid].n1, mem[rid].n2))
        if not node_ids:
            QMessageBox.information(self, "Move",
                                   "Select nodes or members first "
                                   "(Ctrl-click for several).")
            return
        from editing import MoveDialog
        import transforms
        vec = MoveDialog.get(self, self._project)
        if vec is None:
            return
        dx, dy, dz = vec
        self._apply_edit("Move", lambda: transforms.move_nodes(
            self._project, node_ids, dx, dy, dz))

    def copy_selected(self) -> None:
        refs = self._selected_refs()
        node_ids = {rid for (k, rid) in refs if k == "node"}
        member_ids = {rid for (k, rid) in refs if k == "member"}
        if not node_ids and not member_ids:
            QMessageBox.information(self, "Copy",
                                   "Select nodes or members first "
                                   "(Ctrl-click for several).")
            return
        from editing import CopyDialog
        import transforms
        res = CopyDialog.get(self, self._project)
        if res is None:
            return
        dx, dy, dz, count = res
        self._apply_edit(f"Copy x{count}", lambda: transforms.copy_selection(
            self._project, node_ids, member_ids, dx, dy, dz, count))

    def _sel_nodes_members(self):
        refs = self._selected_refs()
        node_ids = {rid for (k, rid) in refs if k == "node"}
        member_ids = {rid for (k, rid) in refs if k == "member"}
        mem = {m.id: m for m in self._project.members}
        for mid in member_ids:
            if mid in mem:
                node_ids.update((mem[mid].n1, mem[mid].n2))
        return node_ids, member_ids

    def mirror_selected(self) -> None:
        node_ids, member_ids = self._sel_nodes_members()
        if not node_ids and not member_ids:
            QMessageBox.information(self, "Mirror", "Select nodes or members "
                                   "first (Ctrl-click for several).")
            return
        from editing import MirrorDialog
        import transforms
        res = MirrorDialog.get(self, self._project)
        if res is None:
            return
        axis, coord = res
        self._apply_edit("Mirror", lambda: transforms.mirror_selection(
            self._project, node_ids, member_ids, axis, coord))

    def rotate_selected(self) -> None:
        node_ids, _member_ids = self._sel_nodes_members()
        if not node_ids:
            QMessageBox.information(self, "Rotate", "Select nodes or members "
                                   "first (Ctrl-click for several).")
            return
        from editing import RotateDialog
        import transforms
        res = RotateDialog.get(self, self._project)
        if res is None:
            return
        axis, center, angle = res
        self._apply_edit("Rotate", lambda: transforms.rotate_selection(
            self._project, node_ids, axis, center, angle))

    def extrude_selected(self) -> None:
        node_ids, member_ids = self._sel_nodes_members()
        if not node_ids and not member_ids:
            QMessageBox.information(self, "Extrude", "Select nodes or members "
                                   "first (Ctrl-click for several).")
            return
        from editing import CopyDialog
        import transforms
        res = CopyDialog.get(self, self._project, "Extrude selection")
        if res is None:
            return
        dx, dy, dz, count = res
        self._apply_edit(f"Extrude x{count}",
                         lambda: transforms.extrude_selection(
                             self._project, node_ids, member_ids,
                             dx, dy, dz, count))

    def deselect_all(self) -> None:
        self._set_selection([])

    # ----------------------------------------------- active / inactive set
    # A view/edit working set (MIDAS-style activation): inactive entities are
    # hidden from the viewport and left out of picking/selection, but the
    # analysis model is always built from the full project, so deactivating a
    # part of the model never silently changes results. Operations here are pure
    # view state (``self._inactive``) — they do not touch the undo stack.
    def _all_geom_refs(self) -> set:
        """Every selectable geometry ref in the project (node / member / area)."""
        refs = {("node", n.id) for n in self._project.nodes}
        refs |= {("member", m.id) for m in self._project.members}
        refs |= {("area", a.id) for a in getattr(self._project, "areas", [])}
        return refs

    def _with_dependency_nodes(self, refs) -> set:
        """Expand a selection with the end/corner nodes the selected members and
        areas need — so 'isolate' keeps those members/areas drawable."""
        keep = {tuple(r) for r in refs}
        by_member = {m.id: m for m in self._project.members}
        by_area = {a.id: a for a in getattr(self._project, "areas", [])}
        for kind, ident in list(keep):
            if kind == "member" and ident in by_member:
                m = by_member[ident]
                keep.add(("node", m.n1))
                keep.add(("node", m.n2))
            elif kind == "area" and ident in by_area:
                for n in by_area[ident].nodes:
                    keep.add(("node", n))
        return keep

    def inactivate_selected(self) -> None:
        sel = {r for r in (tuple(x) for x in self._selection)
               if r[0] in ("node", "member", "area")}
        if not sel:
            self.statusBar().showMessage(
                "Inactivate — select nodes, members or areas first.")
            return
        self._inactive |= sel
        self._set_selection([])
        self._refresh_active(f"Inactivated {len(sel)} item(s)")

    def activate_selected_only(self) -> None:
        sel = {r for r in (tuple(x) for x in self._selection)
               if r[0] in ("node", "member", "area")}
        if not sel:
            self.statusBar().showMessage(
                "Activate selected only — select the part to isolate first.")
            return
        keep = self._with_dependency_nodes(sel)
        self._inactive = self._all_geom_refs() - keep
        self._refresh_active(f"Isolated {len(sel)} selected item(s)")

    def activate_all(self) -> None:
        if not self._inactive:
            self.statusBar().showMessage("Activate all — nothing is inactive.")
            return
        self._inactive = set()
        self._refresh_active("Activated the whole model")

    def invert_active(self) -> None:
        self._inactive = self._all_geom_refs() - self._inactive
        self._set_selection([])
        self._refresh_active("Inverted the active set")

    def _refresh_active(self, message: str) -> None:
        """Rebuild the viewport for the current active set and report status."""
        self._rebuild()
        self._apply_selection_effects()          # restore any kept highlight
        n = len(self._inactive)
        self.statusBar().showMessage(
            f"{message} — {n} item(s) inactive" if n else message)

    def _prune_inactive(self) -> None:
        """Drop inactive refs whose entity no longer exists (after a delete or an
        undo), so the working set never hides a stale id."""
        if self._inactive:
            self._inactive = {r for r in self._inactive if self._ref_exists(r)}

    def _display_project(self):
        """The project restricted to the *active* working set, for rendering.
        Returns ``self._project`` unchanged when nothing is inactive (the fast,
        allocation-free path). Otherwise a shallow copy whose node/member/area
        lists exclude the inactive entities; a node that would be left dangling
        by hiding all of its elements is dropped too, while a truly standalone
        active node is kept."""
        if not self._inactive:
            return self._project
        p = copy.copy(self._project)             # shares props/loads/materials
        inactive = self._inactive
        hidden_node = lambda nid: ("node", nid) in inactive

        def member_vis(m):
            return (("member", m.id) not in inactive
                    and not hidden_node(m.n1) and not hidden_node(m.n2))

        def area_vis(a):
            return (("area", a.id) not in inactive
                    and all(not hidden_node(n) for n in a.nodes))

        p.members = [m for m in self._project.members if member_vis(m)]
        p.areas = [a for a in getattr(self._project, "areas", []) if area_vis(a)]
        referenced, used = set(), set()
        for m in self._project.members:
            referenced.update((m.n1, m.n2))
        for a in getattr(self._project, "areas", []):
            referenced.update(a.nodes)
        for m in p.members:
            used.update((m.n1, m.n2))
        for a in p.areas:
            used.update(a.nodes)

        def node_vis(n):
            if hidden_node(n.id):
                return False
            if n.id in used:
                return True                      # an endpoint of a visible element
            return n.id not in referenced        # else keep only standalone nodes

        p.nodes = [n for n in self._project.nodes if node_vis(n)]
        return p

    def _render_model(self) -> None:
        """Draw the current active working set (see ``_display_project``). The
        full analysis model stays in ``self._model``; only the viewport is
        filtered."""
        disp = self._display_project()
        view_model = (self._model if disp is self._project
                      else disp.build_model(with_loads=False))
        self.view.set_model(view_model)
        self.view.mark_hinges(disp)
        self.view.set_story_grid(disp)

    def _visible_model(self, solved):
        """A render-only view of a *solved* model restricted to the active
        working set, for the results views (deformed / diagram / contour /
        design). It shares the very same Node/Element objects — so the computed
        displacements and forces are intact — and only filters the node/element
        dicts, so results honour the active set exactly as the modeling view
        does. Returns ``solved`` unchanged when nothing is inactive.

        Element tags follow the build convention (a member's tag is its id; an
        area owns the tags from ``_area_element_tags``), so the same inactive
        refs map straight onto the solved model. A node is dropped when it is
        individually inactive or left orphaned by hiding all of its elements;
        standalone active nodes stay."""
        if not self._inactive or solved is None:
            return solved
        proj = self._project
        inactive = self._inactive
        hidden_node_ids = {nid for (k, nid) in inactive if k == "node"}

        hidden_tags: set = set()
        for m in proj.members:
            if (("member", m.id) in inactive
                    or m.n1 in hidden_node_ids or m.n2 in hidden_node_ids):
                hidden_tags.add(m.id)
        for a in getattr(proj, "areas", []):
            if (("area", a.id) in inactive
                    or any(n in hidden_node_ids for n in a.nodes)):
                hidden_tags.update(proj._area_element_tags(a))

        used, referenced = set(), set()
        for tag, el in solved.elements.items():
            nts = tuple(getattr(el, "node_tags", ()) or ())
            referenced.update(nts)
            if tag not in hidden_tags:
                used.update(nts)

        def node_hidden(nid):
            if nid in hidden_node_ids:
                return True
            if nid in used:
                return False
            return nid in referenced          # only hidden elements used it

        v = copy.copy(solved)
        v._nodes = {t: n for t, n in solved.nodes.items()
                    if not node_hidden(t)}
        v._elements = {t: e for t, e in solved.elements.items()
                       if t not in hidden_tags}
        return v

    def _update_snap(self, *_) -> None:
        self.view.set_snap(self.act_snap.isChecked(), self.snap_spin.value())

    def _update_work_plane(self, *_) -> None:
        """Push the draw work-plane (kind + offset) to the viewport (W1b). The
        offset is entered in the display unit; the view works in SI."""
        kind = self.plane_combo.currentData()
        off = self._units().to_si(self.plane_offset.value(), Quantity.LENGTH)
        self.view.set_work_plane(kind, off)

    def _register_toolbar_commands(self) -> None:
        """Expose the common model/edit commands so they can be pinned onto the
        viewport toolbar via right-click → Customize Toolbar (the view already
        registers the tool/orient/zoom commands itself)."""
        from toolbar_commands import ToolCommand
        specs = [
            ("cmd_node", self.act_add_node, "node", "Model"),
            ("cmd_member", self.act_add_member, "member", "Model"),
            ("cmd_section", self.act_add_section, "section", "Model"),
            ("cmd_materials", self.act_materials, "section", "Model"),
            ("cmd_undo", self.act_undo, "undo", "Edit"),
            ("cmd_redo", self.act_redo, "redo", "Edit"),
            ("cmd_delete", self.act_delete, "delete", "Edit"),
            ("cmd_move", self.act_move, "move", "Edit"),
            ("cmd_copy", self.act_copy, "copy", "Edit"),
            ("cmd_mirror", self.act_mirror, "mirror", "Edit"),
            ("cmd_rotate", self.act_rotate, "rotate", "Edit"),
            ("cmd_extrude", self.act_extrude, "extrude", "Edit"),
            ("cmd_run", self.act_run, "run", "Analysis"),
            ("cmd_inactivate", self.act_inactivate, "inactivate", "Active"),
            ("cmd_activate_only", self.act_activate_only, "activate_only",
             "Active"),
            ("cmd_activate_all", self.act_activate_all, "activate_all",
             "Active"),
            ("cmd_invert_active", self.act_invert_active, "invert_active",
             "Active"),
            ("cmd_undeformed", self.act_undef, "undeformed", "Results"),
            ("cmd_design", self.act_design, "design", "Results"),
        ]
        cmds = []
        for cid, act, icon, grp in specs:
            label = act.text().replace("&", "").replace("…", "").strip()
            cmds.append(ToolCommand(cid, label, icon, group=grp, kind="action",
                                    activate=act.trigger))
        self.view._nav_bar.register_many(cmds)
        self.view._nav_bar.rebuild()
        self._install_mode_switch()

    def _install_mode_switch(self) -> None:
        """Pin the pre/post-processing **Model | Results** switch to the viewport
        tool strip (where the retired rotation lock lived). Picking a segment
        drives the ribbon workspace; the ribbon drives it back so the two never
        disagree."""
        from mode_switch import ModeSwitch
        self._mode_switch = ModeSwitch(self)
        self._mode_switch.modeChanged.connect(self._on_mode_switch)
        self.view._nav_bar.set_trailing_widget(self._mode_switch)
        # reflect a workspace the ribbon reaches by any route (a run's auto-raise,
        # a keyboard tab-cycle, a direct tab click) back onto the switch.
        self._ribbon.tabs.currentChanged.connect(self._reflect_mode_switch)
        self._reflect_mode_switch(self._ribbon.tabs.currentIndex())

    def _on_mode_switch(self, mode: str) -> None:
        import mode_switch as ms
        rb = self._ribbon
        if mode == ms.RESULTS:
            rb.set_current("Results")
        else:
            if rb.tabs.tabText(rb.tabs.currentIndex()) == "Results":
                rb.set_current("Home")
            self._show_undeformed()          # drop any deformed / contour overlay

    def _reflect_mode_switch(self, index: int) -> None:
        import mode_switch as ms
        title = self._ribbon.tabs.tabText(index)
        sw = getattr(self, "_mode_switch", None)
        if sw is not None:
            sw.set_mode(ms.RESULTS if title == "Results" else ms.MODEL)

    def _sync_ribbon_mode(self, mode: str) -> None:
        """Reflect the viewport's active tool in the ribbon's selection group.
        Viewport-only nav tools (orbit / pan / zoom-window) leave none checked."""
        ribbon = {"select": self.act_select, "window": self.act_sel_window,
                  "polygon": self.act_sel_poly, "draw_node": self.act_draw_node,
                  "draw_member": self.act_draw_member,
                  "draw_area": self.act_draw_area}
        act = ribbon.get(mode)
        if act is not None:
            act.setChecked(True)
        else:
            self._mode_group.setExclusive(False)
            for a in self._mode_group.actions():
                a.setChecked(False)
            self._mode_group.setExclusive(True)

    def _set_mode(self, mode: str) -> None:
        self.view.set_mode(mode)
        hints = {
            "select": "Single select — click a node or member.",
            "window": "Window select — drag a box; items fully inside are "
                      "selected.",
            "polygon": "Polygon select — click vertices; double-click or "
                       "right-click to close.",
            "draw_node": "Draw node — click the ground plane to place nodes "
                         "(snapped to 0.5 m).",
            "draw_member": "Draw member — click two nodes to connect them.",
            "draw_area": "Draw area — click corner nodes (3+, in order); click "
                         "the first corner again to close the panel."}
        self.statusBar().showMessage(hints.get(mode, ""))
        rb = getattr(self, "_ribbon", None)    # R4: surface the Draw tab
        if rb is not None and not rb.is_collapsed():
            rb.set_current("Draw")

    def _on_region_select(self, refs, additive=False) -> None:
        picked = [tuple(r) for r in refs]
        combined = (self._selection + picked) if additive else picked
        self._set_selection(combined)
        self.statusBar().showMessage(
            f"Selected {len(self._selection)} item(s)")

    def select_all_nodes(self) -> None:
        self._on_region_select([("node", n.id) for n in self._project.nodes])

    def select_all_members(self) -> None:
        self._on_region_select([("member", m.id) for m in self._project.members])

    def select_all(self) -> None:
        refs = ([("node", n.id) for n in self._project.nodes]
                + [("member", m.id) for m in self._project.members])
        self._on_region_select(refs)

    def select_by_section(self) -> None:
        if not self._project.sections:
            return
        from PySide6.QtWidgets import QInputDialog
        items = [f"{s.id}: {s.name}" for s in self._project.sections]
        choice, ok = QInputDialog.getItem(self, "Select by section",
                                          "Section:", items, 0, False)
        if not ok:
            return
        sid = int(choice.split(":")[0])
        self._on_region_select([("member", m.id) for m in self._project.members
                                if m.section == sid])

    def _draw_add_node(self, x, y, z) -> None:
        nid = max((n.id for n in self._project.nodes), default=0) + 1
        node = Node(id=nid, x=float(x), y=float(y), z=float(z))
        self._apply_edit("Draw node",
                         lambda: self._project.nodes.append(node),
                         ("node", nid))

    def _draw_add_member(self, n1, n2) -> None:
        p = self._project
        if not p.sections or not p.materials:
            QMessageBox.information(self, "Draw member",
                                    "Add a section and material first.")
            return
        mid = max((m.id for m in p.members), default=0) + 1
        member = Member(id=mid, n1=n1, n2=n2, section=p.sections[0].id,
                        material=p.materials[0].id)
        self._apply_edit("Draw member",
                         lambda: p.members.append(member), ("member", mid))

    def _draw_add_area(self, node_ids) -> None:
        """Create an Area from corner nodes picked in the viewport (slab S2
        click-to-draw). Uses the first thickness + material and a 2×2 mesh; edit
        it afterwards via the properties panel / Area dialog."""
        from project import Area
        p = self._project
        if p.ndm != 3:
            QMessageBox.information(self, "Draw area", "Areas need a 3-D model.")
            return
        if not p.shell_sections or not p.materials:
            QMessageBox.information(
                self, "Draw area",
                "Add a thickness (Home ▸ Thickness) and a material first.")
            return
        nodes = [int(n) for n in node_ids]
        if len(nodes) < 3 or len(set(nodes)) != len(nodes):
            return
        aid = p.next_area_id()
        area = Area(id=aid, nodes=nodes, shell_section=p.shell_sections[0].id,
                    material=p.materials[0].id, mesh=(2, 2))
        self._apply_edit("Draw area",
                         lambda: p.areas.append(area), ("area", aid))

    def _apply_from_inspector(self, kind, key, new) -> None:
        p = self._project
        if kind == "node":
            self._apply_edit("Edit node",
                             lambda: _replace(p.nodes, key, new), ("node", key))
        elif kind == "member":
            self._apply_edit("Edit member",
                             lambda: _replace(p.members, key, new),
                             ("member", key))
        elif kind == "section":
            self._apply_edit("Edit section",
                             lambda: _replace(p.sections, key, new),
                             ("section", key))
        elif kind == "load" and 0 <= key < len(p.loads):
            self._apply_edit("Edit load",
                             lambda: p.loads.__setitem__(key, new),
                             ("load", key))

    def _apply_bulk(self, kind, ids, payload) -> None:
        p = self._project
        idset = set(ids)
        if kind == "member":
            sec, mat = payload["section"], payload["material"]

            def mutate():
                for m in p.members:
                    if m.id in idset:
                        m.section = sec
                        m.material = mat
            self._apply_edit(f"Set section/material ({len(ids)} members)",
                             mutate)
        elif kind == "node":
            sup = payload["supports"]
            sup = sup if any(sup) else ()

            def mutate():
                for n in p.nodes:
                    if n.id in idset:
                        n.supports = sup
            self._apply_edit(f"Set fixity ({len(ids)} nodes)", mutate)

    # --------------------------------------------------------------- internals
    def _apply_edit(self, text, mutate, select=None) -> None:
        before = copy.deepcopy(self._project)
        mutate()
        after = copy.deepcopy(self._project)
        self._undo_stack.push(EditCommand(self, text, before, after, select))

    def _restore(self, snapshot, select) -> None:
        self._project = copy.deepcopy(snapshot)
        self._rebuild()
        self._update_title()
        if select:
            self._select(select)

    def _rebuild(self) -> None:
        try:
            self._model = self._project.build_model()
            self._prune_inactive()
            self._render_model()
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.critical(self, "Model error", str(exc))
        self._refresh_tree()
        self._refresh_status()

    def _update_title(self) -> None:
        name = self._project.name if self._project else "Untitled"
        star = "" if self._undo_stack.isClean() else "*"
        where = f" — {self._path}" if self._path else ""
        self.setWindowTitle(f"{star}{name}{where} — femsolver desktop (preview)")

    # ------------------------------------------------------------- model tree
    def _super(self, title):
        """A bold, un-selectable top-level super-group header (Properties /
        Structures / Loads / Analysis) — pure organisation, carries no ref."""
        it = QTreeWidgetItem(self.tree, [title])
        f = it.font(0)
        f.setBold(True)
        it.setFont(0, f)
        it.setFlags(Qt.ItemFlag.ItemIsEnabled)      # not selectable
        it.setFirstColumnSpanned(True)
        it.setData(0, KEY_ROLE, f"grp:{title}")     # persist expand state (N3)
        return it

    def _category(self, parent, title, count, cat_key, icon_name=None):
        """A category row: name in col 0, count badge in col 1, a ``("cat",
        key)`` tag (so the context menu / double-click know what to open) but
        **no** selection ref. Empty categories are muted so the tree also shows
        what the model lacks (charter B2)."""
        it = QTreeWidgetItem(parent, [title, str(count)])
        it.setData(0, CAT_ROLE, ("cat", cat_key))
        it.setData(0, KEY_ROLE, f"cat:{_cat_key_str(cat_key)}")   # N3
        it.setFlags(Qt.ItemFlag.ItemIsEnabled)      # header, not a selectable ref
        if icon_name:
            it.setIcon(0, icons.icon(icon_name, style.ICON))
        it.setTextAlignment(1, Qt.AlignmentFlag.AlignRight
                            | Qt.AlignmentFlag.AlignVCenter)
        muted = QBrush(QColor(style.MUTED))
        it.setForeground(1, muted)
        if not count:
            it.setForeground(0, muted)              # grey the whole empty slot
        return it

    def _leaf(self, parent, label, ref):
        it = QTreeWidgetItem(parent, [label])
        it.setData(0, Qt.ItemDataRole.UserRole, ref)
        return it

    # --- leaf label builders (shared by the full build and the in-place N5
    # refresh, so the two paths can never drift apart) -----------------------
    @staticmethod
    def _node_leaf_label(n, labels, ndm) -> str:
        sup = ([labels[k] for k in range(min(len(n.supports), len(labels)))
                if n.supports[k]] if n.supports else [])
        tag = f"  [{','.join(sup)}]" if sup else ""
        coord = (f"{n.x:g}, {n.y:g}" if ndm == 2
                 else f"{n.x:g}, {n.y:g}, {n.z:g}")
        return f"{n.id}:  ({coord}){tag}"

    @staticmethod
    def _support_leaf_label(n, labels) -> str:
        mask = [labels[k] for k in range(min(len(n.supports), len(labels)))
                if n.supports[k]]
        return f"{n.id}:  [{','.join(mask)}]"

    @staticmethod
    def _section_leaf_label(s) -> str:
        shape = f"  [{s.shape}]" if s.shape else ""
        return f"{s.id}:  {s.name}{shape}"

    @staticmethod
    def _member_leaf_label(m) -> str:
        return (f"{m.id}:  {m.n1} → {m.n2}  "
                f"(sec {m.section}, mat {m.material})")

    @staticmethod
    def _load_leaf_label(ld) -> str:
        vals = ", ".join(f"{v:g}" for v in ld.values)
        return f"node {ld.node}:  ({vals})"

    @staticmethod
    def _mload_leaf_label(ml, ndm) -> str:
        comps = f"wy={ml.wy:g}" + (f", wz={ml.wz:g}" if ndm == 3 else "")
        return f"member {ml.member}:  ({comps})"

    def _populate_tree(self) -> None:
        p = self._project
        labels = dof_labels(p.ndm, p.ndf)
        leaves = not self._summary_mode           # summary mode = counts only (N6)
        scroll = self.tree.verticalScrollBar().value()   # keep the view steady
        self.tree.clear()

        # ---- Properties -------------------------------------------------
        props = self._super("Properties")
        self._category(props, "Materials", len(p.materials),
                       "materials", "design")
        sections = self._category(props, "Sections", len(p.sections),
                                  "sections", "section")
        if leaves:
            for s in p.sections:
                self._leaf(sections, self._section_leaf_label(s),
                           ("section", s.id))
        self._category(props, "Hinge properties", len(p.hinges),
                       "hinges", "run")

        # ---- Structures -------------------------------------------------
        struct = self._super("Structures")
        nodes = self._category(struct, "Nodes", len(p.nodes), "nodes", "node")
        if leaves:
            for n in p.nodes:
                self._leaf(nodes, self._node_leaf_label(n, labels, p.ndm),
                           ("node", n.id))
        # Elements, grouped by element type (charter B3). The type rows carry
        # counts and stay even in summary mode; only the members drop out.
        elements = self._category(struct, "Elements", len(p.members),
                                  "elements", "member")
        by_type: dict[str, list] = {}
        for m in p.members:
            by_type.setdefault(_element_type(m), []).append(m)
        for etype in sorted(by_type):
            bucket = by_type[etype]
            trow = self._category(elements, etype, len(bucket),
                                  ("elements", etype), "member")
            if leaves:
                for m in bucket:
                    self._leaf(trow, self._member_leaf_label(m),
                               ("member", m.id))
        supported = [n for n in p.nodes if n.supports and any(n.supports)]
        supports = self._category(struct, "Supports", len(supported),
                                  "supports", "node")
        if leaves:
            for n in supported:
                self._leaf(supports, self._support_leaf_label(n, labels),
                           ("node", n.id))

        # ---- Loads ------------------------------------------------------
        loadgrp = self._super("Loads")
        self._category(loadgrp, "Load patterns", len(p.load_cases),
                       "load_cases", "load")
        loads = self._category(loadgrp, "Nodal loads", len(p.loads),
                               "loads", "load")
        if leaves:
            for i, ld in enumerate(p.loads):
                self._leaf(loads, self._load_leaf_label(ld), ("load", i))
        mloads = self._category(loadgrp, "Line loads", len(p.member_loads),
                                "member_loads", "load")
        if leaves:
            for i, ml in enumerate(p.member_loads):
                self._leaf(mloads, self._mload_leaf_label(ml, p.ndm),
                           ("member_load", i))
        self._category(loadgrp, "Load combinations", len(p.combinations),
                       "combinations", "load")

        # ---- Analysis ---------------------------------------------------
        an = self._super("Analysis")
        case_rows = (["Linear Static"]
                     + [f"{c.name} (nonlinear)" for c in p.nonlinear_cases]
                     + ["Modal", "Response Spectrum", "Buckling", "Time History"])
        cases = self._category(an, "Analysis cases", len(case_rows),
                               "analysis_cases", "run")
        if leaves:
            for name in case_rows:
                self._leaf(cases, name, None)
        runs = self._category(an, "Results", len(p.runs), "results", "run")
        if leaves:
            for i, r in enumerate(p.runs):
                self._leaf(runs, getattr(r, "name", f"Run {i + 1}"), None)

        # Restore each branch's expand/collapse state from the persisted set
        # (nav N3) so an edit-triggered rebuild keeps the user's outline; then
        # re-apply any active filter (nav N4) to the fresh items.
        self._restore_expansion()
        self._sync_tree_selection(self._selection)       # re-show selection (N6)
        self.tree.verticalScrollBar().setValue(scroll)
        flt = getattr(self, "_nav_filter", None)
        if flt is not None and flt.text().strip():
            self._apply_filter(flt.text())
        self._apply_selection_effects()      # refresh highlight/Properties/status
        self._last_tree_sig = self._tree_signature(p)

    @staticmethod
    def _tree_signature(p):
        """A hashable of everything that determines the tree's *structure* — the
        set of branches/leaves and every count. When two builds share it, only
        leaf *values* changed, so the N5 fast path can refresh labels in place
        instead of tearing the tree down. Ref-less leaves (analysis-case / run
        names) are folded in so they never go stale under the fast path."""
        by_type: dict[str, list] = {}
        for m in p.members:
            by_type.setdefault(_element_type(m), []).append(m.id)
        return (
            tuple(s.id for s in p.sections),
            tuple(n.id for n in p.nodes),
            tuple(sorted((t, tuple(ids)) for t, ids in by_type.items())),
            tuple(n.id for n in p.nodes if n.supports and any(n.supports)),
            len(p.materials), len(p.hinges), len(p.loads), len(p.member_loads),
            len(p.load_cases), len(p.combinations),
            tuple(c.name for c in p.nonlinear_cases),
            tuple(getattr(r, "name", "") for r in p.runs),
        )

    def _refresh_tree(self) -> None:
        """Entry point from a rebuild (N5): refresh leaf labels + counts in
        place when the structure is unchanged, else do a full rebuild. Keeps
        expansion, scroll and selection when nothing structural moved."""
        if (self._last_tree_sig is not None
                and self._tree_signature(self._project) == self._last_tree_sig):
            self._refresh_labels_in_place()
        else:
            self._populate_tree()

    def _refresh_labels_in_place(self) -> None:
        """Update only the leaf label text from the current project — no
        teardown, so scroll / selection / transient expansion all survive.
        Only reached when the structure signature is unchanged, so counts and
        branch membership are already correct."""
        p = self._project
        labels = dof_labels(p.ndm, p.ndf)
        nodes = {n.id: n for n in p.nodes}
        secs = {s.id: s for s in p.sections}
        mems = {m.id: m for m in p.members}
        for it in self._iter_tree_items():
            ref = it.data(0, Qt.ItemDataRole.UserRole)
            if not ref:
                continue
            kind, key = ref
            parent = it.parent()
            pkey = parent.data(0, KEY_ROLE) if parent is not None else None
            if kind == "node" and key in nodes:
                it.setText(0, self._support_leaf_label(nodes[key], labels)
                           if pkey == "cat:supports"
                           else self._node_leaf_label(nodes[key], labels, p.ndm))
            elif kind == "member" and key in mems:
                it.setText(0, self._member_leaf_label(mems[key]))
            elif kind == "section" and key in secs:
                it.setText(0, self._section_leaf_label(secs[key]))
            elif kind == "load" and 0 <= key < len(p.loads):
                it.setText(0, self._load_leaf_label(p.loads[key]))
            elif kind == "member_load" and 0 <= key < len(p.member_loads):
                it.setText(0, self._mload_leaf_label(p.member_loads[key], p.ndm))
        self._apply_selection_effects()      # values changed → refresh Properties

    def _iter_tree_items(self, parent=None):
        """Depth-first walk of every item (used by the recursive finder)."""
        if parent is None:
            tops = [self.tree.topLevelItem(i)
                    for i in range(self.tree.topLevelItemCount())]
        else:
            tops = [parent.child(i) for i in range(parent.childCount())]
        for it in tops:
            yield it
            yield from self._iter_tree_items(it)

    def _find_item(self, ref):
        target = tuple(ref)
        for it in self._iter_tree_items():
            if it.data(0, Qt.ItemDataRole.UserRole) == target:
                return it
        return None

    def _select(self, ref) -> None:
        """Make ``ref`` the sole selection (the tree mirrors it in leaf mode;
        the viewport highlights it in both modes)."""
        self._set_selection([ref])

    # --------------------------------------------------- expand/collapse (N3)
    def _build_nav_panel(self):
        """Wrap the tree in a panel with an Expand-all / Collapse-all header
        (nav plan N3) so the whole outline is one click away either way."""
        panel = QWidget()
        col = QVBoxLayout(panel)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        # Live filter (nav plan N4): hides non-matching branches as you type;
        # matches by id / name / element-type (the leaf label text).
        self._nav_filter = QLineEdit()
        self._nav_filter.setObjectName("navFilter")
        self._nav_filter.setPlaceholderText("Search model…")
        self._nav_filter.setClearButtonEnabled(True)
        self._nav_filter.textChanged.connect(self._apply_filter)
        fwrap = QHBoxLayout()
        fwrap.setContentsMargins(style.SP_XS, style.SP_XS, style.SP_XS,
                                 style.SP_XS)
        fwrap.addWidget(self._nav_filter)
        col.addLayout(fwrap)
        bar = QHBoxLayout()
        bar.setContentsMargins(style.SP_XS, style.SP_XS, style.SP_XS, 0)
        bar.setSpacing(style.SP_XS)
        # Nav-panel-local actions (not app commands) — kept off the ``act_*``
        # namespace the ribbon "homed exactly once" test guards.
        self._nav_expand_act = _action(self, "Expand all", None,
                                       self.expand_all_tree, "fit")
        self._nav_collapse_act = _action(self, "Collapse all", None,
                                         self.collapse_all_tree, "fitsel")
        for act in (self._nav_expand_act, self._nav_collapse_act):
            btn = QToolButton()
            btn.setDefaultAction(act)
            btn.setToolButtonStyle(
                Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            btn.setAutoRaise(True)
            bar.addWidget(btn)
        bar.addStretch(1)
        # Summary-mode toggle (nav plan N6): drop the individual leaves so the
        # tree is counts-only; selection then lives on the viewport + tables.
        self._nav_summary_act = _action(self, "Summary", None,
                                        self.toggle_summary_mode, "frame")
        self._nav_summary_act.setCheckable(True)
        self._nav_summary_act.setChecked(self._summary_mode)
        self._nav_summary_act.setToolTip(
            "Summary view — show category counts only, no individual items")
        self._nav_summary_btn = QToolButton()
        self._nav_summary_btn.setDefaultAction(self._nav_summary_act)
        self._nav_summary_btn.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._nav_summary_btn.setAutoRaise(True)
        bar.addWidget(self._nav_summary_btn)
        col.addLayout(bar)
        col.addWidget(self.tree)
        return panel

    def toggle_summary_mode(self, checked=None) -> None:
        """Flip the tree between the full outline (leaves) and a counts-only
        summary (nav N6). Structure differs, so force a full rebuild; the
        selection is unaffected (it lives in ``self._selection``)."""
        if checked is None:
            checked = self._nav_summary_act.isChecked()
        self._summary_mode = bool(checked)
        self._nav_summary_act.setChecked(self._summary_mode)
        self._settings.setValue("nav/summary", self._summary_mode)
        self._last_tree_sig = None            # structure changed → full rebuild
        if self._project is not None:
            self._populate_tree()

    def _restore_expansion(self) -> None:
        """Set every branch's expanded state from the persisted set, without
        letting the programmatic changes churn the persist signals (N3)."""
        self._building_tree = True
        for it in self._iter_tree_items():
            key = it.data(0, KEY_ROLE)
            if key is not None:
                it.setExpanded(key in self._expanded)
        self._building_tree = False

    # ------------------------------------------------------------- filter (N4)
    def _apply_filter(self, text) -> None:
        """Live-hide branches that don't match ``text`` (by id / name / type).
        A container is shown when it matches or any descendant does; when the
        query clears, everything is unhidden and the saved outline restored."""
        q = (text or "").strip().lower()
        if not q:
            for it in self._iter_tree_items():
                it.setHidden(False)
            self._restore_expansion()
            return

        self._building_tree = True          # expansion here is transient

        def visit(item, forced=False):
            match = forced or q in item.text(0).lower()
            child_hit = False
            for i in range(item.childCount()):
                if visit(item.child(i), forced=match):
                    child_hit = True
            visible = match or child_hit
            item.setHidden(not visible)
            if child_hit or (match and item.childCount()):
                item.setExpanded(True)
            return visible

        for i in range(self.tree.topLevelItemCount()):
            visit(self.tree.topLevelItem(i))
        self._building_tree = False

    def _on_branch_expanded(self, item) -> None:
        if self._building_tree:
            return
        key = item.data(0, KEY_ROLE)
        if key is not None:
            self._expanded.add(key)
            self._persist_expanded()

    def _on_branch_collapsed(self, item) -> None:
        if self._building_tree:
            return
        key = item.data(0, KEY_ROLE)
        if key is not None:
            self._expanded.discard(key)
            self._persist_expanded()

    def _persist_expanded(self) -> None:
        self._settings.setValue("nav/expanded", sorted(self._expanded))

    def expand_all_tree(self) -> None:
        self._building_tree = True
        self.tree.expandAll()
        self._building_tree = False
        for it in self._iter_tree_items():
            key = it.data(0, KEY_ROLE)
            if key is not None:
                self._expanded.add(key)
        self._persist_expanded()

    def collapse_all_tree(self) -> None:
        """Collapse every category/type row but keep the four super-groups open
        so the category list stays visible (the useful 'collapsed' outline)."""
        self._building_tree = True
        for it in self._iter_tree_items():
            key = it.data(0, KEY_ROLE)
            if key is None:
                continue
            keep = key.startswith("grp:")
            it.setExpanded(keep)
            (self._expanded.add if keep else self._expanded.discard)(key)
        self._building_tree = False
        self._persist_expanded()


def _set_icon(act, icon_name):
    """Give an action the themed icon ink (``style.ICON``) and remember the icon
    name on it, so ``MainWindow._retheme_icons()`` can recolour it when the theme
    changes (the ribbon buttons that mirror the action follow automatically)."""
    act.setIcon(icons.icon(icon_name, style.ICON))
    act.setProperty("iconName", icon_name)
    return act


def _action(parent, text, shortcut, slot, icon_name=None) -> QAction:
    act = QAction(text, parent)
    if icon_name:
        _set_icon(act, icon_name)
    if shortcut:
        act.setShortcut(shortcut)
    act.triggered.connect(slot)
    label = text.replace("&", "").rstrip("…")
    act.setToolTip(f"{label}  ({shortcut})" if shortcut else label)
    return act


class RibbonBar(QWidget):
    """CSiBridge-style tabbed ribbon (plan ribbon R1).

    A File "backstage" button plus a tab strip whose active tab swaps a single
    row of captioned tool-groups (:func:`_ribbon_group`). This one compact
    widget replaces both the classic menu bar and the old three stacked ribbon
    rows, so the top chrome is a single strip. Buttons mirror their ``QAction``
    (``setDefaultAction``), so enabled state, re-inked icons and Ctrl-shortcuts
    stay live. Selecting a tab is a pure view swap — no model state.

    The group row can be **collapsed** (plan ribbon R5) to just the tab strip —
    double-click the active tab or Ctrl+F1 — reclaiming a row on demand; while
    collapsed, a single tab click reveals the row transiently until the next
    click outside the ribbon. ``collapsedChanged`` lets the shell persist it.
    """

    collapsedChanged = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ribbonRoot")
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Fixed)
        self._tab_titles: list[str] = []
        self._collapsed = False
        self._transient = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # tab strip: [ File ▾ ]  Home  Draw  Loads  Analysis  Results  View
        strip = QWidget()
        strip.setObjectName("ribbonStrip")
        srow = QHBoxLayout(strip)
        srow.setContentsMargins(style.SP_SM, 0, style.SP_SM, 0)
        srow.setSpacing(style.SP_SM)
        self.file_btn = QToolButton()
        self.file_btn.setObjectName("ribbonFile")
        self.file_btn.setText("File")
        self.file_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.file_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        # The File button opens the backstage (R8), wired by the shell via
        # file_btn.clicked; file_menu stays the canonical list of file actions
        # that the backstage renders and that the homed-once invariant checks.
        self.file_menu = QMenu(self.file_btn)
        srow.addWidget(self.file_btn)
        # Quick Access Toolbar (plan ribbon R11): a few common actions beside
        # File, visible on every tab. Populated by the shell via
        # ``set_quick_actions``; hidden until then.
        self._qat = QWidget()
        self._qat.setObjectName("ribbonQat")
        self._qat_row = QHBoxLayout(self._qat)
        self._qat_row.setContentsMargins(style.SP_SM, 0, style.SP_XS, 0)
        self._qat_row.setSpacing(style.SP_XS)
        srow.addWidget(self._qat)
        self._qat_sep = QFrame()
        self._qat_sep.setObjectName("ribbonVSep")
        self._qat_sep.setFrameShape(QFrame.Shape.VLine)
        srow.addWidget(self._qat_sep)
        self._qat.hide()
        self._qat_sep.hide()
        self.tabs = QTabBar()
        self.tabs.setObjectName("ribbonTabs")
        self.tabs.setDrawBase(False)
        self.tabs.setExpanding(False)
        self.tabs.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        srow.addWidget(self.tabs)
        srow.addStretch(1)
        outer.addWidget(strip)

        # the single group row, swapped per tab
        self.stack = QStackedWidget()
        self.stack.setObjectName("ribbonStack")
        outer.addWidget(self.stack)
        self.tabs.currentChanged.connect(self.stack.setCurrentIndex)
        self.tabs.tabBarDoubleClicked.connect(lambda _i: self.toggle_collapsed())
        self.tabs.tabBarClicked.connect(self._on_tab_clicked)

    # ---- collapse / expand (R5) ----
    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed) -> None:
        """Hide (or show) the group row, leaving just the tab strip."""
        collapsed = bool(collapsed)
        self._end_transient()
        self._collapsed = collapsed
        self.stack.setVisible(not collapsed)
        self.collapsedChanged.emit(collapsed)

    def toggle_collapsed(self) -> None:
        self.set_collapsed(not self._collapsed)

    def _on_tab_clicked(self, _index) -> None:
        # a click while collapsed reveals the row transiently (Office-style)
        if self._collapsed and not self.stack.isVisible():
            self._transient = True
            self.stack.setVisible(True)
            app = QApplication.instance()
            if app is not None:
                app.installEventFilter(self)

    def _end_transient(self) -> None:
        if not self._transient:
            return
        self._transient = False
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        if self._collapsed:
            self.stack.setVisible(False)

    def eventFilter(self, obj, event):
        if self._transient and event.type() == QEvent.Type.MouseButtonPress:
            w = QApplication.widgetAt(event.globalPosition().toPoint())
            if w is None or not (w is self or self.isAncestorOf(w)):
                self._end_transient()          # clicked away → re-collapse
        return super().eventFilter(obj, event)

    def add_tab(self, title, groups) -> QWidget:
        """Add a ribbon tab whose body is a row of captioned groups."""
        page = RibbonPage(groups)
        self.tabs.addTab(title)
        self.stack.addWidget(page)
        self._tab_titles.append(title)
        return page

    def set_quick_actions(self, actions) -> None:
        """R11 — fill the Quick Access Toolbar with icon-only buttons mirroring
        ``actions`` (tab-independent common commands). Empty hides the QAT."""
        self.qat_buttons: list[QToolButton] = []
        while self._qat_row.count():
            item = self._qat_row.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for act in actions:
            btn = QToolButton()
            btn.setObjectName("qatBtn")
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            btn.setDefaultAction(act)
            btn.setAutoRaise(True)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._qat_row.addWidget(btn)
            self.qat_buttons.append(btn)
        self._qat.setVisible(bool(actions))
        self._qat_sep.setVisible(bool(actions))

    def set_current(self, which) -> None:
        idx = which if isinstance(which, int) else self._tab_titles.index(which)
        self.tabs.setCurrentIndex(idx)
        self.stack.setCurrentIndex(idx)

    def page(self, title) -> QWidget:
        """The group-row widget behind ``title`` (for tests / lookups)."""
        return self.stack.widget(self._tab_titles.index(title))


class RibbonPage(QWidget):
    """One ribbon tab's body: a left-aligned row of captioned tool-groups
    (:func:`_ribbon_group`) separated by thin vertical rules (plan ribbon R1).

    Width overflow (plan ribbon R7): when the row is wider than the tab, the
    lowest-priority groups — from the right — collapse into a single ``»`` popup
    button (their actions become a grouped menu) instead of being clipped. The
    split is recomputed on every resize and reverses when the width returns.
    """

    def __init__(self, groups, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ribbonPage")
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(style.SP_SM, style.SP_XS, style.SP_SM,
                                     style.SP_XS)
        self._row.setSpacing(0)
        self._groups: list[dict] = []
        for i, (caption, items) in enumerate(groups):
            sep = None
            if i:
                sep = QFrame()
                sep.setObjectName("ribbonVSep")
                sep.setFrameShape(QFrame.Shape.VLine)
                self._row.addWidget(sep)
            gw = _ribbon_group(caption, items)
            self._row.addWidget(gw)
            self._groups.append(
                {"caption": caption, "widget": gw, "sep": sep,
                 "actions": [it[0] for it in items if isinstance(it, tuple)]})
        # overflow "»" button — hidden until a group spills off the right
        self._more = QToolButton()
        self._more.setObjectName("ribbonMore")
        self._more.setText("»")
        self._more.setToolTip("More groups")
        self._more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._more.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._more_menu = QMenu(self._more)
        self._more.setMenu(self._more_menu)
        self._more_menu.aboutToShow.connect(self._fill_overflow)
        self._row.addWidget(self._more)
        self._more.hide()
        self._row.addStretch(1)
        self._shown = len(self._groups)          # groups currently inline

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._relayout()

    def _group_width(self, g) -> int:
        w = g["widget"].sizeHint().width()
        if g["sep"] is not None:
            w += g["sep"].sizeHint().width() + style.SP_SM
        return w

    def _relayout(self) -> None:
        avail = self.width() - 2 * style.SP_SM
        widths = [self._group_width(g) for g in self._groups]
        if sum(widths) <= avail:
            shown = len(self._groups)
        else:
            budget = avail - self._more.sizeHint().width() - style.SP_SM
            shown, acc = 0, 0
            for w in widths:
                if acc + w > budget:
                    break
                acc += w
                shown += 1
            shown = max(shown, 1)                # always keep one group inline
        if shown == self._shown:
            return
        self._shown = shown
        for idx, g in enumerate(self._groups):
            vis = idx < shown
            g["widget"].setVisible(vis)
            if g["sep"] is not None:
                g["sep"].setVisible(vis)
        self._more.setVisible(shown < len(self._groups))

    def _fill_overflow(self) -> None:
        """(Re)build the overflow menu from the groups that don't fit."""
        self._more_menu.clear()
        for g in self._groups[self._shown:]:
            self._more_menu.addSection(g["caption"])
            for a in g["actions"]:
                self._more_menu.addAction(a)

    def hidden_captions(self) -> list[str]:
        """Captions of the groups currently collapsed into the overflow popup."""
        return [g["caption"] for g in self._groups[self._shown:]]


def _ribbon_group(caption, items) -> QWidget:
    """A captioned, text-under-icon tool-button cluster — one ribbon group
    (plan ribbon R1). Each item is either a ``(QAction, short_label)`` pair —
    the short label is shown under the icon (the action keeps its full text) —
    or a bare ``QWidget`` (e.g. the grid-snap spin box) added as-is. Buttons
    mirror their action via ``setDefaultAction`` so enabled state and
    Ctrl-shortcuts (e.g. act_run's Ctrl+R) stay live.
    """
    box = QWidget()
    box.setObjectName("ribbonGroup")
    col = QVBoxLayout(box)
    col.setContentsMargins(style.SP_SM, style.SP_XS, style.SP_SM, 0)
    col.setSpacing(style.SP_XS)
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(style.SP_XS)
    for it in items:
        if isinstance(it, tuple):
            act, label = it
            act.setIconText(label)
            btn = QToolButton()
            btn.setObjectName("ribbonBtn")
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            btn.setDefaultAction(act)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            row.addWidget(btn)
        else:
            row.addWidget(it)          # a bare widget, e.g. the grid-snap spin
    col.addLayout(row)
    cap = QLabel(caption.upper())
    cap.setObjectName("ribbonCap")
    cap.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    col.addWidget(cap)
    return box


def _write_slab_csv(path, table) -> None:
    """Write a ``(headers, rows)`` slab-results table to ``path`` as CSV
    (slab plan S9 export)."""
    import csv
    headers, rows = table
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def _find(items, item_id):
    for it in items:
        if it.id == item_id:
            return it
    return None


def _replace(items, item_id, new) -> None:
    for i, it in enumerate(items):
        if it.id == item_id:
            items[i] = new
            return
