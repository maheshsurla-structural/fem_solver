"""The application shell: a docked-panel main window over the 3-D viewport.

The native version of the AdSec-style layout — a central viewport, a left
model tree, a bottom output log — the frame every future view (analysis,
design, drawings) docks into. Edits act on the ``Project`` (the source of
truth); the solver ``Model`` is recompiled and re-rendered after each change.
"""
from __future__ import annotations

import copy

from PySide6.QtCore import QItemSelectionModel, QSettings, QSize, Qt
from PySide6.QtGui import QAction, QActionGroup, QUndoStack
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox,
                               QDockWidget, QDoubleSpinBox, QFileDialog,
                               QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                               QPlainTextEdit, QSlider, QToolButton,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                               QWidget)

import icons
import model_geometry as mg
import style
from commands import EditCommand
from editing import (LoadDialog, MemberDialog, NodeDialog, SectionDialog,
                     dof_labels)
from hinge_editor import HingeAssignmentDialog, HingeManagerDialog
from material_editor import MaterialManagerDialog
from model_view import ModelView
from project import Material, Member, Node, Project, Section
from properties import PropertiesPanel

PRODUCT_MONO = "FS"              # femsolver desktop — titlebar / taskbar mark
PRODUCT_ACCENT = "#2563eb"       # stable brand blue (independent of theme)


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
        self.setCentralWidget(self.view)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Model"])
        self.tree.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
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
        self.view.set_region_callback(self._on_region_select)

        self._build_menu()
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
        self._st_units = _lbl()
        for w in (self._st_model, self._st_sel, self._st_coord, self._st_units):
            sb.addPermanentWidget(w)

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
        self._st_units.setText(f"{p.force_unit} · {p.length_unit}")
        self._update_sel_status(len(self._selected_refs()))   # stay consistent

    def _update_sel_status(self, n: int) -> None:
        self._st_sel.setText(f"{n} selected" if n else "")

    def _on_cursor_coords(self, x, y) -> None:
        unit = self._project.length_unit if self._project else "m"
        self._st_coord.setText(f"X {x:.2f}  Y {y:.2f} {unit}")

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

    # --------------------------------------------------------------- menu / UI
    def _build_menu(self) -> None:
        self.act_undo = self._undo_stack.createUndoAction(self, "&Undo")
        self.act_undo.setShortcut("Ctrl+Z")
        self.act_redo = self._undo_stack.createRedoAction(self, "&Redo")
        self.act_redo.setShortcut("Ctrl+Y")

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
        self.act_add_load = _action(self, "Add &load…", None, self.add_load,
                                    "load")
        self.act_add_lineload = _action(self, "Add l&ine load…", None,
                                        self.add_line_load, "load")
        self.act_add_section = _action(self, "Add &section…", None,
                                       self.add_section, "section")
        self.act_materials = _action(self, "&Materials…", None,
                                     self.manage_materials)
        self.act_hinges = _action(self, "&Hinges…", None, self.manage_hinges)
        self.act_assign_hinges = _action(self, "Assign &hinges…", None,
                                         self.assign_hinges)
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
        self.act_runanalysis = _action(self, "Run &analysis…", None,
                                       self.run_analysis, "run")
        self.act_run = _action(self, "&Run (linear static)", "Ctrl+R",
                               self.run_linear_static, "run")
        self.act_pushover = _action(self, "Nonlinear &pushover…", None,
                                    self.run_pushover_dialog, "run")
        self.act_timehistory = _action(self, "Nonlinear &time history…", None,
                                       self.run_timehistory_dialog, "run")
        self.act_runhistory = _action(self, "Run &history…", None,
                                      self.show_run_history)
        self.act_checkmodel = _action(self, "&Check model…", None,
                                      self.check_model)
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

        self.act_diag_n = _set_icon(QAction("Axial &N", self), "axial")
        self.act_diag_n.triggered.connect(lambda *_: self.show_diagram("N"))
        self.act_diag_v = _set_icon(QAction("Shear &V", self), "shear")
        self.act_diag_v.triggered.connect(lambda *_: self.show_diagram("V"))
        self.act_diag_m = _set_icon(QAction("Moment &M", self), "moment")
        self.act_diag_m.triggered.connect(lambda *_: self.show_diagram("M"))
        self.act_design = _set_icon(QAction("&Design (DCR)", self), "design")
        self.act_design.triggered.connect(self.show_design)
        self.act_loadcases = _action(self, "Load &cases…", None,
                                     self.manage_load_cases, "load")
        self.act_editcombos = _action(self, "Load com&binations…", None,
                                      self.manage_combinations, "loadsgen")
        self.act_gencombos = _action(self, "Generate ASCE-7 &combinations", None,
                                     self.generate_combinations, "loadsgen")
        self.act_drawings = _action(self, "&Drawings…", None, self.open_drawings,
                                    "drawings")
        self.act_sectiondesigner = _action(
            self, "&Section Designer…", None, self.open_section_designer,
            "sectiondesigner")
        self.act_theme = _action(self, "Toggle &theme (light / dark)", None,
                                 self.toggle_theme)
        self.act_density = _action(self, "Compact &density", None,
                                   self.toggle_density)
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
        self._mode_group = QActionGroup(self)
        for a in (self.act_select, self.act_sel_window, self.act_sel_poly,
                  self.act_draw_node, self.act_draw_member):
            self._mode_group.addAction(a)
        self.act_snap = _set_icon(QAction("&Snap to grid", self), "snap")
        self.act_snap.setCheckable(True)
        self.act_snap.setChecked(True)
        self.act_snap.toggled.connect(self._update_snap)
        self.act_sel_all_nodes = _action(self, "Select all &nodes", None,
                                         self.select_all_nodes)
        self.act_sel_all_members = _action(self, "Select all &members", None,
                                           self.select_all_members)
        self.act_sel_all = _action(self, "Select &all", "Ctrl+A", self.select_all)
        self.act_sel_by_section = _action(self, "Select by &section…", None,
                                          self.select_by_section)
        self.snap_spin = QDoubleSpinBox()
        self.snap_spin.setRange(0.05, 10.0)
        self.snap_spin.setSingleStep(0.05)
        self.snap_spin.setDecimals(2)
        self.snap_spin.setValue(0.5)
        self.snap_spin.setPrefix("grid ")
        self.snap_spin.setSuffix(" m")
        self.snap_spin.valueChanged.connect(lambda _v: self._update_snap())

        file_menu = self.menuBar().addMenu("&File")
        for a in (self.act_new, self.act_new3d, self.act_open, self.act_save,
                  self.act_saveas):
            file_menu.addAction(a)
        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(self.act_undo)
        edit_menu.addAction(self.act_redo)
        edit_menu.addSeparator()
        for a in (self.act_add_node, self.act_add_member, self.act_add_section,
                  self.act_materials, self.act_hinges, self.act_assign_hinges,
                  self.act_delete):
            edit_menu.addAction(a)
        edit_menu.addSeparator()
        edit_menu.addAction(self.act_deselect)
        edit_menu.addAction(self.act_move)
        edit_menu.addAction(self.act_copy)
        edit_menu.addAction(self.act_mirror)
        edit_menu.addAction(self.act_rotate)
        edit_menu.addAction(self.act_extrude)
        gen_menu = self.menuBar().addMenu("&Generate")
        gen_menu.addAction(self.act_gen)

        # Loads home (plan S1) — every load-definition surface in one place,
        # instead of "Add load" buried under Edit and "Generate loads" under
        # Generate. Reads: define cases -> apply loads -> combine.
        loads_menu = self.menuBar().addMenu("&Loads")
        loads_menu.addAction(self.act_loadcases)
        loads_menu.addSeparator()
        loads_menu.addAction(self.act_add_load)
        loads_menu.addAction(self.act_add_lineload)
        loads_menu.addAction(self.act_genloads)
        loads_menu.addSeparator()
        loads_menu.addAction(self.act_editcombos)
        loads_menu.addAction(self.act_gencombos)

        # Analysis — slimmed to define -> run -> view (plan S1). The direct
        # pushover / time-history launchers now live in the Analysis-cases
        # home (act_analysiscases); linear static keeps its Ctrl+R via the
        # Run-analysis control and the Analysis toolbar.
        analysis_menu = self.menuBar().addMenu("&Analysis")
        analysis_menu.addAction(self.act_analysiscases)
        analysis_menu.addAction(self.act_runanalysis)
        analysis_menu.addSeparator()
        results_menu = analysis_menu.addMenu(icons.icon("undeformed", style.ICON),
                                             "&Results && diagrams")
        _set_icon(results_menu.menuAction(), "undeformed")
        results_menu.addAction(self.act_undef)
        results_menu.addSeparator()
        results_menu.addAction(self.act_diag_n)
        results_menu.addAction(self.act_diag_v)
        results_menu.addAction(self.act_diag_m)
        results_menu.addSeparator()
        results_menu.addAction(self.act_runhistory)
        analysis_menu.addSeparator()
        analysis_menu.addAction(self.act_design)
        analysis_menu.addAction(self.act_checkmodel)
        select_menu = self.menuBar().addMenu("&Select")
        select_menu.addAction(self.act_select)
        select_menu.addAction(self.act_sel_window)
        select_menu.addAction(self.act_sel_poly)
        select_menu.addSeparator()
        select_menu.addAction(self.act_sel_all_nodes)
        select_menu.addAction(self.act_sel_all_members)
        select_menu.addAction(self.act_sel_all)
        select_menu.addAction(self.act_sel_by_section)
        select_menu.addSeparator()
        select_menu.addAction(self.act_deselect)
        draw_menu = self.menuBar().addMenu("&Draw")
        for a in (self.act_draw_node, self.act_draw_member):
            draw_menu.addAction(a)
        draw_menu.addSeparator()
        draw_menu.addAction(self.act_snap)
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.act_fit)
        view_menu.addSeparator()
        for a in (self.act_v_iso, self.act_v_top, self.act_v_bottom,
                  self.act_v_front, self.act_v_back, self.act_v_left,
                  self.act_v_right):
            view_menu.addAction(a)
        view_menu.addSeparator()
        view_menu.addAction(self.act_drawings)
        view_menu.addSeparator()
        view_menu.addAction(self.act_theme)
        view_menu.addAction(self.act_density)
        tools_menu = self.menuBar().addMenu("&Tools")
        tools_menu.addAction(self.act_sectiondesigner)

        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)

        def _toolbar(name, items):
            tb = self.addToolBar(name)
            for a in items:
                tb.addSeparator() if a is None else tb.addAction(a)
            return tb

        def _ribbon_toolbar(name, groups):
            tb = self.addToolBar(name)
            tb.setObjectName("ribbonBar")
            tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            tb.setMovable(False)
            for i, (caption, items) in enumerate(groups):
                if i:
                    tb.addSeparator()
                tb.addWidget(_ribbon_group(caption, items))
            return tb

        # One cohesive, captioned ribbon band across the top (plan S3): a Home
        # row over the S2 Loads and Analysis rows, each a set of grouped,
        # text-under-icon buttons with eyebrow captions — evoking the
        # CSiBridge/Midas ribbon without a tab framework. It replaces the old
        # scattered icon toolbars (File/Edit/Generate/View/Tools); their full
        # action sets stay in the menus (plan S1) and the ribbon carries the
        # common ones. The modal draw/select tool palettes stay icon-only on
        # their own row below the band. Loads and Analysis keep their own rows
        # (the Analysis groups are too wide to share one — see S2).
        _ribbon_toolbar("Home", (
            ("File", ((self.act_new, "New"), (self.act_open, "Open"),
                      (self.act_save, "Save"))),
            ("Model", ((self.act_add_node, "Node"),
                       (self.act_add_member, "Member"),
                       (self.act_add_section, "Section"))),
            ("Edit", ((self.act_undo, "Undo"), (self.act_redo, "Redo"))),
            ("View", ((self.act_fit, "Fit"), (self.act_v_iso, "Iso"))),
            ("Tools", ((self.act_sectiondesigner, "Designer"),)),
        ))
        self.addToolBarBreak()
        _ribbon_toolbar("Loads", (
            ("Loads", ((self.act_loadcases, "Cases"),
                       (self.act_add_load, "Nodal"),
                       (self.act_add_lineload, "Line"),
                       (self.act_genloads, "Generate"))),
            ("Combinations", ((self.act_editcombos, "Combos"),
                              (self.act_gencombos, "ASCE-7"))),
        ))
        self.addToolBarBreak()
        _ribbon_toolbar("Analysis", (
            ("Analyse", ((self.act_analysiscases, "Cases"),
                         (self.act_runanalysis, "Run"),
                         (self.act_run, "Linear"))),
            ("Results", ((self.act_undef, "Undeformed"),
                         (self.act_diag_n, "Axial"),
                         (self.act_diag_v, "Shear"),
                         (self.act_diag_m, "Moment"))),
            ("Design", ((self.act_design, "Design"),
                        (self.act_checkmodel, "Check"))),
        ))
        self.addToolBarBreak()
        # Modal drafting/selection tool palettes — checkable, icon-only.
        _toolbar("Select", (self.act_select, self.act_sel_window,
                            self.act_sel_poly, self.act_deselect))
        tb_draw = _toolbar("Draw", (self.act_draw_node, self.act_draw_member,
                                    self.act_snap))
        tb_draw.addWidget(self.snap_spin)

    # ---------------------------------------------------------------- analysis
    def _solve(self):
        if self._model is None or not self._model.elements:
            self.statusBar().showMessage("Nothing to solve — add members first.")
            return None
        from femsolver import LinearStaticAnalysis
        return LinearStaticAnalysis(self._model).run()

    def run_linear_static(self):
        info = self._solve()
        if info is None:
            return None
        dmax = mg.max_translation(self._model)
        span = mg.model_span(self._model)
        scale = (0.08 * span / dmax) if dmax > 0 else 1.0
        self.view.show_deformed(self._model, scale)
        self.log.appendPlainText(
            f"Linear static solved: neq={info.get('neq', '?')}, "
            f"max|u| = {dmax:.4e} m, deformation ×{scale:.0f}")
        self.statusBar().showMessage(
            f"Solved · max|u| {dmax:.3e} m · deformation ×{scale:.0f}")
        return info

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

    def run_timehistory_dialog(self) -> None:
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
        dlg = TimeHistoryDialog(self, p)
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
        self.view.show_nl_step(self._model, st.node_disp, st.member_damage,
                               self._nl_scale, member_state=st.member_state,
                               color_mode=mode)

    def show_diagram(self, kind: str) -> None:
        if self._solve() is None:
            return
        vmax = self.view.show_diagram(self._model, kind)
        names = {"N": "Axial N", "V": "Shear V", "M": "Moment M"}
        units = {"N": "N", "V": "N", "M": "N·m"}
        self.log.appendPlainText(
            f"{names[kind]} diagram — max |{kind}| = {vmax:.4e} {units[kind]}")
        self.statusBar().showMessage(
            f"{names[kind]} · max |{kind}| {vmax:.3e} {units[kind]}")

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
        self.view.show_design(self._model, dcrs)
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
        """Add / rename / remove load cases. Loads whose case is deleted fall
        back to the first case, and combinations drop factors for removed
        cases."""
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
        self._apply_edit("Edit load cases", _mut)
        self.log.appendPlainText(
            f"Load cases: {', '.join(c.name for c in cases)}")

    def manage_combinations(self) -> None:
        """Open the load-combination editor (plan L2): add / rename / delete
        combinations and set each case's factor, with a one-click ASCE 7-22
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
        set generated from the load cases' natures."""
        if self._project is None:
            return
        combos = self._project.generate_asce7_combinations()
        if not combos:
            QMessageBox.information(
                self, "Combinations",
                "No combinations generated — add load cases with natures "
                "(Dead / Live / Wind …) in Analysis ▸ Load cases first.")
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
            self.view.set_model(self._model)

    # ------------------------------------------------------------- project I/O
    def load_project(self, project, path=None) -> None:
        self._project = project
        self._path = path
        self._undo_stack.clear()
        self._rebuild()
        self._update_title()
        self.log.appendPlainText(
            f"Loaded project '{project.name}' — {self._model!r}")

    def new_project(self) -> None:
        p = Project()
        p.materials.append(Material(id=1, name="A992", E=200e9, nu=0.3))
        p.sections.append(Section(id=1, name="W12x65", A=0.012323, Iz=2.2185e-4,
                                  shape="W12x65"))
        self.load_project(p, None)

    def new_project_3d(self) -> None:
        from demo_model import demo_project_3d
        self.load_project(demo_project_3d(), None)

    def generate_frame(self) -> None:
        import generators
        from editing import FrameDialog
        params = FrameDialog.get(self)
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
            self._update_title()
            self.statusBar().showMessage(f"Saved {path}")
            self.log.appendPlainText(f"Saved project to {path}")
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(exc))

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

    def add_load(self) -> None:
        if not self._project.nodes:
            QMessageBox.information(self, "Add load", "Add a node first.")
            return
        load = LoadDialog.edit(self, self._project)
        if load is None:
            return
        idx = len(self._project.loads)
        self._apply_edit("Add load",
                         lambda: self._project.loads.append(load),
                         ("load", idx))

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

    def manage_hinges(self) -> None:
        result = HingeManagerDialog.manage(self, self._project)
        if result is None:
            return
        self._apply_edit(
            "Edit hinges",
            lambda: setattr(self._project, "hinges", result))

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
        run the batchable linear-static inline (live status), then open the
        interactive dialogs for any queued nonlinear / time-history cases."""
        if self._project is None:
            return
        from run_analysis_dialog import RunAnalysisDialog
        requests = RunAnalysisDialog.run(self, self._project,
                                         self.run_linear_static)
        for req in requests:
            if req[0] == "nonlinear":
                self.run_pushover_dialog(preselect_case=req[1])
            elif req[0] == "timehistory":
                self.run_timehistory_dialog()

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
        cases, run = res
        if cases != self._project.nonlinear_cases:
            self._apply_edit(
                "Edit analysis cases",
                lambda: setattr(self._project, "nonlinear_cases", cases))
        if run is None:
            return
        kind = run[0]
        if kind == "linear":
            self.run_linear_static()
        elif kind == "nonlinear":
            self.run_pushover_dialog(preselect_case=run[1])
        elif kind == "timehistory":
            self.run_timehistory_dialog()

    def _on_double_click(self, item, _col) -> None:
        ref = item.data(0, Qt.ItemDataRole.UserRole)
        if not ref:
            return
        kind, key = ref
        {"node": self._edit_node, "member": self._edit_member,
         "section": self._edit_section, "load": self._edit_load,
         "member_load": self._edit_member_load}[kind](key)

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
        item = self.tree.currentItem()
        ref = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        if not ref:
            return
        kind, key = ref
        p = self._project
        if kind == "section" and any(m.section == key for m in p.members):
            QMessageBox.information(
                self, "Delete section",
                "Section is used by a member — reassign it first.")
            return
        def mutate():
            if kind == "node":
                p.nodes = [n for n in p.nodes if n.id != key]
                p.members = [m for m in p.members if key not in (m.n1, m.n2)]
                p.loads = [ld for ld in p.loads if ld.node != key]
            elif kind == "member":
                p.members = [m for m in p.members if m.id != key]
            elif kind == "section":
                p.sections = [s for s in p.sections if s.id != key]
            elif kind == "load" and 0 <= key < len(p.loads):
                del p.loads[key]
            elif kind == "member_load" and 0 <= key < len(p.member_loads):
                del p.member_loads[key]
        self._apply_edit(f"Delete {kind}", mutate)

    def _on_selection_changed(self) -> None:
        refs = self._selected_refs()
        self._update_sel_status(len(refs))
        geom = [r for r in refs if r[0] in ("node", "member")]
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

    def _on_pick(self, kind, ident) -> None:
        additive = bool(QApplication.keyboardModifiers() & (
            Qt.KeyboardModifier.ShiftModifier
            | Qt.KeyboardModifier.ControlModifier))
        if not additive:
            self._select((kind, ident))
            return
        item = self._find_item((kind, ident))
        if item is not None:
            item.setSelected(not item.isSelected())
            self.tree.setCurrentItem(
                item, 0, QItemSelectionModel.SelectionFlag.NoUpdate)

    def _selected_refs(self):
        refs = []
        for it in self.tree.selectedItems():
            r = it.data(0, Qt.ItemDataRole.UserRole)
            if r:
                refs.append(tuple(r))
        return refs

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
        self.tree.clearSelection()
        self.tree.setCurrentItem(None)
        self.props.clear_selection()
        self.view.clear_highlight()

    def _update_snap(self, *_) -> None:
        self.view.set_snap(self.act_snap.isChecked(), self.snap_spin.value())

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
            "draw_member": "Draw member — click two nodes to connect them."}
        self.statusBar().showMessage(hints.get(mode, ""))

    def _on_region_select(self, refs, additive=False) -> None:
        targets = {tuple(r) for r in refs}
        if additive:
            targets |= set(self._selected_refs())
        self.tree.blockSignals(True)
        self.tree.clearSelection()
        first = None
        for i in range(self.tree.topLevelItemCount()):
            grp = self.tree.topLevelItem(i)
            for j in range(grp.childCount()):
                child = grp.child(j)
                if child.data(0, Qt.ItemDataRole.UserRole) in targets:
                    child.setSelected(True)
                    first = first or child
        if first:
            self.tree.setCurrentItem(
                first, 0, QItemSelectionModel.SelectionFlag.NoUpdate)
        self.tree.blockSignals(False)
        self._on_selection_changed()
        self.statusBar().showMessage(f"Selected {len(targets)} item(s)")

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
            self.view.set_model(self._model)
            self.view.mark_hinges(self._project)
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.critical(self, "Model error", str(exc))
        self._populate_tree()
        self._refresh_status()

    def _update_title(self) -> None:
        name = self._project.name if self._project else "Untitled"
        star = "" if self._undo_stack.isClean() else "*"
        where = f" — {self._path}" if self._path else ""
        self.setWindowTitle(f"{star}{name}{where} — femsolver desktop (preview)")

    def _populate_tree(self) -> None:
        p = self._project
        labels = dof_labels(p.ndm, p.ndf)
        self.tree.clear()
        nodes = QTreeWidgetItem(self.tree, [f"Nodes ({len(p.nodes)})"])
        for n in p.nodes:
            sup = ([labels[k] for k in range(min(len(n.supports), len(labels)))
                    if n.supports[k]] if n.supports else [])
            tag = f"  [{','.join(sup)}]" if sup else ""
            coord = (f"{n.x:g}, {n.y:g}" if p.ndm == 2
                     else f"{n.x:g}, {n.y:g}, {n.z:g}")
            it = QTreeWidgetItem(nodes, [f"{n.id}:  ({coord}){tag}"])
            it.setData(0, Qt.ItemDataRole.UserRole, ("node", n.id))
        members = QTreeWidgetItem(self.tree, [f"Members ({len(p.members)})"])
        for m in p.members:
            it = QTreeWidgetItem(members, [
                f"{m.id}:  {m.n1} → {m.n2}  (sec {m.section}, mat {m.material})"])
            it.setData(0, Qt.ItemDataRole.UserRole, ("member", m.id))
        sections = QTreeWidgetItem(self.tree, [f"Sections ({len(p.sections)})"])
        for s in p.sections:
            shape = f"  [{s.shape}]" if s.shape else ""
            it = QTreeWidgetItem(sections, [f"{s.id}:  {s.name}{shape}"])
            it.setData(0, Qt.ItemDataRole.UserRole, ("section", s.id))
        loads = QTreeWidgetItem(self.tree, [f"Loads ({len(p.loads)})"])
        for i, ld in enumerate(p.loads):
            vals = ", ".join(f"{v:g}" for v in ld.values)
            it = QTreeWidgetItem(loads, [f"node {ld.node}:  ({vals})"])
            it.setData(0, Qt.ItemDataRole.UserRole, ("load", i))
        mloads = QTreeWidgetItem(
            self.tree, [f"Line loads ({len(p.member_loads)})"])
        for i, ml in enumerate(p.member_loads):
            comps = f"wy={ml.wy:g}" + (f", wz={ml.wz:g}" if p.ndm == 3 else "")
            it = QTreeWidgetItem(mloads, [f"member {ml.member}:  ({comps})"])
            it.setData(0, Qt.ItemDataRole.UserRole, ("member_load", i))
        for grp in (nodes, members, sections, loads, mloads):
            grp.setExpanded(True)

    def _find_item(self, ref):
        target = tuple(ref)
        for i in range(self.tree.topLevelItemCount()):
            grp = self.tree.topLevelItem(i)
            for j in range(grp.childCount()):
                child = grp.child(j)
                if child.data(0, Qt.ItemDataRole.UserRole) == target:
                    return child
        return None

    def _select(self, ref) -> None:
        item = self._find_item(ref)
        if item is not None:
            self.tree.clearSelection()
            item.setSelected(True)
            self.tree.setCurrentItem(item)


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


def _ribbon_group(caption, items) -> QWidget:
    """A captioned, text-under-icon tool-button cluster — one ribbon group
    (plan S2). ``items`` is a list of ``(QAction, short_label)`` pairs; the
    short label is the caption shown under the icon, while the menus keep the
    action's full text. Buttons mirror their action via ``setDefaultAction`` so
    enabled state and Ctrl-shortcuts (e.g. act_run's Ctrl+R) stay live.
    """
    box = QWidget()
    box.setObjectName("ribbonGroup")
    col = QVBoxLayout(box)
    col.setContentsMargins(style.SP_SM, style.SP_XS, style.SP_SM, 0)
    col.setSpacing(style.SP_XS)
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(style.SP_XS)
    for act, label in items:
        act.setIconText(label)
        btn = QToolButton()
        btn.setObjectName("ribbonBtn")
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        btn.setDefaultAction(act)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        row.addWidget(btn)
    col.addLayout(row)
    cap = QLabel(caption.upper())
    cap.setObjectName("ribbonCap")
    cap.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    col.addWidget(cap)
    return box


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
