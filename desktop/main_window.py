"""The application shell: a docked-panel main window over the 3-D viewport.

The native version of the AdSec-style layout — a central viewport, a left
model tree, a bottom output log — the frame every future view (analysis,
design, drawings) docks into. Edits act on the ``Project`` (the source of
truth); the solver ``Model`` is recompiled and re-rendered after each change.
"""
from __future__ import annotations

import copy

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QUndoStack
from PySide6.QtWidgets import (QDockWidget, QDoubleSpinBox, QFileDialog,
                               QMainWindow, QMessageBox, QPlainTextEdit,
                               QTreeWidget, QTreeWidgetItem)

import model_geometry as mg
from commands import EditCommand
from editing import (LoadDialog, MemberDialog, NodeDialog, SectionDialog,
                     dof_labels)
from model_view import ModelView
from project import Material, Member, Node, Project, Section
from properties import PropertiesPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1280, 820)
        self._model = None
        self._project = None
        self._path = None
        self._undo_stack = QUndoStack(self)

        self.view = ModelView(self)
        self.setCentralWidget(self.view)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Model"])
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

        self.props = PropertiesPanel(self._apply_from_inspector, self)
        dock_props = QDockWidget("Properties", self)
        dock_props.setWidget(self.props)
        dock_props.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                                   | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_props)
        self.tree.currentItemChanged.connect(self._on_tree_selection)
        self.view.set_pick_callback(self._on_pick)
        self.view.set_add_node_callback(self._draw_add_node)
        self.view.set_add_member_callback(self._draw_add_member)

        self._build_menu()
        self.statusBar().showMessage("Ready")

    # --------------------------------------------------------------- menu / UI
    def _build_menu(self) -> None:
        self.act_undo = self._undo_stack.createUndoAction(self, "&Undo")
        self.act_undo.setShortcut("Ctrl+Z")
        self.act_redo = self._undo_stack.createRedoAction(self, "&Redo")
        self.act_redo.setShortcut("Ctrl+Y")

        self.act_new = _action(self, "&New", "Ctrl+N", self.new_project)
        self.act_new3d = _action(self, "New &3-D frame", None, self.new_project_3d)
        self.act_gen = _action(self, "&Generate frame…", "Ctrl+G",
                               self.generate_frame)
        self.act_open = _action(self, "&Open…", "Ctrl+O", self.open_project)
        self.act_save = _action(self, "&Save", "Ctrl+S", self.save_project)
        self.act_saveas = _action(self, "Save &As…", "Ctrl+Shift+S",
                                  self.save_project_as)

        self.act_add_node = _action(self, "Add &node…", "Ctrl+Shift+N",
                                    self.add_node)
        self.act_add_member = _action(self, "Add &member…", "Ctrl+Shift+M",
                                      self.add_member)
        self.act_add_load = _action(self, "Add &load…", None, self.add_load)
        self.act_add_section = _action(self, "Add &section…", None,
                                       self.add_section)
        self.act_delete = _action(self, "&Delete", "Del", self.delete_selected)

        self.act_run = _action(self, "&Run (linear static)", "Ctrl+R",
                               self.run_linear_static)
        self.act_undef = _action(self, "&Undeformed", None, self._show_undeformed)
        self.act_fit = _action(self, "&Fit", "F", self.view.fit)
        self.act_diag_n = QAction("Axial &N", self)
        self.act_diag_n.triggered.connect(lambda *_: self.show_diagram("N"))
        self.act_diag_v = QAction("Shear &V", self)
        self.act_diag_v.triggered.connect(lambda *_: self.show_diagram("V"))
        self.act_diag_m = QAction("Moment &M", self)
        self.act_diag_m.triggered.connect(lambda *_: self.show_diagram("M"))
        self.act_design = QAction("&Design (DCR)", self)
        self.act_design.triggered.connect(self.show_design)
        self.act_drawings = _action(self, "&Drawings…", None, self.open_drawings)

        self.act_select = QAction("&Select", self, checkable=True)
        self.act_select.setChecked(True)
        self.act_select.triggered.connect(lambda: self._set_mode("select"))
        self.act_draw_node = QAction("Draw n&ode", self, checkable=True)
        self.act_draw_node.triggered.connect(lambda: self._set_mode("draw_node"))
        self.act_draw_member = QAction("Draw m&ember", self, checkable=True)
        self.act_draw_member.triggered.connect(
            lambda: self._set_mode("draw_member"))
        self._mode_group = QActionGroup(self)
        for a in (self.act_select, self.act_draw_node, self.act_draw_member):
            self._mode_group.addAction(a)
        self.act_snap = QAction("&Snap to grid", self, checkable=True)
        self.act_snap.setChecked(True)
        self.act_snap.toggled.connect(self._update_snap)
        self.snap_spin = QDoubleSpinBox()
        self.snap_spin.setRange(0.05, 10.0)
        self.snap_spin.setSingleStep(0.05)
        self.snap_spin.setDecimals(2)
        self.snap_spin.setValue(0.5)
        self.snap_spin.setPrefix("grid ")
        self.snap_spin.setSuffix(" m")
        self.snap_spin.valueChanged.connect(lambda _v: self._update_snap())

        file_menu = self.menuBar().addMenu("&File")
        for a in (self.act_new, self.act_new3d, self.act_gen, self.act_open,
                  self.act_save, self.act_saveas):
            file_menu.addAction(a)
        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(self.act_undo)
        edit_menu.addAction(self.act_redo)
        edit_menu.addSeparator()
        for a in (self.act_add_node, self.act_add_member, self.act_add_section,
                  self.act_add_load, self.act_delete):
            edit_menu.addAction(a)
        analysis_menu = self.menuBar().addMenu("&Analysis")
        analysis_menu.addAction(self.act_run)
        analysis_menu.addAction(self.act_undef)
        analysis_menu.addSeparator()
        for a in (self.act_diag_n, self.act_diag_v, self.act_diag_m,
                  self.act_design):
            analysis_menu.addAction(a)
        draw_menu = self.menuBar().addMenu("&Draw")
        for a in (self.act_select, self.act_draw_node, self.act_draw_member):
            draw_menu.addAction(a)
        draw_menu.addSeparator()
        draw_menu.addAction(self.act_snap)
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.act_fit)
        view_menu.addAction(self.act_drawings)

        tb = self.addToolBar("Main")
        for a in (self.act_open, self.act_save, None, self.act_undo,
                  self.act_redo, None, self.act_add_node, self.act_add_member,
                  self.act_add_section, self.act_add_load, self.act_delete,
                  None, self.act_select, self.act_draw_node,
                  self.act_draw_member, None, self.act_run, self.act_undef,
                  self.act_diag_n, self.act_diag_v, self.act_diag_m,
                  self.act_design, None, self.act_fit, self.act_drawings):
            tb.addSeparator() if a is None else tb.addAction(a)
        tb.addSeparator()
        tb.addAction(self.act_snap)
        tb.addWidget(self.snap_spin)

    # ---------------------------------------------------------------- analysis
    def _solve(self):
        if self._model is None or not self._model.elements:
            self.statusBar().showMessage("Nothing to solve — add members first.")
            return None
        from femsolver import LinearStaticAnalysis
        return LinearStaticAnalysis(self._model).run()

    def run_linear_static(self) -> None:
        info = self._solve()
        if info is None:
            return
        dmax = mg.max_translation(self._model)
        span = mg.model_span(self._model)
        scale = (0.08 * span / dmax) if dmax > 0 else 1.0
        self.view.show_deformed(self._model, scale)
        self.log.appendPlainText(
            f"Linear static solved: neq={info.get('neq', '?')}, "
            f"max|u| = {dmax:.4e} m, deformation ×{scale:.0f}")
        self.statusBar().showMessage(
            f"Solved · max|u| {dmax:.3e} m · deformation ×{scale:.0f}")

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
        if self._solve() is None:
            return
        import design
        dcrs = design.design_all(self._model, self._project)
        self.view.show_design(self._model, dcrs)
        vals = {t: d for t, d in dcrs.items() if d is not None}
        if not vals:
            self.log.appendPlainText(
                "Design: no members have a steel shape — set a section's shape "
                "(e.g. 'W12x65') to run the AISC checks.")
            self.statusBar().showMessage("Design: assign a W-shape first.")
            return
        worst = max(vals, key=vals.get)
        mx = vals[worst]
        verdict = "PASS" if mx <= 1.0 else "FAIL"
        self.log.appendPlainText(
            f"Design (AISC 360-22 §H1): {len(vals)} members checked, "
            f"max DCR = {mx:.2f} at member {worst} — {verdict}")
        self.statusBar().showMessage(f"Design · max DCR {mx:.2f} · {verdict}")

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

    def _on_double_click(self, item, _col) -> None:
        ref = item.data(0, Qt.ItemDataRole.UserRole)
        if not ref:
            return
        kind, key = ref
        {"node": self._edit_node, "member": self._edit_member,
         "section": self._edit_section, "load": self._edit_load}[kind](key)

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
        self._apply_edit(f"Delete {kind}", mutate)

    def _on_tree_selection(self, current, _previous) -> None:
        ref = current.data(0, Qt.ItemDataRole.UserRole) if current else None
        if not ref:
            self.props.clear_selection()
            self.view.clear_highlight()
            return
        self.props.show_item(self._project, ref[0], ref[1])
        if ref[0] in ("node", "member"):
            self.view.highlight(ref[0], ref[1])
        else:
            self.view.clear_highlight()

    def _on_pick(self, kind, ident) -> None:
        self._select((kind, ident))

    def _update_snap(self, *_) -> None:
        self.view.set_snap(self.act_snap.isChecked(), self.snap_spin.value())

    def _set_mode(self, mode: str) -> None:
        self.view.set_mode(mode)
        hints = {
            "select": "Select — click a node or member to select it.",
            "draw_node": "Draw node — click the ground plane to place nodes "
                         "(snapped to 0.5 m).",
            "draw_member": "Draw member — click two nodes to connect them."}
        self.statusBar().showMessage(hints.get(mode, ""))

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
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.critical(self, "Model error", str(exc))
        self._populate_tree()
        self.statusBar().showMessage(
            f"{len(self._project.nodes)} nodes · {len(self._project.members)} "
            f"members · {len(self._project.loads)} loads")

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
        for grp in (nodes, members, sections, loads):
            grp.setExpanded(True)

    def _select(self, ref) -> None:
        target = tuple(ref)
        for i in range(self.tree.topLevelItemCount()):
            grp = self.tree.topLevelItem(i)
            for j in range(grp.childCount()):
                child = grp.child(j)
                if child.data(0, Qt.ItemDataRole.UserRole) == target:
                    self.tree.setCurrentItem(child)
                    return


def _action(parent, text, shortcut, slot) -> QAction:
    act = QAction(text, parent)
    if shortcut:
        act.setShortcut(shortcut)
    act.triggered.connect(slot)
    return act


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
