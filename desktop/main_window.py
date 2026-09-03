"""The application shell: a docked-panel main window over the 3-D viewport.

The native version of the AdSec-style layout — a central viewport, a left
model tree, a bottom output log — the frame every future view (analysis,
design, drawings) docks into. Edits act on the ``Project`` (the source of
truth); the solver ``Model`` is recompiled and re-rendered after each change.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QDockWidget, QFileDialog, QMainWindow,
                               QMessageBox, QPlainTextEdit, QTreeWidget,
                               QTreeWidgetItem)

import model_geometry as mg
from editing import LoadDialog, MemberDialog, NodeDialog, dof_labels
from model_view import ModelView
from project import Material, Project, Section


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1280, 820)
        self._model = None
        self._project = None
        self._path = None
        self._dirty = False

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

        self._build_menu()
        self.statusBar().showMessage("Ready")

    # --------------------------------------------------------------- menu / UI
    def _build_menu(self) -> None:
        self.act_new = _action(self, "&New", "Ctrl+N", self.new_project)
        self.act_open = _action(self, "&Open…", "Ctrl+O", self.open_project)
        self.act_save = _action(self, "&Save", "Ctrl+S", self.save_project)
        self.act_saveas = _action(self, "Save &As…", "Ctrl+Shift+S",
                                  self.save_project_as)

        self.act_add_node = _action(self, "Add &node…", "Ctrl+Shift+N",
                                    self.add_node)
        self.act_add_member = _action(self, "Add &member…", "Ctrl+Shift+M",
                                      self.add_member)
        self.act_add_load = _action(self, "Add &load…", None, self.add_load)
        self.act_delete = _action(self, "&Delete", "Del", self.delete_selected)

        self.act_run = _action(self, "&Run (linear static)", "Ctrl+R",
                               self.run_linear_static)
        self.act_undef = _action(self, "&Undeformed", None, self._show_undeformed)
        self.act_fit = _action(self, "&Fit", "F", self.view.fit)

        file_menu = self.menuBar().addMenu("&File")
        for a in (self.act_new, self.act_open, self.act_save, self.act_saveas):
            file_menu.addAction(a)
        edit_menu = self.menuBar().addMenu("&Edit")
        for a in (self.act_add_node, self.act_add_member, self.act_add_load,
                  self.act_delete):
            edit_menu.addAction(a)
        analysis_menu = self.menuBar().addMenu("&Analysis")
        analysis_menu.addAction(self.act_run)
        analysis_menu.addAction(self.act_undef)
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.act_fit)

        tb = self.addToolBar("Main")
        for a in (self.act_open, self.act_save, None, self.act_add_node,
                  self.act_add_member, self.act_add_load, self.act_delete,
                  None, self.act_run, self.act_undef, None, self.act_fit):
            tb.addSeparator() if a is None else tb.addAction(a)

    # ---------------------------------------------------------------- analysis
    def run_linear_static(self) -> None:
        if self._model is None or not self._model.elements:
            self.statusBar().showMessage("Nothing to solve — add members first.")
            return
        from femsolver import LinearStaticAnalysis
        info = LinearStaticAnalysis(self._model).run()
        dmax = mg.max_translation(self._model)
        span = mg.model_span(self._model)
        scale = (0.08 * span / dmax) if dmax > 0 else 1.0
        self.view.show_deformed(self._model, scale)
        self.log.appendPlainText(
            f"Linear static solved: neq={info.get('neq', '?')}, "
            f"max|u| = {dmax:.4e} m, deformation ×{scale:.0f}")
        self.statusBar().showMessage(
            f"Solved · max|u| {dmax:.3e} m · deformation ×{scale:.0f}")

    def _show_undeformed(self) -> None:
        if self._model is not None:
            self.view.set_model(self._model)

    # ------------------------------------------------------------- project I/O
    def load_project(self, project, path=None) -> None:
        self._project = project
        self._path = path
        self._dirty = False
        self._rebuild()
        self._update_title()
        self.log.appendPlainText(
            f"Loaded project '{project.name}' — {self._model!r}")

    def new_project(self) -> None:
        p = Project()
        p.materials.append(Material(id=1, name="Steel", E=200e9, nu=0.3))
        p.sections.append(Section(id=1, name="Default", A=6.0e-3, Iz=2.0e-4))
        self.load_project(p, None)

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
            self._dirty = False
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
        self._project.nodes.append(node)
        self._after_edit(("node", node.id))

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
        p.members.append(member)
        self._after_edit(("member", member.id))

    def add_load(self) -> None:
        if not self._project.nodes:
            QMessageBox.information(self, "Add load", "Add a node first.")
            return
        load = LoadDialog.edit(self, self._project)
        if load is None:
            return
        self._project.loads.append(load)
        self._after_edit(("load", len(self._project.loads) - 1))

    def _on_double_click(self, item, _col) -> None:
        ref = item.data(0, Qt.ItemDataRole.UserRole)
        if not ref:
            return
        kind, key = ref
        {"node": self._edit_node, "member": self._edit_member,
         "load": self._edit_load}[kind](key)

    def _edit_node(self, nid) -> None:
        new = NodeDialog.edit(self, self._project, _find(self._project.nodes, nid))
        if new is not None:
            _replace(self._project.nodes, nid, new)
            self._after_edit(("node", nid))

    def _edit_member(self, mid) -> None:
        new = MemberDialog.edit(self, self._project,
                                _find(self._project.members, mid))
        if new is not None:
            _replace(self._project.members, mid, new)
            self._after_edit(("member", mid))

    def _edit_load(self, index) -> None:
        if not 0 <= index < len(self._project.loads):
            return
        new = LoadDialog.edit(self, self._project, self._project.loads[index])
        if new is not None:
            self._project.loads[index] = new
            self._after_edit(("load", index))

    def delete_selected(self) -> None:
        item = self.tree.currentItem()
        ref = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        if not ref:
            return
        kind, key = ref
        p = self._project
        if kind == "node":
            p.nodes = [n for n in p.nodes if n.id != key]
            p.members = [m for m in p.members if key not in (m.n1, m.n2)]
            p.loads = [ld for ld in p.loads if ld.node != key]
        elif kind == "member":
            p.members = [m for m in p.members if m.id != key]
        elif kind == "load" and 0 <= key < len(p.loads):
            del p.loads[key]
        self._after_edit()

    # --------------------------------------------------------------- internals
    def _after_edit(self, select=None) -> None:
        self._dirty = True
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
        star = "*" if self._dirty else ""
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
            it = QTreeWidgetItem(nodes, [f"{n.id}:  ({n.x:g}, {n.y:g}){tag}"])
            it.setData(0, Qt.ItemDataRole.UserRole, ("node", n.id))
        members = QTreeWidgetItem(self.tree, [f"Members ({len(p.members)})"])
        for m in p.members:
            it = QTreeWidgetItem(members, [
                f"{m.id}:  {m.n1} → {m.n2}  (sec {m.section}, mat {m.material})"])
            it.setData(0, Qt.ItemDataRole.UserRole, ("member", m.id))
        loads = QTreeWidgetItem(self.tree, [f"Loads ({len(p.loads)})"])
        for i, ld in enumerate(p.loads):
            vals = ", ".join(f"{v:g}" for v in ld.values)
            it = QTreeWidgetItem(loads, [f"node {ld.node}:  ({vals})"])
            it.setData(0, Qt.ItemDataRole.UserRole, ("load", i))
        for grp in (nodes, members, loads):
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
