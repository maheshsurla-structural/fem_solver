"""Material editor — define elastic **and inelastic (fiber)** materials with a
live stress-strain preview (GUI-1, plan §14).

``MaterialDialog`` edits one :class:`project.Material`: an *Identity* card (id /
name / kind) over a *Parameters* card of kind-specific fields, beside a
matplotlib σ-ε curve that redraws as you type — the curve is computed by the
engine via :func:`materials.stress_strain_curve`, so what you see is exactly the
fiber law. ``MaterialManagerDialog`` is the list/add/edit/delete manager.

Rebuilt onto the shared card scaffold (:mod:`analysis_ui`) for the GUI-polish
work stream (plan gui-polish D1); the `.data()` / `.edit()` / `.manage()`
contracts and every test-referenced attribute are unchanged. Pure Qt +
matplotlib (Agg canvas) — headless-constructible under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

import copy

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog,
                               QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

import materials as M
import style
from analysis_ui import GroupCard, dialog_buttons
from project import Material

# param key -> (label, unit, factor): the spin shows value/factor, reads value*factor
PARAM_SPECS: dict[str, tuple[str, str, float]] = {
    "E": ("E", "GPa", 1.0e9),
    "nu": ("nu", "", 1.0),
    "fc": ("f'c", "MPa", 1.0e6),
    "eps_c0": ("eps_c0", "‰", 1.0e-3),
    "eps_cu": ("eps_cu", "‰", 1.0e-3),
    "Ec": ("Ec", "GPa", 1.0e9),
    "fpcu_ratio": ("fpcu / f'c", "", 1.0),
    "fy": ("fy", "MPa", 1.0e6),
    "fu": ("fu", "MPa", 1.0e6),
    "eps_sh": ("eps_sh", "‰", 1.0e-3),
    "eps_su": ("eps_su", "‰", 1.0e-3),
}


def _param_spin(key: str, value: float) -> QDoubleSpinBox:
    _lbl, _unit, factor = PARAM_SPECS.get(key, (key, "", 1.0))
    spin = QDoubleSpinBox()
    spin.setRange(0.0 if key != "nu" else -0.99, 1.0e6)
    spin.setDecimals(4)
    spin.setValue(value / factor)
    spin.setSingleStep(0.1 if factor != 1.0 else 0.01)
    return spin


def _next_id(ids) -> int:
    return (max(ids) + 1) if ids else 1


class MaterialDialog(QDialog):
    def __init__(self, parent, project, material=None):
        super().__init__(parent)
        self.setWindowTitle("Edit material" if material else "Add material")
        self._project = project
        outer = QHBoxLayout(self)
        outer.setContentsMargins(style.SP_LG, style.SP_LG, style.SP_LG,
                                 style.SP_LG)
        outer.setSpacing(style.SP_MD)

        # ---- left: grouped input cards ----
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(style.SP_MD)

        ident = GroupCard("Identity")
        self.id_spin = QSpinBox()
        self.id_spin.setRange(1, 10_000_000)
        self.id_spin.setValue(material.id if material
                              else _next_id([m.id for m in project.materials]))
        self.id_spin.setEnabled(material is None)
        ident.add_row("Material id", self.id_spin)
        self.name = QLineEdit(material.name if material else "New material")
        self.name.textChanged.connect(self._redraw)
        ident.add_row("Name", self.name)
        self.kind = QComboBox()
        for key, label in M.MATERIAL_KINDS.items():
            self.kind.addItem(label, key)
        if material:
            i = self.kind.findData(material.kind)
            if i >= 0:
                self.kind.setCurrentIndex(i)
        ident.add_row("Type", self.kind)
        lv.addWidget(ident)

        # dynamic parameter rows live on the Parameters card's form; we clear +
        # rebuild them on a kind change.
        self._param_card = GroupCard("Parameters")
        self._param_form = self._param_card.body_layout()
        self._spins: dict[str, QDoubleSpinBox] = {}
        self._seed = material          # source values while (re)building rows
        self._build_param_rows()
        lv.addWidget(self._param_card)

        hint = QLabel("The σ-ε curve is the actual fiber law the engine will use.")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        lv.addWidget(hint)
        lv.addStretch(1)
        lv.addWidget(dialog_buttons(self))
        outer.addWidget(left, 0)

        # ---- right: σ-ε preview ----
        self._fig = Figure(figsize=(3.8, 3.0), layout="constrained")
        self._ax = self._fig.add_subplot(111)
        self._canvas = Canvas(self._fig)
        self._canvas.setMinimumWidth(360)
        outer.addWidget(self._canvas, 1)

        self.kind.currentIndexChanged.connect(self._on_kind_changed)
        style.apply(self)
        self._redraw()

    # ------------------------------------------------ dynamic parameter rows
    def _current_kind(self) -> str:
        return self.kind.currentData()

    def _param_keys(self, kind: str) -> list[str]:
        if kind == "elastic_isotropic":
            return ["E", "nu"]
        return list(M.KIND_DEFAULTS.get(kind, {}).keys())

    def _build_param_rows(self) -> None:
        while self._param_form.rowCount():
            self._param_form.removeRow(0)
        self._spins.clear()
        kind = self._current_kind()
        # seed values: existing material of this kind, else kind defaults
        seed = dict(M.KIND_DEFAULTS.get(kind, {}))
        if kind == "elastic_isotropic":
            seed = {"E": 200.0e9, "nu": 0.3}
        if self._seed is not None and self._seed.kind == kind:
            if kind == "elastic_isotropic":
                seed = {"E": self._seed.E, "nu": self._seed.nu}
            else:
                seed = {**seed, **(self._seed.params or {})}
        for key in self._param_keys(kind):
            lbl, unit, _f = PARAM_SPECS.get(key, (key, "", 1.0))
            spin = _param_spin(key, float(seed.get(key, 0.0)))
            spin.valueChanged.connect(self._redraw)
            self._spins[key] = spin
            self._param_form.addRow(f"{lbl} [{unit}]" if unit else lbl, spin)

    def _on_kind_changed(self) -> None:
        self._seed = None            # switching kind -> use fresh defaults
        self._build_param_rows()
        self._redraw()

    # ------------------------------------------------ read + preview
    def _read_params(self) -> dict:
        out = {}
        for key, spin in self._spins.items():
            _lbl, _unit, factor = PARAM_SPECS.get(key, (key, "", 1.0))
            out[key] = spin.value() * factor
        return out

    def data(self) -> Material:
        kind = self._current_kind()
        p = self._read_params()
        if kind == "elastic_isotropic":
            return Material(id=self.id_spin.value(), name=self.name.text(),
                            E=float(p.get("E", 200e9)),
                            nu=float(p.get("nu", 0.3)), kind=kind, params={})
        # representative modulus for the linear frame stiffness fallback
        if kind == "concrete_kentpark":
            E_rep = 2.0 * p["fc"] / p["eps_c0"]
        elif kind == "concrete_mander":
            E_rep = p.get("Ec", 25e9)
        else:                                    # steel
            E_rep = p.get("E", 200e9)
        params = {k: v for k, v in p.items() if k not in ("E", "nu")}
        mat = Material(id=self.id_spin.value(), name=self.name.text(),
                       E=float(E_rep), nu=0.2, kind=kind, params=params)
        if kind in ("reinforcing_steel", "cyclic_steel"):
            mat.fy, mat.fu = float(p["fy"]), float(p["fu"])
            mat.params["E"] = float(p.get("E", 200e9))
        return mat

    def _redraw(self) -> None:
        self._ax.clear()
        try:
            eps, sig = M.stress_strain_curve(self.data(), n=200)
            self._ax.plot(eps * 1.0e3, sig / 1.0e6, "-", lw=1.8,
                          color=style.C_PRIMARY)
            self._ax.axhline(0, color=style.AX_SPINE, lw=0.6)
            self._ax.axvline(0, color=style.AX_SPINE, lw=0.6)
            self._ax.set_xlabel("strain (‰)")
            self._ax.set_ylabel("stress (MPa)")
            self._ax.set_title(self.name.text())
            style.beautify_axes(self._ax)
        except Exception as exc:                       # noqa: BLE001
            self._ax.text(0.5, 0.5, f"({exc})", ha="center", va="center",
                          transform=self._ax.transAxes, fontsize=8,
                          color=style.MUTED)
        self._canvas.draw_idle()

    @classmethod
    def edit(cls, parent, project, material=None):
        dlg = cls(parent, project, material)
        return dlg.data() if dlg.exec() else None


class MaterialManagerDialog(QDialog):
    """List / add / edit / delete the project's materials. Returns the edited
    material list via :meth:`result_materials` after a successful close."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Materials")
        self._project = project
        self._materials = copy.deepcopy(project.materials)
        v = QVBoxLayout(self)
        v.setContentsMargins(style.SP_LG, style.SP_LG, style.SP_LG, style.SP_LG)
        v.setSpacing(style.SP_SM)

        head = QLabel("Material library")
        head.setObjectName("h2")
        v.addWidget(head)
        sub = QLabel("Elastic and inelastic (fiber) materials used by members.")
        sub.setObjectName("sub")
        v.addWidget(sub)

        body = QHBoxLayout()
        body.setSpacing(style.SP_MD)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["id", "name", "type"])
        self.table.setMinimumSize(440, 240)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.doubleClicked.connect(lambda *_: self._edit())
        body.addWidget(self.table, 1)

        col = QVBoxLayout()
        col.setSpacing(style.SP_SM)
        for label, cb in (("Add…", self._add), ("Edit…", self._edit),
                          ("Delete", self._delete)):
            b = QPushButton(label)
            b.clicked.connect(cb)
            col.addWidget(b)
        col.addStretch(1)
        body.addLayout(col)
        v.addLayout(body)

        v.addWidget(dialog_buttons(self))
        style.apply(self)
        self._refresh()

    def _refresh(self) -> None:
        self.table.setRowCount(len(self._materials))
        for r, m in enumerate(self._materials):
            self.table.setItem(r, 0, QTableWidgetItem(str(m.id)))
            self.table.setItem(r, 1, QTableWidgetItem(m.name))
            self.table.setItem(r, 2, QTableWidgetItem(
                M.MATERIAL_KINDS.get(m.kind, m.kind)))

    def _selected_row(self) -> int:
        return self.table.currentRow()

    def _proxy_project(self):
        # MaterialDialog only reads .materials (for id defaults) — hand it ours
        proxy = copy.copy(self._project)
        proxy.materials = self._materials
        return proxy

    def _add(self) -> None:
        mat = MaterialDialog.edit(self, self._proxy_project())
        if mat is None:
            return
        if any(m.id == mat.id for m in self._materials):
            QMessageBox.warning(self, "Duplicate",
                                f"Material {mat.id} already exists.")
            return
        self._materials.append(mat)
        self._refresh()

    def _edit(self) -> None:
        r = self._selected_row()
        if r < 0:
            return
        mat = MaterialDialog.edit(self, self._proxy_project(), self._materials[r])
        if mat is not None:
            self._materials[r] = mat
            self._refresh()

    def _delete(self) -> None:
        r = self._selected_row()
        if r < 0:
            return
        mid = self._materials[r].id
        used = [mb.id for mb in self._project.members if mb.material == mid]
        if used:
            QMessageBox.warning(self, "In use",
                                f"Material {mid} is used by member(s) "
                                f"{', '.join(map(str, used))}.")
            return
        del self._materials[r]
        self._refresh()

    def result_materials(self) -> list:
        return self._materials

    @classmethod
    def manage(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.result_materials() if dlg.exec() else None
