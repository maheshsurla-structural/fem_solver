"""Registry of saveable analysis-case types (analysis-cases-manager plan).

Each built-in analysis type (modal, buckling, response spectrum, moving load,
…) is described by a :class:`CaseType` adapter so the unified Analysis-cases
home (:class:`analysis_cases_dialog.AnalysisCasesDialog`) can treat it as a
persistent, named, multi-instance :class:`project.AnalysisCase`: list it, add /
modify it (seeding the type's setup dialog from the saved ``params``), and run
it from those saved params — with **no re-prompt**.

An adapter maps a case's JSON-friendly ``params`` both ways:

* :meth:`CaseType.edit` opens the type's setup dialog seeded from a case and
  returns the (re)named :class:`project.AnalysisCase`, or ``None`` if cancelled.
* :meth:`CaseType.build_config` rebuilds, headlessly, the runtime ``config`` the
  matching ``MainWindow.run_*`` consumes. For most types the params *are* the
  config; response-spectrum (a later increment) rebuilds its derived
  ``ResponseSpectrum`` object here.
* :meth:`CaseType.dispatch` calls that ``run_*`` with the config.
* :meth:`CaseType.detail` / :meth:`CaseType.default_params` feed the list row and
  a fresh Add.

Pure-Python + Qt dialogs; headless-constructible under
``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

from project import AnalysisCase


class CaseType:
    """Base adapter — subclass per analysis type, register in :data:`_ORDER`."""

    type_id: str = ""
    type_label: str = ""
    icon: str = "run"

    # -------------------------------------------------------------- list row
    def detail(self, project, params: dict) -> str:            # noqa: ARG002
        """One-line summary for the Details column."""
        return ""

    def default_params(self, project) -> dict:                 # noqa: ARG002
        """Params for a fresh Add (before the user touches the dialog)."""
        return {}

    # -------------------------------------------------------- add / modify
    def edit(self, parent, project, case: AnalysisCase | None = None):
        """Open the setup dialog (seeded from ``case``) and return the edited
        :class:`AnalysisCase`, or ``None`` if cancelled. A *new* case carries
        ``id = 0`` — the caller assigns a fresh id."""
        raise NotImplementedError

    # ---------------------------------------------------------------- run
    def build_config(self, project, params: dict):             # noqa: ARG002
        """Rebuild the runtime config the runner consumes. Default: identity."""
        return params

    def dispatch(self, win, config):
        """Invoke the matching ``MainWindow.run_*`` with ``config``."""
        raise NotImplementedError

    # ------------------------------------------------------------- helper
    def _mk(self, case, *, name, params, notes):
        """Build an :class:`AnalysisCase` of this type, preserving the id of an
        edited case (``0`` for a new one)."""
        return AnalysisCase(id=(case.id if case else 0), name=name,
                            type=self.type_id, params=params, notes=notes)

    def _edit_cap(self, project) -> int:
        """A cheap upper bound on the mode count for the editor's spinbox — the
        runner re-clamps to the true ``neq`` at run time."""
        return max(1, int(project.ndf) * max(1, len(project.nodes)))


class ModalType(CaseType):
    """Free-vibration modal analysis → :meth:`MainWindow.run_modal`."""

    type_id = "modal"
    type_label = "Modal"
    icon = "undeformed"

    def detail(self, project, params):
        mass = "lumped" if params.get("lumped") else "consistent"
        return f"{int(params.get('num_modes', 6))} modes · {mass} mass"

    def default_params(self, project):
        return {"num_modes": 6, "lumped": False}

    def edit(self, parent, project, case=None):
        from modal_dialog import ModalDialog
        params = dict(case.params) if case else self.default_params(project)
        dlg = ModalDialog(parent, max_modes=self._edit_cap(project),
                          default_modes=int(params.get("num_modes", 6)),
                          initial=params,
                          name=(case.name if case else "Modal"),
                          notes=(case.notes if case else ""))
        if not dlg.exec():
            return None
        num_modes, lumped = dlg.result()
        return self._mk(case, name=dlg.header.name() or "Modal",
                        params={"num_modes": int(num_modes),
                                "lumped": bool(lumped)},
                        notes=dlg.header.notes())

    def build_config(self, project, params):
        return (int(params.get("num_modes", 6)), bool(params.get("lumped")))

    def dispatch(self, win, config):
        num_modes, lumped = config
        return win.run_modal(num_modes=num_modes, lumped=lumped)


class BucklingType(CaseType):
    """Linear (eigenvalue) buckling → :meth:`MainWindow.run_buckling`."""

    type_id = "buckling"
    type_label = "Buckling"
    icon = "run"

    def detail(self, project, params):
        sel = params.get("selection") or ["all", None]
        ref = {"all": "all load cases",
               "case": f"case {sel[1]}",
               "combination": f"combo {sel[1]}"}.get(sel[0], "all load cases")
        return (f"{int(params.get('num_modes', 4))} modes · {ref} · "
                f"{int(params.get('subdivisions', 6))} subdiv/member")

    def default_params(self, project):
        return {"selection": ["all", None], "num_modes": 4, "subdivisions": 6}

    def edit(self, parent, project, case=None):
        from buckling_dialog import BucklingDialog
        params = dict(case.params) if case else self.default_params(project)
        dlg = BucklingDialog(parent, project, max_modes=self._edit_cap(project),
                             default_modes=int(params.get("num_modes", 4)),
                             initial=params,
                             name=(case.name if case else "Buckling"),
                             notes=(case.notes if case else ""))
        if not dlg.exec():
            return None
        selection, num_modes, subdivisions = dlg.result()
        return self._mk(case, name=dlg.header.name() or "Buckling",
                        params={"selection": list(selection),
                                "num_modes": int(num_modes),
                                "subdivisions": int(subdivisions)},
                        notes=dlg.header.notes())

    def build_config(self, project, params):
        sel = tuple(params.get("selection") or ("all", None))
        return (sel, int(params.get("num_modes", 4)),
                int(params.get("subdivisions", 6)))

    def dispatch(self, win, config):
        return win.run_buckling(config=config)


class MovingLoadType(CaseType):
    """Moving-load / influence-line → :meth:`MainWindow.run_moving_load`. The
    dialog's config dict *is* the params (identity ``build_config``)."""

    type_id = "movingload"
    type_label = "Moving Load"

    def detail(self, project, params):
        veh = params.get("vehicle", "hl93")
        resp = params.get("response") or ("M", None, "i")
        what = {"M": f"moment @ member {resp[1]}",
                "V": f"shear @ member {resp[1]}",
                "disp": f"disp @ node {resp[1]}",
                "reaction": f"reaction @ node {resp[1]}"}.get(resp[0], resp[0])
        return f"{veh} · {what} · {len(params.get('lane') or [])}-node lane"

    def edit(self, parent, project, case=None):
        from moving_load_dialog import MovingLoadDialog
        dlg = MovingLoadDialog(parent, project,
                               initial=(dict(case.params) if case else None),
                               name=(case.name if case else "Moving Load"),
                               notes=(case.notes if case else ""))
        if not dlg.exec():
            return None
        return self._mk(case, name=dlg.header.name() or "Moving Load",
                        params=dlg.result(), notes=dlg.header.notes())

    def dispatch(self, win, config):
        return win.run_moving_load(config=config)


class TemperatureGradientType(CaseType):
    """Temperature gradient → :meth:`MainWindow.run_temperature_gradient`."""

    type_id = "tempgradient"
    type_label = "Temperature Gradient"

    def detail(self, project, params):
        n = len(params.get("members") or [])
        if params.get("source") == "linear":
            grad = f"linear {params.get('dt_top', 0):g}→{params.get('dt_bot', 0):g}°C"
        else:
            grad = f"AASHTO zone {params.get('zone', '?')}"
        return f"{grad} · {n} members"

    def edit(self, parent, project, case=None):
        from temperature_gradient_dialog import TemperatureGradientDialog
        dlg = TemperatureGradientDialog(
            parent, project,
            initial=(dict(case.params) if case else None),
            name=(case.name if case else "Temperature Gradient"),
            notes=(case.notes if case else ""))
        if not dlg.exec():
            return None
        return self._mk(case, name=dlg.header.name() or "Temperature Gradient",
                        params=dlg.result(), notes=dlg.header.notes())

    def dispatch(self, win, config):
        return win.run_temperature_gradient(config=config)


class LoadRatingType(CaseType):
    """AASHTO LRFR load rating → :meth:`MainWindow.run_load_rating`. Capacity /
    dead loads are persisted in SI (the dialog converts to/from display)."""

    type_id = "loadrating"
    type_label = "Load Rating"

    def detail(self, project, params):
        resp = params.get("response") or ("M", None, "i")
        kind = {"M": "moment", "V": "shear"}.get(resp[0], resp[0])
        levels = ["inv/op"]
        if params.get("adtt") is not None:
            levels.append("legal")
        if params.get("permit_gamma_LL") is not None:
            levels.append("permit")
        return f"{kind} @ member {resp[1]} ({resp[2]}) · {'+'.join(levels)}"

    def edit(self, parent, project, case=None):
        from load_rating_dialog import LoadRatingDialog
        dlg = LoadRatingDialog(parent, project,
                               initial=(dict(case.params) if case else None),
                               name=(case.name if case else "Load Rating"),
                               notes=(case.notes if case else ""))
        if not dlg.exec():
            return None
        return self._mk(case, name=dlg.header.name() or "Load Rating",
                        params=dlg.result(), notes=dlg.header.notes())

    def dispatch(self, win, config):
        return win.run_load_rating(config=config)


# Ordered registry — also the order of the Add ▾ menu. Extend as types migrate
# (response spectrum, vehicle dynamics, influence surface, cable tuning, …).
_ORDER: list[CaseType] = [ModalType(), BucklingType(), MovingLoadType(),
                          TemperatureGradientType(), LoadRatingType()]
TYPES: dict[str, CaseType] = {ct.type_id: ct for ct in _ORDER}


def get(type_id: str) -> CaseType | None:
    return TYPES.get(type_id)
