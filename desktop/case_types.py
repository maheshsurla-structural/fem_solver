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
    def build_config(self, project, params: dict, *,           # noqa: ARG002
                     initial_condition=("zero",)):
        """Rebuild the runtime config the runner consumes. Default: identity.

        ``initial_condition`` (E2) is the case's stiffness-to-use setting
        (``("zero",)`` or ``("state", nl_case_id)``). Most types ignore it (they
        run against the current model); a type that can start from a nonlinear
        case's committed state threads it into its config here."""
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
        # E1: Modal is edited in the unified Load-Case-Data editor (Type ▾).
        from case_editor import CaseEditorDialog
        return CaseEditorDialog.edit(parent, project, self.type_id, case)

    def build_config(self, project, params, *, initial_condition=("zero",)):
        return (int(params.get("num_modes", 6)), bool(params.get("lumped")),
                initial_condition)

    def dispatch(self, win, config):
        num_modes, lumped, ic = config
        return win.run_modal(num_modes=num_modes, lumped=lumped,
                             initial_condition=ic)


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
        # E1: Buckling is edited in the unified Load-Case-Data editor (Type ▾).
        from case_editor import CaseEditorDialog
        return CaseEditorDialog.edit(parent, project, self.type_id, case)

    def build_config(self, project, params, *, initial_condition=("zero",)):
        sel = tuple(params.get("selection") or ("all", None))
        return (sel, int(params.get("num_modes", 4)),
                int(params.get("subdivisions", 6)), initial_condition)

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
        from case_editor import CaseEditorDialog        # E1 unified editor
        return CaseEditorDialog.edit(parent, project, self.type_id, case)

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
        from case_editor import CaseEditorDialog        # E1 unified editor
        return CaseEditorDialog.edit(parent, project, self.type_id, case)

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
        from case_editor import CaseEditorDialog        # E1 unified editor
        return CaseEditorDialog.edit(parent, project, self.type_id, case)

    def dispatch(self, win, config):
        return win.run_load_rating(config=config)


class ResponseSpectrumType(CaseType):
    """Response-spectrum (modal superposition) → run_response_spectrum. The
    *object* case: the dialog's result is a derived ``ResponseSpectrum``, so the
    saved params are the spectrum **inputs** and ``build_config`` rebuilds the
    object headlessly via ``response_spectrum_dialog.spectrum_from_params``."""

    type_id = "responsespectrum"
    type_label = "Response Spectrum"

    def detail(self, project, params):
        src = {"asce7": "ASCE 7", "ec8": "EC8", "is1893": "IS 1893",
               "custom": "custom"}.get(params.get("source"),
                                       params.get("source", "?"))
        return (f"{src} · {int(params.get('num_modes', 6))} modes · "
                f"{str(params.get('direction', 'x')).upper()} · "
                f"{str(params.get('combination', 'cqc')).upper()}")

    def edit(self, parent, project, case=None):
        # E1: Response Spectrum is edited in the unified Load-Case-Data editor.
        from case_editor import CaseEditorDialog
        return CaseEditorDialog.edit(parent, project, self.type_id, case)

    def build_config(self, project, params, *, initial_condition=("zero",)):
        from response_spectrum_dialog import spectrum_from_params
        return (spectrum_from_params(params), int(params.get("num_modes", 6)),
                params.get("direction", "x"),
                params.get("combination", "cqc"), initial_condition)

    def dispatch(self, win, config):
        return win.run_response_spectrum(config=config)


class VehicleDynamicsType(CaseType):
    """Vehicle dynamics (moving-load time-history) → run_vehicle_dynamics."""

    type_id = "vehicledynamics"
    type_label = "Vehicle Dynamics"

    def detail(self, project, params):
        kind = "sprung-mass" if params.get("kind") == "vbi" else "moving force"
        spd = float(params.get("speed", 16.667)) * 3.6
        tail = (f" · {params.get('vehicle', '')}"
                if params.get("kind") != "vbi" else "")
        return f"{kind}{tail} · {spd:.0f} km/h @ node {params.get('node')}"

    def edit(self, parent, project, case=None):
        from case_editor import CaseEditorDialog        # E1 unified editor
        return CaseEditorDialog.edit(parent, project, self.type_id, case)

    def dispatch(self, win, config):
        return win.run_vehicle_dynamics(config=config)


class InfluenceSurfaceType(CaseType):
    """Influence surface / multi-lane (3-D deck) → run_influence_surface."""

    type_id = "influencesurface"
    type_label = "Influence Surface"

    def detail(self, project, params):
        resp = params.get("response") or ("disp", None)
        what = {"disp": "disp", "reaction": "reaction"}.get(resp[0], resp[0])
        mp = " · MP" if params.get("multi_presence", True) else ""
        return (f"{what} @ node {resp[1]} · {params.get('vehicle', '')} · "
                f"{len(params.get('deck') or [])} deck nodes{mp}")

    def edit(self, parent, project, case=None):
        from case_editor import CaseEditorDialog        # E1 unified editor
        return CaseEditorDialog.edit(parent, project, self.type_id, case)

    def dispatch(self, win, config):
        return win.run_influence_surface(config=config)


class CableTuningType(CaseType):
    """Cable-stayed tuning (unknown load factor) → run_cable_tuning."""

    type_id = "cabletuning"
    type_label = "Cable Tuning"

    def detail(self, project, params):
        return (f"{len(params.get('cables') or [])} stays · "
                f"{len(params.get('targets') or [])} target nodes")

    def edit(self, parent, project, case=None):
        from case_editor import CaseEditorDialog        # E1 unified editor
        return CaseEditorDialog.edit(parent, project, self.type_id, case)

    def dispatch(self, win, config):
        return win.run_cable_tuning(config=config)


class TimeHistoryType(CaseType):
    """Nonlinear time history → the interactive :meth:`run_timehistory_dialog`.

    References a :class:`project.TimeHistoryFunction` by id (the record lives in
    the project's function library, TH-1); ``build_config`` resolves it into the
    runner's seed. The run is interactive (fiber solve + response plot), so
    ``dispatch`` opens the runner *seeded* rather than running headlessly."""

    type_id = "timehistory"
    type_label = "Time History"
    icon = "function"

    def default_params(self, project):
        fid = project.th_functions[0].id if project.th_functions else None
        return {"function_id": fid, "control_node": None, "direction": "y",
                "scale": 1.0, "zeta": 0.05, "density": 2400.0}

    def detail(self, project, params):
        f = project.th_function(params.get("function_id"))
        fname = f.name if f else "(function deleted)"
        return (f"{fname} · node {params.get('control_node')} "
                f"{str(params.get('direction', 'y')).upper()} · "
                f"×{params.get('scale', 1.0):g}")

    def edit(self, parent, project, case=None):
        # E1: Time History is edited in the unified Load-Case-Data editor; its
        # body is IC-capable and the shell injects hold_source_loads.
        from case_editor import CaseEditorDialog
        return CaseEditorDialog.edit(parent, project, self.type_id, case)

    def build_config(self, project, params, *, initial_condition=("zero",)):
        f = project.th_function(params.get("function_id"))
        if f is None:
            raise ValueError("the referenced time-history function was deleted "
                             "— pick another in the case's Modify dialog")
        return {"values": list(f.values), "dt": f.dt, "in_g": f.in_g,
                "name": f.name, "control_node": params.get("control_node"),
                "direction": params.get("direction", "y"),
                "scale": float(params.get("scale", 1.0)),
                "zeta": float(params.get("zeta", 0.05)),
                "density": float(params.get("density", 2400.0)),
                "initial_condition": initial_condition,
                "hold_source_loads": bool(
                    params.get("hold_source_loads", False))}

    def dispatch(self, win, config):
        return win.run_timehistory_dialog(seed=config)


# Ordered registry — also the order of the Add ▾ menu.
_ORDER: list[CaseType] = [ModalType(), BucklingType(), MovingLoadType(),
                          TemperatureGradientType(), LoadRatingType(),
                          ResponseSpectrumType(), VehicleDynamicsType(),
                          InfluenceSurfaceType(), CableTuningType(),
                          TimeHistoryType()]
TYPES: dict[str, CaseType] = {ct.type_id: ct for ct in _ORDER}


def get(type_id: str) -> CaseType | None:
    return TYPES.get(type_id)
