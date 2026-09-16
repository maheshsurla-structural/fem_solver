"""In-app model checks & diagnostics (plan §16 G-S5).

A commercial solver tells you *before* you run that the model is sound, and
*after* a failed run what to try next. This is the pure, Qt-free core of that:

* :func:`check_project` — static sanity checks over a :class:`project.Project`:
  topology (missing/duplicate ids, zero-length members), restraint (unsupported
  ⇒ mechanism), **unit-scale plausibility** (an ``E`` or ``f'c`` that looks like
  it was typed in MPa/GPa instead of Pa is the classic silent blunder),
  section/fiber readiness, and nonlinear-case wiring.
* :func:`convergence_advice` — actionable guidance when a nonlinear run fails to
  converge or stops short.

Each finding is a :class:`Check` (level + message + fix hint). Levels are
``"error"`` (won't run / wrong answer), ``"warning"`` (suspicious), ``"info"``
(capability note). The GUI (`model_checks_dialog`, the pushover dialog) renders
them; this module stays testable headless.
"""
from __future__ import annotations

from dataclasses import dataclass

LEVELS = ("error", "warning", "info")

# Plausible SI ranges (Pa) — outside these, the value was almost certainly typed
# in the wrong unit (the #1 silent modelling error).
_E_MIN, _E_MAX = 1.0e8, 1.0e13          # elastic modulus: ~1 GPa .. ~10 TPa
_FC_MIN = 1.0e6                          # concrete f'c: below 1 MPa ⇒ likely MPa
_FY_MIN = 1.0e6                          # steel f_y: below 1 MPa ⇒ likely MPa


@dataclass(frozen=True)
class Check:
    level: str            # "error" | "warning" | "info"
    message: str          # what is wrong
    hint: str = ""        # how to fix it


def _n(project) -> int:
    return int(getattr(project, "ndf", 3))


def check_project(project) -> list[Check]:
    """Return all findings for ``project``, most-severe first."""
    out: list[Check] = []
    out += _check_topology(project)
    out += _check_restraint(project)
    out += _check_units(project)
    out += _check_sections(project)
    out += _check_nonlinear_cases(project)
    order = {lvl: i for i, lvl in enumerate(LEVELS)}
    out.sort(key=lambda c: order.get(c.level, 9))
    return out


# --------------------------------------------------------------- topology
def _check_topology(project) -> list[Check]:
    out: list[Check] = []
    nodes = list(project.nodes)
    members = list(project.members)
    if not nodes:
        out.append(Check("error", "The model has no nodes.",
                         "Add nodes before analysing."))
    if not members:
        out.append(Check("error", "The model has no members.",
                         "Add at least one member."))
    ids = [n.id for n in nodes]
    dupes = {i for i in ids if ids.count(i) > 1}
    for i in sorted(dupes):
        out.append(Check("error", f"Duplicate node id {i}.",
                         "Node ids must be unique."))
    node_ids = set(ids)
    sec_ids = {s.id for s in project.sections}
    mat_ids = {m.id for m in project.materials}
    coords = {n.id: n for n in nodes}
    for mb in members:
        for end in (mb.n1, mb.n2):
            if end not in node_ids:
                out.append(Check("error",
                                 f"Member {mb.id} references missing node {end}.",
                                 "Fix the member's end nodes."))
        if mb.section not in sec_ids:
            out.append(Check("error",
                             f"Member {mb.id} references missing section "
                             f"{mb.section}.", "Assign an existing section."))
        if mb.material not in mat_ids:
            out.append(Check("error",
                             f"Member {mb.id} references missing material "
                             f"{mb.material}.", "Assign an existing material."))
        a, b = coords.get(mb.n1), coords.get(mb.n2)
        if a is not None and b is not None:
            d2 = (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2
            if d2 <= 1.0e-18:
                out.append(Check("error",
                                 f"Member {mb.id} has zero length "
                                 f"(nodes {mb.n1} and {mb.n2} coincide).",
                                 "Move one end node."))
    return out


# --------------------------------------------------------------- restraint
def _check_restraint(project) -> list[Check]:
    if not project.nodes:
        return []
    total_fix = sum(sum(1 for f in (n.supports or ()) if f)
                    for n in project.nodes)
    if total_fix == 0:
        return [Check("error",
                      "No supports — the structure is unrestrained "
                      "(rigid-body mechanism).",
                      "Fix enough DOFs to remove all rigid-body motion.")]
    need = 3 if project.ndm == 2 else 6
    if total_fix < need:
        return [Check("warning",
                      f"Only {total_fix} DOF(s) restrained; a stable "
                      f"{project.ndm}-D model needs at least {need}.",
                      "Check the structure can't move or spin as a rigid body.")]
    return []


# --------------------------------------------------------------- units
def _flag_modulus(name, val, out):
    if val <= 0:
        out.append(Check("warning", f"{name} = {val:g} is not positive.",
                         "Elastic modulus must be > 0 (Pa)."))
    elif val < _E_MIN:
        out.append(Check("warning",
                         f"{name} = {val:g} Pa looks too small — did you enter "
                         f"MPa or GPa instead of Pa?",
                         "The project stores SI base units: steel E ≈ 2.0e11 Pa."))
    elif val > _E_MAX:
        out.append(Check("warning", f"{name} = {val:g} Pa looks implausibly "
                         "large.", "Expected ~1e9–1e12 Pa."))


def _check_units(project) -> list[Check]:
    out: list[Check] = []
    for m in project.materials:
        _flag_modulus(f"Material '{m.name}' E", float(getattr(m, "E", 0.0)), out)
        p = getattr(m, "params", None) or {}
        for key in ("E", "Ec", "Es"):
            if key in p and p[key]:
                _flag_modulus(f"Material '{m.name}' {key}", float(p[key]), out)
        fc = p.get("fc")
        if fc and 0 < float(fc) < _FC_MIN:
            out.append(Check("warning",
                             f"Material '{m.name}' f'c = {float(fc):g} Pa is "
                             "below 1 MPa — did you enter MPa instead of Pa?",
                             "e.g. 35 MPa concrete is f'c = 35e6 Pa."))
        for key in ("fy", "fu"):
            v = p.get(key)
            if v and 0 < float(v) < _FY_MIN:
                out.append(Check("warning",
                                 f"Material '{m.name}' {key} = {float(v):g} Pa "
                                 "is below 1 MPa — MPa instead of Pa?",
                                 "e.g. 500 MPa steel is f_y = 500e6 Pa."))
        for key in ("eps_cu", "eps_su", "eps_c0", "eps_sh"):
            v = p.get(key)
            if v is not None and float(v) >= 1.0:
                out.append(Check("warning",
                                 f"Material '{m.name}' {key} = {float(v):g} is "
                                 "≥ 1 — strains are dimensionless (0.003, not 3).",
                                 "Enter strain as a fraction, not ‰ or %."))
    return out


# --------------------------------------------------------------- sections
def _check_sections(project) -> list[Check]:
    out: list[Check] = []
    has_fiber = any(getattr(s, "gsd_spec", None) for s in project.sections)
    if not has_fiber:
        out.append(Check("info",
                         "No fiber (Section Designer) section in the model — "
                         "nonlinear pushover / time-history is unavailable.",
                         "Author a Section Designer section and assign it."))
    for s in project.sections:
        spec = getattr(s, "gsd_spec", None)
        if not spec:
            continue
        try:
            import section_gui_core as core
            sp = core.Spec(**{k: v for k, v in spec.items()
                              if k in core.Spec.__dataclass_fields__})
            core.build_case(sp)                       # can it build at all?
            conf, _ = core._section_confinement(sp, {})
            if (conf is not None and sp.kind in ("Rectangular", "Circular")
                    and core.confined_core_polygon(sp) is None):
                out.append(Check("warning",
                                 f"Section '{s.name}': the cover leaves no "
                                 "distinct core — confinement won't apply.",
                                 "Reduce the cover or increase the section."))
        except Exception as exc:                       # noqa: BLE001
            out.append(Check("error",
                             f"Section '{s.name}' can't build: {exc}",
                             "Check its geometry / cover / reinforcement."))
    return out


# --------------------------------------------------------------- NL cases
def _check_nonlinear_cases(project) -> list[Check]:
    out: list[Check] = []
    node_ids = {n.id for n in project.nodes}
    case_ids = {c.id for c in project.nonlinear_cases}
    ndf = _n(project)
    for c in project.nonlinear_cases:
        tag = f"Nonlinear case '{c.name}'"
        if c.control_node not in node_ids:
            out.append(Check("error", f"{tag}: control node {c.control_node} "
                             "doesn't exist.", "Pick an existing node."))
        if not (0 <= c.control_dof < ndf):
            out.append(Check("error", f"{tag}: control DOF {c.control_dof} is "
                             f"out of range 0..{ndf - 1}.", "Pick a valid DOF."))
        if not c.target:
            out.append(Check("warning", f"{tag}: target displacement is 0.",
                             "Set a non-zero target."))
        if c.n_steps <= 0:
            out.append(Check("warning", f"{tag}: {c.n_steps} steps.",
                             "Use a positive number of steps."))
        if c.axial and c.axial_node is not None and c.axial_node not in node_ids:
            out.append(Check("error", f"{tag}: axial node {c.axial_node} "
                             "doesn't exist.", "Pick an existing node."))
        if c.continue_from is not None and c.continue_from not in case_ids:
            out.append(Check("error", f"{tag}: continues from missing case "
                             f"{c.continue_from}.", "Fix the staged sequence."))
    # E2: an initial condition ("state", id) must name an existing nonlinear
    # case — on nonlinear *and* saved analysis cases (a dangling reference is
    # left by deleting the source, or by importing a partial project).
    both = list(project.nonlinear_cases) + list(
        getattr(project, "analysis_cases", []))
    for c in both:
        ic = tuple(getattr(c, "initial_condition", ("zero",)) or ("zero",))
        if len(ic) >= 2 and ic[0] == "state" and ic[1] not in case_ids:
            out.append(Check("error", f"Case '{c.name}': initial condition "
                             f"references missing nonlinear case {ic[1]}.",
                             "Pick an existing source case, or start unstressed."))
    return out


# --------------------------------------------------------------- advice
def convergence_advice(*, error=None, got_steps: int = 0,
                       requested_steps: int = 0) -> list[str]:
    """Actionable next steps after a nonlinear run failed or stopped short."""
    tips: list[str] = []
    if error is not None:
        e = str(error).lower()
        if "singular" in e or "mechanism" in e or "snap" in e:
            tips.append("Singular tangent — check the structure is fully "
                        "supported (no mechanism), and that a softening section "
                        "hasn't lost all stiffness at this drift.")
        tips.append("Reduce the target displacement or increase the number of "
                    "steps so each increment is smaller.")
        tips.append("Loosen the convergence tolerance — base shear is in "
                    "newtons, so 1e-3–1.0 is typical (not 1e-8).")
        tips.append("Increase the maximum iterations per step.")
    elif requested_steps and got_steps < requested_steps:
        tips.append(f"The run stopped at step {got_steps} of {requested_steps} "
                    "— the section most likely reached its capacity (concrete "
                    "crushing / bar fracture) or the tangent went singular.")
        tips.append("If you expected it to go further, reduce the target or add "
                    "steps; the GUI already retries with adaptive step-cutting.")
    return tips


def summarize(checks) -> tuple[int, int, int]:
    """(errors, warnings, infos) counts."""
    return (sum(1 for c in checks if c.level == "error"),
            sum(1 for c in checks if c.level == "warning"),
            sum(1 for c in checks if c.level == "info"))
