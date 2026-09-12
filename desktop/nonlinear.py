"""Nonlinear (fiber) analysis for the desktop — the "background" behind the
nonlinear-run UI (plan §14 GUI-4/5).

Compiles the project's General-Section-Designer (GSD) column sections into
inelastic ``FiberSection2D`` + force-based beam-columns and runs a
displacement-controlled pushover (optionally holding an axial preload). It
reuses the ``section_gui_core`` material factories and the engine's
staged / nonlinear analysis — no new constitutive or solver code (plan §15:
one section engine).

2-D only for now. Units are the project's own (the desktop stores SI).
"""
from __future__ import annotations


def fiber_section_from_spec(spec, *, target: int = 1200, n_z: int = 16,
                            n_y: int = 40):
    """Build an inelastic :class:`FiberSection2D` from a ``section_gui_core.Spec``.

    Concrete and rebar use the Spec's own constitutive models via the shared
    ``section_gui_core`` factories (so the fiber law matches what the Section
    Designer previews). Circular sections use the unified polar mesh (U2
    :func:`polar_cells`); other shapes use the Cartesian grid. One concrete
    material for now — confined core/cover split is a follow-up.
    """
    import section_gui_core as core
    from femsolver.sections.response.fiber import (Fiber, FiberSection2D,
                                                   polar_cells, polar_divisions)
    from femsolver.sections.section import _discretize_polygon_to_fibers

    case = core.build_case(spec)
    poly = case.section.geometry.polygon
    conc = core.concrete_uniaxial_from(dict(
        fc=spec.fc, conc_model=spec.conc_model, eps_c0=spec.eps_c0,
        eps_cu=spec.eps_cu, fcu_ratio=spec.fcu_ratio, fr_model=spec.fr_model,
        fr_coeff=spec.fr_coeff, eps_decay=spec.eps_decay,
        conc_f1_ratio=spec.conc_f1_ratio))
    steel = core.steel_uniaxial_from(dict(
        fy=spec.fy, Es=spec.Es, steel_model=spec.steel_model,
        steel_b=spec.steel_b, steel_fu_ratio=spec.steel_fu_ratio,
        steel_eps_sh=spec.steel_eps_sh, steel_eps_su=spec.steel_eps_su))

    if getattr(spec, "kind", "") == "Circular":
        R = float(spec.D) / 2.0
        n_r, n_theta = polar_divisions(target)
        fibers = [Fiber(y=y, z=z, area=a, material=conc.clone())
                  for (y, z, a) in polar_cells(0.0, R, n_r, n_theta)]
    else:
        fibers = list(_discretize_polygon_to_fibers(poly, conc, n_z=n_z, n_y=n_y))

    bars = case.section.reinforcement.bars if case.section.reinforcement else []
    for b in bars:
        fibers.append(Fiber(y=float(b.y), z=float(b.z), area=float(b.area),
                            material=steel.clone()))
    return FiberSection2D(fibers)


def build_nonlinear_model(project):
    """Compile ``project`` into a nonlinear ``femsolver.Model``: members whose
    section carries a ``gsd_spec`` become fiber ``ForceBeamColumn2DCorotational``;
    the rest stay elastic ``BeamColumn2D``. Raises if no fiber section is found.
    """
    from femsolver import (BeamColumn2D, BeamColumn2DCorotational,
                           ElasticIsotropic, Model)
    from project import _gsd_modulus, _resolve_section, _spec_from_gsd

    if project.ndm != 2:
        raise ValueError("nonlinear fiber model is 2-D only for now")
    m = Model(ndm=2, ndf=3)
    mats = {}
    for mat in project.materials:
        obj = ElasticIsotropic(mat.id, E=mat.E, nu=mat.nu)
        m.add_material(obj)
        mats[mat.id] = obj
    for nd in project.nodes:
        m.add_node(nd.id, nd.x, nd.y)

    secs = {s.id: s for s in project.sections}
    n_fiber = 0
    for mb in project.members:
        sec = secs[mb.section]
        if getattr(sec, "gsd_spec", None):
            fs = fiber_section_from_spec(_spec_from_gsd(sec.gsd_spec))
            try:
                Ec = _gsd_modulus(sec.gsd_spec)
            except Exception:                          # noqa: BLE001
                Ec = mats[mb.material].E if mb.material in mats else 3.0e10
            base = ElasticIsotropic(100_000 + mb.id, E=Ec, nu=0.2)
            m.add_material(base)
            hinge = (project.hinge(mb.hinge)
                     if getattr(mb, "hinge", None) else None)
            if hinge is not None:
                # Lumped finite-length fiber hinge (P9 / GUI-3): elastic member
                # with a fiber plastic hinge of length lp at each end (CSI
                # "Fiber P-M2-M3" / Midas lumped hinge).
                from femsolver.elements.beam_fiber_hinge import \
                    FiberHingeBeamColumn2D
                lp_i, lp_j = project.resolve_hinge_lengths(mb, hinge)
                m.add_element(FiberHingeBeamColumn2D(
                    mb.id, (mb.n1, mb.n2), base, section=fs,
                    lp=lp_i, lp_j=lp_j))
            else:
                # Displacement-based corotational fiber element: robust for GUI
                # pushover to large drift (the force-based element's flexibility
                # can go singular near softening — see plan §8 / P8).
                m.add_element(BeamColumn2DCorotational(
                    mb.id, (mb.n1, mb.n2), base, section=fs))
            n_fiber += 1
        else:
            A, Iz, _Iy, _J = _resolve_section(sec)
            m.add_element(BeamColumn2D(mb.id, (mb.n1, mb.n2),
                                       mats[mb.material], A, Iz))

    if n_fiber == 0:
        raise ValueError(
            "no fiber (Section Designer) sections found — assign a GSD section "
            "to at least one member for nonlinear analysis")
    for nd in project.nodes:
        if nd.supports and any(nd.supports):
            m.fix(nd.id, list(nd.supports))
    return m


def _section_def(el, ip: int = 0):
    """Committed base-section deformation (eps_a, kappa) at integration point
    ``ip`` — from the displacement-based corotational element's ``_e_sections``
    or the force-based (hinge) element's ``_e_committed``; None before the first
    commit / if unavailable."""
    for attr in ("_e_sections", "_e_committed"):
        arr = getattr(el, attr, None)
        if arr is not None and len(arr) > ip:
            return float(arr[ip][0]), float(arr[ip][1])
    return None


def _peak_abs_strain(el) -> float:
    d = _section_def(el, 0)
    if d is None:
        return 0.0
    eps_a, kappa = d
    return max((abs(eps_a - f.y * kappa) for f in el.sections[0].fibers),
               default=0.0)


class _Capturer:
    """Per-step recorder for the GUI-6 post-processing: the monitored base
    section's per-fiber (y, z, σ, ε) snapshot, the whole model's nodal
    deformation, and each fiber member's peak |fiber strain| (hinge state).
    Snapshots come from a section *clone*, so the live state is untouched."""

    def __init__(self, model, mon_el, fiber_members,
                 capture_fibers: bool, capture_shape: bool):
        self._m = model
        self._mon = mon_el
        self._fiber_members = fiber_members
        self._fibers = capture_fibers
        self._shape = capture_shape
        self.frames: list = []
        self.shape_frames: list = []
        self.damage_frames: list = []

    @property
    def active(self) -> bool:
        return self._fibers or self._shape

    def capture(self) -> None:
        if self._fibers and self._mon is not None:
            d = _section_def(self._mon, 0)
            if d is not None:
                eps_a, kappa = d
                snap = self._mon.sections[0].clone()
                self.frames.append([
                    (float(f.y), float(f.z),
                     float(f.material.get_response(eps_a - f.y * kappa)[0]),
                     float(eps_a - f.y * kappa))
                    for f in snap.fibers])
        if self._shape:
            self.shape_frames.append(
                {nid: (float(n.disp[0]), float(n.disp[1]))
                 for nid, n in self._m.nodes.items()})
            self.damage_frames.append(
                {eid: _peak_abs_strain(self._m.elements[eid])
                 for eid in self._fiber_members})

    def result_into(self, result: dict) -> None:
        if self._fibers:
            result["fiber_frames"] = self.frames
        if self._shape:
            result["shape_frames"] = self.shape_frames
            result["damage_frames"] = self.damage_frames


def _fiber_members(project, model):
    """(fiber member ids, monitored element) — the members whose section is a
    fiber (Section-Designer) section; the monitored element is the first one
    (IP 0 = its n1 end, the base of a cantilever)."""
    secs = {s.id: s for s in project.sections}
    ids = [mb.id for mb in project.members
           if getattr(secs.get(mb.section), "gsd_spec", None)]
    return ids, (model.elements[ids[0]] if ids else None)


def run_pushover(project, *, control_node: int, control_dof: int,
                 target: float, n_steps: int = 40, axial: float = 0.0,
                 axial_node: int | None = None, axial_dof: int = 0,
                 tol: float = 1e-6, max_iter: int = 60,
                 on_step=None, should_cancel=None,
                 capture_fibers: bool = False,
                 capture_shape: bool = False) -> dict:
    """Displacement-controlled pushover of the nonlinear fiber model.

    Pushes ``control_node`` DOF ``control_dof`` (0=Ux, 1=Uy, 2=Rz) to
    ``target`` over ``n_steps``. With ``axial`` (compression magnitude) at
    ``axial_node`` the preload is applied first (load control) and *held
    constant* while pushing (staged). Returns ``{"disp", "shear"}`` — abs tip
    displacement and abs base shear (the displacement-control load factor of
    the unit reference load).

    ``on_step(info)`` (if given) is called after each push step with
    ``{step, num_steps, disp, shear}`` for live UI progress; ``should_cancel()``
    (if given) is polled each step and, when true, stops the push early
    (keeping the curve so far) — the hooks a threaded UI runner uses.
    """
    from femsolver import NonlinearStaticAnalysis, StagedAnalysis, monotonic
    from femsolver.analysis.static_integrator import DisplacementControl

    m = build_nonlinear_model(project)
    targets = monotonic(target, n_steps)
    du = float(targets[1] - targets[0])
    ref = [0.0, 0.0, 0.0]
    ref[control_dof] = -1.0

    # Optional per-step capture at the monitored member's base section (feeds
    # the GUI-6 fiber contour / deformed-shape / hinge-state views).
    fiber_members, mon_el = _fiber_members(project, m)
    cap = _Capturer(m, mon_el, fiber_members, capture_fibers, capture_shape)

    def _step_cb(info):
        if on_step is not None:
            on_step({"step": info["step"], "num_steps": info["num_steps"],
                     "disp": abs(info["tracked"] or 0.0),
                     "shear": abs(info["lambda"])})
        if cap.active:
            cap.capture()
        if should_cancel is not None and should_cancel():
            return False                               # cooperative cancel
        return True

    def _push(mm):
        mm.add_nodal_load(control_node, ref)
        return NonlinearStaticAnalysis(
            mm, num_steps=n_steps,
            integrator=DisplacementControl(control_node, control_dof, du),
            track=(control_node, control_dof), tol=tol, max_iter=max_iter,
            step_callback=_step_cb)

    if axial and axial_node:
        aref = [0.0, 0.0, 0.0]
        aref[axial_dof] = -abs(axial)
        sa = StagedAnalysis(m)
        sa.add_stage("axial", lambda mm: (
            mm.add_nodal_load(axial_node, aref),
            NonlinearStaticAnalysis(mm, num_steps=8, dlambda=0.125,
                                    integrator="load_control", tol=tol,
                                    max_iter=max_iter))[1])
        sa.add_stage("push", _push)
        out = sa.run()["push"]
    else:
        out = _push(m).run()

    result = {"disp": [abs(float(x)) for x in out["tracked"]],
              "shear": [abs(float(x)) for x in out["lambdas"]],
              "protocol": "monotonic"}
    cap.result_into(result)
    return result


# --------------------------------------------------------------- case runner

def _case_chain(project, case) -> list:
    """The ordered ``continue_from`` chain ending at ``case`` — ``[root, …,
    case]`` — so a staged run replays each ancestor's push, then this case's,
    continuing from committed state. Broken/cyclic links stop the walk."""
    chain = [case]
    seen = {case.id}
    cur = case
    while cur.continue_from:
        prev = project.nonlinear_case(cur.continue_from)
        if prev is None or prev.id in seen:
            break
        chain.append(prev)
        seen.add(prev.id)
        cur = prev
    chain.reverse()
    return chain


def _case_du(case):
    """(du, n_steps) for a case's protocol: a scalar increment (monotonic) or a
    signed per-step increment schedule (cyclic), plus the step count."""
    from femsolver.analysis.protocols import monotonic, stepped_cyclic
    import numpy as np
    if case.protocol == "cyclic":
        targets = stepped_cyclic(tuple(case.amplitudes),
                                 scale=float(case.target),
                                 cycles=int(case.cycles),
                                 pts_per_cycle=int(case.pts_per_cycle))
        du = np.diff(targets)
        return du, int(len(du))
    targets = monotonic(float(case.target), int(case.n_steps))
    return float(targets[1] - targets[0]), int(case.n_steps)


def case_total_steps(project, case) -> int:
    """Total push steps a case run will report (summed over the continue-from
    chain) — for sizing a progress bar."""
    return sum(_case_du(c)[1] for c in _case_chain(project, case))


def run_case(project, case, *, on_step=None, should_cancel=None,
             capture_fibers: bool = False, capture_shape: bool = False) -> dict:
    """Run a saved :class:`project.NonlinearCase` — monotonic or cyclic,
    optional held axial preload, optional ``continue_from`` staged continuation.

    Returns ``{"disp", "shear", "protocol"}`` (signed control-DOF displacement
    and total base shear, so cyclic runs trace the hysteresis) plus the GUI-6
    capture frames. Progress/cancel hooks match :func:`run_pushover`."""
    from femsolver import NonlinearStaticAnalysis, StagedAnalysis
    from femsolver.analysis.static_integrator import DisplacementControl

    m = build_nonlinear_model(project)
    chain = _case_chain(project, case)
    root = chain[0]
    need_axial = bool(root.axial and root.axial_node)

    fiber_members, mon_el = _fiber_members(project, m)
    cap = _Capturer(m, mon_el, fiber_members, capture_fibers, capture_shape)
    total_steps = case_total_steps(project, case)
    state = {"active": False, "i": 0}

    def _step_cb(info):
        if state["active"]:
            state["i"] += 1
            if on_step is not None:
                on_step({"step": state["i"], "num_steps": total_steps,
                         "disp": float(info["tracked"] or 0.0),
                         "shear": -float(info["lambda"] or 0.0)})
            if cap.active:
                cap.capture()
        if should_cancel is not None and should_cancel():
            return False
        return True

    def _push_factory(c):
        du, nsteps = _case_du(c)
        ref = [0.0, 0.0, 0.0]
        ref[c.control_dof] = -1.0

        def factory(mm):
            state["active"] = True
            mm.add_nodal_load(c.control_node, ref)
            return NonlinearStaticAnalysis(
                mm, num_steps=nsteps,
                integrator=DisplacementControl(c.control_node, c.control_dof,
                                               du),
                track=(c.control_node, c.control_dof),
                tol=float(c.tol), max_iter=int(c.max_iter),
                step_callback=_step_cb)
        return factory

    def _axial_factory(c):
        aref = [0.0, 0.0, 0.0]
        aref[c.axial_dof] = -abs(float(c.axial))

        def factory(mm):
            state["active"] = False           # don't record the preload stage
            mm.add_nodal_load(c.axial_node, aref)
            return NonlinearStaticAnalysis(
                mm, num_steps=8, dlambda=0.125, integrator="load_control",
                tol=float(c.tol), max_iter=int(c.max_iter),
                step_callback=_step_cb)
        return factory

    disp: list = []
    shear: list = []
    if len(chain) == 1 and not need_axial:
        out = _push_factory(case)(m).run()
        disp = [float(x) for x in out["tracked"]]
        shear = [-float(x) for x in out["lambdas"]]
    else:
        sa = StagedAnalysis(m)
        if need_axial:
            sa.add_stage("axial", _axial_factory(root))
        for i, c in enumerate(chain):
            sa.add_stage(f"push{i}", _push_factory(c))
        out = sa.run()
        offset = 0.0                          # cumulative held lateral factor
        for i in range(len(chain)):
            r = out[f"push{i}"]
            for u, lam in zip(r["tracked"], r["lambdas"]):
                disp.append(float(u))
                shear.append(-(offset + float(lam)))
            if r["lambdas"]:
                offset += float(r["lambdas"][-1])

    result = {"disp": disp, "shear": shear, "protocol": case.protocol}
    cap.result_into(result)
    return result
