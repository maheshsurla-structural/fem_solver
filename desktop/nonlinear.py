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


def fiber_section_from_spec(spec, *, materials=None, target: int = 1200,
                            n_z: int = 16, n_y: int = 40):
    """Build an inelastic :class:`FiberSection2D` from a ``section_gui_core.Spec``.

    Concrete and rebar use the Spec's own constitutive models via the shared
    ``section_gui_core`` factories (so the fiber law matches what the Section
    Designer previews).

    **Confined core / unconfined cover (plan §16 C1/G-S4):** when the section
    defines transverse confinement — a tie **Link** rebar group, or a manual
    confinement override on the Spec — the fibre section is the **same** two-zone
    confined-core + unconfined-cover section the Section Designer's confined M-φ
    uses (``section_gui_core._confined_fiber_section``), sourced from the same
    ``_section_confinement`` (one confinement source, no parallel path — plan
    §15). ``materials`` maps material name → props so the hoop grade's ``f_yh``
    is read from the tie's material (defaults when absent / not provided).

    With no confinement the whole section uses a single unconfined law (circular:
    polar mesh; other shapes: Cartesian grid) — the historical behaviour, so
    existing (unconfined) runs are unchanged.
    """
    import section_gui_core as core
    from femsolver.sections.response.fiber import (Fiber, FiberSection2D,
                                                   polar_cells, polar_divisions)
    from femsolver.sections.section import _discretize_polygon_to_fibers

    conf, _info = core._section_confinement(spec, materials or {})
    if conf is not None:
        # Unified confined two-zone section — identical to the confined M-φ path.
        fs, *_rest = core._confined_fiber_section(
            spec, conf, n_z=max(n_z, 28), n_y=max(n_y, 56))
        return fs

    # --- unconfined: a single concrete law over the whole section ---
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


def build_nonlinear_model(project, *, materials=None, density: float = 0.0):
    """Compile ``project`` into a nonlinear ``femsolver.Model``: members whose
    section carries a ``gsd_spec`` become fiber ``ForceBeamColumn2DCorotational``;
    the rest stay elastic ``BeamColumn2D``. Raises if no fiber section is found.

    ``density`` (mass per unit volume) is assigned to every element's material
    so the model carries consistent mass (element mass = rho·A·L) — needed for a
    dynamic time-history (plan §16 C3). Default 0 (massless) leaves the static
    pushover behaviour unchanged.
    """
    from femsolver import (BeamColumn2D, BeamColumn2DCorotational,
                           BeamColumn3D, BeamColumn3DCorotational,
                           ElasticIsotropic, Model)
    from femsolver.sections.response.fiber import FiberSection3D
    from project import _gsd_modulus, _resolve_section, _spec_from_gsd

    if project.ndm not in (2, 3):
        raise ValueError("nonlinear fiber model supports ndm 2 or 3")
    threeD = project.ndm == 3
    rho = float(density)
    m = Model(ndm=project.ndm, ndf=(6 if threeD else 3))
    mats = {}
    for mat in project.materials:
        obj = ElasticIsotropic(mat.id, E=mat.E, nu=mat.nu, rho=rho)
        m.add_material(obj)
        mats[mat.id] = obj
    for nd in project.nodes:
        if threeD:
            m.add_node(nd.id, nd.x, nd.y, nd.z)
        else:
            m.add_node(nd.id, nd.x, nd.y)

    secs = {s.id: s for s in project.sections}
    n_fiber = 0
    for mb in project.members:
        sec = secs[mb.section]
        if getattr(sec, "gsd_spec", None):
            fs = fiber_section_from_spec(_spec_from_gsd(sec.gsd_spec),
                                         materials=materials)
            try:
                Ec = _gsd_modulus(sec.gsd_spec)
            except Exception:                          # noqa: BLE001
                Ec = mats[mb.material].E if mb.material in mats else 3.0e10
            base = ElasticIsotropic(100_000 + mb.id, E=Ec, nu=0.2, rho=rho)
            m.add_material(base)
            if threeD:
                # 3-D biaxial (P-M2-M3): wrap the fibres in a FiberSection3D
                # (torsion held elastic via GJ) on the disp-based corotational
                # 3-D element. Lumped hinges are 2-D only, so 3-D uses the
                # distributed element.
                _A, _Iz, _Iy, J = _resolve_section(sec)
                GJ = Ec / (2.0 * (1.0 + 0.2)) * max(float(J), 1e-9)
                fs3 = FiberSection3D(list(fs.fibers), GJ=GJ)
                m.add_element(BeamColumn3DCorotational(
                    mb.id, (mb.n1, mb.n2), base, section=fs3))
                n_fiber += 1
                continue
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
        elif threeD:
            A, Iz, Iy, J = _resolve_section(sec)
            m.add_element(BeamColumn3D(mb.id, (mb.n1, mb.n2),
                                       mats[mb.material], A, Iy, Iz, J))
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
    """Committed base-section deformation vector at integration point ``ip`` —
    ``[eps_a, kappa_z]`` in 2-D or ``[eps_a, kappa_z, kappa_y, gamma]`` in 3-D —
    from the corotational element's ``_e_sections`` or the force-based (hinge)
    element's ``_e_committed``; None before the first commit / if unavailable."""
    for attr in ("_e_sections", "_e_committed"):
        arr = getattr(el, attr, None)
        if arr is not None and len(arr) > ip:
            return [float(x) for x in arr[ip]]
    return None


def _fiber_strain(f, e) -> float:
    """Plane-section fibre strain ``eps_a - y*kappa_z (+ z*kappa_y)`` for a 2-D
    or 3-D section-deformation vector ``e``."""
    eps = e[0] - f.y * e[1]
    if len(e) > 2:                              # 3-D biaxial: + z*kappa_y
        eps += f.z * e[2]
    return eps


def _peak_abs_strain(el) -> float:
    e = _section_def(el, 0)
    if e is None:
        return 0.0
    return max((abs(_fiber_strain(f, e)) for f in el.sections[0].fibers),
               default=0.0)


def _section_accept_state(el) -> int:
    """Governing ASCE 41 acceptance level (0 Elastic .. 3 CP) of the element's
    base section (§16 C5). Handles 2-D and 3-D (biaxial) sections."""
    from femsolver.performance.acceptance import section_state
    e = _section_def(el, 0)
    if e is None:
        return 0
    return section_state(el.sections[0].fibers, e[0], e[1],
                         kappa_y=(e[2] if len(e) > 2 else 0.0))


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
        self.accept_frames: list = []          # per step: {eid: level 0..3}

    @property
    def active(self) -> bool:
        return self._fibers or self._shape

    def capture(self) -> None:
        if self._fibers and self._mon is not None:
            e = _section_def(self._mon, 0)
            if e is not None:
                snap = self._mon.sections[0].clone()
                self.frames.append([
                    (float(f.y), float(f.z),
                     float(f.material.get_response(_fiber_strain(f, e))[0]),
                     float(_fiber_strain(f, e)))
                    for f in snap.fibers])
        if self._shape:
            ndm = self._m.ndm
            self.shape_frames.append(
                {nid: tuple(float(n.disp[d]) for d in range(ndm))
                 for nid, n in self._m.nodes.items()})
            self.damage_frames.append(
                {eid: _peak_abs_strain(self._m.elements[eid])
                 for eid in self._fiber_members})
            self.accept_frames.append(
                {eid: _section_accept_state(self._m.elements[eid])
                 for eid in self._fiber_members})

    def result_into(self, result: dict) -> None:
        if self._fibers:
            result["fiber_frames"] = self.frames
        if self._shape:
            result["shape_frames"] = self.shape_frames
            result["damage_frames"] = self.damage_frames
            result["accept_frames"] = self.accept_frames


def _fiber_members(project, model):
    """(fiber member ids, monitored element) — the members whose section is a
    fiber (Section-Designer) section; the monitored element is the first one
    (IP 0 = its n1 end, the base of a cantilever)."""
    secs = {s.id: s for s in project.sections}
    ids = [mb.id for mb in project.members
           if getattr(secs.get(mb.section), "gsd_spec", None)]
    return ids, (model.elements[ids[0]] if ids else None)


def _add_accept_milestones(result: dict) -> None:
    """From per-step acceptance levels + the control displacement, record the
    step/displacement at which the model first reaches IO, LS and CP (§16 C5)."""
    from femsolver.performance.acceptance import LEVELS
    af = result.get("accept_frames")
    disp = result.get("disp")
    if not af or not disp:
        return
    worst = [max(fr.values()) if fr else 0 for fr in af]
    milestones: dict = {}
    for lvl in (1, 2, 3):                       # IO, LS, CP
        for k, w in enumerate(worst):
            if w >= lvl:
                milestones[LEVELS[lvl]] = {
                    "step": k + 1,
                    "disp": (disp[k] if k < len(disp) else None)}
                break
    result["accept_milestones"] = milestones


def run_pushover(project, *, control_node: int, control_dof: int,
                 target: float, n_steps: int = 40, axial: float = 0.0,
                 axial_node: int | None = None, axial_dof: int = 0,
                 tol: float = 1e-6, max_iter: int = 60,
                 on_step=None, should_cancel=None,
                 capture_fibers: bool = False,
                 capture_shape: bool = False, materials=None) -> dict:
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

    m = build_nonlinear_model(project, materials=materials)
    targets = monotonic(target, n_steps)
    du = float(targets[1] - targets[0])
    ref = [0.0] * project.ndf
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
            step_callback=_step_cb, substep=True)      # C4: auto step-cutting

    if axial and axial_node:
        aref = [0.0] * project.ndf
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
    _add_accept_milestones(result)
    return result


# ----------------------------------------------------- dynamic time-history (C3)

def run_time_history(project, accel, dt, *, control_node: int,
                     control_dof: int = 1, direction: str = "y",
                     zeta: float = 0.05, density: float = 2400.0,
                     num_steps: int | None = None, tol: float = 1.0,
                     max_iter: int = 30, materials=None,
                     initial_case=None, hold_source_loads: bool = False,
                     on_step=None, should_cancel=None) -> dict:
    """Nonlinear **dynamic time-history** of the fiber model under rigid-base
    ground acceleration (plan §16 C3).

    Applies the base excitation ``accel`` (a ground-acceleration record sampled
    at ``dt``) in ``direction`` via the standard ``-M·ι·ü_g(t)`` inertia load
    (:func:`ground_motion_force`), with Rayleigh damping calibrated to ``zeta``
    at the first two natural modes (from an eigen analysis of the mass/stiffness
    model), and integrates with Newmark + Newton (``NonlinearTransientAnalysis``).
    Mass comes from ``density`` (rho·A·L per element).

    Returns ``{"times", "disp", "velocity", "acceleration", "dt", "peak_disp",
    "protocol": "time_history"}`` — the monitored DOF's response history (the
    standard seismic demand). ``control_dof`` defaults to 1 (Uy); ``direction``
    sets the excitation axis.

    **Initial conditions (E2):** when ``initial_case`` names a Nonlinear Static
    case, the run starts from *its* committed deformed + materially-committed
    state (via :func:`seed_to_committed_state`) instead of the unstressed state —
    the base excitation then perturbs the preloaded structure. With
    ``hold_source_loads`` the source case's held force is carried through the
    dynamic run (added as a constant term to the excitation) so the preload stays
    in equilibrium; without it only the stiffness + state are inherited (SAP
    semantics). The held vector and the excitation share the model's stable DOF
    numbering, so they superpose in equation space.
    """
    import numpy as np
    from femsolver import EigenAnalysis, NonlinearTransientAnalysis, RayleighDamping
    from femsolver.analysis.response_spectrum import ground_motion_force

    accel = np.asarray(accel, dtype=float).ravel()
    if accel.size < 2:
        raise ValueError("accel must have at least two samples")
    n = int(num_steps) if num_steps is not None else accel.size - 1

    F_hold = None
    if initial_case is not None:
        src = (initial_case if hasattr(initial_case, "control_node")
               else project.nonlinear_case(initial_case))
        if src is None:
            raise ValueError(
                f"the initial-condition source case {initial_case!r} was "
                "deleted — pick another in the case's Modify dialog")
        m, F_const = seed_to_committed_state(project, src, materials=materials,
                                             density=density)
        if hold_source_loads:
            F_hold = F_const
    else:
        m = build_nonlinear_model(project, materials=materials, density=density)
    m.number_dofs()
    if m.neq == 0:
        raise RuntimeError("model is fully constrained — no dynamic DOFs")

    # Rayleigh damping from the two lowest distinct modes at ratio zeta.
    n_modes = max(1, min(4, m.neq - 1))
    eig = EigenAnalysis(m, num_modes=n_modes).run()
    omegas = [2.0 * np.pi * float(f) for f in eig["frequencies_hz"]
              if f > 1e-9]
    distinct = []
    for w in omegas:
        if all(abs(w - d) > 1e-6 * max(w, 1.0) for d in distinct):
            distinct.append(w)
    if len(distinct) >= 2:
        damping = RayleighDamping.from_modes(distinct[0], zeta,
                                             distinct[1], zeta)
    elif distinct:                       # single mode -> mass-proportional only
        damping = RayleighDamping(alpha_M=2.0 * zeta * distinct[0], alpha_K=0.0)
    else:
        damping = None

    def accel_fn(t):
        x = t / dt
        i = int(x)
        if i < 0:
            return float(accel[0])
        if i >= accel.size - 1:
            return float(accel[-1])
        frac = x - i
        return float(accel[i] * (1.0 - frac) + accel[i + 1] * frac)

    load_fn = ground_motion_force(m, direction=direction, accel_function=accel_fn)

    if F_hold is not None:                     # hold the source loads (E2): the
        base_fn = load_fn                      # excitation + a constant preload
        F_hold = np.asarray(F_hold, dtype=float).ravel()

        def load_fn(t, _base=base_fn, _hold=F_hold):
            return np.asarray(_base(t), dtype=float).ravel() + _hold

    def _step_cb(info):
        if on_step is not None:
            on_step({"step": info["step"], "num_steps": info["num_steps"],
                     "time": info["time"], "disp": abs(info["disp"] or 0.0)})
        if should_cancel is not None and should_cancel():
            return False
        return True

    out = NonlinearTransientAnalysis(
        m, num_steps=n, dt=dt, damping=damping, load_function=load_fn,
        tol=tol, max_iter=max_iter,
        track=(control_node, control_dof), step_callback=_step_cb).run()

    disp = [float(x) for x in out["tracked_disp"]]
    return {
        "times": [float(t) for t in out["times"]],
        "disp": disp,
        "velocity": [float(v) for v in out["tracked_velocity"]],
        "acceleration": [float(a) for a in out["tracked_acceleration"]],
        "dt": float(dt),
        "peak_disp": max((abs(d) for d in disp), default=0.0),
        "protocol": "time_history",
    }


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


def _push_factory(ndf, c, *, step_cb=None, on_active=None):
    """A displacement-controlled push stage for nonlinear case ``c`` (its own
    monotonic/cyclic protocol). ``on_active(True)`` marks the stage as recorded
    (progress + capture); ``step_cb`` is the per-step hook. Shared by
    :func:`run_case` and :func:`seed_to_committed_state` so a case seeds to
    exactly the state its own pushover reaches."""
    from femsolver import NonlinearStaticAnalysis
    from femsolver.analysis.static_integrator import DisplacementControl
    du, nsteps = _case_du(c)
    ref = [0.0] * ndf
    ref[c.control_dof] = -1.0

    def factory(mm):
        if on_active is not None:
            on_active(True)
        mm.add_nodal_load(c.control_node, ref)
        return NonlinearStaticAnalysis(
            mm, num_steps=nsteps,
            integrator=DisplacementControl(c.control_node, c.control_dof, du),
            track=(c.control_node, c.control_dof),
            tol=float(c.tol), max_iter=int(c.max_iter),
            step_callback=step_cb, substep=True)   # C4 (monotonic only; a
        # cyclic schedule advertises supports_substep=False -> no-op)
    return factory


def _axial_factory(ndf, c, *, step_cb=None, on_active=None):
    """A load-controlled axial-preload stage, held constant by
    :class:`StagedAnalysis` while later stages run. ``on_active(False)`` keeps
    the preload out of the recorded pushover."""
    from femsolver import NonlinearStaticAnalysis
    aref = [0.0] * ndf
    aref[c.axial_dof] = -abs(float(c.axial))

    def factory(mm):
        if on_active is not None:
            on_active(False)                  # don't record the preload stage
        mm.add_nodal_load(c.axial_node, aref)
        return NonlinearStaticAnalysis(
            mm, num_steps=8, dlambda=0.125, integrator="load_control",
            tol=float(c.tol), max_iter=int(c.max_iter), step_callback=step_cb)
    return factory


def run_case(project, case, *, on_step=None, should_cancel=None,
             capture_fibers: bool = False, capture_shape: bool = False,
             materials=None) -> dict:
    """Run a saved :class:`project.NonlinearCase` — monotonic or cyclic,
    optional held axial preload, optional ``continue_from`` staged continuation.

    Returns ``{"disp", "shear", "protocol"}`` (signed control-DOF displacement
    and total base shear, so cyclic runs trace the hysteresis) plus the GUI-6
    capture frames. Progress/cancel hooks match :func:`run_pushover`."""
    from femsolver import StagedAnalysis

    m = build_nonlinear_model(project, materials=materials)
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

    def _on_active(v):                        # toggle stage recording
        state["active"] = v

    ndf = project.ndf
    disp: list = []
    shear: list = []
    if len(chain) == 1 and not need_axial:
        out = _push_factory(ndf, case, step_cb=_step_cb,
                            on_active=_on_active)(m).run()
        disp = [float(x) for x in out["tracked"]]
        shear = [-float(x) for x in out["lambdas"]]
    else:
        sa = StagedAnalysis(m)
        if need_axial:
            sa.add_stage("axial", _axial_factory(ndf, root, step_cb=_step_cb,
                                                 on_active=_on_active))
        for i, c in enumerate(chain):
            sa.add_stage(f"push{i}", _push_factory(ndf, c, step_cb=_step_cb,
                                                   on_active=_on_active))
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
    _add_accept_milestones(result)
    return result


def seed_to_committed_state(project, case, *, materials=None, density=0.0):
    """Run a Nonlinear Static ``case`` (its axial preload + ``continue_from``
    push chain) to its committed **end state**, without recording pushover curves
    or capture frames, and return ``(model, F_const)`` for a downstream analysis
    to build on (E2 — initial conditions).

    ``density`` is forwarded to :func:`build_nonlinear_model` so a downstream
    *dynamic* analysis (time history) receives a mass-bearing model; it does not
    affect the static committed state (the preload applies no mass-based load).

    * ``model`` is left at the committed deformed + materially-committed state
      (element / material history intact), ready to hand to a modal / response-
      spectrum / buckling / time-history solve seeded from this state.
    * ``F_const`` is the eqn-space constant-force vector summing every stage's
      converged applied load — the "held source loads" a downstream case carries
      when the user opts to hold them. SAP carries stiffness + state only, so
      holding these loads is opt-in (E2 checkbox); the model's own load pattern
      is cleared, so the held load lives *only* in the returned vector.

    Physics is identical to :func:`run_case`'s state production (same stage
    factories), so a case seeds to exactly the state its own pushover reaches.
    ``case`` is a :class:`project.NonlinearCase`; only nonlinear cases are valid
    initial-condition sources (SAP parity)."""
    from femsolver import StagedAnalysis
    from femsolver.analysis.assembler import assemble_force

    m = build_nonlinear_model(project, materials=materials, density=density)
    chain = _case_chain(project, case)
    root = chain[0]
    need_axial = bool(root.axial and root.axial_node)
    ndf = project.ndf

    if len(chain) == 1 and not need_axial:
        analysis = _push_factory(ndf, case)(m)
        analysis.run()
        f_const = float(analysis.integrator.lambd) * assemble_force(m)
        for node in m.nodes.values():         # held load lives in f_const only
            node._load[:] = 0.0
        return m, f_const

    sa = StagedAnalysis(m)
    if need_axial:
        sa.add_stage("axial", _axial_factory(ndf, root))
    for i, c in enumerate(chain):
        sa.add_stage(f"push{i}", _push_factory(ndf, c))
    sa.run()
    return m, sa.const_force_final
