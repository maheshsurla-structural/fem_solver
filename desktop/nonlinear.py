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
            # Displacement-based corotational fiber element: robust for GUI
            # pushover to large drift (the force-based element's flexibility can
            # go singular near softening — see plan §8 / P8).
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


def run_pushover(project, *, control_node: int, control_dof: int,
                 target: float, n_steps: int = 40, axial: float = 0.0,
                 axial_node: int | None = None, axial_dof: int = 0,
                 tol: float = 1e-6, max_iter: int = 60,
                 on_step=None, should_cancel=None) -> dict:
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

    def _step_cb(info):
        if on_step is not None:
            on_step({"step": info["step"], "num_steps": info["num_steps"],
                     "disp": abs(info["tracked"] or 0.0),
                     "shear": abs(info["lambda"])})
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

    return {"disp": [abs(float(x)) for x in out["tracked"]],
            "shear": [abs(float(x)) for x in out["lambdas"]]}
