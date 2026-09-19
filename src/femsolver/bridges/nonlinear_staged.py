"""Nonlinear staged cable-stayed erection (bridge plan T2.2, second half).

Cable-stayed and suspension bridges are *erected* stay-by-stay: a segment is
added, its stay is tensioned, the deck deflects, the next segment is added at
the **new deformed tip**, and so on.  Getting the locked-in force distribution
right needs three nonlinearities that a one-shot linear analysis misses:

1. **Geometric (large-displacement) stiffness** — a taut stay is stiff
   transversely in proportion to its tension (the string / P-Δ effect); a slack
   one is not.  The chord direction and length are taken from the *current*
   configuration each iteration (a corotational axial formulation).
2. **Cable sag (Ernst)** — a real stay sags under self-weight, so part of every
   tension change is taken up by the sag rather than by pure elongation.  The
   effective axial modulus :func:`ernst_equivalent_modulus` depends on the
   current tension and is iterated to consistency.
3. **Stress-free birth in the deformed geometry** — a segment installed at
   stage *k* is unstressed in the geometry that exists *at stage k*, not in the
   original drawing geometry.  Each element therefore measures strain from its
   **birth length**, so it carries force only from deformation that happens
   after it is born.

:class:`NonlinearStagedErection` is a self-contained active-set Newton driver
for a **2-D or 3-D** pin-jointed cable / truss network (the cable-erection
idealisation: stays, hangers, and axial deck/pylon chords). The spatial
dimension is inferred from the node coordinates; gravity (for the Ernst sag
correction) acts along the last axis (``y`` in 2-D, ``z`` in 3-D). It consumes
a list of
:class:`ErectionStage` (birth / death / incremental load / stay pretension) and
returns the per-stage and final tensions and displacements, validated against
closed forms (taut-string deflection, Ernst reduction, stress-free birth,
linear-limit agreement) in ``tests/test_bridge_nonlinear_staged.py``.

Deck **bending** during erection (a corotational *beam* with a birth datum) is
a deliberate future extension; this increment delivers the cable nonlinearities,
which is where staged cable-stayed erection differs from a linear run.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from femsolver.bridges.cable import ernst_equivalent_modulus


# ============================================================ data
@dataclass
class CableSegment:
    """One axial member of a staged cable network (stay, hanger, or a
    pin-jointed deck / pylon chord).

    Parameters
    ----------
    tag : int
        Unique id (referenced by the stage script).
    node_a, node_b : int
        End node ids.
    E, A : float
        Axial modulus (Pa) and area (m²).
    gamma_eff : float, default 0.0
        Cable weight per unit length perpendicular to the chord (N/m) for the
        Ernst sag correction.  ``0`` → a straight (sag-free) member.
    tension_only : bool, default False
        A true cable goes slack (carries no compression) — its stiffness drops
        to a small residual when the axial force would be compressive.
    name : str, default ""
    """

    tag: int
    node_a: int
    node_b: int
    E: float
    A: float
    gamma_eff: float = 0.0
    tension_only: bool = False
    name: str = ""


@dataclass
class ErectionStage:
    """One step of a staged erection.

    Attributes
    ----------
    name : str
    add : list[int]
        Segment tags **born** at the start of this stage — installed
        stress-free in the geometry that exists now.
    remove : list[int]
        Segment tags **removed** (temporary stays, falsework); their force
        redistributes to the remaining structure when equilibrium is re-solved.
    loads : dict[int, tuple]
        ``{node: (Fx, Fy[, Fz])}`` nodal loads applied at this stage (2-D or
        3-D, matching the model; incremental — added to the running total held
        through later stages).
    pretension : dict[int, float]
        ``{segment_tag: N0}`` stay pretension introduced at this stage (a
        lack-of-fit initial axial force, N, positive = tension).  Held into
        later stages; reassigning a tag overwrites it.
    """

    name: str = ""
    add: list = field(default_factory=list)
    remove: list = field(default_factory=list)
    loads: dict = field(default_factory=dict)
    pretension: dict = field(default_factory=dict)


@dataclass
class NonlinearStagedResult:
    """Outcome of :meth:`NonlinearStagedErection.run`.

    Attributes
    ----------
    stage_names : list[str]
    displacements : dict[int, tuple]
        Final ``{node: (ux, uy)}``.
    tensions : dict[int, float]
        Final axial force per segment active at the end (N, + = tension).
    tension_history : dict[int, list]
        Per segment, its axial force at the end of each stage (``None`` when
        inactive that stage).
    stage_displacements : list[dict]
        Per stage, ``{node: (ux, uy)}`` at the end of that stage.
    iterations : list[int]
        Newton iterations taken at each stage.
    converged : list[bool]
    """

    stage_names: list
    displacements: dict
    tensions: dict
    tension_history: dict
    stage_displacements: list
    iterations: list
    converged: list


# ============================================================ driver
class NonlinearStagedErection:
    """Active-set corotational Newton driver for staged cable erection.

    Parameters
    ----------
    nodes : dict[int, tuple]
        ``{node: (x, y)}`` (2-D) or ``{node: (x, y, z)}`` (3-D) reference
        coordinates. The dimension is taken from these and fixes the DOF count
        per node; ``supports`` and ``loads`` must match it.
    segments : iterable[CableSegment]
        Every segment used across the stages (birth/death selects which are
        active).
    stages : list[ErectionStage]
    supports : dict[int, tuple], optional
        ``{node: (fix_x, fix_y)}`` booleans; unlisted nodes are free.
    initial_active : iterable[int], optional
        Segment tags active before stage 1 (default: none — everything is born
        by some stage's ``add``).
    tol : float, default 1e-10
        Convergence tolerance on the residual norm (relative to the load norm)
        and the displacement-increment norm.
    max_iter : int, default 60
        Maximum Newton iterations per stage.
    slack_stiffness : float, default 1e-6
        Residual axial-stiffness fraction kept for a slack tension-only cable
        (keeps the tangent non-singular).
    """

    def __init__(self, nodes, segments, stages, *, supports=None,
                 initial_active=None, tol=1e-10, max_iter=60,
                 slack_stiffness=1e-6):
        self.X = {int(t): np.asarray(c, dtype=float) for t, c in nodes.items()}
        self.seg = {int(s.tag): s for s in segments}
        self.stages = list(stages)
        if not self.stages:
            raise ValueError("at least one erection stage required")
        # spatial dimension from the node coordinates (2-D or 3-D cable net)
        self.ndim = len(next(iter(self.X.values()))) if self.X else 2
        if self.ndim not in (2, 3):
            raise ValueError("node coordinates must be 2-D or 3-D")
        supports = supports or {}
        self.fix = {int(t): tuple(bool(b) for b in f)
                    for t, f in supports.items()}
        self.initial_active = set(int(t) for t in (initial_active or []))
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.slack = float(slack_stiffness)

        # DOF numbering: ndim per node, free DOFs only (fixed → eqn -1)
        self._node_ids = sorted(self.X)
        self._dof = {}
        eq = 0
        for nid in self._node_ids:
            fixes = self.fix.get(nid, (False,) * self.ndim)
            dofs = []
            for k in range(self.ndim):
                if k < len(fixes) and fixes[k]:
                    dofs.append(-1)
                else:
                    dofs.append(eq)
                    eq += 1
            self._dof[nid] = tuple(dofs)
        self.neq = eq

        self.u = {nid: np.zeros(self.ndim)          # displacements
                  for nid in self._node_ids}
        self.L0 = {}          # per-segment birth (reference) length
        self.N_pre = {}       # per-segment pretension (lack-of-fit force)
        self._N_last = {}     # per-segment axial force (lagged, for Ernst T)

    # ---------------------------------------------------------------- geometry
    def _pos(self, nid):
        return self.X[nid] + self.u[nid]

    def _chord(self, s):
        """Current chord vector, length, unit vector (from a→b)."""
        d = self._pos(s.node_b) - self._pos(s.node_a)
        L = float(np.linalg.norm(d))
        if L == 0.0:
            raise ValueError(f"segment {s.tag}: end nodes coincide")
        return d, L, d / L

    def _axial(self, s):
        """Return ``(N, E_eff, L0, L, n, slack)`` for an active segment at the
        current state — the total axial force incl. pretension (floored at 0
        for a slack tension-only cable), with the Ernst modulus iterated from
        the current tension."""
        _, L, n = self._chord(s)
        L0 = self.L0[s.tag]
        eps = (L - L0) / L0
        E = s.E
        # Ernst sag: E_eff depends on the current tension. Use the segment's
        # last axial force as the operating tension (a fixed point that is
        # self-consistent at convergence), falling back to the bare-E estimate
        # on the first pass.
        if s.gamma_eff > 0.0:
            T = self._N_last.get(s.tag, 0.0)
            if T <= 0.0:
                T = s.E * s.A * eps + self.N_pre.get(s.tag, 0.0)
            if T > 0.0:
                # horizontal span (gravity acts along the last axis: y in 2-D,
                # z in 3-D), so the horizontal projection is all-but-last.
                dchord = self._pos(s.node_b) - self._pos(s.node_a)
                L_h = float(np.linalg.norm(dchord[:-1]))
                if L_h > 0.0:
                    E = ernst_equivalent_modulus(
                        E=s.E, A=s.A, L_h=L_h, gamma_eff=s.gamma_eff, T=T)
        N = E * s.A * eps + self.N_pre.get(s.tag, 0.0)
        slack = s.tension_only and N <= 0.0
        if slack:                                     # a cable carries no compression
            N = 0.0
        self._N_last[s.tag] = N
        return N, E, L0, L, n, slack

    # ------------------------------------------------------------------- solve
    def _assemble(self, active, F):
        """Tangent K (neq×neq) and residual r = F_ext − f_int (neq) for the
        active set at the current displacement."""
        K = np.zeros((self.neq, self.neq))
        f_int = np.zeros(self.neq)
        nd = self.ndim
        I_n = np.eye(nd)
        for tag in active:
            s = self.seg[tag]
            N, E_eff, L0, L, n, slack = self._axial(s)
            km = E_eff * s.A / L0                       # material axial stiff
            if slack:
                km *= self.slack                        # residual stiffness
            nn = np.outer(n, n)
            kg = (N / L)                                # geometric (string)
            Ablk = km * nn + kg * (I_n - nn)
            fe = np.concatenate((-N * n, N * n))        # global 2·ndim vector
            Kblk = np.block([[Ablk, -Ablk], [-Ablk, Ablk]])
            eqs = (*self._dof[s.node_a], *self._dof[s.node_b])
            for a in range(2 * nd):
                ea = eqs[a]
                if ea < 0:
                    continue
                f_int[ea] += fe[a]
                for b in range(2 * nd):
                    eb = eqs[b]
                    if eb >= 0:
                        K[ea, eb] += Kblk[a, b]
        return K, F - f_int

    def _load_vector(self, load_total):
        F = np.zeros(self.neq)
        for nid, vec in load_total.items():
            dofs = self._dof[int(nid)]
            for k, val in enumerate(np.asarray(vec, dtype=float)):
                if k < len(dofs) and dofs[k] >= 0:
                    F[dofs[k]] += float(val)
        return F

    def _newton(self, active, F):
        """Newton-Raphson to equilibrium of the active set under load F.
        Returns (iterations, converged)."""
        fnorm = max(np.linalg.norm(F), 1.0)
        for it in range(1, self.max_iter + 1):
            K, r = self._assemble(active, F)
            if np.linalg.norm(r) <= self.tol * fnorm:
                return it - 1, True
            try:
                du = np.linalg.solve(K, r)
            except np.linalg.LinAlgError:
                return it, False
            if not np.all(np.isfinite(du)):
                return it, False
            # scatter increment
            for nid in self._node_ids:
                for k, eqk in enumerate(self._dof[nid]):
                    if eqk >= 0:
                        self.u[nid][k] += du[eqk]
            if np.linalg.norm(du) <= self.tol * max(
                    self._u_norm(), 1e-12):
                # one more residual check after the update
                _, r2 = self._assemble(active, F)
                return it, bool(np.linalg.norm(r2) <= 1e-6 * fnorm)
        _, r = self._assemble(active, F)
        return self.max_iter, bool(np.linalg.norm(r) <= 1e-6 * fnorm)

    def _u_norm(self):
        return float(np.sqrt(sum(float(v @ v) for v in self.u.values())))

    def _birth_length(self, s):
        d = self._pos(s.node_b) - self._pos(s.node_a)
        return float(np.linalg.norm(d))

    def run(self) -> NonlinearStagedResult:
        active = set(self.initial_active)
        for tag in active:                              # born at original geom
            self.L0[tag] = self._birth_length(self.seg[tag])
        load_total: dict = {}
        names, iters, conv = [], [], []
        stage_disps, tension_hist = [], {t: [] for t in self.seg}

        for stage in self.stages:
            # ---- death: drop from the active set (force redistributes on solve)
            for tag in stage.remove:
                if tag not in active:
                    raise ValueError(
                        f"stage {stage.name!r}: cannot remove segment {tag} — "
                        "it is not active")
                active.discard(tag)
            # ---- birth: install stress-free in the current deformed geometry
            for tag in stage.add:
                if tag not in self.seg:
                    raise ValueError(
                        f"stage {stage.name!r}: unknown segment {tag}")
                active.add(tag)
                self.L0[tag] = self._birth_length(self.seg[tag])
            # ---- pretension held from this stage on
            for tag, N0 in stage.pretension.items():
                self.N_pre[int(tag)] = float(N0)
            # ---- accumulate loads (2-D or 3-D force vectors)
            for nid, load in stage.loads.items():
                vec = np.asarray(load, dtype=float)
                cur = load_total.get(int(nid))
                load_total[int(nid)] = vec if cur is None else cur + vec

            if not active:
                raise RuntimeError(f"stage {stage.name!r}: nothing active")

            F = self._load_vector(load_total)
            n_it, ok = self._newton(active, F)

            names.append(stage.name)
            iters.append(n_it)
            conv.append(ok)
            stage_disps.append({nid: tuple(self.u[nid]) for nid in self._node_ids})
            for tag in self.seg:
                if tag in active:
                    tension_hist[tag].append(self._axial(self.seg[tag])[0])
                else:
                    tension_hist[tag].append(None)

        tensions = {tag: self._axial(self.seg[tag])[0] for tag in active}
        return NonlinearStagedResult(
            stage_names=names,
            displacements={nid: tuple(self.u[nid]) for nid in self._node_ids},
            tensions=tensions,
            tension_history=tension_hist,
            stage_displacements=stage_disps,
            iterations=iters,
            converged=conv,
        )
