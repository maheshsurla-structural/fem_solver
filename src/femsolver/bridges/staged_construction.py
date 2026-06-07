"""Staged-construction analysis with time-dependent creep redistribution.

Bridges (especially segmental PSC and cable-stayed) are built in
**stages**: a piece of structure is cast / erected, then loaded by
later additions before the concrete in the original piece has fully
matured. Each stage sees a different *effective* concrete modulus
because of creep, so the long-term distribution of internal forces
differs from a one-shot analysis where everything is loaded at once.

This module provides two cooperating ingredients:

* :class:`ConstructionStage` -- a dataclass describing one stage:
  which load patterns are applied, how many days elapse before the
  next stage, and which elements are "born" in this stage (i.e., are
  active from this stage onward).

* :class:`StagedConstructionAnalysis` -- a driver that runs a linear
  static analysis at each stage with the **Effective Modulus Method
  (EMM)**::

      E_eff(t, t0) = E_c / (1 + chi · phi(t, t0))

  where ``phi(t, t0)`` is the CEB-FIP creep coefficient (per
  :mod:`femsolver.bridges.creep_shrinkage`) and ``chi`` is the
  *ageing coefficient* (typically 0.8 for long-term sustained loads;
  Trost 1967 / Bazant 1972). For lacks of long-term laboratory data,
  ``chi = 0.8`` is the AASHTO/CEB-FIP default.

The driver returns a :class:`StagedConstructionResult` holding the
per-stage displacement increment and per-stage cumulative
displacement.

References
----------
* Trost, H. (1967). "Auswirkungen des Superpositionsprinzips auf
  Kriech- und Relaxationsprobleme bei Beton und Spannbeton."
  *Beton- und Stahlbetonbau*, 62 (10).
* Ghali, A., Favre, R. & Elbadry, M. (2002). *Concrete Structures:
  Stresses and Deformations*. CRC Press.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from femsolver.bridges.creep_shrinkage import cebfip_creep_coefficient


# ============================================================ stage data

@dataclass
class ConstructionStage:
    """One stage of a staged-construction analysis.

    Attributes
    ----------
    name : str
        Free-text label for reporting.
    duration_days : float
        Days between this stage's loading and the next stage's loading
        (used to advance the creep clock for subsequent stages).
    load_pattern : dict
        ``{node_tag: [Fx, Fy, ...]}`` of loads applied AT THIS STAGE.
    activated_elements : list, optional
        Element tags that become active in this stage. Cumulative:
        once activated, they remain active in later stages.
    age_at_loading_days : float, default 28.0
        Concrete age at the moment this stage's load is applied.
        Used to compute the creep coefficient.
    """

    name: str
    duration_days: float
    load_pattern: dict
    activated_elements: list = field(default_factory=list)
    age_at_loading_days: float = 28.0


# ============================================================ result

@dataclass
class StagedConstructionResult:
    """Outcome of a staged-construction analysis.

    Attributes
    ----------
    n_stages : int
    stage_names : list
    u_incremental : list of np.ndarray
        Per-stage displacement increment (size = model.neq).
    u_cumulative : np.ndarray
        Cumulative displacement at the END of all stages
        (size = model.neq).
    creep_factors : list of float
        Per-stage effective-modulus reduction ``1 / (1 + chi · phi)``
        used when solving that stage.
    """

    n_stages: int
    stage_names: list
    u_incremental: list
    u_cumulative: np.ndarray
    creep_factors: list


# ============================================================ driver

class StagedConstructionAnalysis:
    """Multi-stage linear analysis with EMM creep redistribution.

    For each stage:

    1. Compute the effective concrete modulus ``E_eff`` for the
       NEW load applied in this stage, based on the concrete age
       and the time until the END of the analysis.
    2. Scale the model's stiffness by ``E_eff / E_c`` (uniform
       scalar approximation) and solve for the stage's
       displacement increment.
    3. Accumulate the incremental displacement onto the running total.

    Parameters
    ----------
    model : Model
        Reference model. The stiffness ``K_0`` is computed once
        from this model and then rescaled per stage.
    stages : list of ConstructionStage
    f_cm : float
        Mean concrete strength (Pa). Used for the creep coefficient.
    chi : float, default 0.8
        Ageing coefficient (Trost/Bazant).
    RH : float, default 70.0
        Relative humidity (percent).
    h_0 : float, default 0.20
        Notional member size (m).
    final_age_days : float, optional
        Concrete age at the end of the analysis (typically 50 years =
        18250 days). If omitted, the creep coefficient is evaluated
        out to ``stage.age_at_loading_days + sum(durations)``.
    """

    def __init__(
        self,
        model,
        *,
        stages: list,
        f_cm: float,
        chi: float = 0.8,
        RH: float = 70.0,
        h_0: float = 0.20,
        final_age_days: float | None = None,
    ):
        if not stages:
            raise ValueError("at least one stage required")
        if f_cm <= 0.0:
            raise ValueError("f_cm must be > 0")
        if not (0.0 < chi <= 1.0):
            raise ValueError("chi must be in (0, 1]")
        self.model = model
        self.stages = list(stages)
        self.f_cm = float(f_cm)
        self.chi = float(chi)
        self.RH = float(RH)
        self.h_0 = float(h_0)
        # Build the timeline: total days from start of stage 1
        cumulative = []
        t = 0.0
        for s in self.stages:
            cumulative.append(t)
            t += s.duration_days
        self._stage_start_days = cumulative
        if final_age_days is None:
            # default: max stage start + 50 yr
            self.final_age_days = float(cumulative[-1] + 18250.0)
        else:
            self.final_age_days = float(final_age_days)

    def _creep_factor_for_stage(self, stage_idx: int) -> float:
        """``1 / (1 + chi · phi(t_final, t_load))`` for the given stage."""
        stage = self.stages[stage_idx]
        t_load = stage.age_at_loading_days
        t_final = max(t_load + 1.0, self.final_age_days)
        creep = cebfip_creep_coefficient(
            t_days=t_final, t0_days=t_load,
            f_cm=self.f_cm, RH=self.RH, h_0=self.h_0,
        )
        return float(1.0 / (1.0 + self.chi * creep.phi))

    def run(self) -> StagedConstructionResult:
        """Run the multi-stage analysis."""
        from femsolver.analysis.assembler import (
            assemble_force,
            assemble_stiffness,
        )
        from scipy.sparse.linalg import spsolve

        m = self.model
        m.number_dofs()
        # Assemble the base stiffness once (assumes E is uniform across
        # elements, which is the canonical concrete-bridge case)
        K0 = assemble_stiffness(m).tocsc()
        u_cum = np.zeros(m.neq)
        u_increments = []
        creep_factors = []
        names = []
        # Reset model's load slot, then apply each stage's loads in turn
        for s_idx, stage in enumerate(self.stages):
            # Clear nodal loads, apply only this stage's loads
            for node in m.nodes.values():
                node._load[:] = 0.0
            for node_tag, load in stage.load_pattern.items():
                m.node(node_tag)._load[: len(load)] = np.asarray(
                    load, dtype=float,
                )
            f = assemble_force(m)
            factor = self._creep_factor_for_stage(s_idx)
            K_eff = K0 * factor
            try:
                du = spsolve(K_eff, f)
            except Exception as exc:
                raise RuntimeError(
                    f"stage '{stage.name}' (index {s_idx}) solve failed: "
                    f"{exc}"
                ) from exc
            du = np.asarray(du).ravel()
            u_cum += du
            u_increments.append(du)
            creep_factors.append(factor)
            names.append(stage.name)
        # Scatter the cumulative displacement back to nodes for
        # convenient post-analysis access (mirrors LinearStaticAnalysis).
        for node in m.nodes.values():
            for d in range(node.ndf):
                eq = int(node.eqn[d])
                if eq >= 0:
                    node.disp[d] = float(u_cum[eq])
        return StagedConstructionResult(
            n_stages=len(self.stages),
            stage_names=names,
            u_incremental=u_increments,
            u_cumulative=u_cum,
            creep_factors=creep_factors,
        )


# ============================================================ incremental staged (birth + death)

@dataclass
class ErectionStage:
    """One stage of an *incremental* staged-erection analysis.

    Unlike :class:`ConstructionStage` (which feeds the simplified
    scalar-EMM driver), this stage description drives a true
    active-set incremental analysis with element **birth** and
    **death**.

    Attributes
    ----------
    name : str
        Label for reporting.
    add_elements : list, optional
        Element tags that become active ("born") at the START of this
        stage. A born element is installed **stress-free in the current
        deformed geometry** -- it accumulates force only from load
        increments applied in this and later stages.
    remove_elements : list, optional
        Element tags removed ("die") at the start of this stage
        (falsework, temporary props, erection equipment, demolished
        segments). The locked-in internal force of each removed element
        is **released** onto the remaining structure (the exact
        equilibrium release load ``K_e u_e``).
    loads : dict, optional
        ``{node_tag: [Fx, Fy, ...]}`` nodal loads applied in this stage
        (incremental -- added to the running total).
    stiffness_factor : float, default 1.0
        Optional uniform scale on the active stiffness for this stage's
        solve (e.g. an EMM creep factor ``1/(1+chi*phi)`` for a
        long-duration sustained-load stage). Default 1.0 = elastic.
    """

    name: str
    add_elements: list = field(default_factory=list)
    remove_elements: list = field(default_factory=list)
    loads: dict = field(default_factory=dict)
    stiffness_factor: float = 1.0


@dataclass
class IncrementalStagedResult:
    """Outcome of an :class:`IncrementalStagedAnalysis`.

    Attributes
    ----------
    stage_names : list[str]
    u_increments : list[np.ndarray]
        Per-stage free-DOF displacement increment (size neq).
    u_cumulative : np.ndarray
        Final cumulative displacement (size neq).
    element_forces : dict[int, np.ndarray]
        Final locked-in global end-force vector per element that is
        active at the end (``K_e u_e^active``). Removed elements are
        absent.
    element_force_history : dict[int, list]
        Per element, the end-force vector at the end of each stage
        (``None`` for stages where the element was inactive).
    active_history : list[set]
        Set of active element tags at the end of each stage.
    """

    stage_names: list
    u_increments: list
    u_cumulative: np.ndarray
    element_forces: dict
    element_force_history: dict
    active_history: list


class IncrementalStagedAnalysis:
    """True incremental staged-construction analysis with element
    birth and death.

    The model must contain **all** elements that appear in any stage;
    the driver activates / deactivates them per the stage script. At
    each stage it:

    1. Updates the active element set (add births, remove deaths).
    2. For each **dying** element, adds its current locked-in force
       ``K_e u_e`` to the stage release load (so the remaining
       structure picks up what the removed element was carrying).
    3. Assembles the stiffness from the **active elements only**,
       restricted to the DOFs those elements touch (so DOFs belonging
       only to not-yet-born / removed elements stay out of the solve).
    4. Solves for the displacement increment under (stage loads +
       release loads), accumulates displacement, and accumulates each
       active element's force by ``f_e += K_e du_e`` -- which makes a
       newly-born element stress-free with respect to all deformation
       that occurred before its birth.

    Parameters
    ----------
    model : Model
        Contains every element used across the stages, plus supports.
    stages : list[ErectionStage]
    initial_active : list, optional
        Element tags active before stage 1 (default: empty -- every
        element must be born by some stage's ``add_elements``).

    Notes
    -----
    Linear, small-displacement. MP constraints (rigid links /
    diaphragms) are not yet supported by this driver.
    """

    def __init__(self, model, stages, *, initial_active=None):
        if not stages:
            raise ValueError("at least one stage required")
        if model.mp_constraints:
            raise NotImplementedError(
                "IncrementalStagedAnalysis does not yet support MP "
                "constraints (rigid links / diaphragms)."
            )
        self.model = model
        self.stages = list(stages)
        self.initial_active = set(initial_active or [])

    def _element_dofmaps(self):
        m = self.model
        return {e.tag: m.element_dof_map(e) for e in m.elements.values()}

    def run(self) -> IncrementalStagedResult:
        m = self.model
        m.reset_results()
        m.number_dofs()
        neq = m.neq
        if neq == 0:
            raise RuntimeError("model has no free DOFs")

        dofmaps = self._element_dofmaps()
        active = set(self.initial_active)
        # locked-in global end forces per active element
        f_elem: dict[int, np.ndarray] = {
            tag: np.zeros(dofmaps[tag].size) for tag in active
        }

        u_cum = np.zeros(neq)
        u_increments = []
        names = []
        force_history = {e.tag: [] for e in m.elements.values()}
        active_history = []

        for stage in self.stages:
            # ---- birth / death --------------------------------------
            release = np.zeros(neq)
            for tag in stage.remove_elements:
                if tag not in active:
                    raise ValueError(
                        f"stage '{stage.name}': cannot remove element "
                        f"{tag} -- it is not active"
                    )
                # release the dying element's locked-in force
                fe = f_elem[tag]
                dm = dofmaps[tag]
                for i, eq in enumerate(dm):
                    if eq >= 0:
                        release[eq] += fe[i]
                active.discard(tag)
                f_elem.pop(tag, None)
            for tag in stage.add_elements:
                if tag not in m.elements:
                    raise ValueError(
                        f"stage '{stage.name}': unknown element {tag}"
                    )
                active.add(tag)
                f_elem[tag] = np.zeros(dofmaps[tag].size)  # born stress-free

            if not active:
                raise RuntimeError(
                    f"stage '{stage.name}': no active elements to solve"
                )

            # ---- assemble active stiffness + active free DOF set -----
            active_tags = sorted(active)
            Ke_cache = {tag: m.element(tag).K_global() for tag in active_tags}
            free_eqs = set()
            for tag in active_tags:
                for eq in dofmaps[tag]:
                    if eq >= 0:
                        free_eqs.add(int(eq))
            ix = np.array(sorted(free_eqs), dtype=int)
            if ix.size == 0:
                raise RuntimeError(
                    f"stage '{stage.name}': active structure has no free DOFs"
                )
            g2l = {int(g): l for l, g in enumerate(ix)}

            rows, cols, vals = [], [], []
            for tag in active_tags:
                dm = dofmaps[tag]
                Ke = Ke_cache[tag] * stage.stiffness_factor
                ndof = dm.size
                for a in range(ndof):
                    ga = dm[a]
                    if ga < 0:
                        continue
                    la = g2l[int(ga)]
                    for b in range(ndof):
                        gb = dm[b]
                        if gb < 0:
                            continue
                        rows.append(la)
                        cols.append(g2l[int(gb)])
                        vals.append(Ke[a, b])
            n = ix.size
            K_aa = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()

            # ---- stage load vector (external + release) --------------
            f_full = release.copy()
            for node_tag, load in stage.loads.items():
                node = m.node(node_tag)
                ld = np.asarray(load, dtype=float)
                for d in range(min(node.ndf, ld.size)):
                    eq = int(node.eqn[d])
                    if eq >= 0:
                        f_full[eq] += ld[d]
            f_a = f_full[ix]

            # ---- solve increment -------------------------------------
            try:
                du_a = np.asarray(spsolve(K_aa, f_a)).ravel()
            except Exception as exc:
                raise RuntimeError(
                    f"stage '{stage.name}' solve failed: {exc} (the active "
                    "sub-structure may be a mechanism -- check supports / "
                    "connectivity of the elements active this stage)."
                ) from exc
            if not np.all(np.isfinite(du_a)):
                raise RuntimeError(
                    f"stage '{stage.name}': non-finite increment (active "
                    "sub-structure is singular / a mechanism)."
                )
            du = np.zeros(neq)
            du[ix] = du_a
            u_cum = u_cum + du
            u_increments.append(du)

            # ---- accumulate element forces ---------------------------
            for tag in active_tags:
                dm = dofmaps[tag]
                du_e = np.array([du[eq] if eq >= 0 else 0.0 for eq in dm])
                f_elem[tag] = f_elem[tag] + Ke_cache[tag] @ du_e

            # ---- record -------------------------------------------------
            names.append(stage.name)
            active_history.append(set(active))
            for tag in force_history:
                if tag in active:
                    force_history[tag].append(f_elem[tag].copy())
                else:
                    force_history[tag].append(None)

        # scatter final displacement to nodes
        for node in m.nodes.values():
            for d in range(node.ndf):
                eq = int(node.eqn[d])
                node.disp[d] = float(u_cum[eq]) if eq >= 0 else 0.0

        return IncrementalStagedResult(
            stage_names=names,
            u_increments=u_increments,
            u_cumulative=u_cum,
            element_forces={t: f.copy() for t, f in f_elem.items()},
            element_force_history=force_history,
            active_history=active_history,
        )


# ============================================================ EMM helper

def effective_modulus_EMM(
    *,
    E_c: float,
    phi: float,
    chi: float = 0.8,
) -> float:
    """Effective Modulus Method::

        E_eff = E_c / (1 + chi · phi)

    Parameters
    ----------
    E_c : float
        Concrete elastic modulus at the age of loading (Pa).
    phi : float
        Creep coefficient phi(t, t_0).
    chi : float, default 0.8
        Ageing coefficient.

    Returns
    -------
    E_eff : float
    """
    if E_c <= 0.0:
        raise ValueError("E_c must be > 0")
    if phi < 0.0:
        raise ValueError("phi must be >= 0")
    if not (0.0 < chi <= 1.0):
        raise ValueError("chi must be in (0, 1]")
    return float(E_c / (1.0 + chi * phi))
