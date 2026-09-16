"""Staged (sequential) static analysis (plan G6).

:class:`StagedAnalysis` runs a sequence of static stages on **one** model,
each continuing from the previous stage's committed state -- the "apply the
axial preload and hold it, then push laterally" workflow of the fiber-hinge
benchmark (plan §2.6). It is the femsolver counterpart of an OpenSees
``loadConst`` between load patterns / a CSI staged construction case.

Between stages it:

1. keeps the deformed node state and every element's committed material history
   (the next stage's :class:`NonlinearStaticAnalysis` runs with
   ``keep_state=True``, so displacements are not zeroed);
2. folds the just-converged stage's applied load into a **constant** baseline
   force carried into all later stages (via the integrator's
   ``set_constant_force``), so a preload stays applied at full value while a
   later stage is displacement-controlled;
3. clears the model's nodal load pattern so the next stage defines only its own
   incremental loads.

Because the constant baseline holds prior loads, a later stage's load factor
scales *only* that stage's pattern -- e.g. displacement control on the lateral
DOF drives the lateral load while the axial preload is held.

Usage::

    sa = StagedAnalysis(model)
    sa.add_stage("axial", lambda m: (
        m.add_nodal_load(2, [-2400.0, 0.0, 0.0]),
        NonlinearStaticAnalysis(m, num_steps=10, dlambda=0.1,
                                integrator="load_control"))[1])
    sa.add_stage("push", lambda m: (
        m.add_nodal_load(2, [0.0, -1.0, 0.0]),
        NonlinearStaticAnalysis(m, num_steps=40,
                                integrator=DisplacementControl(2, 1, -0.02),
                                track=(2, 1)))[1])
    out = sa.run()          # {"axial": {...}, "push": {...}}
"""
from __future__ import annotations

import numpy as np

from femsolver.analysis.assembler import assemble_force
from femsolver.analysis.nonlinear_static import NonlinearStaticAnalysis


class StagedAnalysis:
    """Run a sequence of static stages on one model, holding earlier stages'
    loads constant and continuing from their committed state.

    Parameters
    ----------
    model :
        The model. Stages mutate its loads and state in place; after
        :meth:`run` it is left at the final stage's committed state.
    """

    def __init__(self, model):
        self.model = model
        self._stages: list[tuple[str, object]] = []
        # After run(): the eqn-space constant-force vector held into a
        # (hypothetical) further stage — the sum of every stage's converged
        # applied load. Exposed so a downstream analysis seeded from this
        # committed state can optionally hold these loads (E2). None until run.
        self.const_force_final = None

    def add_stage(self, name: str, analysis_factory) -> "StagedAnalysis":
        """Add a stage. ``analysis_factory(model)`` must set the stage's
        incremental nodal loads on ``model`` and return a configured
        :class:`NonlinearStaticAnalysis`. Returns ``self`` for chaining."""
        self._stages.append((str(name), analysis_factory))
        return self

    def run(self) -> dict:
        """Run every stage in order. Returns ``{stage_name: analysis_result}``
        (the dict each stage's ``NonlinearStaticAnalysis.run`` returns)."""
        if not self._stages:
            raise RuntimeError("StagedAnalysis has no stages")
        results: dict[str, dict] = {}
        F_const: np.ndarray | None = None
        for i, (name, factory) in enumerate(self._stages):
            analysis = factory(self.model)
            if not isinstance(analysis, NonlinearStaticAnalysis):
                raise TypeError(
                    f"stage {name!r} factory must return a "
                    f"NonlinearStaticAnalysis, got {type(analysis).__name__}")
            analysis.keep_state = i > 0        # continue from prior state
            analysis.const_force = F_const     # hold prior stages' loads
            results[name] = analysis.run()
            # fold THIS stage's converged applied load into the baseline held
            # constant for later stages (numbering is stable across stages, so
            # the eqn-space vector stays valid).
            applied = analysis.integrator.lambd * assemble_force(self.model)
            F_const = applied if F_const is None else F_const + applied
            # clear the load pattern so the next stage defines its own
            for node in self.model.nodes.values():
                node._load[:] = 0.0
        self.const_force_final = F_const
        return results
