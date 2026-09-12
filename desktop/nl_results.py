"""Step-indexed nonlinear-results model (plan GUI-I1).

A nonlinear run (``nonlinear.run_pushover`` / ``run_case``) returns several
per-step arrays -- the base-shear/displacement curve plus, when capture is on,
a per-step monitored-section fiber snapshot, whole-model nodal deformation, and
per-member peak fiber strain (hinge state). Before this, every view reached
into those parallel lists by index. :class:`NonlinearResults` consolidates them
into one queryable model so any view reads **state at step k** the same way::

    results = NonlinearResults.from_run(run_pushover(...))
    st = results.step(k)          # -> StepState(disp, shear, node_disp, ...)

It is pure data (no Qt), so it is testable headless and shared by the pushover
dialog, the report/export path, and future views (e.g. scrubbing the main
model view). ``from_run`` / ``to_dict`` round-trip the run dict, so the export
and report code keep working unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StepState:
    """The model's state at one push step.

    ``node_disp``/``member_damage``/``fibers`` are ``None`` when that quantity
    was not captured (the run did not request fiber/shape recording).
    """
    index: int
    disp: float
    shear: float
    node_disp: dict | None       # {node_id: (dx, dy)}
    member_damage: dict | None   # {member_id: peak |fiber strain|}
    fibers: list | None          # [(y, z, sigma, eps), ...] monitored section


class NonlinearResults:
    """Step-indexed view over a nonlinear run's output.

    Parameters mirror the run dict's keys. ``disp``/``shear`` are the
    base-shear vs control-displacement curve (signed for cyclic runs). The
    optional frame lists are per-step captures, aligned one-to-one with the
    curve points.
    """

    def __init__(self, disp, shear, *, protocol: str = "monotonic",
                 fiber_frames=None, shape_frames=None, damage_frames=None):
        self.disp = [float(x) for x in (disp or [])]
        self.shear = [float(x) for x in (shear or [])]
        self.protocol = str(protocol)
        self._fibers = list(fiber_frames) if fiber_frames else []
        self._shape = list(shape_frames) if shape_frames else []
        self._damage = list(damage_frames) if damage_frames else []

    # ------------------------------------------------------------- builders
    @classmethod
    def from_run(cls, result: dict) -> "NonlinearResults":
        """Wrap the dict returned by ``run_pushover`` / ``run_case``."""
        return cls(
            result.get("disp", []), result.get("shear", []),
            protocol=result.get("protocol", "monotonic"),
            fiber_frames=result.get("fiber_frames"),
            shape_frames=result.get("shape_frames"),
            damage_frames=result.get("damage_frames"))

    def to_dict(self) -> dict:
        """Reconstruct the run-dict form (for the export / report path)."""
        d: dict = {"disp": list(self.disp), "shear": list(self.shear),
                   "protocol": self.protocol}
        if self._fibers:
            d["fiber_frames"] = self._fibers
        if self._shape:
            d["shape_frames"] = self._shape
            d["damage_frames"] = self._damage
        return d

    # ------------------------------------------------------------- queries
    @property
    def n_curve(self) -> int:
        return len(self.disp)

    @property
    def n_steps(self) -> int:
        """Number of scrubbable steps (captured post-processing frames)."""
        return max(len(self._fibers), len(self._shape))

    @property
    def has_fibers(self) -> bool:
        return bool(self._fibers)

    @property
    def has_shape(self) -> bool:
        return bool(self._shape)

    def curve(self) -> tuple[list, list]:
        return list(self.disp), list(self.shear)

    def step(self, k: int) -> StepState:
        """State at step ``k`` (0-based). Missing captures come back as
        ``None`` on the :class:`StepState`."""
        n = max(self.n_curve, self.n_steps)
        if n == 0:
            raise IndexError("results have no steps")
        if not (0 <= k < n):
            raise IndexError(f"step {k} out of range [0, {n})")
        return StepState(
            index=k,
            disp=self.disp[k] if k < len(self.disp) else float("nan"),
            shear=self.shear[k] if k < len(self.shear) else float("nan"),
            node_disp=self._shape[k] if k < len(self._shape) else None,
            member_damage=self._damage[k] if k < len(self._damage) else None,
            fibers=self._fibers[k] if k < len(self._fibers) else None)

    def __len__(self) -> int:
        return max(self.n_curve, self.n_steps)

    # ------------------------------------------------------------- summary
    def peak_shear(self) -> float:
        """Signed base shear of largest magnitude (0 if empty)."""
        return max(self.shear, key=abs) if self.shear else 0.0

    def disp_at_peak(self) -> float:
        if not self.shear:
            return 0.0
        i = max(range(len(self.shear)), key=lambda j: abs(self.shear[j]))
        return self.disp[i] if i < len(self.disp) else float("nan")

    def max_abs_disp(self) -> float:
        return max((abs(d) for d in self.disp), default=0.0)
