"""Persisted nonlinear-run history (plan §16 G-S2).

A nonlinear run (pushover / case / time-history) is *expensive* and tied to one
analysis, so — unlike the cheap-to-recompute linear results the :class:`Project`
deliberately never stores — its **curve is worth saving**. This is what SAP2000
and Midas do: the nonlinear results live with the model.

:class:`RunRecord` is the lean, JSON-safe unit that does it: the base-shear vs
control-displacement curve (or the response history), a few summary scalars, and
the ASCE 41 first-reach milestones — everything needed to *re-view and compare*
a run after save/load, without the heavy per-step fiber/shape frames (those stay
session-only; re-run to scrub the model again). The project carries a list of
them (``Project.runs``); the Run-history dialog lists and re-plots them.

Pure data (no Qt, no numpy in the stored form), so it round-trips through the
project JSON and is testable headless.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime

from nl_results import NonlinearResults

_LEVELS = ("IO", "LS", "CP")


@dataclass
class RunRecord:
    """One saved nonlinear run: its curve + summary (plan G-S2).

    ``x`` / ``y`` are the curve samples — control displacement vs base shear for
    a pushover/cyclic run, or time vs response for a time-history. ``peak`` /
    ``peak_x`` are the signed largest-magnitude ``y`` and its ``x``.
    ``milestones`` maps an ASCE 41 level (``"IO"``/``"LS"``/``"CP"``) to the
    control displacement at which the model first reached it. ``meta`` holds
    small descriptive strings (control node/DOF, units) for display only.
    """
    id: int
    name: str
    kind: str = "pushover"                 # pushover | cyclic | case | time_history
    created: str = ""                      # ISO-8601 (local)
    protocol: str = "monotonic"
    x_label: str = "displacement"
    y_label: str = "base shear"
    x: list = field(default_factory=list)
    y: list = field(default_factory=list)
    peak: float = 0.0
    peak_x: float = 0.0
    milestones: dict = field(default_factory=dict)   # {"IO": disp, ...}
    meta: dict = field(default_factory=dict)          # {str: str}

    # ------------------------------------------------------------- builders
    @classmethod
    def from_results(cls, results: NonlinearResults, *, id: int, name: str,
                     kind: str = "pushover", x_label: str = "displacement",
                     y_label: str = "base shear", meta: dict | None = None,
                     created: str | None = None) -> "RunRecord":
        """Build a record from a :class:`NonlinearResults` (the curve + its
        summary + acceptance milestones), coercing every number to a plain
        ``float`` so the record is JSON-safe regardless of numpy inputs."""
        x, y = results.curve()
        ms: dict = {}
        for lvl in _LEVELS:
            m = results.accept_milestones.get(lvl)
            if isinstance(m, dict) and "disp" in m:
                ms[lvl] = float(m["disp"])
        return cls(
            id=int(id), name=str(name), kind=str(kind),
            created=created or datetime.now().isoformat(timespec="seconds"),
            protocol=str(results.protocol), x_label=str(x_label),
            y_label=str(y_label),
            x=[float(v) for v in x], y=[float(v) for v in y],
            peak=float(results.peak_shear()), peak_x=float(results.disp_at_peak()),
            milestones=ms, meta={str(k): str(v) for k, v in (meta or {}).items()})

    @classmethod
    def from_time_history(cls, result: dict, *, id: int, name: str,
                          meta: dict | None = None,
                          created: str | None = None) -> "RunRecord":
        """Build a record from a ``run_time_history`` result dict (time vs the
        monitored displacement response)."""
        t = [float(v) for v in result.get("times", [])]
        d = [float(v) for v in result.get("disp", [])]
        peak = max(d, key=abs) if d else 0.0
        peak_x = t[d.index(peak)] if d and peak in d else 0.0
        return cls(
            id=int(id), name=str(name), kind="time_history",
            created=created or datetime.now().isoformat(timespec="seconds"),
            protocol="time_history", x_label="time", y_label="displacement",
            x=t, y=d, peak=float(peak), peak_x=float(peak_x),
            meta={str(k): str(v) for k, v in (meta or {}).items()})

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        """Construct from a stored dict, ignoring unknown keys (forward-compat)."""
        fields = set(cls.__dataclass_fields__)
        kept = {k: v for k, v in d.items() if k in fields}
        kept["milestones"] = {str(k): float(v)
                              for k, v in (kept.get("milestones") or {}).items()}
        kept["meta"] = {str(k): str(v)
                        for k, v in (kept.get("meta") or {}).items()}
        kept["x"] = [float(v) for v in (kept.get("x") or [])]
        kept["y"] = [float(v) for v in (kept.get("y") or [])]
        return cls(**kept)

    def to_dict(self) -> dict:
        return asdict(self)

    # ------------------------------------------------------------- queries
    def results(self) -> NonlinearResults:
        """A curve-only :class:`NonlinearResults` for (re-)plotting / comparison —
        no per-step frames (those are not persisted)."""
        r = NonlinearResults(self.x, self.y, protocol=self.protocol)
        if self.milestones:
            r.accept_milestones = {lvl: {"disp": d}
                                   for lvl, d in self.milestones.items()}
        return r

    def summary(self) -> str:
        """A one-line human summary (peak + governing acceptance level)."""
        lvl = next((l for l in reversed(_LEVELS) if l in self.milestones), None)
        tail = f" · reached {lvl}" if lvl else ""
        return (f"{self.y_label} peak {self.peak:.4g} @ {self.x_label} "
                f"{self.peak_x:.4g}{tail}")


def next_run_id(runs) -> int:
    return max((r.id for r in runs), default=0) + 1
