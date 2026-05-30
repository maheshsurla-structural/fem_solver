"""Model QA / sanity checker.

Runs a battery of model-integrity checks and returns a structured
report of warnings. Categories:

* **ERROR** -- the model is structurally inconsistent (e.g., orphan
  nodes, no constraints) and analysis will fail or give wrong
  answers.
* **WARNING** -- non-fatal anomaly that the engineer should review
  (e.g., very high aspect ratio, unusual property value).
* **INFO** -- inventory: total nodes, elements, free DOFs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class QaWarning:
    category: str          # "ERROR", "WARNING", "INFO"
    message: str
    affected: list = field(default_factory=list)


@dataclass
class QaReport:
    warnings: list = field(default_factory=list)

    @property
    def errors(self) -> list:
        return [w for w in self.warnings if w.category == "ERROR"]

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def __str__(self) -> str:
        lines = []
        by_cat = {"ERROR": [], "WARNING": [], "INFO": []}
        for w in self.warnings:
            by_cat.setdefault(w.category, []).append(w)
        for cat in ("ERROR", "WARNING", "INFO"):
            for w in by_cat[cat]:
                tag = f"[{cat}]"
                detail = (f" (affected: {w.affected[:5]}"
                          + (", ..." if len(w.affected) > 5 else "")
                          + ")" if w.affected else "")
                lines.append(f"{tag} {w.message}{detail}")
        if not lines:
            lines = ["[OK] no issues detected"]
        return "\n".join(lines)


def _collect_used_node_tags(model) -> set:
    used = set()
    for e in model.elements.values():
        for t in e.node_tags:
            used.add(int(t))
    # Also include MP-constrained nodes
    for c in getattr(model, "mp_constraints", []) or []:
        for bc in c.basic_constraints(model):
            used.add(int(bc.c_node))
            for (rn, _, _) in bc.r_terms:
                used.add(int(rn))
    return used


def run_qa_checks(model) -> QaReport:
    """Run the standard battery of QA checks on a :class:`Model`."""
    rep = QaReport()

    # Inventory
    n_nodes = len(getattr(model, "_nodes", {}))
    n_elem = len(getattr(model, "_elements", {}))
    rep.warnings.append(QaWarning(
        "INFO",
        f"Model has {n_nodes} nodes and {n_elem} elements",
    ))

    # ERROR: zero nodes / zero elements
    if n_nodes == 0:
        rep.warnings.append(QaWarning("ERROR", "Model has no nodes"))
    if n_elem == 0:
        rep.warnings.append(QaWarning("ERROR", "Model has no elements"))

    # ERROR: no fixities defined anywhere
    fixed_any = False
    for n in model.nodes.values():
        if any(int(f) == 1 for f in n.fixity):
            fixed_any = True
            break
    if not fixed_any and n_nodes > 0:
        rep.warnings.append(QaWarning(
            "ERROR",
            "No fixities defined -- the model has no supports; "
            "linear-static analysis will produce a singular system",
        ))

    # WARNING: orphan nodes (no element / constraint references them)
    used = _collect_used_node_tags(model)
    orphans = sorted(
        n.tag for n in model.nodes.values()
        if int(n.tag) not in used
        and not any(int(f) == 1 for f in n.fixity)
    )
    if orphans:
        rep.warnings.append(QaWarning(
            "WARNING",
            f"{len(orphans)} orphan node(s) -- referenced by no element or "
            "fixity and not constrained",
            affected=orphans,
        ))

    # WARNING: zero-length / near-zero-length line elements
    bad_len = []
    for e in model.elements.values():
        tags = e.node_tags
        if len(tags) == 2:
            n1 = model.node(tags[0]).coords
            n2 = model.node(tags[1]).coords
            dist = float(np.linalg.norm(np.asarray(n2) - np.asarray(n1)))
            if dist < 1.0e-9:
                bad_len.append(e.tag)
    if bad_len:
        rep.warnings.append(QaWarning(
            "WARNING",
            f"{len(bad_len)} line element(s) with zero length",
            affected=bad_len,
        ))

    # WARNING: duplicate (parallel, coincident) elements
    seen = {}
    duplicates = []
    for e in model.elements.values():
        tags = e.node_tags
        if len(tags) == 2:
            key = tuple(sorted([int(tags[0]), int(tags[1])]))
            if key in seen:
                duplicates.append((seen[key], e.tag))
            else:
                seen[key] = e.tag
    if duplicates:
        rep.warnings.append(QaWarning(
            "WARNING",
            f"{len(duplicates)} pair(s) of duplicate elements "
            "(same end nodes)",
            affected=[f"{a}/{b}" for a, b in duplicates[:10]],
        ))

    # WARNING: very large coordinate spread (indicating unit-mix bugs)
    if n_nodes > 1:
        coords = np.array(
            [n.coords[:model.ndm] for n in model.nodes.values()],
            dtype=float,
        )
        diag = float(np.linalg.norm(coords.max(0) - coords.min(0)))
        if diag > 1.0e4:
            rep.warnings.append(QaWarning(
                "WARNING",
                f"Model diagonal extent = {diag:.0f} (very large; check "
                "units: should typically be meters not millimeters)",
            ))
        elif diag < 1.0e-2:
            rep.warnings.append(QaWarning(
                "WARNING",
                f"Model diagonal extent = {diag:.2e} (very small; check "
                "units)",
            ))

    return rep
