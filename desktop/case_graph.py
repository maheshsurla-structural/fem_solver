"""Case dependency graph (E4) — the data behind the Load Case Tree.

Analysis cases chain through two links, both pointing at a Nonlinear Static case:

* a :class:`project.NonlinearCase`'s ``continue_from`` — staged pushover
  continuation; and
* any case's ``initial_condition = ("state", id)`` (E2) — start from that
  nonlinear case's committed state.

This module turns those links into a dependency graph the tree dialog renders
(:mod:`case_tree_dialog`), and it surfaces the two things a chain can get wrong:
a **dangling** reference (the source case was deleted) and a **cycle**. Pure
Python — no Qt — so it is unit-testable headless.
"""
from __future__ import annotations


def _parent_id(case, kind: str):
    """The nonlinear-case id ``case`` depends on, or ``None``. ``continue_from``
    (the pushover chain) takes precedence; otherwise a state initial condition."""
    if kind == "nonlinear":
        cf = getattr(case, "continue_from", None)
        if cf is not None:
            return int(cf)
    ic = tuple(getattr(case, "initial_condition", ("zero",)) or ("zero",))
    if len(ic) >= 2 and ic[0] == "state" and ic[1] is not None:
        return int(ic[1])
    return None


def case_nodes(project) -> list[dict]:
    """One node per stored case (nonlinear cases then saved analysis cases).

    Each is a dict with ``key`` (``(kind, id)``), ``kind``, ``id``, ``name``,
    ``type`` (display label), ``parent`` (``("nonlinear", id)`` or ``None``) and
    ``dangling`` (the parent id names no existing nonlinear case)."""
    import case_types
    nl_ids = {c.id for c in project.nonlinear_cases}
    out: list[dict] = []
    for c in project.nonlinear_cases:
        pid = _parent_id(c, "nonlinear")
        out.append({"key": ("nonlinear", c.id), "kind": "nonlinear", "id": c.id,
                    "name": c.name, "type": "Nonlinear Static",
                    "parent": (("nonlinear", pid) if pid is not None else None),
                    "dangling": pid is not None and pid not in nl_ids})
    for c in getattr(project, "analysis_cases", []):
        ct = case_types.get(c.type)
        pid = _parent_id(c, "analysis")
        out.append({"key": ("analysis", c.id), "kind": "analysis", "id": c.id,
                    "name": c.name, "type": (ct.type_label if ct else c.type),
                    "parent": (("nonlinear", pid) if pid is not None else None),
                    "dangling": pid is not None and pid not in nl_ids})
    for n in out:                              # E5c/E4: stale = source ran later
        n["stale"] = _is_stale(project, n["key"], n["parent"])
    return out


def _is_stale(project, key, parent) -> bool:
    """True when the case's source ran *after* the case last ran — so the case's
    result is out of date. Needs both run times (session status, E5c); unknown
    when either hasn't run."""
    if parent is None or not hasattr(project, "case_status"):
        return False
    ns = project.case_status(key)
    ps = project.case_status(parent)
    if not (ns and ps and ns.get("when") and ps.get("when")):
        return False
    return ps["when"] > ns["when"]


def case_forest(project):
    """``(roots, cycle_keys)``.

    ``roots`` is a nested list of ``{**node, "children": [...]}`` where a node's
    children are the cases that depend on it — rooted at cases with no parent (or
    a dangling one). Cases caught in a dependency cycle cannot be nested (that
    would recurse forever): their keys are returned in ``cycle_keys`` and they
    are appended as flat pseudo-roots (no children) so the view can still list
    and flag them."""
    nodes = {n["key"]: n for n in case_nodes(project)}
    children: dict = {}
    for key, n in nodes.items():
        p = n["parent"]
        if p is not None and p in nodes and not n["dangling"]:
            children.setdefault(p, []).append(key)

    def build(key, path):
        if key in path:                       # cycle — do not descend
            return None
        node = dict(nodes[key])
        node["children"] = [t for c in children.get(key, [])
                            if (t := build(c, path | {key})) is not None]
        return node

    roots = []
    for key, n in nodes.items():
        if n["parent"] is None or n["dangling"] or n["parent"] not in nodes:
            t = build(key, frozenset())
            if t is not None:
                roots.append(t)

    reached: set = set()

    def mark(t):
        reached.add(t["key"])
        for ch in t["children"]:
            mark(ch)
    for r in roots:
        mark(r)

    cycle_keys = [k for k in nodes if k not in reached]
    for k in cycle_keys:                       # list cycle nodes flat, no nesting
        flat = dict(nodes[k])
        flat["children"] = []
        roots.append(flat)
    return roots, cycle_keys
