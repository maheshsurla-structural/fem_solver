"""JSON serialization for :class:`Section` (Theme II.8).

Section round-trip preserves identity, geometry, reinforcement, and
material descriptors. Materials themselves are stored as opaque
dictionaries with a ``class`` tag and (where possible) a kwargs
mapping that lets the user re-instantiate them on load.

Trade-off note
--------------
femsolver's material objects are deliberately flexible -- they range
from plain dataclasses (``ConcreteMaterial`` with ``fc_prime``, ``fy``)
to stateful uniaxial materials (``ConcreteKentPark``,
:class:`UniaxialBilinear`, ...) to high-level descriptors. Persisting
ALL material state through JSON would couple this module to every
material class in the codebase.

The chosen compromise:

* The geometry, identity, reinforcement layout, and catalogue
  reference round-trip *exactly*.
* Material zones round-trip their **class name** and any attributes
  that are basic Python types (int, float, str, bool, list of same).
* On load, if the material's class is importable, we attempt to
  reconstruct it via its ``__init__``; if that fails, the zone's
  ``material`` slot is set to ``None`` and the load proceeds.

For round-trip-critical workflows (e.g. nonlinear analyses), users
should attach their own materials after loading.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any

from femsolver.sections.geometry.polygon import PolygonGeometry
from femsolver.sections.section import (
    MaterialZone,
    RebarBar,
    ReinforcementLayout,
    Section,
)


# ============================================================ encoders

def _encode_material(material: Any) -> dict | None:
    """Best-effort material encoding. Stores the class name + any
    JSON-friendly attributes. Returns ``None`` for ``None``."""
    if material is None:
        return None
    cls = type(material)
    blob: dict = {
        "class": f"{cls.__module__}.{cls.__qualname__}",
    }
    # Try dataclass asdict first (lossless for dataclasses with
    # JSON-friendly fields)
    if is_dataclass(material) and not isinstance(material, type):
        try:
            blob["args"] = {
                k: v for k, v in asdict(material).items()
                if _is_json_simple(v)
            }
            return blob
        except Exception:
            pass
    # Otherwise pull JSON-simple attributes
    args: dict[str, Any] = {}
    for k in dir(material):
        if k.startswith("_"):
            continue
        try:
            v = getattr(material, k)
        except Exception:
            continue
        if callable(v):
            continue
        if _is_json_simple(v):
            args[k] = v
    if args:
        blob["args"] = args
    return blob


def _is_json_simple(v: Any) -> bool:
    if v is None or isinstance(v, (bool, int, float, str)):
        return True
    if isinstance(v, (list, tuple)):
        return all(_is_json_simple(x) for x in v)
    if isinstance(v, dict):
        return (all(isinstance(k, str) for k in v.keys())
                and all(_is_json_simple(x) for x in v.values()))
    return False


def _encode_geometry(geom) -> dict:
    """Encode a Geometry to a JSON-friendly dict via its polygon
    coordinates."""
    polygon = geom.polygon
    ext = list(polygon.exterior.coords)
    if len(ext) > 1 and ext[0] == ext[-1]:
        ext = ext[:-1]
    holes = []
    for ring in polygon.interiors:
        h = list(ring.coords)
        if len(h) > 1 and h[0] == h[-1]:
            h = h[:-1]
        holes.append([list(c) for c in h])
    return {
        "type": type(geom).__name__,
        "exterior": [list(c) for c in ext],
        "holes": holes,
        # Carry the catalogue-overridden gross properties so they
        # survive round-trip (otherwise the load would recompute them
        # from the polygon and lose any catalogue precision).
        "area": float(geom.area),
        "I_zz": float(geom.I_zz),
        "I_yy": float(geom.I_yy),
        "J": float(geom.J),
    }


def _encode_reinforcement(rl: ReinforcementLayout | None) -> dict | None:
    if rl is None:
        return None
    return {
        "bars": [
            {
                "z": b.z, "y": b.y, "area": b.area,
                "designation": b.designation,
                "material": _encode_material(b.material),
            }
            for b in rl.bars
        ],
        "stirrup_designation": rl.stirrup_designation,
        "stirrup_spacing": rl.stirrup_spacing,
        "stirrup_legs": rl.stirrup_legs,
    }


def section_to_dict(section: Section) -> dict:
    """Encode a :class:`Section` to a JSON-friendly dict."""
    return {
        "femsolver_section_version": 1,
        "name": section.name,
        "family": section.family,
        "catalogue_ref": section.catalogue_ref,
        "geometry": _encode_geometry(section.geometry),
        "zones": [
            {
                "name": z.name,
                "material": _encode_material(z.material),
                # Sub-geometries on zones are not yet round-tripped
                # (composite sections often share the parent geometry);
                # the load reconstructs them as None.
            }
            for z in section.zones
        ],
        "reinforcement": _encode_reinforcement(section.reinforcement),
    }


def section_to_json(section: Section, *, indent: int | None = 2) -> str:
    return json.dumps(section_to_dict(section), indent=indent)


# ============================================================ decoders

def _decode_material(blob: dict | None) -> Any:
    """Best-effort material reconstruction. Returns ``None`` if the
    class can't be imported or the kwargs don't match its __init__."""
    if blob is None:
        return None
    cls_path = blob.get("class")
    if not cls_path or "." not in cls_path:
        return None
    mod_path, cls_name = cls_path.rsplit(".", 1)
    try:
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
    except (ImportError, AttributeError):
        return None
    args = blob.get("args", {}) or {}
    try:
        return cls(**args)
    except TypeError:
        # Constructor signature mismatch -- give up gracefully.
        return None


def _decode_geometry(blob: dict):
    """Reconstruct a Geometry. For catalogued sections, wrap the
    polygon in a :class:`CataloguedGeometry` to preserve the
    overridden A/I/J values."""
    from femsolver.sections.catalogue.geometry import CataloguedGeometry

    exterior = [tuple(c) for c in blob["exterior"]]
    holes = [
        [tuple(c) for c in h] for h in blob.get("holes", [])
    ]
    base = PolygonGeometry(exterior, holes=holes if holes else None)
    # If the recorded properties differ meaningfully from the polygon-
    # computed ones, wrap in a CataloguedGeometry override.
    recorded_A = float(blob.get("area", base.area))
    rel_diff_A = abs(recorded_A - base.area) / max(base.area, 1e-30)
    if rel_diff_A > 1e-9:
        return CataloguedGeometry(
            base,
            area=recorded_A,
            I_zz=float(blob.get("I_zz", base.I_zz)),
            I_yy=float(blob.get("I_yy", base.I_yy)),
            J=max(float(blob.get("J", base.J)), 1e-30),
        )
    return base


def _decode_reinforcement(blob: dict | None) -> ReinforcementLayout | None:
    if blob is None:
        return None
    bars = [
        RebarBar(
            z=b["z"], y=b["y"], area=b["area"],
            designation=b.get("designation", ""),
            material=_decode_material(b.get("material")),
        )
        for b in blob.get("bars", [])
    ]
    return ReinforcementLayout(
        bars=bars,
        stirrup_designation=blob.get("stirrup_designation", "#3"),
        stirrup_spacing=blob.get("stirrup_spacing", 0.150),
        stirrup_legs=blob.get("stirrup_legs", 2),
    )


def section_from_dict(data: dict) -> Section:
    """Reconstruct a :class:`Section` from its JSON-style dict."""
    if data.get("femsolver_section_version") != 1:
        raise ValueError(
            f"unknown section version: {data.get('femsolver_section_version')!r}"
        )
    geom = _decode_geometry(data["geometry"])
    zones = [
        MaterialZone(
            material=_decode_material(z.get("material")),
            name=z.get("name", ""),
        )
        for z in data.get("zones", [])
    ]
    return Section(
        geometry=geom,
        zones=zones,
        name=data.get("name", ""),
        family=data.get("family", ""),
        catalogue_ref=data.get("catalogue_ref"),
        reinforcement=_decode_reinforcement(data.get("reinforcement")),
    )


def section_from_json(text: str) -> Section:
    return section_from_dict(json.loads(text))
