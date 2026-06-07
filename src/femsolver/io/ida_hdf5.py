"""HDF5 persistence for IDA results (Phase 48.1).

For a typical PBE study -- 30 records x 50 IM levels x 1000 time
steps x 1000 DOFs -- the in-memory storage of an :class:`IDASummary`
is fine, but **time-history traces** would not fit. This module
gives a non-invasive way to spill IDA results to a single HDF5 file
that can be re-loaded later for fragility fitting, P-58 input, or
just inspection.

Storage layout
--------------

::

    /
    ├── meta/
    │   ├── n_records         : int
    │   ├── edp_names         : list[str] (variable-length strings)
    │   └── created           : ISO timestamp
    ├── records/
    │   ├── <record_name_0>/
    │   │   ├── IMs                  : (n_levels,) float64
    │   │   ├── scale_factors        : (n_levels,) float64
    │   │   ├── converged            : (n_levels,) bool
    │   │   ├── n_steps_completed    : (n_levels,) int64
    │   │   └── edps/<edp_name>      : (n_levels,) float64
    │   └── <record_name_1>/ ...
    └── collapse/                    [only present when saving an IDASummary]
        ├── collapse_IMs    : (n_records,) float64
        ├── causes          : (n_records,) variable-length string
        └── point_indices   : (n_records,) int64  (-1 sentinel for None)

``h5py`` is imported lazily. Calling these helpers without h5py
installed raises a clear ImportError.
"""
from __future__ import annotations

import datetime
import math
from typing import Iterable

import numpy as np

from femsolver.performance.ida import IDAPoint, IDARecord
from femsolver.performance.ida_collapse import CollapseResult, IDASummary


def _h5():
    try:
        import h5py
    except ImportError as exc:                                  # pragma: no cover
        raise ImportError(
            "h5py is required for femsolver.io.ida_hdf5 (install with "
            "`pip install h5py`)."
        ) from exc
    return h5py


# ============================================================ save

def _collect_edp_names(record: IDARecord) -> list[str]:
    names: list[str] = []
    for p in record.points:
        for k in p.EDPs:
            if k not in names:
                names.append(k)
    return names


def _write_record_group(g, record: IDARecord, edp_names: Iterable[str]) -> None:
    pts = record.points
    n = len(pts)
    IMs = np.array([p.IM for p in pts], dtype=np.float64)
    sf = np.array([p.scale_factor for p in pts], dtype=np.float64)
    conv = np.array([p.converged for p in pts], dtype=bool)
    nstep = np.array([p.n_steps_completed for p in pts], dtype=np.int64)
    g.create_dataset("IMs", data=IMs, compression="gzip")
    g.create_dataset("scale_factors", data=sf, compression="gzip")
    g.create_dataset("converged", data=conv, compression="gzip")
    g.create_dataset("n_steps_completed", data=nstep, compression="gzip")
    edps_g = g.create_group("edps")
    for name in edp_names:
        col = np.array(
            [p.EDPs.get(name, math.nan) for p in pts],
            dtype=np.float64,
        )
        edps_g.create_dataset(name, data=col, compression="gzip")


def save_ida_record(record: IDARecord, path: str) -> None:
    """Save one :class:`IDARecord` to ``path`` (HDF5)."""
    h5py = _h5()
    edp_names = _collect_edp_names(record)
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["n_records"] = 1
        meta.attrs["created"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        dt = h5py.string_dtype(encoding="utf-8")
        meta.create_dataset(
            "edp_names",
            data=np.array(edp_names, dtype=object),
            dtype=dt,
        )
        recs_g = f.create_group("records")
        g = recs_g.create_group(record.record_name)
        _write_record_group(g, record, edp_names)


def save_ida_summary(summary: IDASummary, path: str) -> None:
    """Save a full :class:`IDASummary` (multi-record + collapse) to HDF5."""
    h5py = _h5()
    edp_names: list[str] = []
    for rec in summary.records:
        for name in _collect_edp_names(rec):
            if name not in edp_names:
                edp_names.append(name)
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["n_records"] = len(summary.records)
        meta.attrs["created"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        dt = h5py.string_dtype(encoding="utf-8")
        meta.create_dataset(
            "edp_names",
            data=np.array(edp_names, dtype=object),
            dtype=dt,
        )
        recs_g = f.create_group("records")
        for rec in summary.records:
            g = recs_g.create_group(rec.record_name)
            _write_record_group(g, rec, edp_names)
        # Collapse block
        coll = f.create_group("collapse")
        coll.create_dataset("collapse_IMs", data=summary.collapse_IMs)
        coll.attrs["n_collapsed"] = int(summary.n_collapsed)
        coll.attrs["median_collapse_IM"] = float(summary.median_collapse_IM) \
            if math.isfinite(summary.median_collapse_IM) else math.nan
        causes = np.array(
            [c.cause for c in summary.collapse_results], dtype=object,
        )
        coll.create_dataset("causes", data=causes, dtype=dt)
        idx = np.array(
            [-1 if c.collapse_point_index is None
                else int(c.collapse_point_index)
                for c in summary.collapse_results],
            dtype=np.int64,
        )
        coll.create_dataset("point_indices", data=idx)


# ============================================================ load

def _read_record_group(g, edp_names: Iterable[str]) -> IDARecord:
    IMs = np.asarray(g["IMs"])
    sf = np.asarray(g["scale_factors"])
    conv = np.asarray(g["converged"]).astype(bool)
    nstep = np.asarray(g["n_steps_completed"]).astype(int)
    edps_g = g["edps"]
    edp_cols = {
        name: np.asarray(edps_g[name])
        for name in edp_names if name in edps_g
    }
    pts = []
    for i in range(IMs.size):
        EDPs = {name: float(col[i]) for name, col in edp_cols.items()}
        pts.append(IDAPoint(
            IM=float(IMs[i]),
            scale_factor=float(sf[i]),
            EDPs=EDPs,
            converged=bool(conv[i]),
            n_steps_completed=int(nstep[i]),
        ))
    return IDARecord(record_name=g.name.rsplit("/", 1)[-1], points=pts)


def load_ida_record(path: str, record_name: str | None = None) -> IDARecord:
    """Load a single :class:`IDARecord` from ``path``.

    If ``record_name`` is None and the file contains exactly one
    record, that record is returned. Otherwise the name must be given.
    """
    h5py = _h5()
    with h5py.File(path, "r") as f:
        edp_names = [s.decode() if isinstance(s, bytes) else s
                     for s in f["meta/edp_names"][...]]
        recs_g = f["records"]
        if record_name is None:
            keys = list(recs_g.keys())
            if len(keys) != 1:
                raise ValueError(
                    f"{path} has {len(keys)} records; record_name required."
                )
            record_name = keys[0]
        if record_name not in recs_g:
            raise KeyError(
                f"record '{record_name}' not found in {path}"
            )
        return _read_record_group(recs_g[record_name], edp_names)


def load_ida_summary(path: str) -> IDASummary:
    """Load an :class:`IDASummary` written by :func:`save_ida_summary`."""
    h5py = _h5()
    with h5py.File(path, "r") as f:
        edp_names = [s.decode() if isinstance(s, bytes) else s
                     for s in f["meta/edp_names"][...]]
        recs_g = f["records"]
        records = [_read_record_group(recs_g[name], edp_names)
                   for name in recs_g.keys()]
        if "collapse" not in f:
            # Summary without collapse info; reconstruct trivially
            return IDASummary(
                records=records,
                collapse_results=[],
                collapse_IMs=np.empty(0),
                n_collapsed=0,
                median_collapse_IM=math.nan,
            )
        coll = f["collapse"]
        collapse_IMs = np.asarray(coll["collapse_IMs"], dtype=np.float64)
        causes = [s.decode() if isinstance(s, bytes) else s
                  for s in coll["causes"][...]]
        idxs = np.asarray(coll["point_indices"], dtype=np.int64)
        collapse_results = [
            CollapseResult(
                collapse_IM=float(collapse_IMs[i]),
                cause=causes[i],
                collapse_point_index=(None if int(idxs[i]) < 0
                                        else int(idxs[i])),
            )
            for i in range(len(causes))
        ]
        n_coll = int(coll.attrs.get("n_collapsed", 0))
        med = float(coll.attrs.get("median_collapse_IM", math.nan))
        return IDASummary(
            records=records,
            collapse_results=collapse_results,
            collapse_IMs=collapse_IMs,
            n_collapsed=n_coll,
            median_collapse_IM=med,
        )
