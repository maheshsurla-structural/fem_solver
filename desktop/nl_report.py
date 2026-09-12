"""Nonlinear result export + analysis report (GUI-7, plan §14 stage 7).

Pure functions (no Qt) that turn a ``nonlinear.run_pushover`` / ``run_case``
result into exportable artifacts:

* :func:`curve_csv` — the pushover / hysteresis curve (step, displacement,
  base shear).
* :func:`fibers_csv` — one captured step's fiber state (y, z, σ, ε).
* :func:`run_summary` — headline metrics (peak shear, drift, stiffness,
  dissipated energy for cyclic runs, peak fiber strain).
* :func:`member_peak_strain` — each fiber member's peak |strain| over the run
  (the hinge-state summary).
* :func:`report_html` — a self-contained, printable HTML report (metadata +
  metrics + hinge-state table + an embedded curve image).

Kept Qt-free so it is unit-testable headless; the pushover dialog wires these
to ``QFileDialog`` save buttons.
"""
from __future__ import annotations

import html as _html

import numpy as np

# strain past which a fiber section is taken as inelastic (rebar yield ε_y ≈
# f_y/E ≈ 0.0025 for 500 MPa; a coarse "has this hinge yielded?" flag only).
YIELD_STRAIN = 2.0e-3


def curve_csv(disp, shear, *, length_unit="m", force_unit="N") -> str:
    """CSV text of the base-shear vs displacement curve (signed, so a cyclic
    run round-trips the hysteresis)."""
    out = [f"step,displacement [{length_unit}],base_shear [{force_unit}]"]
    for i, (d, s) in enumerate(zip(disp, shear)):
        out.append(f"{i + 1},{float(d):.10g},{float(s):.10g}")
    return "\n".join(out) + "\n"


def fibers_csv(frame, *, length_unit="m") -> str:
    """CSV text of one captured step's fiber state — ``frame`` is a list of
    ``(y, z, stress, strain)`` tuples (as recorded by the capture)."""
    out = [f"y [{length_unit}],z [{length_unit}],stress [Pa],strain"]
    for y, z, sig, eps in frame:
        out.append(f"{float(y):.10g},{float(z):.10g},"
                   f"{float(sig):.10g},{float(eps):.10g}")
    return "\n".join(out) + "\n"


def member_peak_strain(result) -> dict:
    """``{member_id: peak |fiber strain| over all steps}`` from the run's
    ``damage_frames`` — the per-member hinge-state summary."""
    out: dict = {}
    for frame in result.get("damage_frames") or []:
        for mid, v in frame.items():
            out[mid] = max(out.get(mid, 0.0), abs(float(v)))
    return out


def run_summary(result) -> dict:
    """Headline metrics for a run result. Keys present depend on the data:
    always ``protocol``/``steps``; with a curve also ``peak_shear``,
    ``disp_at_peak_shear``, ``max_abs_disp``, ``initial_stiffness``; cyclic adds
    ``dissipated_energy``; captured runs add ``peak_fiber_strain``."""
    disp = [float(x) for x in result.get("disp", [])]
    shear = [float(x) for x in result.get("shear", [])]
    out: dict = {"protocol": result.get("protocol", "monotonic"),
                 "steps": len(disp)}
    if disp:
        d = np.asarray(disp)
        s = np.asarray(shear)
        i = int(np.argmax(np.abs(s)))
        out["peak_shear"] = float(s[i])
        out["disp_at_peak_shear"] = float(d[i])
        out["max_abs_disp"] = float(np.max(np.abs(d)))
        if len(d) > 1 and d[1] != 0.0:
            out["initial_stiffness"] = float(s[1] / d[1])
        if out["protocol"] == "cyclic" and len(d) > 1:
            out["dissipated_energy"] = float(
                abs(np.sum(0.5 * (s[1:] + s[:-1]) * np.diff(d))))
    strains = member_peak_strain(result)
    if strains:
        out["peak_fiber_strain"] = max(strains.values())
    return out


def _fmt(x, sig=4) -> str:
    return f"{x:.{sig}g}"


def report_html(result, meta: dict, *, length_unit="m", force_unit="N",
                curve_png_b64: str | None = None,
                title="Nonlinear analysis report") -> str:
    """A self-contained, printable HTML report: a metadata table (``meta``:
    label→value strings), a results-summary table, a per-member hinge-state
    table, and — if ``curve_png_b64`` (base64 PNG) is given — the embedded
    curve image."""
    s = run_summary(result)
    esc = _html.escape

    def _table(pairs) -> str:
        rows = "".join(
            f"<tr><th>{esc(str(k))}</th><td>{esc(str(v))}</td></tr>"
            for k, v in pairs)
        return f"<table>{rows}</table>"

    meta_rows = list(meta.items())

    metric_rows = [("Protocol", s["protocol"]), ("Steps", s["steps"])]
    if "peak_shear" in s:
        metric_rows += [
            ("Peak base shear", f"{_fmt(s['peak_shear'])} {force_unit}"),
            ("Displacement at peak",
             f"{_fmt(s['disp_at_peak_shear'])} {length_unit}"),
            ("Max |displacement|",
             f"{_fmt(s['max_abs_disp'])} {length_unit}")]
    if "initial_stiffness" in s:
        metric_rows.append(("Initial stiffness",
                            f"{_fmt(s['initial_stiffness'])} "
                            f"{force_unit}/{length_unit}"))
    if "dissipated_energy" in s:
        metric_rows.append(("Dissipated energy",
                            f"{_fmt(s['dissipated_energy'])} "
                            f"{force_unit}·{length_unit}"))
    if "peak_fiber_strain" in s:
        metric_rows.append(("Peak fiber strain", _fmt(s["peak_fiber_strain"])))

    strains = member_peak_strain(result)
    hinge_html = ""
    if strains:
        rows = "".join(
            f"<tr><td>{mid}</td><td>{_fmt(v)}</td>"
            f"<td>{'yielded' if v > YIELD_STRAIN else 'elastic'}</td></tr>"
            for mid, v in sorted(strains.items()))
        hinge_html = (
            "<h2>Member hinge state</h2>"
            "<table class='grid'><tr><th>member</th>"
            "<th>peak |fiber strain|</th><th>state</th></tr>"
            f"{rows}</table>")

    img_html = ""
    if curve_png_b64:
        img_html = ("<h2>Response curve</h2>"
                    f"<img alt='response curve' "
                    f"src='data:image/png;base64,{curve_png_b64}'>")

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{esc(title)}</title>
<style>
  body {{ font: 14px/1.5 -apple-system, Segoe UI, Roboto, sans-serif;
          color: #1a1a1a; margin: 2rem; max-width: 760px; }}
  h1 {{ font-size: 1.4rem; margin: 0 0 .2rem; }}
  h2 {{ font-size: 1.05rem; margin: 1.4rem 0 .4rem;
        border-bottom: 1px solid #ddd; padding-bottom: .2rem; }}
  table {{ border-collapse: collapse; margin: .3rem 0; }}
  th, td {{ text-align: left; padding: .25rem .8rem .25rem 0;
            vertical-align: top; }}
  table.grid th, table.grid td {{ border: 1px solid #ddd; padding: .3rem .6rem; }}
  table:not(.grid) th {{ color: #555; font-weight: 600; padding-right: 1.2rem; }}
  img {{ max-width: 100%; border: 1px solid #eee; margin-top: .4rem; }}
  footer {{ margin-top: 2rem; color: #888; font-size: .85rem; }}
</style></head><body>
<h1>{esc(title)}</h1>
<h2>Run</h2>
{_table(meta_rows)}
<h2>Results</h2>
{_table(metric_rows)}
{hinge_html}
{img_html}
<footer>Generated by femsolver desktop — nonlinear (fiber) analysis.</footer>
</body></html>
"""
