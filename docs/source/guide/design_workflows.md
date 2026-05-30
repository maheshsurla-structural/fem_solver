# Design workflows

femsolver ships code-based member-design checks for four major codes:

| Code | Module |
|------|--------|
| **ACI 318-19** (US concrete) | {mod}`femsolver.design.concrete` |
| **AISC 360-22** (US steel) | {mod}`femsolver.design.steel` |
| **IS 456:2000** (Indian concrete) | {mod}`femsolver.design.is456` |
| **IS 800:2007** (Indian steel) | {mod}`femsolver.design.is800` |

Seismic detailing is provided in parallel via
{mod}`femsolver.design.seismic` (ACI 18 / AISC 341) and
{mod}`femsolver.design.is13920`.

## Typical design pipeline

```
analysis envelopes (LoadCombination + LoadPattern)
        ↓
member-level demand pairs (P, M, V) per combo per element
        ↓
member-level capacity check (code module)
        ↓
seismic-detailing check (SCWB, capacity shear, confinement)
        ↓
HTML / CSV report
```

## RC beam example (IS 456)

```python
from femsolver.design import is456

res = is456.is456_beam_flexure(
    M_u=150.0e3,                       # factored moment (N·m)
    f_ck=is456.fck_M(25),              # M25 concrete
    f_y=is456.fy_Fe(415),              # Fe-415 steel
    b=0.300, d=0.460,
)
print(res.A_st * 1e6, "mm^2")          # required tension steel
print(res.note)
```

## Steel beam example (AISC 360)

See `examples/50_steel_frame_design.py` for a complete frame walkthrough
using the {class}`~femsolver.design.steel.designer.SteelMemberDesigner`.

## Capacity design (seismic)

IS 13920 SCWB joint check:

```python
from femsolver.design import is13920

res = is13920.is13920_scwb_check(sum_Mc=600e3, sum_Mb=400e3)
assert res.passes      # ratio 1.5 >= 1.4 limit
```

ACI 18 / AISC 341 equivalents live in
{mod}`femsolver.design.seismic.scwb`.

## Reports

```python
from femsolver.design.reports import (
    make_html_report, make_csv_summary, MemberReportEntry,
)
# entries = [...] (build from design results)
write_html_report(entries, path="design_report.html")
write_csv_summary(entries, path="design_report.csv")
```

See `examples/53_full_design_report.py` for a complete walkthrough.
