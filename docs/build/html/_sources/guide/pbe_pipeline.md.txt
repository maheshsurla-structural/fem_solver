# Performance-Based Earthquake Engineering pipeline

The PBE pipeline ties together IDA, collapse, fragility, record
selection, soil-structure interaction, and FEMA P-58 component damage.

```
PSHA target ──► record selection (CMS + ASCE 7 scaling)
                            │
                            ▼
                multi-record IDA / NLTHA  ──► EDPs (PSD, PFA, collapse IM)
                            │
                            ├──► fragility fit (theta, beta)
                            ▼
            FEMA P-58 component damage + Monte-Carlo
                            │
                            ▼
                   loss curve E[L | IM]
```

## 1. Record selection (Phase 26)

```python
from femsolver import (
    compute_sdof_response_spectrum,
    conditional_mean_spectrum,
    scale_record_suite,
    period_range_mask,
)
```

See {mod}`femsolver.analysis.cms` and
{mod}`femsolver.analysis.record_scaling`.

## 2. IDA + collapse (Phase 25)

```python
from femsolver import IDADriver, multi_record_ida, detect_collapse

summary = multi_record_ida(
    model_factory=factory,
    records=records,
    IM_levels=IM_levels,
    edp_extractor=max_drift_edp(...),
    drift_limit=0.10,
)
```

See {mod}`femsolver.analysis.ida` and
{mod}`femsolver.analysis.ida_collapse`.

## 3. Fragility fit (Phase 25.3)

```python
from femsolver import fit_collapse_fragility

fit = fit_collapse_fragility(
    summary.collapse_IMs[finite_mask],
    no_collapse_IM_max=nc_IM_max,
)
print(fit.theta, fit.beta)        # lognormal median + dispersion
```

See {mod}`femsolver.analysis.fragility`.

## 4. SSI (Phase 27)

Gazetas footing impedance and API p-y/t-z/q-z soil springs:

```python
from femsolver import (
    HalfspaceSoil, gazetas_surface_footing,
    py_curve_sand, py_curve_soft_clay,
    tz_curve_sand, tz_curve_clay, qz_curve,
)
```

See {mod}`femsolver.analysis.ssi` and
{mod}`femsolver.analysis.soil_springs`.

## 5. P-58 component damage (Phase 28)

```python
from femsolver import (
    DamageState, ComponentFragility, ComponentGroup,
    ComponentDamageAssessment,
)

drywall = ComponentFragility(
    name="B1041 drywall partition",
    edp_type="PSD",
    damage_states=[
        DamageState("DS1", 0.005, 0.4, cost_median=200.0),
        DamageState("DS2", 0.012, 0.4, cost_median=1000.0),
        DamageState("DS3", 0.025, 0.4, cost_median=3000.0),
    ],
)
groups = [ComponentGroup(drywall, quantity=100.0, edp_value=0.015)]
assess = ComponentDamageAssessment(groups)
result = assess.monte_carlo(n_realisations=5000, seed=42)
print(result.mean_loss, result.p95_loss)
```

See {mod}`femsolver.analysis.p58`.

## Capstone

The full pipeline lives end-to-end in
`examples/43_pbe_full_workflow.py` — IDA → EDPs → P-58 → loss curve.
