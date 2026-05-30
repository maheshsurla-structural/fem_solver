# Analysis types

| Class | What it does | Typical use |
|-------|--------------|-------------|
| {class}`~femsolver.LinearStaticAnalysis` | Solves `K u = f` once | Service loads, gravity, code combinations |
| {class}`~femsolver.NonlinearStaticAnalysis` | Newton or arc-length, with `LoadControl` / `DisplacementControl` / `ArcLength` | Pushover, P-Delta, snap-through, pile-soil |
| {class}`~femsolver.EigenAnalysis` | Generalised eigenvalue (K, M) | Modal periods, mode shapes |
| {class}`~femsolver.LinearBucklingAnalysis` | Linearised buckling about a static pre-load | Euler columns, plate / shell buckling |
| {class}`~femsolver.TransientAnalysis` | Linear time integration | Direct linear dynamics |
| {class}`~femsolver.NonlinearTransientAnalysis` | Newmark / HHT / Gen-α with Newton inner loop | NLTHA, IDA cells |
| {class}`~femsolver.ResponseSpectrumAnalysis` | Modal-superposition response spectrum (SRSS / CQC) | Design-spectrum response |
| {class}`~femsolver.ModalPushoverAnalysis` | Chopra–Goel multi-mode pushover | Modal pushover assessment |
| {class}`~femsolver.IDADriver` | Single-record IDA sweep over IM levels | PBE Stage 1 |
| {func}`~femsolver.multi_record_ida` | Multi-record IDA driver | PBE record-suite IDA |
| {class}`~femsolver.EnvelopeAnalysis` | LoadCombination + LoadPattern envelope | Design-envelope assembly |

## Integrators (static)

```{eval-rst}
.. autoclass:: femsolver.LoadControl
.. autoclass:: femsolver.DisplacementControl
.. autoclass:: femsolver.ArcLength
```

## Integrators (transient)

```{eval-rst}
.. autoclass:: femsolver.Newmark
.. autoclass:: femsolver.NewmarkNonlinear
.. autoclass:: femsolver.HHTAlpha
.. autoclass:: femsolver.GeneralizedAlpha
.. autoclass:: femsolver.CentralDifference
```

## Algorithms (nonlinear iteration)

```{eval-rst}
.. autoclass:: femsolver.Newton
.. autoclass:: femsolver.ModifiedNewton
.. autoclass:: femsolver.LineSearchNewton
```

## Convergence tests

```{eval-rst}
.. autoclass:: femsolver.NormDispIncr
.. autoclass:: femsolver.NormUnbalance
.. autoclass:: femsolver.EnergyIncr
```

## Picking an analysis

| Question | Use |
|----------|-----|
| Will any material yield? | Nonlinear (static or transient) |
| Is the load history important? | Transient or step-by-step nonlinear |
| Only need natural periods? | `EigenAnalysis` |
| Linearised P-Delta / buckling? | `LinearBucklingAnalysis` (must use corotational elements) |
| Code-spectrum response? | `ResponseSpectrumAnalysis` |
| Full PBE / collapse fragility? | `IDADriver` → `multi_record_ida` → `fit_collapse_fragility` |
