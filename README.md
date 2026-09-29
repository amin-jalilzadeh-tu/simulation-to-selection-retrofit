# From simulation to selection: home retrofit choices

This repository contains the code and data for *From simulation to selection:
effects of surrogate prediction errors on envelope-retrofit decisions under
present and future weather* (Jalilzadeh et al.). It compares **625 combinations**
of window, façade, roof and floor upgrades for a Dutch terraced house across
three weather cases.

The question is practical: **will a fast prediction model recommend the same
upgrade package as the full calculation?** For each package, the analysis
compares heating energy, the number of days one room's daily average temperature
stays below 24 °C, and comparative material-cost and product-stage carbon
**indices**. The indices are not whole-building cost or carbon totals. Different
preferences give these outcomes different weights; for example, one profile
prioritises the temperature measure.

The code first finds the best trade-offs using the archived simulation outputs.
It then tests prediction models that estimate either two outcomes (energy and
temperature) or all four. It reports prediction errors, which packages remain
good trade-offs, and whether the weighted choices agree with the full analysis.
In one saved present-weather case, the temperature-focused choice was **package
25** with the full analysis but **package 16** with the four-outcome model.

## What is in the repository

| Folder | Contents |
| --- | --- |
| [`inputs/`](inputs/) | Reduced daily EnergyPlus outputs for the three weather cases |
| [`scripts/`](scripts/) | Exact analysis and prediction-model validation |
| [`results/`](results/) | Released metrics, predictions, trade-off sets and choices |
| [`figures/`](figures/) and [`supplementary_analysis/`](supplementary_analysis/) | Paper figures, tables and supporting analysis |
| [`tests/`](tests/) | Checks for the reported calculations |

The package starts from **saved daily simulation outputs**; it does not include
the base EnergyPlus model or rerun the building simulation. The reported
prediction-based choices combine held-out predictions from several validation
folds; they are a research check, not the output of one deployed model.

## Reproduce the analysis

Run these commands from the repository root. The released runs used Python
3.13.9 for the exact analysis and 3.14.6 for the prediction models. The
analysis commands write to new `rerun_*` directories, preserving the released
results.

```bash
python3.13 -m venv .venv-exact
.venv-exact/bin/pip install -r requirements-exact.txt
.venv-exact/bin/python scripts/reproduce_exhaustive_analysis.py \
  --output-dir results/rerun_exhaustive_analysis

python3.14 -m venv .venv-surrogate
.venv-surrogate/bin/pip install -r requirements-surrogate.txt
.venv-surrogate/bin/python scripts/validate_mtl_family.py \
  --learned-targets two --output-dir results/rerun_surrogate_validation
.venv-surrogate/bin/python scripts/validate_mtl_family.py \
  --learned-targets four --output-dir results/rerun_surrogate_validation_family4
.venv-surrogate/bin/python -m unittest discover -s tests
```

The model-training commands take longer than inspecting the released results.
For the full protocol, definitions, diagnostics and figure commands, see
[Reproducibility](REPRODUCIBILITY.md). The accompanying manuscript supplement is
in [`supplementary_analysis/SUPPLEMENTARY_S1.md`](supplementary_analysis/SUPPLEMENTARY_S1.md).

## Licences

The currently published code uses the [MIT Licence](LICENSE). The data, results
and figures identified in [LICENSE-DATA.md](LICENSE-DATA.md) use CC BY 4.0.
