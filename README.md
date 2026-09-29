# Dutch Terraced-House Retrofit Analysis — Supplementary File S1

Code and data accompanying the article *From simulation to selection: effects of
surrogate prediction errors on envelope-retrofit decisions under present and
future weather* (Jalilzadeh et al.).

This archive holds the reduced daily EnergyPlus outputs for 625 envelope
packages under three weather datasets, the exact enumeration analysis, the
multi-task surrogate family screen under both task designs, the MCDA
selection-layer analysis, the released result tables and figures, and the test
suite that guards the reported numbers.

Exact enumeration is the reference simulation analysis. The surrogate screen is
evaluated against it: package-grouped out-of-fold point accuracy, then whether
surrogate-derived Pareto sets and weighted selections reproduce the exact
decisions.

Everything here runs from the extracted archive root. No EnergyPlus run, no
network access, and no GPU are required.

## 1. Environments

The released results were produced in **two different environments**, so this
archive ships two requirement files rather than one:

| File | Reproduces | Interpreter |
| --- | --- | --- |
| `requirements-exact.txt` | `results/exhaustive_analysis/` | Python 3.13.9 |
| `requirements-surrogate.txt` | `results/surrogate_validation*/` | Python 3.14.6 |

Each pin is copied from the `software_versions` block that the corresponding
run wrote into its own metadata file (`results/exhaustive_analysis/run_metadata.json`
and `results/surrogate_validation*/model_metadata.json`). Installing one file
over the other will not reproduce both sets of numbers: the two environments
differ in Python, numpy and pandas.

Use separate virtual environments:

```bash
python3.13 -m venv .venv-exact
.venv-exact/bin/pip install -r requirements-exact.txt

python3.14 -m venv .venv-surrogate
.venv-surrogate/bin/pip install -r requirements-surrogate.txt
```

The `supplementary_analysis/` scripts additionally need `scipy` (for
`scipy.optimize.linprog`) and `matplotlib`. The main-figure rendering environment
is recorded in `supplementary_analysis/figure_environment.json`; these versions
are separate from the two analysis environments above. The scipy version is
not pinned.

## 2. Reproduce the manuscript results

### 2.1 Exact enumeration

```bash
.venv-exact/bin/python scripts/reproduce_exhaustive_analysis.py \
    --output-dir results/rerun_exhaustive_analysis
```

Flags: `--input-dir` (default `inputs/`), `--output-dir` (default
`results/exhaustive_analysis/`, i.e. it overwrites the released copy unless you
redirect it) and `--skip-known-checks`.

The script enumerates all 625 packages for each of 2020, 2050 and 2100,
computes the four objectives, finds the exact Pareto front, applies the five
weighted stakeholder profiles, and repeats the analysis at upper temperature
thresholds of 23, 24, 25 and 26 °C. Unless `--skip-known-checks` is given it
asserts the manuscript's Pareto counts, threshold-specific counts and reference
deltas, weighted-sum selection IDs, temperature-provenance counts and
present-climate correlations, and fails loudly if any of them move.

### 2.2 Surrogate family screen — the two task designs

The manuscript reports one model family under two task designs, selected with
`--learned-targets`:

```bash
# exact-index design: energy and D24 learned, cost and GWP stay exact
.venv-surrogate/bin/python scripts/validate_mtl_family.py \
    --learned-targets two \
    --output-dir results/rerun_surrogate_validation

# all-predicted design: all four objectives learned and used in the front
.venv-surrogate/bin/python scripts/validate_mtl_family.py \
    --learned-targets four \
    --output-dir results/rerun_surrogate_validation_family4
```

Always pass `--output-dir`. The default is `results/surrogate_validation/`,
which holds the released exact-index results.

`scripts/validate_surrogate_publication.py` contains the earlier three-model
configuration and is kept as a reference for the regression tests. Use
`scripts/validate_mtl_family.py` to reproduce the reported results.

Each run trains, per outer fold: the shared two-head MTL network, one matched
single-task network per learned target, five further family variants (shared
with MGDA; the `Separate` port with and without MGDA; the `Deep_Balanced` port
with and without MGDA), one multi-output random forest, and one
gradient-boosted-tree regressor per learned target. With the default three
repeat seeds and five outer folds that is 120 neural fits, 15 forests and 30
boosters for `--learned-targets two`, and 150 neural fits, 15 forests and 60
boosters for `--learned-targets four`. Everything runs on one CPU thread for
determinism, so budget accordingly.

Other flags, all defaulting to the released settings: `--input-dir`, `--seeds`
(17 29 43), `--outer-folds` (5), `--validation-fraction` (0.15),
`--trunk-widths` (128 128 64), `--head-hidden-size` (32), `--max-epochs` (300),
`--patience` (30), `--min-delta` (1e-6), `--learning-rate` (1e-3),
`--weight-decay` (1e-5), `--lr-scheduler-factor` (0.5),
`--lr-scheduler-patience` (10), `--minimum-learning-rate` (1e-6),
`--batch-size` (128), `--gradient-clip-norm` (5.0), `--rf-estimators` (300),
`--rf-min-samples-leaf` (1), `--rf-max-depth` (unset).

### 2.3 The dropout diagnostic

`results/surrogate_validation_family4_dropout_diagnostic/` was produced under
the exact-index design with a single repeat seed and the `Deep_Balanced`
dropout lowered from 0.5 to 0.1. Neither the dropout rate nor the two other
family-only settings (`separate_hidden_size`, `hgb_max_iter`,
`hgb_learning_rate`) has a command-line flag, so this run needs a short driver
rather than a flag:

```bash
.venv-surrogate/bin/python - <<'PY'
from pathlib import Path
from scripts import validate_mtl_family as family

config = family.ValidationConfig(seeds=(17,), deep_balanced_dropout=0.1)
configurations, input_files = family.load_exact_dataset(Path("inputs"))
artifacts = family.run_validation(configurations, config, input_files)
family.write_artifacts(
    Path("results/rerun_dropout_diagnostic"), artifacts
)
PY
```

The released directory keeps only `fold_metrics.csv`, `metrics_summary.csv` and
`model_metadata.json`; the driver above writes the full artifact set.

### 2.4 MCDA selection layer and figures

```bash
cd supplementary_analysis
python mcda_supportedness_sweep.py     # writes mcda_outputs.csv, mcda_profile_sensitivity.csv
python mcda_joint_sweep.py             # writes mcda_joint_outputs.csv
python mcda_weight_agreement.py        # writes mcda_weight_agreement.csv
python mcda_central_weights.py         # writes mcda_central_weights.csv
python make_supplementary_tables.py    # emits the generated subset of the tables
python make_s1_agreement_tables.py     # Tables S1.13, S1.14 and S1.20
python within_fold_selection_check.py               # Tables S1.21-S1.24, retained outputs only
python recompute_decision_validation.py            # decision tables from the predictions
```

These scripts resolve this archive's `results/` directory automatically when run
from inside `supplementary_analysis/`; `S1_EXHAUSTIVE`, `S1_SURROGATE`, `FIGDIR`
and `S1_OUT` override the locations for the preceding generators. The held-out-fold
diagnostic instead accepts `--s1-root` and `--output-dir`, with defaults relative to
its installed script location. All are deterministic (fixed seed 42 for the weight
sweeps). Run the two sweeps before `make_supplementary_tables.py` if
you regenerate the CSVs from scratch.

`SUPPLEMENTARY_S1.md` is partly hand-maintained: sections S1.0, S1.10 and S1.19
are not produced by any generator. `make_supplementary_tables.py` emits only the
generated subset and refuses to overwrite the document unless
`S1_ALLOW_OVERWRITE=1` is set; write to a scratch path with `S1_OUT` to inspect
what it produces. `make_mtl_s1_tables.py` and `make_s1_agreement_tables.py` splice
their tables into the existing document in place and are idempotent.

`recompute_decision_validation.py` regenerates the decision tables from the
released out-of-fold predictions and verifies them against the shipped tables.
Use its default verification mode with this archive.

The figure generators in the same directory write to `FIGDIR` (default
`./figures_out`). Their plotting helpers are included in
`supplementary_analysis/`, so they run from inside that directory.
`make_attainment_figure.py` generates manuscript Figures 2 and 3 using
`make_retrofit_figures.py`, including the plotted data and numerical checks.

### 2.5 Tests

```bash
.venv-surrogate/bin/python -m unittest discover -s tests
```

Run from the archive root, and use the surrogate environment: the suite imports
torch and scikit-learn. The suite covers

- `test_reproduce_exhaustive_analysis.py` — infiltration design flow, the
  heating-energy row extractor, zero-span normalisation, the reconciled state
  table, and the threshold Pareto summary against the manuscript values;
- `test_rate_indices.py` — the cost and GWP rate indices and their alias
  functions;
- `test_mcda_selection.py` — the selection tie rule, the released
  unique-selectability counts, ASF agreement, and the sweep coverage counts;
- `test_mtl_family_task_designs.py` — the `two`/`four` task-design switch, the
  per-design model shapes, and end-to-end runs of both designs;
- `test_mtl_family_mgda.py` — the min-norm task weights (exact closed form for
  two tasks, Frank-Wolfe for four) and the MGDA update, where the
  shared-trunk gradient is replaced by the min-norm combination;
- `test_mtl_family_comparators.py` — the gradient-boosted-tree comparator, the
  command-line contract, and the bitwise regression of the three published
  models against `validate_surrogate_publication.py`;
- `test_mtl_family_decision_layer.py` — the D24 clipping rule, that heating
  energy is not clipped and day counts are not rounded, that raw predictions survive for
  point metrics, and that the four-target design really routes predicted cost
  and GWP into the front;
- `test_surrogate_publication_validation.py` — regression checks for
  `validate_surrogate_publication.py`.

## 3. What the surrogate run writes

`results/surrogate_validation*/` contains:

- `fold_metrics.csv`, `metrics_summary.csv` — three-repeat, five-fold
  package-grouped MAE, RMSE and R² per model and target;
- `horizon_fold_metrics.csv`, `horizon_metrics_summary.csv` — the same, split by
  weather dataset;
- `oof_predictions.csv` — every out-of-fold prediction, with the true value and
  residual;
- `split_assignments.csv` — every inner-train / inner-validation / outer-test
  assignment;
- `surrogate_pareto_validation.csv` — exact-versus-surrogate Pareto precision,
  recall, F1, Jaccard, and the raw out-of-range day-count diagnostics;
- `weighted_profile_validation.csv` — weighted-profile selection agreement and
  exact-objective regret;
- `model_metadata.json` — protocol, full configuration, software versions,
  per-fold train-only scaler values, every training seed, and the SHA-256 of
  each input file.

Point metrics use the raw predictions. For the decision comparison only, the
predicted day count is clipped to its physical range of 0–365 without rounding;
in the four-target design the predicted cost and GWP indices are clipped at zero
for the same reason, and heating energy is not clipped. Within each repeat and weather dataset the
predictions from the five separately fitted outer-fold models are stitched into
an out-of-fold composite covering all 625 packages. These fronts diagnose the
cross-validated training protocol; they are not the output of a single deployed
model. The unbounded diagnostics are recorded in `model_metadata.json`.

The all-predicted design predicts all four objectives: annual heating energy,
days with daily-mean air temperature below 24 °C, and the cost and product-stage
GWP indices. The exact-index design predicts the first two objectives and
supplies the cost and GWP indices as exact component-rate sums. Decision analysis
uses the corresponding predicted or exact indices for each design.

## 4. Definitions

- **Annual heating energy** is the archived daily heating series summed over the
  year and converted to GJ/year. In the released merged CSV files that series is
  the row labelled `Gas Consumption [J](Daily)`, in joules per day. No efficiency
  and no fuel conversion are applied, so the quantity is not purchased gas,
  primary energy or measured demand.
- **The primary temperature indicator** counts days when the selected zone's
  daily mean air temperature is below 24 °C. Outputs also report the strict
  `17.5 < T < 24 °C` band count and sensitivity at upper thresholds of 23, 24,
  25 and 26 °C. It is a warm-side screen, not a full comfort assessment.
- **`infiltration_design_flow_m3_s`** reproduces the configuration-specific
  design-flow calculation used by the archived IDF-generation workflow.
- **`retrofit_state_specifications.csv`** reconciles each discrete state ID with
  the generator's thickness/conductivity value, the value stored in the
  reduced CSV metadata (the surrogate input), component-rate indices and infiltration factors.
  It also records that the façade state changes the north/south windowed walls
  rather than every outdoor-coupled surface.
- **Cost and product-stage GWP** are comparative sums of component-specific unit
  rates. Envelope areas are not applied, so these are indices, not
  whole-building costs or embodied-carbon totals. `utils/cost.py` and
  `utils/carbon.py` also provide `calculate_total_cost` and
  `calculate_total_carbon` as aliases; they return the same index values as
  `calculate_cost_rate_index` and `calculate_gwp_rate_index`.
- **Dataset label `2100`** is kept for compatibility. The corresponding EPW
  header identifies an SSP5-8.5 Future Weather Generator timeframe of 2080, so
  the manuscript calls it the late-century dataset.

## 5. Archive contents

```
inputs/                     three reduced daily-output CSVs (2020, 2050, 2100)
scripts/
  reproduce_exhaustive_analysis.py   exact enumeration and MCDM selection
  validate_mtl_family.py             current surrogate family screen
  validate_surrogate_publication.py  three-model reference for the regression tests
utils/                      cost and GWP rate-index functions
tests/                      the test suite described in section 2.5
results/exhaustive_analysis/                     released exact-analysis outputs
results/surrogate_validation/                    released exact-index design
results/surrogate_validation_family4/            released all-predicted design
results/surrogate_validation_family4_dropout_diagnostic/  dropout-0.1 diagnostic
supplementary_analysis/     MCDA sweeps, figure generators, SUPPLEMENTARY_S1.md
figures/                    manuscript and supplementary figures
simulation_assumptions.csv
weather_file_descriptors.csv
README_S1.txt               the submitted supplementary-file description
```

In `weather_file_descriptors.csv`, HDD18 and CDD18 use the daily mean outdoor
dry-bulb temperature: `sum(max(18 - T_day, 0))` and `sum(max(T_day - 18, 0))`.
JJA P95 is the linearly interpolated 95th percentile of the 92 June–August daily
means.

The input files are also version-controlled at
<https://github.com/amin-jalilzadeh-tu/IsaChao_Retrofit-/tree/main/inputs>.

## 6. Limits

- This archive starts from the daily simulation outputs in `inputs/` and
  reproduces the post-processing, surrogate training, reported metrics, Pareto
  comparisons and weighted selections. The base EnergyPlus model and weather
  files are available from the corresponding author.
- Re-running the surrogate screen is deterministic on CPU within a fixed
  environment. Results can still shift across platforms and library versions, so
  install the pinned environment before comparing numbers.
- The released `model_metadata.json` files for the two exact-index runs predate
  one metadata field that the current script always writes
  (`decision_validation.predicted_front_cost_carbon_source`). A re-run therefore
  adds that key; the CSV artifacts are unaffected.


## 7. Held-out-fold decision diagnostics (Tables S1.21–S1.24)

From the archive root:

```bash
python supplementary_analysis/within_fold_selection_check.py
```

This NumPy/pandas script postprocesses retained OOF predictions, exact enumeration
and full-grid decision tables. It performs no model fitting or EnergyPlus execution
and preserves its inputs. The package-relative default output is
`supplementary_analysis/decision_fidelity_diagnostics/`; `--s1-root PACKAGE` and
`--output-dir OUTPUT` override those locations. The folder contains individual
fold/profile decisions, model summaries, exact score margins, regret tails, physical
case details, input SHA256 fingerprints, and Markdown/LaTeX versions of Tables
S1.21–S1.24. The script does not overwrite `SUPPLEMENTARY_S1.md`.

For each formulation and task design, 3 repeat seeds × 5 held-out folds × 3 weather
cases give **45 evaluations**, each comparing the same 125 candidates. Five named
profiles give **225 profile decisions** per formulation/design. Pareto F1 and
sampled-weight agreement are means and sample SDs over those 45 evaluations. The
same 10,000 Dirichlet(1,1,1,1) weight draws (seed 42) are reused in each evaluation.
Each front has its own ideal/nadir; regret evaluates the selected package under
the exact front's normalisation for that 125-candidate set. Subset nondominance and
regret concern those 125 packages. The repeated data, partitions, weather cases
and profiles yield dependent comparisons.

The retained 625-package composite has 9 repeat/weather evaluations and 45 profile
decisions per formulation/design. Pooling five formulations gives 225 full-grid
or 1,125 held-out profile comparisons per task design in Table S1.23. Its
mismatch-only rows exclude exact package agreements. The two candidate-set sizes
have different exact fronts and normalisation anchors, so their regret magnitudes
refer to different decision problems.

In full-grid reconstruction, the five exact-index formulations have 127 of 133
mismatches (95.5%) on the exact front and median mismatch regret 0.0047. The
all-predicted design has 160 of 180 (88.9%) and median 0.0164. Pooling across
both task designs gives 287 of 313 (91.7%) and median 0.0078.

Exact winner-to-runner-up margins are counted once per candidate set and profile
(15 full-grid and 225 held-out decisions). Their median exact-front-normalised
margins are **0.001705 (0.0017 rounded)** and **0.006630**, respectively. The
association of larger margins with higher package agreement is descriptive;
changing candidate count also changes the front and its normalisation, so this
comparison does not isolate a causal effect of margins or cross-fitting.

Manuscript Figure 4 (`fig_model_comparison.pdf/png`) pairs exact-index raw point
errors with full-grid decision fidelity for both task designs. The lower panels
show exact selection agreement over 45 named-profile cases per model/design and
individual mismatch regrets, median markers and interquartile-range segments
on a logarithmic axis (25–42 mismatches per formulation/design). The 95th
percentiles and maxima remain in Table S1.23 and the `full625` / `mismatches`
rows of `supplementary_analysis/decision_fidelity_diagnostics/regret_tails.csv`.
The main figure generator accepts `S1_EXHAUSTIVE`, `MTL_SURROGATE`,
`MTL4_SURROGATE` and `S1_ANALYSIS` for alternative input directories. The all-predicted point-error plot remains in
`figures/fig_model_comparison_all_predicted.pdf/png`.

The five main figures can be regenerated from the released tables, without
simulation or fitting, from the extracted archive root:

```bash
FIGDIR="$PWD/figures" python supplementary_analysis/make_framework_figure.py
FIGDIR="$PWD/figures" python supplementary_analysis/make_attainment_figure.py
FIGDIR="$PWD/figures" python supplementary_analysis/make_mtl_figures.py --comparison-only
FIGDIR="$PWD/figures" python supplementary_analysis/make_smaa_summary_figure.py
```

Figures 2-5 share the `make_main_figures.py` entry point. Figures 2 and 3 are
generated by `make_retrofit_figures.py`: `fig_package_tradeoffs.pdf/png` shows
all 625 package outcomes per case, and `fig_component_comparisons.pdf/png`
shows all 125 paired component-option comparisons per case. Window, façade
and roof pairs are grouped by the unchanged floor option: existing (25 pairs)
or insulated (100). The ground-floor row changes the floor option itself.
All option changes include their assigned infiltration change. The generator
exports plotted values, pooled and grouped summaries, and source hashes in
`figures/retrofit_tradeoff_data/`.

Figure 5 retains selection-frequency and conditional-average-weight bars,
with coverage bars below. Place Figures 2 and 3 at 0.94 text width, Figure 4
at 0.95 and Figure 5 at 0.98 to preserve the intended lettering.

The attainment curves and their matched-effect panels remain available as
Supplementary Figure S1.7:

```bash
FIGDIR="$PWD/figures" python supplementary_analysis/make_attainment_s1_figure.py
```

The `--comparison-only` option updates only manuscript Figure 4. Table S1.4 and
Appendix Table B.4 give weather-specific accuracy for the exact-index design
first, followed by the all-predicted design. Refresh only S1.4 with
`python supplementary_analysis/make_mtl_s1_tables.py --weather-only`.


The environment used for the main figures is recorded in
supplementary_analysis/figure_environment.json.

## Licence

Code (all `.py` files) is released under the MIT Licence (`LICENSE`). Data and
results (`inputs/`, `results/`, `figures/`, and the `.csv`/`.json` files in
`supplementary_analysis/`) are released under the Creative Commons Attribution
4.0 International licence (`LICENSE-DATA.md`).
