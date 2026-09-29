Supplementary File S1: exact analysis and surrogate-validation package

This archive supports the single integrated manuscript. It contains the three
reduced daily EnergyPlus-output datasets, the exact-analysis and surrogate
family-screen scripts, publication-focused tests, component-rate helper
functions, all derived result tables, every out-of-fold prediction and split
assignment, model metadata, figures, and weather-file descriptors.

Recommended environment and commands, run from the extracted archive root. The
exact analysis and the surrogate screen were produced in two different
environments, so each has its own requirement file; install them in separate
virtual environments (see REPRODUCIBILITY.md, section 1):

    python -m pip install -r requirements-exact.txt
    python scripts/reproduce_exhaustive_analysis.py

    python -m pip install -r requirements-surrogate.txt
    python scripts/validate_mtl_family.py --learned-targets four \
        --output-dir results/rerun_surrogate_validation_family4
    python scripts/validate_mtl_family.py --learned-targets two \
        --output-dir results/rerun_surrogate_validation
    python -m unittest discover -s tests

The two family-screen commands correspond to the manuscript's two task designs:
"four" is the all-predicted design and "two" is the exact-index design. Write
them to new output directories, as shown; the script's default output directory
is results/surrogate_validation, which holds the released exact-index results.
scripts/validate_surrogate_publication.py contains the earlier three-model
configuration used as a regression reference; scripts/validate_mtl_family.py
reproduces the reported results.

The exact-analysis script enumerates all 625 packages in each of the three
weather datasets, calculates the four stated objectives, identifies exact
Pareto sets, applies the five weighted profiles, and repeats the upper-
temperature-threshold sensitivity analysis.

The all-predicted design predicts all four objectives: annual heating energy,
days with daily-mean air temperature below 24 degrees C, and the cost and
product-stage GWP indices. The exact-index design predicts the first two
objectives and supplies the cost and GWP indices as exact component-rate sums.
Validation uses
three repeats of five package-grouped outer folds. Every package's three weather
rows remain together, the inner early-stopping split is also group-disjoint,
and all scalers are fitted only on inner-training rows. The manuscript reports
five formulations: the shared multi-task network under equal-weighted and
gradient-balanced training, matched single-task networks, a 300-tree multi-output
random forest, and per-objective gradient-boosted trees. Four further
architecture variants are included in the archive and reported in Appendix
Table B.3 and S1. Raw predictions are retained for point metrics. Immediately before the
decision analysis, predictions are returned to their feasible ranges: predicted
day counts to 0--365 without rounding, and in the four-target design the
predicted cost and GWP indices to zero or above, both indices being sums of
non-negative component rates. Heating energy is not clipped. Five outer-fold prediction blocks are stitched to form
an OOF composite covering all 625 packages; this is a cross-validated diagnostic,
not the output of one fitted deployment model. Outputs include pooled and
horizon-specific point metrics, Pareto precision/recall/F1, weighted-selection
agreement, exact-objective regret, and raw prediction-domain diagnostics.

The largest named-profile regret was 0.472262 for the equally weighted joint
network in the all-predicted design, repeat seed 29, present weather and the
D24-priority profile. It selected package 16 instead of exact package 25.
Package 16 had raw predictions of 376.8 days and -15.26 GWP-index points,
clipped to 365 and zero before scoring. The same package is selected without
clipping.

The published validation artifacts are included under
results/surrogate_validation_family4/ (all-predicted design) and
results/surrogate_validation/ (exact-index design). Re-running the neural validation is deterministic
on CPU with the pinned software versions recorded in requirements-surrogate.txt
and model_metadata.json, subject to platform-level numerical differences.

The input files are also version-controlled at:

    https://github.com/amin-jalilzadeh-tu/IsaChao_Retrofit-/tree/main/inputs

The package starts from the daily simulation outputs and reproduces the
manuscript's post-processing, surrogate training, reported prediction metrics,
Pareto comparisons, and weighted selections.

Model and source documentation:

  * Appendix Table A.1 of the manuscript specifies the prescribed operating
    schedules, internal gains and boundary conditions. The monthly ground
    temperatures are the same across floor states and weather cases.
  * Appendix Table A.2 lists, for each component state, the model-generation
    value and the surrogate-input value stored in the reduced dataset; the
    surrogates use the surrogate-input values.
  * Component cost and product-stage GWP rates are taken from the underlying
    master's thesis and enter the analysis as comparative component-rate
    indices.
  * The reduction workflow renames "Boiler Heating Energy" as "Gas Consumption".
    It also renames "TERRACED_HOUSE_2_72B36ACE" as
    "BLOCK BUILDINGBLOCK1 STOREY 0". The base geometry places this temperature
    zone between 3 and 6 m above ground. The archive retains daily means for
    this first-storey zone; the heating series is boiler-delivered heat and
    excludes domestic hot water.

The descriptive baseline-case summaries in Results 3.3 use the existing
out-of-fold predictions and weighted-profile tables. They can be reproduced
without fitting models or rerunning simulations or optimisation:

    python supplementary_analysis/summarize_baseline_cases.py

The script prints JSON to standard output. The corresponding archived summary
is supplementary_analysis/baseline_case_summary.json. It also distinguishes
baseline errors from non-baseline regret outliers and reports the constant-
baseline weight-space benchmark from the existing sweep tables.

Important interpretation limits:

  * Annual heating energy is the archived daily heating series summed over the
    year and converted to GJ/year. In the released merged CSV files that series
    is the row labelled "Gas Consumption [J](Daily)", in joules per day. No
    efficiency and no fuel conversion are applied, so the quantity is not
    purchased gas, primary energy or measured demand.
  * Cost and product-stage GWP values are sums of component-specific unit rates;
    envelope areas are not applied, so they are comparative indices rather than
    whole-building totals.
  * The primary temperature indicator counts days when one zone's daily-mean air
    temperature is below 24 degrees C. It is a warm-side screen, not a complete
    thermal-comfort assessment.
  * Dataset label 2100 is retained for compatibility; its EPW header identifies
    an SSP5-8.5 Future Weather Generator 2080 timeframe, described in the paper
    as late-century.

In weather_file_descriptors.csv, HDD18 and CDD18 use daily-mean outdoor dry-bulb
temperature: sum(max(18 - T_day, 0)) and sum(max(T_day - 18, 0)). JJA P95 is the
linearly interpolated 95th percentile of the 92 June-August daily means. The
original Windows-side execution log did not retain EPW hashes; the listed
hashes identify the surviving files named by the later simulation runner.

Supplementary analysis and figure scripts (added with the revised manuscript):

    supplementary_analysis/
        mcda_supportedness_sweep.py   LP unique-selectability test for every exact
                                      Pareto member, 10,000-draw uniform weight
                                      sweep, ASF comparison with rho grid, all-625
                                      normalisation-anchor check, and warm-side
                                      baseline safeguard (writes mcda_outputs.csv)
        mcda_joint_sweep.py           joint weight x temperature-threshold sweep
                                      (writes mcda_joint_outputs.csv)
        mcda_weight_agreement.py      continuous weight-space selection agreement
                                      between surrogate and exact fronts, both task
                                      designs (writes mcda_weight_agreement.csv)
        mcda_central_weights.py       1,000,000-draw SMAA first-rank acceptability
                                      and central weight vectors per weather case
                                      (writes mcda_central_weights.csv)
        make_main_figures.py          shared entry point for manuscript Figures 2-5
        make_retrofit_figures.py      manuscript Figures 2-3 (package outcomes
                                      and floor-stratified paired comparisons)
        make_attainment_figure.py     entry point for the same Figure 2-3 pair
        make_attainment_s1_figure.py  Supplementary Figure S1.7 (attainment curves
                                      and matched component effects)
        make_framework_figure.py     manuscript Figure 1 (both candidate-pool
                                      evaluations and task designs)
        make_results_figures.py       workflow and warm-side figures
        make_fidelity_map_figure.py   front-recovery vs selection-agreement
                                      figure
        make_smaa_summary_figure.py   manuscript Figure 5 (SMAA acceptability +
                                      central-weight signatures + coverage bars)
        make_mcda_figures.py          Supplementary Figures S1.1-S1.2
        make_value_paths_figure.py    Appendix Figure B.1 and Supplementary
                                      Figure S1.4 (value paths of the exact fronts)
        make_decision_robustness_figure.py  Supplementary Figure S1.6 (cumulative
                                      coverage curves)
        make_graphical_abstract.py    graphical abstract (13 x 5 cm vector artwork)
        make_supplementary_tables.py  emits the generated subset of the tables;
                                      refuses to overwrite SUPPLEMENTARY_S1.md
                                      unless S1_ALLOW_OVERWRITE=1, because S1.0,
                                      S1.10 and S1.19 are hand-maintained
        make_s1_agreement_tables.py   Tables S1.13, S1.14 and S1.20
        recompute_decision_validation.py  decision tables from the released
                                      out-of-fold predictions
        mcda_outputs.csv              per-member unique-selectability class, LP
                                      margin t* and first-rank acceptability
        mcda_profile_sensitivity.csv  per-case per-profile weighted-sum and ASF
                                      selections, agreement, rho-grid stability
        mcda_joint_outputs.csv        joint weight x threshold acceptability
        mcda_weight_agreement.csv     per-composite weight-space agreement
        mcda_central_weights.csv      per-package acceptability and central weights
        within_fold_selection_check.py  Tables S1.21-S1.24 from retained outputs
        decision_fidelity_diagnostics/  held-out metrics, margins, tails and cases
        SUPPLEMENTARY_S1.md           compiled supplement tables S1.1-S1.24

Each script resolves this archive's results/ directory automatically when run
from inside supplementary_analysis/; set S1_EXHAUSTIVE, S1_SURROGATE, or FIGDIR
to override. Figure scripts write to FIGDIR (default: ./figures_out inside the
archive). All are deterministic (fixed seed 42 for the weight sweeps). Run
mcda_supportedness_sweep.py and mcda_joint_sweep.py before
make_supplementary_tables.py if regenerating the CSVs from scratch.

FAMILY-SCREEN ADDENDUM (four-head design)
- scripts/validate_mtl_family.py: extended validation script (architecture families, equal-weight
  and MGDA trainers, gradient-boosting comparator, --learned-targets two|four).
- results/surrogate_validation/: exact-index design outputs (two predicted, two exact).
- results/surrogate_validation_family4/: all-predicted design outputs (four objectives predicted).
- results/surrogate_validation_family4_dropout_diagnostic/: Deep-balanced dropout-0.1 diagnostic.
- supplementary_analysis/make_mtl_figures.py: manuscript Figure 4 (exact-index point
  accuracy and full-grid decision fidelity), the preserved all-predicted accuracy plot, and Supplementary Figures
  S1.3 (architecture) and S1.5 (clipping mechanism) generators.
- supplementary_analysis/make_mtl_s1_tables.py: regenerates Tables S1.3-S1.5, S1.9, S1.11-S1.12.
- supplementary_analysis/make_s1_agreement_tables.py: regenerates Tables S1.13, S1.14, S1.20.
- supplementary_analysis/recompute_decision_validation.py: regenerates and verifies
  the decision tables from the released out-of-fold predictions. Use its default
  verification mode with this archive.

MCDA SELECTION PROCEDURES
- Named-profile selections in mcda_supportedness_sweep.py use
  np.isclose(..., rtol=0.0, atol=1e-12), breaking ties by lowest simulation ID.
  The 10,000-draw sweeps use exact-equality argmin over ID-sorted scores,
  also breaking exact ties by lowest ID.
- Pareto members are classified as uniquely_selectable when the LP margin
  t* > 1e-9, and not_uniquely_selectable otherwise, under w_j >= 1e-6.
  mcda_outputs.csv records selectability and t_star in scientific notation.
- ASF selections are unchanged for rho in {1e-8, 1e-6, 1e-4, 1e-2}.
- Using Eq. (6) scaling constants over all 625 packages leaves all profile
  selections unchanged. A warm-side baseline safeguard requiring D24 >=
  same-case baseline changes only the late-century energy-priority selection,
  from package 250 to 400.
- mcda_weight_agreement.py generates Table S1.14; mcda_central_weights.py
  generates Table S1.16.
- tests/test_mcda_selection.py checks the tie rule, classification counts,
  ASF agreement, and sweep and joint coverage.

HELD-OUT-FOLD DECISION DIAGNOSTICS (Tables S1.21-S1.24)

From the archive root, postprocess the retained predictions without fitting models
or running EnergyPlus:

    python supplementary_analysis/within_fold_selection_check.py

The default output directory is supplementary_analysis/decision_fidelity_diagnostics/.
The --s1-root and --output-dir arguments allow explicit input and output paths.
The script uses NumPy and pandas from either documented environment and reads the
released OOF predictions, exact enumeration, and full-grid decision tables unchanged.
It writes model-level summaries, individual fold/profile decisions, exact score
margins, regret tails, physical case details, input SHA256 fingerprints, and the
Markdown/LaTeX source for Tables S1.21-S1.24. It does not replace the compiled
SUPPLEMENTARY_S1.md document.

Each formulation and task design has 3 repeat seeds x 5 held-out folds x 3 weather
cases = 45 evaluations, each containing the same 125 candidates for exact and
predicted comparisons, and 5 named profiles per evaluation = 225 profile decisions.
Pareto F1 and sampled-weight agreement are means and sample SDs over the 45
evaluations. Weight agreement uses the same 10,000 Dirichlet(1,1,1,1) draws (seed 42)
for each evaluation. Repeated partitions, weather cases and profiles are dependent.
Each front uses its own ideal/nadir; regret uses the exact front of the same
125-candidate subset. Subset nondominance and regret do not use the 625-package
reference. The retained full-grid composite contributes 9 repeat/weather evaluations
and 45 profile decisions per formulation/design. The regret-tail table pools five
formulations, giving 225 full-grid or 1,125 held-out profile comparisons per design;
separate mismatch-only rows exclude exact package agreements.

Across the five formulations in full-grid reconstruction, the exact-index design
has 127 of 133 mismatches (95.5%) on the exact front and median mismatch regret
0.0047. The all-predicted design has 160 of 180 (88.9%) and median 0.0164. Pooling
across both task designs gives 287 of 313 (91.7%) and median 0.0078.

Exact winner-to-runner-up margins are counted once per candidate set and profile:
15 full-grid and 225 held-out decisions. Their median exact-front-normalised margins
are 0.001705 (0.0017 rounded) and 0.006630, respectively. The observed relationship
between margin size and package agreement is descriptive, not causal: restricting
the candidate set also changes the front and its normalisation.

Manuscript Figure 4, fig_model_comparison.pdf/png, shows exact-index raw point
errors above and full-grid selection agreement and mismatch-regret distributions
for both designs below. Agreement uses all 45 named-profile cases per model and
design; regret uses only mismatches. Small points show individual mismatches,
large markers show medians, and thick segments span the interquartile range on
a logarithmic axis. These distributions use 25--42 cases per formulation and
design and are not confidence intervals. The 95th percentiles and maxima remain
in decision_fidelity_diagnostics/regret_tails.csv (scope full625, population
mismatches) and Table S1.23.
The four-objective all-predicted accuracy plot is preserved separately as
figures/fig_model_comparison_all_predicted.pdf/png. The --comparison-only option
of make_mtl_figures.py regenerates only manuscript Figure 4. The main figure
generator accepts S1_EXHAUSTIVE, MTL_SURROGATE, MTL4_SURROGATE and S1_ANALYSIS
for alternative input directories, and FIGDIR for output.

Regenerate the five main figures from the released tables without simulation or
model fitting, from the extracted archive root:

    FIGDIR="$PWD/figures" python supplementary_analysis/make_framework_figure.py
    FIGDIR="$PWD/figures" python supplementary_analysis/make_attainment_figure.py
    FIGDIR="$PWD/figures" python supplementary_analysis/make_mtl_figures.py --comparison-only
    FIGDIR="$PWD/figures" python supplementary_analysis/make_smaa_summary_figure.py

The attainment curves are retained as Supplementary Figure S1.7:

    FIGDIR="$PWD/figures" python supplementary_analysis/make_attainment_s1_figure.py

Place manuscript Figures 2 and 3 at 0.94 text width, Figure 4 at 0.95 and
Figure 5 at 0.98. Figure 2 (fig_package_tradeoffs.pdf/png) shows all 625
packages per weather case, heating reductions relative to each case's baseline,
and minimum-heating and illustrative four-objective equal-weight selections.
There are no two-objective Pareto rings. Figure 3
(fig_component_comparisons.pdf/png) shows all 125 matched pairs per component
and weather case. Window, facade and roof pairs are grouped by the unchanged
floor option: existing (25) or insulated (100). The ground-floor row changes
the floor option itself. Option changes include their infiltration assignments.
Day changes in both figures have the opposite sign to changes in D24.

The generator exports plotted package values, paired differences, pooled and
floor-stratified summaries, and source hashes in figures/retrofit_tradeoff_data/.
These are deterministic descriptive comparisons of the retained simulations;
their ranges and interquartile ranges are not confidence intervals.

Table S1.4 now presents the exact-index design first and the all-predicted design
second, matching Appendix Table B.4. To refresh only this table:

    python supplementary_analysis/make_mtl_s1_tables.py --weather-only

The environment used for the revised main figures is recorded in
supplementary_analysis/figure_environment.json.
