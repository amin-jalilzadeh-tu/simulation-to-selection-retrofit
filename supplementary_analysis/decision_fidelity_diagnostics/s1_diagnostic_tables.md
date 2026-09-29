## Table S1.21. Decision fidelity within held-out folds

Decision fidelity within each held-out fold. Every comparison uses the same 125 candidates for exact and predicted decisions. Each formulation and task design has 45 repeat–fold–weather evaluations and 225 named-profile comparisons. F1 and sampled-weight agreement are means with sample standard deviations over the 45 evaluations; these evaluations share data. Nondominance and regret refer to the exact front of the 125 candidates. The nondominated percentage and median regret include mismatches only.

Each formulation and task design has 3 repeat seeds × 5 held-out folds × 3 weather cases = 45 candidate-set evaluations, with 5 profiles each (225 profile comparisons). Sampled-weight agreement uses the same 10,000 Dirichlet(1,1,1,1) draws (seed 42) in every evaluation. Both fronts use their own ideal/nadir normalisation; selected packages are scored with the exact-front normalisation. Raw predictions are clipped for decision analysis as in Methods. These repeated comparisons are dependent, and the reported SDs describe variation across the 45 evaluations rather than uncertainty from independent deployment trials.

| Design | Formulation | Pareto F1 (mean ± SD) | Profile matches / 225 (%) | Weight agreement, % (mean ± SD) | Mismatches on subset exact front (%) | Median mismatch regret | Maximum regret |
|---|---|---:|---:|---:|---:|---:|---:|
| all-predicted | Equally weighted joint network | 0.860 ± 0.028 | 88/225 (39.1) | 39.9 ± 15.0 | 95.6 | 0.0159 | 0.4719 |
| all-predicted | Gradient-balanced joint network | 0.854 ± 0.028 | 80/225 (35.6) | 38.1 ± 14.6 | 95.2 | 0.0122 | 0.4441 |
| all-predicted | Single-task networks | 0.865 ± 0.030 | 95/225 (42.2) | 42.1 ± 12.6 | 100.0 | 0.0116 | 0.1158 |
| all-predicted | Random forest | 0.882 ± 0.023 | 123/225 (54.7) | 52.2 ± 12.5 | 100.0 | 0.0083 | 0.0706 |
| all-predicted | Gradient-boosted trees | 0.918 ± 0.023 | 147/225 (65.3) | 72.1 ± 11.3 | 100.0 | 0.0053 | 0.1109 |
| exact-index | Equally weighted joint network | 0.942 ± 0.023 | 148/225 (65.8) | 75.4 ± 9.3 | 98.7 | 0.0033 | 0.1215 |
| exact-index | Gradient-balanced joint network | 0.941 ± 0.021 | 147/225 (65.3) | 73.7 ± 12.4 | 97.4 | 0.0039 | 0.4441 |
| exact-index | Single-task networks | 0.939 ± 0.024 | 143/225 (63.6) | 72.5 ± 15.2 | 100.0 | 0.0045 | 0.1158 |
| exact-index | Random forest | 0.934 ± 0.022 | 156/225 (69.3) | 70.8 ± 11.9 | 100.0 | 0.0045 | 0.0782 |
| exact-index | Gradient-boosted trees | 0.939 ± 0.025 | 160/225 (71.1) | 77.7 ± 9.6 | 100.0 | 0.0039 | 0.1109 |

Pooling the five exact-index formulations, 368 of 371 mismatches (99.2%) remain nondominated within their 125-candidate set; their median regret is 0.0042. Only 80.3% are also on the 625-package exact front. The subset and full-grid references therefore should not be interchanged.

## Table S1.22. Exact winner-to-runner-up score margins

Margins are the second-lowest minus the lowest weighted score among exact-front members, using each candidate set’s own exact-front ideal/nadir. The full-grid calculation counts 15 weather–profile decisions once each; the held-out calculation counts 225 repeat–fold–weather–profile decisions once each. Values are not duplicated across formulations or task designs.

| Candidate set | Profile | Decisions | Margin < 0.004 (%) | Median margin |
|---|---|---:|---:|---:|
| Full grid (625) | All profiles | 15 | 80.0 | 0.00171 |
| Full grid (625) | Equal weight | 3 | 100.0 | 0.00102 |
| Full grid (625) | GWP priority | 3 | 100.0 | 0.00179 |
| Full grid (625) | Cost priority | 3 | 0.0 | 0.04331 |
| Full grid (625) | Warm-side priority | 3 | 100.0 | 0.00185 |
| Full grid (625) | Heating priority | 3 | 100.0 | 0.00105 |
| Held-out (125) | All profiles | 225 | 37.8 | 0.00663 |
| Held-out (125) | Equal weight | 45 | 17.8 | 0.00663 |
| Held-out (125) | GWP priority | 45 | 20.0 | 0.01913 |
| Held-out (125) | Cost priority | 45 | 13.3 | 0.02087 |
| Held-out (125) | Warm-side priority | 45 | 51.1 | 0.00370 |
| Held-out (125) | Heating priority | 45 | 86.7 | 0.00191 |

The full-grid median is 0.001705 (0.0017 rounded), and the held-out-subset median is 0.006630. Among exact-index comparisons within held-out sets, identity agreement is 37.4% when the exact margin is below 0.004 and 85.0% otherwise. This is a descriptive association: reducing the candidate count also changes the front and its normalisation, so the comparison does not isolate a causal effect of score margins or cross-fitting. The diagnostic CSV also reports the runner-up among all candidates under the same exact-front scaling; this differs from the front-only runner-up in one of the 225 held-out decisions, without changing its classification at 0.004.

## Table S1.23. Regret tails for full-grid and held-out decisions

Values pool the five main formulations. All-profile rows include exact package agreements (zero regret); mismatch rows exclude them. Each formulation contributes 45 full-grid profile comparisons (3 repeats × 3 weather cases × 5 profiles) or 225 held-out profile comparisons (3 repeats × 5 folds × 3 weather cases × 5 profiles). The 95th percentile uses linear interpolation. Regret uses the exact-front scaling of the corresponding 625- or 125-candidate set, so magnitudes across candidate-set sizes use different anchors.

| Candidate set | Design | Population | Comparisons | Median | 95th percentile | Maximum | Regret > 0.05, count (%) |
|---|---|---|---:|---:|---:|---:|---:|
| Full grid (625) | all-predicted | All profiles | 225 | 0.0078 | 0.0696 | 0.4723 | 26 (11.6) |
| Full grid (625) | all-predicted | Mismatches | 180 | 0.0164 | 0.0939 | 0.4723 | 26 (14.4) |
| Full grid (625) | exact-index | All profiles | 225 | 0.0010 | 0.0400 | 0.1174 | 8 (3.6) |
| Full grid (625) | exact-index | Mismatches | 133 | 0.0047 | 0.0767 | 0.1174 | 8 (6.0) |
| Held-out (125) | all-predicted | All profiles | 1125 | 0.0008 | 0.0542 | 0.4719 | 72 (6.4) |
| Held-out (125) | all-predicted | Mismatches | 592 | 0.0111 | 0.0747 | 0.4719 | 72 (12.2) |
| Held-out (125) | exact-index | All profiles | 1125 | 0.0000 | 0.0209 | 0.4441 | 20 (1.8) |
| Held-out (125) | exact-index | Mismatches | 371 | 0.0042 | 0.0563 | 0.4441 | 20 (5.4) |

## Table S1.24. Physical outcomes of the largest exact-index held-out regret

Physical outcomes and weighted contributions for the largest held-out-subset regret under the exact-index design. The gradient-balanced joint network (repeat seed 43, fold 1) selects package 1 instead of package 145 under present weather and the warm-side priority profile. Positive contributions increase regret; index savings partially offset the heating and warm-side penalties.

| Quantity | Exact selection (145) | Surrogate selection (1) | Weighted contribution to regret |
|---|---:|---:|---:|
| Annual heating, $E_H$ (GJ/year) | 22.712 | 37.571 | +0.07406 |
| Cost index, $I_C$ (points) | 507.000 | 0.000 | -0.04734 |
| GWP index, $I_G$ (points) | 97.000 | 0.000 | -0.04932 |
| Warm-side count, $D_{24}$ (days) | 365.000 | 361.000 | +0.46667 |
| Total regret | | | 0.44406 |

The unrenovated package selected by the surrogate uses 14.86 GJ/year more heating and has four fewer days below 24 °C than the exact selection. The exact subset front spans six $D_{24}$ days, giving the warm-side contribution $0.7 × 4/6 = 0.46667$. Cost and GWP index savings partly offset that penalty. Both selected packages are nondominated within the subset and within the full grid. Raw $D_{24}$ predictions for the exact and surrogate selections are 365.68 and 371.77 days, respectively, and both enter the decision analysis at the clipped value of 365 days.

The retained exact-output checks in `physical_reference_checks.json` also confirm that the late-century façade/roof warm-side-priority package (ID 15) cuts heating by 30.02% and gains five $D_{24}$ days relative to the unrenovated package. Among the minimum-heating selections across the three weather cases, the late-century selection has the largest warm-side penalty (57 additional warm days). Across all 625 late-century packages, the largest penalty is 59 days (IDs 580, 604 and 605).

Tables S1.21–S1.24 are generated from retained outputs by `supplementary_analysis/within_fold_selection_check.py`. Machine-readable tables, case details, input SHA256 fingerprints and LaTeX tables are in `supplementary_analysis/decision_fidelity_diagnostics/`. No model fitting or EnergyPlus execution is performed.
