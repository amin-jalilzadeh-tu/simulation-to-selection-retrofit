# Decision fidelity diagnostics

Generated from retained out-of-fold predictions and exact enumeration outputs.
No model is fitted and no physical simulation is run. Inputs remain read-only.

The script requires Python, NumPy and pandas. Install it in the S1 package's
`supplementary_analysis` directory to use its package-relative defaults, or pass
`--s1-root` and `--output-dir` explicitly.

- `summary.csv`: 45 held-out-fold/weather evaluations and 225 profile comparisons
  per model/task design. Weight agreement uses the same 10,000 Dirichlet(1,1,1,1)
  draws (seed 42) for every evaluation.
- `group_metrics.csv` and `profile_decisions.csv`: underlying 125-package results.
- `exact_score_margins.csv`: 15 full-grid and 225 held-out-subset exact decisions,
  counted once each rather than duplicated over models or task designs. The main
  margin compares the best two exact-front members. An additional column compares
  the best two candidates anywhere in the candidate set under the same scaling.
- `margin_summary.csv`: summaries of the exact-front margins.
- `margin_identity_association.csv`: descriptive association between exact score
  margin and surrogate identity agreement. Shared data and profiles limit causal
  interpretation.
- `regret_tails.csv`: median, linearly interpolated 95th percentile, maximum and
  share strictly above 0.05, separately for all comparisons and mismatches only.
  Full-grid and held-out-subset regret use their respective exact-front scaling.
- `worst_exact_index_within_fold.json`: physical outcomes, raw predictions and
  additive weighted contributions for the largest exact-index subset regret.
- `physical_reference_checks.json`: retained simulation checks of the headline
  façade/roof comparison and minimum-heating packages across weather cases.
- `s1_diagnostic_tables.md` and `.tex`: Tables S1.21–S1.24, including model-level
  held-out-fold fidelity. LaTeX requires booktabs and graphicx.

Clipping follows Methods: D24 to [0,365], predicted cost/GWP to nonnegative values.
Each exact and reconstructed front uses its own ideal/nadir. Named profiles break
ties within 1e-12 by lowest simulation ID; sampled weights use exact argmin over
ID-sorted candidates. A lower bound or zero-width objective is handled as in S1.
Nondominance and regret in subset evaluations concern those 125 candidates.
Shrinking the candidate set also changes the decision problem; differences from
the 625-package composite do not isolate an effect of cross-fitting alone.
Exact-front winner-to-runner-up margins have medians 0.001705 (full grid) and
0.006630 (held-out subsets). Margin/identity association is descriptive, not causal.
