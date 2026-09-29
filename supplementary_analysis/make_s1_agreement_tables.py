"""Generate Tables S1.13, S1.14 and S1.20 of SUPPLEMENTARY_S1.md.

S1.13  substitution-only ablation: predicted E_H and D24 from the four-target
       models with the two indices taken from the component tables, so the
       effect of substituting the indices is separated from the effect of
       refitting on two targets.
S1.14  weight-space selection agreement for all nine formulations, all three
       arms (regenerated here because the arm set changed).
S1.20  normalisation-anchor check and training-cap counts.

Inputs
  mcda_weight_agreement.csv                     written by mcda_weight_agreement.py
  mcda_common_anchor.csv                        written by mcda_common_anchor.py
  <results>/*/weighted_profile_validation*.csv  written by recompute_decision_validation.py
  <results>/surrogate_validation_family4/model_metadata.json

The splice is idempotent: each table is replaced in place when present.
"""
import json
import os
import re

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
S1MD = os.path.join(_HERE, "SUPPLEMENTARY_S1.md")
RESULTS = os.environ.get(
    "S1_RESULTS", os.path.join(_HERE, os.pardir, ".build", "s1data",
                               "Supplementary_File_S1_analysis_reproducibility", "results"))
FOUR = os.path.join(RESULTS, "surrogate_validation_family4")
TWO = os.path.join(RESULTS, "surrogate_validation")

MAIN5 = ["shared_mtl_nn", "shared_mtl_nn_mgda", "independent_stl_nn",
         "random_forest", "gradient_boosting"]
LABEL = {
    "shared_mtl_nn": "Joint NN, equal weighting",
    "shared_mtl_nn_mgda": "Joint NN, gradient-balanced",
    "independent_stl_nn": "Single-task NNs",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient-boosted trees",
    "separate_mtl_nn": "Separate family, equal weighting",
    "separate_mtl_nn_mgda": "Separate family, gradient-balanced",
    "deep_balanced_mtl_nn": "Deep-balanced family, equal weighting",
    "deep_balanced_mtl_nn_mgda": "Deep-balanced family, gradient-balanced",
}
ORDER = MAIN5 + ["separate_mtl_nn", "separate_mtl_nn_mgda",
                 "deep_balanced_mtl_nn", "deep_balanced_mtl_nn_mgda"]


def _round_half_up(value, digits=1):
    factor = 10 ** digits
    return float(int(value * factor + (0.5 if value >= 0 else -0.5))) / factor


wa = pd.read_csv(os.path.join(_HERE, "mcda_weight_agreement.csv"))
sel = {
    "all-predicted": pd.read_csv(os.path.join(FOUR, "weighted_profile_validation.csv")),
    "substitution-only": pd.read_csv(
        os.path.join(FOUR, "weighted_profile_validation_substitution_only.csv")),
    "exact-index": pd.read_csv(os.path.join(TWO, "weighted_profile_validation.csv")),
}
ARMS = ["all-predicted", "substitution-only", "exact-index"]


def agreement(design, model):
    values = wa[(wa.task_design == design) & (wa.model == model)].agreement_pct
    if values.empty:
        raise SystemExit(f"mcda_weight_agreement.csv has no rows for {design}/{model}")
    return _round_half_up(values.mean()), _round_half_up(values.std(ddof=1))


def exact_selections(design, model):
    frame = sel[design]
    return int(frame[frame.model == model].selection_agreement.sum())


# ---------------------------------------------------------------- S1.13
def _raw_mean(arm, model):
    return wa[(wa.task_design == arm) & (wa.model == model)].agreement_pct.mean()


mean_ws = {arm: sum(_raw_mean(arm, m) for m in MAIN5) / len(MAIN5) for arm in ARMS}
tot_sel = {arm: sum(exact_selections(arm, m) for m in MAIN5) for arm in ARMS}
share = ((mean_ws["substitution-only"] - mean_ws["all-predicted"])
         / (mean_ws["exact-index"] - mean_ws["all-predicted"]) * 100.0)

rows13 = []
for model in MAIN5:
    cells = [LABEL[model]]
    for arm in ARMS:
        mean, _ = agreement(arm, model)
        cells += [f"{mean:.1f}", f"{exact_selections(arm, model)}"]
    rows13.append("| " + " | ".join(cells) + " |")
mean_cells = []
for arm in ARMS:
    mean_cells += [f"**{mean_ws[arm]:.1f}**", f"**{tot_sel[arm]}**"]
rows13.append("| **Mean over the five / total of 225** | " + " | ".join(mean_cells) + " |")

t113 = "\n".join([
    "## Table S1.13: Substitution-only ablation",
    "",
    "The exact-index design changes two things at once for the joint networks and the "
    "random forest: the two indices are supplied from the component tables, and the "
    "model is refitted on two targets instead of four. The substitution-only arm "
    "changes only the first. It retains the four-target models' predicted $E_H$ and "
    "$D_{24}$ and replaces the two indices by their table values, with no model "
    "refitted, so the two effects can be read separately. The single-task networks and "
    "the per-objective boosted trees fit each target independently, so for them the "
    "substitution-only and exact-index arms coincide by construction.",
    "",
    "Agreement is the mean over the nine repeat-weather case composites, per cent; "
    "selections are exact profile selections of 45.",
    "",
    "| Formulation | all-predicted agr. | all-predicted sel. | substitution-only agr. "
    "| substitution-only sel. | exact-index agr. | exact-index sel. |",
    "|---|---|---|---|---|---|---|",
    *rows13,
    "",
    f"Substituting the indices accounts for {share:.0f} per cent of the mean "
    f"weight-space gain between the all-predicted and exact-index arms "
    f"({mean_ws['all-predicted']:.1f} to {mean_ws['substitution-only']:.1f} against "
    f"{mean_ws['exact-index']:.1f} per cent). Exact selections move the same way, "
    f"{tot_sel['all-predicted']} of 225 to {tot_sel['substitution-only']} of 225 under "
    f"substitution alone and {tot_sel['exact-index']} of 225 after refitting. The "
    "improvement therefore came from substituting the deterministic indices, while "
    "refitting contributed the remainder.",
    "",
])

# ---------------------------------------------------------------- S1.14
rows14 = []
for arm in ARMS:
    for i, model in enumerate(ORDER):
        mean, sd = agreement(arm, model)
        rows14.append(f"| {arm if i == 0 else ''} | {LABEL[model]} | {mean:.1f} ± {sd:.1f} |")

t114 = "\n".join([
    "## Table S1.14: Weight-space selection agreement, all nine formulations",
    "",
    "Share of the 10,000 uniform Dirichlet(1,1,1,1) weight draws (seed 42, the Section 3.4",
    "draws) for which the reconstructed and exact fronts select the same package under the",
    "identical Eq. (6)-(7) protocol with the sweep tie rule. Predictions are returned to their",
    "feasible ranges before decision analysis: $D_{24}$ to [0, 365] and the predicted cost and",
    "GWP indices to [0, inf). Mean ± SD over the nine repeat-weather case composites; the SD",
    "describes between-composite variation, not a confidence interval, since all composites",
    "share the same draws. Generated by `mcda_weight_agreement.py`; per-composite values in",
    "`mcda_weight_agreement.csv`. The first five rows of each arm are the main-text",
    "formulations (Table 3).",
    "",
    "| Task design | Model | Weight-space agreement (%) |",
    "|---|---|---|",
    *rows14,
    "",
])

# ---------------------------------------------------------------- S1.20
anchor = pd.read_csv(os.path.join(_HERE, "mcda_common_anchor.csv"))
anchor["delta"] = anchor.common_anchor_pct - anchor.own_anchor_pct
rows20 = []
for design in ("all-predicted", "exact-index"):
    for i, model in enumerate(MAIN5):
        sub = anchor[(anchor.task_design == design) & (anchor.model == model)]
        rows20.append(
            f"| {design if i == 0 else ''} | {LABEL[model]} "
            f"| {_round_half_up(sub.own_anchor_pct.mean()):.1f} "
            f"| {_round_half_up(sub.common_anchor_pct.mean()):.1f} "
            f"| {sub.delta.mean():+.2f} |")

meta = json.load(open(os.path.join(FOUR, "model_metadata.json")))
cap = meta["configuration"]["max_epochs"]
counts = {}
for run in meta["training_runs"]:
    # Parameter-count summaries are metadata records, not fitted networks.
    if "epochs_ran" not in run:
        continue
    model = run.get("model")
    total, hit = counts.get(model, (0, 0))
    counts[model] = (total + 1, hit + (1 if run.get("epochs_ran") == cap else 0))
rows_cap = [f"| {LABEL[m]} | {counts[m][1]} of {counts[m][0]} |"
            for m in ORDER if m in counts and counts[m][1]]

t120 = "\n".join([
    "## Table S1.20: Normalisation anchors and the training cap",
    "",
    "**Normalisation anchors.** The exact and reconstructed fronts are each normalised over",
    "their own ideal and nadir, so a reconstructed front with a different range could in",
    "principle disagree for reasons of scaling rather than of package ordering. Repeating the",
    "weight-space agreement with common anchors --- the exact front's ideal and nadir applied",
    "to both --- moves agreement by 0.89 points on average across the 90 composites. The",
    "largest formulation-level mean shift is 2.52 points and the largest single-composite",
    "shift is 4.20 points. Written by `mcda_common_anchor.py` to `mcda_common_anchor.csv`.",
    "",
    "| Task design | Model | Own anchors (%) | Common anchors (%) | Difference |",
    "|---|---|---|---|---|",
    *rows20,
    "",
    f"**Training cap.** Neural training stops early on the inner validation partition or at",
    f"{cap} epochs, whichever comes first. The runs that reached the cap are:",
    "",
    "| Formulation | Runs at the cap |",
    "|---|---|",
    *rows_cap,
    "",
    "Nine of the fifteen equally weighted joint-network runs reached the 300-epoch cap; the",
    "reported results for that formulation are conditional on this training budget.",
    "",
])

# ---------------------------------------------------------------- splice
src = open(S1MD).read()


def _pos(text, number):
    match = re.search(r"^## Table S1\.%s(?!\d)" % re.escape(number), text, re.M)
    return None if match is None else match.start()


def splice(text, number, new, before):
    """Replace the S1.<number> block if present, else insert it before S1.<before>."""
    start = _pos(text, number)
    if start is not None:
        end = _pos(text, before)
        if end is None or end < start:
            raise SystemExit(f"cannot bound Table S1.{number}: S1.{before} missing or earlier")
        return text[:start] + new + "\n" + text[end:]
    at = _pos(text, before)
    if at is None:
        return text.rstrip() + "\n\n" + new
    return text[:at] + new + "\n" + text[at:]


src = splice(src, "13", t113, "14")
src = splice(src, "14", t114, "15")
# S1.20 must be bounded, not truncated to end-of-file: later tables follow it.
start20 = _pos(src, "20")
if start20 is not None:
    after = _pos(src, "21")
    src = (src[:start20] + t120 + "\n" + src[after:]) if after is not None else src[:start20] + t120
else:
    after = _pos(src, "21")
    src = (src[:after] + t120 + "\n" + src[after:]) if after is not None else src.rstrip() + "\n\n" + t120
open(S1MD, "w").write(src)
print(f"S1 updated: S1.13 ({len(rows13)} rows), S1.14 ({len(rows14)} rows), "
      f"S1.20 ({len(rows20)} anchor rows, {len(rows_cap)} cap rows)")
