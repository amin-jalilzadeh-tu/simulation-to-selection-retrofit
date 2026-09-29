"""Regenerate the surrogate tables of SUPPLEMENTARY_S1.md from the family-screen runs.

Replaces Tables S1.3-S1.5 and S1.9 with five-formulation, two-task-design versions,
appends Tables S1.11 (full nine-model grids) and S1.12 (architecture screen), and
extends the S1.0 methods notes (trainers, boosting comparator, task designs).
Every value is computed from the run CSVs; nothing is transcribed by hand.
"""
import os
import argparse
import re
from collections import defaultdict

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--weather-only", action="store_true",
                    help="Update only Table S1.4 from the released weather metrics.")
ARGS = parser.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
S1MD = os.path.join(HERE, "SUPPLEMENTARY_S1.md")
RUNS = {
    "all-predicted": os.environ.get(
        "MTL4_SURROGATE",
        os.path.join(HERE, os.pardir, "results", "surrogate_validation_family4")),
    "exact-index": os.environ.get(
        "MTL2_SURROGATE",
        os.path.join(HERE, os.pardir, "results", "surrogate_validation")),
}
DIAG = os.environ.get(
    "MTL_DIAG",
    os.path.join(HERE, os.pardir, "results",
                 "surrogate_validation_family4_dropout_diagnostic"))

FIVE = ["shared_mtl_nn", "shared_mtl_nn_mgda", "independent_stl_nn",
        "random_forest", "gradient_boosting"]
NINE = FIVE[:2] + ["separate_mtl_nn", "separate_mtl_nn_mgda",
                   "deep_balanced_mtl_nn", "deep_balanced_mtl_nn_mgda"] + FIVE[2:]
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
TGT = {
    "annual_heating_energy_gj": ("E_H (GJ/year)", 3),
    "days_below_24_c": ("D24 (days/year)", 3),
    "cost_rate_sum_proxy": ("I_C (index points)", 1),
    "carbon_rate_sum_proxy": ("I_G (index points)", 2),
}
HZ = {2020: "Present", 2050: "Mid-century", 2100: "Late-century"}


def load(design):
    d = RUNS[design]
    return {
        "fold": pd.read_csv(os.path.join(d, "fold_metrics.csv")),
        "hfold": pd.read_csv(os.path.join(d, "horizon_fold_metrics.csv")),
        "pareto": pd.read_csv(os.path.join(d, "surrogate_pareto_validation.csv")),
        "sel": pd.read_csv(os.path.join(d, "weighted_profile_validation.csv")),
    }


DATA = {k: load(k) for k in RUNS}


def ms(series, dec):
    return f"{series.mean():.{dec}f} ± {series.std(ddof=1):.{dec}f}"


def acc_rows(fold, models, targets, dec_r2=4):
    rows = []
    for m in models:
        first = True
        for t in targets:
            lab, dec = TGT[t]
            sub = fold[(fold.model == m) & (fold.target == t)]
            mae = sub[sub.metric == "mae"]["value"]
            rmse = sub[sub.metric == "rmse"]["value"]
            r2 = sub[sub.metric == "r2"]["value"]
            rows.append(f"| {LABEL[m] if first else ''} | {lab} | {ms(mae, dec)} | "
                        f"{ms(rmse, dec)} | {ms(r2, dec_r2)} |")
            first = False
    return rows


# ---------------------------------------------------------------- S1.3
t13 = ["## Table S1.3: Out-of-fold prediction accuracy, pooled over weather cases "
       "(all-predicted task design)", "",
       "Mean ± SD over the 15 package-grouped outer folds (3 seeds × 5 folds); folds "
       "share packages across seeds and are not independent replicates. All four "
       "objectives are predicted; exact-index accuracies for the two simulated "
       "outcomes are in Table S1.11.", "",
       "| Model | Target | MAE | RMSE | R² |", "|---|---|---|---|---|"]
t13 += acc_rows(DATA["all-predicted"]["fold"], FIVE, list(TGT))
t13 = "\n".join(t13) + "\n"

# ---------------------------------------------------------------- S1.4
t14 = ["## Table S1.4: Weather-case-specific prediction accuracy, both task designs", "",
       "Mean ± descriptive SD over 15 package-grouped folds. The cost and carbon "
       "indices are identical across weather cases, so only the two simulated "
       "outcomes are shown per case. E_H MAE is in GJ/year and D24 MAE in days.", ""]
for design in ("exact-index", "all-predicted"):
    t14 += [f"**{design.capitalize()} task design**", "",
            "| Model | Weather case | E_H MAE | E_H R² | D24 MAE | D24 R² |",
            "|---|---|---|---|---|---|"]
    hf = DATA[design]["hfold"]
    for m in FIVE:
        first = True
        for hz in (2020, 2050, 2100):
            sub = hf[(hf.model == m) & (hf.horizon == hz)]
            e = sub[sub.target == "annual_heating_energy_gj"]
            d = sub[sub.target == "days_below_24_c"]
            t14.append(
                f"| {LABEL[m] if first else ''} | {HZ[hz]} | "
                f"{ms(e[e.metric == 'mae']['value'], 3)} | {ms(e[e.metric == 'r2']['value'], 4)} | "
                f"{ms(d[d.metric == 'mae']['value'], 3)} | {ms(d[d.metric == 'r2']['value'], 4)} |")
            first = False
    t14.append("")
t14 = "\n".join(t14) + "\n"

if ARGS.weather_only:
    with open(S1MD) as stream:
        src = stream.read()
    src, count = re.subn(r"^## Table S1\.4(?!\d).*?(?=^## Table S1\.5(?!\d))",
                         lambda match: t14 + "\n", src, count=1,
                         flags=re.M | re.S)
    if count != 1:
        raise SystemExit("Table S1.4 block not found")
    with open(S1MD, "w") as stream:
        stream.write(src)
    print("Updated Table S1.4 only:", S1MD)
    raise SystemExit(0)

# ---------------------------------------------------------------- S1.5
t15 = ["## Table S1.5: Decision-fidelity summary, both task designs", "",
       "Pareto precision/recall/F1: mean ± SD over the 9 seed–weather case composite "
       "fronts. Exact agreement: profile selections (of 15 unique weather case–profile "
       "cases) reproduced exactly, per repeat seed. Regret: Eq. 10 true-score regret "
       "over all 45 selection cases. In the all-predicted design the predicted fronts "
       "use predicted cost/carbon; the exact reference never changes.", "",
       "| Task design | Model | Precision | Recall | F1 | Exact agreement "
       "(seeds 17/29/43) | On exact front | Regret median | Regret p90 | Regret max |",
       "|---|---|---|---|---|---|---|---|---|---|"]
for design in ("all-predicted", "exact-index"):
    pv, sel = DATA[design]["pareto"], DATA[design]["sel"]
    first = True
    for m in FIVE:
        p = pv[pv.model == m]
        w = sel[sel.model == m]
        per_seed = "/".join(str(int(w[w.repeat_seed == s].selection_agreement.sum()))
                            for s in (17, 29, 43))
        t15.append(
            f"| {design if first else ''} | {LABEL[m]} | {ms(p.precision, 3)} | "
            f"{ms(p.recall, 3)} | {ms(p.f1, 3)} | {per_seed} of 15 | "
            f"{int(w.predicted_selection_on_exact_pareto.sum())} of 45 | "
            f"{w.true_score_regret.median():.4f} | "
            f"{w.true_score_regret.quantile(0.9):.3f} | "
            f"{w.true_score_regret.max():.3f} |")
        first = False
t15 = "\n".join(t15) + "\n"

# ---------------------------------------------------------------- S1.9
t19 = ["## Table S1.9: Per repeat–weather case Pareto precision and recall", "",
       "Values are per-composite (seed × weather case); seeds ordered 17/29/43.", ""]
for design in ("all-predicted", "exact-index"):
    t19 += [f"**{design.capitalize()} task design**", "",
            "| Model | Weather case | Precision (seeds 17/29/43) | Recall (seeds 17/29/43) |",
            "|---|---|---|---|"]
    pv = DATA[design]["pareto"]
    for m in FIVE:
        first = True
        for hz in (2020, 2050, 2100):
            sub = pv[(pv.model == m) & (pv.horizon == hz)].sort_values("repeat_seed")
            prec = "/".join(f"{v:.3f}" for v in sub.precision)
            rec = "/".join(f"{v:.3f}" for v in sub.recall)
            t19.append(f"| {LABEL[m] if first else ''} | {HZ[hz]} | {prec} | {rec} |")
            first = False
    t19.append("")
t19 = "\n".join(t19) + "\n"

# ---------------------------------------------------------------- S1.11
t111 = ["## Table S1.11: Full model screen, both task designs (nine formulations)", "",
        "Pooled out-of-fold MAE (mean over 15 folds) per objective, with Pareto recall "
        "(mean over 9 composites) and exact selection agreement (of 45). The Separate "
        "and Deep-balanced rows use the alternative architectures described in "
        "Table S1.12.", ""]
for design in ("all-predicted", "exact-index"):
    fold, pv, sel = (DATA[design][k] for k in ("fold", "pareto", "sel"))
    targets = [t for t in TGT if not fold[fold.target == t].empty]
    header = "| Model | " + " | ".join(TGT[t][0] + " MAE" for t in targets) + \
             " | Pareto recall | Agreement |"
    t111 += [f"**{design.capitalize()} task design**", "", header,
             "|" + "---|" * (len(targets) + 3)]
    for m in NINE:
        if fold[fold.model == m].empty:
            continue
        cells = []
        for t in targets:
            dec = TGT[t][1]
            v = fold[(fold.model == m) & (fold.target == t) & (fold.metric == "mae")]["value"].mean()
            cells.append(f"{v:.{dec}f}")
        rec = pv[pv.model == m].recall.mean()
        agr = int(sel[sel.model == m].selection_agreement.sum())
        t111.append(f"| {LABEL[m]} | " + " | ".join(cells) +
                    f" | {rec:.3f} | {agr} of 45 |")
    t111.append("")
t111 = "\n".join(t111) + "\n"

# ---------------------------------------------------------------- S1.12
diag = pd.read_csv(os.path.join(DIAG, "fold_metrics.csv"))
db = diag[(diag.model == "deep_balanced_mtl_nn")]
db_e = db[(db.target == "annual_heating_energy_gj") & (db.metric == "mae")]["value"].mean()
db_d = db[(db.target == "days_below_24_c") & (db.metric == "mae")]["value"].mean()
main = DATA["exact-index"]["fold"]
mb = main[(main.model == "deep_balanced_mtl_nn") & (main.repeat_seed == 17)] \
    if "repeat_seed" in main.columns else None


def fold_mae(design, model, target):
    f = DATA[design]["fold"]
    return f[(f.model == model) & (f.target == target)
             & (f.metric == "mae")]["value"].mean()


def agree45(design, model):
    s = DATA[design]["sel"]
    return int(s[s.model == model].selection_agreement.sum())


SEP, SHR = "separate_mtl_nn", "shared_mtl_nn"
sep_e = fold_mae("all-predicted", SEP, "annual_heating_energy_gj")
shr_e = fold_mae("all-predicted", SHR, "annual_heating_energy_gj")
sep_d = fold_mae("all-predicted", SEP, "days_below_24_c")
shr_d = fold_mae("all-predicted", SHR, "days_below_24_c")
sep_c = fold_mae("all-predicted", SEP, "cost_rate_sum_proxy")
shr_c = fold_mae("all-predicted", SHR, "cost_rate_sum_proxy")
sep_g = fold_mae("all-predicted", SEP, "carbon_rate_sum_proxy")
shr_g = fold_mae("all-predicted", SHR, "carbon_rate_sum_proxy")
sep_a = agree45("all-predicted", SEP)
shr_a = agree45("all-predicted", SHR)

# Inner-training sample size, per outer fold, from the released split assignments.
split = pd.read_csv(os.path.join(RUNS["all-predicted"], "split_assignments.csv"))
itr = split[split.role == "inner_train"].groupby(["repeat_seed", "outer_fold"])
n_pkg = itr.configuration_group_id.nunique().unique()
n_row = itr.size().unique()
if n_pkg.size != 1 or n_row.size != 1:
    raise ValueError(f"inner_train size varies across folds: {n_pkg}, {n_row}")
n_pkg, n_row = int(n_pkg[0]), int(n_row[0])

t112 = "\n".join([
    "## Table S1.12: Architecture screen and trainer notes", "",
    "Two further architecture families were screened on identical folds (full rows "
    "in Table S1.11): the Separate family (one shared layer, deep task branches, "
    "linear outputs for standardised targets) and the Deep-balanced "
    "family (200–100 shared trunk with dropout 0.5, asymmetric task heads). The "
    "deeper architecture did not consistently improve prediction accuracy or "
    "selection agreement relative to the shared-trunk network. "
    "The Separate family with equal weighting improves on the shared trunk on the "
    "two simulated outcomes in the all-predicted design (heating MAE "
    f"{sep_e:.3f} against {shr_e:.3f} GJ/year; day-count MAE {sep_d:.3f} against "
    f"{shr_d:.3f} days/year) but is worse on the two rate-table indices (cost "
    f"{sep_c:.1f} against {shr_c:.1f}; GWP {sep_g:.2f} against {shr_g:.2f} index "
    f"points) and reproduces {sep_a} of the 45 exact selections against {shr_a}; "
    "in the exact-index design it is behind the shared trunk on both simulated "
    "outcomes and on exact selection agreement. Neither family displaces the main "
    "formulation. The Deep-balanced collapse is consistent with excessive dropout "
    f"for the inner-training sample, 0.5 against {n_pkg} training packages "
    f"({n_row:,} rows): a one-seed diagnostic at dropout 0.1 recovers its "
    f"exact-index heating MAE to {db_e:.3f} GJ/year and its day-count MAE to "
    f"{db_d:.3f} days/year, still clearly behind the shared trunk.", "",
    "**Trainer implementation note.** MGDA replaces the shared-trunk gradient with "
    "a minimum-norm convex combination of task gradients, while each output head "
    "retains its own gradient. Combination weights are computed in closed form "
    "for two tasks and by deterministic Frank–Wolfe iteration for four.", ""])


# ---------------------------------------------------------------- splice
src = open(S1MD).read()

def _header_pos(text, number):
    """Position of '## Table S1.<number>' regardless of the punctuation that
    follows it.  The two generators in this directory have used ':' and ' -- ',
    so match the table number itself and require a non-digit after it."""
    match = re.search(r"^## Table S1\.%s(?!\d)" % re.escape(number), text, re.M)
    if match is None:
        raise SystemExit(f"make_mtl_s1_tables: heading for Table S1.{number} not found")
    return match.start()


def replace_block(text, start_number, end_number, new):
    start = _header_pos(text, start_number)
    end = _header_pos(text, end_number)
    return text[:start] + new + "\n" + text[end:]

src = replace_block(src, "3", "6", t13 + "\n" + t14 + "\n" + t15)
src = replace_block(src, "9", "10", t19)

note = ("\n**Gradient-boosting comparator.** One HistGradientBoostingRegressor per "
        "objective: squared-error loss, 300 boosting iterations, learning rate 0.1, "
        "no early stopping, seeded per fold and objective; fitted on the same "
        "standardised inputs and inner-training rows as the networks.\n\n"
        "**Trainers and task designs.** The joint network is trained under equal "
        "weighting (unweighted mean of the four standardised MSEs) and under "
        "gradient balancing with the MGDA implementation (Table S1.12). "
        "Two task designs are evaluated: all four objectives predicted (predicted "
        "fronts use predicted cost/carbon) and the exact-index design (only the two "
        "simulated outcomes predicted; cost/carbon taken from the component tables). "
        "The exact reference is identical in both.\n")
anchor = "**Unique-selectability LP.**"
if anchor not in src:
    raise SystemExit(f"make_mtl_s1_tables: anchor {anchor!r} not found in SUPPLEMENTARY_S1.md")
if "**Gradient-boosting comparator.**" not in src:
    src = src.replace(anchor, note + "\n" + anchor, 1)

# S1.11/S1.12 are regenerated in place when they exist and inserted ahead of
# S1.14 when they do not, so a from-scratch rebuild keeps the tables in order.
# The earlier append-only guard silently left a stale block whenever one was
# already present, so a corrected run never reached the document.
has11 = re.search(r"^## Table S1\.11(?!\d)", src, re.M) is not None
has14 = re.search(r"^## Table S1\.14(?!\d)", src, re.M) is not None
if has11 and has14:
    src = replace_block(src, "11", "14", t111 + "\n" + t112)
elif has14:
    src = replace_block(src, "14", "14", t111 + "\n" + t112)
else:
    src = src.rstrip() + "\n\n" + t111 + "\n" + t112
open(S1MD, "w").write(src)
print("S1 updated:",
      len(t13.splitlines()), "S1.3 lines;",
      len(t15.splitlines()), "S1.5 lines;",
      len(t111.splitlines()), "S1.11 lines")
