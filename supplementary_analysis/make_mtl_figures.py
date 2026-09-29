"""Generate manuscript figures from archived surrogate-validation CSVs.

Outputs (to FIGDIR):
  fig_model_comparison.(pdf|png) - exact-index E_H/D24 MAE and full-grid
                                 agreement / mismatch-regret comparisons
  fig_model_comparison_all_predicted.(pdf|png) - four-objective diagnostic
  fig_mtl_architecture.(pdf|png) - separate all-predicted and exact-index routes
  fig_d24_mechanism.(pdf|png)    - present-weather raw OOF D24 predictions
                                 (equal-weighting joint network, seed 29)
The accuracy panels show mean +/- sample SD and all 15 held-out-fold values.
Decision panels show 45 named-profile comparisons per model and design, using
the full 625-package grid; regret summaries include mismatches only.
Use --comparison-only to write only fig_model_comparison, preserving other plots.
Data overrides: MTL_SURROGATE (exact-index), MTL4_SURROGATE (all-predicted),
S1_EXHAUSTIVE, S1_DECISION_DIAGNOSTICS (directory containing regret_tails.csv),
and FIGDIR. Repository and extracted-S1 paths are resolved automatically.
Deterministic; no simulation or training required.

Typography and palette come from figstyle: every figure is authored at the width
it occupies on the page (frac of \\linewidth), so nominal point sizes are the
printed point sizes, and finalize() refuses to write sub-7 pt ordinary type.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FixedLocator, MaxNLocator, NullLocator, ScalarFormatter

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import figstyle as fs

fs.apply()


def _data_dir(env_key, sub):
    """Resolve both the repository wrapper and extracted S1 layouts."""
    explicit = os.environ.get(env_key)
    candidates = ([Path(explicit)] if explicit else [
        Path(_HERE).parent / "results" / sub,
        Path(_HERE) / "Supplementary_File_S1_analysis_reproducibility"
        / "results" / sub,
    ])
    for path in candidates:
        if path.is_dir():
            return str(path.resolve()) + os.sep
    raise FileNotFoundError(f"{sub} data not found; set {env_key}")


def _figdir():
    d = os.environ.get("FIGDIR", os.path.join(os.getcwd(), "figures_out"))
    os.makedirs(d, exist_ok=True)
    return d if d.endswith(os.sep) else d + os.sep


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--comparison-only", action="store_true",
                    help="write only the four-panel manuscript comparison")
args = parser.parse_args()
SV = _data_dir("MTL_SURROGATE", "surrogate_validation")
SV4 = _data_dir("MTL4_SURROGATE", "surrogate_validation_family4")
EX = _data_dir("S1_EXHAUSTIVE", "exhaustive_analysis")
FIGDIR = _figdir()

# Colour and marker encode model family; position identifies formulation.
MODELS = [
    ("shared_mtl_nn", "Joint network,\nequal weighting", "network"),
    ("shared_mtl_nn_mgda", "Joint network,\ngradient-balanced", "network"),
    ("independent_stl_nn", "Single-task\nnetworks", "network"),
    ("random_forest", "Random forest", "tree"),
    ("gradient_boosting", "Gradient-boosted\ntrees", "tree"),
]
TARGET_COLUMN = {
    "E_H":  ("annual_heating_energy_gj", "GJ/year"),
    "I_C":  ("cost_rate_sum_proxy", "index points"),
    "I_G":  ("carbon_rate_sum_proxy", "index points"),
    "D_24": ("days_below_24_c", "days"),
}
FRAC_COMPARISON = 0.95

# Within-weather ranges describe the exact reference; plotted MAEs remain in
# their own units and pool the three weather cases within each held-out fold.
with open(EX + "all_configurations.csv", newline="") as fh:
    _exh = list(csv.DictReader(fh))
NARROW = {}
for _k, (_col, _unit) in TARGET_COLUMN.items():
    _spans = []
    for _hz in ("2020", "2050", "2100"):
        _v = [float(r[_col]) for r in _exh if r["horizon"] == _hz]
        _spans.append(max(_v) - min(_v))
    NARROW[_k] = min(_spans)
assert round(NARROW["E_H"], 2) == 13.07, NARROW["E_H"]
assert round(NARROW["I_C"], 2) == 1091.00, NARROW["I_C"]
assert round(NARROW["I_G"], 2) == 201.45, NARROW["I_G"]
assert NARROW["D_24"] == 6.0, NARROW["D_24"]


def _paired_mae(data_dir, objective_keys, stem, expected_mae):
    """Read the paired fold values and check the reported accuracy summaries."""
    paired = defaultdict(dict)
    with open(data_dir + "fold_metrics.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["metric"] != "mae":
                continue
            key = (row["model"], row["target"])
            fold = (int(row["repeat_seed"]), int(row["outer_fold"]))
            assert fold not in paired[key], (key, fold)
            paired[key][fold] = float(row["value"])
    fold_keys = sorted(paired[(MODELS[0][0], TARGET_COLUMN["E_H"][0])])
    assert len(fold_keys) == 15, fold_keys
    vals = {}
    for model, _, _ in MODELS:
        for key in objective_keys:
            target = TARGET_COLUMN[key][0]
            assert sorted(paired[(model, target)]) == fold_keys, (model, target)
            vals[(model, target)] = np.array(
                [paired[(model, target)][fold] for fold in fold_keys])
    for (model, key), expected in expected_mae.items():
        observed = float(np.mean(vals[(model, TARGET_COLUMN[key][0])]))
        assert round(observed, 3) == expected, (stem, model, key, observed)

    return vals


def comparison(data_dir, objective_keys, stem, expected_mae):
    """Preserve the four-objective all-predicted accuracy comparison."""
    vals = _paired_mae(data_dir, objective_keys, stem, expected_mae)
    rows = 1 if len(objective_keys) == 2 else 2
    fig, axes = plt.subplots(rows, 2,
                             figsize=fs.size(FRAC_COMPARISON,
                                             aspect=0.49 if rows == 1 else 0.82),
                             sharey=True, layout="constrained", squeeze=False)
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.04, wspace=0.05, hspace=0.10)
    stats = {}
    for panel, (ax, key) in enumerate(zip(axes.ravel(), objective_keys)):
        target, unit = TARGET_COLUMN[key]
        y = np.arange(len(MODELS))[::-1]
        xmax = max(max(vals[(model, target)]) for model, _, _ in MODELS) * 1.06
        for yi, (model, _, family) in zip(y, MODELS):
            folds = vals[(model, target)]
            mean, sd = float(np.mean(folds)), float(np.std(folds, ddof=1))
            stats[(model, key)] = (mean, sd)
            color, marker = fs.MODEL_FAMILY[family], fs.MODEL_MARKER[family]
            ax.scatter(folds, np.full(len(folds), yi), s=8, marker=marker,
                       color=color, alpha=0.28, lw=0, zorder=2)
            ax.hlines(yi, mean - sd, mean + sd, color=color, lw=1.5, zorder=3)
            ax.scatter([mean], [yi], s=34, marker=marker, color=color,
                       edgecolor="white", lw=0.8, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([label for _, label, _ in MODELS], fontsize=7.5,
                           linespacing=1.05)
        ax.set_title(f"({'abcd'[panel]}) {fs.OBJECTIVE_LABEL[key]}", loc="left",
                     fontsize=9, pad=3)
        ax.set_xlabel(f"{fs.OBJECTIVE_LABEL[key]} MAE ({unit})",
                      fontsize=8.7, labelpad=2)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.65, len(MODELS) - 0.35)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax.tick_params(axis="x", labelsize=8, pad=1.5)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.grid(False, axis="y")
        ax.grid(True, axis="x", color=fs.GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D([0], [0], lw=0, marker=fs.MODEL_MARKER[family],
               markerfacecolor=fs.MODEL_FAMILY[family], markeredgecolor="white",
               markeredgewidth=0.8, markersize=5.5, label=label)
        for family, label in (("network", "Neural networks"),
                              ("tree", "Tree ensembles"))
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, fontsize=8,
               frameon=False, handletextpad=0.35, columnspacing=2.0,
               borderaxespad=0.1)
    fs.finalize(fig, FIGDIR + stem, frac=FRAC_COMPARISON)
    print(f"{stem} written; pooled MAE mean +/- sample SD over 15 folds:")
    for (model, key), (mean, sd) in stats.items():
        print(f"  {model}/{key}: {mean:.9f} +/- {sd:.9f}")

    present = defaultdict(list)
    with open(data_dir + "horizon_fold_metrics.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            if (row["metric"] == "mae" and row["horizon"] == "2020"
                    and row["target"] == "days_below_24_c"):
                present[row["model"]].append(float(row["value"]))
    print("  present-weather D24 mean MAE:",
          {model: round(float(np.mean(present[model])), 6)
           for model, _, _ in MODELS})



EXACT_MAE = {
    ("shared_mtl_nn", "E_H"): 0.081,
    ("shared_mtl_nn_mgda", "E_H"): 0.082,
    ("independent_stl_nn", "E_H"): 0.083,
    ("random_forest", "E_H"): 0.191,
    ("gradient_boosting", "E_H"): 0.128,
    ("shared_mtl_nn", "D_24"): 0.572,
    ("shared_mtl_nn_mgda", "D_24"): 0.593,
    ("independent_stl_nn", "D_24"): 0.562,
    ("random_forest", "D_24"): 0.383,
    ("gradient_boosting", "D_24"): 0.353,
}


def _decision_diagnostics_dir():
    explicit = os.environ.get("S1_DECISION_DIAGNOSTICS")
    candidates = ([Path(explicit)] if explicit else [
        Path(SV).parent.parent / "supplementary_analysis"
        / "decision_fidelity_diagnostics",
        Path(_HERE) / "decision_fidelity_diagnostics",
        Path(_HERE) / "Supplementary_File_S1_analysis_reproducibility"
        / "supplementary_analysis" / "decision_fidelity_diagnostics",
    ])
    for path in candidates:
        if (path / "regret_tails.csv").is_file():
            return path
    raise FileNotFoundError("regret_tails.csv not found; set "
                            "S1_DECISION_DIAGNOSTICS")


def _decision_metrics():
    """Load saved full-grid decisions and independently check mismatch summaries."""
    with (_decision_diagnostics_dir() / "regret_tails.csv").open(newline="") as fh:
        tails = {(row["design"], row["model"]): row
                 for row in csv.DictReader(fh)
                 if row["scope"] == "full625"
                 and row["population"] == "mismatches"
                 and row["model"] in {model for model, _, _ in MODELS}}
    expected_matches = {
        "exact-index": [18, 16, 19, 20, 19],
        "all-predicted": [8, 3, 9, 8, 17],
    }
    metrics = {}
    for design, directory in (("exact-index", SV), ("all-predicted", SV4)):
        cases = defaultdict(list)
        with open(directory + "weighted_profile_validation.csv", newline="") as fh:
            for row in csv.DictReader(fh):
                cases[row["model"]].append(row)
        for (model, _, _), expected in zip(MODELS, expected_matches[design]):
            rows = cases[model]
            identifiers = {(row["repeat_seed"], row["horizon"], row["profile"])
                           for row in rows}
            assert len(rows) == len(identifiers) == 45, (design, model)
            matches = sum(row["selection_agreement"].lower() == "true"
                          for row in rows)
            assert matches == expected, (design, model, matches)
            mismatch_regrets = np.array([
                float(row["true_score_regret"]) for row in rows
                if row["selection_agreement"].lower() != "true"
            ])
            summary = tails[(design, model)]
            assert int(summary["n"]) == len(mismatch_regrets) == 45 - matches
            observed = (float(np.median(mismatch_regrets)),
                        float(np.quantile(mismatch_regrets, .95, method="linear")),
                        float(np.max(mismatch_regrets)))
            retained = tuple(float(summary[key]) for key in
                             ("median_regret", "p95_regret", "max_regret"))
            assert np.allclose(observed, retained, rtol=0, atol=1e-12), (
                design, model, observed, retained)
            assert 0 < retained[0] <= retained[1] <= retained[2]
            metrics[(design, model)] = dict(matches=matches,
                agreement=100 * matches / 45, mismatch_count=len(mismatch_regrets),
                median=retained[0], p95=retained[1], maximum=retained[2])
    return metrics


def main_comparison():
    """Generate the main comparison and the other outputs."""
    from make_main_figures import fig4

    fig4(surrogate_dir=SV, all_predicted_dir=SV4, output_dir=FIGDIR)

main_comparison()
if args.comparison_only:
    sys.exit(0)

comparison(SV4, fs.OBJECTIVE_ORDER, "fig_model_comparison_all_predicted", {
    ("shared_mtl_nn", "E_H"): 0.120,
    ("shared_mtl_nn_mgda", "E_H"): 0.085,
    ("independent_stl_nn", "E_H"): 0.083,
    ("random_forest", "E_H"): 0.283,
    ("gradient_boosting", "E_H"): 0.128,
    ("shared_mtl_nn", "D_24"): 0.887,
    ("shared_mtl_nn_mgda", "D_24"): 0.620,
    ("independent_stl_nn", "D_24"): 0.562,
    ("random_forest", "D_24"): 0.663,
    ("gradient_boosting", "D_24"): 0.353,
})

# ------------------------------------------------------------- architecture schematic
# The two task designs are separately fitted models.  Showing them as two lanes
# prevents the exact-index refit from looking like a four-head network whose
# index outputs are merely ignored at inference.
FRAC_ARCH = 1.0

# Column geometry, in data units, sized to the text each column must hold at the
# printed width.  ARCH_XMAX data units span fs.LINEWIDTH_IN.
ARCH_COL_W = [2.45, 1.60, 2.35, 1.70, 1.60]
ARCH_GAP = 0.40
ARCH_MARGIN = 0.15
ARCH_COL_X = []
_x = ARCH_MARGIN
for _w in ARCH_COL_W:
    ARCH_COL_X.append(_x)
    _x += _w + ARCH_GAP
ARCH_XMAX = _x - ARCH_GAP + ARCH_MARGIN
ARCH_YMAX = 5.00

fig, ax = plt.subplots(figsize=fs.size(FRAC_ARCH,
                                       aspect=ARCH_YMAX / ARCH_XMAX))
ax.set_xlim(0, ARCH_XMAX)
ax.set_ylim(0, ARCH_YMAX)
ax.axis("off")

_FITS = []   # (text artist, box extent in data coords) -- checked after draw


def _box(col, y, h, text, fc, fontsize=7.2, ec=fs.MUTED, ls="-", tc=fs.INK,
         w=None, lw=0.9):
    x = ARCH_COL_X[col]
    w = ARCH_COL_W[col] if w is None else w
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=fc, ec=ec, lw=lw, linestyle=ls))
    t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=tc, linespacing=1.25)
    _FITS.append((t, (x, y, w, h)))


def _arrow(x0, y0, x1, y1, color=fs.MUTED, lw=0.9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=7, lw=lw, color=color))


def _right(col):
    return ARCH_COL_X[col] + ARCH_COL_W[col]


INPUT_TEXT = ("six inputs\nweather code, window U\n"
              "three envelope R values\ninfiltration flow")

# One pale fill per box CLASS, so the six classes stay distinguishable.  Every
# fill is fs.tint() of a namespaced hue -- no pale hex is invented here -- and the
# hue carries the box's structural role: fs.SCHEMATIC for data entering the study,
# the predicted track, the exact reference track and the comparison layer, and
# fs.MODEL_FAMILY['network'] for the fitted heads (the same navy that means
# "neural network" in Figure 4).
FILL_INPUT = fs.tint(fs.SCHEMATIC["input"], 0.86)
FILL_TRUNK = fs.tint(fs.SCHEMATIC["surrogate"], 0.80)
FILL_HEADS = fs.tint(fs.MODEL_FAMILY["network"], 0.82)
FILL_EXACT = fs.tint(fs.SCHEMATIC["exact"], 0.72)
FILL_VECTOR = fs.tint(fs.SCHEMATIC["comparison"], 0.80)
FILL_DECIDE = fs.tint(fs.SCHEMATIC["comparison"], 0.66)
C_NET = fs.MODEL_FAMILY["network"]
C_EXACT = fs.TASK_DESIGN["exact_index"]


def _lane(y, title, exact_index=False):
    h = 1.42
    ax.text(ARCH_MARGIN, y + h + 0.12, title, fontsize=8.5,
            fontweight="bold", color=fs.INK, va="bottom")
    _box(0, y, h, INPUT_TEXT, FILL_INPUT, fontsize=7.0)
    _box(1, y, h, "shared trunk\n128 - 128 - 64\n(SiLU)", FILL_TRUNK,
         fontsize=7.2)
    _arrow(_right(0) + 0.08, y + h / 2, ARCH_COL_X[1] - 0.06, y + h / 2)
    if exact_index:
        _box(2, y + 0.80, 0.62, "two 64 - 32 - 1 heads\nheating, D24",
             FILL_HEADS, fontsize=7.2, ec=C_NET)
        _box(2, y, 0.62, "component tables\nexact cost/GWP indices",
             FILL_EXACT, fontsize=7.2, ec=C_EXACT, lw=1.5)
        _arrow(_right(1) + 0.08, y + h / 2, ARCH_COL_X[2] - 0.06, y + 1.11)
        _arrow(_right(2) + 0.08, y + 1.11, ARCH_COL_X[3] - 0.06, y + 0.86)
        _arrow(_right(2) + 0.08, y + 0.31, ARCH_COL_X[3] - 0.06, y + 0.58,
               color=C_EXACT, lw=1.4)
    else:
        _box(2, y, h, "four 64 - 32 - 1 heads\nheating, D24, cost, GWP",
             FILL_HEADS, fontsize=7.2, ec=C_NET)
        _arrow(_right(1) + 0.08, y + h / 2, ARCH_COL_X[2] - 0.06, y + h / 2)
        _arrow(_right(2) + 0.08, y + h / 2, ARCH_COL_X[3] - 0.06, y + h / 2)
    # Plain target names keep every label legible at the diagram's printed size.
    _box(3, y, h, "objective vector\nheating, D24,\ncost, GWP",
         FILL_VECTOR, fontsize=7.2)
    _box(4, y, h, "Pareto fronts\nand selections", FILL_DECIDE, fontsize=7.2)
    _arrow(_right(3) + 0.08, y + h / 2, ARCH_COL_X[4] - 0.06, y + h / 2)


_lane(3.00, "All-predicted design: one four-head fit", exact_index=False)
_lane(0.62, "Exact-index design: a separate two-head refit + exact table indices",
      exact_index=True)
ax.text(ARCH_XMAX / 2, 0.08,
        "Same inputs, architecture family, grouped folds and preprocessing; "
        "separately fitted models",
        ha="center", fontsize=7.0, color=fs.MUTED, style="italic")
fig.tight_layout(pad=0.2)

# Fail loudly if any box text outgrew its box at the printed width, rather than
# shipping a schematic whose labels spill over the frames.
fig.canvas.draw()
_rend = fig.canvas.get_renderer()
_over = []
for _t, (_bx, _by, _bw, _bh) in _FITS:
    tb = _t.get_window_extent(renderer=_rend)
    p0 = ax.transData.transform((_bx, _by))
    p1 = ax.transData.transform((_bx + _bw, _by + _bh))
    if (tb.x0 < p0[0] - 0.5 or tb.x1 > p1[0] + 0.5
            or tb.y0 < p0[1] - 0.5 or tb.y1 > p1[1] + 0.5):
        _over.append((_t.get_text().splitlines()[0],
                      round((tb.x1 - tb.x0) - (p1[0] - p0[0]), 1),
                      round((tb.y1 - tb.y0) - (p1[1] - p0[1]), 1)))
if _over:
    raise ValueError("fig_mtl_architecture: box text overflows its frame "
                     "(label, dx_pt, dy_pt): " + repr(_over))
fs.finalize(fig, FIGDIR + "fig_mtl_architecture", frac=FRAC_ARCH)
print("fig_mtl_architecture written")

# ------------------------------------------------------------- D24 decision mechanism
# The worst-regret composite of the manuscript (Section 3.3): repeat seed 29,
# equally weighted joint network, present weather. Package 16 (simulated 361 d,
# raw prediction 376.76 d) is clipped to 365 and wins the D24-priority selection.
FRAC_D24 = 1.0
WORST_SEED, WORST_PKG = "29", 16
with open(SV4 + "oof_predictions.csv", newline="") as fh:
    oof = [r for r in csv.DictReader(fh)
           if r["model"] == "shared_mtl_nn" and r["repeat_seed"] == WORST_SEED
           and r["horizon"] == "2020"]
sim_id = np.array([int(r["simulation_id"]) for r in oof])
true_d = np.array([float(r["true_days_below_24_c"]) for r in oof])
pred_d = np.array([float(r["predicted_days_below_24_c"]) for r in oof])
rng_j = np.random.default_rng(7)
jit = rng_j.uniform(-0.22, 0.22, len(true_d))
w = sim_id == WORST_PKG
# Guard every number the caption states: the two tie groups are DIFFERENT sets
# (84 packages are simulated at 365; 51 raw predictions clip to 365; 45 overlap),
# and the highlighted package must be the D24-priority winner with regret 0.472.
n365 = int((true_d == 365).sum())
n_clip = int((pred_d >= 365).sum())
assert (n365, n_clip, int(((pred_d >= 365) & (true_d == 365)).sum())) == (84, 51, 45)
assert w.sum() == 1 and abs(pred_d[w][0] - 376.764) < 0.01 and true_d[w][0] == 361
with open(SV4 + "weighted_profile_validation.csv", newline="") as fh:
    sel = [r for r in csv.DictReader(fh)
           if r["model"] == "shared_mtl_nn" and r["repeat_seed"] == WORST_SEED
           and r["horizon"] == "2020"
           and r["profile"] == "days_below_24_priority"]
assert (int(sel[0]["predicted_simulation_id"]) == WORST_PKG
        and abs(float(sel[0]["true_score_regret"]) - 0.4723) < 5e-4), sel

# Navy is the network family (the predictions are joint-network output); the
# highlighted package is flagged in the D24-priority SELECTION role, which is
# what it is here -- the winner of that profile -- not the D24 objective itself.
# The role hue is text-safe (4.9:1 on white), so the annotation carries it too
# and the reader does not have to match a black caption to a coloured marker.
C_D24 = fs.SELECTION["D_24_priority"]
MK_D24 = fs.SELECTION_MARKER["D_24_priority"]
fig, ax = plt.subplots(figsize=fs.size(FRAC_D24, aspect=0.60))
top = max(pred_d.max(), 365) + 0.8
ax.axhspan(365, top, color=fs.MINOR_FILL, zorder=0)
ax.axhline(365, color=fs.MUTED, lw=1.0, zorder=1)
ax.text(358.7, 365.25, "physical maximum\n(clipped for decisions)", fontsize=8,
        color=fs.MUTED, va="bottom", linespacing=1.15)
ax.plot([358.5, 365.5], [358.5, 365.5], ls="--", color=fs.INK, lw=1.0, zorder=1)
ax.text(359.15, 359.95, "$y=x$", fontsize=8, color=fs.INK, rotation=8)
ax.scatter(true_d[~w] + jit[~w], pred_d[~w], s=10,
           color=fs.MODEL_FAMILY["network"], alpha=0.4, lw=0, zorder=2)
# the observation the text discusses: clipping promotes it into the tied group
ax.scatter(true_d[w], pred_d[w], s=62, marker=MK_D24, facecolor="none",
           edgecolor=C_D24, lw=1.6, zorder=5)
ax.annotate("", xy=(true_d[w][0], 365.12), xytext=(true_d[w][0], pred_d[w][0] - 0.30),
            arrowprops=dict(arrowstyle="-|>", lw=1.3, color=C_D24))
ax.text(true_d[w][0] + 0.30, pred_d[w][0] - 0.35,
        f"package {WORST_PKG}: raw {pred_d[w][0]:.1f} d for a simulated "
        f"{true_d[w][0]:.0f} d;\nclipped to 365, its other predicted objectives\n"
        "make it the D24-priority winner (regret 0.472)",
        fontsize=7.5, ha="left", va="top", color=C_D24)
ax.annotate(f"{n_clip} predictions clip\nto 365 for decisions", xy=(364.55, 366.1),
            xytext=(362.20, 369.4), fontsize=7.5, ha="center", color=fs.MUTED,
            arrowprops=dict(arrowstyle="->", lw=0.8, color=fs.MUTED))
ax.set_xticks(range(359, 366))
ax.set_xlabel("Simulated D24 (days/year; horizontal jitter for visibility)",
              fontsize=8.5, labelpad=2)
ax.set_ylabel("Raw out-of-fold predicted D24 (days/year)", fontsize=8.5,
              labelpad=2)
ax.set_xlim(358.5, 365.6)
ax.annotate(f"{n365} packages simulated\nat the 365-day maximum", xy=(364.75, 362.6),
            xytext=(363.45, 356.9), fontsize=7.5, ha="center", color=fs.MUTED,
            arrowprops=dict(arrowstyle="->", lw=0.8, color=fs.MUTED))
ax.tick_params(labelsize=8, pad=1.5)
ax.grid(False, axis="x")
ax.grid(True, axis="y", color=fs.GRID, lw=0.5)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.3)
fs.finalize(fig, FIGDIR + "fig_d24_mechanism", frac=FRAC_D24)
print(f"fig_d24_mechanism written (seed {WORST_SEED}, n={len(true_d)}, sim-365 {n365}, "
      f"clipped {n_clip}, package {WORST_PKG} raw {pred_d[w][0]:.2f})")
