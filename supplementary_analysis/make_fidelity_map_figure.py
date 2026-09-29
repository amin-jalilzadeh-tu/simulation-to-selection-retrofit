"""Front recovery vs recommendation recovery (manuscript Figure 5).

One point-pair per formulation: mean Pareto F1 (x) against mean weight-space
selection agreement (y), with an arrow from the all-predicted to the exact-index
task design. Error bars are the descriptive SD across the nine repeat-weather
case composites (which share the weight draws). The figure shows in one view
that respectable front recovery does not deliver recommendation recovery: the
all-predicted formulations sit at F1 0.69-0.83 but 14-47% agreement, exact
indices move every formulation up and to the right, and even at F1 = 0.86-0.89
agreement stays near half.

F1 comes from surrogate_pareto_validation.csv of both designs; agreement from
mcda_weight_agreement.csv. Means are asserted against the manuscript's Table 4.
Writes fig_fidelity_map.(pdf|png). Deterministic.
"""
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import offset_copy

import figstyle as fs

fs.apply()

_HERE = os.path.dirname(os.path.abspath(__file__))

FRAC = 0.92          # manuscript includes this figure at 0.92\linewidth


def _data_dir(env_key, sub):
    for p in (os.environ.get(env_key),
              os.path.join(_HERE, os.pardir, "results", sub)):
        if p and os.path.isdir(p):
            return p if p.endswith(os.sep) else p + os.sep
    raise FileNotFoundError(f"{sub} data not found; set {env_key}")

RESULTS = os.path.dirname(_data_dir("S1_EXHAUSTIVE", "exhaustive_analysis").rstrip(os.sep))
DESIGN_DIR = {"exact-index": "surrogate_validation",
              "all-predicted": "surrogate_validation_family4"}


def _figdir():
    d = os.environ.get("FIGDIR", os.path.join(os.getcwd(), "figures_out"))
    os.makedirs(d, exist_ok=True)
    return d if d.endswith(os.sep) else d + os.sep

FIGDIR = _figdir()

# Task design is a paired before/after of the same quantity, so it is encoded as
# a light -> dark value ramp (open -> filled marker), not as two hues; the
# connecting arrow carries the direction of the move.
AP_C, EI_C = fs.TASK_DESIGN["all_predicted"], fs.TASK_DESIGN["exact_index"]
AP_M, EI_M = fs.TASK_MARKER["all_predicted"], fs.TASK_MARKER["exact_index"]

MAIN5 = [("shared_mtl_nn", "Joint network,\nequal weighting"),
         ("shared_mtl_nn_mgda", "Joint network,\ngradient-balanced"),
         ("independent_stl_nn", "Single-task\nnetworks"),
         ("random_forest", "Random forest"),
         ("gradient_boosting", "Gradient-boosted\ntrees")]
# Table 4 means, asserted
# Anchors match Table 3 of the manuscript. The all-predicted agreement values
# are those obtained after the predicted cost and GWP indices are clipped at
# zero before the decision layer; Pareto F1 is unchanged by that clipping.
EXPECTED = {("shared_mtl_nn", "all-predicted"): (0.718, 9.7),
            ("shared_mtl_nn", "exact-index"): (0.891, 49.6),
            ("shared_mtl_nn_mgda", "all-predicted"): (0.689, 10.0),
            ("shared_mtl_nn_mgda", "exact-index"): (0.887, 46.4),
            ("independent_stl_nn", "all-predicted"): (0.723, 17.8),
            ("independent_stl_nn", "exact-index"): (0.891, 50.7),
            ("random_forest", "all-predicted"): (0.793, 13.9),
            ("random_forest", "exact-index"): (0.864, 46.5),
            ("gradient_boosting", "all-predicted"): (0.832, 47.2),
            ("gradient_boosting", "exact-index"): (0.872, 52.9)}

f1 = defaultdict(list)
for design, d in DESIGN_DIR.items():
    with open(os.path.join(RESULTS, d, "surrogate_pareto_validation.csv")) as fh:
        for r in csv.DictReader(fh):
            f1[(r["model"], design)].append(float(r["f1"]))
agr = defaultdict(list)
with open(os.path.join(_HERE, "mcda_weight_agreement.csv")) as fh:
    for r in csv.DictReader(fh):
        agr[(r["model"], r["task_design"])].append(float(r["agreement_pct"]))

# ---------------------------------------------------------------------------
# Layout constants.  Each formulation carries ONE label -- the long name used in
# Table 4 and Figure 4 -- set beside its all-predicted marker, which is the end
# of the pair that has room; the arrow then leads the eye to the exact-index
# square.  A second, short-form label column on the right was tried and removed:
# it named the same five models twice, and no leader could reach the exact-index
# cluster without crossing another leader or the error bars.
XLO, XHI = 0.645, 0.918
YLO, YHI = 0.0, 70.0               # 70 clears the tallest SD whisker (69.1)
LBL_PT = 7.5                       # printed size; floor is fs.MIN_PT = 7.0

# label anchor (data coords) and alignment, one per formulation
AP_LBL = {"shared_mtl_nn":      (0.7460, 7.6, "center"),
          "shared_mtl_nn_mgda": (0.6790, 5.0, "center"),
          "independent_stl_nn": (0.7225, 37.0, "center"),
          "random_forest":      (0.7933, 2.6, "center"),
          "gradient_boosting":  (0.7860, 47.2, "center")}
# Curvature and head clearance per connector.  Distinct curvature keeps the five
# individually traceable where the data forces them to converge.
#
# The joint-equal and single-task exact-index means are (0.891, 49.6) and
# (0.891, 50.7): identical to three decimals in F1 and 1.1 pp apart in
# agreement, i.e. 4.1 pt apart on this axis against a 7.07 pt marker, so the
# lower square was ~42% hidden behind the upper one and neither arrowhead was
# separable.  Fanning the connectors does not fix that -- the collision is
# between the MARKERS, not the connectors.
#
# Fix: displace only the DRAWN square, in display points, and keep everything
# that carries information at the true coordinate -- the error bars, the stem
# root, and a printed value label.  The near-tie is thereby stated rather than
# hidden, which is the point: these two formulations really are indistinguishable
# here, and the figure should say so rather than imply a separation.
# (dx, dy) in POINTS applied to the drawn exact-index square only.  Data-space
# positions are untouched; anything not listed is drawn at its true location.
# Vertical only: displacing x as well would distort apparent F1, and F1 is the
# coordinate these two share exactly (0.891 both).  Agreement is the axis that
# differs, it is the axis displaced, and it is the value printed -- so every
# number a reader might take off the figure is either true or stated.
EI_NUDGE = {"shared_mtl_nn":      (0.0, -11.0),
            "independent_stl_nn": (0.0,  11.0)}

ARROW = {"shared_mtl_nn":      (0.08, 4.0),
         "shared_mtl_nn_mgda": (0.09, 9.0),
         "independent_stl_nn": (-0.14, 3.0),
         "random_forest":      (-0.16, 9.0),
         "gradient_boosting":  (0.0, 9.0)}

fig, ax = plt.subplots(figsize=fs.size(FRAC, aspect=0.70))

for m, lab in MAIN5:
    pts = {}
    for design, key in (("all-predicted", "all_predicted"),
                        ("exact-index", "exact_index")):
        fx = np.array(f1[(m, design)]); ay = np.array(agr[(m, design)])
        assert len(fx) == 9 and len(ay) == 9, (m, design)
        exp_f, exp_a = EXPECTED[(m, design)]
        assert abs(fx.mean() - exp_f) < 5e-4 and abs(ay.mean() - exp_a) < 0.06, (
            m, design, fx.mean(), ay.mean())
        pts[design] = (fx.mean(), ay.mean())
        ap = key == "all_predicted"
        color = AP_C if ap else EI_C
        # Error bars always sit at the true mean, nudge or no nudge.
        ax.errorbar(fx.mean(), ay.mean(), xerr=fx.std(ddof=1), yerr=ay.std(ddof=1),
                    fmt="none", ecolor=color, elinewidth=0.9,
                    alpha=0.70 if ap else 0.35, zorder=2)

        dx, dy = (0.0, 0.0) if ap else EI_NUDGE.get(m, (0.0, 0.0))
        if dx or dy:
            tr = offset_copy(ax.transData, fig=fig, x=dx, y=dy, units="points")
            # hairline stem: true coordinate -> drawn square, so the reader can
            # always recover where the point actually is
            ax.annotate("", xy=(fx.mean(), ay.mean()), xycoords="data",
                        xytext=(dx, dy), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", lw=0.6, color=EI_C,
                                        shrinkA=0, shrinkB=3.2, alpha=0.9),
                        zorder=3)
            ax.annotate(f"{ay.mean():.1f}", xy=(fx.mean(), ay.mean()),
                        xycoords="data", xytext=(dx + 7.5, dy),
                        textcoords="offset points", fontsize=LBL_PT,
                        ha="left", va="center", color=fs.MUTED, zorder=6)
        else:
            tr = ax.transData
        ax.scatter([fx.mean()], [ay.mean()], s=50, marker=AP_M if ap else EI_M,
                   facecolor="white" if ap else EI_C,
                   edgecolor=AP_C if ap else "white",
                   lw=1.5 if ap else 0.8, zorder=4, transform=tr)
    rad, shrink_b = ARROW[m]
    ndx, ndy = EI_NUDGE.get(m, (0.0, 0.0))
    head = pts["exact-index"]
    if ndx or ndy:
        # convert the point offset into data coords so the arrow head follows
        # the square the reader can actually see
        x0, y0 = ax.transData.transform(head)
        head = ax.transData.inverted().transform((x0 + ndx * fig.dpi / 72.0,
                                                  y0 + ndy * fig.dpi / 72.0))
    ax.add_patch(FancyArrowPatch(
        pts["all-predicted"], tuple(head), arrowstyle="-|>",
        connectionstyle=f"arc3,rad={rad}", mutation_scale=8, lw=0.85,
        color=fs.MUTED, alpha=0.85, shrinkA=5, shrinkB=shrink_b, zorder=3))

    lx, ly, ha = AP_LBL[m]
    ax.text(lx, ly, lab, fontsize=LBL_PT, ha=ha, va="center", color=fs.MUTED,
            linespacing=1.10, zorder=6)

ax.axhline(50, color=fs.FAINT, lw=0.8, ls=":", zorder=1)
ax.text(XLO + 0.004, 51.2, "half of draws", fontsize=LBL_PT, color=fs.MUTED,
        ha="left", va="bottom")

ax.scatter([], [], s=50, marker=AP_M, facecolor="white", edgecolor=AP_C, lw=1.5,
           label="all-predicted")
ax.scatter([], [], s=50, marker=EI_M, facecolor=EI_C, edgecolor="white", lw=0.8,
           label="exact-index")
ax.legend(loc="upper left", handletextpad=0.5, borderaxespad=0.2,
          labelspacing=0.45)

ax.set_xlabel("Pareto-front recovery, mean F1 over nine composites")
ax.set_ylabel("Weight-space selection agreement (%)")
ax.set_xlim(XLO, XHI)
ax.set_ylim(YLO, YHI)
ax.set_xticks([0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])
fig.tight_layout(pad=0.4)
fs.finalize(fig, FIGDIR + "fig_fidelity_map", frac=FRAC)
print("fig_fidelity_map written (means asserted against Table 4)")
