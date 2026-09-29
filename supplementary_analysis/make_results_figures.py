"""Generate the manuscript's warm-side figure from the released S1 CSVs.

Outputs (to FIGDIR):
  fig_warmside.(pdf|png)  - three weather-case panels: below-threshold day count of the
                            equal-weight selection and the minimum-E_H package relative
                            to the same-case baseline, across Tu = 23-26 C
fig_model_comparison and fig_d24_mechanism are produced by make_mtl_figures.py from
the four-head run.
Deterministic; no simulation or training required.

fig_workflow moved to make_framework_figure.py when it was rebuilt as a framework
figure rather than a two-track pipeline; this script no longer writes it.

Typography and colour are governed by figstyle: the figure is authored at the
width it occupies on the page (0.88\\linewidth), so a nominal point size is the
printed point size, and fs.finalize refuses to write a figure carrying type
below 7 pt.
"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle as fs

fs.apply()

# Data resolution: env override first, else paths inside the S1 archive this
# script ships in (../results/...). No machine-specific fallbacks.
_HERE = os.path.dirname(os.path.abspath(__file__))


def _data_dir(env_key, sub):
    for p in (os.environ.get(env_key),
              os.path.join(_HERE, os.pardir, "results", sub)):
        if p and os.path.isdir(p):
            return p if p.endswith(os.sep) else p + os.sep
    raise FileNotFoundError(f"{sub} data not found; set {env_key}")


def _figdir():
    d = os.environ.get("FIGDIR", os.path.join(os.getcwd(), "figures_out"))
    os.makedirs(d, exist_ok=True)
    return d if d.endswith(os.sep) else d + os.sep

EX = _data_dir("S1_EXHAUSTIVE", "exhaustive_analysis")
FIGDIR = _figdir()

# Selection roles, not weather cases: the two series are the equal-weight
# selection and the minimum-E_H package, so they take the SELECTION namespace
# (which inherits the objective hue each role prioritises).
C_EQ = fs.SELECTION["equal_weight"]
C_MIN = fs.SELECTION["E_H_priority"]

HORIZONS = [("2020", "Present"), ("2050", "Mid-century"), ("2100", "Late-century")]
THRESHOLDS = [23, 24, 25, 26]

WARM_FRAC = 0.88     # fig_warmside occupies 0.88\linewidth
# ------------------------------------------------------- warm-side threshold panels
with open(EX + "all_configurations.csv") as fh:
    allc = list(csv.DictReader(fh))
with open(EX + "pareto_fronts.csv") as fh:
    fronts = list(csv.DictReader(fh))


def equal_weight_selection(hz):
    """Recompute the equal-weight Eq. (6)-(7) selection on the exact front."""
    sub = [r for r in fronts if r["horizon"] == hz]
    ids = np.array([int(r["simulation_id"]) for r in sub])
    F = np.array([[float(r["annual_heating_energy_gj"]), float(r["cost_rate_sum_proxy"]),
                   float(r["carbon_rate_sum_proxy"]), -float(r["days_below_24_c"])]
                  for r in sub])
    fmin, fmax = F.min(0), F.max(0)
    Fn = (F - fmin) / np.where(fmax > fmin, fmax - fmin, 1.0)
    scores = Fn @ np.array([0.25, 0.25, 0.25, 0.25])
    best = np.min(scores)
    cand = np.where(np.isclose(scores, best, rtol=0.0, atol=1e-12))[0]
    return int(ids[cand[np.argmin(ids[cand])]])


fig, axes = plt.subplots(1, 3, figsize=fs.size(WARM_FRAC, aspect=0.42), sharey=True)
for ax, (hz, lab) in zip(axes, HORIZONS):
    sub = {int(r["simulation_id"]): r for r in allc if r["horizon"] == hz}
    eq_id = equal_weight_selection(hz)
    min_id = min(sub, key=lambda i: float(sub[i]["annual_heating_energy_gj"]))
    d_eq = [float(sub[eq_id][f"days_below_{t}_c"]) - float(sub[1][f"days_below_{t}_c"])
            for t in THRESHOLDS]
    d_min = [float(sub[min_id][f"days_below_{t}_c"]) - float(sub[1][f"days_below_{t}_c"])
             for t in THRESHOLDS]
    ax.axhline(0, color=fs.MUTED, lw=0.8, zorder=1)
    # same x positions; the open marker keeps both series visible where they
    # nearly coincide (packages are selected once, at 24 C, then evaluated
    # unchanged at every threshold)
    ax.plot(THRESHOLDS, d_eq, marker=fs.SELECTION_MARKER["equal_weight"], ms=4.4,
            lw=1.4, color=C_EQ, markerfacecolor="none", markeredgewidth=1.2,
            label="equal-weight", zorder=3)
    ax.plot(THRESHOLDS, d_min, marker=fs.SELECTION_MARKER["E_H_priority"], ms=4.4,
            lw=1.4, color=C_MIN, label="minimum heating", zorder=3)
    # The two series are named in the legend (panel 1), not inline: at 26 C in
    # the Present case both land on 0 and inline labels collided with each
    # other and with the zero line.
    v24 = d_min[THRESHOLDS.index(24)]
    # One rule in all three panels: just right of and ~4 pt below the 24 C point
    # the label describes, so it sits beside its own marker and needs no leader.
    # U+2212 for the sign, so the annotation and the y-axis ticks agree.
    v24_text = (24.12, v24 - 8.0)
    v24_label = f"{v24:+.0f}".replace("-", "−")
    ax.annotate(f"{v24_label} d at 24 °C", xy=(24, v24), xytext=v24_text,
                fontsize=7.5, color=C_MIN)
    ax.set_title(lab, fontsize=9, pad=3)
    ax.set_xticks(THRESHOLDS)
    ax.set_xlim(22.7, 26.3)
    ax.tick_params(labelsize=8)
    print(f"warmside {lab}: equal-weight {eq_id} {d_eq}; min-EH {min_id} {d_min}")

# The count is named in words, not as $D_{T_u}$: matplotlib sets a subscript
# inside a subscript at 0.49x, so that label printed its innermost glyph at
# 4.57 pt.  The manuscript has no symbol for the count at a general threshold
# ($D_{24}$ is the 24 degC one), and names it "the below-threshold count", so
# the words are the manuscript's own; the x label carries $T_u$.
axes[0].set_ylabel("Change from baseline\nin below-threshold days", fontsize=9)
axes[0].set_ylim(-67, 11)
axes[0].set_yticks([0, -20, -40, -60])
# Legend and gloss live in the empty lower half of the Present panel, where
# neither series goes.
axes[0].legend(loc="upper left", bbox_to_anchor=(0.015, 0.50), fontsize=8,
               handlelength=1.6, handletextpad=0.5, labelspacing=0.35,
               borderaxespad=0.0)
axes[0].text(0.015, 0.02, "negative = more\nabove-threshold days",
             transform=axes[0].transAxes, fontsize=7.2, color=fs.MUTED,
             ha="left", va="bottom", linespacing=1.25)
# One shared x label under the middle panel: three copies no longer fit the
# 0.88-linewidth panels, and the axis is identical in all three.
axes[1].set_xlabel("Upper threshold $T_u$ (°C)", fontsize=9, labelpad=2)
fig.tight_layout(w_pad=0.6)
fs.finalize(fig, FIGDIR + "fig_warmside", frac=WARM_FRAC)
print("fig_warmside written")
