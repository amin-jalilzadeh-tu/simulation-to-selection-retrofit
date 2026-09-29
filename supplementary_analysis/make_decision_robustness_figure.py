"""Cumulative coverage curves of the weight and weight-threshold sweeps (Supplementary Figure S1.6).

Two panels from the released MCDA CSVs:
  (a) cumulative first-rank acceptability over the most-frequent winning packages,
      weights only, per weather case, with the 90%-coverage crossings 15/13/17
      (mcda_outputs.csv);
  (b) the same under joint weight x threshold sampling, crossings 26/13/19
      (mcda_joint_outputs.csv).
The main-text summary is the coverage-bar panel of the SMAA figure
(fig_smaa_summary); the full curves live in
the supplement. Weather cases use the shared WEATHER palette with distinct line
styles, markers and direct labels, so identity does not rest on color alone.
Writes fig_coverage_curves.(pdf|png). Deterministic; crossings are asserted.
"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import figstyle as fs

fs.apply()

_HERE = os.path.dirname(os.path.abspath(__file__))

FRAC = 1.0


def _figdir():
    d = os.environ.get("FIGDIR", os.path.join(os.getcwd(), "figures_out"))
    os.makedirs(d, exist_ok=True)
    return d if d.endswith(os.sep) else d + os.sep

FIGDIR = _figdir()

# Weather cases: shared WEATHER hues (blue -> orange -> red) plus distinct line
# style and marker, so identity never rests on colour alone.
CASES = [("2020", "Present", fs.WEATHER["present"], "-", fs.WEATHER_MARKER["present"]),
         ("2050", "Mid-century", fs.WEATHER["mid"], "--", fs.WEATHER_MARKER["mid"]),
         ("2100", "Late-century", fs.WEATHER["late"], "-.", fs.WEATHER_MARKER["late"])]
EXPECTED_K90 = {"weights": {"2020": 15, "2050": 13, "2100": 17},
                "joint": {"2020": 26, "2050": 13, "2100": 19}}

# Direct labels sit as a stacked block in the empty lower-right of each panel;
# the canvas is too narrow at final width to hang them off each crossing dot.
LABEL_Y = {"2020": 0.60, "2050": 0.47, "2100": 0.34}


def read(name):
    with open(os.path.join(_HERE, name), newline="") as fh:
        return list(csv.DictReader(fh))


fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=fs.size(FRAC, aspect=0.46))

# ---------------------------------------------- (a)/(b) cumulative coverage curves
def coverage_panel(ax, rows, value_key, expected, title):
    # x-limit covers EVERY winning package of every case in this panel, so the
    # cumulative curves are drawn complete rather than silently truncated.
    xmax = 0
    for hz, lab, color, ls, mk in CASES:
        acc = sorted((float(r[value_key]) for r in rows
                      if r["horizon"] == hz and float(r[value_key]) > 0), reverse=True)
        cum = np.cumsum(acc)
        k90 = int(np.searchsorted(cum, 90.0)) + 1
        assert k90 == expected[hz], (title, hz, k90, expected[hz])
        x = np.arange(1, len(cum) + 1)
        xmax = max(xmax, len(cum))
        ax.plot(x, cum, color=color, lw=1.4, ls=ls, drawstyle="steps-post",
                marker=mk, ms=3.2, markevery=8, zorder=3)
        ax.scatter([k90], [cum[k90 - 1]], s=22, color=color, zorder=4)
        ax.text(0.985, LABEL_Y[hz], f"{lab}: {k90}", transform=ax.transAxes,
                ha="right", va="center", color=color, fontweight="bold")
    xmax += 2
    ax.axhline(90, color=fs.MUTED, lw=0.8, ls=":", zorder=2)
    ax.text(xmax - 0.8, 91.4, "90%", ha="right", va="bottom", color=fs.MUTED)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 104)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.set_xlabel("Most-frequent winning packages")
    ax.set_title(title)


coverage_panel(ax_b, read("mcda_outputs.csv"), "acceptability_pct",
               EXPECTED_K90["weights"], "(a) Coverage, weights only")
coverage_panel(ax_c, read("mcda_joint_outputs.csv"), "joint_acceptability_pct",
               EXPECTED_K90["joint"], "(b) Coverage, weights × threshold")
ax_b.set_ylabel("Cumulative first-rank\nshare (%)")
fig.subplots_adjust(left=0.115, right=0.995, bottom=0.185, top=0.90, wspace=0.20)
fs.finalize(fig, FIGDIR + "fig_coverage_curves", frac=FRAC)
print("fig_coverage_curves written (coverage crossings verified)")
