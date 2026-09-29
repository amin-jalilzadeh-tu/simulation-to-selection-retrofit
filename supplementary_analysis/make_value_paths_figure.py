"""Value paths (parallel coordinates) of the exact Pareto fronts — all four objectives at once.

The standard four-objective visualisation in the multiobjective literature
(parallel coordinates / value paths; cf. the Pareto-front visualisation surveys
and the pymoo/PAVED tooling in the project library). One panel per weather case;
one polyline per exact Pareto member over the four objective axes, each axis
min-max scaled over that case's front and ORIENTED SO UP IS BETTER (E_H, I_C,
I_G inverted; D24 as-is), with the real best/worst values printed at the axis
ends. Highlighted paths: the do-nothing baseline, the equal-weight selection and
the E_H-priority selection, so the compromise structure of each choice is
readable across all four objectives simultaneously — the equal-weight selection
runs high on every axis, the E_H-priority selection tops the heating axis while
dropping on cost, carbon and (in the warmer cases) the warm-side axis, and the
baseline tops the two index axes while sitting at the bottom of the heating axis.

Writes fig_value_paths.(pdf|png). Deterministic; selections asserted against the
manuscript's Table 3.
"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

CASES = [("2020", "(a) Present"), ("2050", "(b) Mid-century"), ("2100", "(c) Late-century")]
EXPECTED_SEL = {"2020": {"Energy": 550, "Balanced": 25},
                "2050": {"Energy": 395, "Balanced": 13},
                "2100": {"Energy": 250, "Balanced": 13}}
PROFILES = {"Energy": np.array([0.7, 0.1, 0.1, 0.1]),
            "Balanced": np.array([0.25, 0.25, 0.25, 0.25])}
AXES = [("$E_H$", "GJ/yr", True), ("$I_C$", "pts", True),
        ("$I_G$", "pts", True), ("$D_{24}$", "d", False)]  # invert => up is better
HL = {"base": ("#2b6cb0", "baseline (no retrofit)"),
      "Balanced": ("#0f766e", "equal-weight selection"),
      "Energy": ("#c1440e", "$E_H$-priority selection")}

rows_all = list(csv.DictReader(open(EX + "pareto_fronts.csv")))
fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.3), sharey=True)

for ax, (hz, title) in zip(axs, CASES):
    sub = [r for r in rows_all if r["horizon"] == hz]
    ids = np.array([int(r["simulation_id"]) for r in sub])
    F = np.column_stack([
        [float(r["annual_heating_energy_gj"]) for r in sub],
        [float(r["cost_rate_sum_proxy"]) for r in sub],
        [float(r["carbon_rate_sum_proxy"]) for r in sub],
        [float(r["days_below_24_c"]) for r in sub],
    ])
    lo, hi = F.min(0), F.max(0)
    V = (F - lo) / np.where(hi > lo, hi - lo, 1.0)
    for j, (_, _, invert) in enumerate(AXES):
        if invert:
            V[:, j] = 1.0 - V[:, j]           # up = better on every axis

    # verify the highlighted selections against Table 3 (Eq. 6-7 on this front)
    Fm = np.column_stack([F[:, 0], F[:, 1], F[:, 2], -F[:, 3]])
    fmin, fmax = Fm.min(0), Fm.max(0)
    Fn = (Fm - fmin) / np.where(fmax > fmin, fmax - fmin, 1.0)
    order = np.argsort(ids)
    sel = {}
    for name, w in PROFILES.items():
        sel[name] = int(ids[order][np.argmin((Fn @ w)[order])])
        assert sel[name] == EXPECTED_SEL[hz][name], (hz, name, sel[name])

    x = np.arange(4)
    for v in V:
        ax.plot(x, v, color="#9db9d4", lw=0.5, alpha=0.25, zorder=1)
    idx = {int(s): i for i, s in enumerate(ids)}
    for key, sid in (("base", 1), ("Balanced", sel["Balanced"]),
                     ("Energy", sel["Energy"])):
        c, lab = HL[key]
        ax.plot(x, V[idx[sid]], color=c, lw=2.2, zorder=3,
                marker="o", ms=4, label=f"{lab}" if ax is axs[0] else None)
    def _fmt(v):
        s = f"{v:.1f}"
        return s[:-2] if s.endswith(".0") else s

    for j, (name, unit, invert) in enumerate(AXES):
        ax.axvline(j, color="0.75", lw=0.8, zorder=0)
        top, bot = (lo[j], hi[j]) if invert else (hi[j], lo[j])
        ax.text(j, 1.045, _fmt(top), ha="center", fontsize=6.6, color="0.35")
        ax.text(j, -0.075, _fmt(bot), ha="center", fontsize=6.6, color="0.35")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} ({u})" for n, u, _ in AXES], fontsize=8)
    ax.set_yticks([])
    ax.set_ylim(-0.12, 1.10)
    ax.set_title(title, fontsize=9.5, pad=10)
    for s in ax.spines.values():
        s.set_visible(False)
axs[0].set_ylabel("better →", fontsize=8.5, rotation=90)
axs[0].annotate("", xy=(-0.14, 0.85), xytext=(-0.14, 0.45),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="0.35"))
fig.legend(loc="lower center", ncol=3, fontsize=8, frameon=False,
           bbox_to_anchor=(0.5, -0.035))
fig.tight_layout(rect=(0.01, 0.06, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(FIGDIR + f"fig_value_paths.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("fig_value_paths written (axis ends carry real objective values)")
