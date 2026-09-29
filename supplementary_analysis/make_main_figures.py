"""Generate manuscript Figures 2-5 from retained S1 analysis outputs.

Figures 2 and 3 are the package-outcome and paired-component figures,
implemented in make_retrofit_figures.py. Figure 4 compares surrogate models;
Figure 5 summarises preference sensitivity. No simulation or fitting is run.

S1_EXHAUSTIVE, MTL_SURROGATE, MTL4_SURROGATE and S1_ANALYSIS select inputs;
FIGDIR selects the output directory. Extracted-S1 and project layouts are
resolved automatically. --figure 2 or 3 regenerates the package/component pair.
"""
import argparse
import collections
import csv
import os
from functools import wraps
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import figstyle as fs

_HERE = Path(__file__).resolve().parent
_S1_NAME = "Supplementary_File_S1_analysis_reproducibility"


def _directory(env_key, candidates, override=None, required=()):
    """Use an explicit directory strictly; otherwise try supported layouts."""
    explicit = override if override is not None else os.environ.get(env_key)
    paths = [Path(explicit).expanduser()] if explicit else candidates
    for path in paths:
        if path.is_dir() and all((path / name).is_file() for name in required):
            return path.resolve()
    locations = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"Required data not found in {locations}; set {env_key}")


def _results_dir(env_key, subdirectory, override=None):
    candidates = [
        _HERE.parent / "results" / subdirectory,
        _HERE.parent / ".build" / "s1data" / _S1_NAME / "results" / subdirectory,
        _HERE / _S1_NAME / "results" / subdirectory,
    ]
    filename = ("all_configurations.csv" if subdirectory == "exhaustive_analysis"
                else "weighted_profile_validation.csv")
    return _directory(env_key, candidates, override, required=(filename,))


def _analysis_dir(override=None):
    candidates = [
        _HERE,
        _HERE.parent / ".build" / "s1data" / _S1_NAME / "supplementary_analysis",
        _HERE / _S1_NAME / "supplementary_analysis",
        _HERE.parent / "supplementary_analysis",
    ]
    return _directory("S1_ANALYSIS", candidates, override,
                      required=("mcda_central_weights.csv", "mcda_joint_outputs.csv"))


def _output_dir(override=None):
    path = Path(override if override is not None else
                os.environ.get("FIGDIR", Path.cwd() / "figures_out"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _figure_style(function):
    """Apply main-figure typography without altering other legacy figures."""
    @wraps(function)
    def styled(*args, **kwargs):
        with matplotlib.rc_context():
            fs.apply(base_pt=8.0)
            return function(*args, **kwargs)
    return styled


HZ = {"present": "2020", "mid": "2050", "late": "2100"}
HZ_LABEL = {"present": "Present", "mid": "Mid-century", "late": "Late-century"}
FLOOR_COL = {"uninsulated": "#1F9A8A", "insulated": "#9C5227"}


def rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Exhaustive reference
# ---------------------------------------------------------------------------
STATE_VAL = {
    "windows": {2.9: 0, 1.2: 1, 1.21: 2, 0.8: 3, 0.81: 4},
    "floor":   {0.41: 0, 4.8: 1, 5.0: 2, 5.5: 3, 5.6: 4},
    "facade":  {0.45: 0, 4.2: 1, 4.4: 2, 6.5: 3, 6.7: 4},
    "roof":    {0.48: 0, 4.5: 1, 4.7: 2, 8.5: 3, 8.7: 4},
}
COL = {"windows": "windows_U_Factor", "floor": "groundfloor_thermal_resistance",
       "facade": "ext_walls_thermal_resistance", "roof": "roof_thermal_resistance"}
STRONG = {"windows": 3, "floor": 4, "facade": 4, "roof": 4}   # lowest U / highest R

def states(r):
    return {c: STATE_VAL[c][float(r[COL[c]])] for c in COL}

PKG = {}   # (weather case, simulation identifier) -> component states and outputs
SEL = {}
MINEH = {}

def label_of(sim):
    s = PKG[("2020", sim)]
    parts = [(n, s[c]) for c, n in (("windows", "Windows"), ("floor", "Floor"),
                                   ("facade", "Façade"), ("roof", "Roof")) if s[c] > 0]
    if not parts:
        return "No retrofit"
    txt = " + ".join(f"{n} {v}" for n, v in parts)
    return txt[0] + txt[1:].replace("Windows", "windows").replace("Floor", "floor") \
                          .replace("Façade", "façade").replace("Roof", "roof")

def _load_reference(exhaustive_dir=None):
    """Load the package reference only for figures that need it."""
    global PKG, SEL, MINEH
    data = _results_dir("S1_EXHAUSTIVE", "exhaustive_analysis", exhaustive_dir)
    configurations = rows(data / "all_configurations.csv")
    PKG = {
        (r["horizon"], int(r["simulation_id"])): dict(
            EH=float(r["annual_heating_energy_gj"]),
            D24=float(r["days_below_24_c"]), **states(r))
        for r in configurations
    }
    assert len(configurations) == len(PKG) == 1875
    SEL = {
        (r["horizon"], r["profile"]): int(r["simulation_id"])
        for r in rows(data / "mcdm_weighted_sum_selections.csv")
    }
    MINEH = {
        r["horizon"]: int(r["simulation_id"])
        for r in rows(data / "reference_solutions.csv")
        if r["selection"] == "minimum_heating_energy"
    }
    assert MINEH == {"2020": 625, "2050": 625, "2100": 599}


def nondominated_mask(objectives):
    """Exact minimisation dominance, preserving every tied objective vector."""
    objectives = np.asarray(objectives, dtype=float)
    assert objectives.ndim == 2 and np.isfinite(objectives).all()
    no_worse = (objectives[None, :, :] <= objectives[:, None, :]).all(axis=2)
    better = (objectives[None, :, :] < objectives[:, None, :]).any(axis=2)
    return ~(no_worse & better).any(axis=1)

# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------
def matched_effects(hz, comp):
    idx = {}
    for (h, sim), p in PKG.items():
        if h == hz:
            idx[tuple(p[c] for c in COL)] = p
    saved, added = [], []
    for key, p0 in idx.items():
        if p0[comp] != 0:
            continue
        p1 = idx[tuple(STRONG[comp] if c == comp else p0[c] for c in COL)]
        saved.append(p0["EH"] - p1["EH"]); added.append(p0["D24"] - p1["D24"])
    assert len(saved) == 125
    return np.array(saved), np.array(added)


BAND = "#F5F5F5"
SLATE, SLATE_DARK, SLATE_LIGHT = "#6D8EA0", "#3F5B6E", "#CBD5DC"


def soften(ax, keep_left=True):
    ax.grid(True, color="#ECECEC", lw=0.5)
    ax.tick_params(length=2.5, width=0.5, colors="#9A9A9A", labelcolor=fs.INK)
    for name, sp in ax.spines.items():
        sp.set_linewidth(0.6); sp.set_color("#A0A0A0")
        if name == "left" and not keep_left:
            sp.set_visible(False)
    ax.title.set_fontsize(9.5)


def fig2(exhaustive_dir=None, output_dir=None):
    """Generate the package and component pair, manuscript Figures 2 and 3."""
    from make_retrofit_figures import make_figures
    make_figures(exhaustive_dir, output_dir)


# ---------------------------------------------------------------------------
# Figure 4
# ---------------------------------------------------------------------------
MAIN = [("shared_mtl_nn", "Multi-task NN,\nequal task-loss weights", "network"),
        ("shared_mtl_nn_mgda", "Multi-task NN,\nMGDA", "network"),
        ("independent_stl_nn", "Separate single-\noutput NNs", "network"),
        ("random_forest", "Multi-output\nrandom forest", "tree"),
        ("gradient_boosting", "Separate gradient-\nboosted regressors", "tree")]


@_figure_style
def fig4(surrogate_dir=None, all_predicted_dir=None, output_dir=None):
    SV = _results_dir("MTL_SURROGATE", "surrogate_validation", surrogate_dir)
    SV4 = _results_dir("MTL4_SURROGATE", "surrogate_validation_family4", all_predicted_dir)
    out = _output_dir(output_dir)
    fm = rows(os.path.join(SV, "fold_metrics.csv"))
    mae = collections.defaultdict(list)
    expected_mae = {
        "annual_heating_energy_gj": [0.081, 0.082, 0.083, 0.191, 0.128],
        "days_below_24_c": [0.572, 0.593, 0.562, 0.383, 0.353],
    }
    for r in fm:
        if r["metric"] == "mae":
            mae[(r["model"], r["target"])].append(float(r["value"]))
    wpv = {"exact_index": rows(os.path.join(SV, "weighted_profile_validation.csv")),
           "all_predicted": rows(os.path.join(SV4, "weighted_profile_validation.csv"))}
    agree, regret = {}, {}
    for d, rr in wpv.items():
        for m, _, _ in MAIN:
            sub = [r for r in rr if r["model"] == m]; assert len(sub) == 45
            agree[(d, m)] = sum(r["selection_agreement"] == "True" for r in sub)
            regret[(d, m)] = np.array([float(r["true_score_regret"]) for r in sub if r["selection_agreement"] != "True"])
    assert [agree[("exact_index", m)] for m, _, _ in MAIN] == [18, 16, 19, 20, 19]
    assert [agree[("all_predicted", m)] for m, _, _ in MAIN] == [8, 3, 9, 8, 17]
    assert all(np.isfinite(v).all() and (v > 0).all() for v in regret.values())
    assert sum(len(v) for v in regret.values()) == 313
    for design in wpv:
        for model, _, _ in MAIN:
            assert len(regret[(design, model)]) == 45 - agree[(design, model)]

    fig = plt.figure(figsize=fs.size(0.95, aspect=0.98))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.14, left=0.19, right=0.99, top=0.95, bottom=0.08)
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    n = len(MAIN)
    for a in ax:
        a.set_yticks(range(n)); a.set_ylim(-0.6, n - 0.4); a.invert_yaxis()
        for i in range(0, n, 2):
            a.axhspan(i - 0.5, i + 0.5, color=BAND, zorder=0, lw=0)
        soften(a, keep_left=False); a.grid(axis="y", visible=False)
    ax[0].set_yticklabels([l for _, l, _ in MAIN]); ax[2].set_yticklabels([l for _, l, _ in MAIN])
    ax[1].tick_params(labelleft=False); ax[3].tick_params(labelleft=False)

    for a, tgt, ttl, unit in ((ax[0], "annual_heating_energy_gj", "(a) Heating MAE; cost/GWP calculated", "MAE (GJ/year)"),
                              (ax[1], "days_below_24_c", "(b) D₂₄ MAE; cost/GWP calculated", "MAE (days)")):
        for i, (m, _, fam) in enumerate(MAIN):
            v = np.array(mae[(m, tgt)]); assert len(v) == 15
            assert np.isfinite(v).all() and round(float(v.mean()), 3) == expected_mae[tgt][i]
            col = fs.MODEL_FAMILY[fam]; mk = fs.MODEL_MARKER[fam]
            a.scatter(v, np.full(15, i), s=11, color=col, alpha=0.28, linewidths=0, zorder=2)
            sd = v.std(ddof=1)
            a.plot([v.mean() - sd, v.mean() + sd], [i, i], color=col, lw=2.4, solid_capstyle="round", zorder=3)
            a.plot([v.mean()], [i], marker=mk, ms=7.5, color=col, markeredgecolor="white", markeredgewidth=0.9, zorder=4)
        a.set_title(ttl, loc="left", fontweight="medium"); a.set_xlabel(unit + "  (lower is better)"); a.set_xlim(left=0)
    ax[0].legend(handles=[Line2D([], [], marker="o", ms=6, color=fs.MODEL_FAMILY["network"], ls="", label="Neural network"),
                          Line2D([], [], marker="D", ms=5.5, color=fs.MODEL_FAMILY["tree"], ls="", label="Tree ensemble"),
                          Line2D([], [], marker="o", ms=3.5, color=fs.MODEL_FAMILY["network"], alpha=0.35, ls="", label="One outer test fold")],
                 loc="upper right", borderaxespad=0.3, labelspacing=0.35, handletextpad=0.4)

    a = ax[2]
    for i, (m, _, _) in enumerate(MAIN):
        p0, p1 = 100 * agree[("all_predicted", m)] / 45, 100 * agree[("exact_index", m)] / 45
        a.plot([p0, p1], [i, i], color="#CFCFCF", lw=2.2, solid_capstyle="round", zorder=2)
        a.plot([p0], [i], marker="o", ms=6.5, markerfacecolor="white", markeredgecolor=fs.TASK_DESIGN["all_predicted"],
               markeredgewidth=1.3, ls="", zorder=3)
        a.plot([p1], [i], marker="s", ms=6.5, color=fs.TASK_DESIGN["exact_index"], ls="", zorder=4)
        a.annotate(f"{agree[('all_predicted', m)]}/45", (p0, i), xytext=(-6, 0), textcoords="offset points",
                   ha="right", va="center", color=fs.MUTED)
        a.annotate(f"{agree[('exact_index', m)]}/45", (p1, i), xytext=(6, 0), textcoords="offset points",
                   ha="left", va="center", color=fs.INK)
    a.set_xlim(-14, 100); a.set_xticks([0, 25, 50, 75, 100])
    a.set_xlabel("Agreement (%), higher is better")
    a.set_title("(c) Selection agreement", loc="left", fontweight="medium")
    fig.legend(handles=[Line2D([], [], marker="o", ms=6.5, markerfacecolor="white", markeredgecolor=fs.TASK_DESIGN["all_predicted"],
                               markeredgewidth=1.3, ls="", label="All objectives predicted"),
                        Line2D([], [], marker="s", ms=6.5, color=fs.TASK_DESIGN["exact_index"], ls="",
                               label="Predict heating and D₂₄; calculate cost/GWP")],
               loc="center", ncol=2, bbox_to_anchor=(0.59, 0.515), columnspacing=2.0, handletextpad=0.4)

    a = ax[3]
    rng = np.random.default_rng(1)
    for i, (m, _, _) in enumerate(MAIN):
        for d, off, mk, face in (("all_predicted", 0.2, "o", "white"), ("exact_index", -0.2, "s", fs.TASK_DESIGN["exact_index"])):
            v = regret[(d, m)]; col = fs.TASK_DESIGN[d]; y = i + off
            a.scatter(v, y + rng.uniform(-0.07, 0.07, len(v)), s=9, color=col, alpha=0.5, linewidths=0, zorder=2)
            q1, q2, q3 = np.percentile(v, [25, 50, 75])
            a.plot([q1, q3], [y, y], color=col, lw=3.0, solid_capstyle="round", zorder=3)
            a.plot([q2], [y], marker=mk, ms=7, markerfacecolor=face, markeredgecolor=col, markeredgewidth=1.3, ls="", zorder=4)
    allv = np.concatenate(list(regret.values()))
    a.set_xscale("log"); a.set_xlim(allv.min() * 0.6, allv.max() * 1.6)
    a.set_xticks([1e-3, 1e-2, 1e-1, 0.5]); a.set_xticklabels(["0.001", "0.01", "0.1", "0.5"])
    a.set_xlabel("Normalised regret (log scale)\nLower is better")
    a.set_title("(d) Regret when selections differ", loc="left", fontweight="medium")
    fs.finalize(fig, os.path.join(out, "fig_model_comparison"), frac=0.95)


# ---------------------------------------------------------------------------
# Figure 5: selection frequency, conditional weights and coverage
# ---------------------------------------------------------------------------
OBJ_WORD = {"E_H": "heating", "I_C": "cost", "I_G": "GWP", "D_24": "D₂₄"}


def short_label(sim):
    s = PKG[("2020", sim)]
    parts = [f"{a}{s[c]}" for c, a in (("windows", "W"), ("floor", "Fl"), ("facade", "Fa"), ("roof", "Ro")) if s[c] > 0]
    return " + ".join(parts) if parts else "No retrofit"


def k90(pairs):
    tot = 0.0
    for k, (sim, p) in enumerate(sorted(pairs, key=lambda t: -t[1]), 1):
        tot += p
        if tot >= 90.0:
            return k
    raise ValueError


@_figure_style
def fig5(exhaustive_dir=None, analysis_dir=None, output_dir=None):
    _load_reference(exhaustive_dir)
    SA = _analysis_dir(analysis_dir)
    out = _output_dir(output_dir)
    cw = rows(os.path.join(SA, "mcda_central_weights.csv"))
    jo = rows(os.path.join(SA, "mcda_joint_outputs.csv"))
    acc, acc10k, wc = collections.defaultdict(dict), collections.defaultdict(dict), {}
    for r in cw:
        hz, sim = r["horizon"], int(r["simulation_id"])
        acc[hz][sim] = float(r["acceptability_pct_1m"]); acc10k[hz][sim] = float(r["acceptability_pct_10k"])
        wc[(hz, sim)] = [float(r[c]) for c in ("wc_energy", "wc_cost", "wc_gwp", "wc_d24")]
    joint = collections.defaultdict(dict)
    for r in jo:
        joint[r["horizon"]][int(r["simulation_id"])] = float(r["joint_acceptability_pct"])
    hzs = [HZ[k] for k in ("present", "mid", "late")]
    kw = [k90(list(acc10k[hz].items())) for hz in hzs]; kj = [k90(list(joint[hz].items())) for hz in hzs]
    assert kw == [15, 13, 17] and kj == [26, 13, 19]
    assert [round(acc[hz][1], 1) for hz in hzs] == [10.3, 28.0, 32.1]
    assert [SEL[(hz, "balanced")] for hz in hzs] == [25, 13, 13]
    expected_top = {
        "2020": [25, 19, 3, 1, 13],
        "2050": [1, 15, 13, 18, 394],
        "2100": [1, 15, 13, 63, 145],
    }

    fig = plt.figure(figsize=fs.size(0.98, aspect=1.10))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1.15], hspace=0.5, wspace=0.10,
                          left=0.20, right=0.98, top=0.91, bottom=0.30)
    gsd = fig.add_gridspec(1, 1, left=0.20, right=0.98, top=0.17, bottom=0.06)
    OBJ = list(fs.OBJECTIVE_ORDER)
    for k, key in enumerate(("present", "mid", "late")):
        hz = HZ[key]
        top = sorted(acc[hz].items(), key=lambda t: -t[1])[:5]
        sims = [s for s, _ in top]; vals = [v for _, v in top]
        assert sims == expected_top[hz]
        assert round(sum(vals)) == {"2020": 69, "2050": 72, "2100": 69}[hz]
        assert all(np.isclose(sum(wc[(hz, sim)]), 1.0, atol=2e-4) for sim in sims)
        ab = fig.add_subplot(gs[k, 0]); aw = fig.add_subplot(gs[k, 1])
        y = np.arange(5)
        cols = [fs.BASELINE_FILL if s == 1 else SLATE for s in sims]
        ab.barh(y, vals, height=0.6, color=cols, zorder=2)
        for i, (s, v) in enumerate(zip(sims, vals)):
            ab.text(v + 0.7, i, f"{v:.0f}%", va="center", color=fs.MUTED)
            if s == SEL[(hz, "balanced")]:
                ab.barh([i], [v], height=0.6, fill=False, edgecolor=fs.OBJECTIVE["I_C"], lw=1.8, zorder=3)
        ab.set_yticks(y); ab.set_yticklabels([f"{short_label(s)} (ID {s})" for s in sims]); ab.invert_yaxis()
        ab.set_xlim(0, 40); ab.set_xticks([0, 10, 20, 30, 40]); soften(ab, keep_left=False); ab.grid(axis="y", visible=False)
        ab.set_title(f"({'abc'[k]}) {HZ_LABEL[key]}: reference Pareto set", loc="left", fontweight="medium")
        if k == 2:
            ab.set_xlabel("Selection frequency under sampled weights (%)")
        else:
            ab.tick_params(labelbottom=False)
        left = np.zeros(5)
        for j, o in enumerate(OBJ):
            w = np.array([100 * wc[(hz, s)][j] for s in sims])
            aw.barh(y, w, left=left, height=0.6, color=fs.OBJECTIVE[o], edgecolor="white", lw=0.8, zorder=2)
            for i in range(5):
                if w[i] >= 3.2 * len(OBJ_WORD[o]) + 2:
                    aw.text(left[i] + w[i] / 2, i, OBJ_WORD[o], ha="center", va="center",
                            color=fs.INK if o == "I_G" else "white")
            left += w
        aw.set_yticks(y); aw.tick_params(labelleft=False); aw.invert_yaxis(); aw.set_xlim(0, 100)
        aw.set_xticks([0, 50, 100]); soften(aw, keep_left=False); aw.grid(False)
        if k == 0:
            aw.set_title("Mean preference weights when selected", loc="left", fontweight="medium")
        if k == 2:
            aw.set_xlabel("Mean preference weight (%)")
        else:
            aw.tick_params(labelbottom=False)
    # (d) shortlist size: paired bars, direct labels
    ad = fig.add_subplot(gsd[0, 0])
    yl = np.arange(3); h = 0.34
    c0, c1 = SLATE_LIGHT, SLATE_DARK
    ad.barh(yl - h / 2, kw, height=h, color=c0, zorder=2)
    ad.barh(yl + h / 2, kj, height=h, color=c1, zorder=2)
    for i in range(3):
        ad.text(kw[i] + 0.4, yl[i] - h / 2, f"{kw[i]}", va="center", color=fs.MUTED)
        ad.text(kj[i] + 0.4, yl[i] + h / 2, f"{kj[i]}", va="center", color=fs.INK)
    ad.set_yticks(yl); ad.set_yticklabels([HZ_LABEL[k] for k in ("present", "mid", "late")]); ad.invert_yaxis()
    ad.set_xlim(0, 44); ad.set_xticks([0, 10, 20, 30, 40]); soften(ad, keep_left=False); ad.grid(axis="y", visible=False)
    ad.legend(handles=[Patch(color=c0, label="Weights at 24 °C"),
                       Patch(color=c1, label="Pooled over four thresholds")],
              loc="lower right", borderaxespad=0.3, labelspacing=0.3, handlelength=1.2)
    ad.set_xlabel("Number of packages accounting for at least 90% of selections")
    ad.set_title("(d) Packages accounting for at least 90% of selections", loc="left", fontweight="medium")
    fig.legend(handles=[Line2D([], [], marker="s", ms=8, color=fs.OBJECTIVE[o], ls="", label=OBJ_WORD[o] + " weight") for o in OBJ]
                       + [Line2D([], [], marker="s", ms=8, markerfacecolor="white", markeredgecolor=fs.OBJECTIVE["I_C"],
                                 markeredgewidth=1.6, ls="", label="equal-weight selection")],
               loc="upper center", ncol=3, bbox_to_anchor=(0.56, 1.0), columnspacing=1.4, handletextpad=0.3)
    fs.finalize(fig, os.path.join(out, "fig_smaa_summary"), frac=0.98)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("2", "3", "4", "5", "all"), default="all",
                        help="select a manuscript figure; 2 or 3 regenerates the pair")
    args = parser.parse_args()
    if args.figure in ("2", "3", "all"):
        fig2()
    if args.figure in ("4", "all"):
        fig4()
    if args.figure in ("5", "all"):
        fig5()


if __name__ == "__main__":
    main()
