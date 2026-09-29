"""Generate the base SUPPLEMENTARY_S1.md sections from the released S1 CSVs.

Tables:
  S1.1  Simulation assumptions
  S1.2  Retrofit state specifications (incl. bundled infiltration factors)
  S1.3  Out-of-fold prediction accuracy, pooled (MAE/RMSE/R2, mean +/- SD over 15 folds)
  S1.4  Horizon-specific prediction accuracy (MAE and R2, mean +/- SD over 15 folds)
  S1.5  Decision-fidelity summary (Pareto precision/recall/F1; per-seed exact
        selection agreement; regret distribution)
  S1.6  Weather-file descriptors
  S1.7  MCDA robustness summary (unique selectability, weight sweep, joint sweep)
        plus the normalisation-anchor and warm-side-safeguard results
  S1.14 Weight-space selection agreement (from mcda_weight_agreement.csv)
  S1.15 Equal-weight exchange equivalents of one D24 day (from pareto_fronts.csv)
  S1.16 First-rank acceptability and central weight vectors (from
        mcda_central_weights.csv)
Deterministic; derives every value from the released CSVs.

Pipeline note: this script writes the BASE document. make_mtl_s1_tables.py then
replaces Tables S1.3-S1.5 and S1.9 with the five-formulation two-task-design
versions and appends S1.11-S1.12 at the end of the document.
Table S1.10 and the Figure S1.1/S1.2 captions are maintained directly in the
shipped SUPPLEMENTARY_S1.md. Set S1_OUT to write elsewhere (used by the tests).
"""
import csv
import os
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _data_dir(env_key, sub):
    for p in (os.environ.get(env_key),
              os.path.join(_HERE, os.pardir, "results", sub)):
        if p and os.path.isdir(p):
            return p if p.endswith(os.sep) else p + os.sep
    raise FileNotFoundError(f"{sub} data not found; set {env_key}")


SV = _data_dir("S1_SURROGATE", "surrogate_validation")
EX = _data_dir("S1_EXHAUSTIVE", "exhaustive_analysis")
ROOTDIR = os.path.dirname(EX.rstrip(os.sep))            # .../results
ARCH = os.path.dirname(ROOTDIR)                          # archive root
OUT = os.environ.get("S1_OUT", os.path.join(_HERE, "SUPPLEMENTARY_S1.md"))

MODEL = {"shared_mtl_nn": "Multi-task NN", "independent_stl_nn": "Single-task NNs",
         "random_forest": "Random forest"}
TARGET = {"annual_heating_energy_gj": "E_H (GJ/year)", "days_below_24_c": "D24 (days/year)"}
HZ = {"2020": "Present", "2050": "Mid-century", "2100": "Late-century"}
ORDER = ["shared_mtl_nn", "independent_stl_nn", "random_forest"]


def read(path):
    return list(csv.DictReader(open(path)))


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


parts = ["# Supplementary S1 — methods notes and tables\n",
         "All values in this document are derived from the CSV files in this archive; "
         "the generating script is `supplementary_analysis/make_supplementary_tables.py`. "
         "E_H is the annual sum of the archived daily heating series; no boiler "
         "efficiency or fuel conversion is applied. D24 counts days with the "
         "selected zone's daily-mean air temperature below 24 \u00b0C.\n"]

# ---- S1.0 methods notes
parts.append("## S1.0 Methods notes\n")
parts.append(
    "**Surrogate feature encoding.** The six model inputs (weather-case label; window "
    "U-value; ground-floor, façade, and roof thermal resistances; infiltration design flow) "
    "enter the feature vector with the values stored in the reduced dataset. For the opaque "
    "states these are the dataset values listed in Table S1.2; for example, ground-floor "
    "state 1 enters as 4.8 m²K/W and roof state 0 as 0.48 m²K/W, so the released "
    "out-of-fold predictions are exactly reproducible from the released files. The window "
    "U-values are identical in both sources.\n\n"
    "**Random-forest comparator.** 300 trees, unlimited depth, minimum leaf size 1, all "
    "features considered at each split, bootstrap resampling; fitted on the same "
    "standardised inputs and targets and the same inner-training rows as the networks, with "
    "the early-stopping partition unused.\n\n"
    "**Unique-selectability LP.** A Pareto member is *uniquely selectable* when some "
    "admissible weight vector makes it the unique minimiser of the Eq. (7) weighted sum over "
    "the front's Eq. (6)-normalised objectives, under the nonnegative normalised weighted-sum "
    "model with weight floor w_j ≥ 10⁻⁶. For each member the LP maximises the worst-case "
    "score margin t\\* over admissible weights; a member is classified uniquely selectable "
    "when t\\* > 10⁻⁹ and not uniquely selectable otherwise. No finer boundary/unsupported "
    "split is reported, because several margins sit at solver-tolerance scale (e.g. "
    "−4.9×10⁻⁸) and certifying such a split would need exact rather than floating-point "
    "arithmetic. Per-member t\\* values (scientific notation) are in `mcda_outputs.csv`.\n\n"
    "**Tie conventions.** Two deliberate, distinct rules are used. The five named-profile "
    "selections (weighted sum and achievement scalarisation) treat scores within an absolute "
    "tolerance of 10⁻¹² as tied (`np.isclose` with rtol = 0.0) and break ties towards the "
    "lowest simulation identifier; the relative tolerance must be zero, since numpy's default "
    "treats genuinely unequal scores as tied. The 10,000-draw sweeps use exact-equality "
    "argmin over identifier-sorted scores (deterministic, lowest identifier first); a "
    "tolerance rule there could manufacture ties between genuinely unequal scores and is "
    "intentionally not used.\n")

# The ASF-comparison claims are derived from the released structured outputs, not
# transcribed: mcda_profile_sensitivity.csv carries the per-case per-profile
# weighted-sum and ASF selections with agreement and rho-grid stability flags, and
# mcda_outputs.csv carries the unique-selectability classification.
_mcda = read(os.path.join(_HERE, "mcda_outputs.csv"))
_prof = read(os.path.join(_HERE, "mcda_profile_sensitivity.csv"))
_agreeing = [r for r in _prof if r["agreement"] == "True"]
_n_changed = len(_prof) - len(_agreeing)
assert all(r["asf_stable_over_rho_grid"] == "True" for r in _prof)
assert len(_agreeing) == 1, _agreeing
_ag = _agreeing[0]
_asf_eq = {r["horizon"]: r["asf_selection"] for r in _prof if r["profile"] == "Balanced"}
assert set(_asf_eq.values()) == {"138"}, _asf_eq
_p138 = {r["horizon"]: r["selectability"] for r in _mcda if r["simulation_id"] == "138"}
assert set(_p138.values()) == {"not_uniquely_selectable"}, _p138
parts.append(
    "**Achievement-scalarising comparison.** The alternative rule minimises "
    "max_i w_i f̄_i(x) + ρ Σ_i w_i f̄_i(x) with ρ = 10⁻⁶ over each front, with f̄_i the "
    "Eq. (6)-normalised objectives (ideal reference point at the origin) and the same "
    f"named-profile tie rule. It selects a different package from the weighted sum in "
    f"{_n_changed} of the {len(_prof)} weather case–profile cases, the "
    f"{HZ[_ag['horizon']].lower()} {_ag['profile']}-priority case alone agreeing; its "
    "equal-weight choice, package 138, is not uniquely selectable under the nonnegative "
    "normalised weighted-sum model with w_j ≥ 10⁻⁶ and the stated solver tolerance, in any "
    f"weather case. The {len(_prof)} achievement-scalarising selections are unchanged for "
    "ρ ∈ {10⁻⁸, 10⁻⁶, 10⁻⁴, 10⁻²} (per-case selections in "
    "`mcda_profile_sensitivity.csv`).\n")
parts.append(
    "**Weight-space selection agreement.** For each task design, formulation and "
    "repeat–weather case composite, the predicted front is the Pareto set of the out-of-fold "
    "predicted objective vectors over all 625 packages (predicted D24 clipped to [0, 365]; "
    "in the exact-index design the two tabulated indices enter exactly). For each of the "
    "10,000 Section 3.4 weight draws, both the predicted and the exact front are normalised "
    "over their own ideal and nadir (Eq. 6), scored with the weighted sum (Eq. 7) under the "
    "sweep tie rule, and the agreement is the share of draws on which both fronts select the "
    "same simulation identifier (Table S1.14).\n\n"
    "**Normalisation-anchor check.** The 15 profile selections are recomputed with the "
    "Eq. (6) scaling constants taken over all 625 packages of the weather case instead of "
    "the front's ideal and nadir; the candidate set and tie rule are unchanged. Results "
    "follow Table S1.7.\n\n"
    "**Warm-side baseline safeguard.** Front members whose D24 falls below the same-case "
    "baseline count are removed before scoring; the surviving members are scored with the "
    "front's original Eq. (6) normalisation (a feasibility screen, not a new scaling) and "
    "the named-profile tie rule. Re-normalising over the screened subset instead also moves "
    "further near-tied selections, consistent with the scale dependence documented above; "
    "the reported safeguard keeps the original scaling. Results follow Table S1.7.\n\n"
    "**First-rank acceptability and central weight vectors.** One million "
    "Dirichlet(1,1,1,1) draws (seed 42, chunked) are scored per weather case under "
    "Eq. (6)–(7) with the sweep tie rule; every ever-winning package's first-rank "
    "acceptability a_i and central weight vector w^c (the mean of its winning draws) are in "
    "`mcda_central_weights.csv`, generated by "
    "`supplementary_analysis/mcda_central_weights.py`. 9,604 draws already bound the Monte "
    "Carlo error on first-rank shares to ±0.01 at 95% confidence (Tervonen and Figueira, "
    "2008); the leading packages' "
    "one-million-draw shares differ from the 10,000-draw estimates by at most 0.4 percentage "
    "points, with the same leading package in every weather case, and the larger sample "
    "stabilises the central vectors of low-acceptability packages. Sampled minimum and "
    "maximum winning weights are not reported: sample extrema are not theoretical weight "
    "bounds.\n")

# ---- S1.1 assumptions
rows = read(os.path.join(ARCH, "simulation_assumptions.csv"))
parts.append("## Table S1.1 — Simulation assumptions\n")
parts.append(md_table(["Setting", "Value"],
                      [(r["setting"], r["value"]) for r in rows]))

# ---- S1.2 state specifications
rows = read(EX + "retrofit_state_specifications.csv")
parts.append("\n## Table S1.2 — Retrofit state specifications and bundled infiltration factors\n")
parts.append("`generator_value` is the thickness/conductivity-derived property in the executed "
             "generator; `archive_metadata_value` identifies the same discrete state in the "
             "reduced archive. The infiltration factor multiplies the 0.025 m³/s baseline "
             "Flow/Zone rate through the component-weighted generator equation "
             "(clipped to 0.013–0.025 m³/s).\n")
hdr = ["Component", "State", "Source option label", "Property", "Generator", "Archive",
       "Cost rate (€/m²)", "GWP A1–A3 (kgCO₂e/m²)", "Infiltration factor"]
parts.append(md_table(hdr, [(r["component"], r["state_id"], r["source_option_label"],
                             r["reported_property"], r["generator_value"],
                             r["archive_metadata_value"], r["cost_rate_eur_per_m2"],
                             r["gwp_a1_a3_kgco2e_per_m2"], r["infiltration_factor"])
                            for r in rows]))

# ---- S1.3 pooled prediction accuracy
fm = defaultdict(list)
for r in read(SV + "fold_metrics.csv"):
    fm[(r["model"], r["target"], r["metric"])].append(float(r["value"]))
parts.append("\n## Table S1.3 — Out-of-fold prediction accuracy, pooled over horizons\n")
parts.append("Mean ± SD over the 15 package-grouped outer folds (3 seeds × 5 folds); "
             "folds share packages across seeds and are not independent replicates.\n")
rows3 = []
for m in ORDER:
    for t in TARGET:
        cells = [MODEL[m] if t == "annual_heating_energy_gj" else "", TARGET[t]]
        for met in ("mae", "rmse", "r2"):
            v = fm[(m, t, met)]
            fmtstr = "{:.3f} ± {:.3f}" if met != "r2" else "{:.4f} ± {:.4f}"
            cells.append(fmtstr.format(np.mean(v), np.std(v, ddof=1)))
        rows3.append(cells)
parts.append(md_table(["Model", "Target", "MAE", "RMSE", "R²"], rows3))

# ---- S1.4 horizon-specific
hm = {}
for r in read(SV + "horizon_metrics_summary.csv"):
    hm[(r["model"], r["horizon"], r["target"], r["metric"])] = (
        float(r["mean"]), float(r["standard_deviation"]))
parts.append("\n## Table S1.4 — Horizon-specific prediction accuracy (mean ± SD over 15 folds)\n")
rows4 = []
for m in ORDER:
    for hz in ("2020", "2050", "2100"):
        cells = [MODEL[m] if hz == "2020" else "", HZ[hz]]
        for t in TARGET:
            for met in ("mae", "r2"):
                mu, sd = hm[(m, hz, t, met)]
                fmtstr = "{:.3f} ± {:.3f}" if met == "mae" else "{:.4f} ± {:.4f}"
                cells.append(fmtstr.format(mu, sd))
        rows4.append(cells)
parts.append(md_table(["Model", "Horizon", "E_H MAE", "E_H R²", "D24 MAE", "D24 R²"], rows4))

# ---- S1.5 decision fidelity
pv = defaultdict(lambda: defaultdict(list))
for r in read(SV + "surrogate_pareto_validation.csv"):
    for k in ("precision", "recall", "f1"):
        pv[r["model"]][k].append(float(r[k]))
wp = read(SV + "weighted_profile_validation.csv")
agree = defaultdict(lambda: defaultdict(int))
regret = defaultdict(list)
on_front = defaultdict(int)
for r in wp:
    m = r["model"]
    if r["selection_agreement"] == "True":
        agree[m][r["repeat_seed"]] += 1
    if r["predicted_selection_on_exact_pareto"] == "True":
        on_front[m] += 1
    regret[m].append(float(r["true_score_regret"]))
parts.append("\n## Table S1.5 — Decision-fidelity summary\n")
parts.append("Pareto precision/recall/F1: mean ± SD over the 9 seed–horizon composite fronts. "
             "Exact agreement: profile selections (of 15 unique horizon–profile cases) "
             "reproduced exactly, per repeat seed. Regret: Eq. 10 true-score regret over all "
             "45 selection cases.\n")
rows5 = []
for m in ORDER:
    reg = np.array(regret[m])
    rows5.append([
        MODEL[m],
        "{:.3f} ± {:.3f}".format(np.mean(pv[m]["precision"]), np.std(pv[m]["precision"], ddof=1)),
        "{:.3f} ± {:.3f}".format(np.mean(pv[m]["recall"]), np.std(pv[m]["recall"], ddof=1)),
        "{:.3f} ± {:.3f}".format(np.mean(pv[m]["f1"]), np.std(pv[m]["f1"], ddof=1)),
        "/".join(str(agree[m][s]) for s in ("17", "29", "43")) + " of 15",
        f"{on_front[m]} of 45",
        "{:.4f}".format(np.median(reg)),
        "{:.3f}".format(np.quantile(reg, 0.9)),
        "{:.3f}".format(reg.max()),
    ])
parts.append(md_table(["Model", "Pareto precision", "Pareto recall", "F1",
                       "Exact agreement (seeds 17/29/43)", "On exact front",
                       "Regret median", "Regret p90", "Regret max"], rows5))

# ---- S1.6 weather descriptors
rows = read(os.path.join(ARCH, "weather_file_descriptors.csv"))
parts.append("\n## Table S1.6 — Weather-file descriptors\n")
parts.append(md_table(
    ["Weather case", "EPW header", "SHA-256 (first 16)", "Annual mean T (°C)", "JJA mean T (°C)",
     "HDD18 (°C·day)", "CDD18 (°C·day)", "JJA daily-mean p95 (°C)"],
    [(r["weather_case"], r["epw_header"], r["sha256"][:16] + "…", r["annual_mean_dry_bulb_c"],
      r["jja_mean_dry_bulb_c"], r["hdd18_c_day"], r["cdd18_c_day"],
      r["jja_daily_mean_p95_c"]) for r in rows]))
parts.append("\nFull 64-character SHA-256 hashes are in `weather_file_descriptors.csv`. The hashes "
             "document the surviving EPW files themselves.\n")

# ---- S1.8 objective correlations per horizon
allc = read(EX + "all_configurations.csv")
parts.append("\n## Table S1.8 — Pearson correlations between objectives, all 625 packages per weather case\n")
rows8 = []
for hz in ("2020", "2050", "2100"):
    sub = [r for r in allc if r["horizon"] == hz]
    eh = np.array([float(r["annual_heating_energy_gj"]) for r in sub])
    icx = np.array([float(r["cost_rate_sum_proxy"]) for r in sub])
    igx = np.array([float(r["carbon_rate_sum_proxy"]) for r in sub])
    d24 = np.array([float(r["days_below_24_c"]) for r in sub])
    c = lambda a, b: "{:+.3f}".format(np.corrcoef(a, b)[0, 1])
    rows8.append([HZ[hz], c(eh, icx), c(eh, igx), c(icx, igx), c(eh, d24)])
parts.append(md_table(["Horizon", "E_H–I_C", "E_H–I_G", "I_C–I_G", "E_H–D24"], rows8))

# ---- S1.9 per repeat-horizon Pareto precision/recall (MTL)
pvr = read(SV + "surrogate_pareto_validation.csv")
parts.append("\n## Table S1.9 — Per repeat–horizon Pareto precision and recall\n")
rows9 = []
for m in ORDER:
    for hz in ("2020", "2050", "2100"):
        sub = [r for r in pvr if r["model"] == m and r["horizon"] == hz]
        cells = [MODEL[m] if hz == "2020" else "", HZ[hz]]
        cells.append("/".join("{:.3f}".format(float(r["precision"])) for r in sub))
        cells.append("/".join("{:.3f}".format(float(r["recall"])) for r in sub))
        rows9.append(cells)
parts.append(md_table(["Model", "Horizon", "Precision (seeds 17/29/43)", "Recall (seeds 17/29/43)"], rows9))

# ---- S1.7 MCDA robustness summary
mc = read(os.path.join(_HERE, "mcda_outputs.csv"))
jc = read(os.path.join(_HERE, "mcda_joint_outputs.csv"))
parts.append("\n## Table S1.7 — MCDA robustness summary per horizon\n")
parts.append("Unique selectability at weight floor 10⁻⁶ and threshold t\\* > 10⁻⁹ (see S1.0); "
             "sweep = 10,000 uniform Dirichlet(1,1,1,1) draws, seed 42; joint sweep additionally "
             "samples T_u uniformly over {23, 24, 25, 26} °C. Full per-member results are in "
             "`supplementary_analysis/mcda_outputs.csv` and `mcda_joint_outputs.csv`.\n")
rows7 = []
for hz in ("2020", "2050", "2100"):
    sub = [r for r in mc if r["horizon"] == hz]
    notun = sum(1 for r in sub if r["selectability"] == "not_uniquely_selectable")
    acc = sorted((float(r["acceptability_pct"]) for r in sub
                  if float(r["acceptability_pct"]) > 0), reverse=True)
    k90 = int(np.searchsorted(np.cumsum(acc), 90.0)) + 1
    jsub = sorted((float(r["joint_acceptability_pct"]) for r in jc if r["horizon"] == hz),
                  reverse=True)
    jk90 = int(np.searchsorted(np.cumsum(jsub), 90.0)) + 1
    rows7.append([HZ[hz], f"{len(sub)}", f"{notun} of {len(sub)}", f"{len(acc)}", f"{k90}",
                  f"{len(jsub)}", f"{jk90}"])
parts.append(md_table(["Horizon", "Pareto members", "Not uniquely selectable",
                       "Sweep: distinct winners", "Sweep: 90% coverage",
                       "Joint: distinct winners", "Joint: 90% coverage"], rows7))
parts.append("\n*The cumulative-coverage curves behind the 90% columns are drawn in the "
             "manuscript's decision-robustness figure, produced by "
             "`supplementary_analysis/make_decision_robustness_figure.py`.*\n")

# ---- anchor and safeguard results (computed, not transcribed)
PROFILES = {
    "Cost": np.array([0.1, 0.7, 0.1, 0.1]),
    "GWP": np.array([0.1, 0.1, 0.7, 0.1]),
    "Energy": np.array([0.7, 0.1, 0.1, 0.1]),
    "D24": np.array([0.1, 0.1, 0.1, 0.7]),
    "Balanced": np.array([0.25, 0.25, 0.25, 0.25]),
}
front_rows = read(EX + "pareto_fronts.csv")


def _objmat(sub):
    return np.array(
        [[float(r["annual_heating_energy_gj"]), float(r["cost_rate_sum_proxy"]),
          float(r["carbon_rate_sum_proxy"]), -float(r["days_below_24_c"])] for r in sub])


def _select(scores, ids):
    best = np.min(scores)
    cand = np.where(np.isclose(scores, best, rtol=0.0, atol=1e-12))[0]
    return int(ids[cand[np.argmin(ids[cand])]])


anchor_changed, guard_moves = [], []
for hz in ("2020", "2050", "2100"):
    sub = [r for r in front_rows if r["horizon"] == hz]
    ids = np.array([int(r["simulation_id"]) for r in sub])
    F = _objmat(sub)
    fmin, fmax = F.min(0), F.max(0)
    Fn = (F - fmin) / np.where(fmax > fmin, fmax - fmin, 1.0)
    ws = {p: _select(Fn @ w, ids) for p, w in PROFILES.items()}
    sub625 = [r for r in allc if r["horizon"] == hz]
    F625 = _objmat(sub625)
    lo, hi = F625.min(0), F625.max(0)
    Fn_alt = (F - lo) / np.where(hi > lo, hi - lo, 1.0)
    anchor_changed += [p for p, w in PROFILES.items() if _select(Fn_alt @ w, ids) != ws[p]]
    d24 = -F[:, 3]
    feas = d24 >= float(d24[ids == 1][0])
    for p, w in PROFILES.items():
        new = _select((Fn[feas]) @ w, ids[feas])
        if new != ws[p]:
            io, in_ = np.where(ids == ws[p])[0][0], np.where(ids == new)[0][0]
            dF = F[in_] - F[io]
            guard_moves.append((HZ[hz], p, ws[p], new, dF))
assert not anchor_changed, anchor_changed
guard_txt = "; ".join(
    f"the {hzl} {p}-priority selection moves {old} → {new} "
    f"({-dF[3]:+.0f} below-threshold days for {dF[0]:+.3f} GJ E_H, "
    f"{dF[1]:+.0f} cost-index and {dF[2]:+.0f} GWP-index points)"
    for hzl, p, old, new, dF in guard_moves) or "no selection changes"
parts.append(f"\n**Anchor and safeguard results.** With Eq. (6) scaling constants taken over "
             f"all 625 packages, all 15 profile selections are unchanged. Under the warm-side "
             f"baseline safeguard, {guard_txt}; all other selections are unchanged.\n")

# ---- S1.14 weight-space selection agreement
wa = read(os.path.join(_HERE, "mcda_weight_agreement.csv"))
MAIN5 = ["shared_mtl_nn", "shared_mtl_nn_mgda", "independent_stl_nn",
         "random_forest", "gradient_boosting"]
WLABEL = {
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
parts.append("\n## Table S1.14 — Weight-space selection agreement, all nine formulations\n")
parts.append("Share of the 10,000 uniform Dirichlet(1,1,1,1) weight draws (seed 42, the "
             "Section 3.4 draws) for which the reconstructed and exact fronts select the same "
             "package under the identical Eq. (6)–(7) protocol with the sweep tie rule; "
             "predicted D24 clipped to [0, 365] before decision analysis. Mean ± SD over the "
             "nine repeat–weather case composites; the SD describes between-composite "
             "variation, not a confidence interval, since all composites share the same draws. "
             "Generated by `supplementary_analysis/mcda_weight_agreement.py`; per-composite "
             "values in `mcda_weight_agreement.csv`. The first five rows are the main-text "
             "formulations (Table 4).\n")
rows14 = []
for design in ("all-predicted", "exact-index"):
    order14 = MAIN5 + [m for m in sorted({r["model"] for r in wa}) if m not in MAIN5]
    for i, m in enumerate(order14):
        vals = np.array([float(r["agreement_pct"]) for r in wa
                         if r["task_design"] == design and r["model"] == m])
        rows14.append([design if i == 0 else "", WLABEL[m],
                       "{:.1f} ± {:.1f}".format(vals.mean(), vals.std(ddof=1))])
parts.append(md_table(["Task design", "Model", "Weight-space agreement (%)"], rows14))

# ---- S1.15 equal-weight exchange equivalents
parts.append("\n## Table S1.15 — Equal-weight exchange equivalents of one D24 day\n")
parts.append("Eq. (6) spans of each weather case's Pareto front (its ideal-to-nadir range per "
             "objective) and the resulting equal-weight exchange rates: sacrificing one "
             "below-threshold day is score-neutral against the tabulated amounts of each other "
             "objective. The cost- and carbon-index spans are identical in all three cases "
             "(I_C 1091.0 points, I_G 200.45 points), since the indices do not depend on "
             "weather. For the D24-priority stress profile (weight 0.7 against 0.1), multiply "
             "each equivalent by seven.\n")
rows15 = []
for hz in ("2020", "2050", "2100"):
    sub = [r for r in front_rows if r["horizon"] == hz]
    F = _objmat(sub)
    span = F.max(0) - F.min(0)  # D24 enters negated; its span magnitude is unchanged
    rows15.append([HZ[hz], "{:.0f}".format(span[3]), "{:.2f}".format(span[0]),
                   "{:.3g}".format(span[0] / span[3]), "{:.1f}".format(span[1] / span[3]),
                   "{:.3g}".format(span[2] / span[3])])
parts.append(md_table(["Weather case", "D24 span (d)", "E_H span (GJ)", "1 day ≡ E_H (GJ)",
                       "1 day ≡ I_C (pts)", "1 day ≡ I_G (pts)"], rows15))

# ---- S1.16 first-rank acceptability and central weight vectors
cw = read(os.path.join(_HERE, "mcda_central_weights.csv"))
parts.append("\n## Table S1.16 — First-rank acceptability and central weight vectors "
             "(1,000,000 draws)\n")
parts.append("Top three packages per weather case by first-rank acceptability a_i, with "
             "central weight vectors w^c = (E_H, I_C, I_G, D24), both from the same "
             "1,000,000-draw run (see S1.0); all ever-winning packages are in "
             "`mcda_central_weights.csv`.\n")
rows16 = []
for hz in ("2020", "2050", "2100"):
    sub = sorted((r for r in cw if r["horizon"] == hz),
                 key=lambda r: -float(r["acceptability_pct_1m"]))[:3]
    for i, r in enumerate(sub):
        rows16.append([HZ[hz] if i == 0 else "", r["simulation_id"],
                       "{:.2f}".format(float(r["acceptability_pct_1m"])),
                       "({:.3f}, {:.3f}, {:.3f}, {:.3f})".format(
                           float(r["wc_energy"]), float(r["wc_cost"]),
                           float(r["wc_gwp"]), float(r["wc_d24"]))])
parts.append(md_table(["Weather case", "Package", "a_i (%)", "w^c (E_H, I_C, I_G, D24)"],
                      rows16))

# ---- S1.17 per-case per-profile selections under the two scalarising rules
PROF_LABEL = {"Cost": "Cost", "GWP": "GWP", "Energy": "Energy", "D24": "D24",
              "Balanced": "Equal-weight"}
PROF_ORDER = ["Cost", "GWP", "Energy", "D24", "Balanced"]
parts.append("\n## Table S1.17 — Weighted-sum and achievement-scalarising selections "
             "per case and profile\n")
parts.append("Per-case per-profile selections under the two scalarising rules (from "
             "`mcda_profile_sensitivity.csv`); the achievement-scalarising selections are "
             "identical for ρ ∈ {10⁻⁸, 10⁻⁶, 10⁻⁴, 10⁻²}. The two rules agree in "
             f"{len(_agreeing)} of the {len(_prof)} cases.\n")
rows17 = []
for hz in ("2020", "2050", "2100"):
    for i, pname in enumerate(PROF_ORDER):
        r = next(x for x in _prof if x["horizon"] == hz and x["profile"] == pname)
        rows17.append([HZ[hz] if i == 0 else "", PROF_LABEL[pname],
                       r["weighted_sum_selection"], r["asf_selection"],
                       "yes" if r["agreement"] == "True" else "no"])
parts.append(md_table(["Weather case", "Profile", "Weighted-sum package", "ASF package",
                       "Same?"], rows17))

# ---- S1.18 bound-violation rates of the raw day-count predictions
MAIN5_LBL = [("shared_mtl_nn", "Joint NN, equal weighting"),
             ("shared_mtl_nn_mgda", "Joint NN, gradient-balanced"),
             ("independent_stl_nn", "Single-task NNs"),
             ("random_forest", "Random forest"),
             ("gradient_boosting", "Gradient-boosted trees")]
_pv = {"exact-index": read(SV + "surrogate_pareto_validation.csv"),
       "all-predicted": read(os.path.join(ROOTDIR, "surrogate_validation_family4",
                                          "surrogate_pareto_validation.csv"))}
parts.append("\n## Table S1.18 — Raw day-count predictions outside the physical bound\n")
parts.append("Share of the raw out-of-fold D24 predictions above 365 days, per formulation, "
             "task design and weather case (percent of the 1,875 predictions per cell: "
             "625 packages × 3 repeats). The reported percentages use a strict comparison "
             "with 365 days. The forest’s exceedances are floating-point roundoff, with a "
             "maximum prediction of 365.0000000000002 days; the neural and boosted-tree "
             "formulations produce larger exceedances. All predictions are clipped to "
             "[0, 365] before any "
             "decision analysis. No prediction falls below 0 in either design. From "
             "`surrogate_pareto_validation.csv` of both designs.\n")
rows18 = []
for design in ("all-predicted", "exact-index"):
    for i, (m, lbl) in enumerate(MAIN5_LBL):
        cells = [design if i == 0 else "", lbl]
        for hz in ("2020", "2050", "2100"):
            sub = [r for r in _pv[design] if r["model"] == m and r["horizon"] == hz]
            assert len(sub) == 3, (design, m, hz, len(sub))
            above = sum(int(r["raw_predicted_days_above_365_count"]) for r in sub)
            assert all(int(r["raw_predicted_days_below_zero_count"]) == 0 for r in sub)
            cells.append(f"{100.0 * above / 1875.0:.1f}%")
        rows18.append(cells)
parts.append(md_table(["Task design", "Model", "Present", "Mid-century", "Late-century"],
                      rows18))

# This script emits the generated tables only.  SUPPLEMENTARY_S1.md also carries
# hand-maintained sections (S1.0 methods notes, S1.10, S1.19) that are not
# reproduced here, so overwriting it in place destroys them.  Refuse unless the
# caller asks for it explicitly.
if os.path.exists(OUT) and os.environ.get("S1_ALLOW_OVERWRITE") != "1":
    raise SystemExit(
        f"make_supplementary_tables: {OUT} already exists and this script emits "
        "only the generated subset of the document (S1.0, S1.10 and S1.19 are "
        "hand-maintained and would be lost).\n"
        "Set S1_OUT to a scratch path to inspect the generated tables, or "
        "S1_ALLOW_OVERWRITE=1 if you really intend to replace the document."
    )
open(OUT, "w").write("\n".join(parts) + "\n")
print("wrote", OUT)
