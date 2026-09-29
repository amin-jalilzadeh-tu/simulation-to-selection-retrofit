"""Read-only summaries of archived S1 CSVs; no fitting, simulation or selection reruns."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MAIN5 = ["shared_mtl_nn", "shared_mtl_nn_mgda", "independent_stl_nn", "random_forest", "gradient_boosting"]

def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))

def yes(v):
    return str(v).lower() == "true"

def describe(rows):
    vals = [float(r["true_score_regret"]) for r in rows]
    return {
        "n": len(rows), "median_regret": statistics.median(vals) if vals else None,
        "max_regret": max(vals) if vals else None,
        "regret_above_0.05": sum(v > .05 for v in vals),
        "regret_above_0.1": sum(v > .1 for v in vals),
        "exact_pareto_count": sum(yes(r["predicted_selection_on_exact_pareto"]) for r in rows),
    }

oof = [r for r in read(RESULTS / "surrogate_validation_family4/oof_predictions.csv") if r["model"] in MAIN5]
groups = defaultdict(list)
for r in oof:
    groups[(r["model"], r["repeat_seed"], r["horizon"])].append(r)

baseline = {}
for model in MAIN5:
    detail = []
    for (m, seed, horizon), rows in groups.items():
        if m != model:
            continue
        base = next(r for r in rows if r["simulation_id"] == "1")
        cost = float(base["predicted_cost_rate_sum_proxy"])
        lower = sum(float(r["predicted_cost_rate_sum_proxy"]) < cost for r in rows)
        lower_clipped = sum(max(0.0, float(r["predicted_cost_rate_sum_proxy"])) < max(0.0, cost) for r in rows)
        detail.append({"seed": seed, "horizon": horizon, "baseline_predicted_cost": cost,
                       "baseline_predicted_gwp": float(base["predicted_carbon_rate_sum_proxy"]),
                       "packages_predicted_cheaper": lower,
                       "packages_predicted_cheaper_after_clipping": lower_clipped})
    costs = [r["baseline_predicted_cost"] for r in detail]
    baseline[model] = {"cost_min": min(costs), "cost_median": statistics.median(costs), "cost_max": max(costs),
                       "composites_with_cheaper_predicted_package": sum(r["packages_predicted_cheaper"] > 0 for r in detail),
                       "composites_with_cheaper_predicted_package_after_clipping": sum(r["packages_predicted_cheaper_after_clipping"] > 0 for r in detail),
                       "detail": detail}

arms = {}
for arm, folder in [("all-predicted", "surrogate_validation_family4"), ("exact-index", "surrogate_validation")]:
    rows = [r for r in read(RESULTS / folder / "weighted_profile_validation.csv") if r["model"] in MAIN5]
    arms[arm] = rows
arms["substitution-only"] = [r for r in read(RESULTS / "surrogate_validation_family4/weighted_profile_validation_substitution_only.csv") if r["model"] in MAIN5]

selection_summary = {}
for arm, rows in arms.items():
    selection_summary[arm] = {}
    for m in MAIN5:
        mr = [r for r in rows if r["model"] == m]
        br = [r for r in mr if r["exact_simulation_id"] == "1"]
        selection_summary[arm][m] = {
            "all_cases": len(mr), "all_matches": sum(yes(r["selection_agreement"]) for r in mr),
            "exact_baseline_cases": len(br), "baseline_matches": sum(yes(r["selection_agreement"]) for r in br),
            "baseline_mismatches": describe([r for r in br if not yes(r["selection_agreement"])]),
        }

pooled_regret = {}
for arm, rows in list(arms.items()) + [("both_designs", arms["all-predicted"] + arms["exact-index"])]:
    pooled_regret[arm] = {}
    for subset, rr in [("all_cases", rows), ("mismatches_only", [r for r in rows if not yes(r["selection_agreement"])])]:
        pooled_regret[arm][subset] = {
            "all": describe(rr),
            "exact_baseline_winner": describe([r for r in rr if r["exact_simulation_id"] == "1"]),
            "other_exact_winner": describe([r for r in rr if r["exact_simulation_id"] != "1"]),
        }
    pooled_regret[arm]["largest_regret_cases"] = sorted(rows, key=lambda r: float(r["true_score_regret"]), reverse=True)[:8]

wa = [r for r in read(ROOT / "supplementary_analysis/mcda_weight_agreement.csv") if r["model"] in MAIN5]
ws = {arm: {m: statistics.mean(float(r["agreement_pct"]) for r in wa if r["task_design"] == arm and r["model"] == m) for m in MAIN5}
      for arm in ("all-predicted", "exact-index", "substitution-only")}
mcda = [r for r in read(ROOT / "supplementary_analysis/mcda_outputs.csv") if r["simulation_id"] == "1"]
ws["always_baseline_10k"] = {r["horizon"]: float(r["acceptability_pct"]) for r in mcda}
ws["always_baseline_10k_mean"] = statistics.mean(ws["always_baseline_10k"].values())

by_model = {}
for m in ("shared_mtl_nn", "shared_mtl_nn_mgda"):
    by_model[m] = {(r["repeat_seed"], r["horizon"], r["profile"]): r for r in arms["all-predicted"] if r["model"] == m}
comparison = []
for key, eq in by_model["shared_mtl_nn"].items():
    mg = by_model["shared_mtl_nn_mgda"][key]
    if yes(eq["selection_agreement"]) != yes(mg["selection_agreement"]):
        comparison.append({"key": key, "equal_matches": yes(eq["selection_agreement"]), "mgda_matches": yes(mg["selection_agreement"]),
                           "equal_regret": float(eq["true_score_regret"]), "mgda_regret": float(mg["true_score_regret"]),
                           "exact_simulation_id": eq["exact_simulation_id"]})

print(json.dumps({"source": str(RESULTS.relative_to(ROOT)), "baseline_predictions": baseline, "selection_counts": selection_summary,
                  "pooled_regret": pooled_regret, "weight_space": ws, "equal_vs_mgda_discordant_matches": comparison}, indent=2))
