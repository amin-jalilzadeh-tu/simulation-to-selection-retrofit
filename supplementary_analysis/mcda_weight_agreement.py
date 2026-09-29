"""Continuous weight-space selection agreement between surrogate and exact fronts.

Weight-space selection agreement is the proportion of uniformly sampled admissible
weight vectors for which the surrogate-derived front and the exact front select the
same package under the identical Eq. (6)-(7) protocol: build the Pareto set, min-max
normalise over that set's own ideal and nadir, score with the weighted sum, and take
the lowest-scoring member (exact-equality argmin over id-sorted scores, lowest
simulation id first - the sweep tie convention of mcda_supportedness_sweep.py).
It complements the five-profile selection agreement of the manuscript by integrating
over the admissible weight space instead of evaluating five chosen weight points; it
remains conditional on the weighted-sum rule and the normalisation.

Inputs (released S1 data):
  results/exhaustive_analysis/pareto_fronts.csv                exact fronts
  results/surrogate_validation/oof_predictions.csv             exact-index design
  results/surrogate_validation_family4/oof_predictions.csv     all-predicted design

Three arms are evaluated: the two task designs of the manuscript plus a
substitution-only ablation that reuses the four-target predictions of E_H and D24
with the two indices taken from the component tables.

For each of the 3 arms x 9 formulations x 9 repeat-case composites, predictions are
returned to their feasible ranges before decision analysis, as in the manuscript:
predicted D24 to [0, 365] and, in the all-predicted arm, the predicted cost and GWP
indices to [0, inf);
the predicted front is the Pareto set of the predicted objective vectors over all 625
packages; agreement is evaluated on the same 10,000 Dirichlet(1,1,1,1) draws (seed 42)
used in Section 3.4. Writes mcda_weight_agreement.csv (per-composite values) and
prints the pooled mean +/- SD per formulation and design. The SD is descriptive
variation across the nine composites, which share the same weight draws - not a
confidence interval.

Deterministic: fixed seed 42.
"""
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))


def _dir(env_key, sub):
    for p in (os.environ.get(env_key), os.path.join(_HERE, os.pardir, "results", sub)):
        if p and os.path.isdir(p):
            return p
    raise FileNotFoundError(f"{sub} not found; set {env_key}")


EX = _dir("S1_EXHAUSTIVE", "exhaustive_analysis")
# The two task-design OOF files live next to the exhaustive_analysis directory.
RESULTS = os.path.dirname(EX.rstrip(os.sep))
_FOUR = os.path.join(RESULTS, "surrogate_validation_family4", "oof_predictions.csv")
DESIGNS = {
    "exact-index": os.path.join(RESULTS, "surrogate_validation", "oof_predictions.csv"),
    "all-predicted": _FOUR,
    # Substitution-only ablation: the four-target models' E_H and D24 predictions are
    # retained and only the two tabulated indices are replaced by their exact values,
    # so the effect of substituting the indices is separated from the effect of
    # refitting the joint networks and the forest on two targets.
    "substitution-only": _FOUR,
}
OUTCSV = os.path.join(_HERE, "mcda_weight_agreement.csv")

HORIZONS = [2020, 2050, 2100]
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

rng = np.random.default_rng(42)
W = rng.dirichlet(np.ones(4), size=10000)


def pareto_mask(F):
    """Boolean mask of non-dominated rows of F (all objectives minimised)."""
    leq = (F[None, :, :] <= F[:, None, :]).all(axis=2)
    lt = (F[None, :, :] < F[:, None, :]).any(axis=2)
    return ~(leq & lt).any(axis=1)


def winners(F, ids):
    """Per-draw winning simulation id on the front F under Eq. (6)-(7)."""
    fmin, fmax = F.min(0), F.max(0)
    Fn = (F - fmin) / np.where(fmax > fmin, fmax - fmin, 1.0)
    order = np.argsort(ids)
    return ids[order[np.argmin((W @ Fn.T)[:, order], axis=1)]]


# --- exact winners per horizon (from the released exact fronts)
exact = pd.read_csv(os.path.join(EX, "pareto_fronts.csv"))
EXACT_WIN = {}
for hz in HORIZONS:
    sub = exact[exact["horizon"] == hz]
    F = np.column_stack([
        sub["annual_heating_energy_gj"].to_numpy(float),
        sub["cost_rate_sum_proxy"].to_numpy(float),
        sub["carbon_rate_sum_proxy"].to_numpy(float),
        -sub["days_below_24_c"].to_numpy(float),
    ])
    EXACT_WIN[hz] = winners(F, sub["simulation_id"].to_numpy(int))

# --- predicted winners per design x model x repeat x horizon
out = []
for design, path in DESIGNS.items():
    oof = pd.read_csv(path)
    for model, mdf in oof.groupby("model"):
        for repeat, rdf in mdf.groupby("repeat_seed"):
            for hz in HORIZONS:
                sub = rdf[rdf["horizon"] == hz]
                assert len(sub) == 625, (design, model, repeat, hz, len(sub))
                d24 = np.clip(sub["predicted_days_below_24_c"].to_numpy(float), 0.0, 365.0)
                if design == "all-predicted":
                    # Predicted indices are clipped at zero: both are sums of
                    # non-negative component rates, so a negative prediction is
                    # outside the feasible range and would otherwise become the
                    # ideal used in the Eq. (6) normalisation.
                    cost = np.clip(sub["predicted_cost_rate_sum_proxy"].to_numpy(float), 0.0, None)
                    carbon = np.clip(sub["predicted_carbon_rate_sum_proxy"].to_numpy(float), 0.0, None)
                else:  # exact-index and substitution-only: indices taken exactly
                    cost = sub["cost_rate_sum_proxy"].to_numpy(float)
                    carbon = sub["carbon_rate_sum_proxy"].to_numpy(float)
                F = np.column_stack([
                    sub["predicted_annual_heating_energy_gj"].to_numpy(float),
                    cost, carbon, -d24,
                ])
                ids = sub["simulation_id"].to_numpy(int)
                mask = pareto_mask(F)
                pred_win = winners(F[mask], ids[mask])
                agree = float(np.mean(pred_win == EXACT_WIN[hz]) * 100.0)
                out.append({"task_design": design, "model": model,
                            "repeat_seed": int(repeat), "horizon": hz,
                            "agreement_pct": round(agree, 2)})

df = pd.DataFrame(out).sort_values(["task_design", "model", "repeat_seed", "horizon"])
df.to_csv(OUTCSV, index=False)
print(f"wrote {OUTCSV} ({len(df)} rows)\n")

print("Weight-space selection agreement, pooled over the nine repeat-case composites")
print("(mean +/- SD across composites; SD is descriptive, the composites share the draws)\n")
for design in ("all-predicted", "substitution-only", "exact-index"):
    print(f"== {design} ==")
    for model in MAIN5 + [m for m in sorted(df["model"].unique()) if m not in MAIN5]:
        vals = df[(df["task_design"] == design) & (df["model"] == model)]["agreement_pct"]
        flag = "" if model in MAIN5 else "  [S1-only formulation]"
        print(f"  {LABEL[model]:38s} {vals.mean():5.1f} +/- {vals.std(ddof=1):4.1f} %{flag}")
    print()
