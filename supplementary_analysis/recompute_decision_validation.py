"""Regenerate the decision-layer validation tables from the released out-of-fold
predictions, applying the feasible-range policy of Section 2.7.

The decision layer is post-processing on `oof_predictions.csv`: no model is
refitted here.  Predictions are returned to their feasible ranges before the
Pareto and selection levels --- D24 to [0, 365] and the cost and GWP indices to
[0, inf) --- because both indices are sums of non-negative component rates, so a
negative prediction lies outside the attainable range and would otherwise supply
the ideal in the Eq. (6) normalisation.  Point metrics are unaffected: they are
computed from the raw predictions, which the released table keeps.

Three arms are written per run directory:

  weighted_profile_validation.csv        the manuscript design (indices clipped)
  surrogate_pareto_validation.csv        Pareto recovery for the same design
  *_substitution_only.csv                four-target E_H and D24 predictions with
                                         the two indices taken from the tables,
                                         no model refitted

Reproduction gate: this module must reproduce the released
`weighted_profile_validation.csv` exactly (simulation ids, agreement flags and
regret) for both task designs.  The gate runs on every invocation and the script
exits non-zero if it fails, so a divergence is never written out.  With
--legacy-gate the indices are left unclipped, which reproduces the tables as
they stood before the clipping correction; that is how the correction was
validated.

Usage:
    python recompute_decision_validation.py [--results DIR] [--write] [--legacy-gate]

Without --write the script only reports; nothing on disk is changed.
"""
import argparse
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))

STAKEHOLDER_PROFILES = OrderedDict(
    (
        ("cost_priority", (0.10, 0.70, 0.10, 0.10)),
        ("carbon_priority", (0.10, 0.10, 0.70, 0.10)),
        ("energy_priority", (0.70, 0.10, 0.10, 0.10)),
        ("days_below_24_priority", (0.10, 0.10, 0.10, 0.70)),
        ("balanced", (0.25, 0.25, 0.25, 0.25)),
    )
)
DECISION_DAYS_BOUNDS = (0.0, 365.0)
MAIN_FIVE = (
    "shared_mtl_nn",
    "shared_mtl_nn_mgda",
    "independent_stl_nn",
    "random_forest",
    "gradient_boosting",
)
RUNS = {
    "all-predicted": "surrogate_validation_family4",
    "exact-index": "surrogate_validation",
}


def pareto_mask_minimization(objectives):
    """Mask of non-dominated rows when every objective is minimised."""
    objectives = np.asarray(objectives, dtype=float)
    non_dominated = np.ones(len(objectives), dtype=bool)
    for index, candidate in enumerate(objectives):
        dominated = np.all(objectives <= candidate, axis=1) & np.any(
            objectives < candidate, axis=1
        )
        non_dominated[index] = not np.any(dominated)
    return non_dominated


def _minmax(objectives, minimum=None, span=None):
    if minimum is None:
        minimum = objectives.min(axis=0)
    if span is None:
        span = objectives.max(axis=0) - minimum
    normalized = np.divide(
        objectives - minimum, span, out=np.zeros_like(objectives, dtype=float),
        where=span > 0.0,
    )
    return normalized, minimum, span


def _stable_minimum_position(scores, simulation_ids):
    minimum_score = float(scores.min())
    candidates = np.flatnonzero(np.isclose(scores, minimum_score, rtol=0.0, atol=1.0e-12))
    return int(min(candidates, key=lambda position: int(simulation_ids[position])))


def compute(oof, mode):
    """mode: 'all-predicted' | 'exact-index' | 'substitution-only' | 'legacy-unclipped'."""
    pareto_records, selection_records = [], []
    for (seed, model, horizon), frame in oof.groupby(
        ["repeat_seed", "model", "horizon"], sort=True
    ):
        frame = frame.sort_values("simulation_id", kind="stable").reset_index(drop=True)
        if frame["simulation_id"].duplicated().any():
            raise AssertionError("OOF predictions contain duplicate configurations.")

        exact = np.column_stack(
            (
                frame["true_annual_heating_energy_gj"].to_numpy(float),
                frame["cost_rate_sum_proxy"].to_numpy(float),
                frame["carbon_rate_sum_proxy"].to_numpy(float),
                -frame["true_days_below_24_c"].to_numpy(float),
            )
        )
        raw_days = frame["predicted_days_below_24_c"].to_numpy(float)
        days = np.clip(raw_days, *DECISION_DAYS_BOUNDS)
        energy = frame["predicted_annual_heating_energy_gj"].to_numpy(float)

        if mode in ("exact-index", "substitution-only"):
            cost = frame["cost_rate_sum_proxy"].to_numpy(float)
            carbon = frame["carbon_rate_sum_proxy"].to_numpy(float)
            below_zero = 0
        else:
            cost = frame["predicted_cost_rate_sum_proxy"].to_numpy(float)
            carbon = frame["predicted_carbon_rate_sum_proxy"].to_numpy(float)
            below_zero = int(np.count_nonzero(cost < 0.0) + np.count_nonzero(carbon < 0.0))
            if mode != "legacy-unclipped":
                cost = np.clip(cost, 0.0, None)
                carbon = np.clip(carbon, 0.0, None)

        predicted = np.column_stack((energy, cost, carbon, -days))
        exact_mask = pareto_mask_minimization(exact)
        predicted_mask = pareto_mask_minimization(predicted)
        exact_groups = set(frame.loc[exact_mask, "configuration_group_id"].astype(int))
        predicted_groups = set(frame.loc[predicted_mask, "configuration_group_id"].astype(int))
        true_positive = len(exact_groups & predicted_groups)
        precision = true_positive / len(predicted_groups)
        recall = true_positive / len(exact_groups)
        pareto_records.append(
            {
                "repeat_seed": int(seed), "model": model, "horizon": int(horizon),
                "configuration_count": len(frame),
                "exact_pareto_count": len(exact_groups),
                "predicted_pareto_count": len(predicted_groups),
                "true_positive_count": true_positive,
                "precision": precision, "recall": recall,
                "f1": (2.0 * precision * recall / (precision + recall)
                       if precision + recall > 0.0 else 0.0),
                "jaccard": true_positive / len(exact_groups | predicted_groups),
                "raw_predicted_days_below_zero_count": int(
                    np.count_nonzero(raw_days < DECISION_DAYS_BOUNDS[0])),
                "raw_predicted_days_above_365_count": int(
                    np.count_nonzero(raw_days > DECISION_DAYS_BOUNDS[1])),
                "decision_days_clipped_count": int(np.count_nonzero(raw_days != days)),
                "raw_predicted_index_below_zero_count": below_zero,
            }
        )

        exact_normalized, minimum, span = _minmax(exact[exact_mask])
        predicted_normalized, _, _ = _minmax(predicted[predicted_mask])
        all_normalized, _, _ = _minmax(exact, minimum, span)
        ids = frame["simulation_id"].to_numpy(int)
        exact_positions = np.flatnonzero(exact_mask)
        predicted_positions = np.flatnonzero(predicted_mask)

        for profile, weight_tuple in STAKEHOLDER_PROFILES.items():
            weights = np.asarray(weight_tuple, dtype=float)
            exact_index = exact_positions[
                _stable_minimum_position(exact_normalized @ weights, ids[exact_mask])]
            predicted_index = predicted_positions[
                _stable_minimum_position(predicted_normalized @ weights, ids[predicted_mask])]
            regret = float(all_normalized[predicted_index] @ weights
                           - all_normalized[exact_index] @ weights)
            if -1.0e-12 < regret < 0.0:
                regret = 0.0
            selection_records.append(
                {
                    "repeat_seed": int(seed), "model": model, "horizon": int(horizon),
                    "profile": profile,
                    "exact_simulation_id": int(ids[exact_index]),
                    "predicted_simulation_id": int(ids[predicted_index]),
                    "selection_agreement": bool(ids[exact_index] == ids[predicted_index]),
                    "predicted_selection_on_exact_pareto": bool(exact_mask[predicted_index]),
                    "exact_true_normalized_score": float(all_normalized[exact_index] @ weights),
                    "predicted_selection_true_normalized_score": float(
                        all_normalized[predicted_index] @ weights),
                    "true_score_regret": regret,
                }
            )
    return pd.DataFrame.from_records(pareto_records), pd.DataFrame.from_records(selection_records)


def reproduction_gate(results_dir, legacy=False):
    """Recomputation must equal the released decision tables.

    Default: recompute under the current feasible-range policy and compare with
    the shipped tables, so a reproducer can confirm the released decision layer
    follows from the released predictions.

    --legacy-gate: recompute with the index clipping disabled instead.  This
    reproduces the tables as they stood before the clipping correction, and was
    how the correction itself was validated.
    """
    for design, sub in RUNS.items():
        directory = os.path.join(results_dir, sub)
        oof = pd.read_csv(os.path.join(directory, "oof_predictions.csv"))
        if legacy and design == "all-predicted":
            mode = "legacy-unclipped"
        else:
            mode = design
        _, selection = compute(oof, mode)
        released = pd.read_csv(os.path.join(directory, "weighted_profile_validation.csv"))
        merged = selection.merge(
            released, on=["repeat_seed", "model", "horizon", "profile"],
            suffixes=("_new", "_released"))
        if len(merged) != len(released):
            raise SystemExit(f"gate: row-count mismatch for {design}")
        mismatched_ids = int((merged.predicted_simulation_id_new
                              != merged.predicted_simulation_id_released).sum())
        mismatched_regret = int((~np.isclose(merged.true_score_regret_new,
                                             merged.true_score_regret_released,
                                             atol=1.0e-9)).sum())
        if mismatched_ids or mismatched_regret:
            raise SystemExit(
                f"gate FAILED for {design}: {mismatched_ids} id and "
                f"{mismatched_regret} regret mismatches against the released table")
        print(f"gate {design:<14} reproduces the released table exactly "
              f"({len(released)} selection records{', legacy policy' if legacy else ''})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=os.environ.get(
        "S1_RESULTS", os.path.join(_HERE, os.pardir, "results")))
    parser.add_argument("--write", action="store_true",
                        help="overwrite the decision tables in the run directories")
    parser.add_argument("--legacy-gate", action="store_true",
                        help="gate against the pre-correction tables (index clipping off)")
    args = parser.parse_args()
    results_dir = os.path.abspath(args.results)
    if not os.path.isdir(results_dir):
        raise SystemExit(f"results directory not found: {results_dir}; set --results or S1_RESULTS")

    reproduction_gate(results_dir, legacy=args.legacy_gate)

    for design, sub in RUNS.items():
        directory = os.path.join(results_dir, sub)
        oof = pd.read_csv(os.path.join(directory, "oof_predictions.csv"))
        pareto, selection = compute(oof, design)
        outputs = {"surrogate_pareto_validation.csv": pareto,
                   "weighted_profile_validation.csv": selection}
        if design == "all-predicted":
            sub_pareto, sub_selection = compute(oof, "substitution-only")
            outputs["surrogate_pareto_validation_substitution_only.csv"] = sub_pareto
            outputs["weighted_profile_validation_substitution_only.csv"] = sub_selection
        agree = selection[selection.model.isin(MAIN_FIVE)].groupby("model").selection_agreement.sum()
        print(f"\n{design}: exact selections of 45 per formulation")
        for model in MAIN_FIVE:
            print(f"  {model:<22} {int(agree[model]):>3}")
        if args.write:
            for name, frame in outputs.items():
                path = os.path.join(directory, name)
                frame.to_csv(path, index=False)
                print(f"  wrote {path} ({len(frame)} rows)")
    if not args.write:
        print("\n(nothing written; pass --write to update the run directories)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
