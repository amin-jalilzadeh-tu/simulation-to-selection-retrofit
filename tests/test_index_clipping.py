"""Regression test for the result-affecting correction: predicted cost and GWP
indices are returned to their feasible range before the decision layer.

Both indices are sums of non-negative component rates. A negative prediction is
outside the attainable range and, left unclipped, becomes the ideal of the
Eq. (6) min-max normalisation, displacing the whole axis and changing which
package the weighted sum selects. These tests pin that behaviour so a future
regeneration cannot silently revert it.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.validate_mtl_family import compute_decision_validation  # noqa: E402


def _frame(cost_prediction_for_second_package):
    """Three packages under one repeat/horizon.

    Exact objectives make package 2 the weighted-sum winner on the cost profile.
    Package 3 carries the manipulated cost prediction.
    """
    rows = []
    # Package 2 is the incremental-zero package, as the unrenovated package is in the
    # study: it holds the true minimum of the cost axis. An unclipped negative
    # prediction on package 3 therefore lands BELOW the true ideal and captures the
    # cost-priority selection; clipping returns it to zero, where the tie rule
    # restores the lowest simulation id.
    exact = [
        # sim, cost, carbon, heating, days
        (1, 100.0, 10.0, 30.0, 360.0),
        (2, 0.0, 20.0, 28.0, 355.0),
        (3, 200.0, 30.0, 20.0, 350.0),
    ]
    for sim, cost, carbon, heat, days in exact:
        prediction_cost = cost if sim != 3 else cost_prediction_for_second_package
        rows.append(
            {
                "repeat_seed": 17, "outer_fold": 0, "model": "unit_test",
                "dataset_row_id": sim, "configuration_group_id": sim,
                "horizon": 2100, "simulation_id": sim,
                "cost_rate_sum_proxy": cost, "carbon_rate_sum_proxy": carbon,
                "true_annual_heating_energy_gj": heat,
                "predicted_annual_heating_energy_gj": heat,
                "true_days_below_24_c": days,
                "predicted_days_below_24_c": days,
                "true_cost_rate_sum_proxy": cost,
                "predicted_cost_rate_sum_proxy": prediction_cost,
                "true_carbon_rate_sum_proxy": carbon,
                "predicted_carbon_rate_sum_proxy": carbon,
            }
        )
    return pd.DataFrame(rows)


class TestIndexClipping(unittest.TestCase):
    def test_negative_index_is_clipped_and_restores_the_exact_selection(self):
        """A negative predicted cost must not become the normalisation ideal."""
        honest = _frame(200.0)
        _, honest_selection = compute_decision_validation(honest, use_predicted_cost_carbon=True)
        cost_row = honest_selection[honest_selection.profile == "cost_priority"].iloc[0]
        self.assertTrue(bool(cost_row.selection_agreement),
                        "control case must already agree; the fixture is wrong otherwise")

        poisoned = _frame(-500.0)
        _, poisoned_selection = compute_decision_validation(poisoned, use_predicted_cost_carbon=True)
        poisoned_row = poisoned_selection[poisoned_selection.profile == "cost_priority"].iloc[0]
        # With clipping in force the impossible value is pulled to zero, so it can no
        # longer sit below the true ideal of the cost axis and steal the selection.
        self.assertEqual(int(poisoned_row.predicted_simulation_id),
                         int(cost_row.exact_simulation_id),
                         "a negative predicted index changed the selection: clipping is not applied")

    def test_clipping_is_reported_in_the_pareto_diagnostics(self):
        pareto, _ = compute_decision_validation(_frame(-500.0), use_predicted_cost_carbon=True)
        self.assertIn("raw_predicted_index_below_zero_count", pareto.columns,
                      "the negative-index diagnostic column is missing from the Pareto table")
        self.assertEqual(int(pareto.iloc[0]["raw_predicted_index_below_zero_count"]), 1)
        clean, _ = compute_decision_validation(_frame(200.0), use_predicted_cost_carbon=True)
        self.assertEqual(int(clean.iloc[0]["raw_predicted_index_below_zero_count"]), 0)

    def test_exact_index_design_never_reports_negative_indices(self):
        pareto, _ = compute_decision_validation(_frame(-500.0), use_predicted_cost_carbon=False)
        self.assertEqual(int(pareto.iloc[0]["raw_predicted_index_below_zero_count"]), 0,
                         "the exact-index design does not use predicted indices at all")

    def test_heating_energy_is_not_clipped(self):
        """Only the two indices and the day count are bounded; heating is not."""
        frame = _frame(200.0)
        frame.loc[frame.simulation_id == 3, "predicted_annual_heating_energy_gj"] = -5.0
        pareto, _ = compute_decision_validation(frame, use_predicted_cost_carbon=True)
        # A negative heating prediction dominates on that axis and must still reach the
        # front, which it cannot do if heating were silently clipped to zero.
        self.assertGreaterEqual(int(pareto.iloc[0]["predicted_pareto_count"]), 1)

    def test_released_all_predicted_table_matches_a_clipped_recomputation(self):
        """End-to-end: the shipped decision table follows from the shipped predictions."""
        directory = os.path.join(REPO_ROOT, "results", "surrogate_validation_family4")
        oof = pd.read_csv(os.path.join(directory, "oof_predictions.csv"))
        _, selection = compute_decision_validation(oof, use_predicted_cost_carbon=True)
        released = pd.read_csv(os.path.join(directory, "weighted_profile_validation.csv"))
        merged = selection.merge(released, on=["repeat_seed", "model", "horizon", "profile"],
                                 suffixes=("_new", "_released"))
        self.assertEqual(len(merged), len(released))
        self.assertEqual(int((merged.predicted_simulation_id_new
                              != merged.predicted_simulation_id_released).sum()), 0)

    def test_released_predictions_really_contain_negative_indices(self):
        """The defect this test guards against is present in the released predictions."""
        directory = os.path.join(REPO_ROOT, "results", "surrogate_validation_family4")
        oof = pd.read_csv(os.path.join(directory, "oof_predictions.csv"))
        negative = ((oof.predicted_cost_rate_sum_proxy < 0)
                    | (oof.predicted_carbon_rate_sum_proxy < 0))
        self.assertGreater(int(negative.sum()), 0)
        for model in ("random_forest", "gradient_boosting"):
            self.assertEqual(int(negative[oof.model == model].sum()), 0,
                             f"{model} produced no negative index in this run")


if __name__ == "__main__":
    unittest.main()
