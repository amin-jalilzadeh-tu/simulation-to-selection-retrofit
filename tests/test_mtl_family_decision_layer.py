"""Tests for the decision layer of scripts/validate_mtl_family.py.

The archive's clipping rule is narrow and load-bearing: predicted days below
24 C are clipped to [0, 365] immediately before the Pareto and weighted-profile
comparisons, heating energy is not clipped, nothing is rounded, and the raw values
survive for every point metric. These tests also check that the clip changes a
front, that the other objectives are left unbounded, and that the four-target
design routes predicted cost and GWP into the front.

Run from the extracted archive root:

    python -m unittest discover -s tests
"""

import unittest

import numpy as np
import pandas as pd

from scripts import validate_mtl_family as family


def oof_frame(rows, model="shared_mtl_nn", repeat_seed=11, horizon=2020):
    """Build a minimal out-of-fold frame with one row per package."""
    records = []
    for index, row in enumerate(rows, start=1):
        record = {
            "repeat_seed": repeat_seed,
            "model": model,
            "horizon": horizon,
            "simulation_id": row.get("simulation_id", index),
            "configuration_group_id": row.get("simulation_id", index),
            "cost_rate_sum_proxy": row["cost"],
            "carbon_rate_sum_proxy": row["carbon"],
            "true_annual_heating_energy_gj": row["true_energy"],
            "true_days_below_24_c": row["true_days"],
            "predicted_annual_heating_energy_gj": row["predicted_energy"],
            "predicted_days_below_24_c": row["predicted_days"],
        }
        if "predicted_cost" in row:
            record["predicted_cost_rate_sum_proxy"] = row["predicted_cost"]
            record["predicted_carbon_rate_sum_proxy"] = row["predicted_carbon"]
        records.append(record)
    return pd.DataFrame.from_records(records)


def fast_config(**overrides) -> family.ValidationConfig:
    settings = dict(
        seeds=(11,),
        outer_folds=3,
        validation_fraction=0.2,
        trunk_widths=(8, 8, 4),
        head_hidden_size=4,
        max_epochs=2,
        patience=1,
        lr_scheduler_patience=1,
        batch_size=16,
        rf_estimators=5,
        separate_hidden_size=8,
        deep_balanced_widths=(12, 8, 8),
        hgb_max_iter=8,
    )
    settings.update(overrides)
    return family.ValidationConfig(**settings)


def synthetic_configurations(group_count: int = 12) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    horizons = (2020, 2050, 2100)
    for group in range(group_count):
        window = (2.9, 1.2, 0.8)[group % 3]
        floor = 0.41 + 0.3 * group
        wall = 0.45 + 0.2 * (group % 5)
        roof = 0.48 + 0.4 * (group % 4)
        infiltration = 0.025 - 0.0004 * group
        for horizon_index, horizon in enumerate(horizons):
            rows.append(
                {
                    "horizon": horizon,
                    "simulation_id": group + 1,
                    "scenario": f"package_{group + 1}",
                    "windows_U_Factor": window,
                    "groundfloor_thermal_resistance": floor,
                    "ext_walls_thermal_resistance": wall,
                    "roof_thermal_resistance": roof,
                    "infiltration_design_flow_m3_s": infiltration,
                    "annual_heating_energy_gj": (
                        30.0
                        - 2.5 * floor
                        - 1.2 * wall
                        - 0.7 * roof
                        + 2.0 * horizon_index
                        + 40.0 * infiltration
                    ),
                    "days_below_24_c": (
                        350.0 - 7.0 * horizon_index - 1.5 * group + 20.0 * infiltration
                    ),
                    "cost_rate_sum_proxy": 10.0 + group,
                    "carbon_rate_sum_proxy": 5.0 + (group % 6),
                }
            )
    return pd.DataFrame.from_records(rows)


class TestClipHelper(unittest.TestCase):
    def test_bounds_are_the_documented_physical_range(self):
        self.assertEqual(family.DECISION_DAYS_BOUNDS, (0.0, 365.0))

    def test_values_are_bounded_but_never_rounded(self):
        clipped = family.clip_predicted_days_for_decisions(
            [-12.4, -0.0001, 0.0, 12.7, 364.9999, 365.0, 410.6]
        )

        np.testing.assert_allclose(
            clipped, [0.0, 0.0, 0.0, 12.7, 364.9999, 365.0, 365.0]
        )

    def test_helper_accepts_arrays_and_returns_float_output(self):
        clipped = family.clip_predicted_days_for_decisions(
            np.array([400, 100, -5], dtype=int)
        )

        self.assertEqual(clipped.dtype, np.dtype(float))
        np.testing.assert_allclose(clipped, [365.0, 100.0, 0.0])


class TestClippingChangesTheDecision(unittest.TestCase):
    def test_clipping_removes_a_package_that_only_impossible_days_kept(self):
        # Unclipped, package 1 is non-dominated only because its predicted day
        # count exceeds the calendar. At 365 it ties package 2 on days and is
        # dominated on energy, so the shipped front must hold package 2 alone.
        oof = oof_frame(
            [
                {
                    "simulation_id": 1,
                    "cost": 5.0,
                    "carbon": 5.0,
                    "true_energy": 20.0,
                    "true_days": 300.0,
                    "predicted_energy": 20.0,
                    "predicted_days": 420.0,
                },
                {
                    "simulation_id": 2,
                    "cost": 5.0,
                    "carbon": 5.0,
                    "true_energy": 10.0,
                    "true_days": 300.0,
                    "predicted_energy": 10.0,
                    "predicted_days": 365.0,
                },
            ]
        )

        pareto, _ = family.compute_decision_validation(oof)

        unclipped_front = family.pareto_mask_minimization(
            np.column_stack(
                (
                    oof["predicted_annual_heating_energy_gj"].to_numpy(float),
                    oof["cost_rate_sum_proxy"].to_numpy(float),
                    oof["carbon_rate_sum_proxy"].to_numpy(float),
                    -oof["predicted_days_below_24_c"].to_numpy(float),
                )
            )
        )

        self.assertEqual(int(unclipped_front.sum()), 2)
        self.assertEqual(int(pareto.loc[0, "predicted_pareto_count"]), 1)
        self.assertEqual(int(pareto.loc[0, "decision_days_clipped_count"]), 1)
        self.assertEqual(int(pareto.loc[0, "raw_predicted_days_above_365_count"]), 1)

    def test_only_days_are_clipped_other_objectives_stay_unbounded(self):
        # Package 1's predicted heating energy is physically impossible
        # (negative). If the decision layer clipped it to zero, package 2
        # would dominate it on cost and GWP and the front would collapse to a
        # single package. It must stay on the front.
        oof = oof_frame(
            [
                {
                    "simulation_id": 1,
                    "cost": 10.0,
                    "carbon": 10.0,
                    "true_energy": 5.0,
                    "true_days": 100.0,
                    "predicted_energy": -5.0,
                    "predicted_days": 100.0,
                },
                {
                    "simulation_id": 2,
                    "cost": 1.0,
                    "carbon": 1.0,
                    "true_energy": 5.0,
                    "true_days": 100.0,
                    "predicted_energy": 0.0,
                    "predicted_days": 100.0,
                },
            ]
        )

        pareto, _ = family.compute_decision_validation(oof)

        self.assertEqual(int(pareto.loc[0, "predicted_pareto_count"]), 2)
        self.assertEqual(int(pareto.loc[0, "decision_days_clipped_count"]), 0)

    def test_raw_predictions_survive_the_decision_pass(self):
        oof = oof_frame(
            [
                {
                    "simulation_id": 1,
                    "cost": 5.0,
                    "carbon": 5.0,
                    "true_energy": 20.0,
                    "true_days": 300.0,
                    "predicted_energy": 20.0,
                    "predicted_days": -25.0,
                },
                {
                    "simulation_id": 2,
                    "cost": 4.0,
                    "carbon": 6.0,
                    "true_energy": 18.0,
                    "true_days": 310.0,
                    "predicted_energy": 18.0,
                    "predicted_days": 410.0,
                },
            ]
        )
        before = oof["predicted_days_below_24_c"].copy()

        pareto, _ = family.compute_decision_validation(oof)

        pd.testing.assert_series_equal(oof["predicted_days_below_24_c"], before)
        self.assertEqual(int(pareto.loc[0, "raw_predicted_days_below_zero_count"]), 1)
        self.assertEqual(int(pareto.loc[0, "raw_predicted_days_above_365_count"]), 1)
        self.assertEqual(int(pareto.loc[0, "decision_days_clipped_count"]), 2)

    def test_diagnostics_report_the_unbounded_prediction_range(self):
        oof = oof_frame(
            [
                {
                    "simulation_id": 1,
                    "cost": 5.0,
                    "carbon": 5.0,
                    "true_energy": 20.0,
                    "true_days": 300.0,
                    "predicted_energy": 20.0,
                    "predicted_days": -25.0,
                },
                {
                    "simulation_id": 2,
                    "cost": 4.0,
                    "carbon": 6.0,
                    "true_energy": 18.0,
                    "true_days": 310.0,
                    "predicted_energy": 18.0,
                    "predicted_days": 410.0,
                },
            ]
        )

        diagnostics = family.raw_days_prediction_diagnostics(oof)

        self.assertEqual(set(diagnostics), {"shared_mtl_nn"})
        entry = diagnostics["shared_mtl_nn"]
        self.assertEqual(entry["prediction_count"], 2)
        self.assertEqual(entry["count_below_0"], 1)
        self.assertEqual(entry["count_above_365"], 1)
        self.assertEqual(entry["minimum_raw_prediction"], -25.0)
        self.assertEqual(entry["maximum_raw_prediction"], 410.0)


class TestPredictedCostAndGwpRouting(unittest.TestCase):
    """The four-target design must actually use the predicted indices."""

    @staticmethod
    def _frame():
        return oof_frame(
            [
                {
                    "simulation_id": 1,
                    "cost": 1.0,
                    "carbon": 1.0,
                    "true_energy": 20.0,
                    "true_days": 300.0,
                    "predicted_energy": 20.0,
                    "predicted_days": 300.0,
                    "predicted_cost": 99.0,
                    "predicted_carbon": 99.0,
                },
                {
                    "simulation_id": 2,
                    "cost": 9.0,
                    "carbon": 9.0,
                    "true_energy": 20.0,
                    "true_days": 300.0,
                    "predicted_energy": 20.0,
                    "predicted_days": 300.0,
                    "predicted_cost": 2.0,
                    "predicted_carbon": 2.0,
                },
            ]
        )

    def test_exact_index_design_ignores_the_predicted_columns(self):
        pareto, _ = family.compute_decision_validation(
            self._frame(), use_predicted_cost_carbon=False
        )

        # Exact cost and GWP favour package 1, which is therefore the only
        # non-dominated package on the predicted front.
        self.assertEqual(int(pareto.loc[0, "predicted_pareto_count"]), 1)
        self.assertEqual(int(pareto.loc[0, "exact_pareto_count"]), 1)
        self.assertAlmostEqual(float(pareto.loc[0, "recall"]), 1.0)

    def test_all_predicted_design_follows_the_predicted_indices(self):
        pareto, _ = family.compute_decision_validation(
            self._frame(), use_predicted_cost_carbon=True
        )

        # Predicted cost and GWP favour package 2, so front recovery collapses.
        self.assertEqual(int(pareto.loc[0, "predicted_pareto_count"]), 1)
        self.assertEqual(int(pareto.loc[0, "true_positive_count"]), 0)
        self.assertAlmostEqual(float(pareto.loc[0, "precision"]), 0.0)
        self.assertAlmostEqual(float(pareto.loc[0, "recall"]), 0.0)

    def test_default_is_the_exact_index_design(self):
        default_pareto, _ = family.compute_decision_validation(self._frame())
        explicit_pareto, _ = family.compute_decision_validation(
            self._frame(), use_predicted_cost_carbon=False
        )

        pd.testing.assert_frame_equal(default_pareto, explicit_pareto)


class TestObjectiveConventions(unittest.TestCase):
    def test_days_enter_the_objective_matrix_negated(self):
        frame = pd.DataFrame(
            {
                "predicted_annual_heating_energy_gj": [12.0],
                "predicted_days_below_24_c": [300.0],
                "cost_rate_sum_proxy": [3.0],
                "carbon_rate_sum_proxy": [4.0],
            }
        )

        objectives = family._objective_matrix(
            frame,
            "predicted_annual_heating_energy_gj",
            "predicted_days_below_24_c",
        )

        np.testing.assert_allclose(objectives, [[12.0, 3.0, 4.0, -300.0]])

    def test_exact_ties_break_to_the_lowest_simulation_id(self):
        position = family._stable_minimum_position(
            np.array([1.0, 1.0]), np.array([7, 3])
        )

        self.assertEqual(position, 1)

    def test_near_ties_are_not_treated_as_ties(self):
        # atol=1e-12 with rtol=0: scores 1e-7 apart are genuinely different
        # and the lower score wins regardless of its simulation id.
        position = family._stable_minimum_position(
            np.array([1.0, 1.0 + 1.0e-7]), np.array([7, 3])
        )

        self.assertEqual(position, 0)


class TestPointMetricsUseRawPredictions(unittest.TestCase):
    def setUp(self):
        family._set_learned_targets("two")

    def test_fold_mae_matches_the_stored_unclipped_predictions(self):
        artifacts = family.run_validation(synthetic_configurations(), fast_config())
        oof = artifacts.oof_predictions
        stored_mae = artifacts.fold_metrics[
            artifacts.fold_metrics["metric"] == "mae"
        ]

        recomputed = []
        for target in family.TARGET_COLUMNS:
            frame = oof.assign(
                absolute_error=(
                    oof[f"predicted_{target}"] - oof[f"true_{target}"]
                ).abs()
            )
            grouped = (
                frame.groupby(["repeat_seed", "outer_fold", "model"], sort=True)[
                    "absolute_error"
                ]
                .mean()
                .reset_index()
            )
            grouped["target"] = target
            recomputed.append(grouped)
        recomputed_frame = pd.concat(recomputed, ignore_index=True)

        merged = stored_mae.merge(
            recomputed_frame,
            on=["repeat_seed", "outer_fold", "model", "target"],
            how="left",
            validate="one_to_one",
        )
        self.assertEqual(len(merged), len(stored_mae))
        self.assertFalse(merged["absolute_error"].isna().any())
        np.testing.assert_allclose(
            merged["value"].to_numpy(float),
            merged["absolute_error"].to_numpy(float),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_metadata_records_the_clipping_policy_for_every_model(self):
        artifacts = family.run_validation(synthetic_configurations(), fast_config())
        policy = artifacts.metadata["decision_validation"]["days_prediction_policy"]
        diagnostics = artifacts.metadata["decision_validation"][
            "raw_days_prediction_diagnostics_by_model"
        ]

        self.assertEqual(policy["lower_bound_days"], 0.0)
        self.assertEqual(policy["upper_bound_days"], 365.0)
        self.assertEqual(set(diagnostics), set(family.MODEL_NAMES))


if __name__ == "__main__":
    unittest.main()
