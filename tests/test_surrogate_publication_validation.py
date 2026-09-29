import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.validate_surrogate_publication import (
    MODEL_NAMES,
    TARGET_COLUMNS,
    ValidationConfig,
    build_repeated_grouped_folds,
    compute_decision_validation,
    fit_fold_scalers,
    prepare_validation_dataset,
    run_validation,
    write_artifacts,
)


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
            energy = (
                30.0
                - 2.5 * floor
                - 1.2 * wall
                - 0.7 * roof
                + 2.0 * horizon_index
                + 40.0 * infiltration
            )
            days = (
                350.0
                - 7.0 * horizon_index
                - 1.5 * group
                + 20.0 * infiltration
            )
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
                    "annual_heating_energy_gj": energy,
                    "days_below_24_c": days,
                    "cost_rate_sum_proxy": 10.0 + group,
                    "carbon_rate_sum_proxy": 5.0 + (group % 6),
                }
            )
    return pd.DataFrame.from_records(rows)


class TestGroupedValidationProtocol(unittest.TestCase):
    def test_outer_and_inner_splits_are_group_disjoint(self):
        data = prepare_validation_dataset(synthetic_configurations())
        groups = data["configuration_group_id"].to_numpy(int)
        folds = build_repeated_grouped_folds(
            groups,
            seeds=(7,),
            n_splits=3,
            validation_fraction=0.2,
        )

        test_row_counts = np.zeros(len(data), dtype=int)
        for fold in folds:
            train_groups = set(groups[fold.inner_train_indices])
            validation_groups = set(groups[fold.inner_validation_indices])
            test_groups = set(groups[fold.outer_test_indices])
            self.assertTrue(train_groups.isdisjoint(validation_groups))
            self.assertTrue(train_groups.isdisjoint(test_groups))
            self.assertTrue(validation_groups.isdisjoint(test_groups))
            test_row_counts[fold.outer_test_indices] += 1

        np.testing.assert_array_equal(test_row_counts, np.ones(len(data), dtype=int))

    def test_scalers_are_fitted_only_on_inner_training_rows(self):
        features = np.asarray([[0.0], [2.0], [1000.0], [2000.0]])
        targets = np.asarray([[10.0], [14.0], [5000.0], [9000.0]])

        scalers = fit_fold_scalers(features, targets, inner_train_indices=[0, 1])

        np.testing.assert_allclose(scalers.features.mean_, [1.0])
        np.testing.assert_allclose(scalers.targets.mean_, [12.0])
        self.assertEqual(int(scalers.features.n_samples_seen_), 2)
        self.assertEqual(int(scalers.targets.n_samples_seen_), 2)

    def test_small_end_to_end_smoke_run_writes_complete_artifacts(self):
        configurations = synthetic_configurations()
        config = ValidationConfig(
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
        )

        artifacts = run_validation(configurations, config)

        expected_oof_rows = len(configurations) * len(MODEL_NAMES)
        self.assertEqual(len(artifacts.oof_predictions), expected_oof_rows)
        self.assertEqual(
            len(artifacts.fold_metrics),
            config.outer_folds * len(MODEL_NAMES) * len(TARGET_COLUMNS) * 3,
        )
        self.assertEqual(
            len(artifacts.horizon_fold_metrics),
            config.outer_folds
            * len(MODEL_NAMES)
            * 3
            * len(TARGET_COLUMNS)
            * 3,
        )
        self.assertEqual(
            len(artifacts.horizon_metrics_summary),
            len(MODEL_NAMES) * 3 * len(TARGET_COLUMNS) * 3,
        )
        predicted_columns = [f"predicted_{target}" for target in TARGET_COLUMNS]
        self.assertFalse(
            artifacts.oof_predictions[predicted_columns].isna().any().any()
        )
        self.assertEqual(
            len(artifacts.pareto_validation),
            len(MODEL_NAMES) * 3,
        )
        self.assertEqual(
            len(artifacts.weighted_profile_validation),
            len(MODEL_NAMES) * 3 * 5,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            write_artifacts(output_dir, artifacts)
            expected_files = {
                "fold_metrics.csv",
                "metrics_summary.csv",
                "horizon_fold_metrics.csv",
                "horizon_metrics_summary.csv",
                "oof_predictions.csv",
                "split_assignments.csv",
                "surrogate_pareto_validation.csv",
                "weighted_profile_validation.csv",
                "model_metadata.json",
            }
            self.assertEqual(
                {path.name for path in output_dir.iterdir()}, expected_files
            )
            metadata = json.loads(
                (output_dir / "model_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["data"]["features"][-1],
                "infiltration_design_flow_m3_s",
            )
            self.assertEqual(metadata["execution"]["device"], "cpu")
            policy = metadata["decision_validation"]["days_prediction_policy"]
            self.assertEqual(policy["lower_bound_days"], 0.0)
            self.assertEqual(policy["upper_bound_days"], 365.0)
            diagnostics = metadata["decision_validation"][
                "raw_days_prediction_diagnostics_by_model"
            ]
            self.assertEqual(set(diagnostics), set(MODEL_NAMES))

            saved_oof = pd.read_csv(
                output_dir / "oof_predictions.csv", float_precision="round_trip"
            )
            recomputed_pareto, recomputed_selections = (
                compute_decision_validation(saved_oof)
            )
            saved_pareto = pd.read_csv(
                output_dir / "surrogate_pareto_validation.csv",
                float_precision="round_trip",
            )
            saved_selections = pd.read_csv(
                output_dir / "weighted_profile_validation.csv",
                float_precision="round_trip",
            )
            pd.testing.assert_frame_equal(
                recomputed_pareto,
                saved_pareto,
                check_dtype=False,
                check_exact=True,
            )
            pd.testing.assert_frame_equal(
                recomputed_selections,
                saved_selections,
                check_dtype=False,
                check_exact=True,
            )

    def test_decision_validation_clips_impossible_days_without_mutating_oof(self):
        oof = pd.DataFrame(
            {
                "repeat_seed": [11, 11, 11, 11],
                "model": ["shared_mtl_nn"] * 4,
                "horizon": [2020] * 4,
                "simulation_id": [1, 2, 3, 4],
                "configuration_group_id": [1, 2, 3, 4],
                "cost_rate_sum_proxy": [0.0, 1.0, 2.0, 3.0],
                "carbon_rate_sum_proxy": [3.0, 2.0, 1.0, 0.0],
                "true_annual_heating_energy_gj": [20.0, 18.0, 16.0, 14.0],
                "true_days_below_24_c": [300.0, 310.0, 320.0, 330.0],
                "predicted_annual_heating_energy_gj": [20.0, 18.0, 16.0, 14.0],
                "predicted_days_below_24_c": [-25.0, 310.0, 320.0, 410.0],
            }
        )
        raw_days = oof["predicted_days_below_24_c"].copy()

        pareto, selections = compute_decision_validation(oof)
        explicitly_clipped = oof.copy()
        explicitly_clipped["predicted_days_below_24_c"] = explicitly_clipped[
            "predicted_days_below_24_c"
        ].clip(0.0, 365.0)
        clipped_pareto, clipped_selections = compute_decision_validation(
            explicitly_clipped
        )

        pd.testing.assert_series_equal(
            oof["predicted_days_below_24_c"], raw_days
        )
        self.assertEqual(int(pareto.loc[0, "raw_predicted_days_below_zero_count"]), 1)
        self.assertEqual(int(pareto.loc[0, "raw_predicted_days_above_365_count"]), 1)
        self.assertEqual(int(pareto.loc[0, "decision_days_clipped_count"]), 2)
        decision_columns = [
            column
            for column in pareto.columns
            if column
            not in {
                "raw_predicted_days_below_zero_count",
                "raw_predicted_days_above_365_count",
                "decision_days_clipped_count",
            }
        ]
        pd.testing.assert_frame_equal(
            pareto[decision_columns],
            clipped_pareto[decision_columns],
            check_exact=True,
        )
        pd.testing.assert_frame_equal(
            selections,
            clipped_selections,
            check_exact=True,
        )


if __name__ == "__main__":
    unittest.main()
