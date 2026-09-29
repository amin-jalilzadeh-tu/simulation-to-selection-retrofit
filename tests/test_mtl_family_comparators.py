"""Tests for the comparator models of scripts/validate_mtl_family.py.

Two things the released suite never checked:

* the gradient-boosted-tree comparator - it is one of the five formulations the
  manuscript reports, is fitted once per learned target, and must be seeded and
  deterministic like every other model in the protocol;
* the regression claim in the script's own docstring, that the three published
  models (shared MTL, matched single-task networks, random forest) run first
  with unchanged seeds and therefore reproduce
  scripts/validate_surrogate_publication.py exactly on identical folds.

Run from the extracted archive root:

    python -m unittest discover -s tests
"""

import contextlib
import io
import sys
import unittest

import numpy as np
import pandas as pd

from scripts import validate_mtl_family as family
from scripts import validate_surrogate_publication as published


def synthetic_configurations(group_count: int = 12) -> pd.DataFrame:
    """A small stand-in for the 625-package archive, same column contract."""
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


def one_fold_setup(config, module=family):
    """Build the first fold and its inner-training scalers, as run_validation does."""
    data = module.prepare_validation_dataset(synthetic_configurations())
    features = data[list(module.FEATURE_COLUMNS)].to_numpy(float)
    targets = data[list(module.TARGET_COLUMNS)].to_numpy(float)
    groups = data["configuration_group_id"].to_numpy(int)
    folds = module.build_repeated_grouped_folds(
        groups, config.seeds, config.outer_folds, config.validation_fraction
    )
    scalers = module.fit_fold_scalers(features, targets, folds[0].inner_train_indices)
    return (
        folds[0],
        scalers.features.transform(features),
        scalers.targets.transform(targets),
        scalers,
    )


class TestGradientBoostingComparator(unittest.TestCase):
    def setUp(self):
        family._set_learned_targets("two")

    def tearDown(self):
        family._set_learned_targets("two")

    def test_comparator_is_part_of_the_reported_roster(self):
        self.assertIn("gradient_boosting", family.MODEL_NAMES)

    def test_one_booster_is_fitted_per_learned_target(self):
        for design, target_count in (("two", 2), ("four", 4)):
            with self.subTest(design=design):
                family._set_learned_targets(design)
                config = fast_config()
                fold, x_scaled, y_scaled, scalers = one_fold_setup(config)

                predictions, records = family._run_fold_models(
                    x_scaled, y_scaled, fold, scalers, config
                )
                boosting_records = [
                    record
                    for record in records
                    if record["model"] == "gradient_boosting"
                ]

                self.assertEqual(len(boosting_records), target_count)
                self.assertEqual(
                    [record["target"] for record in boosting_records],
                    list(family.TARGET_COLUMNS),
                )
                self.assertEqual(
                    predictions["gradient_boosting"].shape,
                    (len(fold.outer_test_indices), target_count),
                )

    def test_each_booster_gets_its_own_seed(self):
        config = fast_config()
        fold, x_scaled, y_scaled, scalers = one_fold_setup(config)

        _, records = family._run_fold_models(
            x_scaled, y_scaled, fold, scalers, config
        )
        seeds = [
            record["training_seed"]
            for record in records
            if record["model"] == "gradient_boosting"
        ]

        self.assertEqual(len(seeds), len(set(seeds)))

    def test_configured_hyperparameters_reach_the_booster_records(self):
        config = fast_config(hgb_max_iter=13, hgb_learning_rate=0.07)
        fold, x_scaled, y_scaled, scalers = one_fold_setup(config)

        _, records = family._run_fold_models(
            x_scaled, y_scaled, fold, scalers, config
        )
        boosting_records = [
            record for record in records if record["model"] == "gradient_boosting"
        ]

        for record in boosting_records:
            self.assertEqual(record["max_iter"], 13)
            self.assertEqual(record["learning_rate"], 0.07)

    def test_repeated_runs_are_bitwise_identical(self):
        config = fast_config()
        fold, x_scaled, y_scaled, scalers = one_fold_setup(config)

        first, _ = family._run_fold_models(x_scaled, y_scaled, fold, scalers, config)
        second, _ = family._run_fold_models(x_scaled, y_scaled, fold, scalers, config)

        for model_name in family.MODEL_NAMES:
            np.testing.assert_array_equal(
                first[model_name], second[model_name], err_msg=model_name
            )

    def test_boosting_predictions_are_returned_in_original_target_units(self):
        # The boosters are fitted on standardised targets; the stored values
        # must come back through the fold's target scaler, like every other
        # model, otherwise the reported MAE would be in standard deviations.
        config = fast_config()
        fold, x_scaled, y_scaled, scalers = one_fold_setup(config)

        predictions, _ = family._run_fold_models(
            x_scaled, y_scaled, fold, scalers, config
        )
        energy = predictions["gradient_boosting"][:, 0]
        days = predictions["gradient_boosting"][:, 1]

        self.assertGreater(float(energy.min()), 5.0)
        self.assertGreater(float(days.min()), 100.0)


class TestPublishedThreeModelRegression(unittest.TestCase):
    """The family screen must not disturb the three published models."""

    def setUp(self):
        family._set_learned_targets("two")

    def test_published_model_names_are_unchanged_and_come_first(self):
        self.assertEqual(family.MODEL_NAMES[:3], published.MODEL_NAMES)

    def test_folds_are_identical_to_the_published_protocol(self):
        config = fast_config()
        data = family.prepare_validation_dataset(synthetic_configurations())
        groups = data["configuration_group_id"].to_numpy(int)

        family_folds = family.build_repeated_grouped_folds(
            groups, config.seeds, config.outer_folds, config.validation_fraction
        )
        published_folds = published.build_repeated_grouped_folds(
            groups, config.seeds, config.outer_folds, config.validation_fraction
        )

        self.assertEqual(len(family_folds), len(published_folds))
        for family_fold, published_fold in zip(family_folds, published_folds):
            self.assertEqual(family_fold.inner_split_seed, published_fold.inner_split_seed)
            for attribute in (
                "inner_train_indices",
                "inner_validation_indices",
                "outer_test_indices",
            ):
                np.testing.assert_array_equal(
                    getattr(family_fold, attribute),
                    getattr(published_fold, attribute),
                    err_msg=attribute,
                )

    def test_published_models_reproduce_bitwise_on_an_identical_fold(self):
        config = fast_config()
        fold, x_scaled, y_scaled, scalers = one_fold_setup(config)

        family_predictions, _ = family._run_fold_models(
            x_scaled, y_scaled, fold, scalers, config
        )
        published_predictions, _ = published._run_fold_models(
            x_scaled, y_scaled, fold, scalers, config
        )

        for model_name in published.MODEL_NAMES:
            np.testing.assert_array_equal(
                family_predictions[model_name],
                published_predictions[model_name],
                err_msg=model_name,
            )

    def test_family_screen_adds_six_formulations_beyond_the_published_three(self):
        config = fast_config()
        fold, x_scaled, y_scaled, scalers = one_fold_setup(config)

        predictions, _ = family._run_fold_models(
            x_scaled, y_scaled, fold, scalers, config
        )

        self.assertEqual(set(predictions), set(family.MODEL_NAMES))
        self.assertEqual(len(set(predictions) - set(published.MODEL_NAMES)), 6)


class TestCommandLineContract(unittest.TestCase):
    """Locks the invocation documented in README.md and README_S1.txt."""

    @staticmethod
    def _parse(argument_list):
        original = sys.argv
        sys.argv = ["validate_mtl_family.py", *argument_list]
        try:
            return family.parse_args()
        finally:
            sys.argv = original

    def test_task_design_flag_accepts_both_documented_values(self):
        self.assertEqual(self._parse([]).learned_targets, "two")
        self.assertEqual(
            self._parse(["--learned-targets", "four"]).learned_targets, "four"
        )

    def test_unknown_task_design_is_refused_at_the_command_line(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self._parse(["--learned-targets", "three"])

    def test_default_output_directory_is_the_released_two_target_directory(self):
        arguments = self._parse([])

        self.assertEqual(arguments.output_dir.name, "surrogate_validation")
        self.assertEqual(arguments.input_dir.name, "inputs")
        self.assertEqual(tuple(arguments.seeds), family.DEFAULT_SEEDS)

    def test_family_only_settings_are_not_command_line_flags(self):
        # README.md documents a short Python driver for the dropout diagnostic
        # precisely because these settings have no flag. If a flag is ever
        # added, that README section must be rewritten, so fail here.
        for flag, value in (
            ("--deep-balanced-dropout", "0.1"),
            ("--separate-hidden-size", "128"),
            ("--hgb-max-iter", "300"),
        ):
            with self.subTest(flag=flag):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        self._parse([flag, value])


if __name__ == "__main__":
    unittest.main()
