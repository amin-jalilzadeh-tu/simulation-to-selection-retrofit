"""Tests for the two task designs of scripts/validate_mtl_family.py.

The manuscript reports the same model family under two task designs, selected
with ``--learned-targets``:

* ``two``  - the exact-index design. Heating energy and days below 24 C are
  learned; the cost and product-stage GWP indices stay exact in the decision
  layer. This is the released results/surrogate_validation/ run.
* ``four`` - the all-predicted design. All four objectives are learned and the
  predicted decision fronts are built from predicted cost and GWP. This is the
  released results/surrogate_validation_family4/ run.

No released test exercised either switch. These tests pin the design switch,
the per-design model shapes, and the fact that the decision layer follows the
design rather than defaulting to exact indices.

Run from the extracted archive root:

    python -m unittest discover -s tests
"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from scripts import validate_mtl_family as family


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
    """A deliberately tiny protocol so the smoke runs stay seconds long."""
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


class TaskDesignTestCase(unittest.TestCase):
    """Always leave the module in the published two-target state."""

    def setUp(self):
        family._set_learned_targets("two")

    def tearDown(self):
        family._set_learned_targets("two")


class TestTaskDesignSwitch(TaskDesignTestCase):
    def test_module_default_is_the_exact_index_design(self):
        self.assertEqual(
            family.TARGET_COLUMNS,
            ("annual_heating_energy_gj", "days_below_24_c"),
        )
        self.assertEqual(family.TARGET_COLUMNS, family.TWO_TARGET_COLUMNS)

    def test_four_target_design_adds_the_cost_and_gwp_indices(self):
        family._set_learned_targets("four")

        self.assertEqual(
            family.TARGET_COLUMNS,
            (
                "annual_heating_energy_gj",
                "days_below_24_c",
                "cost_rate_sum_proxy",
                "carbon_rate_sum_proxy",
            ),
        )
        self.assertEqual(family.TARGET_COLUMNS, family.FOUR_TARGET_COLUMNS)

    def test_switching_back_restores_the_exact_index_design(self):
        family._set_learned_targets("four")
        family._set_learned_targets("two")

        self.assertEqual(family.TARGET_COLUMNS, family.TWO_TARGET_COLUMNS)

    def test_unknown_task_design_is_rejected(self):
        with self.assertRaises(ValueError):
            family._set_learned_targets("three")

        self.assertEqual(family.TARGET_COLUMNS, family.TWO_TARGET_COLUMNS)

    def test_every_learned_target_carries_a_reporting_unit(self):
        for target in family.FOUR_TARGET_COLUMNS:
            self.assertIn(target, family.TARGET_UNITS)

    def test_model_roster_is_the_nine_reported_formulations(self):
        self.assertEqual(len(family.MODEL_NAMES), 9)
        self.assertEqual(
            family.MODEL_NAMES[:3],
            ("shared_mtl_nn", "independent_stl_nn", "random_forest"),
        )
        self.assertIn("gradient_boosting", family.MODEL_NAMES)
        self.assertEqual(
            sum(name.endswith("_mgda") for name in family.MODEL_NAMES), 3
        )


class TestModelShapesFollowTheTaskDesign(TaskDesignTestCase):
    def test_shared_trunk_model_builds_one_head_per_learned_target(self):
        for design, expected_heads in (("two", 2), ("four", 4)):
            with self.subTest(design=design):
                family._set_learned_targets(design)
                model = family.SharedTwoHeadRegressor(6, (16, 8), 4)

                self.assertEqual(len(model.heads), expected_heads)
                self.assertEqual(model(torch.zeros(3, 6)).shape, (3, expected_heads))

    def test_separate_family_builds_one_branch_per_learned_target(self):
        for design, expected_heads in (("two", 2), ("four", 4)):
            with self.subTest(design=design):
                family._set_learned_targets(design)
                model = family.SeparateTwoHeadRegressor(6, 16)

                self.assertEqual(len(model.heads), expected_heads)
                self.assertEqual(model(torch.zeros(3, 6)).shape, (3, expected_heads))

    def test_separate_family_has_no_output_relu(self):
        # The thesis architecture ended each branch in a ReLU. The port drops
        # it because the protocol standardises targets, which take negative
        # values; a surviving ReLU would floor half the target range at zero.
        family._set_learned_targets("two")
        model = family.SeparateTwoHeadRegressor(6, 16)

        self.assertFalse(
            any(isinstance(module, nn.ReLU) for module in model.modules())
        )
        with torch.no_grad():
            for head in model.heads:
                nn.init.constant_(head[-1].weight, 0.0)
                nn.init.constant_(head[-1].bias, -1.5)
            output = model(torch.zeros(2, 6))
        self.assertTrue(bool((output < 0.0).all()))

    def test_deep_balanced_head_widths_follow_the_documented_target_roles(self):
        widths = (200, 100, 50)
        expected = {
            "annual_heating_energy_gj": (50, 25),
            "days_below_24_c": (25, 12),
            "cost_rate_sum_proxy": (100, 50),
            "carbon_rate_sum_proxy": (75, 50),
        }
        for design in ("two", "four"):
            with self.subTest(design=design):
                family._set_learned_targets(design)
                model = family.DeepBalancedTwoHeadRegressor(6, widths, 0.5)

                observed = [
                    (head[0].out_features, head[2].out_features)
                    for head in model.heads
                ]
                self.assertEqual(
                    observed,
                    [expected[target] for target in family.TARGET_COLUMNS],
                )

    def test_deep_balanced_trunk_carries_the_configured_dropout(self):
        family._set_learned_targets("two")
        model = family.DeepBalancedTwoHeadRegressor(6, (200, 100, 50), 0.1)

        dropout_rates = [
            module.p for module in model.shared if isinstance(module, nn.Dropout)
        ]
        self.assertEqual(dropout_rates, [0.1, 0.1])


class TestEndToEndTaskDesigns(TaskDesignTestCase):
    """Small but complete runs of both designs, artifacts included."""

    def _run(self, design):
        family._set_learned_targets(design)
        configurations = synthetic_configurations()
        artifacts = family.run_validation(configurations, fast_config())
        return configurations, artifacts

    def test_exact_index_design_keeps_cost_and_gwp_out_of_the_model(self):
        configurations, artifacts = self._run("two")
        metadata = artifacts.metadata

        self.assertEqual(metadata["data"]["targets"], list(family.TWO_TARGET_COLUMNS))
        self.assertEqual(
            metadata["decision_validation"]["learned_objectives"],
            list(family.TWO_TARGET_COLUMNS),
        )
        self.assertEqual(
            metadata["decision_validation"]["predicted_front_cost_carbon_source"],
            "exact rate-table values",
        )
        self.assertNotIn(
            "predicted_cost_rate_sum_proxy", artifacts.oof_predictions.columns
        )
        self.assertIn("cost_rate_sum_proxy", artifacts.oof_predictions.columns)
        self.assertEqual(
            len(artifacts.oof_predictions),
            len(configurations) * len(family.MODEL_NAMES),
        )

    def test_all_predicted_design_learns_and_uses_four_objectives(self):
        configurations, artifacts = self._run("four")
        metadata = artifacts.metadata

        self.assertEqual(metadata["data"]["targets"], list(family.FOUR_TARGET_COLUMNS))
        self.assertEqual(
            metadata["decision_validation"]["predicted_front_cost_carbon_source"],
            "model predictions",
        )
        for target in family.FOUR_TARGET_COLUMNS:
            self.assertIn(f"predicted_{target}", artifacts.oof_predictions.columns)
            self.assertIn(f"true_{target}", artifacts.oof_predictions.columns)
        self.assertEqual(
            len(artifacts.fold_metrics),
            3 * len(family.MODEL_NAMES) * 4 * 3,
        )
        self.assertEqual(
            len(artifacts.pareto_validation), len(family.MODEL_NAMES) * 3
        )
        self.assertEqual(
            len(artifacts.weighted_profile_validation),
            len(family.MODEL_NAMES) * 3 * len(family.STAKEHOLDER_PROFILES),
        )

    def test_written_artifacts_replay_the_design_they_were_produced_under(self):
        for design, use_predicted in (("two", False), ("four", True)):
            with self.subTest(design=design):
                _, artifacts = self._run(design)
                with tempfile.TemporaryDirectory() as temporary_directory:
                    output_dir = Path(temporary_directory)
                    family.write_artifacts(output_dir, artifacts)
                    metadata = json.loads(
                        (output_dir / "model_metadata.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    saved_oof = pd.read_csv(
                        output_dir / "oof_predictions.csv",
                        float_precision="round_trip",
                    )
                    saved_pareto = pd.read_csv(
                        output_dir / "surrogate_pareto_validation.csv",
                        float_precision="round_trip",
                    )
                    saved_selections = pd.read_csv(
                        output_dir / "weighted_profile_validation.csv",
                        float_precision="round_trip",
                    )

                replayed_pareto, replayed_selections = (
                    family.compute_decision_validation(
                        saved_oof, use_predicted_cost_carbon=use_predicted
                    )
                )

                self.assertEqual(
                    metadata["decision_validation"]["learned_objectives"],
                    list(family.TARGET_COLUMNS),
                )
                pd.testing.assert_frame_equal(
                    replayed_pareto,
                    saved_pareto,
                    check_dtype=False,
                    check_exact=True,
                )
                pd.testing.assert_frame_equal(
                    replayed_selections,
                    saved_selections,
                    check_dtype=False,
                    check_exact=True,
                )


if __name__ == "__main__":
    unittest.main()
