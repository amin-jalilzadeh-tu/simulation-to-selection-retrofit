import unittest

import numpy as np
import pandas as pd

from scripts.reproduce_exhaustive_analysis import (
    EXPECTED_MINIMUM_ENERGY_MINUS_BASELINE_DAYS,
    EXPECTED_THRESHOLD_BALANCED_IDS,
    EXPECTED_THRESHOLD_PARETO_COUNTS,
    HORIZONS,
    REPO_ROOT,
    calculate_infiltration_design_flow,
    extract_horizon,
    extract_heating_energy_daily,
    normalized_objectives,
    retrofit_state_specifications,
    temperature_threshold_pareto_summary,
)


class TestReproducibilityHelpers(unittest.TestCase):
    def test_infiltration_design_flow_matches_archived_formula(self):
        baseline = calculate_infiltration_design_flow(
            windows_U_Factor=2.90,
            groundfloor_thermal_resistance=0.41,
            ext_walls_thermal_resistance=0.45,
            roof_thermal_resistance=0.48,
        )
        all_state_4 = calculate_infiltration_design_flow(
            windows_U_Factor=0.81,
            groundfloor_thermal_resistance=5.60,
            ext_walls_thermal_resistance=6.70,
            roof_thermal_resistance=8.70,
        )

        self.assertAlmostEqual(baseline, 0.025)
        self.assertAlmostEqual(all_state_4, 0.0135)

    def test_direct_boiler_heating_output_is_used(self):
        dates = ["2020-01-01", "2020-01-02"]
        group = pd.DataFrame(
            {
                "Simulation ID": [1],
                "index": [
                    "BOILER:Boiler Heating Energy [J](Daily)",
                ],
                dates[0]: [11.0],
                dates[1]: [22.0],
            }
        )

        daily, source, row_count = extract_heating_energy_daily(group, dates)

        np.testing.assert_allclose(daily, [11.0, 22.0])
        self.assertEqual(source, "direct_boiler_heating_energy_output")
        self.assertEqual(row_count, 1)

    def test_retained_renamed_boiler_output_is_not_reinterpreted_as_fuel(self):
        dates = ["2020-01-01", "2020-01-02"]
        group = pd.DataFrame(
            {
                "Simulation ID": [1],
                "index": ["Gas Consumption [J](Daily)"],
                dates[0]: [11.0],
                dates[1]: [22.0],
            }
        )

        daily, source, row_count = extract_heating_energy_daily(group, dates)

        np.testing.assert_allclose(daily, [11.0, 22.0])
        self.assertEqual(
            source,
            "retained_renamed_boiler_heating_energy_output",
        )
        self.assertEqual(row_count, 1)

    def test_zero_span_normalization_is_safe(self):
        front = pd.DataFrame(
            {
                "annual_heating_energy_gj": [1.0, 2.0],
                "cost_rate_sum_proxy": [5.0, 5.0],
                "carbon_rate_sum_proxy": [7.0, 7.0],
                "days_below_24_c": [300, 300],
            }
        )

        normalized = normalized_objectives(front)

        self.assertTrue(np.isfinite(normalized).all())
        np.testing.assert_allclose(normalized[:, 1:], 0.0)

    def test_reconciled_state_table_records_generator_and_archive_values(self):
        states = retrofit_state_specifications()

        self.assertEqual(len(states), 20)
        floor_state_1 = states.loc[
            states["component"].eq("ground_floor") & states["state_id"].eq(1)
        ].iloc[0]
        self.assertAlmostEqual(floor_state_1["generator_value"], 4.0)
        self.assertAlmostEqual(floor_state_1["archive_metadata_value"], 4.8)

        facade_state_4 = states.loc[
            states["component"].eq("windowed_facade") & states["state_id"].eq(4)
        ].iloc[0]
        self.assertAlmostEqual(facade_state_4["generator_value"], 6.75)
        self.assertEqual(
            facade_state_4["scope_note"],
            "North/south windowed walls only",
        )

    def test_threshold_pareto_summary_matches_manuscript(self):
        configurations = pd.concat(
            [
                extract_horizon(
                    REPO_ROOT / "inputs" / f"{horizon}_merged_simulation_results.csv",
                    horizon,
                )
                for horizon in HORIZONS
            ],
            ignore_index=True,
        )
        summary = temperature_threshold_pareto_summary(configurations)

        observed_counts = {
            int(horizon): {
                int(row["upper_threshold_c"]): int(row["pareto_count"])
                for _, row in group.iterrows()
            }
            for horizon, group in summary.groupby("horizon", sort=True)
        }
        observed_deltas = {
            int(horizon): {
                int(row["upper_threshold_c"]): int(
                    row["minimum_energy_minus_baseline_days"]
                )
                for _, row in group.iterrows()
            }
            for horizon, group in summary.groupby("horizon", sort=True)
        }
        observed_balanced_ids = {
            int(horizon): {
                int(row["upper_threshold_c"]): int(
                    row["selected_equal_weight_simulation_id"]
                )
                for _, row in group.iterrows()
            }
            for horizon, group in summary.groupby("horizon", sort=True)
        }

        self.assertEqual(observed_counts, EXPECTED_THRESHOLD_PARETO_COUNTS)
        self.assertEqual(
            observed_deltas,
            EXPECTED_MINIMUM_ENERGY_MINUS_BASELINE_DAYS,
        )
        self.assertEqual(
            observed_balanced_ids,
            EXPECTED_THRESHOLD_BALANCED_IDS,
        )


if __name__ == "__main__":
    unittest.main()
