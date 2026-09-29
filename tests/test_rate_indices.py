import unittest

from utils.carbon import calculate_gwp_rate_index, calculate_total_carbon
from utils.cost import calculate_cost_rate_index, calculate_total_cost


class TestComparativeRateIndices(unittest.TestCase):
    def test_baseline_indices_are_zero(self):
        design = (2.90, 0.41, 0.45, 0.48)
        self.assertEqual(calculate_cost_rate_index(*design), 0)
        self.assertEqual(calculate_gwp_rate_index(*design), 0)

    def test_all_state_four_matches_archived_analysis(self):
        design = (0.81, 5.6, 6.7, 8.7)
        self.assertAlmostEqual(calculate_cost_rate_index(*design), 1091.0)
        self.assertAlmostEqual(calculate_gwp_rate_index(*design), 146.18)

    def test_legacy_aliases_preserve_results(self):
        design = (1.20, 4.8, 4.2, 4.5)
        self.assertEqual(
            calculate_total_cost(*design),
            calculate_cost_rate_index(*design),
        )
        self.assertEqual(
            calculate_total_carbon(*design),
            calculate_gwp_rate_index(*design),
        )


if __name__ == "__main__":
    unittest.main()
