"""Regression tests for the MCDA selection layer (supplementary_analysis).

Guards the manuscript Section 3.4 numbers:
  - the named-profile tie rule MUST use np.isclose(..., rtol=0.0, atol=1e-12);
    numpy's default relative tolerance treats genuinely unequal scores as tied
    and changes two achievement-scalarisation selections;
  - LP unique-selectability counts (from the released mcda_outputs.csv):
    uniquely selectable 53/51/65, not uniquely selectable 189/207/202,
    package 138 not uniquely selectable in every weather case;
  - achievement scalarisation agrees with the weighted sum in exactly 1 of the
    15 case-profile selections (mid-century D24-priority);
  - 10,000-draw sweep 90%-coverage counts 15/13/17 (weights only) and
    26/13/19 (joint weights x threshold).
"""
import csv
import os
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "results", "exhaustive_analysis")
SUPP = os.path.join(ROOT, "supplementary_analysis")

HORIZONS = ["2020", "2050", "2100"]
PROFILES = {
    "Cost": np.array([0.1, 0.7, 0.1, 0.1]),
    "GWP": np.array([0.1, 0.1, 0.7, 0.1]),
    "Energy": np.array([0.7, 0.1, 0.1, 0.1]),
    "D24": np.array([0.1, 0.1, 0.1, 0.7]),
    "Balanced": np.array([0.25, 0.25, 0.25, 0.25]),
}
RHO = 1e-6


def _front(hz, rows):
    sub = [r for r in rows if r["horizon"] == hz]
    ids = np.array([int(r["simulation_id"]) for r in sub])
    F = np.array(
        [
            [
                float(r["annual_heating_energy_gj"]),
                float(r["cost_rate_sum_proxy"]),
                float(r["carbon_rate_sum_proxy"]),
                -float(r["days_below_24_c"]),
            ]
            for r in sub
        ]
    )
    fmin, fmax = F.min(0), F.max(0)
    Fn = (F - fmin) / np.where(fmax > fmin, fmax - fmin, 1.0)
    return ids, Fn


def _select(scores, ids, rtol):
    best = np.min(scores)
    cand = np.where(np.isclose(scores, best, rtol=rtol, atol=1e-12))[0]
    return ids[cand[np.argmin(ids[cand])]]


class TestTieRule(unittest.TestCase):
    def test_default_rtol_manufactures_a_tie_and_flips_the_winner(self):
        # Two genuinely unequal scores 1e-7 apart: the numpy default relative
        # tolerance (1e-05) calls them tied and hands the win to the lower id.
        ids = np.array([5, 2])
        scores = np.array([1.0, 1.0 + 1e-7])
        self.assertEqual(_select(scores, ids, rtol=0.0), 5)
        self.assertEqual(_select(scores, ids, rtol=1e-05), 2)


class TestSelectionNumbers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(EX, "pareto_fronts.csv")) as f:
            cls.front_rows = list(csv.DictReader(f))
        rng = np.random.default_rng(42)
        cls.W = rng.dirichlet(np.ones(4), size=10000)

    def test_asf_agrees_with_weighted_sum_in_exactly_one_of_fifteen(self):
        agreements = {}
        for hz in HORIZONS:
            ids, Fn = _front(hz, self.front_rows)
            for name, w in PROFILES.items():
                ws = _select(Fn @ w, ids, rtol=0.0)
                asf = _select(np.max(w * Fn, axis=1) + RHO * (Fn @ w), ids, rtol=0.0)
                agreements[(hz, name)] = ws == asf
        agreeing = [k for k, v in agreements.items() if v]
        self.assertEqual(agreeing, [("2050", "D24")])
        self.assertEqual(sum(agreements.values()), 1)  # i.e. ASF changes 14 of 15

    def test_sweep_90pct_coverage_weights_only(self):
        expected = {"2020": 15, "2050": 13, "2100": 17}
        for hz in HORIZONS:
            ids, Fn = _front(hz, self.front_rows)
            order = np.argsort(ids)
            win = ids[order[np.argmin((self.W @ Fn.T)[:, order], axis=1)]]
            _, counts = np.unique(win, return_counts=True)
            freq = np.sort(counts)[::-1] / 100.0
            k90 = int(np.searchsorted(np.cumsum(freq), 90.0)) + 1
            self.assertEqual(k90, expected[hz], hz)

    def test_joint_weight_threshold_90pct_coverage(self):
        expected = {"2020": 26, "2050": 13, "2100": 19}
        with open(os.path.join(EX, "all_configurations.csv")) as f:
            rows = list(csv.DictReader(f))
        for hz in HORIZONS:
            sub = [r for r in rows if r["horizon"] == hz]
            self.assertEqual(len(sub), 625)
            ids = np.array([int(r["simulation_id"]) for r in sub])
            base = np.array(
                [
                    [float(r["annual_heating_energy_gj"]), float(r["cost_rate_sum_proxy"]),
                     float(r["carbon_rate_sum_proxy"])]
                    for r in sub
                ]
            )
            joint = {}
            for tu in (23, 24, 25, 26):
                D = np.array([float(r[f"days_below_{tu}_c"]) for r in sub])
                F = np.column_stack([base, -D])
                leq = (F[None, :, :] <= F[:, None, :]).all(axis=2)
                lt = (F[None, :, :] < F[:, None, :]).any(axis=2)
                P = ~(leq & lt).any(axis=1)
                Fp, idp = F[P], ids[P]
                fmin, fmax = Fp.min(0), Fp.max(0)
                Fn = (Fp - fmin) / np.where(fmax > fmin, fmax - fmin, 1.0)
                order = np.argsort(idp)
                win = idp[order[np.argmin((self.W @ Fn.T)[:, order], axis=1)]]
                for s, c in zip(*np.unique(win, return_counts=True)):
                    joint[s] = joint.get(s, 0) + c
            freq = np.sort(np.array(list(joint.values())))[::-1] / 400.0
            k90 = int(np.searchsorted(np.cumsum(freq), 90.0)) + 1
            self.assertEqual(k90, expected[hz], hz)


class TestS1GeneratorSmoke(unittest.TestCase):
    def test_base_generator_runs_and_carries_corrected_claims(self):
        import subprocess
        import sys
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "S1_generated.md")
            env = dict(os.environ, S1_OUT=out)
            proc = subprocess.run(
                [sys.executable, os.path.join(SUPP, "make_supplementary_tables.py")],
                cwd=SUPP, env=env, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            text = open(out, encoding="utf-8").read()
        for required in (
            "uniquely selectable when t\\* > 10⁻⁹",
            "14 of the 15 weather case–profile cases",
            "189 of 242", "207 of 258", "202 of 267",
            "Not uniquely selectable",
            "all 15 profile selections are unchanged",
            "moves 250 → 400",
            "Table S1.14", "Table S1.15", "Table S1.16", "Table S1.17",
            "Table S1.18",
        ):
            self.assertIn(required, text, required)
        for stale in (
            '"supportedness"', "boundary-supported", "as boundary when",
            "in all 15 horizon–profile cases", "hull boundary",
        ):
            self.assertNotIn(stale, text, stale)


class TestReleasedClassificationCsv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(SUPP, "mcda_outputs.csv")) as f:
            cls.rows = list(csv.DictReader(f))

    def test_uniquely_selectable_counts(self):
        expected = {"2020": (53, 189), "2050": (51, 207), "2100": (65, 202)}
        for hz, (uniq, not_uniq) in expected.items():
            sub = [r for r in self.rows if r["horizon"] == hz]
            self.assertEqual(
                sum(r["selectability"] == "uniquely_selectable" for r in sub), uniq, hz)
            self.assertEqual(
                sum(r["selectability"] == "not_uniquely_selectable" for r in sub), not_uniq, hz)
            self.assertEqual(len(sub), uniq + not_uniq, hz)

    def test_package_138_not_uniquely_selectable_everywhere(self):
        rows138 = [r for r in self.rows if r["simulation_id"] == "138"]
        self.assertEqual(len(rows138), 3)
        for r in rows138:
            self.assertEqual(r["selectability"], "not_uniquely_selectable", r["horizon"])

    def test_t_star_column_is_scientific_notation_floats(self):
        for r in self.rows:
            self.assertIn("e", r["t_star"])  # 1.234e-05 style
            float(r["t_star"])

    def test_profile_sensitivity_csv_matches_published_claims(self):
        with open(os.path.join(SUPP, "mcda_profile_sensitivity.csv")) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 15)
        agreeing = [r for r in rows if r["agreement"] == "True"]
        self.assertEqual([(r["horizon"], r["profile"]) for r in agreeing],
                         [("2050", "D24")])  # ASF changes 14 of 15
        self.assertTrue(all(r["asf_stable_over_rho_grid"] == "True" for r in rows))
        eq_asf = {r["horizon"]: r["asf_selection"] for r in rows
                  if r["profile"] == "Balanced"}
        self.assertEqual(set(eq_asf.values()), {"138"})


if __name__ == "__main__":
    unittest.main()
