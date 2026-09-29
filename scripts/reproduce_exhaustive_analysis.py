#!/usr/bin/env python3
"""Reproduce the manuscript's exhaustive retrofit analysis from raw CSV files.

This script intentionally bypasses the neural-network surrogate and NSGA-II.
For each climate horizon, it:

1. extracts one annual record per simulated envelope configuration;
2. computes the archived heating-energy indicator and days below 24 degrees C;
3. applies the repository's current component-rate cost/carbon proxy functions;
4. identifies the exact four-objective Pareto front; and
5. selects stakeholder solutions with min-max-normalized weighted sums.

The merged CSV label ``Gas Consumption [J](Daily)`` is historical and
misleading. The final archive-generation workflow renamed the EnergyPlus
``BOILER:Boiler Heating Energy [J](Daily)`` output to this label. It is
therefore treated as a boiler heating-energy indicator, not as fuel or metered
gas consumption.

The current cost and carbon functions add component-specific per-m2 rates without
multiplying them by envelope areas. Outputs therefore call these quantities
"proxies"; they must not be described as whole-building totals.

No EnergyPlus simulation or neural-network training is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.carbon import calculate_gwp_rate_index  # noqa: E402
from utils.cost import calculate_cost_rate_index  # noqa: E402


HORIZONS = (2020, 2050, 2100)
DESIGN_COLUMNS = (
    "windows_U_Factor",
    "groundfloor_thermal_resistance",
    "ext_walls_thermal_resistance",
    "roof_thermal_resistance",
)
RETAINED_BOILER_OUTPUT_PATTERN = "Gas Consumption"
BOILER_HEATING_OUTPUT_PATTERN = "BOILER:Boiler Heating Energy"
TEMPERATURE_ROW_PATTERN = "Zone Mean Air Temperature"
TEMPERATURE_UPPER_THRESHOLDS_C = (23.0, 24.0, 25.0, 26.0)

INFILTRATION_FACTOR_MAPS = {
    "windows_U_Factor": {
        2.90: 1.0,
        1.20: 0.8,
        1.21: 0.7,
        0.80: 0.6,
        0.81: 0.5,
    },
    "groundfloor_thermal_resistance": {
        0.41: 1.0,
        4.80: 0.9,
        5.00: 0.8,
        5.50: 0.7,
        5.60: 0.7,
    },
    "ext_walls_thermal_resistance": {
        0.45: 1.0,
        4.20: 0.8,
        4.40: 0.6,
        6.50: 0.5,
        6.70: 0.5,
    },
    "roof_thermal_resistance": {
        0.48: 1.0,
        4.50: 0.9,
        4.70: 0.8,
        8.50: 0.7,
        8.70: 0.7,
    },
}
INFILTRATION_WEIGHTS = {
    "windows_U_Factor": 0.45,
    "ext_walls_thermal_resistance": 0.35,
    "roof_thermal_resistance": 0.10,
    "groundfloor_thermal_resistance": 0.10,
}
INFILTRATION_BASELINE_M3_S = 0.025
INFILTRATION_MIN_M3_S = 0.013
INFILTRATION_MAX_M3_S = 0.025

# The profiles and objective order reproduce Table 5 of the manuscript:
# heating-energy indicator, cost proxy, carbon proxy, and negative days below
# 24 degrees C.
STAKEHOLDER_PROFILES = OrderedDict(
    (
        ("cost_priority", (0.10, 0.70, 0.10, 0.10)),
        ("carbon_priority", (0.10, 0.10, 0.70, 0.10)),
        ("energy_priority", (0.70, 0.10, 0.10, 0.10)),
        ("days_below_24_priority", (0.10, 0.10, 0.10, 0.70)),
        ("balanced", (0.25, 0.25, 0.25, 0.25)),
    )
)

EXPECTED_PARETO_COUNTS = {2020: 242, 2050: 258, 2100: 267}
EXPECTED_THRESHOLD_PARETO_COUNTS = {
    2020: {23: 247, 24: 242, 25: 212, 26: 211},
    2050: {23: 277, 24: 258, 25: 262, 26: 254},
    2100: {23: 265, 24: 267, 25: 270, 26: 288},
}
EXPECTED_MINIMUM_ENERGY_MINUS_BASELINE_DAYS = {
    2020: {23: -3, 24: 1, 25: 1, 26: 0},
    2050: {23: -41, 24: -39, 25: -28, 26: -18},
    2100: {23: -56, 24: -57, 25: -47, 26: -35},
}
EXPECTED_THRESHOLD_BALANCED_IDS = {
    2020: {23: 15, 24: 25, 25: 63, 26: 63},
    2050: {23: 13, 24: 13, 25: 13, 26: 13},
    2100: {23: 13, 24: 13, 25: 13, 26: 13},
}
EXPECTED_TEMPERATURE_PROVENANCE_COUNTS = {
    "at_or_below_17_5_c": 1,
    "at_or_above_24_c": 83384,
}
EXPECTED_2020_CORRELATIONS = {
    ("annual_heating_energy_gj", "cost_rate_sum_proxy"): -0.7495949387073971,
    ("annual_heating_energy_gj", "carbon_rate_sum_proxy"): -0.6055144936306690,
    ("annual_heating_energy_gj", "days_below_24_c"): -0.33439520140558854,
    ("cost_rate_sum_proxy", "carbon_rate_sum_proxy"): 0.5475693957347283,
    ("cost_rate_sum_proxy", "days_below_24_c"): 0.21453053729692012,
    ("carbon_rate_sum_proxy", "days_below_24_c"): 0.12766535295687478,
}
EXPECTED_MCDM_IDS = {
    2020: {
        "cost_priority": 1,
        "carbon_priority": 3,
        "energy_priority": 550,
        "days_below_24_priority": 25,
        "balanced": 25,
    },
    2050: {
        "cost_priority": 1,
        "carbon_priority": 11,
        "energy_priority": 395,
        "days_below_24_priority": 15,
        "balanced": 13,
    },
    2100: {
        "cost_priority": 1,
        "carbon_priority": 1,
        "energy_priority": 250,
        "days_below_24_priority": 15,
        "balanced": 13,
    },
}


def retrofit_state_specifications() -> pd.DataFrame:
    """Return the reconciled state table used to interpret the reduced archive.

    The distributed CSV files retain several rounded resistance
    labels. ``generator_value`` reports thickness/conductivity from the
    executed generator; ``archive_metadata_value`` reports the value used to
    identify the same discrete state in the reduced CSV and rate-index code.
    """
    records = [
        # component, state, label, property, generator, archive, cost, GWP,
        # infiltration factor, density, specific heat, generator SHGC,
        # archive SHGC, scope note
        (
            "window",
            0,
            "Wooden double glazing",
            "U-factor",
            2.90,
            2.90,
            0,
            0,
            1.0,
            np.nan,
            np.nan,
            0.70,
            0.50,
            "All modeled windows",
        ),
        (
            "window",
            1,
            "HR++ double, plastic frame",
            "U-factor",
            1.20,
            1.20,
            184,
            70,
            0.8,
            np.nan,
            np.nan,
            0.70,
            0.50,
            "All modeled windows",
        ),
        (
            "window",
            2,
            "HR++ double, wooden frame",
            "U-factor",
            1.21,
            1.21,
            485,
            50,
            0.7,
            np.nan,
            np.nan,
            0.70,
            0.50,
            "All modeled windows",
        ),
        (
            "window",
            3,
            "Triple, plastic frame",
            "U-factor",
            0.80,
            0.80,
            295,
            150,
            0.6,
            np.nan,
            np.nan,
            0.70,
            0.50,
            "All modeled windows",
        ),
        (
            "window",
            4,
            "Triple, wooden frame",
            "U-factor",
            0.81,
            0.81,
            622,
            120,
            0.5,
            np.nan,
            np.nan,
            0.70,
            0.50,
            "All modeled windows",
        ),
        (
            "ground_floor",
            0,
            "Uninsulated timber",
            "R-value",
            0.41,
            0.41,
            0,
            0,
            1.0,
            370,
            1699,
            np.nan,
            np.nan,
            "Ground-floor material object",
        ),
        (
            "ground_floor",
            1,
            "PIR",
            "R-value",
            4.00,
            4.80,
            59.7,
            10.0,
            0.9,
            370,
            1699,
            np.nan,
            np.nan,
            "Ground-floor material object",
        ),
        (
            "ground_floor",
            2,
            "Hemp fibre",
            "R-value",
            5.00,
            5.00,
            77,
            5.92,
            0.8,
            370,
            1699,
            np.nan,
            np.nan,
            "Ground-floor material object",
        ),
        (
            "ground_floor",
            3,
            "Resol",
            "R-value",
            5.50,
            5.50,
            87.9,
            11.0,
            0.7,
            370,
            1699,
            np.nan,
            np.nan,
            "Ground-floor material object",
        ),
        (
            "ground_floor",
            4,
            "Hemp fibre",
            "R-value",
            5.67,
            5.60,
            108,
            7.0,
            0.7,
            370,
            1699,
            np.nan,
            np.nan,
            "Ground-floor material object",
        ),
        (
            "windowed_facade",
            0,
            "Solid clay brick",
            "R-value",
            0.45,
            0.45,
            0,
            0,
            1.0,
            1500,
            840,
            np.nan,
            np.nan,
            "North/south windowed walls only",
        ),
        (
            "windowed_facade",
            1,
            "EPS",
            "R-value",
            4.20,
            4.20,
            182,
            9.36,
            0.8,
            1500,
            840,
            np.nan,
            np.nan,
            "North/south windowed walls only",
        ),
        (
            "windowed_facade",
            2,
            "Hemp fibre",
            "R-value",
            4.40,
            4.40,
            179,
            4.83,
            0.6,
            1500,
            840,
            np.nan,
            np.nan,
            "North/south windowed walls only",
        ),
        (
            "windowed_facade",
            3,
            "EPS",
            "R-value",
            6.50,
            6.50,
            200,
            17.16,
            0.5,
            1500,
            840,
            np.nan,
            np.nan,
            "North/south windowed walls only",
        ),
        (
            "windowed_facade",
            4,
            "Hemp fibre",
            "R-value",
            6.75,
            6.70,
            222,
            8.50,
            0.5,
            1500,
            840,
            np.nan,
            np.nan,
            "North/south windowed walls only",
        ),
        (
            "roof",
            0,
            "Uninsulated tile",
            "R-value",
            0.49,
            0.48,
            0,
            0,
            1.0,
            437,
            1511,
            np.nan,
            np.nan,
            "Roof material object",
        ),
        (
            "roof",
            1,
            "Mineral wool",
            "R-value",
            4.50,
            4.50,
            89.5,
            23.29,
            0.9,
            437,
            1511,
            np.nan,
            np.nan,
            "Roof material object",
        ),
        (
            "roof",
            2,
            "Hemp fibre",
            "R-value",
            4.75,
            4.70,
            105,
            4.76,
            0.8,
            437,
            1511,
            np.nan,
            np.nan,
            "Roof material object",
        ),
        (
            "roof",
            3,
            "PIR",
            "R-value",
            8.50,
            8.50,
            101,
            18.50,
            0.7,
            437,
            1511,
            np.nan,
            np.nan,
            "Roof material object",
        ),
        (
            "roof",
            4,
            "Hemp fibre",
            "R-value",
            8.75,
            8.70,
            139,
            10.68,
            0.7,
            437,
            1511,
            np.nan,
            np.nan,
            "Roof material object",
        ),
    ]
    columns = [
        "component",
        "state_id",
        "source_option_label",
        "reported_property",
        "generator_value",
        "archive_metadata_value",
        "cost_rate_eur_per_m2",
        "gwp_a1_a3_kgco2e_per_m2",
        "infiltration_factor",
        "density_kg_per_m3",
        "specific_heat_j_per_kgk",
        "generator_shgc",
        "archive_metadata_shgc",
        "scope_note",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def _one_matching_row(group: pd.DataFrame, pattern: str) -> pd.Series:
    """Return the unique output row whose index label contains ``pattern``."""
    matches = group.loc[
        group["index"].astype(str).str.contains(pattern, regex=False, na=False)
    ]
    if len(matches) != 1:
        simulation_id = int(group["Simulation ID"].iloc[0])
        raise ValueError(
            f"Simulation {simulation_id}: expected one row containing "
            f"{pattern!r}, found {len(matches)}."
        )
    return matches.iloc[0]


def _matching_rows(group: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """Return output rows whose label contains ``pattern``."""
    return group.loc[
        group["index"].astype(str).str.contains(pattern, regex=False, na=False)
    ]


def extract_heating_energy_daily(
    group: pd.DataFrame, date_columns: list[str]
) -> tuple[np.ndarray, str, int]:
    """Extract the archived heating-energy indicator and its row provenance.

    Prefer the original boiler Heating Energy row when a less-processed archive
    contains it. The distributed merged CSVs retain that row under the
    historical ``Gas Consumption`` label.
    """
    boiler_matches = _matching_rows(group, BOILER_HEATING_OUTPUT_PATTERN)
    if len(boiler_matches) == 1:
        daily = pd.to_numeric(
            boiler_matches.iloc[0][date_columns], errors="raise"
        ).to_numpy(float)
        return daily, "direct_boiler_heating_energy_output", 1

    if len(boiler_matches) > 1:
        simulation_id = int(group["Simulation ID"].iloc[0])
        raise ValueError(
            f"Simulation {simulation_id}: expected one boiler Heating Energy "
            f"row, found {len(boiler_matches)}."
        )

    retained = _one_matching_row(group, RETAINED_BOILER_OUTPUT_PATTERN)
    daily = pd.to_numeric(retained[date_columns], errors="raise").to_numpy(float)
    return daily, "retained_renamed_boiler_heating_energy_output", 1


def _factor_for_design_value(column: str, value: float) -> float:
    """Return the archived infiltration factor for an exact design level."""
    mapping = INFILTRATION_FACTOR_MAPS[column]
    for design_value, factor in mapping.items():
        if np.isclose(value, design_value, rtol=0.0, atol=1.0e-9):
            return factor
    raise ValueError(
        f"Unsupported {column}={value!r}; expected one of {sorted(mapping)}."
    )


def calculate_infiltration_design_flow(**design: float) -> float:
    """Reproduce the IDF-generation infiltration design-flow calculation."""
    weighted_factor = sum(
        INFILTRATION_WEIGHTS[column]
        * _factor_for_design_value(column, float(design[column]))
        for column in INFILTRATION_WEIGHTS
    )
    return float(
        np.clip(
            INFILTRATION_BASELINE_M3_S * weighted_factor,
            INFILTRATION_MIN_M3_S,
            INFILTRATION_MAX_M3_S,
        )
    )


def extract_horizon(input_csv: Path, horizon: int) -> pd.DataFrame:
    """Convert a horizon's daily-output CSV into one row per configuration."""
    raw = pd.read_csv(input_csv)
    required = {"Simulation ID", "Scenario", "index", *DESIGN_COLUMNS}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"{input_csv}: missing required columns: {missing}")

    date_columns = [
        column for column in raw.columns if column.startswith(f"{horizon}-")
    ]
    if len(date_columns) != 365:
        raise ValueError(
            f"{input_csv}: expected 365 daily columns for {horizon}, "
            f"found {len(date_columns)}."
        )

    records: list[dict[str, object]] = []
    for simulation_id, group in raw.groupby("Simulation ID", sort=True):
        first = group.iloc[0]
        temperature = _one_matching_row(group, TEMPERATURE_ROW_PATTERN)

        design = {column: float(first[column]) for column in DESIGN_COLUMNS}
        (
            heating_energy_daily_j,
            heating_energy_source,
            heating_energy_source_row_count,
        ) = extract_heating_energy_daily(group, date_columns)
        temperature_daily_c = pd.to_numeric(
            temperature[date_columns], errors="raise"
        ).to_numpy(float)

        annual_heating_energy_gj = float(heating_energy_daily_j.sum() / 1.0e9)
        threshold_counts = {
            f"days_below_{int(threshold)}_c": int(
                np.count_nonzero(temperature_daily_c < threshold)
            )
            for threshold in TEMPERATURE_UPPER_THRESHOLDS_C
        }
        days_below_24_c = threshold_counts["days_below_24_c"]
        days_at_or_below_17_5_c = int(np.count_nonzero(temperature_daily_c <= 17.5))
        legacy_static_band_days = int(
            np.count_nonzero(
                (temperature_daily_c > 17.5) & (temperature_daily_c < 24.0)
            )
        )
        days_at_or_above_24_c = int(np.count_nonzero(temperature_daily_c >= 24.0))
        if (
            days_at_or_below_17_5_c + legacy_static_band_days + days_at_or_above_24_c
            != len(date_columns)
        ):
            raise AssertionError(
                f"Simulation {simulation_id}: temperature provenance counts "
                "do not partition the daily series."
            )
        if days_below_24_c != (days_at_or_below_17_5_c + legacy_static_band_days):
            raise AssertionError(
                f"Simulation {simulation_id}: days below 24 C do not match "
                "the provenance-count reconstruction."
            )

        cost_proxy = float(calculate_cost_rate_index(*design.values()))
        carbon_proxy = float(calculate_gwp_rate_index(*design.values()))
        infiltration_design_flow = calculate_infiltration_design_flow(**design)

        records.append(
            {
                "horizon": horizon,
                "simulation_id": int(simulation_id),
                "scenario": str(first["Scenario"]),
                **design,
                "infiltration_design_flow_m3_s": infiltration_design_flow,
                "annual_heating_energy_gj": annual_heating_energy_gj,
                "heating_energy_source": heating_energy_source,
                "heating_energy_source_row_count": heating_energy_source_row_count,
                "temperature_source_row": str(temperature["index"]),
                "cost_rate_sum_proxy": cost_proxy,
                "carbon_rate_sum_proxy": carbon_proxy,
                **threshold_counts,
                "days_at_or_below_17_5_c": days_at_or_below_17_5_c,
                "legacy_static_band_days": legacy_static_band_days,
                "days_at_or_above_24_c": days_at_or_above_24_c,
            }
        )

    configurations = pd.DataFrame.from_records(records).sort_values(
        "simulation_id", kind="stable"
    )
    if len(configurations) != 625:
        raise ValueError(
            f"{input_csv}: expected 625 configurations, found {len(configurations)}."
        )
    return configurations.reset_index(drop=True)


def pareto_mask_minimization(objectives: np.ndarray) -> np.ndarray:
    """Return a mask for non-dominated rows when every objective is minimized."""
    objectives = np.asarray(objectives, dtype=float)
    if objectives.ndim != 2:
        raise ValueError("objectives must be a two-dimensional array")

    non_dominated = np.ones(len(objectives), dtype=bool)
    for index, candidate in enumerate(objectives):
        dominated = np.all(objectives <= candidate, axis=1) & np.any(
            objectives < candidate, axis=1
        )
        non_dominated[index] = not np.any(dominated)
    return non_dominated


def get_pareto_front(
    configurations: pd.DataFrame,
    temperature_column: str = "days_below_24_c",
) -> pd.DataFrame:
    """Identify the exact four-objective Pareto front for one horizon."""
    if temperature_column not in configurations:
        raise KeyError(f"Missing temperature objective column: {temperature_column}")
    objectives = configurations[
        [
            "annual_heating_energy_gj",
            "cost_rate_sum_proxy",
            "carbon_rate_sum_proxy",
        ]
    ].to_numpy(float)
    objectives = np.column_stack(
        (objectives, -configurations[temperature_column].to_numpy(float))
    )
    front = configurations.loc[pareto_mask_minimization(objectives)].copy()
    return front.sort_values("simulation_id", kind="stable").reset_index(drop=True)


def normalized_objectives(
    front: pd.DataFrame,
    temperature_column: str = "days_below_24_c",
) -> np.ndarray:
    """Return min-max-normalized minimization objectives for weighted scoring."""
    if temperature_column not in front:
        raise KeyError(f"Missing temperature objective column: {temperature_column}")
    objectives = front[
        [
            "annual_heating_energy_gj",
            "cost_rate_sum_proxy",
            "carbon_rate_sum_proxy",
        ]
    ].to_numpy(float)
    objectives = np.column_stack(
        (objectives, -front[temperature_column].to_numpy(float))
    )
    minimum = objectives.min(axis=0)
    span = objectives.max(axis=0) - minimum
    return np.divide(
        objectives - minimum,
        span,
        out=np.zeros_like(objectives, dtype=float),
        where=span > 0,
    )


def select_mcdm_solutions(
    configurations: pd.DataFrame, front: pd.DataFrame
) -> pd.DataFrame:
    """Select one Pareto solution for each normalized weighted-sum profile."""
    normalized = normalized_objectives(front)
    baseline = configurations.loc[configurations["simulation_id"].eq(1)].iloc[0]
    rows: list[dict[str, object]] = []

    for profile, weights_tuple in STAKEHOLDER_PROFILES.items():
        weights = np.asarray(weights_tuple, dtype=float)
        scores = normalized @ weights
        # Stable deterministic tie break: lowest Simulation ID among exact minima.
        minimum_score = float(scores.min())
        candidate_positions = np.flatnonzero(
            np.isclose(scores, minimum_score, rtol=0.0, atol=1.0e-12)
        )
        selected_position = min(
            candidate_positions,
            key=lambda position: int(front.iloc[position]["simulation_id"]),
        )
        selected = front.iloc[selected_position]

        rows.append(
            {
                "horizon": int(selected["horizon"]),
                "profile": profile,
                "weight_energy": weights[0],
                "weight_cost": weights[1],
                "weight_carbon": weights[2],
                "weight_days_below_24": weights[3],
                "weighted_score": minimum_score,
                "simulation_id": int(selected["simulation_id"]),
                "scenario": selected["scenario"],
                **{column: selected[column] for column in DESIGN_COLUMNS},
                "infiltration_design_flow_m3_s": selected[
                    "infiltration_design_flow_m3_s"
                ],
                "annual_heating_energy_gj": selected["annual_heating_energy_gj"],
                "heating_energy_reduction_percent": (
                    100.0
                    * (
                        baseline["annual_heating_energy_gj"]
                        - selected["annual_heating_energy_gj"]
                    )
                    / baseline["annual_heating_energy_gj"]
                ),
                "cost_rate_sum_proxy": selected["cost_rate_sum_proxy"],
                "carbon_rate_sum_proxy": selected["carbon_rate_sum_proxy"],
                **{
                    f"days_below_{int(threshold)}_c": int(
                        selected[f"days_below_{int(threshold)}_c"]
                    )
                    for threshold in TEMPERATURE_UPPER_THRESHOLDS_C
                },
                "days_at_or_below_17_5_c": int(selected["days_at_or_below_17_5_c"]),
                "legacy_static_band_days": int(selected["legacy_static_band_days"]),
                "days_at_or_above_24_c": int(selected["days_at_or_above_24_c"]),
            }
        )
    return pd.DataFrame.from_records(rows)


def _reference_row(
    configurations: pd.DataFrame,
    selection: str,
    selected: pd.Series,
    baseline_energy: float,
    tie_count: int = 1,
) -> dict[str, object]:
    return {
        "horizon": int(selected["horizon"]),
        "selection": selection,
        "tie_count": tie_count,
        "simulation_id": int(selected["simulation_id"]),
        "scenario": selected["scenario"],
        **{column: selected[column] for column in DESIGN_COLUMNS},
        "infiltration_design_flow_m3_s": selected["infiltration_design_flow_m3_s"],
        "annual_heating_energy_gj": selected["annual_heating_energy_gj"],
        "heating_energy_reduction_percent": (
            100.0
            * (baseline_energy - selected["annual_heating_energy_gj"])
            / baseline_energy
        ),
        "cost_rate_sum_proxy": selected["cost_rate_sum_proxy"],
        "carbon_rate_sum_proxy": selected["carbon_rate_sum_proxy"],
        **{
            f"days_below_{int(threshold)}_c": int(
                selected[f"days_below_{int(threshold)}_c"]
            )
            for threshold in TEMPERATURE_UPPER_THRESHOLDS_C
        },
        "days_at_or_below_17_5_c": int(selected["days_at_or_below_17_5_c"]),
        "legacy_static_band_days": int(selected["legacy_static_band_days"]),
        "days_at_or_above_24_c": int(selected["days_at_or_above_24_c"]),
    }


def reference_solutions(configurations: pd.DataFrame) -> pd.DataFrame:
    """Return baseline, extrema, and the all-retrofit-state-4 reference package."""
    baseline = configurations.loc[configurations["simulation_id"].eq(1)].iloc[0]
    minimum_energy = configurations.loc[
        configurations["annual_heating_energy_gj"].idxmin()
    ]

    maximum_days_below_24 = int(configurations["days_below_24_c"].max())
    maximum_days_rows = configurations.loc[
        configurations["days_below_24_c"].eq(maximum_days_below_24)
    ].sort_values("simulation_id", kind="stable")
    maximum_days = maximum_days_rows.iloc[0]

    # Scenario ID 625 is retrofit state 4 for every component. This is kept
    # separate from the horizon-specific minimum-energy configuration because
    # they differ in 2100 (IDs 625 and 599, respectively).
    all_state_4 = configurations.loc[configurations["simulation_id"].eq(625)].iloc[0]

    rows = [
        _reference_row(
            configurations,
            "do_nothing_baseline",
            baseline,
            baseline["annual_heating_energy_gj"],
        ),
        _reference_row(
            configurations,
            "minimum_heating_energy",
            minimum_energy,
            baseline["annual_heating_energy_gj"],
        ),
        _reference_row(
            configurations,
            "maximum_days_below_24_first_id",
            maximum_days,
            baseline["annual_heating_energy_gj"],
            tie_count=len(maximum_days_rows),
        ),
        _reference_row(
            configurations,
            "all_components_retrofit_state_4",
            all_state_4,
            baseline["annual_heating_energy_gj"],
        ),
    ]
    return pd.DataFrame.from_records(rows)


def horizon_summary(
    configurations: pd.DataFrame, front: pd.DataFrame
) -> dict[str, object]:
    baseline = configurations.loc[configurations["simulation_id"].eq(1)].iloc[0]
    minimum_energy = configurations.loc[
        configurations["annual_heating_energy_gj"].idxmin()
    ]
    all_state_4 = configurations.loc[configurations["simulation_id"].eq(625)].iloc[0]
    maximum_days_below_24 = configurations["days_below_24_c"].max()
    return {
        "horizon": int(configurations["horizon"].iloc[0]),
        "configuration_count": len(configurations),
        "pareto_count": len(front),
        "days_below_24_min": int(configurations["days_below_24_c"].min()),
        "days_below_24_max": int(maximum_days_below_24),
        "maximum_days_below_24_tie_count": int(
            configurations["days_below_24_c"].eq(maximum_days_below_24).sum()
        ),
        "baseline_simulation_id": int(baseline["simulation_id"]),
        "baseline_annual_heating_energy_gj": baseline["annual_heating_energy_gj"],
        "baseline_days_below_24_c": int(baseline["days_below_24_c"]),
        "minimum_heating_energy_simulation_id": int(minimum_energy["simulation_id"]),
        "minimum_annual_heating_energy_gj": minimum_energy["annual_heating_energy_gj"],
        "minimum_heating_energy_reduction_percent": (
            100.0
            * (
                baseline["annual_heating_energy_gj"]
                - minimum_energy["annual_heating_energy_gj"]
            )
            / baseline["annual_heating_energy_gj"]
        ),
        "minimum_heating_energy_days_below_24_c": int(
            minimum_energy["days_below_24_c"]
        ),
        "all_state_4_simulation_id": int(all_state_4["simulation_id"]),
        "all_state_4_annual_heating_energy_gj": all_state_4["annual_heating_energy_gj"],
        "all_state_4_days_below_24_c": int(all_state_4["days_below_24_c"]),
    }


def create_below_24_figure(
    reference: pd.DataFrame, mcdm: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    """Plot the primary temperature-threshold indicator across horizons."""
    baseline = reference.loc[reference["selection"].eq("do_nothing_baseline")]
    minimum_energy = reference.loc[reference["selection"].eq("minimum_heating_energy")]
    balanced = mcdm.loc[mcdm["profile"].eq("balanced")]

    series = pd.concat(
        (
            baseline.assign(series="Do-nothing baseline"),
            balanced.assign(series="Horizon-specific equal-weight selection"),
            minimum_energy.assign(
                series="Horizon-specific minimum-indicator configuration"
            ),
        ),
        ignore_index=True,
        sort=False,
    )[["horizon", "series", "simulation_id", "days_below_24_c"]]
    series = series.sort_values(["series", "horizon"], kind="stable")
    horizon_labels = {
        2020: "Present",
        2050: "Mid-century",
        2100: "Late-century",
    }
    series["horizon_label"] = series["horizon"].map(horizon_labels)

    styles = {
        "Do-nothing baseline": {
            "color": "black",
            "marker": "o",
            "x_offset": -0.055,
            "legend_label": "Do-nothing baseline",
        },
        "Horizon-specific equal-weight selection": {
            "color": "#0072B2",
            "marker": "s",
            "x_offset": 0.0,
            "legend_label": "Equal-weight selection",
        },
        "Horizon-specific minimum-indicator configuration": {
            "color": "#D55E00",
            "marker": "^",
            "x_offset": 0.055,
            "legend_label": "Minimum-indicator configuration",
        },
    }
    horizon_positions = {horizon: position for position, horizon in enumerate(HORIZONS)}

    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    for label, style in styles.items():
        plotted = series.loc[series["series"].eq(label)].sort_values("horizon")
        x_values = (
            plotted["horizon"].map(horizon_positions).astype(float) + style["x_offset"]
        )
        ax.scatter(
            x_values,
            plotted["days_below_24_c"],
            label=style["legend_label"],
            s=70,
            color=style["color"],
            marker=style["marker"],
        )
        for x_value, (_, row) in zip(x_values, plotted.iterrows()):
            vertical_offset = {
                "Do-nothing baseline": 8,
                "Horizon-specific equal-weight selection": 12,
                "Horizon-specific minimum-indicator configuration": -16,
            }[label]
            ax.annotate(
                f"{int(row['days_below_24_c'])}",
                (x_value, row["days_below_24_c"]),
                xytext=(0, vertical_offset),
                textcoords="offset points",
                ha="center",
                va="center",
                color=style["color"],
                fontsize=10,
                fontweight="bold",
            )

    ax.set_xticks(
        list(horizon_positions.values()),
        labels=[horizon_labels[horizon] for horizon in HORIZONS],
    )
    ax.set_xlabel("Weather case")
    ax.set_ylabel("Days with selected-zone daily-mean T < 24 °C")
    ax.set_ylim(255, 371)
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="lower left", frameon=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    return series.reset_index(drop=True)


def temperature_threshold_sensitivity(
    configurations: pd.DataFrame,
) -> pd.DataFrame:
    """Return per-configuration counts for upper thresholds 23--26 C."""
    identity_columns = ["horizon", "simulation_id", "scenario"]
    threshold_columns = [
        f"days_below_{int(threshold)}_c" for threshold in TEMPERATURE_UPPER_THRESHOLDS_C
    ]
    sensitivity = configurations[[*identity_columns, *threshold_columns]].melt(
        id_vars=identity_columns,
        value_vars=threshold_columns,
        var_name="threshold_column",
        value_name="days_below_threshold",
    )
    threshold_lookup = {
        f"days_below_{int(threshold)}_c": threshold
        for threshold in TEMPERATURE_UPPER_THRESHOLDS_C
    }
    sensitivity["upper_threshold_c"] = sensitivity["threshold_column"].map(
        threshold_lookup
    )
    return (
        sensitivity[
            [
                "horizon",
                "simulation_id",
                "scenario",
                "upper_threshold_c",
                "days_below_threshold",
            ]
        ]
        .sort_values(["horizon", "simulation_id", "upper_threshold_c"], kind="stable")
        .reset_index(drop=True)
    )


def temperature_threshold_pareto_summary(
    configurations: pd.DataFrame,
) -> pd.DataFrame:
    """Reproduce the threshold-specific Pareto counts and reference deltas."""
    records: list[dict[str, object]] = []
    for horizon, horizon_data in configurations.groupby("horizon", sort=True):
        baseline = horizon_data.loc[horizon_data["simulation_id"].eq(1)].iloc[0]
        minimum_energy = horizon_data.loc[
            horizon_data["annual_heating_energy_gj"].idxmin()
        ]

        for threshold in TEMPERATURE_UPPER_THRESHOLDS_C:
            threshold_int = int(threshold)
            temperature_column = f"days_below_{threshold_int}_c"
            front = get_pareto_front(
                horizon_data,
                temperature_column=temperature_column,
            )
            normalized = normalized_objectives(
                front,
                temperature_column=temperature_column,
            )
            balanced_scores = normalized @ np.asarray(
                STAKEHOLDER_PROFILES["balanced"],
                dtype=float,
            )
            minimum_score = float(balanced_scores.min())
            candidate_positions = np.flatnonzero(
                np.isclose(
                    balanced_scores,
                    minimum_score,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            )
            selected_position = min(
                candidate_positions,
                key=lambda position: int(front.iloc[position]["simulation_id"]),
            )
            balanced = front.iloc[selected_position]
            baseline_days = int(baseline[temperature_column])
            minimum_energy_days = int(minimum_energy[temperature_column])
            records.append(
                {
                    "horizon": int(horizon),
                    "upper_threshold_c": threshold,
                    "pareto_count": len(front),
                    "baseline_simulation_id": int(baseline["simulation_id"]),
                    "baseline_days_below_threshold": baseline_days,
                    "minimum_heating_energy_simulation_id": int(
                        minimum_energy["simulation_id"]
                    ),
                    "minimum_heating_energy_days_below_threshold": (
                        minimum_energy_days
                    ),
                    "minimum_energy_minus_baseline_days": (
                        minimum_energy_days - baseline_days
                    ),
                    "selected_equal_weight_simulation_id": int(
                        balanced["simulation_id"]
                    ),
                    "selected_equal_weight_scenario": str(balanced["scenario"]),
                    "selected_equal_weight_days_below_threshold": int(
                        balanced[temperature_column]
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def create_threshold_sensitivity_figure(
    reference: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Plot threshold sensitivity for the three reported reference series."""
    rows: list[dict[str, object]] = []
    fixed_references = pd.concat(
        (
            reference.loc[reference["selection"].eq("do_nothing_baseline")].assign(
                series="Do-nothing baseline"
            ),
            reference.loc[reference["selection"].eq("minimum_heating_energy")].assign(
                series="Minimum heating-energy configuration"
            ),
        ),
        ignore_index=True,
        sort=False,
    )
    for _, selected_row in fixed_references.iterrows():
        for threshold in TEMPERATURE_UPPER_THRESHOLDS_C:
            rows.append(
                {
                    "horizon": int(selected_row["horizon"]),
                    "series": selected_row["series"],
                    "simulation_id": int(selected_row["simulation_id"]),
                    "upper_threshold_c": threshold,
                    "days_below_threshold": int(
                        selected_row[f"days_below_{int(threshold)}_c"]
                    ),
                }
            )
    for _, selected_row in threshold_summary.iterrows():
        rows.append(
            {
                "horizon": int(selected_row["horizon"]),
                "series": "Threshold-specific equal-weight selection",
                "simulation_id": int(
                    selected_row["selected_equal_weight_simulation_id"]
                ),
                "upper_threshold_c": float(selected_row["upper_threshold_c"]),
                "days_below_threshold": int(
                    selected_row["selected_equal_weight_days_below_threshold"]
                ),
            }
        )
    plotted = pd.DataFrame.from_records(rows)

    styles = {
        "Do-nothing baseline": {
            "color": "black",
            "marker": "o",
            "linestyle": "--",
        },
        "Threshold-specific equal-weight selection": {
            "color": "#0072B2",
            "marker": "s",
            "linestyle": "-",
        },
        "Minimum heating-energy configuration": {
            "color": "#D55E00",
            "marker": "^",
            "linestyle": "-",
        },
    }
    horizon_labels = {
        2020: "Present",
        2050: "Mid-century",
        2100: "Late-century",
    }

    fig, axes = plt.subplots(
        1, 3, figsize=(12.5, 4.6), sharex=True, constrained_layout=True
    )
    for ax, horizon in zip(axes, HORIZONS):
        horizon_data = plotted.loc[plotted["horizon"].eq(horizon)]
        for label, style in styles.items():
            series_data = horizon_data.loc[
                horizon_data["series"].eq(label)
            ].sort_values("upper_threshold_c")
            ax.plot(
                series_data["upper_threshold_c"],
                series_data["days_below_threshold"],
                label=label,
                linewidth=2.0,
                markersize=6.5,
                **style,
            )
        ax.set_title(horizon_labels[horizon])
        ax.set_xticks(TEMPERATURE_UPPER_THRESHOLDS_C)
        ax.set_xlabel("Upper threshold (°C)")
        ax.grid(True, linestyle=":", alpha=0.55)
    axes[0].set_ylabel("Days with selected-zone daily mean below threshold")
    axes[-1].legend(loc="lower right", frameon=True, fontsize=8.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return plotted.sort_values(
        ["horizon", "series", "upper_threshold_c"], kind="stable"
    ).reset_index(drop=True)


def create_correlation_figure(
    configurations: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    """Create the corrected 2020 Pearson-correlation heatmap."""
    data_2020 = configurations.loc[configurations["horizon"].eq(2020)].copy()
    columns = OrderedDict(
        (
            ("Annual heating-energy indicator", "annual_heating_energy_gj"),
            ("Component-rate cost index", "cost_rate_sum_proxy"),
            ("Product-stage GWP index", "carbon_rate_sum_proxy"),
            ("Days below 24 C", "days_below_24_c"),
        )
    )
    correlation = data_2020[list(columns.values())].corr(method="pearson")
    correlation.index = list(columns.keys())
    correlation.columns = list(columns.keys())

    display_labels = (
        "Annual heating-energy\nindicator",
        "Component-rate\ncost index",
        "Product-stage\nGWP index",
        "Days with daily-mean\nT < 24 °C",
    )
    values = correlation.to_numpy(float)

    fig, ax = plt.subplots(figsize=(7.2, 5.7), constrained_layout=True)
    boundaries = np.arange(len(display_labels) + 1) - 0.5
    image = ax.pcolormesh(
        boundaries,
        boundaries,
        values,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        shading="flat",
        edgecolors="white",
        linewidth=1.5,
    )
    ax.set_xlim(-0.5, len(display_labels) - 0.5)
    ax.set_ylim(len(display_labels) - 0.5, -0.5)
    ax.set_xticks(np.arange(len(display_labels)), labels=display_labels)
    ax.set_yticks(np.arange(len(display_labels)), labels=display_labels)
    ax.tick_params(axis="x", labelrotation=32, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white" if abs(value) >= 0.45 else "black",
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.052, pad=0.04)
    # Matplotlib rasterizes colorbar meshes with many colour levels by default.
    # Keep the submitted PDF fully vector so journal scaling does not blur it.
    colorbar.solids.set_rasterized(False)
    colorbar.set_label("Pearson correlation coefficient, r", fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return correlation


def create_pareto_mcdm_figure(
    configurations: pd.DataFrame,
    fronts: pd.DataFrame,
    mcdm: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create the corrected two-panel 2020 Pareto and weighted-sum figure."""
    all_2020 = configurations.loc[configurations["horizon"].eq(2020)].copy()
    front_2020 = fronts.loc[fronts["horizon"].eq(2020)].copy()
    selected_2020 = (
        mcdm.loc[mcdm["horizon"].eq(2020)].set_index("profile", drop=False).copy()
    )

    profile_styles = OrderedDict(
        (
            ("cost_priority", ("#E69F00", "Cost priority")),
            ("carbon_priority", ("#009E73", "GWP priority")),
            ("energy_priority", ("#D55E00", "Energy priority")),
            (
                "days_below_24_priority",
                ("#CC79A7", "Warm-side priority"),
            ),
            ("balanced", ("black", "Equal weight")),
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.25))
    x_column = "annual_heating_energy_gj"
    panel_definitions = (
        (
            axes[0],
            "days_below_24_c",
            "Days with selected-zone\ndaily-mean T < 24 °C",
        ),
        (
            axes[1],
            "cost_rate_sum_proxy",
            "Component-rate cost index\n(sum of component €/m² rates)",
        ),
    )

    for ax, y_column, y_label in panel_definitions:
        ax.scatter(
            all_2020[x_column],
            all_2020[y_column],
            s=18,
            color="#C9C9C9",
            edgecolors="none",
            alpha=0.85,
            zorder=1,
        )
        ax.scatter(
            front_2020[x_column],
            front_2020[y_column],
            s=26,
            color="#0072B2",
            edgecolors="white",
            linewidths=0.35,
            zorder=2,
        )

        # Equal-weight and warm-side priorities select the same 2020
        # configuration. Draw the larger balanced star first and the smaller
        # threshold-priority star on top at the exact shared coordinates.
        draw_order = (
            "cost_priority",
            "carbon_priority",
            "energy_priority",
            "balanced",
            "days_below_24_priority",
        )
        for profile in draw_order:
            color, _ = profile_styles[profile]
            selected = selected_2020.loc[profile]
            size = 330 if profile == "balanced" else 245
            if profile == "days_below_24_priority":
                size = 145
            ax.scatter(
                [selected[x_column]],
                [selected[y_column]],
                marker="*",
                s=size,
                color=color,
                edgecolors="black",
                linewidths=1.0,
                zorder=5 if profile == "days_below_24_priority" else 4,
            )

        ax.set_xlabel("Annual heating-energy indicator (GJ/year)", fontsize=15)
        ax.set_ylabel(y_label, fontsize=15)
        ax.grid(True, linestyle=":", color="#BDBDBD", alpha=0.7)
        ax.tick_params(labelsize=12.5)

    axes[0].set_xlim(16.5, 38.7)
    axes[0].set_ylim(358.7, 365.5)
    axes[0].set_yticks(np.arange(359, 366))
    axes[1].set_xlim(16.5, 38.7)
    axes[1].set_ylim(-55, 1125)

    annotation_offsets = {
        "cost_priority": (-10, 0),
        "carbon_priority": (12, 0),
        "energy_priority": (12, 0),
    }
    for ax, y_column, _ in panel_definitions:
        for profile in ("cost_priority", "carbon_priority", "energy_priority"):
            color, label = profile_styles[profile]
            selected = selected_2020.loc[profile]
            horizontal_offset, vertical_offset = annotation_offsets[profile]
            if profile == "cost_priority":
                horizontal_offset, vertical_offset = (-12, 8 if ax is axes[1] else 0)
            if ax is axes[0] and profile == "energy_priority":
                horizontal_offset, vertical_offset = (12, 12)
            ax.annotate(
                label,
                (selected[x_column], selected[y_column]),
                xytext=(horizontal_offset, vertical_offset),
                textcoords="offset points",
                ha="left" if horizontal_offset >= 0 else "right",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=color,
                zorder=6,
            )

        shared = selected_2020.loc["balanced"]
        ax.annotate(
            "Equal weight + warm-side",
            (shared[x_column], shared[y_column]),
            xytext=(12, -8 if ax is axes[0] else 6),
            textcoords="offset points",
            ha="left",
            va="top" if ax is axes[0] else "center",
            fontsize=11,
            fontweight="bold",
            color="black",
            zorder=6,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#C9C9C9",
            markeredgecolor="none",
            markersize=6,
            label="All 625 configurations",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#0072B2",
            markeredgecolor="white",
            markersize=7,
            label="4-objective Pareto front",
        ),
    ]
    for _, (color, label) in profile_styles.items():
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=12,
                label=f"Weighted-sum: {label}",
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        frameon=False,
        fontsize=11.5,
        columnspacing=1.5,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.98, bottom=0.245, wspace=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Return a repository-relative path when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return path.name


def validate_known_results(
    configurations: pd.DataFrame, summaries: pd.DataFrame, mcdm: pd.DataFrame
) -> None:
    observed_counts = summaries.set_index("horizon")["pareto_count"].to_dict()
    if observed_counts != EXPECTED_PARETO_COUNTS:
        raise AssertionError(
            f"Pareto counts changed: expected {EXPECTED_PARETO_COUNTS}, "
            f"observed {observed_counts}"
        )

    observed_ids = {
        int(horizon): group.set_index("profile")["simulation_id"].astype(int).to_dict()
        for horizon, group in mcdm.groupby("horizon", sort=True)
    }
    if observed_ids != EXPECTED_MCDM_IDS:
        raise AssertionError(
            f"MCDM selections changed: expected {EXPECTED_MCDM_IDS}, "
            f"observed {observed_ids}"
        )

    threshold_summary = temperature_threshold_pareto_summary(configurations)
    observed_threshold_counts = {
        int(horizon): {
            int(row["upper_threshold_c"]): int(row["pareto_count"])
            for _, row in group.iterrows()
        }
        for horizon, group in threshold_summary.groupby("horizon", sort=True)
    }
    if observed_threshold_counts != EXPECTED_THRESHOLD_PARETO_COUNTS:
        raise AssertionError(
            "Threshold Pareto counts changed: expected "
            f"{EXPECTED_THRESHOLD_PARETO_COUNTS}, "
            f"observed {observed_threshold_counts}"
        )

    observed_reference_deltas = {
        int(horizon): {
            int(row["upper_threshold_c"]): int(
                row["minimum_energy_minus_baseline_days"]
            )
            for _, row in group.iterrows()
        }
        for horizon, group in threshold_summary.groupby("horizon", sort=True)
    }
    if observed_reference_deltas != EXPECTED_MINIMUM_ENERGY_MINUS_BASELINE_DAYS:
        raise AssertionError(
            "Threshold reference deltas changed: expected "
            f"{EXPECTED_MINIMUM_ENERGY_MINUS_BASELINE_DAYS}, "
            f"observed {observed_reference_deltas}"
        )

    observed_balanced_ids = {
        int(horizon): {
            int(row["upper_threshold_c"]): int(
                row["selected_equal_weight_simulation_id"]
            )
            for _, row in group.iterrows()
        }
        for horizon, group in threshold_summary.groupby("horizon", sort=True)
    }
    if observed_balanced_ids != EXPECTED_THRESHOLD_BALANCED_IDS:
        raise AssertionError(
            "Threshold equal-weight selections changed: expected "
            f"{EXPECTED_THRESHOLD_BALANCED_IDS}, "
            f"observed {observed_balanced_ids}"
        )

    observed_provenance_counts = {
        "at_or_below_17_5_c": int(configurations["days_at_or_below_17_5_c"].sum()),
        "at_or_above_24_c": int(configurations["days_at_or_above_24_c"].sum()),
    }
    if observed_provenance_counts != EXPECTED_TEMPERATURE_PROVENANCE_COUNTS:
        raise AssertionError(
            "Temperature provenance counts changed: expected "
            f"{EXPECTED_TEMPERATURE_PROVENANCE_COUNTS}, "
            f"observed {observed_provenance_counts}"
        )

    data_2020 = configurations.loc[configurations["horizon"].eq(2020)]
    correlation = data_2020[
        [
            "annual_heating_energy_gj",
            "cost_rate_sum_proxy",
            "carbon_rate_sum_proxy",
            "days_below_24_c",
        ]
    ].corr()
    for (first, second), expected in EXPECTED_2020_CORRELATIONS.items():
        observed = float(correlation.loc[first, second])
        if not np.isclose(observed, expected, rtol=0.0, atol=1.0e-12):
            raise AssertionError(
                f"2020 correlation {first}/{second} changed: "
                f"expected {expected}, observed {observed}"
            )


def write_outputs(
    output_dir: Path,
    configurations: pd.DataFrame,
    fronts: pd.DataFrame,
    summaries: pd.DataFrame,
    references: pd.DataFrame,
    mcdm: pd.DataFrame,
    input_files: Iterable[Path],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    configurations.to_csv(output_dir / "all_configurations.csv", index=False)
    fronts.to_csv(output_dir / "pareto_fronts.csv", index=False)
    summaries.to_csv(output_dir / "horizon_summary.csv", index=False)
    references.to_csv(output_dir / "reference_solutions.csv", index=False)
    mcdm.to_csv(output_dir / "mcdm_weighted_sum_selections.csv", index=False)
    retrofit_state_specifications().to_csv(
        output_dir / "retrofit_state_specifications.csv", index=False
    )

    for legacy_name in (
        "comfort_horizons_corrected.png",
        "comfort_horizons_corrected.pdf",
        "comfort_horizon_series.csv",
    ):
        (output_dir / legacy_name).unlink(missing_ok=True)

    plotted = create_below_24_figure(
        references, mcdm, output_dir / "below_24_horizons.png"
    )
    plotted.to_csv(output_dir / "below_24_horizon_series.csv", index=False)

    sensitivity = temperature_threshold_sensitivity(configurations)
    sensitivity.to_csv(
        output_dir / "temperature_threshold_sensitivity.csv", index=False
    )
    sensitivity_summary = (
        sensitivity.groupby(["horizon", "upper_threshold_c"])["days_below_threshold"]
        .agg(
            minimum_days="min",
            mean_days="mean",
            maximum_days="max",
        )
        .reset_index()
    )
    sensitivity_summary.to_csv(
        output_dir / "temperature_threshold_horizon_summary.csv",
        index=False,
    )
    threshold_pareto_summary = temperature_threshold_pareto_summary(configurations)
    threshold_pareto_summary.to_csv(
        output_dir / "temperature_threshold_pareto_summary.csv",
        index=False,
    )
    reference_sensitivity = create_threshold_sensitivity_figure(
        references,
        threshold_pareto_summary,
        output_dir / "temperature_threshold_sensitivity.png",
    )
    reference_sensitivity.to_csv(
        output_dir / "temperature_threshold_reference_series.csv",
        index=False,
    )

    correlation = create_correlation_figure(
        configurations, output_dir / "correlation_2020_corrected.png"
    )
    correlation.to_csv(output_dir / "correlation_2020.csv")
    create_pareto_mcdm_figure(
        configurations,
        fronts,
        mcdm,
        output_dir / "pareto_mcdm_2020_corrected.png",
    )

    metadata = {
        "script": portable_path(Path(__file__)),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "input_sha256": {
            portable_path(path): sha256(path) for path in sorted(input_files)
        },
        "heating_energy_definition": (
            "Annual archived boiler heating-energy indicator: sum of the "
            "daily BOILER:Boiler Heating Energy output, divided by 1e9 "
            "(GJ/year). The final archive-generation workflow renamed this "
            "output to Gas Consumption [J](Daily) in the distributed merged "
            "CSV; it is not a fuel or metered-gas quantity."
        ),
        "heating_energy_source_modes": {
            str(mode): int(count)
            for mode, count in configurations["heating_energy_source"]
            .value_counts()
            .items()
        },
        "heating_energy_source_row_counts": {
            str(int(row_count)): int(count)
            for row_count, count in configurations["heating_energy_source_row_count"]
            .value_counts()
            .items()
        },
        "temperature_primary_definition": (
            "count of selected-zone daily mean air temperatures T < 24 C"
        ),
        "temperature_source_rows": {
            str(label): int(count)
            for label, count in configurations["temperature_source_row"]
            .value_counts()
            .items()
        },
        "temperature_threshold_sensitivity_c": list(TEMPERATURE_UPPER_THRESHOLDS_C),
        "temperature_provenance_counts": {
            "at_or_below_17_5_c": int(configurations["days_at_or_below_17_5_c"].sum()),
            "legacy_strict_17_5_to_24_band": int(
                configurations["legacy_static_band_days"].sum()
            ),
            "at_or_above_24_c": int(configurations["days_at_or_above_24_c"].sum()),
        },
        "infiltration_design_flow_definition": {
            "formula": (
                "clip(0.025 * (0.45*f_window + 0.35*f_wall + "
                "0.10*f_roof + 0.10*f_floor), 0.013, 0.025)"
            ),
            "units": "m3/s",
            "factor_maps": INFILTRATION_FACTOR_MAPS,
        },
        "cost_definition": "unweighted sum of current component-specific cost rates (proxy)",
        "carbon_definition": "unweighted sum of current component-specific A1-A3 rates (proxy)",
        "pareto_objectives": [
            "minimize annual_heating_energy_gj",
            "minimize cost_rate_sum_proxy",
            "minimize carbon_rate_sum_proxy",
            "maximize days_below_24_c",
        ],
        "mcdm_method": "min-max normalization on each horizon's Pareto front, then weighted-sum minimization",
        "stakeholder_profiles": STAKEHOLDER_PROFILES,
        "figure_definitions": {
            "correlation_2020_corrected": (
                "Pearson correlations across all 625 configurations in 2020"
            ),
            "pareto_mcdm_2020_corrected": (
                "All 2020 configurations, exact four-objective Pareto front, "
                "and min-max-normalized weighted-sum stakeholder selections"
            ),
            "below_24_horizons": (
                "Days with selected-zone daily-mean temperature below 24 C "
                "for the baseline, horizon-specific equal-weight selections, "
                "and horizon-specific minimum-indicator configurations"
            ),
            "temperature_threshold_sensitivity": (
                "Reference-solution sensitivity to upper thresholds "
                "23, 24, 25, and 26 C"
            ),
        },
        "temperature_figure_horizon_labels": {
            "2020_dataset": "Present",
            "2050_dataset": "Mid-century",
            "legacy_2100_dataset": "Late-century",
        },
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "inputs",
        help="Directory containing YYYY_merged_simulation_results.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "exhaustive_analysis",
        help="Directory for deterministic CSV, JSON, PNG, and PDF outputs.",
    )
    parser.add_argument(
        "--skip-known-checks",
        action="store_true",
        help="Do not assert the manuscript's known Pareto counts and MCDM IDs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configurations_by_horizon: list[pd.DataFrame] = []
    fronts_by_horizon: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    references_by_horizon: list[pd.DataFrame] = []
    mcdm_by_horizon: list[pd.DataFrame] = []
    input_files: list[Path] = []

    for horizon in HORIZONS:
        input_csv = args.input_dir / f"{horizon}_merged_simulation_results.csv"
        input_files.append(input_csv)
        configurations = extract_horizon(input_csv, horizon)
        front = get_pareto_front(configurations)
        mcdm = select_mcdm_solutions(configurations, front)

        configurations_by_horizon.append(configurations)
        fronts_by_horizon.append(front)
        summaries.append(horizon_summary(configurations, front))
        references_by_horizon.append(reference_solutions(configurations))
        mcdm_by_horizon.append(mcdm)

    all_configurations = pd.concat(configurations_by_horizon, ignore_index=True)
    all_fronts = pd.concat(fronts_by_horizon, ignore_index=True)
    summary_table = pd.DataFrame.from_records(summaries)
    all_references = pd.concat(references_by_horizon, ignore_index=True)
    all_mcdm = pd.concat(mcdm_by_horizon, ignore_index=True)

    if not args.skip_known_checks:
        validate_known_results(all_configurations, summary_table, all_mcdm)

    write_outputs(
        args.output_dir,
        all_configurations,
        all_fronts,
        summary_table,
        all_references,
        all_mcdm,
        input_files,
    )

    print(summary_table.to_string(index=False))
    print()
    print(
        all_mcdm[
            [
                "horizon",
                "profile",
                "simulation_id",
                "heating_energy_reduction_percent",
                "days_below_24_c",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote reproducibility outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
