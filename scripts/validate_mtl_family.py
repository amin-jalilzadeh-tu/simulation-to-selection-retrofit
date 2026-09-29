#!/usr/bin/env python3
"""ISOLATED EXTENSION of validate_surrogate_publication.py (MTL family screen).

Runs in a standalone copy of the inputs so the source repository is untouched.
On top of the published protocol (shared two-head MTL, matched single-task
networks, random forest), this adds on byte-identical folds:

* two-head ports of the thesis architecture families ``Separate`` (one shared
  layer, deep task branches) and ``Deep_Balanced`` (deep shared trunk,
  asymmetric task heads), keeping each family's sharing topology and widths
  while adopting the publication protocol's activation, initialization,
  optimizer, and standardized targets (the original Separate output ReLU is
  removed because standardized targets take negative values);
* an MGDA trainer (task-head losses are
  backpropagated, then the shared-trunk gradient is REPLACED by the min-norm
  combination, never accumulated on top of it); for two tasks the min-norm
  weights have an exact closed form, so no external QP solver is needed;
* a per-target gradient-boosted-trees baseline (HistGradientBoostingRegressor).

The three published models run first with unchanged seeds, so their outputs
must remain byte-identical to the released results (built-in regression check).

Inherited protocol (unchanged from the published script): all horizons of an
envelope package stay in the same split; each outer fold has a group-disjoint
inner early-stopping split; feature and target scalers are fitted on
inner-training rows only; every neural model runs deterministically on CPU.
Cost and carbon rate-sum proxies are deterministic functions of the envelope
package, so they are not learned and stay exact in the decision validation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reproduce_exhaustive_analysis import (  # noqa: E402
    DESIGN_COLUMNS,
    HORIZONS,
    STAKEHOLDER_PROFILES,
    extract_horizon,
    pareto_mask_minimization,
    portable_path,
    sha256,
)


FEATURE_COLUMNS = ("horizon", *DESIGN_COLUMNS, "infiltration_design_flow_m3_s")
TWO_TARGET_COLUMNS = ("annual_heating_energy_gj", "days_below_24_c")
FOUR_TARGET_COLUMNS = (
    "annual_heating_energy_gj",
    "days_below_24_c",
    "cost_rate_sum_proxy",
    "carbon_rate_sum_proxy",
)
TARGET_COLUMNS = TWO_TARGET_COLUMNS
DECISION_EXACT_COLUMNS = ("cost_rate_sum_proxy", "carbon_rate_sum_proxy")


def _set_learned_targets(mode: str) -> None:
    """Switch the module between two-target and four-target modes.

    In ``four`` mode the models also learn the cost and carbon rate indices
    (as the original four-task study did), and predicted decision fronts use
    the PREDICTED cost/carbon values; the exact reference never changes.
    """
    global TARGET_COLUMNS
    if mode == "two":
        TARGET_COLUMNS = TWO_TARGET_COLUMNS
    elif mode == "four":
        TARGET_COLUMNS = FOUR_TARGET_COLUMNS
    else:
        raise ValueError(f"Unknown learned-targets mode: {mode!r}")
MODEL_NAMES = (
    "shared_mtl_nn",
    "independent_stl_nn",
    "random_forest",
    "shared_mtl_nn_mgda",
    "separate_mtl_nn",
    "separate_mtl_nn_mgda",
    "deep_balanced_mtl_nn",
    "deep_balanced_mtl_nn_mgda",
    "gradient_boosting",
)
DEFAULT_SEEDS = (17, 29, 43)
DECISION_DAYS_BOUNDS = (0.0, 365.0)
TARGET_UNITS = {
    "annual_heating_energy_gj": "GJ/year",
    "days_below_24_c": "days/year",
    "cost_rate_sum_proxy": "rate-index points",
    "carbon_rate_sum_proxy": "rate-index points",
}


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for the repeated grouped validation protocol."""

    seeds: tuple[int, ...] = DEFAULT_SEEDS
    outer_folds: int = 5
    validation_fraction: float = 0.15
    trunk_widths: tuple[int, ...] = (128, 128, 64)
    head_hidden_size: int = 32
    max_epochs: int = 300
    patience: int = 30
    min_delta: float = 1.0e-6
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 10
    minimum_learning_rate: float = 1.0e-6
    batch_size: int = 128
    gradient_clip_norm: float = 5.0
    rf_estimators: int = 300
    rf_min_samples_leaf: int = 1
    rf_max_depth: int | None = None
    separate_hidden_size: int = 128
    deep_balanced_widths: tuple[int, int, int] = (200, 100, 50)
    deep_balanced_dropout: float = 0.5
    hgb_max_iter: int = 300
    hgb_learning_rate: float = 0.1

    def validate(self, group_count: int) -> None:
        if not self.seeds:
            raise ValueError("At least one repeat seed is required.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Repeat seeds must be unique.")
        if not 2 <= self.outer_folds <= group_count:
            raise ValueError(
                f"outer_folds must be between 2 and {group_count}, inclusive."
            )
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1.")
        if not self.trunk_widths or any(width <= 0 for width in self.trunk_widths):
            raise ValueError("trunk_widths must contain positive integers.")
        for name in (
            "head_hidden_size",
            "max_epochs",
            "patience",
            "batch_size",
            "rf_estimators",
            "rf_min_samples_leaf",
            "lr_scheduler_patience",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if not 0.0 < self.lr_scheduler_factor < 1.0:
            raise ValueError("lr_scheduler_factor must be between 0 and 1.")
        if self.minimum_learning_rate <= 0.0:
            raise ValueError("minimum_learning_rate must be positive.")
        if self.min_delta < 0.0 or self.weight_decay < 0.0:
            raise ValueError("min_delta and weight_decay cannot be negative.")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive.")


@dataclass(frozen=True)
class GroupedFold:
    repeat_seed: int
    outer_fold: int
    inner_split_seed: int
    inner_train_indices: np.ndarray
    inner_validation_indices: np.ndarray
    outer_test_indices: np.ndarray


@dataclass(frozen=True)
class FoldScalers:
    features: StandardScaler
    targets: StandardScaler


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    epochs_ran: int
    best_validation_loss: float


@dataclass
class ValidationArtifacts:
    fold_metrics: pd.DataFrame
    metrics_summary: pd.DataFrame
    horizon_fold_metrics: pd.DataFrame
    horizon_metrics_summary: pd.DataFrame
    oof_predictions: pd.DataFrame
    split_assignments: pd.DataFrame
    pareto_validation: pd.DataFrame
    weighted_profile_validation: pd.DataFrame
    metadata: dict[str, object]


def _hidden_stack(input_size: int, widths: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_width = input_size
    for width in widths:
        layers.extend((nn.Linear(previous_width, width), nn.SiLU()))
        previous_width = width
    return nn.Sequential(*layers)


class SharedTwoHeadRegressor(nn.Module):
    """Two-output MTL MLP with an equal-capacity linear head per target."""

    def __init__(
        self,
        input_size: int,
        trunk_widths: Sequence[int],
        head_hidden_size: int,
    ) -> None:
        super().__init__()
        self.shared = _hidden_stack(input_size, trunk_widths)
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(trunk_widths[-1], head_hidden_size),
                nn.SiLU(),
                nn.Linear(head_hidden_size, 1),
            )
            for _ in TARGET_COLUMNS
        )
        self.apply(_initialize_linear_layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        representation = self.shared(features)
        return torch.cat([head(representation) for head in self.heads], dim=1)


class SingleTaskRegressor(nn.Module):
    """Single-output MLP matched to one task path of the shared model."""

    def __init__(
        self,
        input_size: int,
        trunk_widths: Sequence[int],
        head_hidden_size: int,
    ) -> None:
        super().__init__()
        self.trunk = _hidden_stack(input_size, trunk_widths)
        self.head = nn.Sequential(
            nn.Linear(trunk_widths[-1], head_hidden_size),
            nn.SiLU(),
            nn.Linear(head_hidden_size, 1),
        )
        self.apply(_initialize_linear_layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(features))


class SeparateTwoHeadRegressor(nn.Module):
    """Two-head port of the thesis ``Separate`` family.

    Topology follows the source study's SeparateMTLModel: one shared
    Linear+activation layer, then a Linear-activation-Linear branch per task.
    The original branches ended in an output ReLU, which is removed here
    because the protocol standardizes targets (negative values are valid).
    Activation and initialization follow the publication protocol.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_size, hidden_size), nn.SiLU())
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1),
            )
            for _ in TARGET_COLUMNS
        )
        self.apply(_initialize_linear_layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        representation = self.shared(features)
        return torch.cat([head(representation) for head in self.heads], dim=1)


class DeepBalancedTwoHeadRegressor(nn.Module):
    """Two-head port of the thesis ``Deep_Balanced`` family.

    Keeps the source study's deep shared trunk (200-100 with dropout) and the
    asymmetric task heads: the energy-style head (100-50-25-1) serves heating
    energy and the comfort-style head (100-25-12-1) serves days below 24 C.
    Activation and initialization follow the publication protocol; the
    log-sigma parameter is omitted because uncertainty weighting is not used.
    """

    def __init__(
        self,
        input_size: int,
        widths: Sequence[int],
        dropout_rate: float,
    ) -> None:
        super().__init__()
        width1, width2, width3 = (int(width) for width in widths)
        self.shared = nn.Sequential(
            nn.Linear(input_size, width1),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(width1, width2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        # Head widths follow the source study per target: energy-style for
        # heating energy, comfort-style for days below 24 C, and (in
        # four-target mode) the original cost-style and emission-style heads.
        head_widths_by_target = {
            "annual_heating_energy_gj": (width3, width3 // 2),
            "days_below_24_c": (width3 // 2, width3 // 4),
            "cost_rate_sum_proxy": (width3 * 2, width3),
            "carbon_rate_sum_proxy": (int(width3 * 1.5), width3),
        }
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(width2, first_width),
                nn.SiLU(),
                nn.Linear(first_width, second_width),
                nn.SiLU(),
                nn.Linear(second_width, 1),
            )
            for first_width, second_width in (
                head_widths_by_target[target] for target in TARGET_COLUMNS
            )
        )
        self.apply(_initialize_linear_layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        representation = self.shared(features)
        return torch.cat([head(representation) for head in self.heads], dim=1)


def _initialize_linear_layers(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _derived_seed(*parts: int) -> int:
    normalized = [int(part) % (2**32) for part in parts]
    return int(
        np.random.SeedSequence(normalized).generate_state(1, dtype=np.uint32)[0]
    )


def set_deterministic_cpu(seed: int) -> None:
    """Set the random state used by one CPU-only training run."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def load_exact_dataset(
    input_dir: Path = REPO_ROOT / "inputs",
) -> tuple[pd.DataFrame, list[Path]]:
    """Load all exact archive rows using the exhaustive-analysis extractor."""
    frames: list[pd.DataFrame] = []
    input_files: list[Path] = []
    for horizon in HORIZONS:
        input_path = input_dir / f"{horizon}_merged_simulation_results.csv"
        input_files.append(input_path)
        frames.append(extract_horizon(input_path, horizon))
    configurations = pd.concat(frames, ignore_index=True)
    observed_horizons = tuple(sorted(configurations["horizon"].unique()))
    if observed_horizons != tuple(HORIZONS):
        raise AssertionError(
            f"Expected horizons {HORIZONS}, observed {observed_horizons}."
        )
    return configurations, input_files


def prepare_validation_dataset(configurations: pd.DataFrame) -> pd.DataFrame:
    """Validate, sort, and attach one stable group ID per envelope package."""
    required = {
        "horizon",
        "simulation_id",
        *DESIGN_COLUMNS,
        "infiltration_design_flow_m3_s",
        *TARGET_COLUMNS,
        *DECISION_EXACT_COLUMNS,
    }
    missing = sorted(required.difference(configurations.columns))
    if missing:
        raise ValueError(f"Validation data are missing columns: {missing}")

    data = configurations.copy()
    numeric_columns = [
        *FEATURE_COLUMNS,
        *TARGET_COLUMNS,
        *DECISION_EXACT_COLUMNS,
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if not np.isfinite(data[numeric_columns].to_numpy(float)).all():
        raise ValueError("Validation inputs and targets must all be finite.")
    if data.duplicated(["horizon", "simulation_id"]).any():
        raise ValueError("Each horizon/simulation_id pair must be unique.")

    data = data.sort_values(["horizon", "simulation_id"], kind="stable").reset_index(
        drop=True
    )
    data["configuration_group_id"] = (
        data.groupby(list(DESIGN_COLUMNS), sort=True, dropna=False).ngroup() + 1
    ).astype(int)
    if data.duplicated(["configuration_group_id", "horizon"]).any():
        raise ValueError(
            "An envelope-package group contains duplicate rows for a horizon."
        )
    data.insert(0, "dataset_row_id", np.arange(len(data), dtype=int))
    return data


def make_grouped_outer_folds(
    groups: Sequence[int] | np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return shuffled, deterministic group-exclusive outer folds."""
    groups_array = np.asarray(groups)
    unique_groups = np.unique(groups_array)
    if not 2 <= n_splits <= len(unique_groups):
        raise ValueError("n_splits must be between 2 and the number of groups.")
    shuffled = np.random.default_rng(seed).permutation(unique_groups)
    test_group_folds = np.array_split(shuffled, n_splits)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for test_groups in test_group_folds:
        test_mask = np.isin(groups_array, test_groups)
        folds.append((np.flatnonzero(~test_mask), np.flatnonzero(test_mask)))
    return folds


def make_group_disjoint_validation_split(
    outer_train_indices: np.ndarray,
    groups: Sequence[int] | np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split outer-training groups into inner train and validation rows."""
    groups_array = np.asarray(groups)
    outer_train_indices = np.asarray(outer_train_indices, dtype=int)
    outer_groups = np.unique(groups_array[outer_train_indices])
    if len(outer_groups) < 2:
        raise ValueError("At least two outer-training groups are required.")
    validation_group_count = int(round(len(outer_groups) * validation_fraction))
    validation_group_count = min(max(validation_group_count, 1), len(outer_groups) - 1)
    shuffled_groups = np.random.default_rng(seed).permutation(outer_groups)
    validation_groups = shuffled_groups[:validation_group_count]
    is_validation = np.isin(
        groups_array[outer_train_indices], validation_groups
    )
    return outer_train_indices[~is_validation], outer_train_indices[is_validation]


def build_repeated_grouped_folds(
    groups: Sequence[int] | np.ndarray,
    seeds: Sequence[int],
    n_splits: int,
    validation_fraction: float,
) -> list[GroupedFold]:
    """Build all outer folds and their group-disjoint early-stopping splits."""
    groups_array = np.asarray(groups)
    grouped_folds: list[GroupedFold] = []
    for repeat_seed in seeds:
        outer = make_grouped_outer_folds(groups_array, n_splits, repeat_seed)
        outer_test_counts = np.zeros(len(groups_array), dtype=int)
        for fold_index, (outer_train, outer_test) in enumerate(outer, start=1):
            inner_seed = _derived_seed(repeat_seed, fold_index, 101)
            inner_train, inner_validation = make_group_disjoint_validation_split(
                outer_train,
                groups_array,
                validation_fraction,
                inner_seed,
            )
            train_groups = set(groups_array[inner_train])
            validation_groups = set(groups_array[inner_validation])
            test_groups = set(groups_array[outer_test])
            if not train_groups.isdisjoint(validation_groups):
                raise AssertionError("Inner training and validation groups overlap.")
            if not train_groups.isdisjoint(test_groups):
                raise AssertionError("Inner training and outer test groups overlap.")
            if not validation_groups.isdisjoint(test_groups):
                raise AssertionError("Inner validation and outer test groups overlap.")
            if set(np.concatenate((inner_train, inner_validation))) != set(
                outer_train
            ):
                raise AssertionError("Inner splits do not partition outer training.")
            outer_test_counts[outer_test] += 1
            grouped_folds.append(
                GroupedFold(
                    repeat_seed=int(repeat_seed),
                    outer_fold=fold_index,
                    inner_split_seed=inner_seed,
                    inner_train_indices=inner_train,
                    inner_validation_indices=inner_validation,
                    outer_test_indices=outer_test,
                )
            )
        if not np.all(outer_test_counts == 1):
            raise AssertionError(
                "Each row must appear in exactly one outer test fold per repeat."
            )
    return grouped_folds


def fit_fold_scalers(
    features: np.ndarray,
    targets: np.ndarray,
    inner_train_indices: Sequence[int] | np.ndarray,
) -> FoldScalers:
    """Fit both scalers exclusively on the inner-training rows."""
    train_indices = np.asarray(inner_train_indices, dtype=int)
    if len(train_indices) == 0:
        raise ValueError("Cannot fit scalers on an empty training split.")
    return FoldScalers(
        features=StandardScaler().fit(features[train_indices]),
        targets=StandardScaler().fit(targets[train_indices]),
    )


def _standardized_mse(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Mean of per-task standardized MSEs (equal task weighting)."""
    return torch.mean(torch.mean(torch.square(prediction - target), dim=0))


def _two_task_min_norm_alpha(grad_a: torch.Tensor, grad_b: torch.Tensor) -> float:
    """Closed-form MGDA min-norm weight for exactly two tasks.

    Minimizes ``||alpha*g_a + (1-alpha)*g_b||^2`` over ``alpha`` in [0, 1].
    Equivalent to the quadratic program solved in the source study's
    ``train_mgda`` (min 0.5*a'GG'a, sum(a)=1, a>=0) for the two-task case,
    but exact and deterministic, so no solver-failure fallback is needed.
    """
    difference = grad_a - grad_b
    denominator = float(torch.dot(difference, difference).item())
    if denominator <= 1.0e-12:
        return 0.5
    alpha = float(torch.dot(grad_b - grad_a, grad_b).item()) / denominator
    return min(1.0, max(0.0, alpha))


def _min_norm_weights(per_task_gradients: list[torch.Tensor]) -> np.ndarray:
    """MGDA min-norm task weights on the simplex for any task count.

    Solves the same quadratic program as the source study's ``train_mgda``
    (min 0.5*a'GG'a, sum(a)=1, a>=0).  Two tasks use the exact closed form;
    more tasks use a deterministic Frank-Wolfe iteration on the Gram matrix,
    the standard MGDA solver, so no external QP dependency is needed.
    """
    task_count = len(per_task_gradients)
    if task_count == 2:
        alpha = _two_task_min_norm_alpha(
            per_task_gradients[0], per_task_gradients[1]
        )
        return np.array([alpha, 1.0 - alpha])
    stacked = torch.stack(per_task_gradients)
    gram = (stacked @ stacked.T).cpu().numpy().astype(np.float64)
    weights = np.full(task_count, 1.0 / task_count)
    for _ in range(250):
        descent_index = int(np.argmin(gram @ weights))
        direction = -weights
        direction[descent_index] += 1.0
        curvature = float(direction @ gram @ direction)
        if curvature <= 1.0e-12:
            break
        step = float(-(weights @ gram @ direction)) / curvature
        step = min(1.0, max(0.0, step))
        if step <= 1.0e-12:
            break
        weights = weights + step * direction
    return weights


def _mgda_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_features: torch.Tensor,
    batch_targets: torch.Tensor,
    gradient_clip_norm: float,
) -> torch.Tensor:
    """One MGDA batch update that replaces the shared-trunk gradient.

    Per-task gradients are taken on the shared trunk only; the min-norm
    combination REPLACES the shared-trunk gradient after the task-head losses
    are backpropagated (the historical implementation accumulated the summed
    gradient on top of the MGDA gradient, which this ordering repairs).
    Returns the equal-weight standardized MSE for logging so that training
    curves stay comparable across trainers.
    """
    optimizer.zero_grad(set_to_none=True)
    prediction = model(batch_features)
    task_losses = [
        torch.mean(torch.square(prediction[:, index] - batch_targets[:, index]))
        for index in range(prediction.shape[1])
    ]
    shared_parameters = [
        parameter
        for parameter in model.shared.parameters()
        if parameter.requires_grad
    ]
    per_task_gradients = []
    for loss in task_losses:
        gradients = torch.autograd.grad(
            loss, shared_parameters, retain_graph=True, allow_unused=False
        )
        per_task_gradients.append(
            torch.cat([gradient.reshape(-1) for gradient in gradients])
        )
    task_weights = _min_norm_weights(per_task_gradients)
    combined_gradient = sum(
        float(weight) * gradient
        for weight, gradient in zip(task_weights, per_task_gradients)
    )

    total_loss = sum(task_losses)
    total_loss.backward()
    index = 0
    for parameter in shared_parameters:
        parameter_size = parameter.numel()
        parameter.grad = (
            combined_gradient[index : index + parameter_size]
            .view_as(parameter)
            .clone()
        )
        index += parameter_size
    if index != combined_gradient.numel():
        raise ValueError("MGDA gradient does not match the shared parameter count.")
    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    return total_loss.detach() / len(task_losses)


def train_neural_regressor(
    model: nn.Module,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    config: ValidationConfig,
    seed: int,
    trainer: str = "equal_mse",
) -> TrainingResult:
    """Train one model with deterministic mini-batches and early stopping."""
    if trainer not in ("equal_mse", "mgda"):
        raise ValueError(f"Unknown trainer: {trainer!r}")
    set_deterministic_cpu(seed)
    device = torch.device("cpu")
    model.to(device)

    x_train = torch.as_tensor(train_features, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(train_targets, dtype=torch.float32, device=device)
    x_validation = torch.as_tensor(
        validation_features, dtype=torch.float32, device=device
    )
    y_validation = torch.as_tensor(
        validation_targets, dtype=torch.float32, device=device
    )
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    if y_validation.ndim == 1:
        y_validation = y_validation[:, None]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=config.weight_decay,
        amsgrad=False,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
        threshold=1.0e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=config.minimum_learning_rate,
    )
    batch_generator = np.random.default_rng(seed)
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    epochs_ran = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        permutation = batch_generator.permutation(len(x_train))
        for start in range(0, len(permutation), config.batch_size):
            batch_indices = torch.as_tensor(
                permutation[start : start + config.batch_size],
                dtype=torch.long,
                device=device,
            )
            if trainer == "mgda":
                _mgda_training_step(
                    model,
                    optimizer,
                    x_train.index_select(0, batch_indices),
                    y_train.index_select(0, batch_indices),
                    config.gradient_clip_norm,
                )
            else:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x_train.index_select(0, batch_indices))
                loss = _standardized_mse(
                    prediction, y_train.index_select(0, batch_indices)
                )
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm
                )
                optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = float(
                _standardized_mse(model(x_validation), y_validation).item()
            )
        scheduler.step(validation_loss)
        epochs_ran = epoch
        if validation_loss < best_validation_loss - config.min_delta:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    model.load_state_dict(best_state)
    return TrainingResult(
        best_epoch=best_epoch,
        epochs_ran=epochs_ran,
        best_validation_loss=best_validation_loss,
    )


@torch.no_grad()
def predict_neural_regressor(model: nn.Module, features: np.ndarray) -> np.ndarray:
    model.eval()
    prediction = model(torch.as_tensor(features, dtype=torch.float32))
    return prediction.detach().cpu().numpy()


def _parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _run_fold_models(
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    fold: GroupedFold,
    scalers: FoldScalers,
    config: ValidationConfig,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    train = fold.inner_train_indices
    validation = fold.inner_validation_indices
    test = fold.outer_test_indices
    input_size = x_scaled.shape[1]
    predictions: dict[str, np.ndarray] = {}
    training_metadata: list[dict[str, object]] = []

    mtl_seed = _derived_seed(fold.repeat_seed, fold.outer_fold, 201)
    set_deterministic_cpu(mtl_seed)
    mtl_model = SharedTwoHeadRegressor(
        input_size,
        config.trunk_widths,
        config.head_hidden_size,
    )
    mtl_result = train_neural_regressor(
        mtl_model,
        x_scaled[train],
        y_scaled[train],
        x_scaled[validation],
        y_scaled[validation],
        config,
        mtl_seed,
    )
    mtl_prediction_scaled = predict_neural_regressor(mtl_model, x_scaled[test])
    predictions["shared_mtl_nn"] = scalers.targets.inverse_transform(
        mtl_prediction_scaled
    )
    training_metadata.append(
        {
            "repeat_seed": fold.repeat_seed,
            "outer_fold": fold.outer_fold,
            "model": "shared_mtl_nn",
            "training_seed": mtl_seed,
            "target": "both",
            "parameter_count": _parameter_count(mtl_model),
            **asdict(mtl_result),
        }
    )

    single_task_predictions: list[np.ndarray] = []
    single_task_parameter_count = 0
    for target_index, target_name in enumerate(TARGET_COLUMNS):
        stl_seed = _derived_seed(
            fold.repeat_seed, fold.outer_fold, 301 + target_index
        )
        set_deterministic_cpu(stl_seed)
        stl_model = SingleTaskRegressor(
            input_size,
            config.trunk_widths,
            config.head_hidden_size,
        )
        stl_result = train_neural_regressor(
            stl_model,
            x_scaled[train],
            y_scaled[train, target_index],
            x_scaled[validation],
            y_scaled[validation, target_index],
            config,
            stl_seed,
        )
        prediction = predict_neural_regressor(stl_model, x_scaled[test])[:, 0]
        single_task_predictions.append(prediction)
        parameter_count = _parameter_count(stl_model)
        single_task_parameter_count += parameter_count
        training_metadata.append(
            {
                "repeat_seed": fold.repeat_seed,
                "outer_fold": fold.outer_fold,
                "model": "independent_stl_nn",
                "training_seed": stl_seed,
                "target": target_name,
                "parameter_count": parameter_count,
                **asdict(stl_result),
            }
        )
    stl_prediction_scaled = np.column_stack(single_task_predictions)
    predictions["independent_stl_nn"] = scalers.targets.inverse_transform(
        stl_prediction_scaled
    )
    training_metadata.append(
        {
            "repeat_seed": fold.repeat_seed,
            "outer_fold": fold.outer_fold,
            "model": "independent_stl_nn",
            "target": "combined_parameter_count",
            "parameter_count": single_task_parameter_count,
        }
    )

    forest_seed = _derived_seed(fold.repeat_seed, fold.outer_fold, 401)
    forest = RandomForestRegressor(
        n_estimators=config.rf_estimators,
        criterion="squared_error",
        min_samples_leaf=config.rf_min_samples_leaf,
        max_depth=config.rf_max_depth,
        max_features=1.0,
        bootstrap=True,
        random_state=forest_seed,
        n_jobs=1,
    )
    forest.fit(x_scaled[train], y_scaled[train])
    forest_prediction_scaled = forest.predict(x_scaled[test])
    predictions["random_forest"] = scalers.targets.inverse_transform(
        forest_prediction_scaled
    )
    training_metadata.append(
        {
            "repeat_seed": fold.repeat_seed,
            "outer_fold": fold.outer_fold,
            "model": "random_forest",
            "training_seed": forest_seed,
            "target": "both",
            "n_estimators": config.rf_estimators,
            "min_samples_leaf": config.rf_min_samples_leaf,
            "max_depth": config.rf_max_depth,
        }
    )

    # ---- isolated extension: thesis architecture families x trainers ----
    def _build_shared() -> nn.Module:
        return SharedTwoHeadRegressor(
            input_size, config.trunk_widths, config.head_hidden_size
        )

    def _build_separate() -> nn.Module:
        return SeparateTwoHeadRegressor(input_size, config.separate_hidden_size)

    def _build_deep_balanced() -> nn.Module:
        return DeepBalancedTwoHeadRegressor(
            input_size, config.deep_balanced_widths, config.deep_balanced_dropout
        )

    mtl_family_variants = (
        ("shared_mtl_nn_mgda", 202, _build_shared, "mgda"),
        ("separate_mtl_nn", 211, _build_separate, "equal_mse"),
        ("separate_mtl_nn_mgda", 212, _build_separate, "mgda"),
        ("deep_balanced_mtl_nn", 221, _build_deep_balanced, "equal_mse"),
        ("deep_balanced_mtl_nn_mgda", 222, _build_deep_balanced, "mgda"),
    )
    for model_name, seed_tag, build_model, trainer in mtl_family_variants:
        variant_seed = _derived_seed(fold.repeat_seed, fold.outer_fold, seed_tag)
        set_deterministic_cpu(variant_seed)
        variant_model = build_model()
        variant_result = train_neural_regressor(
            variant_model,
            x_scaled[train],
            y_scaled[train],
            x_scaled[validation],
            y_scaled[validation],
            config,
            variant_seed,
            trainer=trainer,
        )
        variant_prediction_scaled = predict_neural_regressor(
            variant_model, x_scaled[test]
        )
        predictions[model_name] = scalers.targets.inverse_transform(
            variant_prediction_scaled
        )
        training_metadata.append(
            {
                "repeat_seed": fold.repeat_seed,
                "outer_fold": fold.outer_fold,
                "model": model_name,
                "training_seed": variant_seed,
                "target": "both",
                "trainer": trainer,
                "parameter_count": _parameter_count(variant_model),
                **asdict(variant_result),
            }
        )

    # ---- isolated extension: per-target gradient-boosted trees ----
    boosted_predictions: list[np.ndarray] = []
    for target_index, target_name in enumerate(TARGET_COLUMNS):
        boosting_seed = _derived_seed(
            fold.repeat_seed, fold.outer_fold, 411 + target_index
        )
        booster = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=config.hgb_max_iter,
            learning_rate=config.hgb_learning_rate,
            early_stopping=False,
            random_state=boosting_seed,
        )
        booster.fit(x_scaled[train], y_scaled[train, target_index])
        boosted_predictions.append(booster.predict(x_scaled[test]))
        training_metadata.append(
            {
                "repeat_seed": fold.repeat_seed,
                "outer_fold": fold.outer_fold,
                "model": "gradient_boosting",
                "training_seed": boosting_seed,
                "target": target_name,
                "max_iter": config.hgb_max_iter,
                "learning_rate": config.hgb_learning_rate,
            }
        )
    predictions["gradient_boosting"] = scalers.targets.inverse_transform(
        np.column_stack(boosted_predictions)
    )

    return predictions, training_metadata


def _fold_metric_records(
    truth: np.ndarray,
    prediction: np.ndarray,
    repeat_seed: int,
    outer_fold: int,
    model_name: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for target_index, target_name in enumerate(TARGET_COLUMNS):
        observed = truth[:, target_index]
        estimated = prediction[:, target_index]
        values = {
            "mae": float(mean_absolute_error(observed, estimated)),
            "rmse": float(mean_squared_error(observed, estimated) ** 0.5),
            "r2": float(r2_score(observed, estimated)),
        }
        for metric, value in values.items():
            records.append(
                {
                    "repeat_seed": repeat_seed,
                    "outer_fold": outer_fold,
                    "model": model_name,
                    "target": target_name,
                    "metric": metric,
                    "value": value,
                    "unit": "dimensionless" if metric == "r2" else TARGET_UNITS[target_name],
                    "n_test_rows": len(observed),
                }
            )
    return records


def _split_assignment_records(
    data: pd.DataFrame, fold: GroupedFold
) -> pd.DataFrame:
    role = np.full(len(data), "inner_train", dtype=object)
    role[fold.inner_validation_indices] = "inner_validation"
    role[fold.outer_test_indices] = "outer_test"
    assignments = data[
        ["dataset_row_id", "configuration_group_id", "horizon", "simulation_id"]
    ].copy()
    assignments.insert(0, "outer_fold", fold.outer_fold)
    assignments.insert(0, "repeat_seed", fold.repeat_seed)
    assignments["inner_split_seed"] = fold.inner_split_seed
    assignments["role"] = role
    return assignments


def _oof_prediction_frame(
    data: pd.DataFrame,
    fold: GroupedFold,
    model_name: str,
    prediction: np.ndarray,
) -> pd.DataFrame:
    test_rows = data.iloc[fold.outer_test_indices]
    columns = [
        "dataset_row_id",
        "configuration_group_id",
        "horizon",
        "simulation_id",
        *DESIGN_COLUMNS,
        *DECISION_EXACT_COLUMNS,
    ]
    output = test_rows[columns].copy()
    output.insert(0, "model", model_name)
    output.insert(0, "outer_fold", fold.outer_fold)
    output.insert(0, "repeat_seed", fold.repeat_seed)
    for target_index, target_name in enumerate(TARGET_COLUMNS):
        truth = test_rows[target_name].to_numpy(float)
        output[f"true_{target_name}"] = truth
        output[f"predicted_{target_name}"] = prediction[:, target_index]
        output[f"residual_{target_name}"] = prediction[:, target_index] - truth
    return output


def _objective_matrix(
    frame: pd.DataFrame,
    energy_column: str,
    days_column: str,
    cost_column: str = "cost_rate_sum_proxy",
    carbon_column: str = "carbon_rate_sum_proxy",
) -> np.ndarray:
    return np.column_stack(
        (
            frame[energy_column].to_numpy(float),
            frame[cost_column].to_numpy(float),
            frame[carbon_column].to_numpy(float),
            -frame[days_column].to_numpy(float),
        )
    )


def clip_predicted_days_for_decisions(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Apply only the physical D24 bounds used in decision post-processing."""
    lower_bound, upper_bound = DECISION_DAYS_BOUNDS
    return np.clip(np.asarray(values, dtype=float), lower_bound, upper_bound)


def _minmax_transform(
    objectives: np.ndarray,
    minimum: np.ndarray | None = None,
    span: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if minimum is None:
        minimum = objectives.min(axis=0)
    if span is None:
        span = objectives.max(axis=0) - minimum
    normalized = np.divide(
        objectives - minimum,
        span,
        out=np.zeros_like(objectives, dtype=float),
        where=span > 0.0,
    )
    return normalized, minimum, span


def _stable_minimum_position(scores: np.ndarray, simulation_ids: np.ndarray) -> int:
    minimum_score = float(scores.min())
    candidate_positions = np.flatnonzero(
        np.isclose(scores, minimum_score, rtol=0.0, atol=1.0e-12)
    )
    return int(
        min(candidate_positions, key=lambda position: int(simulation_ids[position]))
    )


def compute_decision_validation(
    oof_predictions: pd.DataFrame,
    use_predicted_cost_carbon: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare OOF surrogate decisions with exact decisions per horizon.

    The exact reference always uses the exact cost/carbon rate indices.  With
    ``use_predicted_cost_carbon`` (four-target mode) the PREDICTED fronts and
    selections are built from the models' predicted cost/carbon values, so
    prediction error in all four objectives propagates into the decision.
    """
    pareto_records: list[dict[str, object]] = []
    selection_records: list[dict[str, object]] = []
    grouped = oof_predictions.groupby(
        ["repeat_seed", "model", "horizon"], sort=True
    )
    for (repeat_seed, model_name, horizon), frame in grouped:
        frame = frame.sort_values("simulation_id", kind="stable").reset_index(drop=True)
        if frame["simulation_id"].duplicated().any():
            raise AssertionError("OOF predictions contain duplicate configurations.")
        exact_objectives = _objective_matrix(
            frame,
            "true_annual_heating_energy_gj",
            "true_days_below_24_c",
        )
        raw_predicted_days = frame["predicted_days_below_24_c"].to_numpy(float)
        decision_predicted_days = clip_predicted_days_for_decisions(
            raw_predicted_days
        )
        decision_frame = frame.assign(
            decision_predicted_days_below_24_c=decision_predicted_days
        )
        if use_predicted_cost_carbon:
            # Both indices are sums of non-negative component rates, so a negative
            # prediction is outside the attainable range.  Left unclipped it becomes
            # the ideal of the min-max normalisation and shifts the whole axis.
            raw_cost = decision_frame["predicted_cost_rate_sum_proxy"].to_numpy(float)
            raw_carbon = decision_frame["predicted_carbon_rate_sum_proxy"].to_numpy(float)
            index_below_zero = int(
                np.count_nonzero(raw_cost < 0.0) + np.count_nonzero(raw_carbon < 0.0))
            decision_frame = decision_frame.assign(
                decision_predicted_cost_rate_sum_proxy=np.clip(raw_cost, 0.0, None),
                decision_predicted_carbon_rate_sum_proxy=np.clip(raw_carbon, 0.0, None),
            )
            predicted_objectives = _objective_matrix(
                decision_frame,
                "predicted_annual_heating_energy_gj",
                "decision_predicted_days_below_24_c",
                cost_column="decision_predicted_cost_rate_sum_proxy",
                carbon_column="decision_predicted_carbon_rate_sum_proxy",
            )
        else:
            index_below_zero = 0
            predicted_objectives = _objective_matrix(
                decision_frame,
                "predicted_annual_heating_energy_gj",
                "decision_predicted_days_below_24_c",
            )
        exact_mask = pareto_mask_minimization(exact_objectives)
        predicted_mask = pareto_mask_minimization(predicted_objectives)
        exact_groups = set(
            frame.loc[exact_mask, "configuration_group_id"].astype(int)
        )
        predicted_groups = set(
            frame.loc[predicted_mask, "configuration_group_id"].astype(int)
        )
        true_positive_count = len(exact_groups.intersection(predicted_groups))
        precision = true_positive_count / len(predicted_groups)
        recall = true_positive_count / len(exact_groups)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        union_count = len(exact_groups.union(predicted_groups))
        pareto_records.append(
            {
                "repeat_seed": int(repeat_seed),
                "model": model_name,
                "horizon": int(horizon),
                "configuration_count": len(frame),
                "exact_pareto_count": len(exact_groups),
                "predicted_pareto_count": len(predicted_groups),
                "true_positive_count": true_positive_count,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "jaccard": true_positive_count / union_count,
                "raw_predicted_days_below_zero_count": int(
                    np.count_nonzero(raw_predicted_days < DECISION_DAYS_BOUNDS[0])
                ),
                "raw_predicted_days_above_365_count": int(
                    np.count_nonzero(raw_predicted_days > DECISION_DAYS_BOUNDS[1])
                ),
                "decision_days_clipped_count": int(
                    np.count_nonzero(raw_predicted_days != decision_predicted_days)
                ),
                "raw_predicted_index_below_zero_count": index_below_zero,
            }
        )

        exact_front_objectives = exact_objectives[exact_mask]
        predicted_front_objectives = predicted_objectives[predicted_mask]
        exact_front_normalized, exact_minimum, exact_span = _minmax_transform(
            exact_front_objectives
        )
        predicted_front_normalized, _, _ = _minmax_transform(
            predicted_front_objectives
        )
        true_all_normalized, _, _ = _minmax_transform(
            exact_objectives, exact_minimum, exact_span
        )
        simulation_ids = frame["simulation_id"].to_numpy(int)
        exact_front_ids = simulation_ids[exact_mask]
        predicted_front_ids = simulation_ids[predicted_mask]
        exact_front_positions = np.flatnonzero(exact_mask)
        predicted_front_positions = np.flatnonzero(predicted_mask)

        for profile, weights_tuple in STAKEHOLDER_PROFILES.items():
            weights = np.asarray(weights_tuple, dtype=float)
            exact_scores = exact_front_normalized @ weights
            predicted_scores = predicted_front_normalized @ weights
            exact_local_position = _stable_minimum_position(
                exact_scores, exact_front_ids
            )
            predicted_local_position = _stable_minimum_position(
                predicted_scores, predicted_front_ids
            )
            exact_global_position = exact_front_positions[exact_local_position]
            predicted_global_position = predicted_front_positions[
                predicted_local_position
            ]
            exact_true_score = float(
                true_all_normalized[exact_global_position] @ weights
            )
            predicted_selection_true_score = float(
                true_all_normalized[predicted_global_position] @ weights
            )
            regret = predicted_selection_true_score - exact_true_score
            if -1.0e-12 < regret < 0.0:
                regret = 0.0
            selection_records.append(
                {
                    "repeat_seed": int(repeat_seed),
                    "model": model_name,
                    "horizon": int(horizon),
                    "profile": profile,
                    "exact_simulation_id": int(
                        simulation_ids[exact_global_position]
                    ),
                    "predicted_simulation_id": int(
                        simulation_ids[predicted_global_position]
                    ),
                    "selection_agreement": bool(
                        simulation_ids[exact_global_position]
                        == simulation_ids[predicted_global_position]
                    ),
                    "predicted_selection_on_exact_pareto": bool(
                        exact_mask[predicted_global_position]
                    ),
                    "exact_true_normalized_score": exact_true_score,
                    "predicted_selection_true_normalized_score": (
                        predicted_selection_true_score
                    ),
                    "true_score_regret": float(regret),
                }
            )
    return (
        pd.DataFrame.from_records(pareto_records),
        pd.DataFrame.from_records(selection_records),
    )


def raw_days_prediction_diagnostics(
    oof_predictions: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    """Summarize unbounded OOF D24 predictions for audit metadata."""
    diagnostics: dict[str, dict[str, float | int]] = {}
    for model_name, frame in oof_predictions.groupby("model", sort=True):
        values = frame["predicted_days_below_24_c"].to_numpy(float)
        diagnostics[str(model_name)] = {
            "prediction_count": int(len(values)),
            "count_below_0": int(np.count_nonzero(values < DECISION_DAYS_BOUNDS[0])),
            "count_above_365": int(
                np.count_nonzero(values > DECISION_DAYS_BOUNDS[1])
            ),
            "minimum_raw_prediction": float(values.min()),
            "maximum_raw_prediction": float(values.max()),
        }
    return diagnostics


def _scaler_metadata(
    fold: GroupedFold, scalers: FoldScalers
) -> dict[str, object]:
    return {
        "repeat_seed": fold.repeat_seed,
        "outer_fold": fold.outer_fold,
        "fit_role": "inner_train",
        "fit_row_count": len(fold.inner_train_indices),
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_mean": scalers.features.mean_.tolist(),
        "feature_scale": scalers.features.scale_.tolist(),
        "target_columns": list(TARGET_COLUMNS),
        "target_mean": scalers.targets.mean_.tolist(),
        "target_scale": scalers.targets.scale_.tolist(),
    }


def run_validation(
    configurations: pd.DataFrame,
    config: ValidationConfig = ValidationConfig(),
    input_files: Iterable[Path] = (),
) -> ValidationArtifacts:
    """Run repeated grouped validation and return all publication artifacts."""
    data = prepare_validation_dataset(configurations)
    group_count = int(data["configuration_group_id"].nunique())
    config.validate(group_count)
    features = data[list(FEATURE_COLUMNS)].to_numpy(float)
    targets = data[list(TARGET_COLUMNS)].to_numpy(float)
    groups = data["configuration_group_id"].to_numpy(int)
    folds = build_repeated_grouped_folds(
        groups,
        config.seeds,
        config.outer_folds,
        config.validation_fraction,
    )

    metric_records: list[dict[str, object]] = []
    horizon_metric_records: list[dict[str, object]] = []
    oof_frames: list[pd.DataFrame] = []
    split_frames: list[pd.DataFrame] = []
    scaler_records: list[dict[str, object]] = []
    training_records: list[dict[str, object]] = []

    for fold in folds:
        split_frames.append(_split_assignment_records(data, fold))
        scalers = fit_fold_scalers(features, targets, fold.inner_train_indices)
        scaler_records.append(_scaler_metadata(fold, scalers))
        x_scaled = scalers.features.transform(features)
        y_scaled = scalers.targets.transform(targets)
        fold_predictions, fold_training_records = _run_fold_models(
            x_scaled, y_scaled, fold, scalers, config
        )
        training_records.extend(fold_training_records)
        truth = targets[fold.outer_test_indices]
        test_horizons = data.iloc[fold.outer_test_indices]["horizon"].to_numpy(int)
        for model_name, prediction in fold_predictions.items():
            metric_records.extend(
                _fold_metric_records(
                    truth,
                    prediction,
                    fold.repeat_seed,
                    fold.outer_fold,
                    model_name,
                )
            )
            for horizon in sorted(np.unique(test_horizons)):
                horizon_mask = test_horizons == horizon
                records = _fold_metric_records(
                    truth[horizon_mask],
                    prediction[horizon_mask],
                    fold.repeat_seed,
                    fold.outer_fold,
                    model_name,
                )
                for record in records:
                    record["horizon"] = int(horizon)
                horizon_metric_records.extend(records)
            oof_frames.append(
                _oof_prediction_frame(
                    data, fold, model_name, prediction
                )
            )

    fold_metrics = pd.DataFrame.from_records(metric_records).sort_values(
        ["model", "target", "metric", "repeat_seed", "outer_fold"],
        kind="stable",
    )
    metrics_summary = (
        fold_metrics.groupby(["model", "target", "metric", "unit"], sort=True)[
            "value"
        ]
        .agg(fold_count="count", mean="mean", standard_deviation="std", minimum="min", maximum="max")
        .reset_index()
    )
    horizon_fold_metrics = pd.DataFrame.from_records(
        horizon_metric_records
    ).sort_values(
        ["model", "horizon", "target", "metric", "repeat_seed", "outer_fold"],
        kind="stable",
    )
    horizon_metrics_summary = (
        horizon_fold_metrics.groupby(
            ["model", "horizon", "target", "metric", "unit"], sort=True
        )["value"]
        .agg(
            fold_count="count",
            mean="mean",
            standard_deviation="std",
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )
    oof_predictions = pd.concat(oof_frames, ignore_index=True).sort_values(
        ["repeat_seed", "model", "horizon", "simulation_id"], kind="stable"
    )
    split_assignments = pd.concat(split_frames, ignore_index=True).sort_values(
        ["repeat_seed", "outer_fold", "dataset_row_id"], kind="stable"
    )
    learned_cost_carbon = "cost_rate_sum_proxy" in TARGET_COLUMNS
    pareto_validation, weighted_profile_validation = compute_decision_validation(
        oof_predictions,
        use_predicted_cost_carbon=learned_cost_carbon,
    )

    input_file_list = [Path(path) for path in input_files]
    metadata: dict[str, object] = {
        "script": portable_path(Path(__file__)),
        "protocol": (
            "Repeated shuffled group-exclusive outer CV with a group-exclusive "
            "inner validation split for early stopping."
        ),
        "configuration": asdict(config),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "execution": {
            "device": "cpu",
            "torch_deterministic_algorithms": True,
            "torch_threads": 1,
        },
        "data": {
            "row_count": len(data),
            "configuration_group_count": group_count,
            "horizons": sorted(int(value) for value in data["horizon"].unique()),
            "features": list(FEATURE_COLUMNS),
            "targets": list(TARGET_COLUMNS),
            "group_definition": list(DESIGN_COLUMNS),
            "input_sha256": {
                portable_path(path): sha256(path) for path in sorted(input_file_list)
            },
        },
        "models": {
            "shared_mtl_nn": (
                f"Shared {len(FEATURE_COLUMNS)}->"
                + "->".join(str(width) for width in config.trunk_widths)
                + " SiLU trunk and one "
                f"{config.trunk_widths[-1]}->{config.head_hidden_size}->1 SiLU "
                "head per target; equal standardized MSE across the two targets."
            ),
            "independent_stl_nn": (
                "Two separately trained networks, each matching one complete "
                "input-to-head path of the MTL network, including the full "
                "trunk and head."
            ),
            "random_forest": (
                "Deterministic multi-output RandomForestRegressor fitted to "
                "the same standardized inner-training data."
            ),
            "shared_mtl_nn_mgda": (
                "Same architecture as shared_mtl_nn, trained with the "
                "MGDA trainer: per-task gradients on the shared "
                "trunk, exact two-task min-norm combination, shared-trunk "
                "gradient replaced (not accumulated) after head backprop."
            ),
            "separate_mtl_nn": (
                "Two-head port of the thesis Separate family: one shared "
                f"{len(FEATURE_COLUMNS)}->{config.separate_hidden_size} SiLU "
                "layer, then one "
                f"{config.separate_hidden_size}->{config.separate_hidden_size}"
                "->1 SiLU branch per target; the original output ReLU is "
                "removed because targets are standardized. Equal standardized "
                "MSE trainer."
            ),
            "separate_mtl_nn_mgda": (
                "separate_mtl_nn architecture trained with the MGDA "
                "trainer."
            ),
            "deep_balanced_mtl_nn": (
                "Two-head port of the thesis Deep_Balanced family: shared "
                f"{len(FEATURE_COLUMNS)}->"
                + "->".join(str(width) for width in config.deep_balanced_widths[:2])
                + f" SiLU trunk with dropout {config.deep_balanced_dropout}, "
                "energy-style head 100->50->25->1 for heating energy and "
                "comfort-style head 100->25->12->1 for days below 24 C. "
                "Equal standardized MSE trainer."
            ),
            "deep_balanced_mtl_nn_mgda": (
                "deep_balanced_mtl_nn architecture trained with the "
                "MGDA trainer."
            ),
            "gradient_boosting": (
                "Per-target HistGradientBoostingRegressor fitted to the same "
                f"standardized inner-training data; max_iter="
                f"{config.hgb_max_iter}, learning_rate="
                f"{config.hgb_learning_rate}, no early stopping, seeded per "
                "fold and target."
            ),
            "neural_optimizer": {
                "name": "AdamW",
                "learning_rate": config.learning_rate,
                "betas": [0.9, 0.999],
                "epsilon": 1.0e-8,
                "weight_decay": config.weight_decay,
                "amsgrad": False,
                "gradient_clip_norm": config.gradient_clip_norm,
            },
            "learning_rate_scheduler": {
                "name": "ReduceLROnPlateau",
                "monitored_value": "inner_validation_standardized_mse",
                "mode": "min",
                "factor": config.lr_scheduler_factor,
                "patience": config.lr_scheduler_patience,
                "threshold": 1.0e-4,
                "threshold_mode": "rel",
                "cooldown": 0,
                "minimum_learning_rate": config.minimum_learning_rate,
            },
            "initialization": "Kaiming uniform for all linear weights; zero biases.",
            "random_forest_hyperparameters": {
                "n_estimators": config.rf_estimators,
                "criterion": "squared_error",
                "max_depth": config.rf_max_depth,
                "min_samples_leaf": config.rf_min_samples_leaf,
                "max_features": 1.0,
                "bootstrap": True,
                "n_jobs": 1,
            },
        },
        "decision_validation": {
            "learned_objectives": list(TARGET_COLUMNS),
            "exact_objectives": list(DECISION_EXACT_COLUMNS),
            "predicted_front_cost_carbon_source": (
                "model predictions" if learned_cost_carbon
                else "exact rate-table values"
            ),
            "pareto_directions": [
                "minimize annual_heating_energy_gj",
                "minimize cost_rate_sum_proxy",
                "minimize carbon_rate_sum_proxy",
                "maximize days_below_24_c",
            ],
            "true_score_regret_definition": (
                "Exact-objective weighted score of the surrogate-selected "
                "configuration minus the exact optimum score, using min-max "
                "bounds from the exact Pareto front."
            ),
            "days_prediction_policy": {
                "raw_oof_and_point_metrics": (
                    "Unclipped predicted days_below_24_c values are preserved "
                    "and used for all MAE, RMSE, and R2 metrics."
                ),
                "pareto_and_weighted_profile_decisions": (
                    "Predicted days_below_24_c is clipped to the physically "
                    "admissible closed interval [0, 365] immediately before "
                    "constructing predicted Pareto fronts and weighted-profile "
                    "selections. In the four-target design the predicted cost and "
                    "GWP indices are clipped at zero for the same reason. Heating "
                    "energy is not clipped."
                ),
                "lower_bound_days": DECISION_DAYS_BOUNDS[0],
                "upper_bound_days": DECISION_DAYS_BOUNDS[1],
            },
            "raw_days_prediction_diagnostics_by_model": (
                raw_days_prediction_diagnostics(oof_predictions)
            ),
            "stakeholder_profiles": STAKEHOLDER_PROFILES,
        },
        "scalers": scaler_records,
        "training_runs": training_records,
    }
    return ValidationArtifacts(
        fold_metrics=fold_metrics.reset_index(drop=True),
        metrics_summary=metrics_summary.reset_index(drop=True),
        horizon_fold_metrics=horizon_fold_metrics.reset_index(drop=True),
        horizon_metrics_summary=horizon_metrics_summary.reset_index(drop=True),
        oof_predictions=oof_predictions.reset_index(drop=True),
        split_assignments=split_assignments.reset_index(drop=True),
        pareto_validation=pareto_validation.reset_index(drop=True),
        weighted_profile_validation=weighted_profile_validation.reset_index(
            drop=True
        ),
        metadata=metadata,
    )


def write_artifacts(output_dir: Path, artifacts: ValidationArtifacts) -> None:
    """Write deterministic, machine-readable validation artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_outputs = {
        "fold_metrics.csv": artifacts.fold_metrics,
        "metrics_summary.csv": artifacts.metrics_summary,
        "horizon_fold_metrics.csv": artifacts.horizon_fold_metrics,
        "horizon_metrics_summary.csv": artifacts.horizon_metrics_summary,
        "oof_predictions.csv": artifacts.oof_predictions,
        "split_assignments.csv": artifacts.split_assignments,
        "surrogate_pareto_validation.csv": artifacts.pareto_validation,
        "weighted_profile_validation.csv": artifacts.weighted_profile_validation,
    }
    for filename, frame in csv_outputs.items():
        frame.to_csv(output_dir / filename, index=False, float_format="%.17g")
    with (output_dir / "model_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(artifacts.metadata, handle, indent=2, allow_nan=False)
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
        default=REPO_ROOT / "results" / "surrogate_validation",
        help="Directory for validation CSV and JSON artifacts.",
    )
    parser.add_argument(
        "--learned-targets",
        choices=("two", "four"),
        default="two",
        help=(
            "two: learn heating energy and days-below-24C, cost/carbon exact "
            "(publication protocol). four: also learn the cost and carbon "
            "rate indices, as in the original four-task study; predicted "
            "decision fronts then use predicted cost/carbon."
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--trunk-widths",
        nargs="+",
        type=int,
        default=[128, 128, 64],
        help="Shared/default MLP trunk widths (publication default: 128 128 64).",
    )
    parser.add_argument("--head-hidden-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1.0e-6)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lr-scheduler-patience", type=int, default=10)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--rf-estimators", type=int, default=300)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_learned_targets(args.learned_targets)
    config = ValidationConfig(
        seeds=tuple(args.seeds),
        outer_folds=args.outer_folds,
        validation_fraction=args.validation_fraction,
        trunk_widths=tuple(args.trunk_widths),
        head_hidden_size=args.head_hidden_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_patience=args.lr_scheduler_patience,
        minimum_learning_rate=args.minimum_learning_rate,
        batch_size=args.batch_size,
        gradient_clip_norm=args.gradient_clip_norm,
        rf_estimators=args.rf_estimators,
        rf_min_samples_leaf=args.rf_min_samples_leaf,
        rf_max_depth=args.rf_max_depth,
    )
    configurations, input_files = load_exact_dataset(args.input_dir)
    artifacts = run_validation(configurations, config, input_files)
    write_artifacts(args.output_dir, artifacts)
    print(artifacts.metrics_summary.to_string(index=False))
    print()
    print(
        artifacts.pareto_validation.groupby("model")[["precision", "recall"]]
        .mean()
        .to_string()
    )
    print(f"\nWrote surrogate validation outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
