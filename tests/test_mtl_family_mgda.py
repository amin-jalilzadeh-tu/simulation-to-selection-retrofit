"""Tests for the MGDA trainer of scripts/validate_mtl_family.py.

Nothing in the released test suite exercised the MGDA code path, although four
of the nine reported formulations are trained with it. These tests pin the two
behaviours the manuscript's MGDA claim rests on:

* the min-norm task weights - the exact closed form used for the two-target
  task design, and the Frank-Wolfe iteration used for the four-target design;
* the gradient semantics - the task-head losses are backpropagated and the
  shared-trunk gradient is then REPLACED by the min-norm combination; these
  tests fail if the min-norm gradient is ever accumulated on top of the summed
  gradient instead.

Run from the extracted archive root:

    python -m unittest discover -s tests
"""

import itertools
import unittest

import numpy as np
import torch

from scripts import validate_mtl_family as family


def _flat_gradients(loss, parameters):
    """Flatten d(loss)/d(parameters) into one vector."""
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True)
    return torch.cat([gradient.reshape(-1) for gradient in gradients])


def _norm_squared(weights, gradients):
    combined = sum(
        float(weight) * gradient for weight, gradient in zip(weights, gradients)
    )
    return float(torch.dot(combined, combined).item())


class TestTwoTaskMinNormWeights(unittest.TestCase):
    """The closed form used whenever exactly two targets are learned."""

    def test_matches_the_hand_computed_optimum(self):
        # min over a of ||a*(2,0) + (1-a)*(0,1)||^2 = 4a^2 + (1-a)^2
        # d/da = 10a - 2 = 0  ->  a = 0.2
        gradient_a = torch.tensor([2.0, 0.0])
        gradient_b = torch.tensor([0.0, 1.0])

        alpha = family._two_task_min_norm_alpha(gradient_a, gradient_b)
        weights = family._min_norm_weights([gradient_a, gradient_b])

        self.assertAlmostEqual(alpha, 0.2, places=12)
        np.testing.assert_allclose(weights, [0.2, 0.8], atol=1.0e-12)

    def test_identical_gradients_split_the_weight_evenly(self):
        gradient = torch.tensor([1.0, -3.0, 0.5])

        alpha = family._two_task_min_norm_alpha(gradient, gradient.clone())

        self.assertEqual(alpha, 0.5)

    def test_directly_opposing_gradients_cancel(self):
        gradient_a = torch.tensor([1.0, 0.0])
        gradient_b = torch.tensor([-1.0, 0.0])

        weights = family._min_norm_weights([gradient_a, gradient_b])

        np.testing.assert_allclose(weights, [0.5, 0.5], atol=1.0e-12)
        self.assertAlmostEqual(_norm_squared(weights, [gradient_a, gradient_b]), 0.0)

    def test_weights_stay_inside_the_unit_interval(self):
        # The unconstrained minimiser is alpha = 1.5; the simplex constraint
        # must clip it to the endpoint rather than extrapolate past it.
        gradient_a = torch.tensor([1.0, 0.0])
        gradient_b = torch.tensor([3.0, 0.0])

        alpha = family._two_task_min_norm_alpha(gradient_a, gradient_b)

        self.assertEqual(alpha, 1.0)

    def test_closed_form_beats_a_dense_line_search(self):
        # Double precision isolates the algebra from float32 rounding; the
        # closed form is exact, so it must never lose to a sampled minimum.
        generator = torch.Generator().manual_seed(20260822)
        for trial in range(25):
            gradient_a = torch.randn(11, generator=generator, dtype=torch.float64)
            gradient_b = torch.randn(11, generator=generator, dtype=torch.float64)

            alpha = family._two_task_min_norm_alpha(gradient_a, gradient_b)
            best = _norm_squared([alpha, 1.0 - alpha], [gradient_a, gradient_b])
            grid = min(
                _norm_squared([value, 1.0 - value], [gradient_a, gradient_b])
                for value in np.linspace(0.0, 1.0, 2001)
            )

            self.assertLessEqual(
                best, grid * (1.0 + 1.0e-12) + 1.0e-12, f"trial {trial}"
            )


class TestFrankWolfeMinNormWeights(unittest.TestCase):
    """The solver used when the four-target task design is selected."""

    # Frank-Wolfe converges at O(1/k) and the released solver stops after a
    # fixed 250 iterations, so the optimum is approached, not hit exactly.
    # Measured over 60 random four-gradient instances, the relative gap to a
    # 200,000-iteration reference is 3e-07 at the median but reaches 1.5e-02
    # on roughly one instance in ten. The bound below sits above every value
    # observed there; it documents the accuracy of the shipped budget rather
    # than asserting exact optimality.
    CONVERGENCE_TOLERANCE = 2.5e-2

    @staticmethod
    def _random_gradients(count, size, seed):
        generator = torch.Generator().manual_seed(seed)
        return [
            torch.randn(size, generator=generator, dtype=torch.float64)
            for _ in range(count)
        ]

    @staticmethod
    def _reference_min_norm(gradients, iterations=50000):
        """Long-run Frank-Wolfe reference for the same quadratic program."""
        stacked = torch.stack(gradients)
        gram = (stacked @ stacked.T).numpy().astype(np.float64)
        count = len(gradients)
        weights = np.full(count, 1.0 / count)
        for _ in range(iterations):
            descent_index = int(np.argmin(gram @ weights))
            direction = -weights
            direction[descent_index] += 1.0
            curvature = float(direction @ gram @ direction)
            if curvature <= 1.0e-16:
                break
            step = float(-(weights @ gram @ direction)) / curvature
            step = min(1.0, max(0.0, step))
            if step <= 1.0e-16:
                break
            weights = weights + step * direction
        return float(weights @ gram @ weights)

    def test_weights_lie_on_the_probability_simplex(self):
        for seed in (1, 2, 3, 4, 5):
            gradients = self._random_gradients(4, 17, seed)

            weights = family._min_norm_weights(gradients)

            self.assertEqual(weights.shape, (4,))
            self.assertAlmostEqual(float(weights.sum()), 1.0, places=10)
            self.assertGreaterEqual(float(weights.min()), -1.0e-12)

    def test_satisfies_the_min_norm_optimality_condition(self):
        # For the minimum-norm element g* of the convex hull of the task
        # gradients, <g_i, g*> >= ||g*||^2 for every task i.
        for seed in (11, 12, 13):
            gradients = self._random_gradients(4, 9, seed)

            weights = family._min_norm_weights(gradients)
            combined = sum(
                float(weight) * gradient
                for weight, gradient in zip(weights, gradients)
            )
            combined_norm_squared = float(torch.dot(combined, combined).item())
            projections = [
                float(torch.dot(gradient, combined).item())
                for gradient in gradients
            ]

            tolerance = self.CONVERGENCE_TOLERANCE * max(
                1.0, combined_norm_squared
            )
            self.assertGreaterEqual(
                min(projections), combined_norm_squared - tolerance, f"seed {seed}"
            )

    def test_within_the_documented_tolerance_of_the_converged_optimum(self):
        for seed in (31, 32, 33):
            gradients = self._random_gradients(4, 9, seed)

            weights = family._min_norm_weights(gradients)
            solver_value = _norm_squared(weights, gradients)
            reference_value = self._reference_min_norm(gradients)

            self.assertGreaterEqual(solver_value, reference_value - 1.0e-12)
            self.assertLessEqual(
                solver_value,
                reference_value * (1.0 + self.CONVERGENCE_TOLERANCE) + 1.0e-12,
                f"seed {seed}",
            )

    def test_two_task_instances_never_reach_the_frank_wolfe_branch(self):
        # The closed form is exact, so the 250-iteration budget never applies
        # to the published two-target task design.
        gradients = self._random_gradients(2, 9, 41)

        weights = family._min_norm_weights(gradients)
        alpha = family._two_task_min_norm_alpha(*gradients)

        np.testing.assert_allclose(weights, [alpha, 1.0 - alpha], atol=0.0)

    def test_no_worse_than_a_dense_simplex_grid(self):
        step = 0.05
        levels = int(round(1.0 / step))
        grid = [
            np.array(point, dtype=float) / levels
            for point in itertools.product(range(levels + 1), repeat=4)
            if sum(point) == levels
        ]
        self.assertEqual(len(grid), 1771)

        for seed in (21, 22, 23):
            gradients = self._random_gradients(4, 13, seed)

            weights = family._min_norm_weights(gradients)
            solver_value = _norm_squared(weights, gradients)
            grid_value = min(
                _norm_squared(candidate, gradients) for candidate in grid
            )

            self.assertLessEqual(solver_value, grid_value + 1.0e-9, f"seed {seed}")

    def test_identical_gradients_stay_uniform(self):
        gradient = torch.tensor([0.5, -0.25, 2.0])
        gradients = [gradient.clone() for _ in range(4)]

        weights = family._min_norm_weights(gradients)

        np.testing.assert_allclose(weights, [0.25] * 4, atol=1.0e-12)

    def test_duplicated_task_pair_reproduces_the_two_task_solution(self):
        gradient_a = torch.tensor([2.0, 0.0, 1.0])
        gradient_b = torch.tensor([0.0, 1.0, -1.0])

        pair_weights = family._min_norm_weights([gradient_a, gradient_b])
        quadruple_weights = family._min_norm_weights(
            [gradient_a, gradient_a.clone(), gradient_b, gradient_b.clone()]
        )

        pair_combined = (
            float(pair_weights[0]) * gradient_a
            + float(pair_weights[1]) * gradient_b
        )
        quadruple_combined = sum(
            float(weight) * gradient
            for weight, gradient in zip(
                quadruple_weights, [gradient_a, gradient_a, gradient_b, gradient_b]
            )
        )

        torch.testing.assert_close(
            quadruple_combined, pair_combined, rtol=0.0, atol=1.0e-3
        )


class TestMgdaSharedGradientSemantics(unittest.TestCase):
    """The shared trunk must be REPLACED, not accumulated."""

    def setUp(self):
        family._set_learned_targets("two")
        torch.manual_seed(20260822)
        self.model = family.SharedTwoHeadRegressor(
            input_size=3, trunk_widths=(5, 4), head_hidden_size=3
        )
        # A zero learning rate keeps the parameters fixed, so the reference
        # gradients computed before the step remain the correct comparison.
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0)
        generator = torch.Generator().manual_seed(7)
        self.features = torch.randn(12, 3, generator=generator)
        self.targets = torch.randn(12, 2, generator=generator)

    def _reference_gradients(self):
        shared_parameters = list(self.model.shared.parameters())
        head_parameters = list(self.model.heads.parameters())
        prediction = self.model(self.features)
        task_losses = [
            torch.mean(torch.square(prediction[:, index] - self.targets[:, index]))
            for index in range(2)
        ]
        per_task = [
            _flat_gradients(loss, shared_parameters) for loss in task_losses
        ]
        weights = family._min_norm_weights(per_task)
        combined = sum(
            float(weight) * gradient for weight, gradient in zip(weights, per_task)
        )
        summed = per_task[0] + per_task[1]
        head_summed = _flat_gradients(sum(task_losses), head_parameters)
        return (
            combined,
            summed,
            head_summed,
            [float(loss.detach()) for loss in task_losses],
        )

    def _observed_shared_gradient(self):
        return torch.cat(
            [
                parameter.grad.reshape(-1)
                for parameter in self.model.shared.parameters()
            ]
        )

    def _observed_head_gradient(self):
        return torch.cat(
            [
                parameter.grad.reshape(-1)
                for parameter in self.model.heads.parameters()
            ]
        )

    def test_shared_gradient_equals_the_min_norm_combination(self):
        combined, _, _, _ = self._reference_gradients()

        family._mgda_training_step(
            self.model,
            self.optimizer,
            self.features,
            self.targets,
            gradient_clip_norm=1.0e9,
        )

        torch.testing.assert_close(
            self._observed_shared_gradient(), combined, rtol=0.0, atol=1.0e-6
        )

    def test_shared_gradient_is_not_the_accumulated_historical_result(self):
        combined, summed, _, _ = self._reference_gradients()
        accumulated = combined + summed

        family._mgda_training_step(
            self.model,
            self.optimizer,
            self.features,
            self.targets,
            gradient_clip_norm=1.0e9,
        )
        observed = self._observed_shared_gradient()

        # Guard the test itself: the two candidate answers must be far apart,
        # otherwise the assertion below would pass for the wrong reason.
        self.assertGreater(float(torch.norm(summed).item()), 1.0e-6)
        self.assertFalse(torch.allclose(observed, accumulated, atol=1.0e-6))
        self.assertFalse(torch.allclose(observed, summed, atol=1.0e-6))

    def test_task_heads_keep_the_summed_loss_gradient(self):
        _, _, head_summed, _ = self._reference_gradients()

        family._mgda_training_step(
            self.model,
            self.optimizer,
            self.features,
            self.targets,
            gradient_clip_norm=1.0e9,
        )

        torch.testing.assert_close(
            self._observed_head_gradient(), head_summed, rtol=0.0, atol=1.0e-6
        )

    def test_reported_loss_is_the_equal_weight_task_mean(self):
        _, _, _, task_losses = self._reference_gradients()

        reported = family._mgda_training_step(
            self.model,
            self.optimizer,
            self.features,
            self.targets,
            gradient_clip_norm=1.0e9,
        )

        self.assertAlmostEqual(
            float(reported), sum(task_losses) / len(task_losses), places=5
        )

    def test_clipping_is_applied_after_the_replacement(self):
        clip_norm = 1.0e-3

        family._mgda_training_step(
            self.model,
            self.optimizer,
            self.features,
            self.targets,
            gradient_clip_norm=clip_norm,
        )

        total_norm = float(
            torch.norm(
                torch.cat(
                    [
                        parameter.grad.reshape(-1)
                        for parameter in self.model.parameters()
                    ]
                )
            ).item()
        )
        self.assertLessEqual(total_norm, clip_norm * (1.0 + 1.0e-6))
        self.assertAlmostEqual(total_norm, clip_norm, places=6)

    def test_four_task_step_uses_the_frank_wolfe_weights(self):
        family._set_learned_targets("four")
        try:
            torch.manual_seed(31)
            model = family.SharedTwoHeadRegressor(
                input_size=3, trunk_widths=(5, 4), head_hidden_size=3
            )
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            generator = torch.Generator().manual_seed(19)
            features = torch.randn(10, 3, generator=generator)
            targets = torch.randn(10, 4, generator=generator)
            shared_parameters = list(model.shared.parameters())
            prediction = model(features)
            self.assertEqual(prediction.shape[1], 4)
            per_task = [
                _flat_gradients(
                    torch.mean(torch.square(prediction[:, index] - targets[:, index])),
                    shared_parameters,
                )
                for index in range(4)
            ]
            weights = family._min_norm_weights(per_task)
            combined = sum(
                float(weight) * gradient
                for weight, gradient in zip(weights, per_task)
            )

            family._mgda_training_step(
                model, optimizer, features, targets, gradient_clip_norm=1.0e9
            )
            observed = torch.cat(
                [parameter.grad.reshape(-1) for parameter in model.shared.parameters()]
            )

            torch.testing.assert_close(observed, combined, rtol=0.0, atol=1.0e-6)
        finally:
            family._set_learned_targets("two")


class TestTrainerSelection(unittest.TestCase):
    def test_unknown_trainer_name_is_rejected(self):
        family._set_learned_targets("two")
        model = family.SharedTwoHeadRegressor(3, (4,), 2)
        features = np.zeros((4, 3), dtype=float)
        targets = np.zeros((4, 2), dtype=float)

        with self.assertRaises(ValueError):
            family.train_neural_regressor(
                model,
                features,
                targets,
                features,
                targets,
                family.ValidationConfig(),
                seed=1,
                trainer="momentum",
            )


if __name__ == "__main__":
    unittest.main()
