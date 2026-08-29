import unittest

import torch

from worker.role_aware_core import (
    calculate_role_scores,
    current_form_score,
    exact_k_indices,
    full_support_momentum_update,
    material_checkpoint,
    open_modulation_gate,
    role_alphas,
    selected_swap_fraction,
)


class RoleAwareCoreTests(unittest.TestCase):
    def test_role_aware_upper_bounds_structural(self):
        e = torch.tensor([2.0, -1.0, 1.0])
        i = torch.tensor([-1.0, 2.0, -3.0])
        x = torch.tensor([0.5, -0.5, 1.0])
        d = torch.tensor([-0.5, 0.5, 2.0])
        scores = calculate_role_scores(e, i, x, d)
        self.assertTrue(torch.all(scores.role_aware_taylor >= scores.structural_taylor))
        self.assertGreater(scores.conflict, 0.0)

    def test_exact_k_ties_use_channel_index(self):
        selected = exact_k_indices(torch.tensor([1.0, 3.0, 3.0, 2.0]), 2)
        self.assertEqual(selected.tolist(), [1, 2])

    def test_swap_fraction(self):
        self.assertEqual(
            selected_swap_fraction(torch.tensor([0, 1]), torch.tensor([1, 2]), 2),
            0.5,
        )

    def test_role_alpha_identity_mean_and_bounds(self):
        dof = torch.zeros(5, 2, requires_grad=True)
        zero = role_alphas(dof)
        self.assertTrue(torch.equal(zero, torch.ones(5, 3)))
        asymmetric = torch.tensor([1.0, 2.0, 4.0]).expand_as(zero)
        gradient = torch.autograd.grad((zero * asymmetric).sum(), dof)[0]
        self.assertGreater(float(gradient.abs().max()), 0.0)
        alpha = role_alphas(torch.tensor([[100.0, -100.0], [-2.0, 3.0]]))
        self.assertTrue(torch.allclose(alpha.mean(-1), torch.ones(2), atol=1e-7))
        self.assertGreaterEqual(float(alpha.min()), 0.75)
        self.assertLessEqual(float(alpha.max()), 1.25)

    def test_probe_momentum_is_separate_and_full_support(self):
        gradient = torch.ones(2, 3)
        momentum = full_support_momentum_update(None, gradient)
        gradient.zero_()
        self.assertTrue(torch.equal(momentum, torch.ones(2, 3)))
        score = current_form_score(torch.full((2, 3), 2.0), momentum)
        self.assertEqual(score.tolist(), [2.0, 2.0])

    def test_material_and_stage_b_gates(self):
        rows = [{"conflict": 0.3, "swap_fraction": 0.2} for _ in range(2)]
        self.assertTrue(material_checkpoint(rows))
        self.assertTrue(open_modulation_gate({0: [1, 1, 0, 0, 1], 1: [1, 0, 1, 0, 1]}))
        self.assertFalse(open_modulation_gate({0: [1, 1, 0, 0, 1], 1: [1, 0, 0, 0, 1]}))

    def test_zero_tied_role_signal_is_invalid(self):
        zeros = torch.zeros(3)
        with self.assertRaises(ValueError):
            calculate_role_scores(zeros, zeros, zeros, torch.ones(3))


if __name__ == "__main__":
    unittest.main()
