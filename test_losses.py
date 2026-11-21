"""Synthetic-only loss checks: python -m unittest test_losses.py."""
import unittest
import torch
from losses import (differentiable_kendall, identity_consistency,
                    objective, ranking_regularizer)


class LossTests(unittest.TestCase):
    def test_sign_and_degenerate_ranks(self):
        x = torch.tensor([0., 1., 2.])
        self.assertAlmostEqual(differentiable_kendall(x, x).item(), 1., places=5)
        self.assertAlmostEqual(differentiable_kendall(x, x.flip(0)).item(), -1., places=5)
        self.assertEqual(differentiable_kendall(x[:1], x[:1]).item(), 0.)

    def test_unequal_branch_dimensions_and_sparse_labels(self):
        torch.manual_seed(17)
        labels = torch.tensor([0, 0, 0, 3, 3, 3, 5, 5, 5])
        g, f = torch.randn(9, 4), torch.randn(9, 7)
        for value in ranking_regularizer(g, f, labels):
            self.assertTrue(torch.isfinite(value))
            self.assertLessEqual(abs(value.item()), 1.)

    def test_objectives_have_finite_gradients(self):
        torch.manual_seed(19)
        labels = torch.arange(3).repeat_interleave(3)
        for v in ("base", "ratr", "ratr_abs", "ratr_hinge", "cov",
                  "consistency", "cov_consistency", "ratr_cov_consistency"):
            with self.subTest(v=v):
                def branch(d):
                    return (torch.randn(9, d, requires_grad=True),
                            torch.randn(9, 3, requires_grad=True))
                g, f, a = branch(4), branch(7), branch(5)
                result = objective({"backbone": g, "fusion": f, "aux": (a,)}, labels, variant=v)
                result["loss"].backward()
                for tensor in (*g, *f, *a):
                    self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_identity_js_zero_for_identical_distributions(self):
        logits = torch.randn(4, 3)
        self.assertAlmostEqual(identity_consistency(logits, logits).item(), 0., places=6)


if __name__ == "__main__":
    unittest.main()
