import math
import unittest

import torch

from wafer_defect_studio.spatial_mil import (
    dense_absent_class_loss,
    positive_spatial_lse_loss,
    present_sparse_budget_loss,
)


class SpatialMilV5LossTest(unittest.TestCase):
    def test_distributes_positive_and_absent_gradients_and_enforces_sparse_budget(self) -> None:
        positive_logits = torch.tensor(
            [[[[[2.0, 1.0], [0.0, -1.0]]]]], requires_grad=True
        )
        positive_spatial_lse_loss(
            positive_logits, torch.tensor([[1]]), temperature=0.5
        ).backward()
        self.assertTrue(torch.all(positive_logits.grad < 0.0))

        absent_logits = torch.tensor(
            [[[[[3.0, 1.0], [0.0, -1.0]]]]], requires_grad=True
        )
        dense_absent_class_loss(
            absent_logits, torch.tensor([[0]]), hardest_fraction=0.25
        ).backward()
        self.assertTrue(torch.all(absent_logits.grad > 0.0))
        self.assertGreater(absent_logits.grad[0, 0, 0, 0, 0], absent_logits.grad[0, 0, 0, 0, 1])

        below = torch.full(
            (1, 1, 1, 2, 2), math.log(0.005 / 0.995), requires_grad=True
        )
        at_or_below_loss = present_sparse_budget_loss(
            below, torch.tensor([[1]]), probability_budget=0.01
        )
        at_or_below_loss.backward()
        self.assertEqual(at_or_below_loss.item(), 0.0)
        self.assertTrue(torch.equal(below.grad, torch.zeros_like(below)))

        above = torch.full(
            (1, 1, 1, 2, 2), math.log(0.02 / 0.98), requires_grad=True
        )
        above_loss = present_sparse_budget_loss(
            above, torch.tensor([[1]]), probability_budget=0.01
        )
        above_loss.backward()
        self.assertGreater(above_loss.item(), 0.0)
        self.assertTrue(torch.all(above.grad > 0.0))

    def test_no_matching_targets_return_autograd_connected_zero(self) -> None:
        logits = torch.randn(1, 2, 1, 2, 2, requires_grad=True)
        targets = torch.zeros(1, 1)
        loss = positive_spatial_lse_loss(logits, targets) + present_sparse_budget_loss(
            logits, targets
        )
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

        logits = torch.randn(1, 2, 1, 2, 2, requires_grad=True)
        loss = dense_absent_class_loss(logits, torch.ones(1, 1))
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))


if __name__ == "__main__":
    unittest.main()
