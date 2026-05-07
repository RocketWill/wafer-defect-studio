import unittest

import torch
import torch.nn.functional as F

from wafer_defect_studio.spatial_mil import positive_spatial_mil_loss


class PositiveSpatialMilLossTest(unittest.TestCase):
    def test_averages_positive_class_max_logit_losses(self) -> None:
        logits = torch.tensor(
            [
                [[[1.0, 3.0]], [[-2.0, -1.0]], [[0.0, 2.0]]],
                [[[4.0, 1.0]], [[5.0, 0.0]], [[-3.0, -4.0]]],
            ]
        )
        targets = torch.tensor([[1, 0, 1], [0, 1, 0]])

        loss = positive_spatial_mil_loss(logits, targets)

        expected = torch.stack(
            [
                F.softplus(torch.tensor(-3.0)),
                F.softplus(torch.tensor(-2.0)),
                F.softplus(torch.tensor(-5.0)),
            ]
        ).mean()
        torch.testing.assert_close(loss, expected)

    def test_gradients_reach_only_positive_class_spatial_maxima(self) -> None:
        logits = torch.tensor(
            [[[[1.0, 4.0]], [[3.0, 2.0]], [[-1.0, 5.0]]]], requires_grad=True
        )
        targets = torch.tensor([[1, 0, 1]])

        positive_spatial_mil_loss(logits, targets).backward()

        gradient = logits.grad
        self.assertIsNotNone(gradient)
        self.assertEqual(0.0, gradient[0, 0, 0, 0].item())
        self.assertLess(gradient[0, 0, 0, 1].item(), 0.0)
        self.assertTrue(torch.equal(gradient[0, 1], torch.zeros_like(gradient[0, 1])))
        self.assertEqual(0.0, gradient[0, 2, 0, 0].item())
        self.assertLess(gradient[0, 2, 0, 1].item(), 0.0)

    def test_zero_positive_target_returns_autograd_connected_zero(self) -> None:
        logits = torch.randn(2, 3, 2, 2, requires_grad=True)
        targets = torch.zeros(2, 3, dtype=torch.int64)

        loss = positive_spatial_mil_loss(logits, targets)
        loss.backward()

        self.assertEqual(0.0, loss.item())
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))


if __name__ == "__main__":
    unittest.main()
