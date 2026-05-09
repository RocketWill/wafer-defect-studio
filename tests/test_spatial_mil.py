import unittest

import torch
import torch.nn.functional as F

from wafer_defect_studio.detection_windows import Rect
from wafer_defect_studio.spatial_mil import (
    absent_class_hard_negative_loss,
    derive_train_positive_class_weights,
    overlap_consistency_loss,
    positive_spatial_mil_loss,
)
from wafer_defect_studio.training_input_bundle import (
    TrainingBundleSource,
    TrainingInputBundle,
    TrainingPatchBag,
)


class PositiveSpatialMilLossTest(unittest.TestCase):
    def test_pools_a_5d_bag_and_applies_class_weights(self) -> None:
        logits = torch.tensor([[[[[1.0]], [[2.0]]], [[[3.0]], [[-4.0]]]]])
        targets = torch.tensor([[1, 1]])
        loss = positive_spatial_mil_loss(logits, targets, torch.tensor([2.0, 1.0]))
        expected = torch.stack((2 * F.softplus(torch.tensor(-3.0)), F.softplus(torch.tensor(-2.0)))).mean()
        torch.testing.assert_close(loss, expected)
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


class AbsentClassHardNegativeLossTest(unittest.TestCase):
    def test_pools_patch_and_spatial_dimensions_for_5d_bags(self) -> None:
        logits = torch.tensor([[[[[1.0]], [[2.0]]], [[[3.0]], [[-4.0]]]]])
        loss = absent_class_hard_negative_loss(logits, torch.tensor([[0, 1]]))
        torch.testing.assert_close(loss, F.softplus(torch.tensor(3.0)))
    def test_averages_absent_class_max_logit_losses(self) -> None:
        logits = torch.tensor(
            [
                [[[1.0, 3.0]], [[-2.0, -1.0]], [[0.0, 2.0]]],
                [[[4.0, 1.0]], [[5.0, 0.0]], [[-3.0, -4.0]]],
            ]
        )
        targets = torch.tensor([[1, 0, 0], [0, 1, 1]])

        loss = absent_class_hard_negative_loss(logits, targets)

        expected = torch.stack(
            [
                F.softplus(torch.tensor(-1.0)),
                F.softplus(torch.tensor(2.0)),
                F.softplus(torch.tensor(4.0)),
            ]
        ).mean()
        torch.testing.assert_close(loss, expected)

    def test_gradients_reach_only_absent_class_spatial_maxima(self) -> None:
        logits = torch.tensor(
            [[[[1.0, 4.0]], [[3.0, 2.0]], [[-1.0, 5.0]]]], requires_grad=True
        )
        targets = torch.tensor([[0, 1, 0]])

        absent_class_hard_negative_loss(logits, targets).backward()

        gradient = logits.grad
        self.assertIsNotNone(gradient)
        self.assertEqual(0.0, gradient[0, 0, 0, 0].item())
        self.assertGreater(gradient[0, 0, 0, 1].item(), 0.0)
        self.assertTrue(torch.equal(gradient[0, 1], torch.zeros_like(gradient[0, 1])))
        self.assertEqual(0.0, gradient[0, 2, 0, 0].item())
        self.assertGreater(gradient[0, 2, 0, 1].item(), 0.0)

    def test_all_present_target_returns_autograd_connected_zero(self) -> None:
        logits = torch.randn(2, 3, 2, 2, requires_grad=True)
        targets = torch.ones(2, 3, dtype=torch.int64)

        loss = absent_class_hard_negative_loss(logits, targets)
        loss.backward()

        self.assertEqual(0.0, loss.item())
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))


class OverlapConsistencyLossTest(unittest.TestCase):
    def test_averages_squared_probability_difference_in_source_overlap(self) -> None:
        logits = torch.stack(
            [
                torch.zeros(1, 2, 2),
                torch.full((1, 2, 2), torch.log(torch.tensor(3.0))),
            ]
        ).requires_grad_()

        loss = overlap_consistency_loss(
            logits,
            [Rect(0, 0, 2, 2), Rect(1, 0, 2, 2)],
            feature_stride=1,
        )

        torch.testing.assert_close(loss, torch.tensor(0.0625))
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[0, :, :, 0], torch.zeros(1, 2)))
        self.assertTrue(torch.all(logits.grad[0, :, :, 1] < 0.0))
        self.assertTrue(torch.all(logits.grad[1, :, :, 0] > 0.0))
        self.assertTrue(torch.equal(logits.grad[1, :, :, 1], torch.zeros(1, 2)))

    def test_disjoint_patches_return_autograd_connected_zero(self) -> None:
        logits = torch.randn(2, 1, 2, 2, requires_grad=True)

        loss = overlap_consistency_loss(
            logits,
            [Rect(0, 0, 2, 2), Rect(2, 0, 2, 2)],
            feature_stride=1,
        )
        loss.backward()

        self.assertEqual(0.0, loss.item())
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))


class TrainPositiveClassWeightsTest(unittest.TestCase):
    def test_caps_rare_class_and_ignores_validation_and_test_labels(self) -> None:
        bundle = _bundle(
            ("scratch", "particle", "void"),
            [("train", ("scratch", "particle"))]
            + [("train", ("particle",))] * 11
            + [("validation", ("scratch", "void")), ("test", ("scratch", "void"))],
        )

        weights = derive_train_positive_class_weights(bundle)

        torch.testing.assert_close(weights, torch.tensor([10.0, 1.0, 1.0]))
        self.assertEqual(torch.float32, weights.dtype)

    def test_preserves_class_order_and_uncapped_ratio(self) -> None:
        bundle = _bundle(
            ("particle", "scratch"),
            [
                ("train", ("scratch",)),
                ("train", ("scratch",)),
                ("train", ("particle", "scratch")),
                ("train", ("particle", "scratch")),
                ("train", ("scratch",)),
            ],
        )

        torch.testing.assert_close(
            derive_train_positive_class_weights(bundle),
            torch.tensor([1.5, 1.0]),
        )


def _bundle(
    class_codes: tuple[str, ...],
    rows: list[tuple[str, tuple[str, ...]]],
) -> TrainingInputBundle:
    sources = tuple(
        TrainingBundleSource(f"wafer-{index}", split, "unused", "unused", "uint8")
        for index, (split, _) in enumerate(rows)
    )
    bags = tuple(
        TrainingPatchBag(
            f"bag-{index}",
            source.image_asset_id,
            0,
            0,
            (Rect(0, 0, 1, 1),),
            labels,
        )
        for index, (source, (_, labels)) in enumerate(zip(sources, rows))
    )
    return TrainingInputBundle("snapshot", "split", class_codes, (), sources, (), 2, bags)


if __name__ == "__main__":
    unittest.main()
