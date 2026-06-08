import unittest

import torch
import torch.nn.functional as F

from wafer_defect_studio.detection_windows import Rect
from wafer_defect_studio.spatial_mil import (
    absent_class_hard_negative_loss,
    dense_absent_class_loss,
    derive_train_positive_class_weights,
    far_negative_suppression_loss,
    negative_ring_suppression_loss,
    overlap_consistency_loss,
    per_instance_coverage_loss,
    positive_spatial_mil_loss,
    positive_spatial_topk_loss,
    same_image_grid_ranking_loss,
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


class GridContrastiveV6LossTest(unittest.TestCase):
    def test_localized_positive_is_enough_while_absent_whole_map_is_penalized(self) -> None:
        localized = torch.full((1, 1, 10, 10), -5.0)
        localized[0, 0, 4, 7] = 5.0
        whole_map = torch.full((1, 1, 10, 10), 5.0)

        localized_positive = positive_spatial_topk_loss(
            localized, torch.tensor([[1]]), top_fraction=0.01
        )
        whole_map_positive = positive_spatial_topk_loss(
            whole_map, torch.tensor([[1]]), top_fraction=0.01
        )
        torch.testing.assert_close(localized_positive, F.softplus(torch.tensor(-5.0)))
        torch.testing.assert_close(localized_positive, whole_map_positive)
        self.assertGreater(
            dense_absent_class_loss(whole_map, torch.tensor([[0]])).item(),
            dense_absent_class_loss(localized, torch.tensor([[0]])).item(),
        )

    def test_ranking_uses_only_normal_grids_from_the_same_wafer_image(self) -> None:
        logits = torch.tensor([2.0, 1.75, 9.0]).reshape(3, 1, 1, 1, 1)
        targets = torch.tensor([[1], [0], [0]])

        loss = same_image_grid_ranking_loss(
            logits, targets, ("wafer-a", "wafer-a", "wafer-b"), margin=0.5
        )
        torch.testing.assert_close(loss, torch.tensor(0.25))

        separated = logits.clone()
        separated[1] = 1.0
        torch.testing.assert_close(
            same_image_grid_ranking_loss(
                separated, targets, ("wafer-a", "wafer-a", "wafer-b"), margin=0.5
            ),
            torch.tensor(0.0),
        )

    def test_ranking_does_not_treat_another_class_assertion_as_a_normal_grid(self) -> None:
        logits = torch.tensor([
            [[[[2.0]], [[0.0]]]],
            [[[[9.0]], [[2.0]]]],
            [[[[1.0]], [[1.0]]]],
        ])
        targets = torch.tensor([[1, 0], [0, 1], [0, 0]])

        torch.testing.assert_close(
            same_image_grid_ranking_loss(
                logits, targets, ("wafer-a",) * 3, margin=0.5
            ),
            torch.tensor(0.0),
        )


class InstanceAwareSpatialLossTest(unittest.TestCase):
    def test_each_instance_requires_a_local_response(self) -> None:
        masks = torch.zeros(2, 4, 4, dtype=torch.bool)
        masks[0, 0, 0] = True
        masks[1, 3, 3] = True
        batch_indices = torch.zeros(2, dtype=torch.long)
        class_indices = torch.zeros(2, dtype=torch.long)

        one_covered = torch.full((1, 1, 4, 4), -4.0)
        one_covered[0, 0, 0, 0] = 4.0
        one_covered.requires_grad_()
        one_loss = per_instance_coverage_loss(
            one_covered, masks, batch_indices, class_indices
        )
        one_loss.backward()

        both_covered = torch.full((1, 1, 4, 4), -4.0)
        both_covered[0, 0, 0, 0] = 4.0
        both_covered[0, 0, 3, 3] = 4.0
        both_loss = per_instance_coverage_loss(
            both_covered, masks, batch_indices, class_indices
        )

        self.assertGreater(one_loss.item(), both_loss.item())
        self.assertLess(one_covered.grad[0, 0, 3, 3].item(), 0.0)

    def test_negative_ring_penalizes_expansion_and_pushes_down(self) -> None:
        ring_masks = torch.zeros(1, 4, 4, dtype=torch.bool)
        ring_masks[0, 1, 2] = True
        batch_indices = torch.zeros(1, dtype=torch.long)
        class_indices = torch.zeros(1, dtype=torch.long)

        high_ring = torch.full((1, 1, 4, 4), -5.0)
        high_ring[0, 0, 1, 2] = 5.0
        high_ring.requires_grad_()
        high_loss = negative_ring_suppression_loss(
            high_ring, ring_masks, batch_indices, class_indices
        )
        high_loss.backward()

        low_ring = torch.full((1, 1, 4, 4), -5.0)
        low_loss = negative_ring_suppression_loss(
            low_ring, ring_masks, batch_indices, class_indices
        )

        self.assertGreater(high_loss.item(), low_loss.item())
        self.assertGreater(high_ring.grad[0, 0, 1, 2].item(), 0.0)

    def test_no_instances_return_connected_zero(self) -> None:
        logits = torch.randn(1, 2, 4, 4, requires_grad=True)
        masks = torch.zeros(0, 4, 4, dtype=torch.bool)
        batch_indices = torch.zeros(0, dtype=torch.long)
        class_indices = torch.zeros(0, dtype=torch.long)

        loss = per_instance_coverage_loss(
            logits, masks, batch_indices, class_indices
        ) + negative_ring_suppression_loss(
            logits, masks, batch_indices, class_indices
        )
        loss.backward()

        self.assertEqual(0.0, loss.item())
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_instance_supervision_rejects_invalid_shape_index_and_masks(self) -> None:
        logits = torch.zeros(1, 1, 2, 2)
        empty_indices = torch.zeros(0, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "instance_masks.*shape"):
            per_instance_coverage_loss(
                logits,
                torch.zeros(1, 2, dtype=torch.bool),
                torch.zeros(1, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
            )
        with self.assertRaisesRegex(ValueError, "mask.*non-empty"):
            per_instance_coverage_loss(
                logits,
                torch.zeros(1, 2, 2, dtype=torch.bool),
                torch.zeros(1, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
            )
        with self.assertRaisesRegex(ValueError, "batch index"):
            per_instance_coverage_loss(
                logits,
                torch.ones(1, 2, 2, dtype=torch.bool),
                torch.ones(1, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
            )
        with self.assertRaisesRegex(ValueError, "ring_masks.*non-empty"):
            negative_ring_suppression_loss(
                logits,
                torch.zeros(1, 2, 2, dtype=torch.bool),
                torch.zeros(1, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
            )
        with self.assertRaisesRegex(ValueError, "batch.*integer"):
            per_instance_coverage_loss(
                logits,
                torch.ones(1, 2, 2, dtype=torch.bool),
                torch.zeros(1, dtype=torch.int16),
                torch.zeros(1, dtype=torch.long),
            )
        with self.assertRaisesRegex(ValueError, "class index"):
            negative_ring_suppression_loss(
                logits,
                torch.ones(1, 2, 2, dtype=torch.bool),
                torch.zeros(1, dtype=torch.long),
                torch.ones(1, dtype=torch.long),
            )
        with self.assertRaisesRegex(ValueError, "boolean"):
            negative_ring_suppression_loss(
                logits,
                torch.ones(1, 2, 2),
                empty_indices,
                empty_indices,
            )

    def test_far_negative_targets_only_hardest_declared_cells(self) -> None:
        logits = torch.tensor(
            [[[[3.0, 7.0, 1.0], [5.0, -2.0, 4.0]]]], requires_grad=True
        )
        far_negative_masks = torch.tensor(
            [[[False, True, True], [True, False, False]]], dtype=torch.bool
        )
        batch_indices = torch.zeros(1, dtype=torch.long)
        class_indices = torch.zeros(1, dtype=torch.long)

        loss = far_negative_suppression_loss(
            logits,
            far_negative_masks,
            batch_indices,
            class_indices,
            hardest_fraction=0.01,
        )

        torch.testing.assert_close(loss, F.softplus(torch.tensor(7.0)))
        loss.backward()

        self.assertGreater(logits.grad[0, 0, 0, 1].item(), 0.0)
        self.assertTrue(torch.equal(
            logits.grad[0, 0] * torch.tensor(
                [[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
            ),
            torch.zeros(2, 3),
        ))
        self.assertEqual(0.0, logits.grad[0, 0, 0, 0].item())
        self.assertEqual(0.0, logits.grad[0, 0, 1, 1].item())

    def test_far_negative_normal_grid_and_empty_rows_are_finite_and_connected(self) -> None:
        normal_logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
        normal_mask = torch.ones(1, 2, 2, dtype=torch.bool)
        normal_loss = far_negative_suppression_loss(
            normal_logits,
            normal_mask,
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int64),
        )
        self.assertTrue(torch.isfinite(normal_loss).item())

        empty_logits = torch.randn(1, 1, 2, 2, requires_grad=True)
        empty_loss = far_negative_suppression_loss(
            empty_logits,
            torch.zeros(0, 2, 2, dtype=torch.bool),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
        )
        empty_loss.backward()
        self.assertEqual(0.0, empty_loss.item())
        self.assertTrue(torch.equal(empty_logits.grad, torch.zeros_like(empty_logits)))

    def test_far_negative_rejects_invalid_fraction_and_mask_contract(self) -> None:
        logits = torch.zeros(1, 1, 2, 2)
        mask = torch.ones(1, 2, 2, dtype=torch.bool)
        indices = torch.zeros(1, dtype=torch.long)

        for fraction in (True, 0.0, float("nan"), float("inf"), "0.5"):
            with self.subTest(fraction=fraction), self.assertRaisesRegex(
                ValueError, "hardest_fraction"
            ):
                far_negative_suppression_loss(
                    logits, mask, indices, indices, hardest_fraction=fraction
                )

        with self.assertRaisesRegex(ValueError, "far_negative_masks.*shape"):
            far_negative_suppression_loss(
                logits,
                torch.ones(1, 2, dtype=torch.bool),
                indices,
                indices,
            )
        with self.assertRaisesRegex(ValueError, "far_negative_masks.*boolean"):
            far_negative_suppression_loss(
                logits,
                torch.ones(1, 2, 2),
                indices,
                indices,
            )
        with self.assertRaisesRegex(ValueError, "far_negative_masks.*non-empty"):
            far_negative_suppression_loss(
                logits,
                torch.zeros(1, 2, 2, dtype=torch.bool),
                indices,
                indices,
            )


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
