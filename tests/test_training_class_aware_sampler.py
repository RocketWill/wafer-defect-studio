import unittest

import torch

from wafer_defect_studio.training_patch_dataset import (
    ClassAwareEqualShapeBatchSampler,
    TrainingPatchDatasetError,
)


class TrainingClassAwareSamplerTest(unittest.TestCase):
    def test_balances_class_and_normal_groups_with_equal_shape_batches(self):
        shapes = ((4,), (4,), (9,), (9,), (4,), (4,))
        targets = ((1, 0), (1, 0), (0, 1), (0, 1), (0, 0), (0, 0))
        sampler = ClassAwareEqualShapeBatchSampler(
            shapes, targets, batch_size=2, seed=17, epoch_size=12
        )

        batches = tuple(tuple(batch) for batch in sampler)
        flattened = tuple(index for batch in batches for index in batch)

        self.assertEqual(len(flattened), 12)
        self.assertEqual(len(sampler), len(batches))
        self.assertTrue(all(len({shapes[index] for index in batch}) == 1 for batch in batches))
        group_counts = (
            sum(index in (0, 1) for index in flattened),
            sum(index in (2, 3) for index in flattened),
            sum(index in (4, 5) for index in flattened),
        )
        self.assertEqual(group_counts, (4, 4, 4))

    def test_multi_label_bag_enters_each_matching_class_pool(self):
        sampler = ClassAwareEqualShapeBatchSampler(
            ((4,), (4,)), ((1, 1), (0, 0)), batch_size=2, seed=5, epoch_size=6
        )

        flattened = tuple(index for batch in sampler for index in batch)

        self.assertEqual(flattened.count(0), 2 * flattened.count(1))

    def test_epoch_is_deterministic_without_changing_global_rng(self):
        shapes = ((4,), (4,), (4,), (9,), (9,), (9,))
        targets = ((1,), (1,), (1,), (0,), (0,), (0,))
        sampler = ClassAwareEqualShapeBatchSampler(
            shapes, targets, batch_size=2, seed=3, epoch_size=8
        )
        torch.manual_seed(91)
        expected = torch.rand(3)
        torch.manual_seed(91)

        epoch_zero = tuple(tuple(batch) for batch in sampler)
        self.assertTrue(torch.equal(torch.rand(3), expected))
        self.assertEqual(epoch_zero, tuple(tuple(batch) for batch in sampler))
        sampler.set_epoch(1)
        self.assertNotEqual(epoch_zero, tuple(tuple(batch) for batch in sampler))
        self.assertEqual(sum(map(len, epoch_zero)), 8)

    def test_reports_the_empty_class_index(self):
        with self.assertRaisesRegex(TrainingPatchDatasetError, "class 1 has no positive samples"):
            ClassAwareEqualShapeBatchSampler(
                ((4,), (4,)), ((1, 0), (0, 0)), batch_size=1, seed=0
            )


if __name__ == "__main__":
    unittest.main()
