import unittest

from wafer_defect_studio.training_patch_dataset import EqualShapeBatchSampler


class TrainingShapeBatchSamplerTest(unittest.TestCase):
    def test_batches_each_shape_once_with_seeded_train_or_sequential_validation_order(self):
        shape_keys = (
            (4, 3, 128, 128),
            (9, 3, 128, 128),
            (4, 3, 128, 128),
            (9, 3, 128, 128),
            (4, 3, 128, 128),
        )

        validation_sampler = EqualShapeBatchSampler(
            shape_keys,
            batch_size=2,
            seed=7,
            shuffle=False,
        )
        validation = tuple(tuple(batch) for batch in validation_sampler)
        self.assertEqual(validation, ((0, 2), (1, 3), (4,)))
        self.assertEqual(len(validation_sampler), 3)

        first_train = tuple(
            tuple(batch)
            for batch in EqualShapeBatchSampler(
                shape_keys,
                batch_size=2,
                seed=7,
                shuffle=True,
            )
        )
        second_train = tuple(
            tuple(batch)
            for batch in EqualShapeBatchSampler(
                shape_keys,
                batch_size=2,
                seed=7,
                shuffle=True,
            )
        )
        self.assertEqual(first_train, second_train)
        self.assertEqual(sorted(index for batch in first_train for index in batch), list(range(5)))
        self.assertTrue(all(
            len({shape_keys[index] for index in batch}) == 1
            for batch in first_train
        ))
        self.assertEqual(len(first_train), 3)


if __name__ == "__main__":
    unittest.main()
