import unittest

import torch

from wafer_defect_studio.training_augmentation import (
    AugmentationConfig,
    apply_augmentation,
    apply_spatial_mil_v4_augmentation,
    preview_transformed_sample,
)


class TrainingAugmentationTest(unittest.TestCase):
    def test_spatial_mil_v4_affine_is_seeded_without_touching_global_rng(self):
        sample = torch.tensor([[[0.2, 0.8]]], dtype=torch.float32)
        torch.manual_seed(41)
        expected_global = torch.rand(3)
        torch.manual_seed(41)

        first = apply_spatial_mil_v4_augmentation(sample, effective_seed=17)
        second = apply_spatial_mil_v4_augmentation(sample, effective_seed=17)

        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(torch.rand(3), expected_global))
        self.assertNotEqual(torch.sign(first[0, 0, 1] - first[0, 0, 0]).item(), 0)
        self.assertEqual(
            torch.sign(first[0, 0, 1] - first[0, 0, 0]).item(),
            torch.sign(sample[0, 0, 1] - sample[0, 0, 0]).item(),
        )
        self.assertFalse(torch.equal(first, apply_spatial_mil_v4_augmentation(sample, effective_seed=18)))

    def test_default_is_identity_and_explicit_preview_is_seeded(self):
        sample = torch.tensor(
            [
                [[0.0, 0.25], [0.5, 0.75]],
                [[0.0, 0.25], [0.5, 0.75]],
                [[0.0, 0.25], [0.5, 0.75]],
            ],
            dtype=torch.float32,
        )

        default = apply_augmentation(sample)
        self.assertTrue(torch.equal(default, sample))

        config = AugmentationConfig(horizontal_flip=True, noise_std=0.1, seed=7)
        first = preview_transformed_sample(sample, config)
        second = preview_transformed_sample(sample, config)
        self.assertEqual(first.input_shape, (3, 2, 2))
        self.assertEqual(first.output_shape, (3, 2, 2))
        self.assertEqual(first.config, config)
        self.assertTrue(torch.equal(first.transformed, second.transformed))
        self.assertFalse(torch.equal(first.transformed, default))


if __name__ == "__main__":
    unittest.main()
