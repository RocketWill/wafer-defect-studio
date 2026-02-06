import unittest

import numpy as np
import torch

from wafer_defect_studio.normalization import NormalizationBounds
from wafer_defect_studio.training_dataset import extract_model_patch


class TrainingDatasetTest(unittest.TestCase):
    def test_native_patch_is_stable_reflect_padded_normalized_and_three_channel(self):
        source8 = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        before8 = source8.copy()
        patch8 = extract_model_patch(
            source8,
            NormalizationBounds("uint8", 0, 255, 10.0, 30.0, 25.0, 75.0),
            top=-1,
            left=-1,
            size=4,
        )

        expected8 = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.5, 0.0, 0.5, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.5, 0.0, 0.5, 0.0],
            ],
            dtype=torch.float32,
        )
        self.assertEqual(patch8.shape, (3, 4, 4))
        self.assertTrue(torch.equal(patch8[0], expected8))
        self.assertTrue(torch.equal(patch8[0], patch8[1]))
        self.assertTrue(torch.equal(patch8[1], patch8[2]))
        np.testing.assert_array_equal(source8, before8)

        source16 = torch.tensor([[1000, 2000], [3000, 4000]], dtype=torch.uint16)
        patch16 = extract_model_patch(
            source16,
            NormalizationBounds("uint16", 0, 65535, 1000.0, 3000.0, 25.0, 75.0),
            top=0,
            left=0,
            size=2,
        )
        expected16 = torch.tensor([[0.0, 0.5], [1.0, 1.0]], dtype=torch.float32)
        self.assertEqual(patch16.shape, (3, 2, 2))
        self.assertTrue(torch.equal(patch16[0], expected16))
        self.assertTrue(torch.equal(patch16[0], patch16[2]))
        self.assertEqual(patch16.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
