import unittest

import torch

from wafer_defect_studio.training_worker import max_pool_patch_logits


class PatchBagPoolingTest(unittest.TestCase):
    def test_per_class_max_preserves_logits_and_routes_gradients(self):
        patch_logits = torch.tensor(
            (
                ((1.0, 4.0), (3.0, 2.0), (2.0, 1.0)),
                ((5.0, 0.5), (1.0, 3.5), (2.0, 2.0)),
            ),
            requires_grad=True,
        )

        pooled = max_pool_patch_logits(patch_logits)

        torch.testing.assert_close(
            pooled,
            torch.tensor(((3.0, 4.0), (5.0, 3.5))),
        )
        pooled.sum().backward()
        torch.testing.assert_close(
            patch_logits.grad,
            torch.tensor(
                (
                    ((0.0, 1.0), (1.0, 0.0), (0.0, 0.0)),
                    ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)),
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
