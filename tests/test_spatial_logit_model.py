import unittest

import torch

from wafer_defect_studio.model_registry import (
    ResNet18Classifier,
    create_resnet18,
    create_resnet18_spatial_logits,
)


class SpatialLogitModelTest(unittest.TestCase):
    def test_produces_stride_four_multiscale_raw_logits(self) -> None:
        model = create_resnet18_spatial_logits(3, device="cpu")
        captured: dict[str, torch.Tensor] = {}
        handles = [
            model.backbone.layer1.register_forward_hook(
                lambda _module, _inputs, output: captured.__setitem__("layer1", output)
            ),
            model.backbone.layer4.register_forward_hook(
                lambda _module, _inputs, output: captured.__setitem__("layer4", output)
            ),
        ]
        try:
            output = model(torch.randn(2, 1, 64, 80))
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual((2, 3, 16, 20), tuple(output.shape))
        self.assertEqual(960, model.spatial_head.in_channels)
        self.assertEqual((2, 64, 16, 20), tuple(captured["layer1"].shape))
        self.assertEqual((2, 512, 2, 3), tuple(captured["layer4"].shape))

    def test_accepts_one_or_three_channels_and_backpropagates_all_scales(self) -> None:
        model = create_resnet18_spatial_logits(1, device="cpu")
        grayscale_logits = model(torch.randn(1, 1, 64, 80))
        rgb_logits = model(torch.randn(1, 3, 64, 80))
        self.assertEqual(tuple(grayscale_logits.shape), tuple(rgb_logits.shape))

        grayscale_logits.sum().backward()
        self.assertIsNotNone(model.backbone.layer1[0].conv1.weight.grad)
        self.assertIsNotNone(model.backbone.layer4[0].conv1.weight.grad)
        self.assertIsNotNone(model.spatial_head.weight.grad)

    def test_existing_classifier_creator_remains_unchanged(self) -> None:
        classifier = create_resnet18(2, device="cpu")
        self.assertIsInstance(classifier, ResNet18Classifier)
        self.assertEqual((1, 2), tuple(classifier(torch.randn(1, 1, 64, 80)).shape))


if __name__ == "__main__":
    unittest.main()
