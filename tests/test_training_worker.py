import json
import hashlib
import multiprocessing as mp
import queue
import threading
import unittest
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory

from PySide6.QtGui import QImage
import torch

from wafer_defect_studio.dataset_snapshot import SnapshotSample
from wafer_defect_studio.detection_windows import Rect
from wafer_defect_studio.model_registry import create_resnet18, create_resnet18_spatial_logits
from wafer_defect_studio.normalization import NormalizationBounds
from wafer_defect_studio.spatial_mil import (
    absent_class_hard_negative_loss,
    overlap_consistency_loss,
    positive_spatial_mil_loss,
)
from wafer_defect_studio.training_protocol import (
    ProgressMessage,
    TerminalMessage,
    TrainingConfig,
    TrainingRequest,
    decode_message,
)
from wafer_defect_studio.training_run import validate_project_checkpoint, validate_staged_artifacts
from wafer_defect_studio.training_worker import (
    create_training_data_loader,
    run_worker,
    start_training_worker,
)
from wafer_defect_studio.training_input_bundle import (
    TrainingBundleSource,
    TrainingInputBundle,
    TrainingPatchBag,
)
from wafer_defect_studio.training_patch_dataset import (
    ClassAwareEqualShapeBatchSampler,
    TrainingPatchDataset,
)


class TrainingWorkerTest(unittest.TestCase):
    def test_v4_worker_rejects_unknown_priority_normal_bag_id_with_train_context(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.png"
            image = QImage(32, 32, QImage.Format.Format_Grayscale8)
            self.assertTrue(image.save(str(source), "PNG"))
            bundle = TrainingInputBundle(
                "snapshot", "split", ("scratch",),
                (NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 1.0, 99.0),),
                (TrainingBundleSource("wafer", "train", str(source), _hash(source), "uint8"),),
                (), 2,
                (
                    TrainingPatchBag("positive", "wafer", 0, 0, (Rect(0, 0, 32, 32),), ("scratch",)),
                    TrainingPatchBag("normal", "wafer", 0, 1, (Rect(0, 0, 32, 32),), ()),
                ),
            )
            bundle_path = root / "bundle.json"
            bundle_path.write_text(bundle.to_json(), encoding="utf-8")
            output = queue.Queue()
            run_worker(
                TrainingRequest(
                    request_id="unknown-priority",
                    config=TrainingConfig(
                        snapshot_id="snapshot", split_id="split", class_count=1,
                        epochs=1, batch_size=1, device="cpu", patch_size=32,
                        patch_stride=32, training_policy="spatial_mil_v4",
                        priority_normal_bag_ids=("not-in-train",),
                        hard_negative_selection_sha256="b" * 64,
                    ),
                    artifact_staging_path=root / "staging",
                    input_bundle_path=bundle_path,
                ),
                output,
                threading.Event(),
            )
            terminal = decode_message(output.get_nowait())
            self.assertEqual(terminal.status, "failed")
            self.assertIn("absent from the train split", terminal.message)
            self.assertIn("not-in-train", terminal.message)

    def test_v4_worker_rejects_missing_training_input_bundle(self):
        with TemporaryDirectory() as temporary_directory:
            output = queue.Queue()
            run_worker(
                TrainingRequest(
                    request_id="missing-v4-bundle",
                    config=TrainingConfig(
                        snapshot_id="snapshot",
                        split_id="split",
                        class_count=1,
                        epochs=1,
                        batch_size=1,
                        device="cpu",
                        patch_size=32,
                        patch_stride=32,
                        training_policy="spatial_mil_v4",
                    ),
                    artifact_staging_path=Path(temporary_directory) / "staging",
                ),
                output,
                threading.Event(),
            )
            terminal = decode_message(output.get_nowait())
            self.assertEqual(terminal.status, "failed")
            self.assertIn("spatial_mil_v4 requires a v2 Training Input Bundle", terminal.message)

    def test_v4_worker_sums_three_unit_weight_losses(self):
        class FakeSpatialModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, inputs):
                return self.value.expand(inputs.shape[0], 2, 8, 8)

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.png"
            image = QImage(48, 32, QImage.Format.Format_Grayscale8)
            self.assertTrue(image.save(str(source), "PNG"))
            rects = (Rect(0, 0, 32, 32), Rect(16, 0, 32, 32))
            bundle = TrainingInputBundle(
                "snapshot", "split", ("scratch", "particle"),
                (NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 1.0, 99.0),),
                (TrainingBundleSource("wafer", "train", str(source), _hash(source), "uint8"),),
                (), 2,
                tuple(
                    TrainingPatchBag(f"bag-{index}", "wafer", 0, index, rects, labels)
                    for index, labels in enumerate((("scratch",), ("particle",), ()))
                ),
            )
            bundle_path = root / "bundle.json"
            bundle_path.write_text(bundle.to_json(), encoding="utf-8")
            output = queue.Queue()
            sampler_epochs = []
            dataset_epochs = []
            original_sampler_set_epoch = ClassAwareEqualShapeBatchSampler.set_epoch
            original_dataset_set_epoch = TrainingPatchDataset.set_epoch
            original_sgd = torch.optim.SGD
            original_loader_factory = create_training_data_loader

            def record_sampler_epoch(sampler, epoch):
                sampler_epochs.append(epoch)
                return original_sampler_set_epoch(sampler, epoch)

            def record_dataset_epoch(dataset, epoch):
                dataset_epochs.append(epoch)
                return original_dataset_set_epoch(dataset, epoch)

            def connected(value):
                return lambda logits, *_args, **_kwargs: logits.sum() * 0.0 + value

            with mock.patch(
                "wafer_defect_studio.training_worker.create_resnet18_spatial_logits",
                return_value=FakeSpatialModel(),
            ), mock.patch(
                "wafer_defect_studio.training_worker.create_training_data_loader",
                wraps=original_loader_factory,
            ) as loader_factory, mock.patch(
                "wafer_defect_studio.training_worker.torch.optim.SGD",
                wraps=original_sgd,
            ) as optimizer_factory, mock.patch.object(
                ClassAwareEqualShapeBatchSampler, "set_epoch", record_sampler_epoch,
            ), mock.patch.object(
                TrainingPatchDataset, "set_epoch", record_dataset_epoch,
            ), mock.patch(
                "wafer_defect_studio.training_worker.positive_spatial_mil_loss",
                side_effect=connected(1.0),
            ), mock.patch(
                "wafer_defect_studio.training_worker.absent_class_hard_negative_loss",
                side_effect=connected(2.0),
            ), mock.patch(
                "wafer_defect_studio.training_worker.overlap_consistency_loss",
                side_effect=connected(4.0),
            ):
                run_worker(
                    TrainingRequest(
                        request_id="loss-sum",
                        config=TrainingConfig(
                            snapshot_id="snapshot", split_id="split", class_count=2,
                            epochs=1, batch_size=1, device="cpu", patch_size=32,
                            patch_stride=16, training_policy="spatial_mil_v4",
                            priority_normal_bag_ids=("bag-2",),
                            hard_negative_selection_sha256="a" * 64,
                        ),
                        artifact_staging_path=root / "staging",
                        input_bundle_path=bundle_path,
                    ),
                    output,
                    threading.Event(),
                )
            messages = []
            while True:
                message = decode_message(output.get_nowait())
                messages.append(message)
                if isinstance(message, TerminalMessage):
                    break
            self.assertEqual(messages[-1].status, "completed", messages[-1].message)
            self.assertEqual(
                [message.loss for message in messages if isinstance(message, ProgressMessage)],
                [7.0] * 18,
            )
            progress = [message for message in messages if isinstance(message, ProgressMessage)]
            self.assertEqual(progress[-1].total_steps, 18)
            self.assertEqual(progress[-1].total_epochs, 6)
            self.assertEqual(sampler_epochs, list(range(6)))
            self.assertEqual(dataset_epochs, list(range(6)))
            self.assertEqual(optimizer_factory.call_count, 1)
            self.assertEqual(loader_factory.call_count, 2)
            self.assertNotIn("priority_normal_indices", loader_factory.call_args_list[0].kwargs)
            self.assertEqual(
                loader_factory.call_args_list[1].kwargs["priority_normal_indices"],
                (2,),
            )
            checkpoint = validate_project_checkpoint(root / "staging" / "model.pt")
            self.assertEqual(checkpoint["hard_negative_refinement"], {
                "selection_sha256": "a" * 64,
                "priority_normal_bag_ids": ["bag-2"],
                "base_epochs": 1,
                "refinement_epochs": 5,
            })
            metrics = json.loads((root / "staging" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["architecture"], "resnet18_spatial_logits")
            self.assertEqual((metrics["epochs"], metrics["base_epochs"], metrics["refinement_epochs"]), (6, 1, 5))

    def test_v4_worker_rejects_v1_bundle_with_policy_and_version_context(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.png"
            image = QImage(32, 32, QImage.Format.Format_Grayscale8)
            self.assertTrue(image.save(str(source), "PNG"))
            bundle = TrainingInputBundle(
                "snapshot", "split", ("scratch",),
                (NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 1.0, 99.0),),
                (TrainingBundleSource("wafer", "train", str(source), _hash(source), "uint8"),),
                (SnapshotSample("wafer", 0, 0, 0, 0, 32, 32, ("scratch",)),),
            )
            terminal = self._run_v4_bundle(root, bundle, patch_size=32)
            self.assertEqual(terminal.status, "failed")
            self.assertIn("spatial_mil_v4 requires bundle version 2", terminal.message)
            self.assertIn("received version 1", terminal.message)

    def test_v4_worker_rejects_patch_size_not_divisible_by_feature_stride(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.png"
            image = QImage(30, 30, QImage.Format.Format_Grayscale8)
            self.assertTrue(image.save(str(source), "PNG"))
            bundle = TrainingInputBundle(
                "snapshot", "split", ("scratch",),
                (NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 1.0, 99.0),),
                (TrainingBundleSource("wafer", "train", str(source), _hash(source), "uint8"),),
                (), 2,
                (TrainingPatchBag("bag", "wafer", 0, 0, (Rect(0, 0, 30, 30),), ("scratch",)),),
            )
            terminal = self._run_v4_bundle(root, bundle, patch_size=30)
            self.assertEqual(terminal.status, "failed")
            self.assertIn("spatial_mil_v4 patch_size", terminal.message)
            self.assertIn("divisible by feature stride 4", terminal.message)

    def _run_v4_bundle(self, root, bundle, *, patch_size):
        bundle_path = root / "training_input_bundle.json"
        bundle_path.write_text(bundle.to_json(), encoding="utf-8")
        output = queue.Queue()
        run_worker(
            TrainingRequest(
                request_id="invalid-v4",
                config=TrainingConfig(
                    snapshot_id=bundle.snapshot_id,
                    split_id=bundle.split_id,
                    class_count=len(bundle.class_codes),
                    epochs=1,
                    batch_size=1,
                    device="cpu",
                    patch_size=patch_size,
                    patch_stride=patch_size,
                    training_policy="spatial_mil_v4",
                ),
                artifact_staging_path=root / "staging",
                input_bundle_path=bundle_path,
            ),
            output,
            threading.Event(),
        )
        messages = []
        while True:
            message = decode_message(output.get_nowait())
            messages.append(message)
            if isinstance(message, TerminalMessage):
                return message

    def test_v4_worker_trains_spatial_head_with_all_bag_groups_and_overlap(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "spatial.png"
            image = QImage(48, 32, QImage.Format.Format_Grayscale8)
            self.assertTrue(image.save(str(source), "PNG"))
            patch_rects = (Rect(0, 0, 32, 32), Rect(16, 0, 32, 32))
            labels = (("scratch",), ("particle",), ())
            bundle = TrainingInputBundle(
                "snapshot-v4", "split-v4", ("scratch", "particle"),
                (NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 1.0, 99.0),),
                (TrainingBundleSource("wafer", "train", str(source), _hash(source), "uint8"),),
                (), 2,
                tuple(
                    TrainingPatchBag(f"bag-{index}", "wafer", 0, index, patch_rects, class_codes)
                    for index, class_codes in enumerate(labels)
                ),
            )
            bundle_path = root / "training_input_bundle.json"
            bundle_path.write_text(bundle.to_json(), encoding="utf-8")
            request = TrainingRequest(
                request_id="spatial-worker-test",
                config=TrainingConfig(
                    snapshot_id="snapshot-v4", split_id="split-v4", class_count=2,
                    epochs=1, batch_size=1, device="cpu", seed=7, learning_rate=0.01,
                    patch_size=32, patch_stride=16, training_policy="spatial_mil_v4",
                ),
                artifact_staging_path=root / "staging",
                input_bundle_path=bundle_path,
            )
            output = queue.Queue()

            torch.manual_seed(7)
            epochs = []
            original_set_epoch = ClassAwareEqualShapeBatchSampler.set_epoch
            original_dataset_set_epoch = TrainingPatchDataset.set_epoch
            dataset_epochs = []

            def record_epoch(sampler, epoch):
                epochs.append(epoch)
                return original_set_epoch(sampler, epoch)

            def record_dataset_epoch(dataset, epoch):
                dataset_epochs.append(epoch)
                return original_dataset_set_epoch(dataset, epoch)

            with mock.patch.object(
                ClassAwareEqualShapeBatchSampler, "set_epoch", record_epoch
            ), mock.patch.object(
                TrainingPatchDataset, "set_epoch", record_dataset_epoch
            ), mock.patch(
                "wafer_defect_studio.training_worker.positive_spatial_mil_loss",
                wraps=positive_spatial_mil_loss,
            ) as positive_loss, mock.patch(
                "wafer_defect_studio.training_worker.absent_class_hard_negative_loss",
                wraps=absent_class_hard_negative_loss,
            ) as absent_loss, mock.patch(
                "wafer_defect_studio.training_worker.overlap_consistency_loss",
                wraps=overlap_consistency_loss,
            ) as overlap_loss:
                run_worker(request, output, threading.Event())

            messages = []
            while True:
                message = decode_message(output.get_nowait())
                messages.append(message)
                if isinstance(message, TerminalMessage):
                    break
            self.assertEqual(messages[-1].status, "completed", messages[-1].message)
            progress = [message for message in messages if isinstance(message, ProgressMessage)]
            self.assertEqual([(message.step, message.total_steps) for message in progress], [(1, 3), (2, 3), (3, 3)])
            self.assertEqual(epochs, [0])
            self.assertEqual(dataset_epochs, [0])
            self.assertEqual(positive_loss.call_count, 3)
            self.assertEqual(absent_loss.call_count, 3)
            self.assertEqual(overlap_loss.call_count, 3)
            checkpoint = validate_project_checkpoint(root / "staging" / "model.pt")
            self.assertEqual(checkpoint["checkpoint_format"], "wafer_defect_studio.resnet18.v4")
            self.assertEqual(checkpoint["architecture"], "resnet18_spatial_logits")
            self.assertEqual(checkpoint["feature_stride"], 4)
            self.assertEqual(checkpoint["patch_size"], 32)
            self.assertEqual(checkpoint["patch_stride"], 16)
            self.assertEqual(checkpoint["training_policy"], "spatial_mil_v4")
            self.assertEqual(checkpoint["loss_weights"], {
                "positive_spatial_mil": 1.0,
                "absent_class_hard_negative": 1.0,
                "overlap_consistency": 1.0,
            })
            self.assertEqual(checkpoint["positive_class_weighting"], {
                "formula": "negative_bag_count / positive_bag_count",
                "minimum": 1.0,
                "maximum": 10.0,
            })
            self.assertEqual(checkpoint["augmentation_policy"], {
                "name": "spatial_mil_v4_defect_preserving_affine",
                "contrast": [0.9, 1.1],
                "brightness": [-0.03, 0.03],
                "seed": 7,
                "seed_formula": "run_seed + epoch * 1_000_003 + bag_index",
            })
            metrics = json.loads((root / "staging" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["architecture"], "resnet18_spatial_logits")
            self.assertIn("spatial_head.weight", checkpoint["state_dict"])
            torch.manual_seed(7)
            initial_model = create_resnet18_spatial_logits(2, weights="none", device="cpu")
            self.assertFalse(torch.equal(
                checkpoint["state_dict"]["spatial_head.bias"],
                initial_model.state_dict()["spatial_head.bias"],
            ))

    def test_v2_bundle_training_pools_patch_logits_per_bag(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "bags.png"
            image = QImage(64, 32, QImage.Format.Format_Grayscale8)
            bits = image.bits()
            stride = image.bytesPerLine()
            for row in range(32):
                bits[row * stride : (row + 1) * stride] = bytes(range(64))
            self.assertTrue(image.save(str(source), "PNG"))
            del bits, image
            bundle = TrainingInputBundle(
                "snapshot-1",
                "split-1",
                ("scratch", "particle"),
                (NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 1.0, 99.0),),
                (TrainingBundleSource("wafer-1", "train", str(source), _hash(source), "uint8"),),
                (),
                2,
                (
                    TrainingPatchBag(
                        "bag-0",
                        "wafer-1",
                        0,
                        0,
                        (Rect(0, 0, 32, 32), Rect(32, 0, 32, 32)),
                        ("scratch",),
                    ),
                    TrainingPatchBag(
                        "bag-1",
                        "wafer-1",
                        0,
                        1,
                        (Rect(0, 0, 32, 32), Rect(32, 0, 32, 32)),
                        ("particle",),
                    ),
                ),
            )
            bundle_path = root / "training_input_bundle.json"
            bundle_path.write_text(bundle.to_json(), encoding="utf-8")
            request = TrainingRequest(
                request_id="patch-bag-worker-test",
                config=TrainingConfig(
                    snapshot_id="snapshot-1",
                    split_id="split-1",
                    class_count=2,
                    epochs=1,
                    batch_size=2,
                    device="cpu",
                    seed=23,
                    learning_rate=0.01,
                    patch_size=32,
                    patch_stride=32,
                ),
                artifact_staging_path=root / "staging",
                input_bundle_path=bundle_path,
            )
            torch.manual_seed(23)
            output = queue.Queue()

            run_worker(request, output, threading.Event())

            messages = []
            while True:
                message = decode_message(output.get_nowait())
                messages.append(message)
                if isinstance(message, TerminalMessage):
                    break
            self.assertEqual(messages[-1].status, "completed")
            progress = [item for item in messages if isinstance(item, ProgressMessage)]
            self.assertEqual([(item.step, item.total_steps) for item in progress], [(1, 1)])
            checkpoint = torch.load(
                root / "staging" / "model.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(
                {
                    "checkpoint_format": checkpoint["checkpoint_format"],
                    "feature_stride": checkpoint["feature_stride"],
                    "patch_size": checkpoint["patch_size"],
                    "patch_stride": checkpoint["patch_stride"],
                    "bag_pooling": checkpoint["bag_pooling"],
                },
                {
                    "checkpoint_format": "wafer_defect_studio.resnet18.v3",
                    "feature_stride": 16,
                    "patch_size": 32,
                    "patch_stride": 32,
                    "bag_pooling": "max",
                },
            )
            torch.manual_seed(23)
            initial_model = create_resnet18(2, weights="none", device="cpu")
            self.assertFalse(
                torch.equal(
                    checkpoint["state_dict"]["backbone.fc.weight"],
                    initial_model.state_dict()["backbone.fc.weight"],
                )
            )

    def test_v2_loader_batches_equal_shapes_with_repeatable_membership(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "bags.png"
            image = QImage(3, 2, QImage.Format.Format_Grayscale8)
            bits = image.bits()
            stride = image.bytesPerLine()
            bits[0:stride] = bytes((10, 20, 30)) + bytes(stride - 3)
            bits[stride : 2 * stride] = bytes((40, 50, 60)) + bytes(stride - 3)
            self.assertTrue(image.save(str(source), "PNG"))
            del bits, image
            bundle = TrainingInputBundle(
                "snapshot-1",
                "split-1",
                ("scratch", "particle"),
                (NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 1.0, 99.0),),
                (TrainingBundleSource("wafer-1", "train", str(source), _hash(source), "uint8"),),
                (),
                2,
                (
                    TrainingPatchBag("bag-0", "wafer-1", 0, 0, (Rect(0, 0, 1, 1),), ("scratch",)),
                    TrainingPatchBag("bag-1", "wafer-1", 0, 1, (Rect(1, 0, 1, 1), Rect(2, 0, 1, 1)), ("particle",)),
                    TrainingPatchBag("bag-2", "wafer-1", 1, 0, (Rect(0, 1, 1, 1),), ("scratch", "particle")),
                    TrainingPatchBag("bag-3", "wafer-1", 1, 1, (Rect(1, 1, 1, 1), Rect(2, 1, 1, 1)), ()),
                ),
            )
            config = TrainingConfig(
                snapshot_id="snapshot-1",
                split_id="split-1",
                class_count=2,
                epochs=1,
                batch_size=2,
                seed=17,
                patch_size=1,
                patch_stride=1,
            )

            def collect():
                loader = create_training_data_loader(
                    TrainingPatchDataset(bundle, "train"),
                    config,
                    split="train",
                )
                return tuple((inputs.clone(), targets.clone()) for inputs, targets in loader)

            first = collect()
            second = collect()
            self.assertEqual(
                {tuple(inputs.shape) for inputs, _targets in first},
                {(2, 1, 3, 1, 1), (2, 2, 3, 1, 1)},
            )
            self.assertTrue(all(
                torch.equal(left, right)
                for left_batch, right_batch in zip(first, second, strict=True)
                for left, right in zip(left_batch, right_batch, strict=True)
            ))
            expected = {
                (1.0, 0.0): (10,),
                (0.0, 1.0): (20, 30),
                (1.0, 1.0): (40,),
                (0.0, 0.0): (50, 60),
            }
            observed = {}
            for inputs, targets in first:
                for bag_inputs, target in zip(inputs, targets, strict=True):
                    observed[tuple(target.tolist())] = tuple(
                        round(float(value) * 255)
                        for value in bag_inputs[:, 0, 0, 0]
                    )
            self.assertEqual(observed, expected)

    def test_bundle_training_uses_real_patches_and_writes_checkpoint(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.png"
            image = QImage(32, 32, QImage.Format.Format_Grayscale8)
            bits = image.bits()
            stride = image.bytesPerLine()
            for row in range(32):
                bits[row * stride : (row + 1) * stride] = bytes([row] * 32) + bytes(stride - 32)
            self.assertTrue(image.save(str(source), "PNG"))
            del bits, image
            bundle = TrainingInputBundle(
                "snapshot-1",
                "split-1",
                ("scratch",),
                (NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 1.0, 99.0),),
                (TrainingBundleSource("wafer-1", "train", str(source), _hash(source), "uint8"),),
                (SnapshotSample("wafer-1", 0, 0, 0, 0, 32, 32, ("scratch",)),),
            )
            bundle_path = root / "training_input_bundle.json"
            bundle_path.write_text(bundle.to_json(), encoding="utf-8")
            request = TrainingRequest(
                request_id="real-worker-test",
                config=TrainingConfig(
                    snapshot_id="snapshot-1",
                    split_id="split-1",
                    class_count=1,
                    epochs=1,
                    batch_size=1,
                    device="cpu",
                    learning_rate=0.001,
                ),
                artifact_staging_path=root / "staging",
                input_bundle_path=bundle_path,
            )
            output = queue.Queue()
            run_worker(request, output, threading.Event(), synthetic_steps=1)
            messages = []
            while True:
                message = decode_message(output.get_nowait())
                messages.append(message)
                if isinstance(message, TerminalMessage):
                    break
            self.assertEqual(messages[-1].status, "completed")
            self.assertTrue((root / "staging" / "model.pt").is_file())
            self.assertIn(
                "model.pt",
                {path.name for path in validate_staged_artifacts(root / "staging")},
            )
            checkpoint = validate_project_checkpoint(root / "staging" / "model.pt")
            self.assertEqual(
                checkpoint["checkpoint_format"],
                "wafer_defect_studio.resnet18.v2",
            )
            self.assertEqual(checkpoint["feature_stride"], 16)
            self.assertEqual(checkpoint["class_codes"], ["scratch"])
            self.assertEqual(checkpoint["input_size"], {"width": 32, "height": 32})

    def _request(self, staging: Path, *, batch_size: int = 1) -> TrainingRequest:
        return TrainingRequest(
            request_id="worker-test",
            config=TrainingConfig(
                snapshot_id="snapshot-1",
                split_id="split-1",
                class_count=2,
                epochs=1,
                batch_size=batch_size,
                device="cpu",
                weights_policy="none",
            ),
            artifact_staging_path=staging,
        )

    def _collect(self, handle):
        messages = []
        while True:
            message = decode_message(handle.queue.get(timeout=30))
            messages.append(message)
            if isinstance(message, TerminalMessage):
                handle.join(timeout=30)
                self.assertFalse(handle.is_alive())
                return messages

    def test_spawned_training_reports_progress_cancel_and_oom_truthfully(self):
        # The spawn context is required on Windows and proves the worker does
        # not depend on a GUI thread or a project SQLite connection.
        self.assertEqual(mp.get_context("spawn").get_start_method(), "spawn")
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            completed_request = self._request(root / "completed")
            completed = start_training_worker(
                completed_request,
                synthetic_steps=1,
                step_delay=0.0,
            )
            completed_messages = self._collect(completed)
            progress = [item for item in completed_messages if isinstance(item, ProgressMessage)]
            terminal = completed_messages[-1]
            self.assertTrue(progress)
            self.assertEqual(progress[-1].phase, "train")
            self.assertEqual(progress[-1].epoch, 1)
            self.assertEqual(progress[-1].step, 1)
            self.assertIsNotNone(progress[-1].loss)
            self.assertIn("batch_size=1", progress[-1].message)
            self.assertIsInstance(terminal, TerminalMessage)
            self.assertEqual(terminal.status, "completed")
            self.assertTrue((root / "completed" / "manifest.json").is_file())
            manifest = json.loads((root / "completed" / "manifest.json").read_text())
            self.assertIn("model.pt", {entry["path"] for entry in manifest["files"]})
            validated = validate_staged_artifacts(root / "completed")
            self.assertEqual({path.name for path in validated}, {"model.pt", "metrics.json"})

            cancelled_request = self._request(root / "cancelled")
            cancelled = start_training_worker(
                cancelled_request,
                synthetic_steps=2,
                step_delay=0.2,
            )
            cancelled.cancel()
            cancelled_messages = self._collect(cancelled)
            self.assertEqual(cancelled_messages[-1].status, "cancelled")
            self.assertIn("cancel", cancelled_messages[-1].message.lower())
            self.assertFalse((root / "cancelled" / "manifest.json").exists())

            oom_request = self._request(root / "oom", batch_size=7)
            oom = start_training_worker(
                oom_request,
                synthetic_steps=1,
                inject_oom_step=1,
            )
            oom_messages = self._collect(oom)
            oom_terminal = oom_messages[-1]
            self.assertEqual(oom_terminal.status, "failed")
            self.assertEqual(oom_terminal.error_code, "out_of_memory")
            self.assertIn("batch_size=7", oom_terminal.message)
            self.assertIn("unchanged", oom_terminal.message.lower())
            self.assertFalse((root / "oom" / "manifest.json").exists())


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
