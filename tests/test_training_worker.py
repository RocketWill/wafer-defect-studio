import json
import hashlib
import multiprocessing as mp
import queue
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PySide6.QtGui import QImage
import torch

from wafer_defect_studio.dataset_snapshot import SnapshotSample
from wafer_defect_studio.detection_windows import Rect
from wafer_defect_studio.model_registry import create_resnet18
from wafer_defect_studio.normalization import NormalizationBounds
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
from wafer_defect_studio.training_patch_dataset import TrainingPatchDataset


class TrainingWorkerTest(unittest.TestCase):
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
