import hashlib
import json
import multiprocessing as mp
import queue
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from wafer_defect_studio.detection_worker import (
    DetectionProgress,
    DetectionRequest,
    DetectionTerminal,
    decode_message,
    run_detection_worker,
    start_detection_worker,
    validate_staged_detection_artifacts,
)
from wafer_defect_studio.model_registry import create_resnet18


class DetectionWorkerTest(unittest.TestCase):
    def test_checkpoint_detection_stages_all_convolutional_sigmoid_maps(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = np.zeros((32, 32), dtype=np.uint8)
            source[:, 16:] = 255
            model = create_resnet18(1, device="cpu")
            checkpoint_path = root / "model.pt"
            torch.save(
                {
                    "checkpoint_format": "wafer_defect_studio.resnet18.v1",
                    "architecture": "resnet18",
                    "class_count": 1,
                    "class_codes": ["edge"],
                    "normalization_bounds": [
                        {
                            "dtype": "uint8",
                            "source_min": 0,
                            "source_max": 255,
                            "low": 0.0,
                            "high": 255.0,
                            "low_percentile": 1.0,
                            "high_percentile": 99.0,
                        }
                    ],
                    "input_size": {"width": 32, "height": 32},
                    "state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
            request = DetectionRequest(
                request_id="checkpoint-detection-1",
                source=source,
                run_id="run-1",
                profile_id="profile-1",
                approved_evaluation_id="evaluation-1",
                class_names=("edge",),
                window_size=(32, 32),
                stride=(32, 32),
                staging_path=root / "checkpoint-stage",
                checkpoint_path=checkpoint_path,
                model_id="checkpoint-model",
                batch_size=1,
            )
            output = queue.Queue()
            run_detection_worker(request, output, threading.Event())
            while True:
                message = decode_message(output.get_nowait())
                if isinstance(message, DetectionTerminal):
                    self.assertEqual(message.status, "completed", message.message)
                    break
            maps = json.loads((root / "checkpoint-stage" / "maps.json").read_text(encoding="utf-8"))
            provenance = json.loads(
                (root / "checkpoint-stage" / "provenance.json").read_text(encoding="utf-8")
            )
            self.assertEqual(maps["window_settings"]["map_method"], "all_convolutional_sigmoid")
            self.assertEqual(provenance["model_id"], "checkpoint-model")

    def _collect(self, handle):
        messages = []
        while True:
            message = decode_message(handle.queue.get(timeout=30))
            messages.append(message)
            if isinstance(message, DetectionTerminal):
                handle.join(timeout=30)
                self.assertFalse(handle.is_alive())
                return messages

    def test_native_source_maps_stage_with_provenance_and_cancel_without_publish(self):
        self.assertEqual(mp.get_context("spawn").get_start_method(), "spawn")
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = np.zeros((8, 8), dtype=np.uint8)
            source[:, 4:] = 255
            request = DetectionRequest(
                request_id="detection-request-1",
                source=source,
                run_id="run-approved",
                profile_id="profile-approved",
                class_names=("edge",),
                window_size=(4, 4),
                stride=(2, 2),
                device="cpu",
                staging_path=root / "completed",
            )
            completed = start_detection_worker(request)
            messages = self._collect(completed)
            self.assertTrue(any(isinstance(item, DetectionProgress) for item in messages))
            self.assertEqual(messages[-1].status, "completed", messages[-1].message)

            stage = root / "completed"
            self.assertTrue((stage / "maps.json").is_file())
            self.assertTrue((stage / "provenance.json").is_file())
            self.assertTrue((stage / "manifest.json").is_file())
            validated = validate_staged_detection_artifacts(stage)
            self.assertEqual({path.name for path in validated}, {"maps.json", "provenance.json"})
            maps = json.loads((stage / "maps.json").read_text(encoding="utf-8"))
            provenance = json.loads((stage / "provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(maps["map_shape"], [8, 8, 1])
            self.assertEqual(maps["source"]["coordinate_system"], "source-image pixels")
            self.assertEqual(provenance["run_id"], "run-approved")
            self.assertEqual(provenance["profile_id"], "profile-approved")
            self.assertEqual(provenance["device"], "cpu")
            self.assertIn("Approximate localization", maps["disclaimers"])
            manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
            for item in manifest["files"]:
                digest = hashlib.sha256((stage / item["path"]).read_bytes()).hexdigest()
                self.assertEqual(digest, item["sha256"])

            cancelled_request = DetectionRequest(
                request_id="detection-request-2",
                source=source,
                run_id="run-approved",
                profile_id="profile-approved",
                class_names=("edge",),
                window_size=(2, 2),
                stride=(1, 1),
                device="cpu",
                staging_path=root / "cancelled",
            )
            cancelled = start_detection_worker(cancelled_request, step_delay=0.1)
            cancelled.cancel()
            cancelled_messages = self._collect(cancelled)
            self.assertEqual(cancelled_messages[-1].status, "cancelled")
            self.assertFalse((root / "cancelled" / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
