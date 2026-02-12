import hashlib
import json
import multiprocessing as mp
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from wafer_defect_studio.detection_worker import (
    DetectionProgress,
    DetectionRequest,
    DetectionTerminal,
    decode_message,
    start_detection_worker,
    validate_staged_detection_artifacts,
)


class DetectionWorkerTest(unittest.TestCase):
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
