import json
import unittest

from wafer_defect_studio.training_protocol import (
    CancelMessage,
    ProgressMessage,
    TerminalMessage,
    TrainingConfig,
    TrainingProtocolError,
    TrainingRequest,
    decode_message,
    encode_message,
)


class TrainingProtocolTest(unittest.TestCase):
    def test_versioned_messages_round_trip_and_keep_worker_value_only(self):
        config = TrainingConfig(
            snapshot_id="snapshot-1",
            split_id="split-1",
            class_count=2,
            epochs=3,
            batch_size=4,
            device="cuda",
            patch_size=128,
            patch_stride=64,
        )
        request = TrainingRequest(
            request_id="run-1",
            config=config,
            artifact_staging_path="C:/project/staging/run-1",
        )

        encoded = encode_message(request)
        payload = json.loads(encoded)
        self.assertEqual(set(payload), {"version", "type", "payload"})
        self.assertEqual(
            set(payload["payload"]),
            {"request_id", "config", "artifact_staging_path"},
        )
        self.assertNotIn("connection", encoded.lower())
        self.assertNotIn("project_service", encoded.lower())
        self.assertEqual(decode_message(encoded), request)
        self.assertEqual(TrainingConfig.from_json(config.to_json()), config)
        self.assertEqual(config.to_dict()["patch_size"], 128)
        self.assertEqual(config.to_dict()["patch_stride"], 64)
        config.validate_patch_geometry(512, 512)
        legacy_payload = config.to_dict()
        legacy_payload.pop("patch_size")
        legacy_payload.pop("patch_stride")
        self.assertIsNone(TrainingConfig.from_dict(legacy_payload).patch_size)

        for patch_size, patch_stride, error in (
            (0, 64, "patch_size must be a positive integer"),
            (128, 0, "patch_stride must be a positive integer"),
            (64, 128, "patch_stride cannot exceed patch_size"),
        ):
            with self.assertRaisesRegex(TrainingProtocolError, error):
                TrainingConfig(
                    snapshot_id="snapshot-1",
                    split_id="split-1",
                    class_count=2,
                    epochs=3,
                    batch_size=4,
                    patch_size=patch_size,
                    patch_stride=patch_stride,
                )
        with self.assertRaisesRegex(
            TrainingProtocolError,
            "patch_size 128 exceeds sample rectangle 64x512",
        ):
            config.validate_patch_geometry(64, 512)

        bundled = TrainingRequest(
            request_id="run-2",
            config=config,
            artifact_staging_path="C:/project/staging/run-2",
            input_bundle_path="C:/project/staging/run-2/training_input_bundle.json",
        )
        self.assertEqual(decode_message(encode_message(bundled)), bundled)

        messages = (
            ProgressMessage(
                request_id="run-1",
                phase="train",
                epoch=1,
                total_epochs=3,
                step=4,
                total_steps=12,
                loss=0.25,
            ),
            TerminalMessage(request_id="run-1", status="completed", message="done"),
            CancelMessage(request_id="run-1", reason="user"),
        )
        for message in messages:
            self.assertEqual(decode_message(encode_message(message)), message)

        unknown_version = dict(payload)
        unknown_version["version"] = 999
        with self.assertRaisesRegex(TrainingProtocolError, "version"):
            decode_message(json.dumps(unknown_version))


if __name__ == "__main__":
    unittest.main()
