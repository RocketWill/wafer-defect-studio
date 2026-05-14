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
    def test_hard_negative_refinement_values_round_trip_and_reject_legacy(self):
        digest = "AB" * 32
        config = TrainingConfig(
            snapshot_id="snapshot-1", split_id="split-1", class_count=1,
            epochs=2, batch_size=1, patch_size=32, patch_stride=32,
            training_policy="spatial_mil_v4",
            priority_normal_bag_ids=["bag-b", "bag-a"],
            hard_negative_selection_sha256=digest,
        )
        self.assertEqual(config.priority_normal_bag_ids, ("bag-b", "bag-a"))
        self.assertEqual(config.hard_negative_selection_sha256, digest.lower())
        self.assertEqual(config.to_dict()["priority_normal_bag_ids"], ["bag-b", "bag-a"])
        self.assertEqual(TrainingConfig.from_json(config.to_json()), config)
        old = config.to_dict()
        old.pop("priority_normal_bag_ids")
        old.pop("hard_negative_selection_sha256")
        self.assertEqual(TrainingConfig.from_dict(old).priority_normal_bag_ids, ())
        with self.assertRaisesRegex(TrainingProtocolError, "provided together"):
            TrainingConfig(
                snapshot_id="snapshot-1", split_id="split-1", class_count=1,
                epochs=1, batch_size=1, patch_size=32, patch_stride=32,
                training_policy="spatial_mil_v4", priority_normal_bag_ids=("bag",),
            )
        with self.assertRaisesRegex(TrainingProtocolError, "provided together"):
            TrainingConfig(
                snapshot_id="snapshot-1", split_id="split-1", class_count=1,
                epochs=1, batch_size=1, patch_size=32, patch_stride=32,
                training_policy="spatial_mil_v4",
                hard_negative_selection_sha256="a" * 64,
            )
        for bag_ids, digest, message in (
            (("bag", "bag"), "a" * 64, "unique non-empty"),
            (("bag",), "z" * 64, "64 hexadecimal"),
            (("bag",), "a" * 63, "64 hexadecimal"),
        ):
            with self.assertRaisesRegex(TrainingProtocolError, message):
                TrainingConfig(
                    snapshot_id="snapshot-1", split_id="split-1", class_count=1,
                    epochs=1, batch_size=1, patch_size=32, patch_stride=32,
                    training_policy="spatial_mil_v4",
                    priority_normal_bag_ids=bag_ids,
                    hard_negative_selection_sha256=digest,
                )
        with self.assertRaisesRegex(TrainingProtocolError, "requires spatial_mil_v4"):
            TrainingConfig(
                snapshot_id="snapshot-1", split_id="split-1", class_count=1,
                epochs=1, batch_size=1, patch_size=32, patch_stride=32,
                priority_normal_bag_ids=("bag",),
                hard_negative_selection_sha256="a" * 64,
            )

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
        legacy_payload.pop("training_policy")
        self.assertIsNone(TrainingConfig.from_dict(legacy_payload).patch_size)
        self.assertEqual(TrainingConfig.from_dict(legacy_payload).training_policy, "legacy")

        spatial = TrainingConfig(
            snapshot_id="snapshot-1",
            split_id="split-1",
            class_count=2,
            epochs=3,
            batch_size=4,
            training_policy="spatial_mil_v4",
            patch_size=128,
            patch_stride=64,
        )
        self.assertEqual(TrainingConfig.from_json(spatial.to_json()), spatial)
        with self.assertRaisesRegex(TrainingProtocolError, "requires patch geometry"):
            TrainingConfig(
                snapshot_id="snapshot-1",
                split_id="split-1",
                class_count=2,
                epochs=3,
                batch_size=4,
                training_policy="spatial_mil_v4",
            )

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
