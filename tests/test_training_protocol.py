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
