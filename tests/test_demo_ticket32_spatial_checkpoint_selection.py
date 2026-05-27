import unittest

from docs.demo.ticket31_development_gate import select_validation_spatial_epoch
from docs.demo.ticket32_development_gate import _training_config


def _row(*, coverage=1.0, precision=1.0, recall=1.0, leakage=0.0, occupancy=0.01):
    return {
        "defect_coverage_recall": coverage,
        "grid_precision": precision,
        "grid_recall": recall,
        "normal_grid_leak_rate": leakage,
        "asserted_grid_occupancy_p95": occupancy,
    }


class Ticket32SpatialCheckpointSelectionTest(unittest.TestCase):
    def test_replay_config_retains_all_epoch_states_without_changing_v5_policy(self) -> None:
        config = _training_config(101)
        self.assertTrue(config.retain_epoch_states)
        self.assertEqual((config.epochs, config.batch_size, config.learning_rate), (30, 4, 0.0003))

    def test_validation_spatial_quality_beats_lower_training_loss(self) -> None:
        candidates = (
            {"epoch": 1, "source_split": "validation", "training_loss": 0.01,
             "per_class": {"scratch": _row(precision=0.5, leakage=0.125),
                           "particle": _row()}},
            {"epoch": 7, "source_split": "validation", "training_loss": 0.5,
             "per_class": {"scratch": _row(), "particle": _row()}},
        )
        selected = select_validation_spatial_epoch(candidates, ("scratch", "particle"))
        self.assertEqual(selected["epoch"], 7)
        self.assertEqual(selected["selection_source"], "validation_spatial_metrics")

    def test_exact_spatial_tie_selects_earliest_epoch(self) -> None:
        candidates = tuple(
            {"epoch": epoch, "source_split": "validation", "per_class": {
                "scratch": _row(), "particle": _row()
            }}
            for epoch in (4, 3)
        )
        self.assertEqual(
            select_validation_spatial_epoch(candidates, ("scratch", "particle"))["epoch"],
            3,
        )

    def test_canonical_json_key_order_does_not_change_class_identity(self) -> None:
        selected = select_validation_spatial_epoch((
            {"epoch": 2, "source_split": "validation", "per_class": {
                "particle": _row(), "scratch": _row()
            }},
        ), ("scratch", "particle"))
        self.assertEqual(selected["epoch"], 2)

    def test_test_members_cannot_select_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation"):
            select_validation_spatial_epoch((
                {"epoch": 1, "source_split": "test", "per_class": {
                    "scratch": _row(), "particle": _row()
                }},
            ), ("scratch", "particle"))


if __name__ == "__main__":
    unittest.main()
