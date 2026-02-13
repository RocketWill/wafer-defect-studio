import unittest

from wafer_defect_studio.detection_windows import Rect
from wafer_defect_studio.proposal_generation import DefectProposal
from wafer_defect_studio.proposal_review import ProposalReviewRevision
from wafer_defect_studio.proposal_queue import (
    ReviewQueueFilters,
    build_review_queue,
    filter_review_queue,
)


class ProposalQueueTest(unittest.TestCase):
    def setUp(self):
        self.proposals = (
            DefectProposal(
                proposal_id="p-2",
                class_name="particle",
                source_rect=Rect(10, 10, 5, 5),
                area=25,
                peak_confidence=0.91,
                mean_confidence=0.82,
                provenance={"image_asset_id": "image-1"},
            ),
            DefectProposal(
                proposal_id="p-1",
                class_name="scratch",
                source_rect=Rect(12, 12, 5, 5),
                area=25,
                peak_confidence=0.72,
                mean_confidence=0.42,
                provenance={
                    "image_asset_id": "image-1",
                    "provenance_disagreement": True,
                },
            ),
            DefectProposal(
                proposal_id="p-3",
                class_name="scratch",
                source_rect=Rect(40, 40, 4, 4),
                area=16,
                peak_confidence=0.99,
                mean_confidence=0.95,
                provenance={"image_asset_id": "image-1"},
            ),
        )
        self.revisions = {
            "p-1": ProposalReviewRevision(
                proposal_id="p-1",
                revision_number=1,
                status="accepted",
                source_rect=Rect(12, 12, 5, 5),
                provenance={"actor": "engineer"},
            ),
            "p-2": ProposalReviewRevision(
                proposal_id="p-2",
                revision_number=2,
                status="corrected",
                source_rect=Rect(11, 11, 6, 6),
                provenance={"actor": "engineer"},
            ),
        }

    def test_builds_status_flags_and_deterministic_identity_order(self):
        queue = build_review_queue(self.proposals, self.revisions)

        self.assertEqual([item.proposal_id for item in queue], ["p-1", "p-2", "p-3"])
        self.assertEqual([item.status for item in queue], ["accepted", "corrected", "unreviewed"])
        self.assertEqual(queue[0].source_rect, Rect(12, 12, 5, 5))
        self.assertTrue(queue[0].cross_class_conflict)
        self.assertTrue(queue[1].cross_class_conflict)
        self.assertTrue(queue[0].provenance_disagreement)
        self.assertFalse(queue[2].cross_class_conflict)
        self.assertFalse(queue[2].provenance_disagreement)

        reversed_queue = build_review_queue(tuple(reversed(self.proposals)), self.revisions)
        self.assertEqual(queue, reversed_queue)

    def test_filters_mean_confidence_conflict_disagreement_and_status(self):
        queue = build_review_queue(self.proposals, self.revisions)

        low = filter_review_queue(
            queue,
            ReviewQueueFilters(low_confidence_threshold=0.50),
        )
        self.assertEqual([item.proposal_id for item in low], ["p-1"])

        conflicts = filter_review_queue(
            queue,
            ReviewQueueFilters(cross_class_conflict=True),
        )
        self.assertEqual([item.proposal_id for item in conflicts], ["p-1", "p-2"])

        disagreements = filter_review_queue(
            queue,
            ReviewQueueFilters(provenance_disagreement=True),
        )
        self.assertEqual([item.proposal_id for item in disagreements], ["p-1"])

        accepted = filter_review_queue(
            queue,
            ReviewQueueFilters(statuses=("accepted",)),
        )
        self.assertEqual([item.proposal_id for item in accepted], ["p-1"])

        combined = filter_review_queue(
            queue,
            ReviewQueueFilters(
                low_confidence_threshold=0.90,
                cross_class_conflict=True,
                statuses=("accepted", "corrected"),
            ),
        )
        self.assertEqual([item.proposal_id for item in combined], ["p-1", "p-2"])


if __name__ == "__main__":
    unittest.main()
