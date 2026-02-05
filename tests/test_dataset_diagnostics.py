import unittest

from wafer_defect_studio.dataset_diagnostics import (
    PreviewImage,
    preview_dataset,
)


class DatasetDiagnosticsTest(unittest.TestCase):
    def test_preview_reports_distributions_warnings_and_records_normal_sampling(self):
        preview = preview_dataset(
            ("scratch", "crack", "missing"),
            (
                PreviewImage("a", "line-a", ("scratch", "crack"), 7),
                PreviewImage("b", "line-a", ("scratch",), 2),
                PreviewImage("c", "line-b", ("scratch",), 0),
                PreviewImage("bad", "line-b", ("missing",), 9, source_valid=False),
            ),
        )

        self.assertEqual(preview.image_distribution, (("eligible", 3), ("invalid", 1)))
        self.assertEqual(
            preview.class_distribution,
            (("scratch", 3), ("crack", 1), ("missing", 0)),
        )
        self.assertEqual(preview.group_distribution, (("line-a", 2), ("line-b", 1)))
        self.assertEqual(
            tuple(warning.code for warning in preview.warnings),
            ("missing-positive-class", "limited-validation", "invalid-source", "imbalance"),
        )
        self.assertEqual(preview.sampling.normal_to_positive_ratio, 1.0)
        self.assertEqual(preview.sampling.explicit_positive_count, 4)
        self.assertEqual(preview.sampling.available_normal_count, 9)
        self.assertEqual(preview.sampling.sampled_normal_count, 4)


if __name__ == "__main__":
    unittest.main()
