import unittest

from wafer_defect_studio.detection_windows import (
    Padding,
    Rect,
    enumerate_inference_windows,
)


class DetectionWindowsTest(unittest.TestCase):
    def test_overlapping_edges_and_source_coordinate_round_trip(self):
        windows = enumerate_inference_windows(7, 5, window_size=4, stride=2)

        self.assertEqual(len(windows), 12)
        self.assertEqual(windows[0].read_rect, Rect(0, 0, 4, 4))
        self.assertEqual(windows[0].source_rect, Rect(0, 0, 4, 4))
        edge = windows[-1]
        self.assertEqual(edge.read_rect, Rect(6, 4, 4, 4))
        self.assertEqual(edge.source_rect, Rect(6, 4, 1, 1))
        self.assertEqual(edge.padding, Padding(left=0, top=0, right=3, bottom=3))
        self.assertEqual(edge.window_to_source(1, 0), (5, 4))
        self.assertEqual(edge.window_to_source(0, 1), (6, 3))

        for window in windows:
            point = (window.source_rect.x, window.source_rect.y)
            local = window.source_to_window(*point)
            self.assertEqual(window.window_to_source(*local), point)

        covered = {
            (x, y)
            for window in windows
            for y in range(window.source_rect.y, window.source_rect.bottom)
            for x in range(window.source_rect.x, window.source_rect.right)
        }
        self.assertEqual(len(covered), 7 * 5)

        for args in ((0, 5, 4, 2), (7, 0, 4, 2), (7, 5, 0, 2), (7, 5, 4, 0), (7, 5, 4, 5)):
            with self.assertRaises(ValueError):
                enumerate_inference_windows(*args)


if __name__ == "__main__":
    unittest.main()
