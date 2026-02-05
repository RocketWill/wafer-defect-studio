from array import array
import unittest

from wafer_defect_studio.normalization import (
    NativePixelSample,
    NormalizationBounds,
    compute_percentile_bounds,
)


class NormalizationTest(unittest.TestCase):
    def test_fixed_percentile_bounds_preserve_uint8_and_uint16_semantics(self):
        pixels8 = array("B", (0, 10, 20, 30, 40))
        pixels16 = array("H", (0, 1000, 2000, 3000, 65535))
        samples = (
            NativePixelSample("uint16", pixels16),
            NativePixelSample("uint8", pixels8),
        )

        expected = (
            NormalizationBounds("uint8", 0, 255, 10.0, 30.0, 25.0, 75.0),
            NormalizationBounds("uint16", 0, 65535, 1000.0, 3000.0, 25.0, 75.0),
        )
        self.assertEqual(compute_percentile_bounds(samples, 25, 75), expected)
        self.assertEqual(compute_percentile_bounds(reversed(samples), 25, 75), expected)

        before = pixels8.tobytes()
        with self.assertRaises(ValueError):
            compute_percentile_bounds((NativePixelSample("uint8", array("B")),), 25, 75)
        self.assertEqual(pixels8.tobytes(), before)


if __name__ == "__main__":
    unittest.main()
