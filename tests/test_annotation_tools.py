import unittest

from wafer_defect_studio.annotation_tools import (
    AnnotationMode,
    AnnotationToolState,
    crossed_annotation_cells,
)


class AnnotationToolsTest(unittest.TestCase):
    def test_modes_shortcuts_and_deduplicated_multi_label_paint_erase(self):
        state = AnnotationToolState()
        state.set_class_key("scratch", "1")
        state.set_class_key("particle", "2")

        self.assertEqual(state.handle_shortcut("b"), AnnotationMode.PAINT)
        self.assertEqual(state.handle_shortcut("1"), ("scratch",))
        self.assertEqual(state.handle_shortcut("2"), ("scratch", "particle"))

        crossed = crossed_annotation_cells((2, 5), (25, 5), 10, 10)
        annotations = state.apply_cells(crossed + crossed[:2], {})
        self.assertEqual(
            annotations,
            {(0, 0): ("scratch", "particle"), (0, 1): ("scratch", "particle"), (0, 2): ("scratch", "particle")},
        )

        self.assertEqual(state.handle_shortcut("e"), AnnotationMode.ERASE)
        state.set_selected_classes(("particle",))
        erased = state.apply_cells(crossed, annotations)
        self.assertEqual(
            erased,
            {(0, 0): ("scratch",), (0, 1): ("scratch",), (0, 2): ("scratch",)},
        )

        self.assertEqual(state.handle_shortcut("a"), AnnotationMode.ANNOTATE)
        self.assertEqual(state.handle_shortcut("p"), AnnotationMode.PAN)


if __name__ == "__main__":
    unittest.main()
