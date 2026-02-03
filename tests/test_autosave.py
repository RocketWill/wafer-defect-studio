from pathlib import Path
import unittest

from wafer_defect_studio.annotation import GridAnnotation
from wafer_defect_studio.autosave import AutosaveGuard


class AutosaveGuardTest(unittest.TestCase):
    def test_failure_rolls_back_blocks_switch_and_actions_are_deterministic(self):
        persisted = {("image-1", 2, 3): ("scratch",)}
        visible = {("image-1", 2, 3): ("scratch",)}
        should_fail = True
        save_as_calls = []

        def save(annotation):
            if should_fail:
                raise OSError("simulated disk full")
            persisted[(annotation.image_asset_id, annotation.row, annotation.column)] = annotation.class_codes

        def restore(annotation, previous):
            visible[(annotation.image_asset_id, annotation.row, annotation.column)] = previous

        def apply(annotation):
            visible[(annotation.image_asset_id, annotation.row, annotation.column)] = (
                annotation.class_codes
            )

        def save_as(target_path, annotation):
            save_as_calls.append((Path(target_path), annotation))

        guard = AutosaveGuard(
            save_callback=save,
            restore_callback=restore,
            apply_callback=apply,
            save_as_callback=save_as,
        )
        annotation = GridAnnotation("image-1", 2, 3, ("scratch", "stain"))

        self.assertFalse(guard.autosave(annotation, ("scratch",)))
        self.assertEqual(persisted[("image-1", 2, 3)], ("scratch",))
        self.assertEqual(visible[("image-1", 2, 3)], ("scratch",))
        self.assertTrue(guard.blocked)
        self.assertFalse(guard.request_image_switch("image-2"))

        self.assertFalse(guard.retry())
        self.assertTrue(guard.blocked)

        should_fail = False
        self.assertTrue(guard.retry())
        self.assertFalse(guard.blocked)
        self.assertEqual(persisted[("image-1", 2, 3)], ("scratch", "stain"))
        self.assertEqual(visible[("image-1", 2, 3)], ("scratch", "stain"))

        should_fail = True
        self.assertFalse(guard.autosave(GridAnnotation("image-1", 2, 3, ("stain",)), ("scratch", "stain")))
        self.assertTrue(guard.blocked)
        self.assertTrue(guard.save_as("recovery.sqlite"))
        self.assertEqual(save_as_calls[0][0], Path("recovery.sqlite"))
        self.assertFalse(guard.blocked)

        self.assertFalse(guard.autosave(annotation, ("scratch", "stain")))
        self.assertTrue(guard.cancel())
        self.assertFalse(guard.blocked)
        self.assertFalse(guard.cancel())


if __name__ == "__main__":
    unittest.main()
