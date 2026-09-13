"""Offline checks. These do not test SAM 2 model inference."""
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import gradio as gr
import app


class EditorTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((40, 60, 3), [100, 120, 140], dtype=np.uint8)
        self.mask = np.zeros((40, 60), dtype=bool)
        self.mask[10:30, 20:40] = True

    def test_transparency_and_background(self):
        rgba = np.array(app.compose(self.rgb, self.mask, "Transparent", "#ffffff", None))
        np.testing.assert_array_equal(rgba[..., :3], self.rgb)
        np.testing.assert_array_equal(rgba[..., 3], self.mask * 255)
        solid = np.array(app.compose(self.rgb, self.mask, "Solid color", "#ff0000", None))
        np.testing.assert_array_equal(solid[self.mask], self.rgb[self.mask])
        self.assertTrue(np.all(solid[~self.mask] == [255, 0, 0]))
        with self.assertRaises(ValueError):
            app.compose(self.rgb, self.mask, "Uploaded background", "#ffffff", None)

    def test_correction_and_rejection_preserve_original(self):
        state = app.upload_image(Image.fromarray(self.rgb))[0]
        evt = gr.SelectData(None, {"index": [25, 15], "value": None})
        state = app.add_point(state, "Include (+)", evt)[0]
        with patch.object(app, "predict_mask", return_value=(self.mask, .9)):
            state = app.segment(state)[0]
        self.assertIsNotNone(state["mask"])
        result, paths, _ = app.export(state, "Transparent", "#ffffff", None)
        self.assertEqual(result.mode, "RGBA")
        np.testing.assert_array_equal(np.array(Image.open(paths[1])), self.mask * 255)
        state = app.add_point(state, "Exclude (-)", evt)[0]
        self.assertIsNone(state["mask"])
        self.assertEqual(state["labels"], [1, 0])
        state = app.undo_point(state)[0]
        self.assertEqual(state["labels"], [1])
        state = app.reject(state)[0]
        self.assertEqual(state["points"], [])
        np.testing.assert_array_equal(state["image"], self.rgb)

    def test_new_image_and_session_isolation(self):
        one = app.upload_image(Image.fromarray(self.rgb))[0]
        two = app.upload_image(Image.fromarray(self.rgb))[0]
        one["points"].append([1, 2])
        self.assertEqual(two["points"], [])
        self.assertIsNone(app.upload_image(None)[0]["image"])
        resized = app.upload_image(Image.new("RGB", (2000, 1000)))[0]
        self.assertEqual(resized["image"].shape, (800, 1600, 3))

    def test_circle_prompt_and_group_undo(self):
        state = app.upload_image(Image.fromarray(self.rgb))[0]
        evt = gr.SelectData(None, {"index": [30, 20], "value": None})
        state, canvas, *_ = app.add_prompt(state, "Include (+)", "Circle prompt", 18, evt)
        self.assertEqual(len(state["points"]), 9)
        self.assertEqual(state["labels"], [1] * 9)
        self.assertEqual(len(state["prompt_groups"]), 1)
        self.assertEqual(state["prompt_groups"][0]["radius"], 18)
        self.assertEqual(canvas.shape, self.rgb.shape)

        state = app.undo_point(state)[0]
        self.assertEqual(state["points"], [])
        self.assertEqual(state["labels"], [])
        self.assertEqual(state["prompt_groups"], [])

    def test_interface_builds(self):
        demo = app.build_app()
        try:
            self.assertGreater(len(demo.config["dependencies"]), 5)
        finally:
            # Gradio may create background resources while constructing Blocks.
            # Close them so the offline test process can terminate cleanly.
            demo.close()


if __name__ == "__main__":
    unittest.main()
