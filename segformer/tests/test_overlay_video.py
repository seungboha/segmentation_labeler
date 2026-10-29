import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import cv2
import numpy as np

from segformer.overlay_video import make_overlay, render_predictions, sequence_name


class OverlayTests(unittest.TestCase):
    def test_roi_is_untouched_and_only_background_is_green(self):
        image = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
        original = image.copy()
        mask = np.zeros((4, 5), dtype=np.uint8);mask[:, 2:] = 255
        overlay = make_overlay(image, mask)
        np.testing.assert_array_equal(overlay[mask != 0], image[mask != 0])
        expected = np.rint(image[mask == 0].astype(np.float32) * 0.8 + np.array([188, 221, 104], dtype=np.float32) * 0.2).astype(np.uint8)
        np.testing.assert_array_equal(overlay[mask == 0], expected)
        np.testing.assert_array_equal(image, original)

    def test_sequence_prefixes_preserve_subfolders(self):
        self.assertEqual(sequence_name("100_lower_00015.bmp"), "100_lower")
        self.assertEqual(sequence_name("nested/100_upper_00100.bmp"), "nested/100_upper")
        self.assertEqual(sequence_name("nested/00001.png"), "nested/sequence")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/FFprobe not installed")
    def test_real_h264_videos_group_sort_pad_and_keep_source_masks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);images = root / "RGB";predictions = root / "predictions"
            images.mkdir();predictions.mkdir()
            records, protected = [], {}
            # Deliberately shuffled: numeric order must place frame 2 before 10.
            for name in ["100_lower_10", "100_upper_2", "100_lower_2", "100_upper_1"]:
                image = np.full((17, 23, 3), (40, 80, 120), dtype=np.uint8)
                mask = np.zeros((17, 23), dtype=np.uint8);mask[3:12, 5:17] = 255
                image_path = images / f"{name}.bmp";mask_path = predictions / f"{name}.png"
                cv2.imwrite(str(image_path), image);cv2.imwrite(str(mask_path), mask)
                for path in [image_path, mask_path]:
                    protected[path] = hashlib.sha256(path.read_bytes()).digest()
                records.append({"image": image_path.name, "height": 17, "width": 23, "masks": {"wide": mask_path.name}})
            manifest = predictions / "predictions.json"
            manifest.write_text(json.dumps({"input_root": str(images), "images": records}))
            original_manifest = manifest.read_bytes()
            result = render_predictions(manifest)
            self.assertEqual(len(result["overlays"]), 4)
            self.assertEqual(len(result["videos"]), 2)
            self.assertEqual(result["roi"], "unchanged")
            lower = next(video for video in result["videos"] if video["sequence"] == "100_lower")
            self.assertEqual(lower["frame_order"], ["100_lower_2.bmp", "100_lower_10.bmp"])
            for video in result["videos"]:
                probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(predictions / video["file"])]))
                self.assertEqual(len(probe["streams"]), 1)
                stream = probe["streams"][0]
                self.assertEqual((stream["codec_name"], stream["pix_fmt"], stream["r_frame_rate"]), ("h264", "yuv420p", "10/1"))
                self.assertEqual((stream["width"], stream["height"], int(stream["nb_frames"])), (24, 66, 2))
            for entry in result["overlays"]:
                overlay = cv2.imread(str(predictions / entry["overlay"]))
                image = cv2.imread(str(images / entry["image"]))
                mask = cv2.imread(str(predictions / records[0]["masks"]["wide"]), cv2.IMREAD_GRAYSCALE)
                np.testing.assert_array_equal(overlay[mask != 0], image[mask != 0])
            for path, digest in protected.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), digest)
            self.assertEqual(manifest.read_bytes(), original_manifest)
            with self.assertRaises(FileExistsError):
                render_predictions(manifest)


if __name__ == "__main__":
    unittest.main()
