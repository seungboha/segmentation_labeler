import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from segformer.inference import binary_masks, collect_images, load_model, parse_args, preprocess, run


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.images = self.root / "RGB";self.images.mkdir()
        for name, size in [("one.bmp", (45, 63)), ("two.bmp", (70, 51))]:
            image = np.full((*size, 3), (30, 20, 10), dtype=np.uint8)
            cv2.imwrite(str(self.images / name), image)
        config = SegformerConfig(num_labels=1, id2label={0: "wide"}, label2id={"wide": 0},
            hidden_sizes=[8, 16, 32, 64], depths=[1, 1, 1, 1], num_attention_heads=[1, 2, 4, 8],
            sr_ratios=[8, 4, 2, 1], decoder_hidden_size=16)
        config.target_label_type = "wide";config.training_input_size = 32
        self.model = SegformerForSemanticSegmentation(config)
        with torch.no_grad():
            self.model.decode_head.classifier.weight.zero_()
            self.model.decode_head.classifier.bias.fill_(1)
        self.checkpoint = self.root / "best"
        self.model.save_pretrained(self.checkpoint)
        (self.checkpoint / "preprocessor_config.json").write_text(json.dumps({
            "size": {"height": 32, "width": 32}, "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225]}))

    def tearDown(self):
        self.temporary.cleanup()

    def args(self, *extra):
        return parse_args(["--weights", str(self.checkpoint), "--input", str(self.images),
                           "--output-dir", str(self.root / "predictions"), "--device", "cpu", *extra])

    def test_binary_sigmoid_and_original_size(self):
        masks = binary_masks(torch.zeros(1, 1, 3, 4), (17, 23), 0.5)
        self.assertEqual(masks.shape, (1, 17, 23))
        self.assertTrue(np.all(masks == 255))  # Single-channel argmax would incorrectly return zeros.
        self.assertTrue(np.all(binary_masks(torch.zeros(1, 1, 3, 4), (17, 23), 0.6) == 0))
        independent = torch.cat([torch.ones(1, 1, 3, 4), -torch.ones(1, 1, 3, 4)], dim=1)
        masks = binary_masks(independent, (17, 23), 0.5)
        self.assertTrue(np.all(masks[0] == 255))
        self.assertTrue(np.all(masks[1] == 0))

    def test_preprocessing_uses_rgb_and_training_normalization(self):
        image = np.full((8, 11, 3), (30, 20, 10), dtype=np.uint8)
        settings = {"input_size": 32, "mean": [0.1, 0.2, 0.3], "std": [0.5, 0.6, 0.7]}
        result = preprocess(image, settings)
        np.testing.assert_allclose(result[:, 0, 0].numpy(),
            (np.array([10, 20, 30]) / 255 - np.array(settings["mean"])) / np.array(settings["std"]), rtol=1e-6)

    def test_real_model_run_and_ordered_file_list(self):
        selected = self.root / "selected.txt"
        selected.write_text("\ufeff# validation\r\ntwo\r\none.bmp\r\n", encoding="utf-8")
        manifest = run(self.args("--file-list", str(selected), "--save-overlays"))
        self.assertEqual(manifest["settings"]["labels"], ["wide"])
        self.assertEqual([image["image"] for image in manifest["images"]], ["two.bmp", "one.bmp"])
        for record in manifest["images"]:
            mask = cv2.imread(str(self.root / "predictions" / record["masks"]["wide"]), cv2.IMREAD_UNCHANGED)
            self.assertEqual(mask.shape, (record["height"], record["width"]))
            self.assertEqual(mask.dtype, np.uint8)
            self.assertTrue(np.all(mask == 255))
        self.assertTrue((self.root / "predictions" / "predictions.json").is_file())
        overlay = cv2.imread(str(self.root / "predictions" / "overlays" / "wide" / "two.png"))
        original = cv2.imread(str(self.images / "two.bmp"))
        np.testing.assert_array_equal(overlay, original)  # This model predicts ROI everywhere.
        with self.assertRaises(FileExistsError):
            run(self.args())

    def test_loads_exact_safetensors_file_and_pytorch_state_dict(self):
        safe, _ = load_model(self.checkpoint / "model.safetensors")
        torch.testing.assert_close(safe.decode_head.classifier.bias, self.model.decode_head.classifier.bias)
        weight_file = self.checkpoint / "custom.pth"
        torch.save({"state_dict": {"module." + key: value for key, value in self.model.state_dict().items()}}, weight_file)
        pt, _ = load_model(weight_file)
        torch.testing.assert_close(pt.decode_head.classifier.bias, self.model.decode_head.classifier.bias)

    def test_lists_reject_missing_duplicate_and_ambiguous_images(self):
        selected = self.root / "selected.txt"
        for content, error in [("missing", FileNotFoundError), ("one\none.bmp", ValueError), ("../one", ValueError)]:
            selected.write_text(content)
            with self.subTest(content=content), self.assertRaises(error):
                collect_images(self.images, selected)
        (self.images / "nested").mkdir()
        cv2.imwrite(str(self.images / "nested" / "one.bmp"), np.zeros((10, 10, 3), dtype=np.uint8))
        selected.write_text("one")
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            collect_images(self.images, selected)

    def test_output_cannot_overwrite_inputs_or_collide(self):
        with self.assertRaisesRegex(ValueError, "outside the input"):
            run(self.args("--output-dir", str(self.images / "predictions")))
        cv2.imwrite(str(self.images / "one.png"), np.zeros((10, 10, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "same output"):
            run(self.args())

    def test_invalid_threshold_and_batch_size(self):
        with self.assertRaisesRegex(ValueError, "threshold"):
            run(self.args("--threshold", "1.1"))
        with self.assertRaisesRegex(ValueError, "batch-size"):
            run(self.args("--batch-size", "0"))


if __name__ == "__main__":
    unittest.main()
