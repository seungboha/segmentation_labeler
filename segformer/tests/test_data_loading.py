from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from segformer.config import DEFAULT_CONFIG_PATH, load_config
from segformer.data_loading import SegFormerDataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary_directory.name)
        (self.data_dir / "RGB").mkdir()
        (self.data_dir / "Label" / "wide").mkdir(parents=True)
        (self.data_dir / "Label" / "narrow").mkdir(parents=True)
        for subject in range(1, 5):
            name = f"{subject}_upper_00001"
            image = np.full((12, 18, 3), subject * 20, dtype=np.uint8)
            wide = np.zeros((12, 18), dtype=np.uint8)
            wide[2:10, 3:15] = 255
            narrow = np.zeros((12, 18), dtype=np.uint8)
            narrow[4:8, 6:12] = 255
            cv2.imwrite(str(self.data_dir / "RGB" / f"{name}.bmp"), image)
            cv2.imwrite(
                str(self.data_dir / "Label" / "wide" / f"{name}.png"), wide
            )
            cv2.imwrite(
                str(self.data_dir / "Label" / "narrow" / f"{name}.png"),
                narrow,
            )
        self.config = load_config(
            DEFAULT_CONFIG_PATH,
            [
                f"DATASET_PATH={self.data_dir}",
                "TRAIN_SUBJECTS=[1,2]",
                "INPUT_SIZE=32",
                "LABEL_TYPE=narrow",
                "NUM_WORKERS=0",
                "AUGMENTATION.ENABLE=false",
            ],
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_configured_subjects_train_and_remaining_subjects_validate(self):
        train = SegFormerDataset(self.config, "train")
        validation = SegFormerDataset(self.config, "val")
        self.assertEqual(train.subjects, (1, 2))
        self.assertEqual(validation.subjects, (3, 4))
        self.assertFalse(set(train.subjects) & set(validation.subjects))
        sample = train[0]
        self.assertEqual(tuple(sample["pixel_values"].shape), (3, 32, 32))
        self.assertEqual(tuple(sample["labels"].shape), (1, 32, 32))
        self.assertEqual(set(sample["labels"].unique().tolist()), {0.0, 1.0})

    def test_loader_discovers_all_frames_without_a_text_file(self):
        for directory, extension in (
            ("RGB", "bmp"),
            ("Label/narrow", "png"),
            ("Label/wide", "png"),
        ):
            original = self.data_dir / directory / f"1_upper_00001.{extension}"
            copy = self.data_dir / directory / f"1_upper_00002.{extension}"
            copy.write_bytes(original.read_bytes())
        train = SegFormerDataset(self.config, "train")
        self.assertEqual(
            train.sample_names,
            ["1_upper_00001", "1_upper_00002", "2_upper_00001"],
        )

    def test_hierarchical_masks_are_stacked_wide_then_narrow(self):
        config = deepcopy(self.config)
        config["LABEL_TYPE"] = "hierarchical"
        labels = SegFormerDataset(config, "val")[0]["labels"]
        self.assertEqual(tuple(labels.shape), (2, 32, 32))
        self.assertTrue((labels[1] <= labels[0]).all().item())

    def test_hierarchy_violation_is_rejected(self):
        config = deepcopy(self.config)
        config["LABEL_TYPE"] = "hierarchical"
        dataset = SegFormerDataset(config, "val")
        name = dataset.sample_names[0]
        bad_narrow = np.full((12, 18), 255, dtype=np.uint8)
        cv2.imwrite(
            str(self.data_dir / "Label" / "narrow" / f"{name}.png"),
            bad_narrow,
        )
        with self.assertRaisesRegex(ValueError, "Narrow mask must be inside"):
            dataset[0]

    def test_missing_training_subject_is_rejected(self):
        config = deepcopy(self.config)
        config["TRAIN_SUBJECTS"] = [1, 99]
        with self.assertRaisesRegex(ValueError, "TRAIN_SUBJECTS not found"):
            SegFormerDataset(config, "train")

    def test_exclusions_apply_before_validation_selection(self):
        config = deepcopy(self.config)
        config["EXCLUDE_SUBJECTS"] = [4]
        validation = SegFormerDataset(config, "val")
        self.assertEqual(validation.subjects, (3,))
        self.assertEqual(validation.excluded_sample_count, 1)

    def test_selected_split_masks_are_checked(self):
        missing = self.data_dir / "Label" / "narrow" / "3_upper_00001.png"
        missing.unlink()
        SegFormerDataset(self.config, "train")
        with self.assertRaisesRegex(FileNotFoundError, "mask.*missing"):
            SegFormerDataset(self.config, "val")

    def test_test_mode_loads_all_non_excluded_subjects(self):
        config = deepcopy(self.config)
        config["EXCLUDE_SUBJECTS"] = [4]
        dataset = SegFormerDataset(config, "test")
        self.assertEqual(dataset.subjects, (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
