from pathlib import Path
import tempfile
import unittest

import yaml

from segformer.config import DEFAULT_CONFIG_PATH, load_config


class ConfigTests(unittest.TestCase):
    def test_load_default_and_apply_nested_override(self):
        config = load_config(
            DEFAULT_CONFIG_PATH,
            [
                "EPOCHS=2",
                "WARMUP_EPOCHS=1",
                "AUGMENTATION.ENABLE=false",
                "LABEL_TYPE=wide",
                "TRAIN_SUBJECTS=[1,2]",
            ],
        )
        self.assertEqual(config["EPOCHS"], 2)
        self.assertFalse(config["AUGMENTATION"]["ENABLE"])
        self.assertEqual(config["LABEL_TYPE"], "wide")
        self.assertEqual(config["TRAIN_SUBJECTS"], [1, 2])
        self.assertEqual(Path(config["OUTPUT_DIR"]).name, "segformer_b0_wide")

    def test_output_directory_follows_label_type_override(self):
        for label_type in ("narrow", "wide", "hierarchical"):
            with self.subTest(label_type=label_type):
                config = load_config(DEFAULT_CONFIG_PATH, [f"LABEL_TYPE={label_type}"])
                self.assertEqual(
                    Path(config["OUTPUT_DIR"]).name,
                    f"segformer_b0_{label_type}",
                )

    def test_explicit_output_path_is_preserved(self):
        config = load_config(
            DEFAULT_CONFIG_PATH,
            ["LABEL_TYPE=wide", "OUTPUT_DIR=./outputs/my_experiment"],
        )
        self.assertEqual(config["OUTPUT_DIR"], "./outputs/my_experiment")

    def test_unknown_override_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown configuration key"):
            load_config(Path(DEFAULT_CONFIG_PATH), ["NOT_A_SETTING=1"])

    def test_training_subjects_must_be_nonempty_unique_integer_ids(self):
        for override in (
            "TRAIN_SUBJECTS=[]",
            "TRAIN_SUBJECTS=[1,1]",
            "TRAIN_SUBJECTS=[-1]",
            "TRAIN_SUBJECTS=[true]",
        ):
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, "TRAIN_SUBJECTS"):
                    load_config(DEFAULT_CONFIG_PATH, [override])

    def test_training_subjects_cannot_be_excluded(self):
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            load_config(
                DEFAULT_CONFIG_PATH,
                ["TRAIN_SUBJECTS=[101]", "EXCLUDE_SUBJECTS=[101]"],
            )

    def test_deprecated_text_file_and_fold_settings_are_rejected(self):
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        for key, value in (
            ("TRAIN_LIST", "train.txt"),
            ("VAL_LIST", "val.txt"),
            ("TRAIN_SUBJECTS_FILE", "subjects.txt"),
            ("NUM_FOLDS", 10),
            ("VAL_FOLD", 1),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                config = dict(raw)
                config[key] = value
                path = Path(directory) / "config.yaml"
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "deprecated split"):
                    load_config(path)


if __name__ == "__main__":
    unittest.main()
