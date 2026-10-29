from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from segformer.train import log_validation_images, validate


class RecordingWriter:
    def __init__(self):
        self.images = {}
        self.scalars = {}

    def add_images(self, tag, images, global_step):
        self.images[tag] = (images.clone(), global_step)

    def add_scalar(self, tag, value, global_step):
        self.scalars[tag] = (value, global_step)


class StaticModel:
    def eval(self):
        return self

    def __call__(self, pixel_values):
        batch_size = pixel_values.shape[0]
        return SimpleNamespace(logits=torch.zeros(batch_size, 1, 2, 2))


def training_config():
    return {
        "LABEL_TYPE": "narrow",
        "THRESHOLD": 0.5,
        "EPOCHS": 3,
        "NORMALIZATION": {
            "MEAN": [0.485, 0.456, 0.406],
            "STD": [0.229, 0.224, 0.225],
        },
        "LOSS": {
            "BCE_WEIGHT": 0.7,
            "DICE_WEIGHT": 0.3,
            "POS_WEIGHT": 1.0,
            "CHANNEL_WEIGHTS": 1.0,
            "HIERARCHY_WEIGHT": 0.1,
            "EPSILON": 1e-6,
        },
    }


class ValidationLoggingTests(unittest.TestCase):
    def test_logs_inputs_ground_truth_and_thresholded_predictions(self):
        config = training_config()
        raw = torch.full((2, 3, 4, 4), 0.25)
        mean = torch.tensor(config["NORMALIZATION"]["MEAN"]).view(1, 3, 1, 1)
        std = torch.tensor(config["NORMALIZATION"]["STD"]).view(1, 3, 1, 1)
        normalized = (raw - mean) / std
        logits = torch.tensor([[[[-1.0, 1.0], [-1.0, 1.0]]]]).repeat(2, 1, 1, 1)
        targets = torch.zeros(2, 1, 4, 4)
        targets[:, :, 1:3, 1:3] = 1.0
        writer = RecordingWriter()

        log_validation_images(
            writer, normalized, logits, targets, config, epoch=2
        )

        inputs, input_step = writer.images["val/first_batch/input"]
        predictions, prediction_step = writer.images[
            "val/first_batch/predicted_narrow"
        ]
        ground_truth, ground_truth_step = writer.images[
            "val/first_batch/ground_truth_narrow"
        ]
        torch.testing.assert_close(inputs, raw)
        torch.testing.assert_close(ground_truth, targets)
        self.assertEqual(
            (input_step, ground_truth_step, prediction_step),
            (3, 3, 3),
        )
        self.assertEqual(tuple(predictions.shape), (2, 1, 4, 4))
        self.assertEqual(set(predictions.unique().tolist()), {0.0, 1.0})

    def test_validation_logs_images_only_for_first_batch(self):
        config = training_config()
        batch = {
            "pixel_values": torch.zeros(2, 3, 4, 4),
            "labels": torch.zeros(2, 1, 4, 4),
        }
        writer = RecordingWriter()
        with patch("segformer.train.log_validation_images") as log_images:
            validate(
                StaticModel(),
                [batch, batch],
                torch.device("cpu"),
                config,
                epoch=0,
                writer=writer,
            )
        log_images.assert_called_once()


if __name__ == "__main__":
    unittest.main()
