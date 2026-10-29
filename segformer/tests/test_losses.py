import unittest

import torch

from segformer.losses import dice_per_sample, segmentation_loss


LOSS_CONFIG = {
    "BCE_WEIGHT": 0.7,
    "DICE_WEIGHT": 0.3,
    "POS_WEIGHT": 1.0,
    "CHANNEL_WEIGHTS": 1.0,
    "HIERARCHY_WEIGHT": 0.1,
    "EPSILON": 1e-6,
}


class LossTests(unittest.TestCase):
    def test_binary_loss_accepts_low_resolution_segformer_logits(self):
        logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
        targets = torch.zeros(2, 1, 32, 32)
        loss, components = segmentation_loss(
            logits, targets, LOSS_CONFIG, hierarchical=False
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(components), {"bce", "dice_loss", "hierarchy"})
        self.assertIsNotNone(logits.grad)

    def test_hierarchy_penalty_detects_narrow_outside_wide_probability(self):
        logits = torch.stack(
            [torch.full((4, 4), -4.0), torch.full((4, 4), 4.0)]
        ).unsqueeze(0)
        targets = torch.zeros(1, 2, 4, 4)
        _, components = segmentation_loss(
            logits, targets, LOSS_CONFIG, hierarchical=True
        )
        self.assertGreater(components["hierarchy"].item(), 0.9)

    def test_empty_prediction_and_target_have_perfect_dice(self):
        logits = torch.full((1, 1, 4, 4), -100.0)
        targets = torch.zeros(1, 1, 4, 4)
        score = dice_per_sample(logits, targets, threshold=0.5, epsilon=1e-6)
        self.assertEqual(score.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
