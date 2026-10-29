"""Loss and metrics for binary and nested binary segmentation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _channel_vector(value: float | list[float], channels: int, tensor: torch.Tensor) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=tensor.dtype, device=tensor.device).flatten()
    if result.numel() == 1:
        result = result.repeat(channels)
    if result.numel() != channels:
        raise ValueError(
            f"Expected one value or {channels} channel values, got {result.numel()}"
        )
    return result


def resize_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Upsample SegFormer's low-resolution logits to the label dimensions."""
    if logits.shape[-2:] == target.shape[-2:]:
        return logits
    return F.interpolate(
        logits,
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


def segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_config: dict[str, Any],
    *,
    hierarchical: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute weighted BCE + soft Dice and an optional nesting penalty."""
    logits = resize_logits(logits, targets)
    channels = logits.shape[1]
    if targets.shape != logits.shape:
        raise ValueError(
            f"Target shape {tuple(targets.shape)} does not match logits "
            f"{tuple(logits.shape)}"
        )
    if hierarchical and channels != 2:
        raise ValueError("Hierarchical loss requires [wide, narrow] output channels")

    channel_weights = _channel_vector(
        loss_config["CHANNEL_WEIGHTS"], channels, logits
    )
    if channel_weights.sum() <= 0:
        raise ValueError("LOSS.CHANNEL_WEIGHTS must contain a positive value")
    normalized_weights = channel_weights / channel_weights.sum()
    pos_weight = _channel_vector(loss_config["POS_WEIGHT"], channels, logits)

    bce_map = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight.view(channels, 1, 1),
        reduction="none",
    )
    bce_by_channel = bce_map.mean(dim=(0, 2, 3))
    bce = (bce_by_channel * normalized_weights).sum()

    probabilities = logits.sigmoid()
    epsilon = float(loss_config["EPSILON"])
    intersection = (probabilities * targets).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
    dice_by_channel = (2.0 * intersection + epsilon) / (denominator + epsilon)
    dice_loss = ((1.0 - dice_by_channel) * normalized_weights).sum()

    hierarchy = logits.new_zeros(())
    if hierarchical:
        hierarchy = F.relu(probabilities[:, 1] - probabilities[:, 0]).mean()
    total = (
        float(loss_config["BCE_WEIGHT"]) * bce
        + float(loss_config["DICE_WEIGHT"]) * dice_loss
        + float(loss_config["HIERARCHY_WEIGHT"]) * hierarchy
    )
    return total, {"bce": bce, "dice_loss": dice_loss, "hierarchy": hierarchy}


@torch.no_grad()
def dice_per_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    epsilon: float,
) -> torch.Tensor:
    """Return Dice for every image and channel, including empty/empty as 1."""
    logits = resize_logits(logits, targets)
    predictions = logits.sigmoid() >= threshold
    truth = targets >= 0.5
    intersection = (predictions & truth).sum(dim=(2, 3), dtype=torch.float32)
    denominator = predictions.sum(dim=(2, 3)) + truth.sum(dim=(2, 3))
    return (2.0 * intersection + epsilon) / (denominator + epsilon)
