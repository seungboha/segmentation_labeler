"""Fine-tune SegFormer-B0 on the intraoral narrow/wide mask dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import yaml

try:
    from .config import DEFAULT_CONFIG_PATH, load_config
    from .data_loading import SegFormerDataset, seed_data_worker
    from .losses import dice_per_sample, resize_logits, segmentation_loss
except ImportError:  # Support ``python train.py`` from this directory.
    from config import DEFAULT_CONFIG_PATH, load_config
    from data_loading import SegFormerDataset, seed_data_worker
    from losses import dice_per_sample, resize_logits, segmentation_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML configuration path (default: config.yaml beside this script)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override an existing YAML key; may be supplied multiple times",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {index} was requested, but only "
                f"{torch.cuda.device_count()} device(s) are visible"
            )
    return device


def output_names(label_type: str) -> list[str]:
    return ["wide", "narrow"] if label_type == "hierarchical" else [label_type]


def build_dataloaders(
    config: dict[str, Any], device: torch.device
) -> tuple[DataLoader, DataLoader, torch.Generator]:
    dataset_class = SegFormerDataset
    if config.get('SNAPSHOT_FILE'):
        try:
            from .snapshot_data import SnapshotDataset
        except ImportError:
            from snapshot_data import SnapshotDataset
        dataset_class = SnapshotDataset
    train_dataset = dataset_class(config, mode="train")
    val_dataset = dataset_class(config, mode="val")
    print(f"Training subjects: {list(train_dataset.subjects)}", flush=True)
    print(f"Validation subjects: {list(val_dataset.subjects)}", flush=True)
    print(
        f"Samples: train={len(train_dataset)}, val={len(val_dataset)}, "
        f"excluded={train_dataset.excluded_sample_count}"
    )

    pin_memory = bool(config["PIN_MEMORY"] and device.type == "cuda")
    train_generator = torch.Generator().manual_seed(config["SEED"])
    val_generator = torch.Generator().manual_seed(config["SEED"] + 1)
    common = {
        "batch_size": config["BATCH_SIZE"],
        "num_workers": config["NUM_WORKERS"],
        "pin_memory": pin_memory,
        "worker_init_fn": seed_data_worker,
        "drop_last": False,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=train_generator,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        generator=val_generator,
        **common,
    )
    if not train_loader or not val_loader:
        raise ValueError("Training and validation loaders must both be nonempty")
    return train_loader, val_loader, train_generator


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    config: dict[str, Any],
) -> LambdaLR:
    total_steps = max(1, config["EPOCHS"] * steps_per_epoch)
    warmup_steps = config["WARMUP_EPOCHS"] * steps_per_epoch
    minimum_ratio = min(
        1.0, config["MIN_LEARNING_RATE"] / config["LEARNING_RATE"]
    )

    def lr_multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1, step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_multiplier)


def build_model(
    config: dict[str, Any], resume_dir: Path | None
) -> SegformerForSemanticSegmentation:
    names = output_names(config["LABEL_TYPE"])
    if resume_dir is not None:
        model = SegformerForSemanticSegmentation.from_pretrained(str(resume_dir))
        if model.config.num_labels != len(names):
            raise ValueError(
                f"Checkpoint has {model.config.num_labels} outputs, but "
                f"LABEL_TYPE={config['LABEL_TYPE']} requires {len(names)}"
            )
        return model

    id2label = {index: name for index, name in enumerate(names)}
    label2id = {name: index for index, name in id2label.items()}
    model = SegformerForSemanticSegmentation.from_pretrained(
        config["MODEL_NAME"],
        num_labels=len(names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    # Store task semantics with the exported model without relying on the generic
    # Transformers problem_type field, because our masks are independent logits.
    model.config.target_label_type = config["LABEL_TYPE"]
    model.config.training_input_size = config["INPUT_SIZE"]
    return model


def save_checkpoint(
    checkpoint_dir: Path,
    model: SegformerForSemanticSegmentation,
    processor: SegformerImageProcessor,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_dice: float,
    bad_epochs: int,
    config: dict[str, Any],
    train_generator: torch.Generator,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    processor.save_pretrained(checkpoint_dir)
    state: dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "best_dice": best_dice,
        "bad_epochs": bad_epochs,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "train_generator_state": train_generator.get_state(),
        "label_type": config["LABEL_TYPE"],
        "input_size": config["INPUT_SIZE"],
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(state, checkpoint_dir / "training_state.pt")


def load_training_state(
    resume_dir: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    train_generator: torch.Generator,
) -> tuple[int, int, float, int]:
    state_path = resume_dir / "training_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"Resume state does not exist: {state_path}")
    # This file contains optimizer and Python/NumPy RNG objects, so it is not a
    # weights-only file. Only resume checkpoints produced by this training code.
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("label_type") != config["LABEL_TYPE"]:
        raise ValueError("RESUME_FROM label type does not match LABEL_TYPE")
    if state.get("input_size") != config["INPUT_SIZE"]:
        raise ValueError("RESUME_FROM input size does not match INPUT_SIZE")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    train_generator.set_state(state["train_generator_state"])
    if torch.cuda.is_available() and "cuda_rng_state" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    return (
        int(state["epoch"]) + 1,
        int(state["global_step"]),
        float(state["best_dice"]),
        int(state["bad_epochs"]),
    )


def train_one_epoch(
    model: SegformerForSemanticSegmentation,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: dict[str, Any],
    epoch: int,
    global_step: int,
    writer: SummaryWriter,
) -> tuple[float, int]:
    model.train()
    if config["NUM_WORKERS"] == 0:
        loader.dataset.seed_augmentation(config["SEED"] + epoch)
    amp_enabled = bool(config["AMP"] and device.type == "cuda")
    hierarchical = config["LABEL_TYPE"] == "hierarchical"
    total_loss = 0.0
    sample_count = 0
    progress = tqdm(loader, desc=f"Train {epoch + 1}/{config['EPOCHS']}")
    for batch_index, batch in enumerate(progress):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(pixel_values=pixel_values).logits
            loss, components = segmentation_loss(
                logits,
                targets,
                config["LOSS"],
                hierarchical=hierarchical,
            )
        scaler.scale(loss).backward()
        clip_norm = float(config["GRADIENT_CLIP_NORM"])
        if clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = pixel_values.shape[0]
        total_loss += loss.item() * batch_size
        sample_count += batch_size
        global_step += 1
        writer.add_scalar("train/loss", loss.item(), global_step)
        writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], global_step)
        for name, value in components.items():
            writer.add_scalar(f"train/{name}", value.item(), global_step)
        progress.set_postfix(loss=f"{loss.item():.4f}")
        if (batch_index + 1) % config["LOG_EVERY_STEPS"] == 0:
            progress.write(
                f"epoch={epoch + 1} step={batch_index + 1}/{len(loader)} "
                f"loss={loss.item():.6f} lr={scheduler.get_last_lr()[0]:.3e}"
            )
    return total_loss / sample_count, global_step


@torch.no_grad()
def log_validation_images(
    writer: SummaryWriter,
    pixel_values: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    config: dict[str, Any],
    epoch: int,
) -> None:
    """Log first-batch inputs, ground truth, and thresholded predictions."""
    mean = torch.as_tensor(
        config["NORMALIZATION"]["MEAN"],
        dtype=pixel_values.dtype,
        device=pixel_values.device,
    ).view(1, 3, 1, 1)
    std = torch.as_tensor(
        config["NORMALIZATION"]["STD"],
        dtype=pixel_values.dtype,
        device=pixel_values.device,
    ).view(1, 3, 1, 1)
    input_images = (pixel_values * std + mean).clamp(0.0, 1.0)
    resized = resize_logits(logits, targets)
    predictions = (resized.sigmoid() >= config["THRESHOLD"]).float()
    step = epoch + 1
    writer.add_images(
        "val/first_batch/input",
        input_images.detach().cpu(),
        global_step=step,
    )
    for channel, name in enumerate(output_names(config["LABEL_TYPE"])):
        writer.add_images(
            f"val/first_batch/ground_truth_{name}",
            targets[:, channel : channel + 1].detach().cpu(),
            global_step=step,
        )
        writer.add_images(
            f"val/first_batch/predicted_{name}",
            predictions[:, channel : channel + 1].detach().cpu(),
            global_step=step,
        )


@torch.no_grad()
def validate(
    model: SegformerForSemanticSegmentation,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    epoch: int,
    writer: SummaryWriter,
) -> tuple[float, float, list[float]]:
    model.eval()
    hierarchical = config["LABEL_TYPE"] == "hierarchical"
    names = output_names(config["LABEL_TYPE"])
    loss_sum = 0.0
    dice_sum = torch.zeros(len(names), device=device)
    sample_count = 0
    for batch_index, batch in enumerate(
        tqdm(loader, desc=f"Val {epoch + 1}/{config['EPOCHS']}")
    ):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)
        logits = model(pixel_values=pixel_values).logits
        if batch_index == 0:
            log_validation_images(
                writer,
                pixel_values,
                logits,
                targets,
                config,
                epoch,
            )
        loss, _ = segmentation_loss(
            logits,
            targets,
            config["LOSS"],
            hierarchical=hierarchical,
        )
        batch_dice = dice_per_sample(
            logits,
            targets,
            threshold=config["THRESHOLD"],
            epsilon=config["LOSS"]["EPSILON"],
        )
        batch_size = pixel_values.shape[0]
        loss_sum += loss.item() * batch_size
        dice_sum += batch_dice.sum(dim=0)
        sample_count += batch_size

    average_loss = loss_sum / sample_count
    channel_dice = (dice_sum / sample_count).tolist()
    mean_dice = float(sum(channel_dice) / len(channel_dice))
    writer.add_scalar("val/loss", average_loss, epoch + 1)
    writer.add_scalar("val/dice", mean_dice, epoch + 1)
    for name, score in zip(names, channel_dice):
        writer.add_scalar(f"val/dice_{name}", score, epoch + 1)
    return average_loss, mean_dice, channel_dice


def main(config: dict[str, Any]) -> None:
    set_global_seed(config["SEED"])
    device = resolve_device(config["DEVICE"])
    print(f"Using device: {device}")
    output_dir = Path(config["OUTPUT_DIR"]).expanduser()
    print(f"Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    resume_dir = (
        Path(config["RESUME_FROM"]).expanduser()
        if config["RESUME_FROM"] is not None
        else None
    )
    if resume_dir is not None and not resume_dir.is_dir():
        raise FileNotFoundError(f"RESUME_FROM is not a directory: {resume_dir}")

    train_loader, val_loader, train_generator = build_dataloaders(config, device)
    subject_lists = {
        "training_subjects": list(train_loader.dataset.subjects),
        "validation_subjects": list(val_loader.dataset.subjects),
    }
    (output_dir / "subjects.json").write_text(json.dumps(subject_lists, indent=2), encoding="utf-8")
    model = build_model(config, resume_dir).to(device)
    processor = SegformerImageProcessor(
        do_resize=True,
        size={"height": config["INPUT_SIZE"], "width": config["INPUT_SIZE"]},
        do_rescale=True,
        do_normalize=True,
        image_mean=config["NORMALIZATION"]["MEAN"],
        image_std=config["NORMALIZATION"]["STD"],
        do_reduce_labels=False,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config["LEARNING_RATE"],
        weight_decay=config["WEIGHT_DECAY"],
    )
    scheduler = build_scheduler(optimizer, len(train_loader), config)
    amp_enabled = bool(config["AMP"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch, global_step, best_dice, bad_epochs = 0, 0, -math.inf, 0
    if resume_dir is not None:
        start_epoch, global_step, best_dice, bad_epochs = load_training_state(
            resume_dir,
            optimizer,
            scheduler,
            scaler,
            config,
            train_generator,
        )
        print(f"Resuming at epoch {start_epoch + 1}; best Dice={best_dice:.6f}")
    if start_epoch >= config["EPOCHS"]:
        print(
            f"Checkpoint already completed epoch {start_epoch}; "
            f"configured EPOCHS={config['EPOCHS']}"
        )
        return

    writer = SummaryWriter(output_dir / "tensorboard", purge_step=global_step)
    for name, subjects in subject_lists.items():
        writer.add_text(f"run/{name}", json.dumps(subjects), global_step=start_epoch)
    writer.flush()
    try:
        for epoch in range(start_epoch, config["EPOCHS"]):
            train_loss, global_step = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                device,
                config,
                epoch,
                global_step,
                writer,
            )
            val_loss, val_dice, channel_dice = validate(
                model, val_loader, device, config, epoch, writer
            )
            scores = ", ".join(
                f"{name}={score:.6f}"
                for name, score in zip(output_names(config["LABEL_TYPE"]), channel_dice)
            )
            print(
                f"Epoch {epoch + 1}: train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} dice={val_dice:.6f} ({scores})"
            )

            improved = val_dice > best_dice
            if improved:
                best_dice = val_dice
                bad_epochs = 0
            else:
                bad_epochs += 1
            save_checkpoint(
                output_dir / "last",
                model,
                processor,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                best_dice,
                bad_epochs,
                config,
                train_generator,
            )
            if improved:
                save_checkpoint(
                    output_dir / "best",
                    model,
                    processor,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step,
                    best_dice,
                    bad_epochs,
                    config,
                    train_generator,
                )
                print(f"Saved new best checkpoint with Dice={best_dice:.6f}")

            patience = config["EARLY_STOPPING_PATIENCE"]
            if patience > 0 and bad_epochs >= patience:
                print(
                    f"Early stopping after {bad_epochs} epochs without improvement; "
                    f"best Dice={best_dice:.6f}"
                )
                break
    finally:
        writer.close()


if __name__ == "__main__":
    arguments = parse_args()
    main(load_config(arguments.config, arguments.overrides))
