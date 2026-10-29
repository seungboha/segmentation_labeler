"""Dataset and augmentation pipeline compatible with HierarchicalUnet data."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def sample_subject(name: str) -> int:
    try:
        return int(name.split("_", 1)[0])
    except ValueError as error:
        raise ValueError(
            f"RGB filename must start with an integer subject ID: {name}.bmp"
        ) from error


def seed_data_worker(worker_id: int) -> None:
    """Seed Python, NumPy, and Albumentations inside a DataLoader worker."""
    del worker_id
    worker = torch.utils.data.get_worker_info()
    if worker is None:
        return
    seed = worker.seed % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    worker.dataset.seed_augmentation(seed)


def _probability(config: dict[str, Any], key: str) -> float:
    return float(config.get(key, 0.0))


def build_transforms(
    input_size: int,
    augmentation_config: dict[str, Any],
    training: bool,
) -> tuple[A.Compose, A.Compose | None]:
    """Build mask-safe geometry and image-only color augmentation."""
    resize = A.Resize(
        input_size,
        input_size,
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
    )
    if not training or not augmentation_config["ENABLE"]:
        return A.Compose([resize]), None

    augmentation_type = augmentation_config["TYPE"]
    if augmentation_type == "light":
        geometry = [
            resize,
            A.HorizontalFlip(
                p=_probability(augmentation_config, "HORIZONTAL_FLIP_P")
            ),
        ]
        color = [
            A.RandomBrightnessContrast(
                brightness_limit=float(augmentation_config["BRIGHTNESS_LIMIT"]),
                contrast_limit=float(augmentation_config["CONTRAST_LIMIT"]),
                p=_probability(augmentation_config, "BRIGHTNESS_CONTRAST_P"),
            )
        ]
        return A.Compose(geometry), A.Compose(color)

    geometry = [
        resize,
        A.HorizontalFlip(
            p=_probability(augmentation_config, "HORIZONTAL_FLIP_P")
        ),
    ]
    if augmentation_type == "elastic_color":
        geometry.append(
            A.VerticalFlip(
                p=_probability(augmentation_config, "VERTICAL_FLIP_P")
            )
        )
    shift_limit = float(augmentation_config["SHIFT_LIMIT"])
    scale_limit = float(augmentation_config["SCALE_LIMIT"])
    geometry.append(
        A.Affine(
            scale=(1.0 - scale_limit, 1.0 + scale_limit),
            translate_percent=(-shift_limit, shift_limit),
            rotate=(
                -float(augmentation_config["ROTATE_LIMIT"]),
                float(augmentation_config["ROTATE_LIMIT"]),
            ),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=_probability(augmentation_config, "AFFINE_P"),
        )
    )
    if augmentation_type == "elastic_color" and _probability(
        augmentation_config, "ELASTIC_P"
    ) > 0:
        geometry.append(
            A.ElasticTransform(
                alpha=float(augmentation_config["ELASTIC_ALPHA"]),
                sigma=float(augmentation_config["ELASTIC_SIGMA"]),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                approximate=bool(augmentation_config["ELASTIC_APPROXIMATE"]),
                p=_probability(augmentation_config, "ELASTIC_P"),
            )
        )

    color: list[A.BasicTransform] = [
        A.RandomBrightnessContrast(
            brightness_limit=float(augmentation_config["BRIGHTNESS_LIMIT"]),
            contrast_limit=float(augmentation_config["CONTRAST_LIMIT"]),
            p=_probability(augmentation_config, "BRIGHTNESS_CONTRAST_P"),
        )
    ]
    if augmentation_type == "elastic_color":
        color.extend(
            [
                A.RandomGamma(
                    gamma_limit=tuple(augmentation_config["GAMMA_LIMIT"]),
                    p=_probability(augmentation_config, "GAMMA_P"),
                ),
                A.HueSaturationValue(
                    hue_shift_limit=int(augmentation_config["HUE_SHIFT_LIMIT"]),
                    sat_shift_limit=int(augmentation_config["SAT_SHIFT_LIMIT"]),
                    val_shift_limit=int(augmentation_config["VAL_SHIFT_LIMIT"]),
                    p=_probability(augmentation_config, "HSV_P"),
                ),
                A.OneOf(
                    [
                        A.MotionBlur(
                            blur_limit=int(augmentation_config["BLUR_LIMIT"]),
                            p=1.0,
                        ),
                        A.GaussianBlur(
                            blur_limit=int(augmentation_config["BLUR_LIMIT"]),
                            sigma_limit=tuple(
                                augmentation_config["GAUSSIAN_BLUR_SIGMA_LIMIT"]
                            ),
                            p=1.0,
                        ),
                    ],
                    p=_probability(augmentation_config, "BLUR_P"),
                ),
                A.GaussNoise(
                    mean_range=tuple(augmentation_config["NOISE_MEAN_RANGE"]),
                    std_range=tuple(augmentation_config["NOISE_STD_RANGE"]),
                    per_channel=bool(augmentation_config["NOISE_PER_CHANNEL"]),
                    noise_scale_factor=float(
                        augmentation_config["NOISE_SCALE_FACTOR"]
                    ),
                    p=_probability(augmentation_config, "NOISE_P"),
                ),
            ]
        )
    return A.Compose(geometry), A.Compose(color)


class SegFormerDataset(Dataset):
    """Load RGB scans and binary narrow/wide masks.

    Images are returned as normalized float tensors named ``pixel_values``.
    Masks are float tensors named ``labels`` and have shape ``[C, H, W]``.
    Hierarchical channels are ordered ``[wide, narrow]``.
    """

    def __init__(self, config: dict[str, Any], mode: str) -> None:
        super().__init__()
        mode = mode.lower()
        if mode not in ("train", "val", "test"):
            raise ValueError("mode must be train, val, or test")
        self.mode = mode
        self.data_dir = Path(config["DATASET_PATH"]).expanduser()
        self.rgb_dir = self.data_dir / "RGB"
        self.label_type = config["LABEL_TYPE"]
        mask_config = config["MASKS"]
        if self.label_type == "hierarchical":
            relative_mask_dirs = [mask_config["WIDE_DIR"], mask_config["NARROW_DIR"]]
        else:
            relative_mask_dirs = [mask_config[f"{self.label_type.upper()}_DIR"]]
        self.mask_dirs = [
            path if (path := Path(item)).is_absolute() else self.data_dir / path
            for item in relative_mask_dirs
        ]

        if not self.rgb_dir.is_dir():
            raise FileNotFoundError(f"RGB directory does not exist: {self.rgb_dir}")
        for mask_dir in self.mask_dirs:
            if not mask_dir.is_dir():
                raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

        available_names = sorted(
            path.stem
            for path in self.rgb_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".bmp"
        )
        if not available_names:
            raise ValueError(f"No .bmp images found in {self.rgb_dir}")
        sample_subjects = {
            name: sample_subject(name) for name in available_names
        }
        available_subjects = set(sample_subjects.values())
        self.training_subjects_requested = tuple(config["TRAIN_SUBJECTS"])
        training_subjects = set(self.training_subjects_requested)
        self.excluded_subjects_requested = tuple(config["EXCLUDE_SUBJECTS"])
        excluded = set(self.excluded_subjects_requested)
        overlap = training_subjects & excluded
        if overlap:
            raise ValueError(
                "TRAIN_SUBJECTS and EXCLUDE_SUBJECTS must not overlap: "
                f"{sorted(overlap)}"
            )
        missing_subjects = training_subjects - available_subjects
        if missing_subjects:
            raise ValueError(
                f"TRAIN_SUBJECTS not found in {self.rgb_dir}: "
                f"{sorted(missing_subjects)}"
            )

        if mode == "train":
            selected_subjects = training_subjects
        elif mode == "val":
            selected_subjects = available_subjects - training_subjects - excluded
        else:
            selected_subjects = available_subjects - excluded
        self.sample_names = [
            name
            for name in available_names
            if sample_subjects[name] in selected_subjects
        ]
        self.excluded_sample_count = sum(
            subject in excluded for subject in sample_subjects.values()
        )
        if not self.sample_names:
            raise ValueError(f"The {mode} split contains no samples")

        for mask_dir in self.mask_dirs:
            missing = [
                f"{name}.png"
                for name in self.sample_names
                if not (mask_dir / f"{name}.png").is_file()
            ]
            if missing:
                preview = ", ".join(missing[:3])
                suffix = " ..." if len(missing) > 3 else ""
                raise FileNotFoundError(
                    f"{len(missing)} mask(s) are missing from {mask_dir}: "
                    f"{preview}{suffix}"
                )

        self.subjects = tuple(sorted(selected_subjects))

        self.geometry_transform, self.color_transform = build_transforms(
            input_size=config["INPUT_SIZE"],
            augmentation_config=config["AUGMENTATION"],
            training=mode == "train",
        )
        self.mean = np.asarray(config["NORMALIZATION"]["MEAN"], dtype=np.float32)
        self.std = np.asarray(config["NORMALIZATION"]["STD"], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.sample_names)

    def seed_augmentation(self, seed: int) -> None:
        """Reset transform-local random generators for reproducibility."""
        self.geometry_transform.set_random_seed(seed % (2**32))
        if self.color_transform is not None:
            self.color_transform.set_random_seed((seed + 1) % (2**32))

    @staticmethod
    def _to_binary(mask: np.ndarray) -> np.ndarray:
        threshold = 0 if mask.max() <= 1 else 127
        return mask > threshold

    def _read_masks(
        self, sample_name: str, image_shape: tuple[int, int]
    ) -> np.ndarray:
        masks: list[np.ndarray] = []
        for mask_dir in self.mask_dirs:
            mask_path = mask_dir / f"{sample_name}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Failed to read mask: {mask_path}")
            if mask.shape != image_shape:
                raise ValueError(
                    f"Size mismatch for {sample_name}: image={image_shape}, "
                    f"mask={mask.shape} ({mask_path})"
                )
            masks.append(self._to_binary(mask))
        if self.label_type == "hierarchical" and np.any(masks[1] & ~masks[0]):
            raise ValueError(f"Narrow mask must be inside wide mask: {sample_name}")
        return np.stack(masks, axis=-1).astype(np.uint8)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_name = self.sample_names[index]
        image_path = self.rgb_dir / f"{sample_name}.bmp"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = self._read_masks(sample_name, image.shape[:2])

        transformed = self.geometry_transform(image=image, mask=mask)
        image, mask = transformed["image"], transformed["mask"]
        if self.color_transform is not None:
            image = self.color_transform(image=image)["image"]

        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        pixel_values = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        labels = torch.from_numpy(mask.astype(np.float32)).permute(2, 0, 1)
        return {
            "pixel_values": pixel_values,
            "labels": labels.contiguous().float(),
            "sample_name": sample_name,
        }
