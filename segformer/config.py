"""YAML configuration loading and validation for SegFormer training."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import re
from typing import Any, Iterable

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


class _ConfigLoader(yaml.SafeLoader):
    """Safe YAML loader that also recognizes values such as ``1e-4``."""


_ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)[eE][-+]?[0-9]+$"),
    list("-+0123456789."),
)


def _entry(config: dict[str, Any], dotted_key: str) -> tuple[dict[str, Any], str]:
    parts = dotted_key.split(".")
    current: Any = config
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"Unknown configuration key: {dotted_key}")
        current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current:
        raise ValueError(f"Unknown configuration key: {dotted_key}")
    return current, parts[-1]


def _value(config: dict[str, Any], dotted_key: str) -> Any:
    parent, key = _entry(config, dotted_key)
    return parent[key]


def apply_overrides(
    config: dict[str, Any], overrides: Iterable[str]
) -> dict[str, Any]:
    """Apply repeatable ``KEY=VALUE`` command-line overrides."""
    resolved = deepcopy(config)
    for override in overrides:
        dotted_key, separator, raw_value = override.partition("=")
        dotted_key = dotted_key.strip()
        if not separator or not dotted_key or not raw_value.strip():
            raise ValueError(f"Expected --set KEY=VALUE, got {override!r}")
        parent, key = _entry(resolved, dotted_key)
        if isinstance(parent[key], dict):
            raise ValueError(
                f"{dotted_key} is a section; override one of its nested keys"
            )
        parent[key] = yaml.load(raw_value, Loader=_ConfigLoader)
    return resolved


def _integer(config: dict[str, Any], key: str, minimum: int) -> int:
    value = _value(config, key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


def _number(
    config: dict[str, Any],
    key: str,
    minimum: float = 0.0,
    maximum: float | None = None,
    *,
    positive: bool = False,
) -> float:
    parent, final_key = _entry(config, key)
    value = parent[final_key]
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"{key} must be a finite number")
    if number < minimum or (positive and number == minimum):
        operator = ">" if positive else ">="
        raise ValueError(f"{key} must be {operator} {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{key} must be <= {maximum}")
    parent[final_key] = number
    return number


def _boolean(config: dict[str, Any], key: str) -> bool:
    value = _value(config, key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a YAML boolean (true or false)")
    return value


def _nonempty_string(config: dict[str, Any], key: str) -> str:
    value = _value(config, key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _validate_scalar_or_list(config: dict[str, Any], key: str) -> None:
    value = _value(config, key)
    values = value if isinstance(value, list) else [value]
    if not values:
        raise ValueError(f"{key} must not be an empty list")
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{key} must be a nonnegative number or list of numbers")
        if not math.isfinite(float(item)) or float(item) < 0:
            raise ValueError(f"{key} must contain only finite nonnegative values")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and numerically normalize a training configuration."""
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")

    for key in ("DATASET_PATH", "MODEL_NAME", "OUTPUT_DIR"):
        _nonempty_string(config, key)

    deprecated_split_keys = (
        "TRAIN_LIST",
        "VAL_LIST",
        "TRAIN_SUBJECTS_FILE",
        "NUM_FOLDS",
        "VAL_FOLD",
    )
    present_deprecated = [key for key in deprecated_split_keys if key in config]
    if present_deprecated:
        raise ValueError(
            "Remove deprecated split settings; configure only TRAIN_SUBJECTS: "
            + ", ".join(present_deprecated)
        )

    training_subjects = _value(config, "TRAIN_SUBJECTS")
    if not isinstance(training_subjects, list) or (not training_subjects and not config.get('SNAPSHOT_FILE')):
        raise ValueError("TRAIN_SUBJECTS must be a nonempty list of subject IDs")
    if any(
        isinstance(subject, bool)
        or not isinstance(subject, int)
        or subject < 0
        for subject in training_subjects
    ):
        raise ValueError("TRAIN_SUBJECTS must contain nonnegative integers")
    if len(training_subjects) != len(set(training_subjects)):
        raise ValueError("TRAIN_SUBJECTS must not contain duplicates")

    label_type = _value(config, "LABEL_TYPE")
    if label_type not in ("narrow", "wide", "hierarchical", "teeth_gum"):
        raise ValueError("LABEL_TYPE must be narrow, wide, hierarchical, or teeth_gum")
    if label_type == "teeth_gum" and not config.get("SNAPSHOT_FILE"):
        raise ValueError("teeth_gum training requires an approved app snapshot")
    required_masks = (
        ("WIDE_DIR", "NARROW_DIR")
        if label_type == "hierarchical"
        else (f"{label_type.upper()}_DIR",)
    )
    for mask_key in (() if config.get('SNAPSHOT_FILE') else required_masks):
        _nonempty_string(config, f"MASKS.{mask_key}")

    excluded = _value(config, "EXCLUDE_SUBJECTS")
    if not isinstance(excluded, list) or any(
        isinstance(subject, bool)
        or not isinstance(subject, int)
        or subject < 0
        for subject in excluded
    ):
        raise ValueError("EXCLUDE_SUBJECTS must be a list of nonnegative integers")
    if len(excluded) != len(set(excluded)):
        raise ValueError("EXCLUDE_SUBJECTS must not contain duplicates")
    if set(training_subjects) & set(excluded):
        raise ValueError("TRAIN_SUBJECTS and EXCLUDE_SUBJECTS must not overlap")

    for key, minimum in (
        ("INPUT_SIZE", 32),
        ("BATCH_SIZE", 1),
        ("EPOCHS", 1),
        ("NUM_WORKERS", 0),
        ("SEED", 0),
        ("WARMUP_EPOCHS", 0),
        ("EARLY_STOPPING_PATIENCE", 0),
        ("LOG_EVERY_STEPS", 1),
    ):
        _integer(config, key, minimum)
    learning_rate = _number(config, "LEARNING_RATE", positive=True)
    _number(config, "WEIGHT_DECAY")
    minimum_learning_rate = _number(config, "MIN_LEARNING_RATE")
    if minimum_learning_rate > learning_rate:
        raise ValueError("MIN_LEARNING_RATE must be <= LEARNING_RATE")
    _number(config, "GRADIENT_CLIP_NORM")
    _number(config, "THRESHOLD", maximum=1.0)
    bce_weight = _number(config, "LOSS.BCE_WEIGHT")
    dice_weight = _number(config, "LOSS.DICE_WEIGHT")
    _number(config, "LOSS.HIERARCHY_WEIGHT")
    _number(config, "LOSS.EPSILON", positive=True)
    if bce_weight == 0 and dice_weight == 0:
        raise ValueError("At least one of LOSS.BCE_WEIGHT or LOSS.DICE_WEIGHT must be > 0")
    _validate_scalar_or_list(config, "LOSS.POS_WEIGHT")
    _validate_scalar_or_list(config, "LOSS.CHANNEL_WEIGHTS")

    for key in ("AMP", "PIN_MEMORY", "AUGMENTATION.ENABLE"):
        _boolean(config, key)
    for key in ("ELASTIC_APPROXIMATE", "NOISE_PER_CHANNEL"):
        _boolean(config, f"AUGMENTATION.{key}")
    if _value(config, "DEVICE") != "auto" and not re.fullmatch(
        r"(?:cpu|cuda(?::\d+)?)", str(_value(config, "DEVICE"))
    ):
        raise ValueError("DEVICE must be auto, cpu, cuda, or cuda:<index>")
    resume = _value(config, "RESUME_FROM")
    if resume is not None and (not isinstance(resume, str) or not resume.strip()):
        raise ValueError("RESUME_FROM must be null or a checkpoint directory")

    mean = _value(config, "NORMALIZATION.MEAN")
    std = _value(config, "NORMALIZATION.STD")
    if not (
        isinstance(mean, list)
        and isinstance(std, list)
        and len(mean) == len(std) == 3
        and all(isinstance(item, (int, float)) for item in mean + std)
        and all(float(item) > 0 for item in std)
    ):
        raise ValueError("NORMALIZATION.MEAN and STD must be three-number lists")

    augmentation_type = _value(config, "AUGMENTATION.TYPE")
    if augmentation_type not in ("light", "heavy", "elastic_color"):
        raise ValueError(
            "AUGMENTATION.TYPE must be light, heavy, or elastic_color"
        )
    for key in (
        "HORIZONTAL_FLIP_P",
        "VERTICAL_FLIP_P",
        "AFFINE_P",
        "ELASTIC_P",
        "BRIGHTNESS_CONTRAST_P",
        "GAMMA_P",
        "HSV_P",
        "BLUR_P",
        "NOISE_P",
    ):
        _number(config, f"AUGMENTATION.{key}", maximum=1.0)
    if config["WARMUP_EPOCHS"] > config["EPOCHS"]:
        raise ValueError("WARMUP_EPOCHS must be <= EPOCHS")

    return config


def load_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    overrides: Iterable[str] = (),
) -> dict[str, Any]:
    """Load a YAML file, apply CLI overrides, and validate the result."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.load(config_file, Loader=_ConfigLoader)
    resolved = validate_config(apply_overrides(config, overrides))
    resolved["OUTPUT_DIR"] = resolved["OUTPUT_DIR"].replace("{label_type}", resolved["LABEL_TYPE"])
    return resolved
