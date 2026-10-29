"""Run local binary/hierarchical SegFormer weights on an image or image folder."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm
try:
    from .overlay_video import PRESETS, render_predictions, validate_options
except ImportError:  # Support python inference.py from the script directory.
    from overlay_video import PRESETS, render_predictions, validate_options


IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path,
                        help="Local best/last checkpoint directory or .safetensors/.pt/.pth/.bin model weights")
    parser.add_argument("--input", "--images", dest="input", required=True, type=Path,
                        help="Input RGB image or image directory (searched recursively)")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for original-size binary PNG masks")
    parser.add_argument("--file-list", type=Path, help="Optional text list of basenames or filenames under --input")
    parser.add_argument("--model-config", type=Path, help="Model config.json for a standalone weights file")
    parser.add_argument("--config", type=Path, help="Training YAML, used when checkpoint preprocessing metadata is unavailable")
    parser.add_argument("--input-size", type=int, help="Override the saved square training input size")
    parser.add_argument("--threshold", type=float, help="Foreground sigmoid threshold (saved training value, otherwise 0.5)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-overlays", action="store_true", help="Save RGB previews with unchanged ROI and mint-green non-ROI")
    parser.add_argument("--save-video", "--save-videos", dest="save_video", action="store_true",
                        help="Save overlays and encode one MP4 per sequence")
    parser.add_argument("--overlay-opacity", type=float, default=0.2)
    parser.add_argument("--non-roi-color", default="#68ddbc")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument("--video-preset", choices=PRESETS, default="medium")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing prediction files")
    return parser.parse_args(argv)


def collect_images(source: Path, file_list: Path | None = None) -> tuple[Path, list[Path]]:
    source = source.expanduser().resolve()
    if source.is_file():
        if file_list:
            raise ValueError("--file-list requires an image directory as --input")
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {source}")
        return source.parent, [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Input image/directory does not exist: {source}")
    images = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No supported images found in {source}")
    if file_list is None:
        return source, images
    by_name: dict[str, list[Path]] = {}
    by_stem: dict[str, list[Path]] = {}
    for image in images:
        by_name.setdefault(image.name, []).append(image)
        by_stem.setdefault(image.stem, []).append(image)
    selected, seen = [], set()
    for line_number, line in enumerate(file_list.expanduser().read_text(encoding="utf-8-sig").splitlines(), 1):
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        if "/" in name or "\\" in name or ":" in name or name in {".", ".."}:
            raise ValueError(f"{file_list}:{line_number}: expected an image basename or filename")
        matches = by_name.get(name) if Path(name).suffix else by_stem.get(name)
        if not matches:
            raise FileNotFoundError(f"Listed image not found in {source}: {name}")
        if len(matches) != 1:
            raise ValueError(f"Ambiguous listed image: {name}")
        if matches[0] in seen:
            raise ValueError(f"Duplicate listed image: {name}")
        seen.add(matches[0]);selected.append(matches[0])
    if not selected:
        raise ValueError(f"Image list contains no samples: {file_list}")
    return source, selected


def load_model(weights: Path, model_config: Path | None = None):
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
    def checked(loaded):
        model, information = loaded
        failures = {key: information.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                    if information.get(key)}
        if failures:
            raise ValueError(f"Weights do not match the supplied model configuration: {failures}")
        return model
    weights = weights.expanduser().resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Weights do not exist: {weights}")
    if weights.is_dir():
        if model_config:
            raise ValueError("--model-config is only used with a standalone weights file")
        model = checked(SegformerForSemanticSegmentation.from_pretrained(
            str(weights), local_files_only=True, output_loading_info=True))
        return model, weights
    config_path = model_config.expanduser().resolve() if model_config else weights.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model configuration missing: {config_path}. Supply --model-config config.json.")
    if weights.name == "training_state.pt":
        raise ValueError("training_state.pt contains optimizer/RNG state, not model weights. Use best/ or model.safetensors.")
    config = SegformerConfig.from_json_file(str(config_path))
    if weights.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(weights), device="cpu")
    elif weights.suffix.lower() in {".pt", ".pth", ".bin"}:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            state = state.get("state_dict", state.get("model_state_dict", state))
    else:
        raise ValueError("Expected .safetensors, .pt, .pth or .bin model weights")
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Weights must contain a model state_dict of tensors")
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    # Transformers applies its version-specific checkpoint key conversions here;
    # a raw load_state_dict cannot read some save_pretrained safetensors files.
    model = checked(SegformerForSemanticSegmentation.from_pretrained(
        None, config=config, state_dict=state, local_files_only=True, output_loading_info=True))
    return model, weights.parent


def inference_settings(model, directory: Path, args: argparse.Namespace) -> dict:
    yaml_path = args.config.expanduser() if args.config else directory.parent / "resolved_config.yaml"
    training = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) if yaml_path.is_file() else {}
    if args.config and not yaml_path.is_file():
        raise FileNotFoundError(f"Training configuration does not exist: {yaml_path}")
    if not isinstance(training, dict):
        raise ValueError("Training YAML must contain a mapping")
    processor_path = directory / "preprocessor_config.json"
    processor = json.loads(processor_path.read_text()) if processor_path.is_file() else {}
    size = args.input_size or getattr(model.config, "training_input_size", None) or training.get("INPUT_SIZE")
    if size is None:
        saved_size = processor.get("size", {})
        if isinstance(saved_size, dict) and saved_size.get("height") == saved_size.get("width"):
            size = saved_size.get("height")
    if not isinstance(size, int) or size < 32:
        raise ValueError("Missing/invalid training input size. Supply --input-size (at least 32).")
    normalization = training.get("NORMALIZATION", {})
    mean = processor.get("image_mean", normalization.get("MEAN"))
    std = processor.get("image_std", normalization.get("STD"))
    if mean is None or std is None:
        raise ValueError("Missing normalization metadata. Keep preprocessor_config.json beside the weights or supply --config.")
    mean, std = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("Normalization must have three finite RGB means and three positive standard deviations")
    threshold = args.threshold if args.threshold is not None else float(training.get("THRESHOLD", 0.5))
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")
    channels = model.config.num_labels
    label_type = getattr(model.config, "target_label_type", None) or training.get("LABEL_TYPE")
    if channels not in {1, 2} or (channels == 2 and label_type != "hierarchical") or (channels == 1 and label_type == "hierarchical"):
        raise ValueError("Expected one binary output, or two independent hierarchical outputs [wide, narrow]")
    names = [str(model.config.id2label.get(index, f"mask_{index}")) for index in range(channels)]
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names) or len(set(names)) != len(names):
        raise ValueError("Checkpoint output labels must be distinct safe directory names")
    return {"input_size": size, "mean": mean.tolist(), "std": std.tolist(), "threshold": threshold, "labels": names}


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not re.fullmatch(r"cpu|cuda(?::\d+)?", value):
        raise ValueError("--device must be auto, cpu, cuda or cuda:<index>")
    device = torch.device(value)
    if device.type == "cuda" and (not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count()):
        raise ValueError(f"Requested CUDA device is unavailable: {device}")
    return device


def preprocess(image: np.ndarray, settings: dict) -> torch.Tensor:
    # Match SegFormerDataset: OpenCV bilinear RGB resize, /255, then normalization.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    size = settings["input_size"]
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    image = (image - np.asarray(settings["mean"], dtype=np.float32)) / np.asarray(settings["std"], dtype=np.float32)
    return torch.from_numpy(image).permute(2, 0, 1).contiguous()


def binary_masks(logits: torch.Tensor, size: tuple[int, int], threshold: float) -> np.ndarray:
    resized = F.interpolate(logits.float(), size=size, mode="bilinear", align_corners=False)
    # A one-channel binary head needs sigmoid, not argmax/softmax.
    return (resized.sigmoid()[0] >= threshold).to(torch.uint8).cpu().numpy() * 255


def run(args: argparse.Namespace) -> dict:
    if args.save_overlays or args.save_video:
        validate_options(args.overlay_opacity, args.non_roi_color, args.fps, args.video_crf, args.video_preset, args.save_video)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.input_size is not None and args.input_size < 32:
        raise ValueError("--input-size must be at least 32")
    root, images = collect_images(args.input, args.file_list)
    output = args.output_dir.expanduser().resolve()
    if args.input.expanduser().resolve().is_dir() and (output == root or root in output.parents):
        raise ValueError("--output-dir must be outside the input image directory")
    model, directory = load_model(args.weights, args.model_config)
    settings = inference_settings(model, directory, args)
    device = resolve_device(args.device)
    model.to(device).eval()
    destinations = []
    seen = set()
    for image in images:
        relative = image.relative_to(root).with_suffix(".png")
        paths = [output / "predictions" / name / relative for name in settings["labels"]] if len(settings["labels"]) > 1 else [output / "predictions" / relative]
        for path in paths:
            if path in seen:
                raise ValueError(f"Different input images map to the same output: {path}")
            if path.resolve() in images:
                raise ValueError(f"Output would overwrite an input image: {path}")
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace predictions.")
            seen.add(path)
        destinations.append(paths)
    manifest_path = output / "predictions.json"
    if manifest_path in seen:
        raise ValueError("Output path collision with predictions.json")
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output manifest already exists: {manifest_path}. Use a new output directory or --overwrite.")
    for folder in ("predictions", "overlays", "videos"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    records = []
    print(f"Device: {device}; images: {len(images)}; labels: {settings['labels']}; threshold: {settings['threshold']}")
    with torch.inference_mode(), tqdm(total=len(images), desc="Inference", unit="image", dynamic_ncols=True) as progress:
        for start in range(0, len(images), args.batch_size):
            batch_paths = images[start:start + args.batch_size]
            originals, tensors = [], []
            for path in batch_paths:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Cannot decode image: {path}")
                originals.append(image.shape[:2]);tensors.append(preprocess(image, settings))
            logits = model(pixel_values=torch.stack(tensors).to(device)).logits
            for offset, (path, size) in enumerate(zip(batch_paths, originals)):
                masks = binary_masks(logits[offset:offset + 1], size, settings["threshold"])
                paths = destinations[start + offset]
                for mask, destination in zip(masks, paths):
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(destination), mask):
                        raise OSError(f"Could not write prediction: {destination}")
                records.append({"image": path.relative_to(root).as_posix(), "height": size[0], "width": size[1],
                                "masks": {name: destination.relative_to(output).as_posix() for name, destination in zip(settings["labels"], paths)}})
                progress.update(1)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"weights": str(args.weights.expanduser().resolve()), "input_root": str(root),
                "created_at": datetime.now(timezone.utc).isoformat(), "settings": settings, "images": records}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.save_overlays or args.save_video:
        render_predictions(manifest_path, make_videos=args.save_video, opacity=args.overlay_opacity,
                           color=args.non_roi_color, fps=args.fps, crf=args.video_crf,
                           preset=args.video_preset, overwrite=args.overwrite)
    return manifest


if __name__ == "__main__":
    try:
        run(parse_args())
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(f"Inference failed: {error}") from error
