"""Render non-ROI overlays and sequence MP4s from an inference manifest."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import uuid

import cv2
import numpy as np
from tqdm.auto import tqdm


PRESETS = ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow")


def validate_options(opacity, color, fps, crf, preset, make_videos):
    if not math.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("Overlay opacity must be between 0 and 1")
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        raise ValueError("Non-ROI color must be a hex RGB color such as #68ddbc")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Video FPS must be positive and finite")
    if not 0 <= crf <= 51 or preset not in PRESETS:
        raise ValueError("Invalid H.264 CRF or encoding preset")
    if make_videos and shutil.which("ffmpeg") is None:
        raise FileNotFoundError("FFmpeg is required for MP4 output. Install ffmpeg or request overlays only.")


def make_overlay(image, mask, opacity=0.2, color="#68ddbc"):
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[:2] != mask.shape:
        raise ValueError("Overlay image and binary mask dimensions must match")
    # OpenCV uses BGR. Leave every ROI pixel byte-for-byte unchanged.
    bgr = np.array([int(color[index:index + 2], 16) for index in (5, 3, 1)], dtype=np.float32)
    result = image.copy()
    background = mask == 0
    result[background] = np.rint(image[background].astype(np.float32) * (1 - opacity) + bgr * opacity).clip(0, 255).astype(np.uint8)
    return result


def safe_path(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Invalid relative path in predictions manifest: {relative!r}")
    result = root.joinpath(*path.parts).resolve()
    if result != root and root not in result.parents:
        raise ValueError(f"Path escapes its input directory: {relative!r}")
    return result


def sequence_name(image):
    path = PurePosixPath(image)
    match = re.fullmatch(r"(.+)_(\d+)", path.stem)
    # 100_lower_00015 -> 100_lower; plain numbered frames share their folder.
    name = match.group(1) if match else "sequence"
    return (path.parent / name).as_posix()


def frame_order(record):
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                 for part in re.split(r"(\d+)", record["image"]))


def add_filename_footer(image, filename):
    """Append a dark caption strip without covering any image pixels."""
    height, width = image.shape[:2]
    result = np.full((height + 48, width, 3), 24, dtype=np.uint8)
    result[:height] = image
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    text_width = cv2.getTextSize(filename, font, scale, 1)[0][0]
    scale *= min(1.0, max(1, width - 16) / max(1, text_width))
    cv2.putText(result, filename, (8, height + 30), font, scale,
                (255, 255, 255), 1, cv2.LINE_AA)
    return result


def encode_video(frames, destination, size, fps, crf, preset, progress, filenames=None):
    width, height = size
    output_height = height + (48 if filenames is not None else 0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.tmp.mp4")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{output_height}",
               "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", "-threads", "2", str(temporary)]
    try:
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
            try:
                for index, frame in enumerate(frames):
                    image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
                    if image is None or image.shape[:2] != (height, width):
                        raise ValueError(f"Video frames must have consistent dimensions: {frame}")
                    if filenames is not None:
                        image = add_filename_footer(image, filenames[index])
                    process.stdin.write(image.tobytes())
                    progress.update(1)
                process.stdin.close()
                returncode = process.wait()
            except BaseException:
                process.kill()
                process.wait()
                if not process.stdin.closed:
                    try:
                        process.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass
                errors.seek(0)
                details = errors.read().decode(errors="replace")[-3000:]
                if details:
                    print(details)
                raise
            if returncode:
                errors.seek(0)
                raise RuntimeError(f"FFmpeg failed: {errors.read().decode(errors='replace')[-3000:]}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def render_predictions(manifest_path, *, make_videos=True, opacity=0.2, color="#68ddbc",
                       fps=10.0, crf=18, preset="medium", overwrite=False):
    validate_options(opacity, color, fps, crf, preset, make_videos)
    manifest_path = Path(manifest_path).expanduser().resolve()
    root = manifest_path.parent
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_root = Path(data["input_root"]).expanduser().resolve()
    if not data["images"]:
        raise ValueError("No predictions to visualize")
    protected = {safe_path(input_root, item["image"]) for item in data["images"]}
    protected.update(safe_path(root, mask) for item in data["images"] for mask in item["masks"].values())
    summary_path = root / "visualizations.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"Visualization manifest already exists: {summary_path}. Use --overwrite.")
    jobs, groups, destinations = [], defaultdict(list), set()
    for item in data["images"]:
        image_path = safe_path(input_root, item["image"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Source image is missing: {image_path}")
        for label, mask in item["masks"].items():
            if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
                raise ValueError(f"Invalid mask label: {label!r}")
            mask_path = safe_path(root, mask)
            if not mask_path.is_file():
                raise FileNotFoundError(f"Predicted mask is missing: {mask_path}")
            relative = PurePosixPath(item["image"]).with_suffix(".png")
            overlay = safe_path(root, (PurePosixPath("overlays") / label / relative).as_posix())
            if overlay in protected or overlay in destinations:
                raise ValueError(f"Overlay output collides with a source or another output: {overlay}")
            if overlay.exists() and not overwrite:
                raise FileExistsError(f"Overlay already exists: {overlay}. Use --overwrite.")
            destinations.add(overlay)
            job = {"image": item["image"], "label": label, "source": image_path, "mask": mask_path,
                   "overlay": overlay, "size": (int(item["width"]), int(item["height"]))}
            jobs.append(job)
            groups[label, sequence_name(item["image"])].append(job)
    for (label, sequence), records in groups.items():
        if len({record["size"] for record in records}) != 1 and make_videos:
            raise ValueError(f"Mixed frame sizes within sequence {sequence!r}; no resizing is applied")
        video = safe_path(root, f"videos/{label}/{sequence}.mp4")
        if make_videos and video.exists() and not overwrite:
            raise FileExistsError(f"Video already exists: {video}. Use --overwrite.")
    rendered = []
    for job in tqdm(jobs, desc="Overlays", unit="image", dynamic_ncols=True):
        image = cv2.imread(str(job["source"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(job["mask"]), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None or mask.ndim != 2:
            raise ValueError(f"Cannot decode source image and binary mask: {job['image']}")
        if image.shape[1::-1] != job["size"] or mask.shape != image.shape[:2]:
            raise ValueError(f"Image/mask dimensions do not match the prediction manifest: {job['image']}")
        if not np.all((mask == 0) | (mask == 255)):
            raise ValueError(f"Expected a binary 0/255 prediction mask: {job['mask']}")
        overlay = make_overlay(image, mask, opacity, color)
        job["overlay"].parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(job["overlay"]), overlay):
            raise OSError(f"Could not save overlay: {job['overlay']}")
        rendered.append({"image": job["image"], "label": job["label"],
                         "overlay": job["overlay"].relative_to(root).as_posix()})
    videos = []
    if make_videos:
        with tqdm(total=len(jobs), desc="Encoding MP4", unit="frame", dynamic_ncols=True) as progress:
            for (label, sequence), records in sorted(groups.items()):
                records.sort(key=frame_order)
                destination = safe_path(root, f"videos/{label}/{sequence}.mp4")
                encode_video([record["overlay"] for record in records], destination,
                             records[0]["size"], fps, crf, preset, progress,
                             filenames=[PurePosixPath(record["image"]).name for record in records])
                width, height = records[0]["size"]
                height += 48
                videos.append({"sequence": sequence, "label": label, "file": destination.relative_to(root).as_posix(),
                               "frames": len(records), "fps": fps, "width": width + width % 2, "height": height + height % 2,
                               "frame_order": [record["image"] for record in records]})
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "predictions": manifest_path.name,
               "roi": "unchanged", "non_roi_color": color, "opacity": opacity,
               "video_footer": {"height": 48, "text": "original frame filename", "background": "#181818"},
               "encoding": {"codec": "libx264", "pixel_format": "yuv420p", "fps": fps, "crf": crf,
                            "preset": preset, "faststart": True, "audio": False},
               "overlays": rendered, "videos": videos}
    temporary = summary_path.with_name(f".{summary_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        os.replace(temporary, summary_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {len(rendered)} overlays and {len(videos)} MP4s under {root}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path, help="Existing predictions.json (inference is not rerun)")
    parser.add_argument("--overlays-only", action="store_true")
    parser.add_argument("--opacity", type=float, default=0.2)
    parser.add_argument("--non-roi-color", default="#68ddbc")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", choices=PRESETS, default="medium")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    render_predictions(args.predictions, make_videos=not args.overlays_only, opacity=args.opacity,
                       color=args.non_roi_color, fps=args.fps, crf=args.crf, preset=args.preset, overwrite=args.overwrite)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(f"Visualization failed: {error}") from error
