"""CVAT for images XML -> staged, full-resolution object masks.

Format: https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/
RLE uses alternating background/foreground runs in row-major order.
"""
from collections import Counter
import math
from pathlib import PurePosixPath
import re
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw


class ImportError(ValueError):
    """A user-facing annotation or mapping error."""


class NoDoctype(ET.TreeBuilder):
    def doctype(self, name, pubid, system):
        raise ImportError("XML document types and entities are not supported.")


def integer(element, attribute, minimum=0):
    value = element.get(attribute, "")
    if not re.fullmatch(r"\d{1,10}", value) or int(value) < minimum:
        raise ImportError(f"Invalid {attribute} on <{element.tag}>.")
    return int(value)


def image_name(name):
    name = name.replace("\\", "/")
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or ":" in name:
        raise ImportError(f"Invalid CVAT image name: {name!r}.")
    return path.as_posix()


def frame_lookup(frames):
    exact, suffixes = {}, {}
    for index, frame in enumerate(frames):
        name = image_name(frame["name"])
        exact.setdefault(name, []).append(index)
        parts = PurePosixPath(name).parts
        for start in range(len(parts)):
            suffixes.setdefault("/".join(parts[start:]), []).append(index)
    return exact, suffixes


def match_image(name, lookup):
    exact, suffixes = lookup
    matches = exact.get(name) or suffixes.get(name)
    if not matches:
        # CVAT may include a root directory that the source folder omits.
        parts = PurePosixPath(name).parts
        for start in range(1, len(parts)):
            matches = exact.get("/".join(parts[start:]))
            if matches:
                break
    if not matches:
        matches = suffixes.get(PurePosixPath(name).name, [])
    if len(matches) != 1:
        reason = "matches more than one frame" if matches else "does not match a frame"
        raise ImportError(f"CVAT image {name!r} {reason}. Open the matching image folder with distinct paths.")
    return matches[0]


def decode_rle(shape, width, height):
    left, top = integer(shape, "left"), integer(shape, "top")
    w, h = integer(shape, "width", 1), integer(shape, "height", 1)
    if left + w > width or top + h > height:
        raise ImportError("CVAT mask bounds extend outside the source image.")
    values = shape.get("rle", "").split(",")
    if not values or any(not re.fullmatch(r"\s*\d{1,20}\s*", value) for value in values):
        raise ImportError("Invalid CVAT mask RLE: expected nonnegative integer runs.")
    runs = [int(value) for value in values]
    if sum(runs) != w * h:
        raise ImportError("CVAT mask RLE length does not match its bounding rectangle.")
    flat = np.zeros(w * h, dtype=bool)
    position = 0
    for index, length in enumerate(runs):
        if index % 2:
            flat[position:position + length] = True
        position += length
    mask = np.zeros((height, width), dtype=bool)
    mask[top:top + h, left:left + w] = flat.reshape(h, w)
    return mask


def decode_polygon(shape, width, height):
    try:
        points = [tuple(map(float, pair.split(","))) for pair in shape.get("points", "").split(";")]
        if len(points) < 3 or any(len(point) != 2 for point in points):
            raise ValueError
        if any(not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= width and 0 <= y <= height)
               for x, y in points):
            raise ValueError
    except ValueError:
        raise ImportError("Invalid CVAT polygon: expected at least three finite points inside the image.")
    surface = Image.new("1", (width, height))
    ImageDraw.Draw(surface).polygon(points, fill=1)
    return np.asarray(surface).astype(bool)


def stage(data, frames, get_size, destination):
    """Validate everything and write masks to a disposable directory, without editing the project."""
    try:
        root = ET.fromstring(data, parser=ET.XMLParser(target=NoDoctype()))
    except ImportError:
        raise
    except (ET.ParseError, LookupError, ValueError):
        raise ImportError("Cannot read XML. Choose an unmodified CVAT annotations.xml file.")
    if root.tag != "annotations":
        raise ImportError("This is not a CVAT annotation XML file (missing <annotations>).")
    version = root.findtext("version", "").strip()
    if version not in {"1.0", "1.1"}:
        raise ImportError(f"Unsupported CVAT XML version: {version or 'missing'}. Expected 1.0 or 1.1.")
    if root.find("track") is not None:
        raise ImportError("This XML contains video tracks. Export as 'CVAT for images' so masks can be matched by image filename.")
    if any((node.text or "").strip().lower() == "true" for node in root.findall(".//flipped")):
        raise ImportError("Flipped CVAT tasks are not supported. Export annotations aligned with the original images.")
    colors = {}
    for label in root.findall(".//labels/label"):
        name = (label.findtext("name", "")).strip()
        color = label.findtext("color", "")
        if re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            colors[name] = color.lower()
    items, skipped, seen_frames = [], Counter(), set()
    lookup = frame_lookup(frames)
    for image in root.findall("image"):
        shapes = [shape for shape in image if shape.tag in {"mask", "polygon"}]
        skipped.update(shape.tag for shape in image if shape.tag not in {"mask", "polygon"})
        if not shapes:
            continue
        name = image_name(image.get("name", ""))
        frame = match_image(name, lookup)
        if frame in seen_frames:
            raise ImportError(f"Multiple CVAT image entries map to {frames[frame]['name']!r}.")
        seen_frames.add(frame)
        size = (integer(image, "width", 1), integer(image, "height", 1))
        if size != get_size(frame):
            raise ImportError(f"Dimensions of {name!r} do not match the source frame. Masks are never resized.")
        groups = {}
        for index, shape in enumerate(shapes):
            label = shape.get("label", "").strip()
            if not label or len(label) > 100:
                raise ImportError("Every CVAT mask or polygon needs a label of 1–100 characters.")
            group = integer(shape, "group_id") if "group_id" in shape.attrib else 0
            track = integer(shape, "track_id") if "track_id" in shape.attrib else None
            # Grouped components of one instance are united within each image.
            identity = ("track", track) if track is not None else (("group", group) if group else ("shape", index))
            key = (label, identity)
            mask = decode_rle(shape, *size) if shape.tag == "mask" else decode_polygon(shape, *size)
            if not mask.any():
                raise ImportError(f"An empty mask was found in {name!r}. Remove empty shapes before importing.")
            if key in groups:
                groups[key]["mask"] |= mask
            else:
                groups[key] = {"mask": mask, "label": label, "track": track, "group": group}
        for item in groups.values():
            filename = f"{len(items):08d}.png"
            Image.fromarray(item.pop("mask").astype(np.uint8) * 255).save(destination / filename)
            items.append({**item, "frame": frame, "file": filename, "width": size[0], "height": size[1],
                          "image_name": name})
    if not items:
        raise ImportError("No masks or polygons found. Export segmentation annotations as 'CVAT for images' XML.")
    labels = list(dict.fromkeys(item["label"] for item in items))
    return {"items": items, "colors": colors,
            "summary": {"mask_count": len(items), "frame_count": len(seen_frames), "first_frame": min(seen_frames), "labels": labels,
                        "skipped_shapes": dict(skipped)}}
