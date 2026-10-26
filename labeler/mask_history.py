"""Immutable whole-project checkpoints with shared, content-addressed mask data."""
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
import re
import uuid
import zipfile

import numpy as np
from PIL import Image


MAX_DELTA_DEPTH = 16


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2).encode("utf-8"))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def binary_png(mask, compact=True):
    stream = io.BytesIO()
    # One bit per pixel before PNG compression; viewers still decode to 0 / 255.
    image = Image.fromarray(mask).convert("1") if compact else Image.fromarray(mask.astype(np.uint8) * 255)
    image.save(stream, format="PNG", optimize=True)
    return stream.getvalue()


def mask_key(mask):
    dimensions = f"{mask.shape[0]}x{mask.shape[1]}:".encode("ascii")
    return hashlib.sha256(dimensions + np.packbits(mask).tobytes()).hexdigest()


def blob_path(directory, key):
    if not re.fullmatch(r"[0-9a-f]{64}", key):
        raise ValueError("Invalid history mask reference.")
    return directory / "history" / "blobs" / key


def read_mask(directory, key):
    """Reconstruct a binary mask; chains are bounded when written."""
    chain = []
    current = key
    while True:
        stem = blob_path(directory, current)
        info = read_json(stem.with_suffix(".json"))
        if info["kind"] == "png":
            with Image.open(stem.with_suffix(".png")) as image:
                mask = np.array(image.convert("L")) != 0
            break
        if info["kind"] != "xor" or len(chain) >= MAX_DELTA_DEPTH:
            raise ValueError("Invalid mask history chain.")
        chain.append((stem, info))
        current = info["parent"]
    for stem, info in reversed(chain):
        x, y, width, height = info["bounds"]
        with np.load(stem.with_suffix(".npz"), allow_pickle=False) as delta:
            changed = np.unpackbits(delta["bits"], count=width * height).reshape(height, width).astype(bool)
        mask[y:y + height, x:x + width] ^= changed
    if mask_key(mask) != key:
        raise ValueError("Mask history checksum does not match.")
    return mask


def store_mask(directory, mask, parent=None):
    key = mask_key(mask)
    stem = blob_path(directory, key)
    if stem.with_suffix(".json").exists():
        return key  # Identical pixels share one immutable blob, even across objects.
    data = binary_png(mask)
    extension = ".png"
    info = {"kind": "png", "depth": 0, "shape": list(mask.shape)}
    if parent:
        previous = read_json(blob_path(directory, parent).with_suffix(".json"))
        if previous["shape"] == list(mask.shape) and previous["depth"] < MAX_DELTA_DEPTH:
            difference = mask ^ read_mask(directory, parent)
            rows = np.flatnonzero(difference.any(axis=1))
            columns = np.flatnonzero(difference.any(axis=0))
            if rows.size:
                y, bottom = int(rows[0]), int(rows[-1]) + 1
                x, right = int(columns[0]), int(columns[-1]) + 1
                stream = io.BytesIO()
                np.savez_compressed(stream, bits=np.packbits(difference[y:bottom, x:right]))
                candidate = {"kind": "xor", "depth": previous["depth"] + 1,
                             "shape": list(mask.shape), "parent": parent,
                             "bounds": [x, y, right - x, bottom - y]}
                delta = stream.getvalue()
                # Include descriptor overhead in the storage comparison.
                if len(delta) + len(json.dumps(candidate, indent=2)) < len(data) + len(json.dumps(info, indent=2)):
                    data, extension, info = delta, ".npz", candidate
    atomic_write(stem.with_suffix(extension), data)
    write_json(stem.with_suffix(".json"), info)
    return key


def list_versions(directory):
    return sorted((read_json(path)["checkpoint"] for path in
                   (directory / "history" / "versions").glob("*.json")),
                  key=lambda item: item["number"])


def get_version(directory, version_id):
    if not re.fullmatch(r"[0-9a-f]{32}", version_id):
        raise FileNotFoundError("Version not found.")
    return read_json(directory / "history" / "versions" / f"{version_id}.json")


def save_version(directory, project, name, note):
    versions = list_versions(directory)
    previous = get_version(directory, versions[-1]["id"]) if versions else None
    parents = {(frame, a["object_id"]): a["history_blob"] for frame, entries in
               (previous["project"]["annotations"].items() if previous else []) for a in entries}
    snapshot = copy.deepcopy(project)
    for frame, entries in snapshot["annotations"].items():
        for annotation in entries:
            with Image.open(directory / annotation["mask_file"]) as image:
                mask = np.asarray(image.convert("L")) != 0
            annotation["history_blob"] = store_mask(directory, mask, parents.get((frame, annotation["object_id"])))
            annotation["mask_file"] = f"masks/{int(frame):06d}_{annotation['object_id']}.png"
    checkpoint = {"id": uuid.uuid4().hex,
                  "number": versions[-1]["number"] + 1 if versions else 1,
                  "name": name, "note": note,
                  "created_at": datetime.now(timezone.utc).isoformat(),
                  "mask_count": sum(len(entries) for entries in snapshot["annotations"].values())}
    # Publishing the manifest is the commit point. Partial saves never appear in the list.
    write_json(directory / "history" / "versions" / f"{checkpoint['id']}.json",
               {"format_version": 1, "checkpoint": checkpoint, "project": snapshot})
    return checkpoint


def export_version(directory, version_id):
    snapshot = get_version(directory, version_id)
    project = snapshot["project"]
    project["mask_set_version"] = snapshot["checkpoint"]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for entries in project["annotations"].values():
            for annotation in entries:
                key = annotation.pop("history_blob")
                archive.writestr(annotation["mask_file"], binary_png(read_mask(directory, key), compact=False))
        archive.writestr("annotations.json", json.dumps(project, indent=2))
        archive.writestr("README.txt", "This is a complete saved mask-set version. Each PNG is binary: 0=background, 255=foreground.\nFrame indices map to frames[].name. Source images are not included.\n")
    return stream.getvalue()
