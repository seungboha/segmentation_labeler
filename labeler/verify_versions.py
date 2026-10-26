"""Whole-mask-set checkpoints: persistence, exact exports, sharing and deltas."""
import base64
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import zipfile

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

import app as backend
import mask_history


def checked(response):
    assert response.status_code == 200, (response.status_code, response.text)
    return response.json()


def encoded(mask):
    stream = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


with tempfile.TemporaryDirectory(prefix="mask-versions-") as temporary:
    root = Path(temporary)
    backend.DATA = root / "projects"
    frames = root / "frames"
    frames.mkdir()
    for index in range(2):
        Image.new("RGB", (512, 512)).save(frames / f"{index}.png")
    with TestClient(backend.app) as client:
        project = checked(client.post("/api/open", json={"folder": str(frames)}))
        pid = project["id"]
        directory = backend.DATA / pid
        assert project["mask_versions"] == []
        checked(client.post("/api/labels", json={"project": pid, "name": "tool"}))
        rng = np.random.default_rng(112)
        wide = rng.integers(0, 2, size=(512, 512)).astype(bool)

        def save(mask, frame=0, oid=1):
            return checked(client.post("/api/save", json={"project": pid, "frame": frame,
                "mask": encoded(mask), "class_name": "tool", "object_id": oid}))

        def checkpoint(name, note=""):
            return checked(client.post("/api/versions", json={"project": pid, "name": name, "note": note}))["mask_versions"][-1]

        def exported(version):
            response = client.get(f"/api/versions/{version['id']}/export", params={"project": pid})
            assert response.status_code == 200
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                metadata = json.loads(archive.read("annotations.json"))
                masks = {}
                for frame, entries in metadata["annotations"].items():
                    for annotation in entries:
                        assert "history_blob" not in annotation
                        with Image.open(io.BytesIO(archive.read(annotation["mask_file"]))) as image:
                            assert image.mode == "L", "Exports must stay ordinary 8-bit grayscale PNGs"
                            pixels = np.asarray(image.convert("L"))
                        assert set(np.unique(pixels)).issubset({0, 255})
                        masks[frame, annotation["object_id"]] = pixels != 0
                assert len([name for name in archive.namelist() if name.endswith(".png")]) == len(masks)
                return metadata, masks

        assert client.post("/api/versions", json={"project": pid, "name": "  "}).status_code == 400
        assert client.post("/api/versions", json={"project": "stale", "name": "Wide"}).status_code == 409
        assert client.get("/api/versions/invalid/export", params={"project": pid}).status_code == 404
        assert client.get(f"/api/versions/{'0' * 32}/export", params={"project": pid}).status_code == 404
        empty = checkpoint("Empty")
        assert empty["mask_count"] == 0

        save(wide)
        save(wide, frame=1, oid=7)
        v1 = checkpoint("Wide", "Initial complete set")
        assert v1["mask_count"] == 2
        files_before = set((directory / "history" / "blobs").iterdir())
        unchanged = checkpoint("Same pixels")
        assert set((directory / "history" / "blobs").iterdir()) == files_before

        narrow = wide.copy()
        narrow[50:52, 60:62] ^= True
        save(narrow)
        checked(client.delete("/api/mask/1/7", params={"project": pid}))
        save(narrow, frame=1, oid=9)
        checked(client.patch("/api/labels/color", json={"project": pid, "name": "tool", "color": "#f46d6d"}))
        checked(client.patch("/api/overlay", json={"project": pid, "opacity": 0.7}))
        v2 = checkpoint("Narrow")
        assert list((directory / "history" / "blobs").glob("*.npz")), "Small edits should use compressed deltas"
        assert len(list((directory / "history" / "blobs").glob("*.json"))) == 2
        # Simulate closing the active project, then reopening it.
        backend.ACTIVE = None
        reopened = checked(client.post("/api/open", json={"folder": str(frames)}))
        assert len(reopened["mask_versions"]) == 4
        assert reopened["annotations"]["1"][0]["object_id"] == 9
        for version, expected, color, object_id in [(v1, wide, "#68ddbc", 7), (v2, narrow, "#f46d6d", 9)]:
            metadata, masks = exported(version)
            assert metadata["mask_set_version"]["id"] == version["id"]
            assert metadata["label_colors"]["tool"] == color
            assert set(masks) == {("0", 1), ("1", object_id)}
            assert all(np.array_equal(mask, expected) for mask in masks.values())

        # A series of small changes reconstructs exactly and periodically stores a full PNG.
        current = narrow.copy()
        for index in range(mask_history.MAX_DELTA_DEPTH + 2):
            current[100, index] ^= True
            save(current)
            version = checkpoint(f"Correction {index}")
            _, masks = exported(version)
            assert np.array_equal(masks["0", 1], current)
        assert len(list((directory / "history" / "blobs").glob("*.png"))) >= 2

        # No-op working saves share storage even though their working filenames change.
        before = set((directory / "history" / "blobs").iterdir())
        save(current)
        checkpoint("No pixel changes")
        assert set((directory / "history" / "blobs").iterdir()) == before

        # A failed manifest publish must not expose a partial version.
        before = mask_history.list_versions(directory)
        real_write = mask_history.atomic_write
        def interrupted(path, data):
            if path.parent.name == "versions":
                raise OSError("Simulated disk failure")
            return real_write(path, data)
        with patch.object(mask_history, "atomic_write", side_effect=interrupted):
            try:
                mask_history.save_version(directory, backend.ACTIVE, "Interrupted", "")
            except OSError:
                pass
            else:
                raise AssertionError("Expected simulated failure")
        assert mask_history.list_versions(directory) == before
        _, masks = exported(v1)
        assert np.array_equal(masks["0", 1], wide)

print("PASS: whole-set versions across frames, additions/deletions, immutable labels, reopen, exact PNG exports")
print("PASS: identical-mask sharing, compressed XOR deltas, bounded chains, no-op saves, failed-save atomicity")
