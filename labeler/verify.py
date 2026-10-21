"""End-to-end check using real SAM inference and temporary source/project files."""
import io
import base64
import json
import tempfile
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient
import app as backend


def checked(response):
    assert response.status_code == 200, (response.status_code, response.text)
    return response.json()


with tempfile.TemporaryDirectory(prefix="labeler-check-") as tmp:
    root = Path(tmp)
    frames = root / "frames"
    frames.mkdir()
    backend.DATA = root / "projects"
    for number in [10, 2]:
        image = Image.new("RGB", (320, 240), "#dddde8")
        draw = ImageDraw.Draw(image)
        draw.ellipse((70, 50, 240, 210), fill="#c84040", outline="#631515", width=5)
        image.save(frames / f"frame{number}.png")
    with TestClient(backend.app) as client:
        assert client.get("/").status_code == 200
        p = checked(client.post("/api/open", json={"folder": str(frames)}))
        assert [f["name"] for f in p["frames"]] == ["frame2.png", "frame10.png"]
        pid = p["id"]
        assert p["labels"] == []
        p = checked(client.post("/api/labels", json={"project": pid, "name": "circle", "color": "#82B6FF"}))
        assert p["label_colors"]["circle"] == "#82b6ff"
        checked(client.post("/api/labels", json={"project": pid, "name": "unused label"}))
        assert client.post("/api/labels", json={"project": pid, "name": " CIRCLE "}).status_code == 409
        assert client.post("/api/labels", json={"project": pid, "name": "  "}).status_code == 400
        assert client.get("/api/frame/0", params={"project": pid}).status_code == 200
        assert client.get("/api/frame/99", params={"project": pid}).status_code == 404
        request = {"project": pid, "frame": 0, "points": [{"x": 150, "y": 125, "label": 1}, {"x": 20, "y": 20, "label": 0}]}
        assert client.post("/api/predict", json={**request, "project": "stale"}).status_code == 409
        result = checked(client.post("/api/predict", json=request))
        assert result["pixels"] > 0
        save = {"project": pid, "frame": 0, "draft": result["draft"], "class_name": "circle"}
        assert client.post("/api/save", json={**save, "class_name": "unknown"}).status_code == 400
        p = checked(client.post("/api/save", json=save))
        assert p["annotations"]["0"][0]["object_id"] == 1
        png = client.get("/api/mask/0/1", params={"project": pid}).content
        mask = np.asarray(Image.open(io.BytesIO(png)))
        assert mask.shape == (240, 320) and set(np.unique(mask)).issubset({0, 255})
        checked(client.patch("/api/labels/color", json={"project": pid, "name": "circle", "color": "#f46d6d"}))
        checked(client.patch("/api/overlay", json={"project": pid, "opacity": 0.7}))
        assert client.get("/api/mask/0/1", params={"project": pid}).content == png
        assert client.patch("/api/labels/color", json={"project": pid, "name": "circle", "color": "invalid"}).status_code == 422
        assert client.patch("/api/overlay", json={"project": pid, "opacity": 1.5}).status_code == 422
        p = checked(client.post("/api/open", json={"folder": str(frames)}))
        assert p["annotations"]["0"][0]["class_name"] == "circle"
        assert p["labels"] == ["circle", "unused label"]
        assert p["label_colors"]["circle"] == "#f46d6d" and p["overlay_opacity"] == 0.7
        assert client.post("/api/save", json=save).status_code == 409
        result = checked(client.post("/api/predict", json={**request, "frame": 1}))
        checked(client.post("/api/save", json={**save, "frame": 1, "draft": result["draft"], "object_id": 1}))
        # Brush edits are independent of the latest SAM draft (which is on frame 1).
        corrected = mask.copy()
        corrected[10:20, 10:20] = 255
        corrected[120:130, 145:155] = 0
        def as_png(array):
            data = io.BytesIO()
            Image.fromarray(array).save(data, format="PNG")
            return base64.b64encode(data.getvalue()).decode()
        manual = {"project": pid, "frame": 0, "class_name": "circle", "object_id": 1,
                  "mask_png": as_png(corrected), "points": request["points"]}
        fixed = checked(client.post("/api/save", json=manual))
        entry = fixed["annotations"]["0"][0]
        assert entry["manually_edited"] and entry["score"] is None
        actual = np.asarray(Image.open(io.BytesIO(client.get("/api/mask/0/1", params={"project": pid}).content)))
        assert np.array_equal(actual, corrected)
        assert client.post("/api/save", json={**manual, "mask_png": "invalid"}).status_code == 400
        assert client.post("/api/save", json={**manual, "mask_png": as_png(np.zeros((10, 10), dtype=np.uint8))}).status_code == 400
        assert client.post("/api/save", json={**manual, "mask_png": as_png(np.zeros_like(corrected))}).status_code == 400
        checked(client.post("/api/open", json={"folder": str(frames)}))
        actual = np.asarray(Image.open(io.BytesIO(client.get("/api/mask/0/1", params={"project": pid}).content)))
        assert np.array_equal(actual, corrected)
        export = client.get("/api/export", params={"project": pid})
        assert export.status_code == 200
        with zipfile.ZipFile(io.BytesIO(export.content)) as archive:
            metadata = json.loads(archive.read("annotations.json"))
            assert len(metadata["annotations"]) == 2
            exported_mask = Image.open(io.BytesIO(archive.read(metadata["annotations"]["0"][0]["mask_file"])))
            assert np.array_equal(np.asarray(exported_mask), corrected)
            assert metadata["labels"] == ["circle", "unused label"]
            assert metadata["label_colors"]["circle"] == "#f46d6d" and metadata["overlay_opacity"] == 0.7
            assert len([n for n in archive.namelist() if n.endswith(".png")]) == 2
        checked(client.delete("/api/mask/0/1", params={"project": pid}))
        assert client.get("/api/mask/0/1", params={"project": pid}).status_code == 404
        Image.new("RGB", (10, 10)).save(frames / "frame3.png")
        assert client.post("/api/open", json={"folder": str(frames)}).status_code == 409
        def import_images():
            return client.post("/api/import", files=[
                ("files", ("selected/frame10.png", (frames / "frame10.png").read_bytes(), "image/png")),
                ("files", ("selected/nested/frame2.png", (frames / "frame2.png").read_bytes(), "image/png")),
                ("files", ("selected/notes.txt", b"ignore me", "text/plain")),
            ])
        imported = checked(import_images())
        checked(client.post("/api/labels", json={"project": imported["id"], "name": "circle"}))
        assert len(imported["frames"]) == 2
        assert client.get("/api/frame/1", params={"project": imported["id"]}).status_code == 200
        imported_prediction = checked(client.post("/api/predict", json={**request, "project": imported["id"]}))
        checked(client.post("/api/save", json={**save, "project": imported["id"], "draft": imported_prediction["draft"]}))
        reopened = checked(import_images())
        assert reopened["id"] == imported["id"]
        assert reopened["annotations"]["0"][0]["class_name"] == "circle"
        # Simulate a project written before the registry existed.
        manifest = backend.DATA / reopened["id"] / "annotations.json"
        legacy = json.loads(manifest.read_text())
        del legacy["labels"]
        del legacy["label_colors"]
        del legacy["overlay_opacity"]
        manifest.write_text(json.dumps(legacy))
        migrated = checked(import_images())
        assert migrated["labels"] == ["circle"]
        assert migrated["label_colors"] == {"circle": "#68ddbc"} and migrated["overlay_opacity"] == 0.45
        assert client.post("/api/import", files=[("files", ("../escape.png", (frames / "frame2.png").read_bytes(), "image/png"))]).status_code == 400
        assert client.post("/api/import", files=[("files", ("bad.png", b"not an image", "image/png"))]).status_code == 400
        print("PASS: folder import, nested images, ignored non-images, duplicate selection resumes labels, invalid image/path rejection")
        print("PASS: label creation, duplicate rejection, unused label persistence, exported registry, old-project migration")
        print("PASS: manual label colors and opacity persist and export; binary PNG masks remain unchanged")
        print("PASS: brush corrections, erased holes, original-size binary export, reopen, malformed/wrong-size/empty mask rejection")
        print("PASS: real GPU inference, natural frame ordering, PNG masks, save/resume, manual ID reuse, export, delete, stale requests, changed-source protection")
