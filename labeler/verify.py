"""End-to-end mask import, manual editing, persistence and export checks."""
import io
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
        home = client.get("/", headers={"If-Modified-Since": "Thu, 31 Dec 2099 23:59:59 GMT"})
        assert home.status_code == 200 and home.headers["cache-control"] == "no-store"
        assert '/static/app.js?v=' in home.text
        assert '/static/controls.css?v=' in home.text
        script = client.get("/static/app.js")
        assert script.headers["cache-control"] == "no-cache, must-revalidate"
        p = checked(client.post("/api/open", json={"folder": str(frames)}))
        assert [f["name"] for f in p["frames"]] == ["frame2.png", "frame10.png"]
        pid = p["id"]
        defaults = ["teeth_gum"]
        assert p["labels"] == defaults
        assert p["label_colors"] == {"teeth_gum": "#68ddbc"}
        p = checked(client.post("/api/labels", json={"project": pid, "name": "circle", "color": "#82B6FF"}))
        assert p["label_colors"]["circle"] == "#82b6ff"
        checked(client.post("/api/labels", json={"project": pid, "name": "unused label"}))
        assert client.post("/api/labels", json={"project": pid, "name": " CIRCLE "}).status_code == 409
        assert client.post("/api/labels", json={"project": pid, "name": "  "}).status_code == 400
        assert client.get("/api/frame/0", params={"project": pid}).status_code == 200
        assert client.get("/api/frame/99", params={"project": pid}).status_code == 404
        buffer = io.BytesIO()
        source = np.zeros((240, 320), dtype=np.uint8)
        source[50:150, 70:200] = 1
        Image.fromarray(source).save(buffer, format="PNG")
        def load_mask(pid=pid, index=0, content=None):
            return client.post("/api/load-mask", data={"project": pid, "frame": str(index)},
                               files={"file": ("mask.png", buffer.getvalue() if content is None else content, "image/png")})
        assert load_mask(pid="stale").status_code == 409
        assert load_mask(index=99).status_code == 404
        assert load_mask(content=b"bad image").status_code == 400
        wrong = io.BytesIO()
        Image.new("L", (10, 10), 255).save(wrong, format="PNG")
        assert load_mask(content=wrong.getvalue()).status_code == 400
        result = checked(load_mask())
        assert result["pixels"] == 13000
        save = {"project": pid, "frame": 0, "mask": result["mask"], "class_name": "circle"}
        assert client.post("/api/save", json={**save, "class_name": "unknown"}).status_code == 400
        assert client.post("/api/save", json={**save, "mask": "!!!"}).status_code == 400
        p = checked(client.post("/api/save", json=save))
        assert p["annotations"]["0"][0]["object_id"] == 1
        # Older brush/SAM frontends send mask_png and points, with a null draft.
        legacy_points = [{"x": 150, "y": 125, "label": 1}, {"x": 20, "y": 20, "label": 0}]
        legacy_save = {"project": pid, "frame": 0, "mask_png": save["mask"],
                       "points": legacy_points, "draft": None, "class_name": "circle", "object_id": 1}
        p = checked(client.post("/api/save", json=legacy_save))
        assert len(p["annotations"]["0"]) == 1
        assert p["annotations"]["0"][0]["sam_points"] == legacy_points
        assert client.post("/api/save", json={**legacy_save, "mask_png": "!!!"}).status_code == 400
        assert client.post("/api/save", json={**legacy_save, "points": [{"x": -1, "y": 0, "label": 1}]}).status_code == 400
        missing_mask = {key: value for key, value in legacy_save.items() if key != "mask_png"}
        assert client.post("/api/save", json=missing_mask).status_code == 422
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
        assert p["labels"] == defaults + ["circle", "unused label"]
        assert p["label_colors"]["circle"] == "#f46d6d" and p["overlay_opacity"] == 0.7
        # Edit actual saved pixels and replace the same object.
        edited = mask.copy()
        edited[60:80, 80:100] = 0
        edited[170:180, 200:210] = 255
        output = io.BytesIO()
        Image.fromarray(edited).save(output, format="PNG")
        import base64
        edited_mask = base64.b64encode(output.getvalue()).decode()
        p = checked(client.post("/api/save", json={**save, "mask": edited_mask, "object_id": 1}))
        assert len(p["annotations"]["0"]) == 1
        actual = np.asarray(Image.open(io.BytesIO(client.get("/api/mask/0/1", params={"project": pid}).content)))
        assert np.array_equal(actual, edited)
        checked(client.post("/api/save", json={**save, "frame": 1, "object_id": 1}))
        export = client.get("/api/export", params={"project": pid})
        assert export.status_code == 200
        with zipfile.ZipFile(io.BytesIO(export.content)) as archive:
            metadata = json.loads(archive.read("annotations.json"))
            assert len(metadata["annotations"]) == 2
            assert metadata["labels"] == defaults + ["circle", "unused label"]
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
        checked(client.post("/api/save", json={**save, "project": imported["id"]}))
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
        print("PASS: mask import and manual pixel replacement, natural frame ordering, PNG masks, save/resume, manual ID reuse, export, delete, stale project requests, changed-source protection")
