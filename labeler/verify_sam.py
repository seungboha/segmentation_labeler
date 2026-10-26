"""SAM2 request validation and real image inference, save, edit and version checks."""
import base64
import io
from pathlib import Path
import tempfile
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image, ImageDraw

import app as backend
from sam_assist import SamUnavailable


def checked(response):
    assert response.status_code == 200, (response.status_code, response.text)
    return response.json()


with tempfile.TemporaryDirectory(prefix="sam-assist-check-") as temporary:
    root = Path(temporary)
    frames = root / "frames";frames.mkdir()
    backend.DATA = root / "projects"
    for index in range(2):
        image = Image.new("RGB", (320, 240), "#dddde8")
        draw = ImageDraw.Draw(image)
        draw.ellipse((35, 50, 135, 190), fill="#c84040", outline="#631515", width=4)
        draw.rectangle((200, 60, 285, 185), fill="#358050")
        image.save(frames / f"{index}.png")
    with TestClient(backend.app) as client:
        p = checked(client.post("/api/open", json={"folder": str(frames)}));pid = p["id"]
        state = checked(client.get("/api/status"))
        assert "sam2" in state
        p = checked(client.post("/api/labels/dental-preset", json={"project": pid}))
        expected = ["teeth_gum", "soft_tissue", "glove", "unusable_region", "background"]
        assert p["labels"] == expected
        p = checked(client.post("/api/labels/dental-preset", json={"project": pid}))
        assert p["labels"] == expected
        checked(client.patch("/api/labels/color", json={"project": pid, "name": "glove", "color": "#123456"}))
        p = checked(client.post("/api/labels/dental-preset", json={"project": pid}))
        assert p["label_colors"]["glove"] == "#123456"
        request = {"project": pid, "frame": 0, "points": [{"x": 80, "y": 120, "label": 1}]}
        assert client.post("/api/predict", json={**request, "project": "stale"}).status_code == 409
        assert client.post("/api/predict", json={**request, "frame": 99}).status_code == 404
        for point in [{"x": -1, "y": 120, "label": 1}, {"x": 320, "y": 120, "label": 1},
                      {"x": 80, "y": 120, "label": 0}]:
            assert client.post("/api/predict", json={**request, "points": [point]}).status_code == 400
        with patch.object(backend.SAM, "predict", side_effect=SamUnavailable("Test: model unavailable")):
            assert client.post("/api/predict", json=request).status_code == 503
        predictor = backend.SAM.get_predictor()
        with patch.object(predictor, "set_image", wraps=predictor.set_image) as embed:
            result = checked(client.post("/api/predict", json=request))
            assert result["pixels"] > 0
            points = [*request["points"], {"x": 10, "y": 10, "label": 0}]
            checked(client.post("/api/predict", json={**request, "points": points}))
            assert embed.call_count == 1, "Repeated prompts on the same frame should reuse embeddings"
            checked(client.post("/api/predict", json={**request, "frame": 1}))
            assert embed.call_count == 2
        mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(result["mask"]))))
        assert mask.shape == (240, 320) and set(np.unique(mask)).issubset({0, 255})
        save = {"project": pid, "frame": 0, "class_name": "teeth_gum", "mask": result["mask"],
                "sam_points": request["points"]}
        p = checked(client.post("/api/save", json=save))
        annotation = p["annotations"]["0"][0]
        assert annotation["source"] == "sam2-assisted" and annotation["sam_points"] == request["points"]
        assert client.post("/api/save", json={**save, "sam_points": [{"x": -1, "y": 10, "label": 1}]}).status_code == 400
        # Manual corrections still save the actual pixels, without rerunning SAM.
        edited = mask.copy();edited[0:10, 0:10] = 255
        stream = io.BytesIO();Image.fromarray(edited).save(stream, format="PNG")
        checked(client.post("/api/save", json={**save, "object_id": annotation["object_id"],
            "mask": base64.b64encode(stream.getvalue()).decode()}))
        png = client.get(f"/api/mask/0/{annotation['object_id']}", params={"project": pid}).content
        assert np.array_equal(np.asarray(Image.open(io.BytesIO(png))), edited)
        version = checked(client.post("/api/versions", json={"project": pid, "name": "SAM and brush"}))["mask_versions"][-1]
        assert client.get(f"/api/versions/{version['id']}/export", params={"project": pid}).status_code == 200

print("PASS: real SAM2 inference, include/exclude prompts, embedding reuse and frame invalidation")
print("PASS: five-label preset, unavailable-model feedback, saved SAM masks, brush corrections and version export")
