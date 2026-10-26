"""CVAT XML import: real RLE pixels, polygon groups, mapping and atomicity."""
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


def checked(response):
    assert response.status_code == 200, (response.status_code, response.text)
    return response.json()


XML = '''<?xml version="1.0" encoding="utf-8"?>
<annotations><version>1.1</version><meta><task><labels>
<label><name>TOOL</name><color>#ff0000</color></label>
<label><name>tissue</name><color>#123456</color></label>
</labels></task></meta>
<image id="99" name="frame2.png" width="8" height="6">
<mask label="TOOL" rle="0,2,2,1,1" left="1" top="2" width="3" height="2" track_id="0"/>
<polygon label="tissue" points="0,0;1,0;1,1;0,1" group_id="5"/>
<polygon label="tissue" points="6,4;7,4;7,5;6,5" group_id="5"/>
<box label="not-a-mask" xtl="0" ytl="0" xbr="7" ybr="5"/>
</image>
<image id="0" name="frame10.png" width="8" height="6">
<mask label="TOOL" rle="1,2,3" left="2" top="1" width="3" height="2" track_id="0"/>
</image></annotations>'''


with tempfile.TemporaryDirectory(prefix="cvat-check-") as temporary:
    root = Path(temporary)
    frames = root / "frames" / "selected"
    frames.mkdir(parents=True)
    for name in ["frame2.png", "frame10.png"]:
        Image.new("RGB", (8, 6)).save(frames / name)
    backend.DATA = root / "projects"
    backend.ACTIVE = None
    with TestClient(backend.app) as client:
        p = checked(client.post("/api/open", json={"folder": str(frames.parent)}))
        pid = p["id"]
        checked(client.post("/api/labels", json={"project": pid, "name": "tool", "color": "#abcdef"}))
        # Existing object must survive additive imports and cancelled previews.
        png = io.BytesIO();Image.new("L", (8, 6), 255).save(png, format="PNG")
        p = checked(client.post("/api/save", json={"project": pid, "frame": 0,
            "mask": base64.b64encode(png.getvalue()).decode(), "class_name": "tool"}))
        version = checked(client.post("/api/versions", json={"project": pid, "name": "Before XML"}))["mask_versions"][0]
        directory = backend.DATA / pid
        manifest = directory / "annotations.json"

        def preview(xml=XML, project=pid):
            return client.post("/api/import/cvat/preview", data={"project": project},
                files={"file": ("annotations.xml", xml.encode() if isinstance(xml, str) else xml, "application/xml")})

        def apply(result):
            return client.post("/api/import/cvat/apply", json={"project": pid, "token": result["token"]})

        before = manifest.read_bytes()
        first = checked(preview())
        assert first["mask_count"] == 3 and first["frame_count"] == 2
        assert first["skipped_shapes"] == {"box": 1}
        assert manifest.read_bytes() == before
        checked(client.delete(f"/api/import/cvat/{first['token']}", params={"project": pid}))
        assert not list(directory.glob("cvat-import-*"))
        assert apply(first).status_code == 409
        assert manifest.read_bytes() == before

        invalid = [b"bad xml", b"<html/>", XML.replace('width="8"', 'width="9"', 1),
            XML.replace('frame2.png', 'missing.png'), XML.replace('0,2,2,1,1', '0,999999999999'),
            XML.replace('0,2,2,1,1', '-1,7'), XML.replace('left="1"', 'left="7"', 1),
            XML.replace('0,0;1,0;1,1;0,1', 'NaN,0;1,0;1,1'),
            XML.replace('0,0;1,0;1,1;0,1', '0,0;1,0'),
            '<annotations><version>1.1</version><track id="0"/></annotations>',
            XML.replace('<task>', '<task><flipped>True</flipped>'),
            b'<!DOCTYPE annotations [<!ENTITY x "test">]><annotations><version>1.1</version>&x;</annotations>',
            '<!DOCTYPE annotations><annotations><version>1.1</version></annotations>'.encode('utf-16')]
        for xml in invalid:
            response = preview(xml)
            assert response.status_code == 400, response.text
            assert manifest.read_bytes() == before
        assert not list(directory.glob("cvat-import-*"))
        assert preview(project="stale").status_code == 409

        # A valid preview becomes stale if another tab changes the project.
        stale = checked(preview())
        checked(client.patch("/api/overlay", json={"project": pid, "opacity": 0.6}))
        assert apply(stale).status_code == 409

        # Simulate disk failure: none of the imported masks or annotations may leak into the project.
        result = checked(preview())
        before = manifest.read_bytes()
        existing = set((directory / "masks").iterdir())
        with patch.object(backend, "atomic_json", side_effect=OSError("Simulated failure")):
            try:
                apply(result)
            except OSError:
                pass
            else:
                raise AssertionError("Expected simulated failure")
        assert manifest.read_bytes() == before
        assert set((directory / "masks").iterdir()) == existing

        imported = checked(apply(result))["project"]
        assert len(imported["annotations"]["0"]) == 3
        assert len(imported["annotations"]["1"]) == 1
        assert imported["labels"] == list(backend.DEFAULT_LABEL_COLORS) + ["tool", "tissue"]
        assert imported["label_colors"] == {**backend.DEFAULT_LABEL_COLORS, "tool": "#abcdef", "tissue": "#123456"}
        a = next(a for a in imported["annotations"]["0"] if a.get("source") == "cvat" and a["class_name"] == "tool")
        b = imported["annotations"]["1"][0]
        assert a["object_id"] == b["object_id"]  # CVAT track_id=0 survives across images.
        pixels = np.asarray(Image.open(io.BytesIO(client.get(f"/api/mask/0/{a['object_id']}", params={"project": pid}).content)))
        expected = np.zeros((6, 8), dtype=np.uint8)
        expected[2, 1:3] = 255;expected[3, 2] = 255
        assert np.array_equal(pixels, expected), pixels
        tissue = next(a for a in imported["annotations"]["0"] if a["class_name"] == "tissue")
        pixels = np.asarray(Image.open(io.BytesIO(client.get(f"/api/mask/0/{tissue['object_id']}", params={"project": pid}).content)))
        assert pixels[0, 0] == pixels[5, 7] == 255 and pixels[3, 4] == 0
        assert apply(result).status_code == 409
        assert not list(directory.glob("cvat-import-*"))

        # Imported objects remain ordinary editable masks, persist, and can be versioned.
        edited = np.zeros((6, 8), dtype=np.uint8);edited[0, 7] = 255
        output = io.BytesIO();Image.fromarray(edited).save(output, format="PNG")
        checked(client.post("/api/save", json={"project": pid, "frame": 0, "object_id": a["object_id"],
            "mask": base64.b64encode(output.getvalue()).decode(), "class_name": "tool"}))
        backend.ACTIVE = None
        reopened = checked(client.post("/api/open", json={"folder": str(frames.parent)}))
        assert len(reopened["annotations"]["0"]) == 3
        with zipfile.ZipFile(io.BytesIO(client.get(f"/api/versions/{version['id']}/export", params={"project": pid}).content)) as archive:
            metadata = json.loads(archive.read("annotations.json"))
            assert sum(map(len, metadata["annotations"].values())) == 1
        after = checked(client.post("/api/versions", json={"project": pid, "name": "After XML"}))["mask_versions"][-1]
        with zipfile.ZipFile(io.BytesIO(client.get(f"/api/versions/{after['id']}/export", params={"project": pid}).content)) as archive:
            metadata = json.loads(archive.read("annotations.json"))
            assert sum(map(len, metadata["annotations"].values())) == 4

        # Duplicate basenames require sufficiently specific paths; never guess.
        (frames / "nested").mkdir();Image.new("RGB", (8, 6)).save(frames / "nested" / "frame2.png")
        other = checked(client.post("/api/open", json={"folder": str(frames)}))
        # Exact relative names still select the intended frame.
        exact = checked(preview(project=other["id"]))
        assert exact["mask_count"] == 3
        ambiguous = root / "ambiguous"
        for name in ["a/frame2.png", "b/frame2.png", "frame10.png"]:
            target = ambiguous / name;target.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 6)).save(target)
        other = checked(client.post("/api/open", json={"folder": str(ambiguous)}))
        assert preview(project=other["id"]).status_code == 400
        backend.clear_pending_import()

print("PASS: CVAT RLE offsets and row order, grouped polygons, labels/colors, track IDs and filename mapping")
print("PASS: preview/cancel, atomic additive import, stale requests, invalid XML/RLE/dimensions and ambiguous names")
print("PASS: imported-mask editing, persistence, whole-set version export and unchanged earlier checkpoints")
