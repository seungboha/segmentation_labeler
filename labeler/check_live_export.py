"""Verify saved demo masks retain source resolution."""
import io
import json
from urllib.request import urlopen
import zipfile
from PIL import Image

project = "6b4043508e17904e3f07"
with urlopen(f"http://127.0.0.1:8000/api/export?project={project}") as response:
    archive_bytes = response.read()
with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
    metadata = json.loads(archive.read("annotations.json"))
    annotation = next(a for a in metadata["annotations"]["0"] if a["class_name"] == "rectangle")
    with Image.open(io.BytesIO(archive.read(annotation["mask_file"]))) as mask:
        assert mask.size == (960, 640), mask.size
        assert mask.getpixel((670, 320)) == 255
        assert mask.getpixel((285, 290)) == 0
    assert {"circle", "rectangle"}.issubset(metadata["labels"])
print("PASS: exported mask stays 960 x 640 and covers the selected rectangle")
