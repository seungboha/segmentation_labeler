"""Local, single-user SAM 2 frame labeler. Run one Uvicorn worker."""
import base64
from contextlib import nullcontext
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import threading
import uuid
import zipfile

import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("LABELER_DATA", ROOT / "projects"))
LOCK = threading.RLock()
ACTIVE = None
PREDICTOR = None
EMBEDDED = None
DRAFT = None
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
app = FastAPI(title="SAM 2 Frame Labeler")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class OpenRequest(BaseModel):
    folder: str


class LabelRequest(BaseModel):
    project: str
    name: str = Field(min_length=1, max_length=100)
    color: str = Field(default="#68ddbc", pattern=r"^#[0-9a-fA-F]{6}$")


class LabelColorRequest(BaseModel):
    project: str
    name: str
    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")


class OverlayRequest(BaseModel):
    project: str
    opacity: float = Field(ge=0, le=1, allow_inf_nan=False)


class Point(BaseModel):
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)
    label: int = Field(ge=0, le=1)


class PredictRequest(BaseModel):
    project: str
    frame: int = Field(ge=0)
    points: list[Point] = Field(min_length=1, max_length=500)


class SaveRequest(BaseModel):
    project: str
    frame: int = Field(ge=0)
    draft: str
    class_name: str = Field(min_length=1, max_length=100)
    object_id: int | None = Field(default=None, ge=1, le=2147483647)


def atomic_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def require_project(project):
    if ACTIVE is None or ACTIVE["id"] != project:
        raise HTTPException(409, "Project changed. Reopen your frame folder.")
    return ACTIVE


def frame_info(project, index):
    p = require_project(project)
    if index < 0 or index >= len(p["frames"]):
        raise HTTPException(404, "Frame does not exist.")
    return p, p["frames"][index]


def read_frame(p, f):
    path = Path(p["source_folder"]) / f["name"]
    stat = path.stat()
    if stat.st_size != f["size"] or stat.st_mtime_ns != f["mtime_ns"]:
        raise HTTPException(409, "Source image changed. Restore the original or use a new folder.")
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def project_dir(p):
    return DATA / p["id"]


def public_project(p):
    return {**p, "save_directory": str(project_dir(p))}


def get_predictor():
    global PREDICTOR
    if PREDICTOR is None:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        checkpoint = os.environ.get("SAM2_CHECKPOINT", str(ROOT.parent / "sam2/checkpoints/sam2.1_hiera_tiny.pt"))
        config = os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_t.yaml")
        if not Path(checkpoint).is_file():
            raise HTTPException(503, "Checkpoint missing. Set SAM2_CHECKPOINT to its WSL path.")
        device = os.environ.get("SAM2_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        PREDICTOR = SAM2ImagePredictor(build_sam2(config, checkpoint, device=device))
    return PREDICTOR


def png_bytes(mask):
    stream = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(stream, format="PNG")
    return stream.getvalue()


@app.get("/")
def home():
    return FileResponse(ROOT / "static/index.html")


@app.get("/api/status")
def status():
    import torch
    return {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "model": "SAM 2 image predictor", "loaded": PREDICTOR is not None}


@app.post("/api/open")
def open_project(req: OpenRequest):
    global ACTIVE, EMBEDDED, DRAFT
    with LOCK:
        folder_text = req.folder.strip().strip('"')
        if re.match(r"^[A-Za-z]:[\\/]", folder_text):
            folder_text = "/mnt/" + folder_text[0].lower() + "/" + folder_text[3:].replace("\\", "/")
        else:
            folder_text = folder_text.replace("\\", "/")
        folder = Path(folder_text).expanduser().resolve()
        if not folder.is_dir():
            raise HTTPException(400, "Folder not found. Enter a WSL path such as /home/seungbo/frames.")
        files = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        files.sort(key=lambda f: [(1, int(s)) if s.isdigit() else (0, s.lower()) for s in re.split(r"(\d+)", f.relative_to(folder).as_posix())])
        if not files:
            raise HTTPException(400, "No supported images found in this folder or its subfolders.")
        frames = [{"name": f.relative_to(folder).as_posix(), "size": f.stat().st_size, "mtime_ns": f.stat().st_mtime_ns} for f in files]
        pid = hashlib.sha256(str(folder).encode()).hexdigest()[:20]
        dest = DATA / pid
        manifest = dest / "annotations.json"
        if manifest.exists():
            p = json.loads(manifest.read_text(encoding="utf-8"))
            if p["frames"] != frames:
                raise HTTPException(409, "Frame folder changed since labeling. Restore its contents or copy the new sequence to a different folder.")
        else:
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "masks").mkdir(exist_ok=True)
            p = {"version": 1, "id": pid, "source_folder": str(folder), "frames": frames,
                 "annotations": {}, "next_object_id": 1}
            atomic_json(manifest, p)
        # Older projects stored class names only on saved annotations.
        labels = list(dict.fromkeys(p.get("labels", []) + [a["class_name"] for entries in p["annotations"].values() for a in entries]))
        colors = {name: p.get("label_colors", {}).get(name, "#68ddbc") for name in labels}
        opacity = p.get("overlay_opacity", 0.45)
        if p.get("labels") != labels or p.get("label_colors") != colors or "overlay_opacity" not in p:
            p.update(labels=labels, label_colors=colors, overlay_opacity=opacity)
            atomic_json(manifest, p)
        ACTIVE, EMBEDDED, DRAFT = p, None, None
        return public_project(p)


@app.post("/api/labels")
def add_label(req: LabelRequest):
    with LOCK:
        p = require_project(req.project)
        name = " ".join(req.name.split())
        if not name:
            raise HTTPException(400, "Enter a label name.")
        if any(label.casefold() == name.casefold() for label in p["labels"]):
            raise HTTPException(409, "This label already exists. Select it from the list.")
        updated = {**p, "labels": [*p["labels"], name], "label_colors": {**p["label_colors"], name: req.color.lower()}}
        atomic_json(project_dir(p) / "annotations.json", updated)
        p.update(updated)
        return public_project(p)


@app.patch("/api/labels/color")
def set_label_color(req: LabelColorRequest):
    with LOCK:
        p = require_project(req.project)
        if req.name not in p["labels"]:
            raise HTTPException(404, "Label not found.")
        updated = {**p, "label_colors": {**p["label_colors"], req.name: req.color.lower()}}
        atomic_json(project_dir(p) / "annotations.json", updated)
        p.update(updated)
        return public_project(p)


@app.patch("/api/overlay")
def set_overlay(req: OverlayRequest):
    with LOCK:
        p = require_project(req.project)
        updated = {**p, "overlay_opacity": req.opacity}
        atomic_json(project_dir(p) / "annotations.json", updated)
        p.update(updated)
        return public_project(p)


@app.post("/api/import")
async def import_route(request: Request):
    async with request.form(max_files=100000, max_fields=10) as form:
        files = form.getlist("files")
        if not files or not all(isinstance(file, UploadFile) for file in files):
            raise HTTPException(400, "Select an image folder first.")
        return await run_in_threadpool(import_folder, files)


def import_folder(files: list[UploadFile]):
    """Copy a browser-selected image folder into local storage, then open it."""
    with LOCK:
        DATA.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix="import-", dir=DATA) as tmp:
                staged = Path(tmp) / "frames"
                staged.mkdir()
                hashes = {}
                for upload in files:
                    name = (upload.filename or "").replace("\\", "/")
                    relative = PurePosixPath(name)
                    if not name or relative.is_absolute() or ".." in relative.parts or ":" in name:
                        raise HTTPException(400, "Invalid image filename in selected folder.")
                    if relative.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    if name in hashes:
                        raise HTTPException(400, "Duplicate image filename in selected folder.")
                    target = staged.joinpath(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    with target.open("wb") as output:
                        while chunk := upload.file.read(1024 * 1024):
                            output.write(chunk)
                            digest.update(chunk)
                    try:
                        with Image.open(target) as image:
                            image.verify()
                    except (OSError, ValueError, Image.DecompressionBombError):
                        raise HTTPException(400, f"Cannot read image: {name}")
                    hashes[name] = digest.hexdigest()
                if not hashes:
                    raise HTTPException(400, "The selected folder contains no supported images.")
                identity = hashlib.sha256(json.dumps(sorted(hashes.items())).encode()).hexdigest()[:24]
                destination = DATA / "imported_frames" / identity
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.move(str(staged), str(destination))
                return open_project(OpenRequest(folder=str(destination)))
        finally:
            for upload in files:
                upload.file.close()


@app.get("/api/frame/{index}")
def get_frame(index: int, project: str):
    with LOCK:
        p, f = frame_info(project, index)
        image = read_frame(p, f)
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return Response(stream.getvalue(), media_type="image/png")


@app.post("/api/predict")
def predict(req: PredictRequest):
    global EMBEDDED, DRAFT
    with LOCK:
        p, f = frame_info(req.project, req.frame)
        image = read_frame(p, f)
        for point in req.points:
            if not (0 <= point.x < image.width and 0 <= point.y < image.height):
                raise HTTPException(400, "Click coordinates are outside the image.")
        if not any(point.label for point in req.points):
            raise HTTPException(400, "Add at least one positive point inside the object.")
        import torch
        try:
            predictor = get_predictor()
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if predictor.device.type == "cuda" and torch.cuda.is_bf16_supported() else nullcontext()
            with torch.inference_mode(), autocast:
                if EMBEDDED != (p["id"], req.frame):
                    predictor.set_image(np.array(image, copy=True))
                    EMBEDDED = (p["id"], req.frame)
                masks, scores, _ = predictor.predict(
                    point_coords=np.array([[pt.x, pt.y] for pt in req.points], dtype=np.float32),
                    point_labels=np.array([pt.label for pt in req.points], dtype=np.int32),
                    multimask_output=True)
            best = int(np.argmax(scores))
            mask = masks[best].astype(bool)
            token = uuid.uuid4().hex
            DRAFT = {"token": token, "project": p["id"], "frame": req.frame, "mask": mask,
                     "points": [pt.model_dump() for pt in req.points], "score": float(scores[best])}
            return {"draft": token, "mask": base64.b64encode(png_bytes(mask)).decode(),
                    "score": DRAFT["score"], "pixels": int(mask.sum())}
        except torch.cuda.OutOfMemoryError:
            EMBEDDED, DRAFT = None, None
            torch.cuda.empty_cache()
            raise HTTPException(503, "GPU memory is full. Close other GPU applications and retry.")


@app.post("/api/save")
def save_mask(req: SaveRequest):
    with LOCK:
        p, f = frame_info(req.project, req.frame)
        if DRAFT is None or (DRAFT["token"], DRAFT["project"], DRAFT["frame"]) != (req.draft, req.project, req.frame):
            raise HTTPException(409, "Preview is stale. Predict again before saving.")
        label = req.class_name.strip()
        if not label:
            raise HTTPException(400, "Enter a class name.")
        if label not in p["labels"]:
            raise HTTPException(400, "Create the label first, then select it before saving.")
        if not DRAFT["mask"].any():
            raise HTTPException(400, "The mask is empty. Add a positive point and try again.")
        oid = req.object_id or p["next_object_id"]
        key = str(req.frame)
        existing = p["annotations"].get(key, [])
        previous = next((a for a in existing if a["object_id"] == oid), None)
        filename = f"masks/{req.frame:06d}_{oid}_{uuid.uuid4().hex[:12]}.png"
        (project_dir(p) / filename).write_bytes(png_bytes(DRAFT["mask"]))
        annotation = {"object_id": oid, "class_name": label, "mask_file": filename,
                      "width": int(DRAFT["mask"].shape[1]), "height": int(DRAFT["mask"].shape[0]),
                      "points": DRAFT["points"], "score": DRAFT["score"]}
        updated = {**p, "annotations": {**p["annotations"], key: [a for a in existing if a["object_id"] != oid] + [annotation]},
                   "next_object_id": max(p["next_object_id"], oid + 1)}
        atomic_json(project_dir(p) / "annotations.json", updated)
        p.update(updated)
        if previous:
            (project_dir(p) / previous["mask_file"]).unlink(missing_ok=True)
        return public_project(p)


@app.get("/api/mask/{index}/{oid}")
def get_mask(index: int, oid: int, project: str):
    with LOCK:
        p, _ = frame_info(project, index)
        annotation = next((a for a in p["annotations"].get(str(index), []) if a["object_id"] == oid), None)
        if annotation is None:
            raise HTTPException(404, "Mask not found.")
        return Response((project_dir(p) / annotation["mask_file"]).read_bytes(), media_type="image/png")


@app.delete("/api/mask/{index}/{oid}")
def delete_mask(index: int, oid: int, project: str):
    with LOCK:
        p, _ = frame_info(project, index)
        entries = p["annotations"].get(str(index), [])
        annotation = next((a for a in entries if a["object_id"] == oid), None)
        if annotation is None:
            raise HTTPException(404, "Mask not found.")
        updated = {**p, "annotations": {**p["annotations"], str(index): [a for a in entries if a["object_id"] != oid]}}
        atomic_json(project_dir(p) / "annotations.json", updated)
        p.update(updated)
        (project_dir(p) / annotation["mask_file"]).unlink(missing_ok=True)
        return public_project(p)


@app.get("/api/export")
def export(project: str):
    with LOCK:
        p = require_project(project)
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("annotations.json", json.dumps(p, indent=2))
            for annotations in p["annotations"].values():
                for annotation in annotations:
                    archive.write(project_dir(p) / annotation["mask_file"], annotation["mask_file"])
            archive.writestr("README.txt", "Each PNG is an independent object mask: 0=background, 255=foreground.\nFrame indices in annotations.json are zero-based and map to frames[].name.\nMasks have the EXIF-oriented source image dimensions. Object masks may overlap.\nClass names are user-defined. Source images are not included.\n")
        return Response(stream.getvalue(), media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="labeler-annotations.zip"'})
