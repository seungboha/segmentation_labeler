"""Local, single-user mask editor. Run one Uvicorn worker."""
import base64
import binascii
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
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import AliasChoices, BaseModel, Field
import mask_history
import cvat_import
import training_workflow as training
from sam_assist import SamAssist, SamUnavailable

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("LABELER_DATA", ROOT / "projects"))
LOCK = threading.RLock()
ACTIVE = None
PENDING_IMPORT = None
SAM = SamAssist(ROOT)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_LABEL_COLORS = {"teeth_gum": "#68ddbc", "soft_tissue": "#ff91ac",
                        "glove": "#82b6ff", "unusable_region": "#f6bf70", "background": "#95a4b9"}
app = FastAPI(title="Frame Mask Editor")
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


class PresetRequest(BaseModel):
    project: str


class SaveRequest(BaseModel):
    project: str
    frame: int = Field(ge=0)
    mask: str = Field(min_length=1, max_length=100_000_000,
                      validation_alias=AliasChoices("mask", "mask_png"))
    class_name: str = Field(min_length=1, max_length=100)
    object_id: int | None = Field(default=None, ge=1, le=2147483647)
    sam_points: list[Point] = Field(default_factory=list, max_length=500,
                                  validation_alias=AliasChoices("sam_points", "points"))


class VersionRequest(BaseModel):
    project: str
    name: str = Field(min_length=1, max_length=100)
    note: str = Field(default="", max_length=2000)


class ApplyImportRequest(BaseModel):
    project: str
    token: str


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
    return {**p, "save_directory": str(project_dir(p)),
            "mask_versions": mask_history.list_versions(project_dir(p))}


def decode_mask(data, size):
    try:
        with Image.open(io.BytesIO(data)) as source:
            mask = ImageOps.exif_transpose(source)
            if mask.size != size:
                raise HTTPException(400, f"Mask must match the frame dimensions: {size[0]} x {size[1]}.")
            # Preserve low-valued foreground (e.g. 0/1 and 16-bit masks).
            if mask.mode in ("1", "L", "I", "F") or mask.mode.startswith("I;16"):
                return np.asarray(mask) != 0
            rgba = np.asarray(mask.convert("RGBA"))
            return np.any(rgba[:, :, :3] != 0, axis=2) & (rgba[:, :, 3] != 0)
    except (OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, "Cannot read mask. Choose a valid mask image.")


def png_bytes(mask):
    stream = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(stream, format="PNG")
    return stream.getvalue()


@app.get("/")
def home():
    # A cached SAM frontend can understand the prediction PNG but not the save
    # protocol. Serve fresh HTML and give every asset a content-specific URL.
    html = (ROOT / "static/index.html").read_text(encoding="utf-8")
    for name in ("app.js", "style.css", "controls.css"):
        version = hashlib.sha256((ROOT / "static" / name).read_bytes()).hexdigest()[:16]
        html = html.replace(f'/static/{name}"', f'/static/{name}?v={version}"')
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.middleware("http")
async def revalidate_assets(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


@app.get("/api/status")
def status():
    return {"device": "Mask editor · SAM2 optional", "sam2": SAM.status()}


@app.post("/api/open")
def open_project(req: OpenRequest):
    global ACTIVE
    with LOCK:
        folder_text = req.folder.strip().strip('"')
        if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", folder_text):
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
                 "annotations": {}, "next_object_id": 1,
                 "labels": ["teeth_gum"], "label_colors": {"teeth_gum": DEFAULT_LABEL_COLORS["teeth_gum"]}}
            atomic_json(manifest, p)
        # Older projects stored class names only on saved annotations.
        labels = list(dict.fromkeys(p.get("labels", []) + [a["class_name"] for entries in p["annotations"].values() for a in entries]))
        colors = {name: p.get("label_colors", {}).get(name, "#68ddbc") for name in labels}
        opacity = p.get("overlay_opacity", 0.45)
        if p.get("labels") != labels or p.get("label_colors") != colors or "overlay_opacity" not in p:
            p.update(labels=labels, label_colors=colors, overlay_opacity=opacity)
            atomic_json(manifest, p)
        clear_pending_import()
        SAM.embedded = None
        ACTIVE = p
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


@app.post("/api/labels/dental-preset")
def dental_labels(req: PresetRequest):
    with LOCK:
        p = require_project(req.project)
        labels, colors = list(p["labels"]), dict(p["label_colors"])
        for name, color in DEFAULT_LABEL_COLORS.items():
            if not any(label.casefold() == name.casefold() for label in labels):
                labels.append(name)
                colors[name] = color
        updated = {**p, "labels": labels, "label_colors": colors}
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


@app.post("/api/load-mask")
async def load_mask(request: Request):
    async with request.form(max_files=1, max_fields=2) as form:
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise HTTPException(400, "Choose a mask image.")
        try:
            index = int(form.get("frame", ""))
        except (ValueError, TypeError):
            raise HTTPException(400, "Select a valid frame.")
        data = await upload.read()
        def preview():
            with LOCK:
                p, f = frame_info(str(form.get("project", "")), index)
                mask = decode_mask(data, read_frame(p, f).size)
                return {"mask": base64.b64encode(png_bytes(mask)).decode(), "pixels": int(mask.sum())}
        return await run_in_threadpool(preview)


def clear_pending_import():
    global PENDING_IMPORT
    if PENDING_IMPORT is not None:
        PENDING_IMPORT["temporary"].cleanup()
        PENDING_IMPORT = None


def project_signature(p):
    return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()


@app.post("/api/import/cvat/preview")
async def preview_cvat(request: Request):
    async with request.form(max_files=1, max_fields=1) as form:
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise HTTPException(400, "Choose a CVAT XML file.")
        data = await upload.read(64 * 1024 * 1024 + 1)
        if len(data) > 64 * 1024 * 1024:
            raise HTTPException(400, "CVAT XML must be 64 MiB or smaller.")
        return await run_in_threadpool(stage_cvat, str(form.get("project", "")), data, upload.filename or "annotations.xml")


def stage_cvat(project, data, filename):
    global PENDING_IMPORT
    with LOCK:
        p = require_project(project)
        clear_pending_import()
        temporary = tempfile.TemporaryDirectory(prefix="cvat-import-", dir=project_dir(p))
        try:
            staged = cvat_import.stage(data, p["frames"], lambda index: read_frame(p, p["frames"][index]).size,
                                       Path(temporary.name))
        except Exception as error:
            temporary.cleanup()
            if isinstance(error, cvat_import.ImportError):
                raise HTTPException(400, str(error))
            raise
        token = uuid.uuid4().hex
        PENDING_IMPORT = {**staged, "temporary": temporary, "token": token, "project": p["id"],
                          "signature": project_signature(p), "filename": PurePosixPath(filename.replace("\\", "/")).name,
                          "sha256": hashlib.sha256(data).hexdigest()}
        return {"token": token, "filename": PENDING_IMPORT["filename"], **staged["summary"]}


@app.delete("/api/import/cvat/{token}")
def cancel_cvat(token: str, project: str):
    with LOCK:
        require_project(project)
        if PENDING_IMPORT and PENDING_IMPORT["project"] == project and PENDING_IMPORT["token"] == token:
            clear_pending_import()
        return {"cancelled": True}


@app.post("/api/import/cvat/apply")
def apply_cvat(req: ApplyImportRequest):
    with LOCK:
        p = require_project(req.project)
        pending = PENDING_IMPORT
        if not pending or pending["project"] != p["id"] or pending["token"] != req.token:
            raise HTTPException(409, "Import preview expired. Choose the XML file again.")
        if pending["signature"] != project_signature(p):
            raise HTTPException(409, "Project changed after the preview. Choose the XML file again.")
        updated = json.loads(json.dumps(p))
        canonical = {name.casefold(): name for name in updated["labels"]}
        for name in pending["summary"]["labels"]:
            if name.casefold() not in canonical:
                canonical[name.casefold()] = name
                updated["labels"].append(name)
                updated["label_colors"][name] = pending["colors"].get(name, "#68ddbc")
        tracks, written = {}, []
        try:
            for item in pending["items"]:
                track_key = (item["track"], item["label"].casefold()) if item["track"] is not None else None
                oid = tracks.get(track_key) if track_key is not None else None
                if oid is None:
                    oid = updated["next_object_id"]
                    if oid > 2147483647:
                        raise HTTPException(400, "No object IDs are available for this import.")
                    updated["next_object_id"] += 1
                    if track_key is not None:
                        tracks[track_key] = oid
                filename = f"masks/{item['frame']:06d}_{oid}_{uuid.uuid4().hex[:12]}.png"
                target = project_dir(p) / filename
                written.append(target)
                shutil.copyfile(Path(pending["temporary"].name) / item["file"], target)
                annotation = {"object_id": oid, "class_name": canonical[item["label"].casefold()],
                              "mask_file": filename, "width": item["width"], "height": item["height"],
                              "source": "cvat", "cvat": {"file": pending["filename"], "sha256": pending["sha256"],
                                "image": item["image_name"], "group_id": item["group"], "track_id": item["track"]}}
                updated["annotations"].setdefault(str(item["frame"]), []).append(annotation)
            training.invalidate(updated, [item['frame'] for item in pending['items']])
            atomic_json(project_dir(p) / "annotations.json", updated)
        except Exception:
            for path in written:
                path.unlink(missing_ok=True)
            raise
        p.update(updated)
        summary = pending["summary"]
        clear_pending_import()
        return {"project": public_project(p), "summary": summary}


@app.post("/api/predict")
def predict(req: PredictRequest):
    with LOCK:
        p, f = frame_info(req.project, req.frame)
        image = read_frame(p, f)
        if any(not (0 <= point.x < image.width and 0 <= point.y < image.height) for point in req.points):
            raise HTTPException(400, "Prompt coordinates are outside the image.")
        if not any(point.label == 1 for point in req.points):
            raise HTTPException(400, "Add an include point inside the desired region first.")
        try:
            mask, score, device = SAM.predict(image, (p["id"], req.frame), req.points)
        except SamUnavailable as error:
            raise HTTPException(503, str(error))
        return {"mask": base64.b64encode(png_bytes(mask)).decode(), "score": score,
                "pixels": int(mask.sum()), "device": device}


@app.post("/api/save")
def save_mask(req: SaveRequest):
    with LOCK:
        p, f = frame_info(req.project, req.frame)
        try:
            data = base64.b64decode(req.mask, validate=True)
        except (ValueError, binascii.Error):
            raise HTTPException(400, "Invalid mask encoding.")
        mask = decode_mask(data, read_frame(p, f).size)
        if any(not (0 <= point.x < mask.shape[1] and 0 <= point.y < mask.shape[0]) for point in req.sam_points):
            raise HTTPException(400, "Saved SAM2 prompt coordinates are outside the image.")
        label = req.class_name.strip()
        if not label:
            raise HTTPException(400, "Enter a class name.")
        if label not in p["labels"]:
            raise HTTPException(400, "Create the label first, then select it before saving.")
        if not mask.any():
            raise HTTPException(400, "The mask is empty. Paint foreground before saving.")
        oid = req.object_id or p["next_object_id"]
        key = str(req.frame)
        existing = p["annotations"].get(key, [])
        previous = next((a for a in existing if a["object_id"] == oid), None)
        filename = f"masks/{req.frame:06d}_{oid}_{uuid.uuid4().hex[:12]}.png"
        (project_dir(p) / filename).write_bytes(png_bytes(mask))
        annotation = {"object_id": oid, "class_name": label, "mask_file": filename,
                      "width": int(mask.shape[1]), "height": int(mask.shape[0]),
                      "source": "sam2-assisted" if req.sam_points else "manual"}
        if req.sam_points:
            annotation["sam_points"] = [point.model_dump() for point in req.sam_points]
        updated = {**p, "annotations": {**p["annotations"], key: [a for a in existing if a["object_id"] != oid] + [annotation]},
                   "next_object_id": max(p["next_object_id"], oid + 1)}
        training.invalidate(updated, [req.frame])
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
        training.invalidate(updated, [index])
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


@app.post("/api/versions")
def save_version(req: VersionRequest):
    with LOCK:
        p = require_project(req.project)
        name = " ".join(req.name.split())
        if not name:
            raise HTTPException(400, "Enter a version name.")
        mask_history.save_version(project_dir(p), p, name, req.note.strip())
        return public_project(p)


@app.get("/api/versions/{version_id}/export")
def export_version(version_id: str, project: str):
    with LOCK:
        p = require_project(project)
        try:
            data = mask_history.export_version(project_dir(p), version_id)
        except FileNotFoundError:
            raise HTTPException(404, "Version not found.")
        return Response(data, media_type="application/zip", headers={
            "Content-Disposition": f'attachment; filename="mask-version-{version_id[:8]}.zip"'})


class ReadyRequest(BaseModel):
    project: str
    frame: int = Field(ge=0)
    ready: bool = True
    allow_empty: bool = False
    subject_id: int | None = Field(default=None, ge=0)


class TrainRequest(BaseModel):
    project: str
    python: str
    config: str
    checkpoint: str
    validation_subjects: list[int] = Field(min_length=1)
    epochs: int = Field(ge=1, le=10000)
    learning_rate: float = Field(gt=0, le=0.1, allow_inf_nan=False)


class ActivateModelRequest(BaseModel):
    project: str
    job_id: str = Field(pattern=r'^[0-9a-f]{32}$')


@app.post('/api/training/ready')
def mark_ready(req: ReadyRequest):
    with LOCK:
        p, f = frame_info(req.project, req.frame)
        updated = {**p}
        try:
            if req.ready:
                training.approve(project_dir(p), updated, req.frame, read_frame(p, f), req.allow_empty, req.subject_id)
            else:
                training.invalidate(updated, [req.frame])
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error))
        atomic_json(project_dir(p) / 'annotations.json', updated)
        p.update(updated)
        return public_project(p)


@app.get('/api/training')
def training_status(project: str):
    with LOCK:
        p = require_project(project)
        records = list(p.get('training_approvals', {}).values())
        defaults = training.defaults()
        if p.get('active_model'):
            defaults['checkpoint'] = p['active_model']['checkpoint']
        return {'defaults': defaults, 'ready_count': len(records),
                'subjects': [{'id': sid, 'count': sum(r['subject'] == sid for r in records)}
                             for sid in sorted({r['subject'] for r in records})],
                'jobs': training.jobs(project), 'active_model': p.get('active_model')}


@app.post('/api/training/start')
def start_training(req: TrainRequest):
    with LOCK:
        p = require_project(req.project)
        try:
            return training.start(project_dir(p), p, req.model_dump())
        except (ValueError, OSError, KeyError, ImportError) as error:
            raise HTTPException(400, str(error))


@app.get('/api/training/{job_id}/log')
def training_log(job_id: str, project: str):
    with LOCK:
        require_project(project)
        job = next((j for j in training.jobs(project) if j['id'] == job_id), None)
        if job is None:
            raise HTTPException(404, 'Training job not found.')
        path = training.LOGS / job_id / 'console.log'
        if not path.exists():
            return {'text': ''}
        with path.open('rb') as file:
            file.seek(max(0, path.stat().st_size - 16000))
            return {'text': file.read().decode('utf-8', errors='replace')}


@app.post('/api/training/activate')
def activate_model(req: ActivateModelRequest):
    with LOCK:
        p = require_project(req.project)
        job = next((j for j in training.jobs(req.project) if j['id'] == req.job_id), None)
        if not job or job['status'] != 'completed' or not (Path(job['checkpoint']) / 'config.json').is_file():
            raise HTTPException(400, 'Only a successfully trained model can be selected.')
        updated = {**p, 'active_model': {'job_id': job['id'], 'checkpoint': job['checkpoint']}}
        atomic_json(project_dir(p) / 'annotations.json', updated)
        p.update(updated)
        return public_project(p)
