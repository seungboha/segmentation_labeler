# Frame mask editor

Local Python backend and browser canvas for loading, editing and SAM2-assisted
annotation of object masks. Manual editing and CVAT import work without a model.

## Start

Install Python dependencies and start one server worker:

```sh
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --workers 1
```

Open http://localhost:8000. On Linux / WSL, `bash run.sh` also starts the server
with `python3`; set `LABELER_PYTHON` to use another interpreter. `PORT` defaults
to 8000. `python start_background.py` runs detached and writes server.log and
server.pid. The app supports one active labeling browser.

## Load and edit masks

1. Choose an image folder, or open a Windows / WSL folder path. Images in
   subfolders are naturally sorted. Selecting the same unchanged folder resumes
   saved annotations. Browser-selected images are copied into project storage.
2. New projects start with teeth_gum, soft_tissue, glove, unusable_region and
   background. Select one from the label list, or add a custom label and color.
3. Click **Load mask for this frame** and select a PNG, TIFF, BMP, or WebP mask.
   Each file represents one object: zero / black is background and nonzero pixels
   are foreground. Transparent pixels are background. Multi-valued masks are
   treated as one foreground object. The mask must match the EXIF-oriented frame
   dimensions exactly; it is never resized. Loading creates an unsaved preview.
4. Drag with **Brush** to add foreground or **Eraser** to remove it. Right drag
   also erases. Brush diameter is measured in original image pixels, independent
   of zoom. **Undo edit** (Z) restores up to 20 edits, including **Clear mask**.
   You can also paint a new mask from scratch.
5. Click **Save object** (Enter) to commit. An object ID is assigned automatically;
   reuse an ID across frames to identify the same object manually. Click
   **Edit mask** on a saved object to edit its actual pixels, including masks
   from older projects. Save replaces that object's mask on the current frame.
6. Navigate with Previous / Next, the slider, or arrow keys. Unsaved changes
   trigger a discard prompt. Download masks and JSON when ready.

Display zoom, fit to view, label colors, and overlay opacity affect previews.
PNG exports always retain the original dimensions and binary values 0 / 255.
Independent object masks may overlap. Empty masks cannot be saved; use Delete
on a saved object to remove it.

## Storage and verification

Projects live in `projects/<folder-hash>/annotations.json`, with PNGs under
`masks/`. Set `LABELER_DATA` to change storage. Exports include annotations,
labels, colors, opacity and masks; original images are excluded. Older saved
annotations remain readable. Edited annotations record `source: manual` or
`source: sam2-assisted` when SAM2 prompt provenance is provided.
Keep source frames unchanged; changed frame sequences must use a new folder.

Run `python -m pip install httpx` and `python verify.py` for backend checks.

## Versions of the complete mask set

Use **Mask versions → Save version…** to give the complete saved mask set a
name, such as Wide or Narrow, and an optional note. Each checkpoint includes all
frames, saved objects, labels, colors and opacity. Save or discard any unsaved
object edits in the dialog first. **Save object** updates the working mask;
**Save version** creates an immutable checkpoint. Existing projects need no migration.

The version list shows each name, timestamp and mask count. Its **Export** button
downloads that checkpoint as a complete ZIP of original-resolution binary PNGs
plus JSON, even after working masks are edited or deleted. Version loading,
switching, restoration and deletion are not implemented. Current working-mask
downloads continue to use **Download masks + JSON**.

History is stored inside `projects/<project-id>/history/`:

```text
history/
  versions/<version-id>.json   # Complete project manifest and checkpoint metadata
  blobs/<pixel-hash>.json      # Shared mask format and dependency metadata
  blobs/<pixel-hash>.png       # Full binary mask, or:
  blobs/<pixel-hash>.npz       # Compressed, bit-packed XOR changes within a rectangle
```

Identical masks share a blob across checkpoints and objects. A changed mask uses
a delta against its previous checkpoint when that is smaller (including metadata)
than a compressed one-bit PNG. New masks and larger edits use PNGs. Chains are
limited to 16 deltas before another full PNG, keeping reconstruction bounded.
Only explicit version saves retain history; individual brush strokes do not.
Repeated unchanged checkpoints add a manifest but no duplicate mask data.

Back up the **entire project directory**, including `history/`: delta blobs may
depend on other blobs. Version manifests publish atomically after their data is
written. An interrupted save can leave unused blobs but cannot publish a partial
checkpoint. Run `python verify_versions.py` to check persistence and storage.

For browser checks, install Playwright in your Node environment, run an isolated
server with `LABELER_DATA` pointing to a disposable directory, and run
`node verify_versions_ui.cjs`. It defaults to Chrome and port 8012; override with
`LABELER_TEST_BROWSER` and `LABELER_TEST_URL`. The test creates a temporary project
and verifies the dialog, unsaved-edit choices, named versions and downloads.

## Load CVAT XML annotations

1. Open the source image folder.
2. Under **Load masks**, choose **Load CVAT XML…** and select the extracted
   `annotations.xml` from a **CVAT for images** export (XML 1.0 / 1.1).
3. Review the number of masks and matched frames, labels and any skipped shapes.
   Cancel leaves the project unchanged. Click **Import masks** to apply.
4. Select **Edit mask** on an imported object, correct its pixels and save.
   **Save version…** can capture the complete imported or corrected mask set.

CVAT raster masks are decoded from row-major, alternating background/foreground
RLE runs and placed at their original bounding-box offsets. Polygons are filled
to binary masks. Components sharing a nonzero group ID and label within an image
are united; per-image `track_id` attributes preserve object identity across frames.
Overlapping masks remain independent. Boxes, points, tags and other non-segmentation
shapes are excluded and counted in the preview.

Images match by relative path, then by an unambiguous path suffix or filename.
XML image IDs and order are not used as frame indices. Dimensions must match the
EXIF-oriented source images exactly. Missing or ambiguous image matches, invalid
RLE, malformed polygons and mismatched dimensions reject the entire preview;
nothing is partially imported. Images with no segmentation shapes are ignored.

Imports **add** objects with new local IDs, keeping existing masks and checkpoints.
Existing labels are matched case-insensitively and retain their colors; new labels
use CVAT colors when supplied. Importing the same XML twice adds duplicate objects.
An import preview expires when another import starts, the project is reopened,
or its saved contents change. XML files are limited to 64 MiB. Flipped tasks and
video-track XML are rejected; export those annotations as **CVAT for images**
aligned with the source images. ZIP archives are not accepted directly.

CVAT parsing is isolated in `cvat_import.py`, which produces validated staged masks.
Other import formats can later supply the same staged objects. Run
`python verify_cvat.py` for decoding, atomicity, persistence and version checks,
or `node verify_cvat_ui.cjs` against a disposable server for the browser workflow.
Browser-test settings match `verify_versions_ui.cjs` above.

Format reference: [CVAT XML specification](https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/).

## SAM2-assisted annotation

Run the server in a Python environment with SAM2 and PyTorch installed. See
[Meta's SAM2 installation guide](https://github.com/facebookresearch/sam2#installation).
The model loads only on the first prediction. A tiny SAM 2.1 checkpoint is
automatically searched for in `../sam2/checkpoints/` and the installed SAM2 source
checkout's `checkpoints/` directory. Override it and the matching config as needed:

```sh
export SAM2_CHECKPOINT=/path/to/sam2.1_hiera_tiny.pt
export SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_t.yaml
export SAM2_DEVICE=cuda  # optional; automatically chooses CUDA or CPU
export LABELER_PYTHON=/path/to/sam-environment/bin/python
bash run.sh
```

On the existing WSL setup, the Python interpreter is
`/home/seungbo/anaconda3/envs/sam/bin/python`. Manual tools stay usable if SAM2 is
unavailable; the SAM2 panel reports missing dependencies or a missing checkpoint.

1. New projects automatically include teeth_gum, soft_tissue, glove,
   unusable_region and background, listed above the custom-label form. Use
   **Add default labels** to add missing defaults to an existing project.
   Existing labels/colors are preserved. Background
   is an explicit annotation label; unannotated pixels are not filled automatically.
2. Select a label, then choose **SAM2 points**. Left click includes a region;
   right click excludes it. Include/Exclude buttons also choose point type.
3. By default the predicted region is added to the current mask. Use **New SAM
   region** to keep the current pixels and start prompts for another disconnected
   region in the same mask. Other application modes remove the predicted region
   from the mask or replace the mask. Negative points refine the predicted region;
   they do not directly erase previously accepted regions.
4. **Undo point** removes the last prompt; removing all points restores the mask
   from before that region. **Generate / retry** retries after a prediction error.
   Saving is blocked while prompts have no valid prediction. Switching to the
   brush editor accepts the last successful result and clears pending prompts.
5. Switch to **Brush editor** for pixel corrections. **Undo edit** undoes brush
   strokes or whole prediction updates. **Save object** commits the final pixels,
   and **Save version…** checkpoints the complete saved mask set.

**Edit mask** always opens saved pixels, including brush corrections and imported
CVAT masks. SAM2 results remain unsaved until Save object. Prompt provenance is
stored for the last predicted region, but the actual saved PNG is authoritative.
The displayed SAM score applies only to that predicted region, not to the combined
or manually corrected mask. Image embeddings are reused on the same frame. This
is single-frame prompting; video propagation is not enabled. The five annotation
labels remain independent; binary SegFormer training mapping is not configured.

Run `python verify_sam.py` in the SAM2 environment for real-model verification.
Run `node verify_sam_ui.cjs` against a disposable SAM2-enabled server for the
browser test, with the same Playwright setup and test URL settings described above.
