# SAM 2 frame labeler

Local Python backend and browser canvas for labeling individual frames with SAM 2.
No tracking or video propagation. Uses the installed SAM package in your `sam`
Conda environment and `../sam2/checkpoints/sam2.1_hiera_tiny.pt` by default.

## Start in WSL Ubuntu

```bash
cd /home/seungbo/projects/Labeler/labeler
bash run.sh
```

Open http://localhost:8000 in your Windows browser. Keep the terminal running;
Ctrl+C stops the server. One server worker and one active labeling browser are
supported. The model loads at the first prediction; frame embeddings are reused
for subsequent clicks on the same frame.

Alternatively, run `python start_background.py` in the SAM environment to keep
the server running after the launching terminal closes. Its log is `server.log`;
its process ID is in `server.pid`. Stop that server in WSL with
`kill "$(cat server.pid)"` from this directory before starting another one.

To try the included sample sequence, run `python make_demo.py` in the SAM
environment and open `/home/seungbo/projects/Labeler/labeler/demo_frames` in the UI.

If dependencies are missing, install them in the existing SAM environment:

```bash
/home/seungbo/anaconda3/envs/sam/bin/python -m pip install -r requirements.txt
```

## Label frames

1. Click **Choose image folder** and select a folder on your computer. Supported
   images in it and its subfolders are read, naturally sorted (`2.jpg` before `10.jpg`),
   and copied to local project storage in WSL. Selecting the same unchanged folder
   resumes its saved annotations. Non-image files are ignored. Alternatively, expand
   **Or open by folder path** and enter a Windows path such as `C:\data\frames`
   or a WSL path such as `/home/seungbo/frames`; this reads directly without copying.
2. Left click includes the object; right click excludes background. Buttons also
   choose include/exclude mode. Add clicks until the overlay looks correct.
3. Create names under **Project labels** with **Add label**. The list shows all
   existing labels and their saved-object counts across the sequence. Click a label
   to select it, or choose it in the **Label name** dropdown. Labels persist even
   before they are used; older projects recover label names from saved annotations.
   Then click **Save object** (Enter). Saving commits immediately
   to disk. New IDs are assigned automatically. Reusing an ID on another frame
   records the same object manually; it does not perform tracking.
4. Use Previous/Next or the slider to change frames. Arrow keys navigate; Z undoes
   a click when focus is outside an input. Each frame is segmented independently.
5. Saved objects can be refined (replays their points) or deleted. Saving a refinement
   replaces that object's mask on the current frame. Preview changes require Save.
6. Download the ZIP export when ready. Reopen the same source folder to resume.

## Display resizing

Use the **Display size** percentage, +/− buttons, or **Fit to view**. Zoomed images
can be scrolled inside the canvas area. Aspect ratio is preserved. Click coordinates
are mapped back to the original image regardless of display zoom. The source
images, SAM 2 inputs, saved PNG mask dimensions, and export dimensions stay unchanged.
Your display preference is remembered by the browser across frames and sessions.

## Label colors and transparency

Choose a preset from the **New label color** palette, or use its custom color
picker before adding the label. Colors are selected manually. Existing labels
start with mint until you choose another color. Click a label's swatch to open
its color editor, choose a preset or custom color, and click **Save color**.
All saved objects and previews with that label use the same color.

The **Overlay opacity** slider ranges from 0% (invisible) to 100% (solid).
Changes preview immediately and save when you release the slider. Label colors
and project-wide opacity persist with the project and are included in exported
JSON as `label_colors` and `overlay_opacity`. Exported PNG masks remain binary
0/255 masks without color or transparency.

## Files and export

Projects are stored in `projects/<folder-hash>/annotations.json` with their masks
under `masks/`. Each saved object PNG has source-image resolution, with pixels
0 for background and 255 for foreground. EXIF orientation is applied consistently
to images and masks. Independent masks may overlap. JSON maps zero-based frame
indices to source filenames, object IDs, class names, prompts, dimensions, and
model scores. Scores describe the model's estimate, not annotation correctness.
Exports include JSON and masks, excluding the original source images.

Keep source files in place and unchanged to resume. Adding/removing/modifying
frames in an existing labeled folder is rejected to prevent frame-index mistakes.
Copy a changed sequence to a new folder to create a separate project. Back up the
project directory along with its original frames.

## Optional settings

```bash
export SAM2_CHECKPOINT=/path/to/checkpoint.pt
export SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_t.yaml
export SAM2_DEVICE=cuda  # or cpu
export LABELER_DATA=/path/to/project-storage
export PORT=8000
bash run.sh
```

The checkpoint and config must match. Set `LABELER_PYTHON` if your environment
uses a different Python interpreter. The backend binds to localhost and accesses
local frame folders; it is intended for personal use on this computer.
