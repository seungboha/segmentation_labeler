# SegFormer-B0 fine-tuning

This directory fine-tunes Hugging Face SegFormer-B0 on the same data contract as
`/home/seungbo/RemovalAI/HierarchicalUnet`.

## Dataset

The default configuration expects:

```text
/home/seungbo/RemovalAI/HierarchicalDataset/train/
  RGB/137_upper_00284.bmp
  Label/narrow/137_upper_00284.png
  Label/wide/137_upper_00284.png
```

Image and mask basenames must match. Masks may contain either `0/1` or `0/255`.
The loader scans `RGB/` directly and never reads a text file to choose training
or validation images. Configure training subjects as an inline integer list:

```yaml
DATASET_PATH: /home/seungbo/RemovalAI/HierarchicalDataset/train
TRAIN_SUBJECTS: [101, 102, 103]
EXCLUDE_SUBJECTS: []
```

Subject IDs are the integer prefix before the first underscore in RGB filenames.
All `.bmp` frames and matching masks for those subjects are used for training.
Every remaining non-excluded subject is used for validation. Empty, duplicate,
invalid, excluded, or missing training subject IDs fail before training starts.
Test mode loads every non-excluded RGB frame independently of `TRAIN_SUBJECTS`.

`LABEL_TYPE` selects the targets:

- `narrow`: one sigmoid output trained on `MASKS.NARROW_DIR` (default)
- `wide`: one sigmoid output trained on `MASKS.WIDE_DIR`
- `hierarchical`: independent `[wide, narrow]` sigmoid outputs with an extra
  penalty when the predicted narrow probability exceeds the wide probability

Hierarchical ground truth is validated so that every narrow pixel is inside the
wide mask. Geometric augmentation is shared by the image and all masks; color
augmentation is applied only to the image.

## Setup and training

Create or activate a Python environment with a CUDA-compatible PyTorch build,
then install the remaining requirements:

```bash
cd /home/seungbo/my_projects/Labeler/segformer
python -m pip install -r requirements.txt
./train.sh
```

The first run downloads `nvidia/mit-b0`. Its ImageNet-pretrained MiT-B0 encoder
is loaded and the SegFormer segmentation decoder is initialized for one or two
output masks. Edit `config.yaml`, or override existing settings from the CLI:

```bash
./train.sh --set EPOCHS=20 --set BATCH_SIZE=4
./train.sh --set LABEL_TYPE=wide --set OUTPUT_DIR=./logs/segformer_b0_wide
./train.sh --set LABEL_TYPE=hierarchical \
  --set OUTPUT_DIR=./logs/segformer_b0_hierarchical
```

`config.yaml` uses `OUTPUT_DIR: ./logs/segformer_b0_{label_type}`.
At startup, after CLI overrides, `{label_type}` becomes `narrow`, `wide` or
`hierarchical` according to `LABEL_TYPE`. Changing only the label type is enough:

```bash
./train.sh --set LABEL_TYPE=wide
# Saves to ./logs/segformer_b0_wide
```

A fixed `--set OUTPUT_DIR=./outputs/my_experiment` still uses that exact path.
The resolved directory is printed at startup and recorded in resolved_config.yaml.
Existing checkpoint directories are not renamed or moved.

The output directory contains the resolved YAML, TensorBoard events, and two
Hugging Face-compatible checkpoint directories:

```text
logs/segformer_b0_narrow/
  resolved_config.yaml
  tensorboard/
  best/
  last/
```

Each checkpoint contains model weights, an image processor, and
`training_state.pt`. Resume only trusted checkpoints produced by this code:

```bash
./train.sh \
  --set RESUME_FROM=./logs/segformer_b0_narrow/last \
  --set EPOCHS=150
```

`EPOCHS` is the total target epoch count, not the number of additional epochs.
Every validation pass logs the first batch under `val/first_batch/input`,
`val/first_batch/ground_truth_<label>`, and
`val/first_batch/predicted_<label>`. The validation subject IDs are printed at
startup and stored under `run/validation_subjects` in TensorBoard. Inspect them
with:

```bash
tensorboard --logdir logs/segformer_b0_narrow/tensorboard
```

## Inference

Use the same Python environment as training. Pass the saved model directory or
the exact model weights file with `--weights`:

```bash
cd /home/seungbo/my_projects/Labeler/segformer
python inference.py \
  --weights logs/segformer_b0_narrow/best/model.safetensors \
  --input /home/seungbo/RemovalAI/HierarchicalDataset/train/RGB \
  --file-list few_shot_val.txt \
  --output-dir outputs/predictions_val \
  --device cuda
```

`--weights logs/segformer_b0_narrow/best` also works. For a single input image,
pass its filename to `--input` and omit `--file-list`. Omit `--file-list` to process
all supported images in an input folder and its subfolders. Supported extensions
are BMP, PNG, JPG/JPEG, TIFF and WebP. Optional lists accept basenames or filenames,
with the same blank-line/comment conventions as training lists.

The script reads the actual output labels and training input size from the
checkpoint, even when a folder name describes a different experiment. OpenCV RGB
resizing and normalization match the training dataset. Normalization is read from
`preprocessor_config.json`; `resolved_config.yaml` above best/last supplies a fallback
and the training threshold. Supply `--config path/to/training.yaml` if metadata was
stored elsewhere. `--input-size` and `--threshold` explicitly override saved settings.

Binary inference uses sigmoid and a threshold (default 0.5), never single-channel
argmax. Logits are resized to each source image's original dimensions before
thresholding. Outputs are ordinary 8-bit grayscale PNGs: background=0, foreground=255.
Disconnected foreground regions remain in the same mask. For one-channel models,
images become `outputs/predictions_val/<basename>.png`. Hierarchical models write
independent masks under `wide/` and `narrow/`. Relative image subfolders are preserved.
`predictions.json` records filenames, dimensions, checkpoint and inference settings.

Standalone `.pt`/`.pth`/`.bin` tensor state dictionaries are supported with a
matching `config.json` beside them, or `--model-config path/to/config.json`.
Inference never loads `training_state.pt` or optimizer state and never downloads
a checkpoint. Output collisions are rejected. Existing prediction files require
`--overwrite`; source image files cannot be overwritten. Options include
`--batch-size 4` and `--device cpu` for CPU inference.


### Overlay previews and MP4s

For predictions that already exist, render without rerunning the model:

```bash
python overlay_video.py \
  --predictions outputs/predictions_wide/predictions.json
```

Defaults: ROI pixels stay unchanged; only non-ROI pixels are tinted mint green
(`#68ddbc`) at opacity 0.2. There is no ROI tint or boundary drawing. PNG previews
retain source resolution. Binary masks and source images are never modified.

```text
outputs/predictions_wide/
  100_lower_00000.png                 # Existing binary prediction
  predictions.json
  overlays/wide/100_lower_00000.png   # Lossless RGB preview
  videos/wide/100_lower.mp4
  videos/wide/100_upper.mp4
  visualizations.json
```

One MP4 is encoded per label and source sequence. A sequence is the filename
prefix before its last numeric suffix (`100_lower_00015` -> `100_lower`). Frames
are sorted by numeric index. Subfolders are preserved; unnumbered filenames in
one folder share a `sequence.mp4`. Each available image contributes one video
frame; gaps in numeric indices are not interpolated. Mixed frame sizes within a
sequence are rejected rather than resized.

Encoding uses 10 FPS, H.264/libx264, CRF 18, medium preset, yuv420p, faststart and
no audio. Odd dimensions are padded by one pixel on the bottom/right as needed.
MP4 encoding is lossy; exact ROI preservation applies to the PNG overlays.
`visualizations.json` records frame order, overlay/video paths and encoding settings.

Optional settings: `--fps 10`, `--opacity 0.2`, `--non-roi-color '#68ddbc'`,
`--crf 18`, `--preset medium`, `--overlays-only`, and `--overwrite`. FFmpeg must
be available in PATH for videos. Existing previews/videos require `--overwrite`.
Both overlay rendering and encoding show tqdm progress.

To include these steps in a new inference run, add `--save-video` (which also
saves overlays), or `--save-overlays` for PNG previews alone:

```bash
python inference.py \
  --weights logs/segformer_b0_wide/best \
  --input /home/seungbo/RemovalAI/HierarchicalDataset/test/RGB \
  --output-dir outputs/predictions_wide_new \
  --save-video --fps 10 --overlay-opacity 0.2
```

Inference also accepts `--non-roi-color`, `--video-crf` and `--video-preset`.
The existing binary mask paths remain unchanged.

## Tests

The tests exercise configuration overrides, subject-level splitting, mask
loading, explicit train/validation lists, hierarchy validation, resizing, and the
custom segmentation loss. From the repository root:

```bash
python -m unittest discover -s segformer/tests -v
```


### Output layout

Training runs (checkpoints, resolved configuration, and TensorBoard events) are stored in `logs/<run>/`. Each inference run contains `predictions/` for binary PNG masks, `overlays/` for previews, and `videos/` for MP4s. `predictions.json` and `visualizations.json` remain at the inference run root; mask paths in the manifest are relative to that root. Videos include an original-filename footer.
