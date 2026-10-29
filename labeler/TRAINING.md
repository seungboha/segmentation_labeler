# Labeling and fine-tuning

1. Open the RGB folder by its WSL path (including `/mnt/...`) to reference images in place. The browser folder picker copies images into project storage.
2. Label every usable teeth/gum region and **Save object**. New projects start with `teeth_gum`; existing labels remain available. Binary training unions every teeth_gum object on a frame. All other pixels are background.
3. Check **Subject ID**, then **Mark ready for training** after reviewing the whole image. Empty images require explicit confirmation and become negative examples. Saving, deleting, or importing masks returns affected images to Draft. **Return to Draft** removes approval without deleting masks.
4. Open **Train model…** and select validation subjects from the Ready images. Both splits use only Ready frames. At least one subject must remain in each split. No TXT lists or external dataset folders are read by this workflow.
5. Choose an existing binary SegFormer checkpoint, epochs, and learning rate. **Create snapshot and train** fine-tunes all layers using every Ready training image. A small approved dataset therefore supports few-shot fine-tuning. Training runs in a separate process; close the dialog to continue labeling.
6. Reopen the dialog to inspect logs, status, and best validation Dice. Compare scores only when the validation IDs match. **Use this model** selects the successful checkpoint as the parent of the next training run. It does not run predictions or overwrite annotations.

## Storage

- `labeler/projects/<project>/annotations.json`: current approvals and selected model (or the configured `LABELER_DATA` directory).
- `labeler/projects/<project>/history/blobs/`: shared immutable compressed mask data, reused by mask versions and training snapshots.
- `labeler/projects/<project>/training/snapshots/<id>.json`: exact train/validation records, source-image checksums, mask references, subjects and binary mapping.
- `segformer/logs/labeler/<job>/`: job metadata, original/generated settings, console log, TensorBoard events, `best/` and `last/` checkpoints.

Snapshots reference original RGB files rather than copying them. Original images and referenced history blobs must remain available. Training checks the source-image checksum whenever loading a frame and fails if it changed after approval. Earlier snapshots remain valid after subsequent mask edits.

## Runtime

The app environment requires `PyYAML` in addition to the existing editor dependencies. Training uses a separate Python executable, defaulting to `/home/seungbo/anaconda3/envs/temp_env/bin/python`; override it through `LABELER_TRAIN_PYTHON` or Runtime settings. The selected YAML supplies optimizer, augmentation, image-size and normalization settings. The app replaces its data-selection settings with the immutable snapshot.

One app-managed training job runs at a time. Jobs continue after closing the dialog or restarting the app. A crashed process is reported as failed; failed jobs cannot be selected. This version does not provide cancellation or automatic prediction refresh.

Restart the app server and refresh the browser after upgrading. No training starts until **Create snapshot and train** is pressed.
