"""Train only on approved immutable app revisions; never reads TXT lists."""
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'labeler'))
import mask_history
try:
    from .data_loading import build_transforms
except ImportError:
    from data_loading import build_transforms


class SnapshotDataset(Dataset):
    def __init__(self, config, mode):
        snapshot = json.loads(Path(config['SNAPSHOT_FILE']).read_text())
        if snapshot['target'] != 'teeth_gum':
            raise ValueError('Unsupported snapshot target')
        if {r['subject'] for r in snapshot['train']} & {r['subject'] for r in snapshot['val']}:
            raise ValueError('Training and validation subjects overlap')
        self.records = snapshot[mode]
        if not self.records:
            raise ValueError(f'Empty {mode} snapshot')
        self.directory = Path(snapshot['project_directory'])
        self.subjects = tuple(sorted({r['subject'] for r in self.records}))
        self.split_file = config['SNAPSHOT_FILE']
        self.training_subjects_requested = ()
        self.training_subjects_file = None
        self.excluded_sample_count = 0
        self.geometry, self.color = build_transforms(config['INPUT_SIZE'], config['AUGMENTATION'], mode == 'train')
        self.mean = np.asarray(config['NORMALIZATION']['MEAN'], dtype=np.float32)
        self.std = np.asarray(config['NORMALIZATION']['STD'], dtype=np.float32)

    def __len__(self):
        return len(self.records)

    def seed_augmentation(self, seed):
        self.geometry.set_random_seed(seed % (2**32))
        if self.color is not None:
            self.color.set_random_seed((seed + 1) % (2**32))

    def __getitem__(self, index):
        record = self.records[index]
        raw = Path(record['image']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record['image_sha256']:
            raise ValueError(f"Source image changed since approval: {record['image']}")
        with Image.open(io.BytesIO(raw)) as image:
            image = np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
        mask = mask_history.read_mask(self.directory, record['mask_blob']).astype(np.uint8)
        if mask.shape != image.shape[:2]:
            raise ValueError('Snapshot image and mask dimensions differ')
        result = self.geometry(image=image, mask=mask)
        image, mask = result['image'], result['mask']
        if self.color is not None:
            image = self.color(image=image)['image']
        image = (image.astype(np.float32) / 255 - self.mean) / self.std
        return {'pixel_values': torch.from_numpy(image).permute(2, 0, 1).contiguous(),
                'labels': torch.from_numpy(mask.astype(np.float32))[None].contiguous(),
                'sample_name': record['name']}
