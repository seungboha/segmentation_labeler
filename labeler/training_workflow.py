"""Approved binary labels, immutable datasets and local training jobs."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone

import numpy as np
from PIL import Image
import mask_history as history

ROOT = Path(__file__).resolve().parent.parent
SEGFORMER = ROOT / "segformer"
LOGS = SEGFORMER / "logs" / "labeler"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(file.read()).hexdigest()


def subject(name):
    match = re.match(r"(\d+)_", Path(name).name)
    if not match:
        raise ValueError(f"Frame filename must start with a subject ID followed by _: {name}")
    return int(match[1])


def invalidate(project, frames):
    approvals = dict(project.get('training_approvals', {}))
    for frame in frames:
        approvals.pop(str(frame), None)
    project['training_approvals'] = approvals


def approve(directory, project, index, image, allow_empty=False, subject_id=None):
    roi = np.zeros((image.height, image.width), dtype=bool)
    for entry in project['annotations'].get(str(index), []):
        if entry['class_name'].casefold() == 'teeth_gum':
            with Image.open(directory / entry['mask_file']) as mask:
                pixels = np.asarray(mask.convert('L')) != 0
            if pixels.shape != roi.shape:
                raise ValueError('Mask dimensions do not match the image.')
            roi |= pixels
    if not roi.any() and not allow_empty:
        raise ValueError('No teeth_gum pixels. Confirm this is a fully reviewed image with no ROI.')
    name = project['frames'][index]['name']
    sid = subject(name) if subject_id is None else subject_id
    previous = project.get('training_approvals', {}).get(str(index), {})
    blob = history.store_mask(directory, roi, previous.get('mask_blob'))
    record = {'image': str(Path(project['source_folder']) / name), 'name': name,
              'subject': sid, 'image_sha256': digest(Path(project['source_folder']) / name),
              'mask_blob': blob, 'approved_at': now(), 'roi_pixels': int(roi.sum())}
    project['training_approvals'] = {**project.get('training_approvals', {}), str(index): record}
    return record


def jobs(project_id):
    result = []
    for path in LOGS.glob('*/job.json'):
        job = history.read_json(path)
        if job['project'] != project_id:
            continue
        process_file = path.parent / 'process.json'
        pid = job.get('pid') or (history.read_json(process_file)['pid'] if process_file.exists() else None)
        if job['status'] == 'running' and pid:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                job.update(status='failed', error='Training process exited before publishing a result.')
                history.write_json(path, job)
        result.append(job)
    return sorted(result, key=lambda job: job['created_at'], reverse=True)


def defaults():
    return {'python': os.environ.get('LABELER_TRAIN_PYTHON', '/home/seungbo/anaconda3/envs/temp_env/bin/python'),
            'config': str(SEGFORMER / 'config.yaml'),
            'checkpoint': str(SEGFORMER / 'logs/segformer_b0_wide/best'),
            'epochs': 10, 'learning_rate': 0.000006}


def create_snapshot(directory, project, validation_subjects):
    held_out = set(validation_subjects)
    approved = list(project.get('training_approvals', {}).values())
    available = {item['subject'] for item in approved}
    if not held_out or not held_out <= available:
        raise ValueError('Select at least one validation subject from Ready images.')
    training = [dict(item) for item in approved if item['subject'] not in held_out]
    validation = [dict(item) for item in approved if item['subject'] in held_out]
    if not training:
        raise ValueError('No Ready training images remain after holding out validation subjects.')
    for item in training + validation:
        if digest(item['image']) != item['image_sha256']:
            raise ValueError(f"Approved source image changed: {item['name']}. Review it again.")
        history.read_mask(directory, item['mask_blob'])
    identity = hashlib.sha256(json.dumps([{k: v for k, v in row.items() if k != 'approved_at'}
                                          for row in sorted(validation, key=lambda r: r['name'])], sort_keys=True).encode()).hexdigest()
    snapshot = {'id': uuid.uuid4().hex, 'created_at': now(), 'project': project['id'],
                'project_directory': str(directory.resolve()), 'target': 'teeth_gum',
                'mapping': {'teeth_gum': 1, 'everything_else': 0},
                'train': training, 'val': validation, 'validation_id': identity,
                'train_subjects': sorted({r['subject'] for r in training}),
                'val_subjects': sorted(held_out),
                'held_out_ready_images': len(approved) - len(training)}
    path = directory / 'training/snapshots' / (snapshot['id'] + '.json')
    history.write_json(path, snapshot)
    return path, snapshot


def start(directory, project, options):
    import yaml
    python = Path(options['python']).expanduser()
    checkpoint = Path(options['checkpoint']).expanduser().resolve()
    if not python.is_file():
        raise ValueError('Training Python executable does not exist.')
    model_config = history.read_json(checkpoint / 'config.json')
    if len(model_config.get('id2label', {})) != 1:
        raise ValueError('Select a binary SegFormer checkpoint directory with one output.')
    if not any((checkpoint / name).is_file() for name in ('model.safetensors', 'pytorch_model.bin')):
        raise ValueError('Checkpoint weights are missing.')
    # A single GPU training job at a time across all projects.
    for path in LOGS.glob('*/job.json'):
        old = history.read_json(path)
        if any(j['status'] == 'running' for j in jobs(old['project'])):
            raise ValueError('A training job is already running. Wait for it to finish.')
    snapshot_path, snapshot = create_snapshot(directory, project, options['validation_subjects'])
    job_id = uuid.uuid4().hex
    run = LOGS / job_id
    run.mkdir(parents=True)
    config = yaml.safe_load(Path(options['config']).read_text())
    for key in ('TRAIN_SUBJECTS_FILE', 'TRAIN_LIST', 'VAL_LIST', 'NUM_FOLDS', 'VAL_FOLD'):
        config.pop(key, None)
    config.update(TRAIN_SUBJECTS=[], EXCLUDE_SUBJECTS=[], LABEL_TYPE='teeth_gum', RESUME_FROM=None,
                  MODEL_NAME=str(checkpoint), OUTPUT_DIR=str(run), EPOCHS=options['epochs'],
                  LEARNING_RATE=options['learning_rate'], MIN_LEARNING_RATE=min(options['learning_rate'], 1e-6),
                  NUM_WORKERS=0, SNAPSHOT_FILE=str(snapshot_path.resolve()), WARMUP_EPOCHS=0)
    (run / 'config.yaml').write_text(yaml.safe_dump(config))
    job = {'id': job_id, 'project': project['id'], 'created_at': now(), 'status': 'running',
           'snapshot': str(snapshot_path), 'validation_id': snapshot['validation_id'],
           'parent_checkpoint': str(checkpoint), 'train_images': len(snapshot['train']),
           'val_images': len(snapshot['val']), 'train_subjects': snapshot['train_subjects'],
           'val_subjects': snapshot['val_subjects'], 'epochs': options['epochs'],
           'learning_rate': options['learning_rate'], 'checkpoint': str(run / 'best')}
    history.write_json(run / 'job.json', job)
    try:
        with (run / 'console.log').open('wb') as log:
            process = subprocess.Popen([str(python), '-u', str(SEGFORMER / 'labeler_train.py'), str(run)],
                                       cwd=SEGFORMER, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        history.write_json(run / 'process.json', {'pid': process.pid})
        def reap():
            code = process.wait()
            final = history.read_json(run / 'job.json')
            if final['status'] == 'running':
                final.update(status='failed', error=f'Training process exited with code {code}. See console log.')
                history.write_json(run / 'job.json', final)
        threading.Thread(target=reap, daemon=True).start()
    except Exception as error:
        job.update(status='failed', error=str(error)); history.write_json(run / 'job.json', job)
        raise
    return job
