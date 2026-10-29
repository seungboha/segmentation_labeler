"""Background app training worker. Publishes terminal state even on failure."""
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'labeler'))
import mask_history


def main(run):
    path = run / 'job.json'
    job = mask_history.read_json(path)
    job['pid'] = os.getpid()
    mask_history.write_json(path, job)
    try:
        from config import load_config
        from train import main as train
        config = load_config(run / 'config.yaml')
        train(config)
        if not (run / 'best/model.safetensors').is_file():
            raise RuntimeError('Training did not produce a best checkpoint')
        # Only load the local training state produced by this process.
        import torch
        state = torch.load(run / 'best/training_state.pt', map_location='cpu', weights_only=False)
        job.update(status='completed', best_dice=float(state['best_dice']))
    except BaseException as error:
        traceback.print_exc()
        job.update(status='failed', error=str(error))
    finally:
        from datetime import datetime, timezone
        job['finished_at'] = datetime.now(timezone.utc).isoformat()
        mask_history.write_json(path, job)
    return 0 if job['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main(Path(sys.argv[1]).resolve()))
