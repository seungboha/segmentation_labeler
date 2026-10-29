"""Exercise the app-to-checkpoint path on a tiny offline CPU model."""
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from transformers import SegformerConfig, SegformerForSemanticSegmentation
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'labeler'))
import training_workflow as workflow
import mask_history
from segformer.snapshot_data import SnapshotDataset


class SnapshotTrainingTests(unittest.TestCase):
    def test_offline_training_job_and_snapshot_integrity(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); images=root/'RGB'; images.mkdir()
            project={'id':'test-project','source_folder':str(images),'frames':[], 'annotations':{}}
            directory=root/'project'; directory.mkdir()
            for i,name in enumerate(['101_lower_1.png','101_lower_2.png','142_lower_1.png']):
                image=Image.new('RGB',(32,32),(80,100,120));image.save(images/name)
                project['frames'].append({'name':name})
                workflow.approve(directory,project,i,image,allow_empty=True)
            model_dir=root/'parent'
            model=SegformerForSemanticSegmentation(SegformerConfig(num_labels=1,
                hidden_sizes=[8,16,32,64],depths=[1,1,1,1],num_attention_heads=[1,2,4,8],
                decoder_hidden_size=16))
            model.save_pretrained(model_dir)
            config=yaml.safe_load((workflow.SEGFORMER/'config.yaml').read_text())
            config.update(INPUT_SIZE=32,BATCH_SIZE=2,DEVICE='cpu',AMP=False,NUM_WORKERS=0)
            config['AUGMENTATION']['ENABLE']=False
            config_path=root/'base.yaml';config_path.write_text(yaml.safe_dump(config))
            options={'python':sys.executable,'checkpoint':str(model_dir),'config':str(config_path),
                     'validation_subjects':[142],'epochs':1,'learning_rate':1e-5}
            with patch.object(workflow,'LOGS',root/'logs'),patch.dict(os.environ,{'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NO_ALBUMENTATIONS_UPDATE':'1','HF_HUB_OFFLINE':'1'}):
                job=workflow.start(directory,project,options)
                path=workflow.LOGS/job['id']/'job.json'
                deadline=time.monotonic()+90
                while time.monotonic()<deadline:
                    state=mask_history.read_json(path)
                    if state['status']!='running':break
                    time.sleep(.2)
                log=(path.parent/'console.log').read_text()
                self.assertEqual(state['status'],'completed',log)
                self.assertTrue((path.parent/'best/model.safetensors').is_file())
                self.assertIn('best_dice',state)
                self.assertEqual(mask_history.read_json(path.parent/'subjects.json'),
                                 {'training_subjects':[101], 'validation_subjects':[142]})
                self.assertIn('Training subjects: [101]', log)
                self.assertIn('Validation subjects: [142]', log)
                self.assertLess(log.index('Validation subjects: [142]'), log.index('Train 1/1'))
                exported=mask_history.read_json(path.parent/'best/config.json')
                self.assertEqual(exported['id2label'],{'0':'teeth_gum'})
                generated=yaml.safe_load((path.parent/'config.yaml').read_text())
                self.assertNotIn('VAL_LIST',generated)
                self.assertNotIn('TRAIN_LIST',generated)
                dataset=SnapshotDataset(generated,'train')
                self.assertEqual(len(dataset),2)
                self.assertEqual(tuple(dataset[0]['labels'].shape),(1,32,32))
                (images/'101_lower_1.png').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'changed'):
                    dataset[0]


if __name__=='__main__':unittest.main()
