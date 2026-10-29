"""Regression tests for approval, immutable snapshots and binary training."""
import base64
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
import app as backend
import mask_history
import training_workflow as workflow


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_data, self.old_active, self.old_logs = backend.DATA, backend.ACTIVE, workflow.LOGS
        backend.DATA = self.root / 'projects'
        workflow.LOGS = self.root / 'logs'
        images = self.root / 'RGB'; images.mkdir()
        for name in ('101_lower_1', '101_lower_2', '142_lower_1'):
            Image.new('RGB', (32, 32), (120, 70, 50)).save(images / (name + '.png'))
        self.client = TestClient(backend.app)
        self.project = self.ok(self.client.post('/api/open', json={'folder': str(images)}))
        self.pid = self.project['id']
        self.directory = backend.DATA / self.pid

    def tearDown(self):
        self.client.close()
        backend.DATA, backend.ACTIVE, workflow.LOGS = self.old_data, self.old_active, self.old_logs
        self.tmp.cleanup()

    def ok(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def save(self, frame=0, offset=0, oid=1):
        mask = np.zeros((32, 32), dtype=np.uint8); mask[4:16, 4+offset:16+offset] = 255
        stream = io.BytesIO(); Image.fromarray(mask).save(stream, format='PNG')
        return self.ok(self.client.post('/api/save', json={'project': self.pid, 'frame': frame,
            'class_name': 'teeth_gum', 'object_id': oid, 'mask': base64.b64encode(stream.getvalue()).decode()}))

    def ready(self, frame=0, empty=False):
        return self.ok(self.client.post('/api/training/ready', json={'project': self.pid,
            'frame': frame, 'allow_empty': empty}))

    def test_empty_approval_requires_confirmation_and_persists(self):
        self.assertEqual(self.project['labels'], ['teeth_gum'])
        r=self.client.post('/api/training/ready', json={'project': self.pid, 'frame': 0})
        self.assertEqual(r.status_code, 400)
        p=self.ready(empty=True)
        self.assertEqual(p['training_approvals']['0']['roi_pixels'], 0)
        self.assertIn('0', json.loads((self.directory/'annotations.json').read_text())['training_approvals'])

    def test_snapshot_uses_ready_only_and_survives_edits(self):
        self.save(); self.ready(); self.ready(2, empty=True)
        path, snapshot = workflow.create_snapshot(self.directory, backend.ACTIVE, [142])
        self.assertEqual(len(snapshot['train']), 1)
        self.assertEqual(len(snapshot['val']), 1)
        self.assertEqual(snapshot['train_subjects'], [101])
        key=snapshot['train'][0]['mask_blob']
        before=mask_history.read_mask(self.directory,key)
        p=self.save(offset=5)
        self.assertNotIn('0', p['training_approvals'])
        np.testing.assert_array_equal(mask_history.read_mask(self.directory,key), before)
        self.assertEqual(json.loads(path.read_text())['train'][0]['mask_blob'], key)
        self.ready()
        p=self.ok(self.client.delete(f'/api/mask/0/1?project={self.pid}'))
        self.assertNotIn('0',p['training_approvals'])

    def test_union_and_source_change_checks(self):
        self.save();self.save(offset=10,oid=2);p=self.ready()
        self.assertGreater(p['training_approvals']['0']['roi_pixels'],144)
        self.ready(2,empty=True)
        with self.assertRaises(ValueError): workflow.create_snapshot(self.directory,backend.ACTIVE,[101,142])
        with self.assertRaises(ValueError): workflow.create_snapshot(self.directory,backend.ACTIVE,[])
        Path(p['training_approvals']['0']['image']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):
            workflow.create_snapshot(self.directory,backend.ACTIVE,[142])

    def test_draft_removal_and_no_txt_settings(self):
        self.ready(empty=True)
        p=self.ok(self.client.post('/api/training/ready',json={'project':self.pid,'frame':0,'ready':False}))
        self.assertEqual(p['training_approvals'],{})
        info=self.ok(self.client.get(f'/api/training?project={self.pid}'))
        self.assertNotIn('validation_list',info['defaults'])
        self.assertEqual(info['ready_count'],0)


if __name__=='__main__':
    unittest.main()
