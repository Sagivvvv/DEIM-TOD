import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from . import test_p3 as p3_tests, test_p4 as p4_tests
from .p5 import choose_family, sampling_group, sampled_ap
from tools import kaggle_train_p5 as runner


class P5Tests(unittest.TestCase):
    def test_selection_and_sampling(self):
        def selection(values, ap, tiny75):
            return choose_family(dict(paired_C_minus_A=dict(Tiny_AP50=dict(values=values),
                AP=dict(mean=ap),Tiny_AP75=dict(mean=tiny75))))
        self.assertEqual(selection([.1,.2,.3],-.1,.0),'C_full')
        self.assertEqual(selection([.1,0,.3],.0,.0),'A_full')
        self.assertEqual(selection([.1,.2,.3],.0,-.11),'A_full')
        self.assertEqual(sampling_group(dict(id=1,file_name='a/bb_V0012_I00012.jpg')),
                         sampling_group(dict(id=2,file_name='a/bb_V0012_I00019.jpg')))
        self.assertEqual(sampling_group(dict(id=2,file_name='u.jpg')),'image:2')
        rows = [(np.array([.8,.7]),np.array([0,1]),1),
                (np.array([.9]),np.array([1]),1)]
        self.assertAlmostEqual(sampled_ap(rows,[0,1]),100*(51+50*2/3)/101)
        self.assertAlmostEqual(sampled_ap(rows,[0,0]),50)

    def test_freeze_and_portable_test_entrypoints(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            fixture = p3_tests.P3Tests()
            fixture.root = root
            gt, common = fixture.make_references()
            p3_tests.runner.main(common+['--p2-report',str(root/'p2/p2_reports.zip'),'--output',str(root/'p3')])
            cache = root/'p0r_train.pt'
            train = ['--train-cache',str(cache),'--train-gt',str(gt)]
            p4_tests.runner.main(train+['--p3-report',str(root/'p3/p3_reports.zip'),'--epochs','2','--threads','1','--output',str(root/'p4')])
            reference = root/'p4/p4_reports.zip'
            with runner.P4Reference(reference) as ref:
                pins = {k:ref.config[k] for k in ('cache_sha256','gt_sha256','manifest_sha256')}
                split = ref.json('split.json')
                sizes = [len(split[k]) for k in ('train_ids','val_ids')]
                anchor = dict(epochs=2,split_sizes=sizes,p4=runner.reference_hashes(ref),
                              source_hashes=ref.config['source_hashes'],train=dict(pins,images=6),test=dict(pins,images=6))
            runner.write(root/'anchor.json',anchor)
            with patch.object(runner,'ANCHOR',root/'anchor.json'):
                def validate(out, *extra):
                    runner.main(['validate',*train,'--p4-report',str(reference),'--threads','1','--output',str(root/out),*extra])
                def test(freeze,out):
                    runner.main(['test','--test-cache',str(cache),'--test-gt',str(gt),'--threads','1',
                                 '--frozen',str(freeze),'--output',str(root/out)])
                validate('smoke','--smoke')
                with self.assertRaises(ValueError):
                    test(root/'smoke','invalid')
                validate('val')
                freeze = root/'val/p5_validation_reports.zip'
                self.assertFalse(runner.read(root/'val/status.json')['test_evaluated'])
                test(freeze,'final')
                with zipfile.ZipFile(root/'final/p5_reports.zip') as archive:
                    self.assertIsNone(archive.testzip())
                    report = json.loads(archive.read('final_report.json'))
                    self.assertEqual(report['status'],'P5_TEST_COMPLETED')
                    self.assertEqual(len(report['metrics']),10)
                    self.assertEqual(report['test_images'],6)
                    self.assertEqual(report['selected_on_validation'],runner.read(root/'val/frozen_config.json')['selected_family'])
                test(root/'val','expanded')
                ck = root/'val/weights/A_full_42.pth'
                ck.write_bytes(ck.read_bytes()+b'x')
                with self.assertRaisesRegex(ValueError,'checkpoint hash'):
                    test(root/'val','tampered')
                self.assertTrue((root/'tampered/p5_error.zip').is_file())


if __name__=='__main__':
    unittest.main()
