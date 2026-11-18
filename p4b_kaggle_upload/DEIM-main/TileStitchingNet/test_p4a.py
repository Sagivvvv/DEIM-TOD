import contextlib
import copy
import gzip
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import torch

from . import test_p4 as p4_tests
from . import test_p3 as p3_tests
from .p3 import PairConfig, pair_tensors
from .p4 import RoiRescorer, gradient_probe
from .p4a import gradient_census, check_original_probe, Accumulator, COMPONENTS

spec = importlib.util.spec_from_file_location('p4a_runner', Path(__file__).resolve().parents[1] / 'tools/kaggle_diagnose_p4a.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class P4ATests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def data(self):
        fixture = p4_tests.P4Tests()
        fixture.root = self.root
        return fixture.data()

    def test_streaming_matches_joint_autograd_and_preserves_state(self):
        samples, batches, pairs, config, _, _ = self.data()
        ids = [s.image_id for s in samples]
        plist = [pairs['pairs'][i] for i in ids]
        for mode in ['full', 'zero']:
            torch.manual_seed(4)
            model = RoiRescorer(78, config, mode).train()
            with torch.no_grad():
                model.rescorer.head[-1].weight.normal_(0, .1)
            for p in model.parameters():
                p.grad = torch.ones_like(p)
            before = copy.deepcopy(model.state_dict())
            grads = [p.grad.clone() for p in model.parameters()]
            rng = torch.get_rng_state().clone()
            expected = gradient_probe(model, batches, plist, config, PairConfig(), .2)
            result, rows = gradient_census(model, batches, plist, ids, config, PairConfig(), .2, ids)
            check_original_probe(result['cohorts']['all_training'], expected)
            self.assertEqual(len(rows), 2)
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
            self.assertTrue(all(torch.equal(g, p.grad) for g, p in zip(grads, model.parameters())))
            loss = result['cohorts']['all_training']['pair_components']
            self.assertAlmostEqual(loss['all']['loss'], sum(loss[k]['loss'] for k in ['base_correct', 'base_wrong', 'base_tie']))
            self.assertAlmostEqual(loss['all']['loss'], sum(loss[k]['loss'] for k in ['same_gt_quality', 'threshold_crossing']))
            wrong = copy.deepcopy(expected)
            wrong['qfl_gradient_norm'] += 1
            with self.assertRaises(ValueError):
                check_original_probe(result['cohorts']['all_training'], wrong)

    def test_no_pairs_skips_invalid_and_training_scope_guard(self):
        samples, batches, _, config, _, _ = self.data()
        model = RoiRescorer(78, config, 'full')
        batches = [{**b, **pair_tensors([])} for b in batches]
        batches[1]['valid'] = torch.zeros_like(batches[1]['valid'])
        result, rows = gradient_census(model, batches, [[], []], ['1', '2'], config, PairConfig(), 0, ['1'])
        self.assertEqual(result['skipped_image_ids'], ['2'])
        self.assertEqual(result['cohorts']['pair_images']['images'], 0)
        pc = result['cohorts']['all_training']['pair_components']['all']
        self.assertEqual(pc['weighted_gradient_norm'], 0)
        self.assertIsNone(pc['cosine_to_qfl'])
        self.assertEqual(pc['per_image_cosine']['defined'], 0)
        with self.assertRaises(ValueError):
            gradient_census(model, batches, [[], []], ['1', '2'], config, PairConfig(), .2, ['validation'])

    def test_vector_aggregation_is_not_average_of_cosines(self):
        acc = Accumulator(2)
        losses = {k: 0. for k in ('qfl',) + COMPONENTS}
        for q, p in [([10., 0.], [1., 0.]), ([-1., 0.], [1., 0.])]:
            acc.add({'qfl': torch.tensor(q, dtype=torch.float64),
                     **{k: torch.tensor(p, dtype=torch.float64) for k in COMPONENTS}}, losses, 1, 1, .2)
        result = acc.finish(.2)['pair_components']['all']
        self.assertEqual(result['cosine_to_qfl'], 1.)
        self.assertEqual(result['per_image_cosine']['negative_fraction'], .5)

    def test_entrypoint_zip_expanded_resume_error_packaging(self):
        fixture = p3_tests.P3Tests()
        fixture.root = self.root
        with contextlib.redirect_stdout(io.StringIO()):
            gt, common = fixture.make_references()
            p3_tests.runner.main(common + ['--p2-report', str(self.root / 'p2/p2_reports.zip'), '--output', str(self.root / 'p3')])
            p4_tests.runner.main(['--train-cache', str(self.root / 'p0r_train.pt'), '--train-gt', str(gt),
                '--p3-report', str(self.root / 'p3/p3_reports.zip'), '--epochs', '2', '--threads', '1', '--output', str(self.root / 'p4')])
            args = ['--train-cache', str(self.root / 'p0r_train.pt'), '--train-gt', str(gt),
                    '--p4-report', str(self.root / 'p4/p4_reports.zip'), '--threads', '1', '--output', str(self.root / 'p4a')]
            runner.main(args)
        summary = json.loads((self.root / 'p4a/p4a_summary.json').read_text())
        self.assertEqual(summary['status'], 'P4A_DIAGNOSTICS_COMPLETED')
        self.assertEqual(summary['input']['new_training_runs'], 0)
        self.assertFalse(summary['input']['validation_evaluated'])
        self.assertFalse(summary['input']['test_evaluated'])
        self.assertTrue(all(r['original_probe_reproduction']['status'] == 'PASS' for r in summary['results'].values()))
        expanded = self.root / 'expanded'
        with zipfile.ZipFile(self.root / 'p4/p4_reports.zip') as z:
            for name in z.namelist():
                target = expanded / (name[:-3] if name.endswith('.gz') else name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(gzip.decompress(z.read(name)) if name.endswith('.gz') else z.read(name))
        with contextlib.redirect_stdout(io.StringIO()):
            runner.main(args + ['--p4-report', str(expanded), '--resume'])
            runner.main(args + ['--p4-report', str(expanded), '--output', str(self.root / 'smoke'), '--smoke'])
        self.assertEqual(json.loads((self.root / 'smoke/p4a_summary.json').read_text())['status'], 'P4A_SMOKE_COMPLETED')
        with zipfile.ZipFile(self.root / 'p4a/p4a_reports.zip') as z:
            self.assertIsNone(z.testzip())
            self.assertFalse(any(n.endswith('.pth') for n in z.namelist()))
        gt.write_text(gt.read_text() + ' ')
        with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            runner.main(args + ['--resume'])
        self.assertEqual(json.loads((self.root / 'p4a/p4a_summary.json').read_text())['status'], 'ERROR')
        self.assertTrue((self.root / 'p4a/p4a_reports.zip').is_file())


if __name__ == '__main__':
    unittest.main()
