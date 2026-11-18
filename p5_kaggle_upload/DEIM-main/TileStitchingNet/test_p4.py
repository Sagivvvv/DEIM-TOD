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

from .p0r import replace_roi
from .p3 import PairConfig, build_pair_dataset, pair_tensors, competition_loss, train_competition
from .p4 import (RoiRescorer, load_p4_checkpoint, train_roi, annotation_audit,
                 full_nms_trace, select_probe, pair_components, gradient_probe)
from .stage2 import QualityConfig, QualityTargets, prepare
from .test_stage2 import sample
from .test_p1 import annotations
from . import test_p3 as p3_tests

spec = importlib.util.spec_from_file_location('p4_runner', Path(__file__).resolve().parents[1]/'tools/kaggle_train_p4.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class P4Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def data(self):
        samples = [sample('1'), sample('2')]
        rng = np.random.default_rng(17)
        for s in samples:
            replace_roi(s, rng.normal(size=(len(s.scores), 64)).astype(np.float32))
        gt = self.root/'gt.json'
        ann = annotations([1, 2])
        gt.write_text(json.dumps(ann), encoding='utf-8')
        config = QualityConfig(epochs=2, batch_tokens=5)
        targets = QualityTargets(gt, config)
        pairs = build_pair_dataset(samples, ann, targets, PairConfig())
        batches = [dict(prepare(s, config, False, targets), **pair_tensors(pairs['pairs'][s.image_id])) for s in samples]
        return samples, batches, pairs, config, ann, targets

    @staticmethod
    def validate(model, epoch):
        return {'X': dict(alpha=.25, metrics={'Tiny_AP50': float(epoch)})}

    def test_roi_mask_invariance_keeps_geometry_and_parameter_shape(self):
        _, batches, _, config, _, _ = self.data()
        data = batches[0]
        torch.manual_seed(42)
        full = RoiRescorer(78, config, 'full')
        zero = RoiRescorer(78, config, 'zero')
        zero.load_state_dict(full.state_dict())
        with torch.no_grad():
            full.rescorer.head[-1].weight.fill_(.3)
            zero.load_state_dict(full.state_dict())
        original = data['features'].clone()
        changed = {**data, 'features': original.clone()}
        changed['features'][:, 12:76] += 100
        torch.testing.assert_close(zero(data)['delta'], zero(changed)['delta'], rtol=0, atol=0)
        self.assertFalse(torch.equal(full(data)['delta'], full(changed)['delta']))
        masked = {**data, 'features': original.clone()}
        masked['features'][:, 12:76] = 0
        torch.testing.assert_close(zero(data)['delta'], full(masked)['delta'], rtol=0, atol=0)
        self.assertTrue(torch.equal(data['features'], original))
        self.assertEqual(sum(p.numel() for p in full.parameters()), sum(p.numel() for p in zero.parameters()))
        empty = {k: v[:0] for k, v in data.items() if k in ['base', 'features']}
        self.assertEqual(zero(empty)['scores'].shape, (0,))
        with self.assertRaises(ValueError):
            RoiRescorer(14, config, 'zero')

    def test_gradient_probe_preserves_state_and_partition_losses(self):
        samples, batches, pairs, config, _, _ = self.data()
        model = RoiRescorer(78, config, 'zero').train()
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        grads = [p.grad.clone() for p in model.parameters()]
        rng = torch.get_rng_state().clone()
        plist = [pairs['pairs'][s.image_id] for s in samples]
        result = gradient_probe(model, batches, plist, config, PairConfig(), .2)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
        self.assertTrue(all(torch.equal(g, p.grad) for g, p in zip(grads, model.parameters())))
        self.assertGreater(result['qfl_gradient_norm'], 0)
        self.assertAlmostEqual(result['pair_components']['all']['weighted_gradient_norm'],
                               .2*result['pair_components']['all']['raw_gradient_norm'])
        out = model(batches[0])
        pc = pair_components(out, batches[0], plist[0], .25)
        torch.testing.assert_close(pc['all'], competition_loss(out, batches[0]))
        torch.testing.assert_close(pc['all'], pc['same_gt_quality']+pc['threshold_crossing'])
        torch.testing.assert_close(pc['all'], pc['base_correct']+pc['base_wrong']+pc['base_tie'])
        a = select_probe(samples, batches, pairs['pairs'])
        self.assertEqual(a, select_probe(samples, batches, pairs['pairs']))
        empty_pairs = [[], []]
        empty_batches = [{**b, **pair_tensors([])} for b in batches]
        no_pairs = gradient_probe(model, empty_batches, empty_pairs, config, PairConfig(), 0)
        self.assertEqual(no_pairs['pair_components']['all']['raw_gradient_norm'], 0)
        self.assertIsNone(no_pairs['pair_components']['all']['cosine_to_qfl'])

    def test_annotation_flags_reconcile_or_require_review_and_nms_order(self):
        samples, _, pairs, _, ann, targets = self.data()
        split = dict(train_ids=['1'], val_ids=['2'])
        result = annotation_audit(ann, targets, split, pairs['geometry'])
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['derived_ignore'], 2)
        broken = copy.deepcopy(pairs['geometry'])
        broken['1']['ignored_boxes'] = []
        self.assertEqual(annotation_audit(ann, targets, split, broken)['status'], 'REVIEW_REQUIRED')
        other = copy.deepcopy(ann)
        other['annotations'].append(dict(id=999, image_id=1, category_id=2, bbox=[0, 0, 10, 10], ignore=1))
        self.assertEqual(annotation_audit(other, targets, split, pairs['geometry'])['status'], 'REVIEW_REQUIRED')
        zero_source = copy.deepcopy(ann)
        for a in zero_source['annotations']:
            a['uncertain'] = 0
        gt = self.root/'zero_gt.json'
        gt.write_text(json.dumps(zero_source), encoding='utf-8')
        zero_targets = QualityTargets(gt, QualityConfig())
        zero_pairs = build_pair_dataset(samples, zero_source, zero_targets, PairConfig())
        result = annotation_audit(zero_source, zero_targets, split, zero_pairs['geometry'])
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['derived_ignore'], 0)
        self.assertIsNotNone(result['ignored_zero_explanation'])
        trace = full_nms_trace(samples, pairs['pairs'])
        for s in samples:
            row = trace[s.image_id]
            self.assertEqual(row['initial_order'], s.scores.argsort()[::-1].tolist())
            self.assertEqual(set(row['keep']) | {p[1] for p in row['edges']}, set(range(len(s.scores))))

    def test_full_roi_training_exactly_reproduces_p3(self):
        _, batches, pairs, config, _, _ = self.data()
        with contextlib.redirect_stdout(io.StringIO()):
            before, _ = train_competition(batches, config, PairConfig(), .2, pairs['signature'], self.root/'old', self.validate)
            after, _ = train_roi(batches, config, PairConfig(), .2, pairs['signature'], 'full', self.root/'new', self.validate)
        a = torch.load(self.root/'old/epoch_02.pth', weights_only=True)
        b = torch.load(self.root/'new/epoch_02.pth', weights_only=True)
        self.assertEqual(before, after)
        for k, v in a['model'].items():
            self.assertTrue(torch.equal(v, b['model'][k]), k)

    def test_zero_roi_resume_and_probe_do_not_change_updates(self):
        samples, batches, pairs, config, _, _ = self.data()
        plist = [pairs['pairs'][s.image_id] for s in samples]
        def with_probe(model, epoch):
            gradient_probe(model, batches, plist, config, PairConfig(), .2)
            return self.validate(model, epoch)
        def interrupted(model, epoch):
            if epoch == 2:
                raise RuntimeError('interruption')
            return with_probe(model, epoch)
        args = (batches, config, PairConfig(), .2, pairs['signature'], 'zero')
        with contextlib.redirect_stdout(io.StringIO()):
            full, _ = train_roi(*args, self.root/'full', self.validate)
            with self.assertRaises(RuntimeError):
                train_roi(*args, self.root/'resume', interrupted)
            resumed, _ = train_roi(*args, self.root/'resume', with_probe)
        self.assertEqual(full, resumed)
        a = torch.load(self.root/'full/resume.pth', weights_only=True)
        b = torch.load(self.root/'resume/resume.pth', weights_only=True)
        self.assertTrue(all(torch.equal(v, b['model'][k]) for k, v in a['model'].items()))
        self.assertEqual(a['scheduler'], b['scheduler'])
        model, payload = load_p4_checkpoint(self.root/'resume/X_best.pth')
        self.assertEqual(payload['roi_mode'], 'zero')
        changed = {**batches[0], 'features': batches[0]['features'].clone()}
        changed['features'][:, 12:76] *= 7
        torch.testing.assert_close(model(batches[0])['delta'], model(changed)['delta'], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            train_roi(batches, config, PairConfig(), .2, pairs['signature'], 'full', self.root/'resume', self.validate)

    def test_full_entrypoint_reference_replay_new_training_smoke_resume_and_guard(self):
        fixture = p3_tests.P3Tests()
        fixture.root = self.root
        with contextlib.redirect_stdout(io.StringIO()):
            gt, common = fixture.make_references()
            p3_tests.runner.main(common+['--p2-report', str(self.root/'p2/p2_reports.zip'), '--output', str(self.root/'p3')])
            args = ['--train-cache', str(self.root/'p0r_train.pt'), '--train-gt', str(gt),
                    '--p3-report', str(self.root/'p3/p3_reports.zip'), '--epochs', '2', '--threads', '1', '--output', str(self.root/'p4')]
            runner.main(args)
        report = json.loads((self.root/'p4/p4_summary.json').read_text())
        self.assertEqual(report['status'], 'P4_ROI_COMPLETED')
        self.assertEqual(report['annotation_audit']['status'], 'PASS')
        self.assertEqual(report['input']['new_training_runs'], 2)
        self.assertFalse(report['input']['test_evaluated'])
        self.assertEqual(set(report['results']), set(runner.METHODS))
        self.assertEqual(len({r['parameter_count'] for r in report['results'].values()}), 1)
        for name in ['A_full', 'C_full']:
            self.assertEqual(report['results'][name]['origin'], 'P3_reference')
            self.assertFalse((self.root/'p4'/name/'resume.pth').exists())
            history = json.loads((self.root/'p4'/name/'history.json').read_text())
            self.assertTrue(all(not any(r['validation'][name]['p3_metric_differences'].values()) for r in history))
        for name in ['A_zero', 'C_zero']:
            self.assertEqual(load_p4_checkpoint(self.root/'p4'/name/(name+'_best.pth'))[1]['roi_mode'], 'zero')
        selection = json.loads((self.root/'p4/gradient_probe_selection.json').read_text())
        split = json.loads((self.root/'p4/split.json').read_text())
        self.assertTrue(set(selection['image_ids']) <= set(split['train_ids']))
        self.assertFalse(set(selection['image_ids']) & set(split['val_ids']))
        with zipfile.ZipFile(self.root/'p4/p4_reports.zip') as z:
            self.assertIsNone(z.testzip())
            for file in ['train_nms_trace.json.gz', 'source_annotations.json.gz', 'C_zero_best_gradients.json', 'sources/TileStitchingNet/p4.py']:
                self.assertIn(file, z.namelist())
        # Reproduce Kaggle recursively expanding both p3_reports.zip and its .gz files.
        expanded = self.root/'expanded/p3_reports'
        with zipfile.ZipFile(self.root/'p3/p3_reports.zip') as z:
            for name in z.namelist():
                content = z.read(name)
                if name.endswith('.gz'):
                    name = name.removesuffix('.gz')
                    content = gzip.decompress(content)
                destination = expanded/name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
        expanded_args = args.copy()
        expanded_args[expanded_args.index('--p3-report')+1] = str(expanded.with_suffix('.zip'))
        with contextlib.redirect_stdout(io.StringIO()):
            runner.main(expanded_args+['--resume'])
            smoke = expanded_args.copy()
            smoke[smoke.index('--p3-report')+1] = str(expanded)
            smoke[smoke.index('--output')+1] = str(self.root/'smoke')
            runner.main(smoke+['--smoke'])
        self.assertEqual(json.loads((self.root/'p4/p4_summary.json').read_text())['status'], 'P4_ROI_COMPLETED')
        self.assertEqual(json.loads((self.root/'smoke/p4_summary.json').read_text())['status'], 'P4_SMOKE_COMPLETED')
        gt.write_text(gt.read_text()+' ')
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
            runner.main(args+['--resume'])
        self.assertEqual(json.loads((self.root/'p4/p4_summary.json').read_text())['status'], 'ERROR')


if __name__ == '__main__':
    unittest.main()
