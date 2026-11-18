import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import torch

from .p0r import replace_roi, frozen_fields
from .p1 import CandidateLabels, nms_trace
from .p2 import train_pointwise
from .p3 import (PairConfig, competition_pairs, build_pair_dataset, pair_tensors,
                 competition_loss, train_competition, load_p3_checkpoint)
from .stage1 import OfficialTinyEvaluator, split_train_validation
from .stage2 import QualityConfig, QualityTargets, prepare, quality_loss, train_quality
from .test_stage2 import sample
from .test_p1 import annotations
from .pipeline_deim_tilestitching import OfflineDatasetIO

spec = importlib.util.spec_from_file_location('p3_runner', Path(__file__).resolve().parents[1]/'tools/kaggle_train_p3.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
import kaggle_train_p2 as p2_runner


class P3Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def labels(self, s, annotation_data=None):
        annotation_data = annotation_data or annotations([int(s.image_id)])
        path = self.root/'gt.json'
        path.write_text(json.dumps(annotation_data), encoding='utf-8')
        targets = QualityTargets(path, QualityConfig())
        return CandidateLabels(annotation_data, targets)(s), targets

    def gray_sample(self):
        s = sample()
        s.boxes_global[:2] = [[1, 0, 11, 10], [4, 0, 14, 10]]
        s.scores[:2] = [.7, .9]
        return s

    def test_actual_gray_conflict_and_label_protection(self):
        s = self.gray_sample()
        labels, targets = self.labels(s)
        _, edges = nms_trace(s.boxes_global, s.scores)
        pairs, excluded = competition_pairs(s, labels, edges, PairConfig())
        self.assertEqual(len(pairs), 1)
        p = pairs[0]
        self.assertEqual((p['winner'], p['suppressed'], p['good'], p['bad']), (1, 0, 0, 1))
        self.assertEqual(p['bad_reason'], 'gray_iou')
        self.assertFalse(p['bad_qfl_valid'])
        self.assertEqual(p['base_order'], 'wrong')
        self.assertTrue(labels['valid'][0])
        data = annotations([1])
        data['annotations'].append(dict(id=99, image_id=1, category_id=1,
            bbox=[4, 0, 10, 10], area=100, iscrowd=0))
        other, _ = self.labels(s, data)
        pp, ex = competition_pairs(s, other, edges, PairConfig())
        self.assertFalse(pp)
        self.assertEqual(ex['different_gt_positives'], 1)
        data['annotations'][-1]['uncertain'] = 1
        ignored, _ = self.labels(s, data)
        self.assertEqual(competition_pairs(s, ignored, edges, PairConfig())[1]['ignore_overlap'], 1)
        ambiguous = copy.deepcopy(labels)
        ambiguous['second'][1] = ambiguous['best'][1] - .01
        self.assertEqual(competition_pairs(s, ambiguous, edges, PairConfig())[1]['ambiguous_assignment'], 1)
        different_gray = copy.deepcopy(labels)
        different_gray['gid'][1] = 1
        self.assertEqual(competition_pairs(s, different_gray, edges, PairConfig())[1]['different_gt_gray'], 1)
        small_gap = copy.deepcopy(labels)
        small_gap['best'][:2] = [.55, .48]
        self.assertEqual(competition_pairs(s, small_gap, edges, PairConfig())[1]['small_quality_gap'], 1)
        boundary = copy.deepcopy(labels)
        boundary['best'][:2] = [.501, .39]
        weight = competition_pairs(s, boundary, edges, PairConfig())[0][0]['weight']
        self.assertGreaterEqual(weight, .25)
        self.assertLess(weight, .3)
        s.scores[:2] = [.9, .7]
        _, edges = nms_trace(s.boxes_global, s.scores)
        p = competition_pairs(s, labels, edges, PairConfig())[0][0]
        self.assertEqual(p['base_order'], 'correct')  # protection pairs retained
        dataset = build_pair_dataset([s], annotations([1]), targets, PairConfig())
        again = build_pair_dataset([s], annotations([1]), targets, PairConfig())
        self.assertEqual(dataset['signature'], again['signature'])
        self.assertEqual(dataset['summary']['pairs'] + sum(dataset['summary']['excluded'].values()), dataset['summary']['edges'])

    def test_pair_gradient_reaches_gray_without_changing_qfl_and_group_normalization(self):
        s = self.gray_sample()
        labels, targets = self.labels(s)
        pairs = competition_pairs(s, labels, nms_trace(s.boxes_global, s.scores)[1], PairConfig())[0]
        data = dict(prepare(s, QualityConfig(), False, targets), **pair_tensors(pairs))
        delta = torch.zeros(len(s.scores), requires_grad=True)
        outputs = dict(delta=delta, logits=torch.logit(data['base']) + delta)
        qfl = quality_loss(outputs, data, QualityConfig())[1]
        self.assertEqual(float(torch.autograd.grad(qfl, delta, retain_graph=True)[0][1]), 0.)
        loss = competition_loss(outputs, data)
        grad = torch.autograd.grad(loss, delta)[0]
        self.assertLess(float(grad[0]), 0.)
        self.assertGreater(float(grad[1]), 0.)
        self.assertEqual(float(grad[2]), 0.)
        other = dict(pairs[0], good=2, bad=3, gt_id=999)
        d = dict(data, **pair_tensors(pairs + [other]))
        expected = competition_loss(outputs, d)
        duplicated = dict(data, **pair_tensors(pairs * 10 + [other]))
        torch.testing.assert_close(competition_loss(outputs, duplicated), expected)
        one = dict(data, **pair_tensors([dict(pairs[0], weight=1.)]))
        weak = dict(data, **pair_tensors([dict(pairs[0], weight=.25)]))
        torch.testing.assert_close(competition_loss(outputs, weak), .25 * competition_loss(outputs, one))
        empty = competition_loss(outputs, dict(data, **pair_tensors([])))
        self.assertEqual(float(empty.detach()), 0.)
        self.assertTrue(torch.equal(torch.autograd.grad(empty, delta)[0], torch.zeros_like(delta)))

    def training_data(self):
        gt = self.root/'train_gt.json'
        ann = annotations([1, 2, 3])
        gt.write_text(json.dumps(ann), encoding='utf-8')
        config = QualityConfig(epochs=2, batch_tokens=5)
        samples = [sample('1'), sample('2')]
        targets = QualityTargets(gt, config)
        pairs = build_pair_dataset(samples, ann, targets, PairConfig())
        batches = [dict(prepare(s, config, False, targets), **pair_tensors(pairs['pairs'][s.image_id])) for s in samples]
        return config, batches, pairs['signature']

    @staticmethod
    def validate(model, epoch):
        return {'C': dict(alpha=.25, metrics={'Tiny_AP50': float(epoch)})}

    def test_zero_weight_reproduces_p2_updates(self):
        config, batches, signature = self.training_data()
        with contextlib.redirect_stdout(io.StringIO()):
            old, _ = train_pointwise(batches, config, None, self.root/'p2', self.validate)
            new, _ = train_competition(batches, config, PairConfig(), 0., signature, self.root/'p3', self.validate)
        a = torch.load(self.root/'p2/epoch_02.pth', weights_only=True)
        b = torch.load(self.root/'p3/epoch_02.pth', weights_only=True)
        for key, value in a['model'].items():
            self.assertTrue(torch.equal(value, b['model'][key]), key)
        for before, after in zip(old[1:], new[1:]):
            self.assertEqual(before['loss'], after['loss'])
            self.assertEqual(before['qfl'], after['qfl'])
            self.assertEqual(before['optimizer_steps'], after['optimizer_steps'])

    def test_resume_preserves_nonzero_pair_training_and_guards_labels(self):
        config, batches, signature = self.training_data()
        def interrupted(model, epoch):
            if epoch == 2:
                raise RuntimeError('interruption')
            return self.validate(model, epoch)
        with contextlib.redirect_stdout(io.StringIO()):
            full, _ = train_competition(batches, config, PairConfig(), .1, signature, self.root/'full', self.validate)
            with self.assertRaises(RuntimeError):
                train_competition(batches, config, PairConfig(), .1, signature, self.root/'resume', interrupted)
            resumed, _ = train_competition(batches, config, PairConfig(), .1, signature, self.root/'resume', self.validate)
        self.assertEqual(full, resumed)
        a = torch.load(self.root/'full/resume.pth', weights_only=True)
        b = torch.load(self.root/'resume/resume.pth', weights_only=True)
        for key, value in a['model'].items():
            self.assertTrue(torch.equal(value, b['model'][key]), key)
        self.assertEqual(a['scheduler'], b['scheduler'])
        model, payload = load_p3_checkpoint(self.root/'resume/C_best.pth')
        self.assertEqual(payload['alpha'], .25)
        self.assertEqual(payload['pair_weight'], .1)
        self.assertTrue(torch.isfinite(model(batches[0])['scores']).all())
        with self.assertRaises(ValueError):
            train_competition(batches, config, PairConfig(), .1, 'changed', self.root/'resume', self.validate)

    def make_references(self):
        gt = self.root/'annotations.json'
        gt.write_text(json.dumps(annotations(range(1, 7))), encoding='utf-8')
        samples = [sample(str(i)) for i in range(1, 7)]
        for s in samples:
            replace_roi(s, np.ones((len(s.scores), 64), np.float32))
        cache = self.root/'p0r_train.pt'
        OfflineDatasetIO.save_samples(samples, str(cache))
        training, validation, split = split_train_validation(samples, gt)
        reference_dir = self.root/'p0r'
        reference_dir.mkdir()
        _, history = train_quality(training, validation, gt, OfficialTinyEvaluator(gt), QualityConfig(epochs=2),
            False, reference_dir/'pointwise_best.pth', snapshot_dir=reference_dir/'pointwise_snapshots')
        baseline = history[0]['validation_before_nms']
        manifest = dict(schema='p0r-roi-v1', smoke=False, candidate_fields_unchanged=True,
            cache_sha256=runner.audit.runner.sha(cache), gt_sha256=runner.audit.runner.sha(gt))
        Path(str(cache)+'.manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        reference = dict(status='REBUILD_AND_TRAIN_COMPLETED', arguments=dict(smoke=False),
            feature_source=dict(source_hashes={p: runner.audit.runner.sha(runner.ROOT/p)
                for p in ['TileStitchingNet/stage1.py', 'TileStitchingNet/stage2.py']}),
            train_refresh=[dict(image_id=s.image_id, frozen=frozen_fields(s)) for s in samples])
        files = {'p0r_summary.json': reference, 'split.json': split, 'p0r_train.pt.manifest.json': manifest,
            'validation_metrics.json': {'Greedy-NMS': baseline}, 'baseline_reproduction.json': {'validation': {'Greedy-NMS': baseline}},
            'pointwise_history.json': history,
            'pointwise_completed.json': dict(checkpoint_sha256=runner.audit.runner.sha(reference_dir/'pointwise_best.pth'))}
        for name, data in files.items():
            (reference_dir/name).write_text(json.dumps(runner.audit.clean(data)), encoding='utf-8')
        for family in ['pointwise', 'local']:
            if family == 'local':
                (reference_dir/'local_best.pth').write_bytes((reference_dir/'pointwise_best.pth').read_bytes())
                (reference_dir/'local_completed.json').write_text('{}')
            for tag in ['epoch_06', 'epoch_18']:
                path = reference_dir/(family+'_snapshots')/(tag+'.pth')
                path.parent.mkdir(exist_ok=True)
                path.write_bytes((reference_dir/'pointwise_best.pth').read_bytes())
        common = ['--train-cache', str(cache), '--train-gt', str(gt), '--p0r-report', str(reference_dir), '--epochs', '2', '--threads', '1']
        p2_runner.main(common + ['--output', str(self.root/'p2')])
        return gt, common

    def test_full_entrypoint_references_diagnostics_resume_and_errors(self):
        with contextlib.redirect_stdout(io.StringIO()):
            gt, common = self.make_references()
            args = common + ['--p2-report', str(self.root/'p2/p2_reports.zip'), '--output', str(self.root/'p3')]
            runner.main(args)
        summary = json.loads((self.root/'p3/p3_summary.json').read_text())
        self.assertEqual(summary['status'], 'P3_TRAINING_COMPLETED')
        self.assertEqual(set(summary['results']), set(runner.METHODS))
        self.assertTrue(summary['input_files_unchanged'])
        self.assertFalse(summary['input']['test_evaluated'])
        history = json.loads((self.root/'p3/A025/history.json').read_text())
        self.assertTrue(all(row['validation']['A025']['p2_max_parameter_abs_diff'] == 0 for row in history))
        self.assertTrue((self.root/'p3/train_pairs.json.gz').is_file())
        self.assertTrue((self.root/'p3/C010_best_before_nms_matches.json.gz').is_file())
        self.assertIn('training_rule_pairs', history[-1]['validation']['A025'])
        with zipfile.ZipFile(self.root/'p3/p3_reports.zip') as z:
            self.assertIsNone(z.testzip())
            self.assertIn('sources/TileStitchingNet/p3.py', z.namelist())
        with contextlib.redirect_stdout(io.StringIO()):
            runner.main(args + ['--resume'])
            runner.main(common + ['--p2-report', str(self.root/'p2'), '--output', str(self.root/'smoke'), '--smoke'])
        self.assertEqual(json.loads((self.root/'smoke/p3_summary.json').read_text())['status'], 'P3_SMOKE_COMPLETED')
        gt.write_text(gt.read_text()+' ')
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
            runner.main(args + ['--resume'])
        self.assertEqual(json.loads((self.root/'p3/p3_summary.json').read_text())['status'], 'ERROR')
        self.assertTrue((self.root/'p3/p3_reports.zip').is_file())


if __name__ == '__main__':
    unittest.main()
