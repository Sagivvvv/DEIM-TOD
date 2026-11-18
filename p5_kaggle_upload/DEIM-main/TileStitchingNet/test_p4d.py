import contextlib
import gzip
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import torch

from . import test_p3 as p3_tests
from . import test_p4 as p4_tests
from .p3 import PairConfig, competition_loss, pair_tensors
from .p4 import RoiRescorer, train_roi
from .p4d import RelationRescorer, train_relation, load_relation_checkpoint, install_initial_core
from .stage2 import prepare, quality_loss

spec = importlib.util.spec_from_file_location('p4d_runner', Path(__file__).resolve().parents[1] / 'tools/kaggle_train_p4d.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RelationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def data(self):
        fixture = p4_tests.P4Tests()
        fixture.root = self.root
        samples, _, pairs, config, ann, targets = fixture.data()
        batches = [dict(prepare(s, config, True, targets), **pair_tensors(pairs['pairs'][s.image_id])) for s in samples]
        return samples, batches, pairs, config

    @staticmethod
    def validate(model, epoch):
        return {'D': dict(alpha=.25, metrics={'Tiny_AP50': float(epoch)})}

    def test_initial_core_identity_and_sparse_permutation_equivariance(self):
        _, batches, _, config = self.data()
        torch.manual_seed(42)
        baseline = RoiRescorer(78, config, 'full')
        torch.manual_seed(42)
        model = RelationRescorer(78, config)
        for k, v in baseline.state_dict().items():
            self.assertTrue(torch.equal(v, model.state_dict()[k]), k)
        with torch.no_grad():
            model.rescorer.encoder[0].weight.add_(.01)
        digest = install_initial_core(model, baseline.state_dict())
        self.assertEqual(len(digest), 64)
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in baseline.state_dict().items()))
        with self.assertRaises(ValueError):
            install_initial_core(model, {})
        data = batches[0]
        torch.testing.assert_close(model(data)['scores'], data['base'], rtol=0, atol=0)
        with torch.no_grad():
            baseline.rescorer.head[-1].weight.normal_(0, .1)
            model.load_state_dict(baseline.state_dict(), strict=False)
        torch.testing.assert_close(model(data)['scores'], baseline(data)['scores'], rtol=0, atol=0)
        with torch.no_grad():
            model.relation_output.weight.normal_(0, .05)
        n = len(data['base'])
        permutation = torch.arange(n-1, -1, -1)
        inverse = torch.argsort(permutation)
        changed = {k: data[k][permutation] for k in ['features', 'base', 'geometry', 'neighbor_mask']}
        changed['neighbors'] = inverse[data['neighbors'][permutation]]
        torch.testing.assert_close(model(changed)['scores'], model(data)['scores'][permutation], rtol=1e-5, atol=1e-6)
        # Labels and pair supervision have no forward dependency.
        contaminated = {**data, 'quality': data['quality']+100, 'gt_id': data['gt_id']+1000}
        torch.testing.assert_close(model(data)['scores'], model(contaminated)['scores'], rtol=0, atol=0)

    def test_mask_empty_singleton_and_branch_receives_gradients(self):
        _, batches, _, config = self.data()
        model = RelationRescorer(78, config)
        data = batches[0]
        with torch.no_grad():
            model.rescorer.head[-1].weight.normal_(0, .1)
            model.relation_output.weight.normal_(0, .1)
        masked = {**data, 'neighbor_mask': torch.zeros_like(data['neighbor_mask'])}
        changed = {**masked, 'geometry': data['geometry']+10, 'neighbors': torch.zeros_like(data['neighbors'])}
        torch.testing.assert_close(model(masked)['scores'], model(changed)['scores'], rtol=0, atol=0)
        baseline = RoiRescorer(78, config, 'full')
        baseline.load_state_dict({k: v for k, v in model.state_dict().items() if k.startswith('rescorer.')})
        torch.testing.assert_close(model(masked)['scores'], baseline(masked)['scores'], rtol=0, atol=0)
        singleton = dict(features=data['features'][:1], base=data['base'][:1],
                         neighbors=torch.zeros((1, 1), dtype=torch.long), geometry=torch.zeros((1, 1, 8)),
                         neighbor_mask=torch.ones((1, 1), dtype=torch.bool))
        torch.testing.assert_close(model(singleton)['scores'], baseline(singleton)['scores'], rtol=0, atol=0)
        self.assertEqual(model(dict(features=data['features'][:0], base=data['base'][:0]))['scores'].numel(), 0)
        # The original fixture has only one nonself neighbor per nonisolated
        # row: its softmax is identically 1. Exercise attention with >1 neighbor.
        data = {**data, 'neighbor_mask': torch.ones_like(data['neighbor_mask'])}
        torch.manual_seed(42)
        model = RelationRescorer(78, config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        norms = {}
        for _ in range(5):
            optimizer.zero_grad()
            out = model(data)
            loss = quality_loss(out, data, config)[1] + .2 * competition_loss(out, data)
            loss.backward()
            norms = {name: float(p.grad.norm()) for name, p in model.named_parameters() if p.grad is not None}
            optimizer.step()
        self.assertGreater(norms['relation_output.weight'], 0)
        self.assertGreater(norms['relation_attention.0.weight'], 0)
        self.assertGreater(norms['relation_value.0.weight'], 0)

    def test_control_exact_updates_and_relation_resume_guard(self):
        _, batches, pairs, config = self.data()
        args = (batches, config, PairConfig(), .2, pairs['signature'])
        with contextlib.redirect_stdout(io.StringIO()):
            before, _ = train_roi(*args, 'full', self.root / 'control', self.validate)
            after, _ = train_relation(*args, 'graph-v1', self.root / 'control2', self.validate, relation=False)
        self.assertEqual(before, after)
        a = torch.load(self.root / 'control/resume.pth', weights_only=True)
        b = torch.load(self.root / 'control2/resume.pth', weights_only=True)
        self.assertTrue(all(torch.equal(v, b['model'][k]) for k, v in a['model'].items()))
        def interrupted(model, epoch):
            if epoch == 2:
                raise RuntimeError('interruption')
            return self.validate(model, epoch)
        with contextlib.redirect_stdout(io.StringIO()):
            full, _ = train_relation(*args, 'graph-v1', self.root / 'full', self.validate)
            with self.assertRaises(RuntimeError):
                train_relation(*args, 'graph-v1', self.root / 'resume', interrupted)
            resumed, _ = train_relation(*args, 'graph-v1', self.root / 'resume', self.validate)
        self.assertEqual(full, resumed)
        a = torch.load(self.root / 'full/resume.pth', weights_only=True)
        b = torch.load(self.root / 'resume/resume.pth', weights_only=True)
        self.assertTrue(all(torch.equal(v, b['model'][k]) for k, v in a['model'].items()))
        self.assertEqual(a['scheduler'], b['scheduler'])
        model, payload = load_relation_checkpoint(self.root / 'resume/D_best.pth')
        self.assertTrue(model.local)
        self.assertGreater(float(model.relation_output.weight.detach().norm()), 0)
        with self.assertRaises(ValueError):
            train_relation(*args, 'changed-graph', self.root / 'resume', self.validate)

    def test_full_entrypoint_reference_graph_latency_smoke_resume(self):
        fixture = p3_tests.P3Tests()
        fixture.root = self.root
        with contextlib.redirect_stdout(io.StringIO()):
            gt, common = fixture.make_references()
            p3_tests.runner.main(common + ['--p2-report', str(self.root / 'p2/p2_reports.zip'), '--output', str(self.root / 'p3')])
            p4_tests.runner.main(['--train-cache', str(self.root / 'p0r_train.pt'), '--train-gt', str(gt),
                '--p3-report', str(self.root / 'p3/p3_reports.zip'), '--epochs', '2', '--threads', '1', '--output', str(self.root / 'p4')])
            args = ['--train-cache', str(self.root / 'p0r_train.pt'), '--train-gt', str(gt),
                    '--p4-report', str(self.root / 'p4/p4_reports.zip'), '--epochs', '2', '--threads', '1', '--output', str(self.root / 'p4d')]
            runner.main(args)
        report = json.loads((self.root / 'p4d/p4d_summary.json').read_text())
        self.assertEqual(report['status'], 'P4D_TRAINING_COMPLETED')
        self.assertEqual(report['input']['new_training_runs'], 1)
        self.assertFalse(report['input']['test_evaluated'])
        self.assertGreater(report['results']['D_local']['parameter_count'], report['results']['C_full']['parameter_count'])
        for name, digest in report['artifact_hashes'].items():
            self.assertEqual(runner.sha_file(self.root / 'p4d' / name), digest)
        for method in runner.METHODS:
            latency = report['latency'][method]
            self.assertEqual(latency['repeats'], 3)
            self.assertGreater(latency['measurements']['total_ms']['mean'], 0)
        graph = json.loads((self.root / 'p4d/train_graph.json').read_text())
        payload = torch.load(self.root / 'p4d/D_local/resume.pth', weights_only=True)
        self.assertEqual(graph['signature'], payload['graph_signature'])
        self.assertIsNotNone(payload['initial_core_hash'])
        expanded = self.root / 'expanded'
        with zipfile.ZipFile(self.root / 'p4/p4_reports.zip') as z:
            for name in z.namelist():
                target = expanded / (name[:-3] if name.endswith('.gz') else name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(gzip.decompress(z.read(name)) if name.endswith('.gz') else z.read(name))
        with contextlib.redirect_stdout(io.StringIO()):
            runner.main(args + ['--p4-report', str(expanded), '--resume'])
            runner.main(args + ['--p4-report', str(expanded), '--output', str(self.root / 'smoke'), '--smoke'])
        self.assertEqual(json.loads((self.root / 'smoke/p4d_summary.json').read_text())['status'], 'P4D_SMOKE_COMPLETED')
        gt.write_text(gt.read_text() + ' ')
        with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            runner.main(args + ['--resume'])
        self.assertEqual(json.loads((self.root / 'p4d/p4d_summary.json').read_text())['status'], 'ERROR')
        with zipfile.ZipFile(self.root / 'p4d/p4d_reports.zip') as z:
            self.assertIsNone(z.testzip())


if __name__ == '__main__':
    unittest.main()
