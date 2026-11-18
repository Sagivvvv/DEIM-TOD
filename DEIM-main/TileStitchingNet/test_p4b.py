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
from .p3 import PairConfig
from .p4 import train_roi
from .p4b import margin_loss, train_margin, load_p4b_checkpoint

spec = importlib.util.spec_from_file_location('p4b_runner', Path(__file__).resolve().parents[1] / 'tools/kaggle_train_p4b.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class P4BTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def data(self):
        fixture = p4_tests.P4Tests()
        fixture.root = self.root
        return fixture.data()

    @staticmethod
    def validate(model, epoch):
        return {'X': dict(alpha=.25, metrics={'Tiny_AP50': float(epoch)})}

    def test_exact_margin_derivatives_denominator_and_reactivation(self):
        delta = torch.tensor([0., 0., .8, 0., -.4, 0.], requires_grad=True)
        data = dict(base=torch.full((6,), .5), pair_good=torch.tensor([0, 2, 4]),
                    pair_bad=torch.tensor([1, 3, 5]), pair_group=torch.tensor([0, 0, 1]),
                    pair_group_count=2, pair_confidence=torch.tensor([.5, 1., 1.]))
        loss = margin_loss({'delta': delta}, data)
        self.assertAlmostEqual(float(loss.detach()), .1125, places=6)
        gradient, = torch.autograd.grad(loss, delta)
        torch.testing.assert_close(gradient, torch.tensor([-.03125, .03125, 0., 0., -.125, .125]))
        with torch.no_grad():
            delta[2] = 0
        grad2, = torch.autograd.grad(margin_loss({'delta': delta}, data), delta)
        self.assertLess(float(grad2[2]), 0)
        with torch.no_grad():
            delta[2] = .4  # exact float32 margin: .25 * .4 == .1
        grad3, = torch.autograd.grad(margin_loss({'delta': delta}, data), delta)
        self.assertEqual(float(grad3[2]), 0.)
        with self.assertRaises(ValueError):
            margin_loss({'delta': delta}, data, margin=float('nan'))
        empty = {**data, 'pair_group_count': 0}
        self.assertEqual(float(margin_loss({'delta': delta}, empty).detach()), 0.)

    def test_softplus_control_exactly_reproduces_p4_updates(self):
        _, batches, pairs, config, _, _ = self.data()
        with contextlib.redirect_stdout(io.StringIO()):
            old, _ = train_roi(batches, config, PairConfig(), .2, pairs['signature'], 'full', self.root / 'old', self.validate)
            new, _ = train_margin(batches, config, PairConfig(), .2, pairs['signature'], self.root / 'new', self.validate, objective='softplus')
        self.assertEqual(old, new)
        a = torch.load(self.root / 'old/resume.pth', weights_only=True)
        b = torch.load(self.root / 'new/resume.pth', weights_only=True)
        self.assertTrue(all(torch.equal(v, b['model'][k]) for k, v in a['model'].items()))

    def test_margin_resume_and_objective_guards(self):
        _, batches, pairs, config, _, _ = self.data()
        def interrupt(model, epoch):
            if epoch == 2:
                raise RuntimeError('interruption')
            return self.validate(model, epoch)
        args = (batches, config, PairConfig(), .2, pairs['signature'])
        with contextlib.redirect_stdout(io.StringIO()):
            history, _ = train_margin(*args, self.root / 'full', self.validate)
            with self.assertRaises(RuntimeError):
                train_margin(*args, self.root / 'resume', interrupt)
            resumed, _ = train_margin(*args, self.root / 'resume', self.validate)
        self.assertEqual(history, resumed)
        a = torch.load(self.root / 'full/resume.pth', weights_only=True)
        b = torch.load(self.root / 'resume/resume.pth', weights_only=True)
        self.assertTrue(all(torch.equal(v, b['model'][k]) for k, v in a['model'].items()))
        self.assertEqual(a['scheduler'], b['scheduler'])
        model, payload = load_p4b_checkpoint(self.root / 'resume/X_best.pth')
        self.assertEqual(payload['objective'], 'hinge')
        self.assertEqual(payload['margin'], .1)
        self.assertTrue(torch.isfinite(model(batches[0])['scores']).all())
        for change in [dict(margin=.2), dict(objective='softplus')]:
            with self.assertRaises(ValueError):
                train_margin(*args, self.root / 'resume', self.validate, **change)

    def test_full_entrypoint_controls_smoke_expanded_resume_and_errors(self):
        fixture = p3_tests.P3Tests()
        fixture.root = self.root
        with contextlib.redirect_stdout(io.StringIO()):
            gt, common = fixture.make_references()
            p3_tests.runner.main(common + ['--p2-report', str(self.root / 'p2/p2_reports.zip'), '--output', str(self.root / 'p3')])
            p4_tests.runner.main(['--train-cache', str(self.root / 'p0r_train.pt'), '--train-gt', str(gt),
                '--p3-report', str(self.root / 'p3/p3_reports.zip'), '--epochs', '2', '--threads', '1', '--output', str(self.root / 'p4')])
            args = ['--train-cache', str(self.root / 'p0r_train.pt'), '--train-gt', str(gt),
                    '--p4-report', str(self.root / 'p4/p4_reports.zip'), '--epochs', '2', '--threads', '1', '--output', str(self.root / 'p4b')]
            runner.main(args)
        report = json.loads((self.root / 'p4b/p4b_summary.json').read_text())
        self.assertEqual(report['status'], 'P4B_TRAINING_COMPLETED')
        self.assertEqual(report['input']['new_training_runs'], 1)
        self.assertFalse(report['input']['test_evaluated'])
        for method in ['A_full', 'C_full']:
            self.assertEqual(report['results'][method]['origin'], 'P4_reference')
            self.assertFalse((self.root / 'p4b' / method / 'resume.pth').exists())
        self.assertEqual(report['results']['M010']['objective'], 'hinge')
        for name, digest in report['artifact_hashes'].items():
            self.assertEqual(runner.sha_file(self.root / 'p4b' / name), digest)
        for method in runner.METHODS:
            h = json.loads((self.root / 'p4b' / method / 'history.json').read_text())
            self.assertEqual([r['epoch'] for r in h], [0, 1, 2])
            for row in h:
                d = row['validation'][method]['margin_diagnostics']['validation']['groups']['all']
                self.assertEqual(d['active'] + d['satisfied'], d['pairs'])
        expanded = self.root / 'expanded'
        with zipfile.ZipFile(self.root / 'p4/p4_reports.zip') as z:
            for name in z.namelist():
                target = expanded / (name[:-3] if name.endswith('.gz') else name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(gzip.decompress(z.read(name)) if name.endswith('.gz') else z.read(name))
        with contextlib.redirect_stdout(io.StringIO()):
            runner.main(args + ['--p4-report', str(expanded), '--resume'])
            runner.main(args + ['--p4-report', str(expanded), '--output', str(self.root / 'smoke'), '--smoke'])
        self.assertEqual(json.loads((self.root / 'smoke/p4b_summary.json').read_text())['status'], 'P4B_SMOKE_COMPLETED')
        gt.write_text(gt.read_text() + ' ')
        with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            runner.main(args + ['--resume'])
        self.assertEqual(json.loads((self.root / 'p4b/p4b_summary.json').read_text())['status'], 'ERROR')
        with zipfile.ZipFile(self.root / 'p4b/p4b_reports.zip') as z:
            self.assertIsNone(z.testzip())


if __name__ == '__main__':
    unittest.main()
