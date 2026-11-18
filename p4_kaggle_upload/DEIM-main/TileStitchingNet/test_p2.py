import contextlib
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
from .p2 import BoundedQualityRescorer, train_pointwise, load_p2_checkpoint
from .p1 import scaled_scores
from .stage1 import OfficialTinyEvaluator, split_train_validation
from .stage2 import QualityConfig, QualityRescorer, QualityTargets, prepare, train_quality
from .test_stage2 import sample
from .test_p1 import annotations
from .pipeline_deim_tilestitching import OfflineDatasetIO

spec = importlib.util.spec_from_file_location('p2_runner', Path(__file__).resolve().parents[1]/'tools/kaggle_train_p2.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class P2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_zero_identity_unit_gradient_bounds_and_empty(self):
        s = sample()
        config = QualityConfig()
        data = prepare(s, config, False)
        for bound in [None, .5, 1.]:
            torch.manual_seed(42)
            model = BoundedQualityRescorer(14, config, bound)
            result = model(data)
            np.testing.assert_array_equal(result['scores'].detach().numpy(), s.scores)
            result['logits'].sum().backward()
            self.assertEqual(float(model.rescorer.head[-1].bias.grad), len(s.scores))
            with torch.no_grad():
                model.rescorer.head[-1].bias.fill_(100.)
            result = model(data)
            self.assertTrue(torch.isfinite(result['scores']).all())
            if bound is not None:
                self.assertLessEqual(float(result['delta'].abs().max().detach()), bound)
            empty = {key:value[:0] for key,value in data.items()}
            self.assertEqual(model(empty)['scores'].shape, (0,))
        torch.manual_seed(42)
        original = QualityRescorer(14, config)
        torch.manual_seed(42)
        wrapped = BoundedQualityRescorer(14, config)
        for name, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, wrapped.rescorer.state_dict()[name]))
        np.testing.assert_array_equal(original(data)['scores'].detach(), wrapped(data)['scores'].detach())

    def make_training(self):
        gt = self.root/'gt.json'
        gt.write_text(json.dumps(annotations([1,2,3])), encoding='utf-8')
        config = QualityConfig(epochs=2, batch_tokens=5)
        training = [sample('1'), sample('2')]
        validation = [sample('3')]
        targets = QualityTargets(gt, config)
        batches = [prepare(s, config, False, targets) for s in training]
        return gt, config, training, validation, batches

    def test_A_updates_match_legacy_training(self):
        gt, config, training, validation, batches = self.make_training()
        old_path = self.root/'legacy_best.pth'
        with contextlib.redirect_stdout(io.StringIO()):
            _, old_history = train_quality(training, validation, gt, OfficialTinyEvaluator(gt),
                config, False, old_path, snapshot_dir=self.root/'legacy')
        def validate(model, epoch):
            return {'A':dict(alpha=1., metrics={'Tiny_AP50':float(epoch)})}
        with contextlib.redirect_stdout(io.StringIO()):
            history, _ = train_pointwise(batches, config, None, self.root/'A', validate)
        old = torch.load(self.root/'legacy/epoch_02.pth', weights_only=True)
        new = torch.load(self.root/'A/epoch_02.pth', weights_only=True)
        for name, value in old['model'].items():
            self.assertTrue(torch.equal(value, new['model']['rescorer.'+name]), name)
        for before, after in zip(old_history[1:], history[1:]):
            self.assertEqual(before['loss'], after['loss'])
            self.assertEqual(before['optimizer_steps'], after['optimizer_steps'])

    def test_epoch_resume_is_identical_and_selection_views_independent(self):
        _, config, _, _, batches = self.make_training()
        def validate(model, epoch):
            return {'A':dict(alpha=1., metrics={'Tiny_AP50':float(2-epoch)}),
                    'A_scaled025':dict(alpha=.25, metrics={'Tiny_AP50':float(epoch)})}
        def interrupted(model, epoch):
            if epoch == 2:
                raise RuntimeError('simulated interruption before epoch commit')
            return validate(model, epoch)
        with contextlib.redirect_stdout(io.StringIO()):
            full_history, _ = train_pointwise(batches, config, None, self.root/'full', validate)
            with self.assertRaises(RuntimeError):
                train_pointwise(batches, config, None, self.root/'resume', interrupted)
            resumed_history, selected = train_pointwise(batches, config, None, self.root/'resume', validate)
        self.assertEqual(full_history, resumed_history)
        full = torch.load(self.root/'full/resume.pth', weights_only=True)
        resumed = torch.load(self.root/'resume/resume.pth', weights_only=True)
        for name, value in full['model'].items():
            self.assertTrue(torch.equal(value, resumed['model'][name]), name)
        self.assertEqual(full['scheduler'], resumed['scheduler'])
        self.assertEqual(selected['A']['epoch'], 0)
        self.assertEqual(selected['A_scaled025']['epoch'], 2)
        model, payload = load_p2_checkpoint(self.root/'resume/A_scaled025_best.pth')
        self.assertEqual(payload['alpha'], .25)
        out = model(batches[0])
        score = scaled_scores(batches[0]['base'].numpy(), out['delta'].detach().numpy(), payload['alpha'])
        self.assertTrue(np.isfinite(score).all())

    def test_full_entrypoint_zip_directory_resume_and_input_guard(self):
        root = self.root
        gt = root/'annotations.json'
        gt.write_text(json.dumps(annotations(range(1,7))), encoding='utf-8')
        samples = [sample(str(i)) for i in range(1,7)]
        for s in samples:
            replace_roi(s, np.ones((len(s.scores),64), np.float32))
        cache = root/'p0r_train.pt'
        with contextlib.redirect_stdout(io.StringIO()):
            OfflineDatasetIO.save_samples(samples, str(cache))
        training, validation, split = split_train_validation(samples, gt)
        config = QualityConfig(epochs=2)
        reference_dir = root/'reference'
        reference_dir.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            _, history = train_quality(training, validation, gt, OfficialTinyEvaluator(gt),
                config, False, reference_dir/'pointwise_best.pth', snapshot_dir=reference_dir/'pointwise_snapshots')
        baseline = history[0]['validation_before_nms']
        manifest = dict(schema='p0r-roi-v1', smoke=False, candidate_fields_unchanged=True,
            cache_sha256=runner.audit.runner.sha(cache), gt_sha256=runner.audit.runner.sha(gt))
        Path(str(cache)+'.manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        reference = dict(status='REBUILD_AND_TRAIN_COMPLETED', arguments=dict(smoke=False),
            feature_source=dict(source_hashes={p:runner.audit.runner.sha(runner.ROOT/p)
                for p in ['TileStitchingNet/stage1.py','TileStitchingNet/stage2.py']}),
            train_refresh=[dict(image_id=s.image_id, frozen=frozen_fields(s)) for s in samples])
        files = {'p0r_summary.json':reference, 'split.json':split, 'p0r_train.pt.manifest.json':manifest,
            'validation_metrics.json':{'Greedy-NMS':baseline},
            'baseline_reproduction.json':{'validation':{'Greedy-NMS':baseline}},
            'pointwise_history.json':history,
            'pointwise_completed.json':dict(checkpoint_sha256=runner.audit.runner.sha(reference_dir/'pointwise_best.pth'))}
        for name, data in files.items():
            (reference_dir/name).write_text(json.dumps(runner.audit.clean(data)), encoding='utf-8')
        # Populate the other small P0-R members used by its directory fingerprint.
        for family in ['pointwise','local']:
            if family == 'local':
                (reference_dir/'local_best.pth').write_bytes((reference_dir/'pointwise_best.pth').read_bytes())
                (reference_dir/'local_completed.json').write_text('{}')
            for tag in ['epoch_06','epoch_18']:
                path = reference_dir/(family+'_snapshots')/(tag+'.pth')
                path.parent.mkdir(exist_ok=True)
                path.write_bytes((reference_dir/'pointwise_best.pth').read_bytes())
        archive = root/'reference.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            for path in reference_dir.rglob('*'):
                if path.is_file():
                    z.write(path, path.relative_to(reference_dir))
        args = ['--train-cache',str(cache), '--train-gt',str(gt), '--p0r-report',str(archive),
                '--threads','1', '--epochs','2', '--output',str(root/'full')]
        with contextlib.redirect_stdout(io.StringIO()):
            runner.main(args)
        summary = json.loads((root/'full/p2_summary.json').read_text())
        self.assertEqual(summary['status'], 'P2_TRAINING_COMPLETED')
        self.assertEqual(set(summary['results']), {'A','A_scaled025','B05','B10'})
        self.assertFalse(summary['input']['test_evaluated'])
        for record in json.loads((root/'full/A/history.json').read_text()):
            self.assertTrue(all(abs(v)<1e-6 for v in record['validation']['A']['p0r_metric_differences'].values()))
        for family in ['A','B05','B10']:
            self.assertTrue((root/'full'/family/'epoch_02.pth').is_file())
        with contextlib.redirect_stdout(io.StringIO()):
            runner.main(args+['--resume'])
        folder_args = args.copy()
        folder_args[folder_args.index('--p0r-report')+1] = str(reference_dir)
        folder_args[folder_args.index('--output')+1] = str(root/'smoke')
        with contextlib.redirect_stdout(io.StringIO()):
            runner.main(folder_args+['--smoke'])
        self.assertEqual(json.loads((root/'smoke/p2_summary.json').read_text())['status'], 'P2_SMOKE_COMPLETED')
        gt.write_text(gt.read_text()+' ')
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
            runner.main(args+['--resume'])
        self.assertEqual(json.loads((root/'full/p2_summary.json').read_text())['status'], 'ERROR')
        self.assertTrue((root/'full/p2_reports.zip').is_file())


if __name__ == '__main__':
    unittest.main()
