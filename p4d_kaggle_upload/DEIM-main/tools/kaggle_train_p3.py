"""P3: reliable frozen-NMS competition supervision on the fixed P0-R cache."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np
import torch
import kaggle_audit_p0 as audit
from kaggle_diagnose_p1 import ReportSource, check_metrics, compressed_json
from kaggle_train_p2 import ValidationDiagnostics, find_reference
from TileStitchingNet.pipeline_deim_tilestitching import OfflineDatasetIO
from TileStitchingNet.p0r import frozen_fields
from TileStitchingNet.p1 import pair_results
from TileStitchingNet.p2 import atomic_json
from TileStitchingNet.p3 import (PairConfig, build_pair_dataset, pair_tensors,
                                train_competition, load_p3_checkpoint)
from TileStitchingNet.stage2 import QualityConfig, QualityTargets, prepare


METHODS = {'A025': 0., 'C005': .05, 'C010': .1, 'C020': .2}


class P2Reference:
    """Read the audited P2 output whether Kaggle expands its ZIP or preserves it."""
    def __init__(self, path=None):
        folder = ROOT / 'TileStitchingNet/reports/p2_20261117'
        candidates = [Path(path)] if path else [folder/'p2_reports.zip', folder/'p2_reports', folder]
        self.path = next((p for p in candidates if p.is_file() or (p/'p2_summary.json').is_file()), None)
        if self.path is None:
            raise FileNotFoundError('Missing P2 reference; provide --p2-report ZIP or expanded directory')
        if self.path.name == 'p2_summary.json':
            self.path = self.path.parent
        self.archive = None

    def __enter__(self):
        if self.path.is_file():
            self.archive = zipfile.ZipFile(self.path)
        self.config = json.loads(self.read('run_config.json'))
        summary = json.loads(self.read('p2_summary.json'))
        if summary['status'] != 'P2_TRAINING_COMPLETED' or self.config['smoke']:
            self.__exit__()
            raise ValueError('P3 requires a completed full P2 reference')
        return self

    def __exit__(self, *args):
        if self.archive is not None:
            self.archive.close()

    def read(self, name):
        return self.archive.read(name) if self.archive else (self.path/name).read_bytes()

    def fingerprint(self):
        names = ['run_config.json', 'p2_summary.json', 'split.json', 'A/history.json']
        names += [f'A/epoch_{i:02d}.pth' for i in range(self.config['config']['epochs'] + 1)]
        h = hashlib.sha256()
        for name in sorted(names):
            h.update(name.encode())
            h.update(hashlib.sha256(self.read(name)).digest())
        return h.hexdigest()


def execute(args, output, report):
    torch.set_num_threads(args.threads)
    cache, gt = Path(args.train_cache), Path(args.train_gt)
    manifest_path = Path(args.manifest or args.train_cache + '.manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if manifest.get('schema') != 'p0r-roi-v1' or manifest.get('smoke'):
        raise ValueError('P3 requires the full P0-R cache')
    with ReportSource(find_reference(args.p0r_report)) as source, P2Reference(args.p2_report) as p2:
        read = lambda name: json.loads(source.read(name))
        reference, split = read('p0r_summary.json'), read('split.json')
        if reference['status'] != 'REBUILD_AND_TRAIN_COMPLETED':
            raise ValueError('P0-R reference is incomplete')
        if manifest != read('p0r_train.pt.manifest.json'):
            raise ValueError('Manifest differs from P0-R')
        cache_sha, gt_sha = audit.runner.sha(cache), audit.runner.sha(gt)
        for key, value in [('cache_sha256', cache_sha), ('gt_sha256', gt_sha)]:
            if manifest[key] != value or p2.config[key] != value:
                raise ValueError('Input differs from P0-R/P2: ' + key)
        p2_split = json.loads(p2.read('split.json'))
        if any(split[k] != p2_split[k] for k in ['train_ids', 'val_ids']):
            raise ValueError('P0-R/P2 split differs')
        if not args.smoke and args.epochs != p2.config['config']['epochs']:
            raise ValueError('Full P3 must use the same epoch budget as P2')
        config = QualityConfig(**{**p2.config['config'], 'epochs': 2 if args.smoke else args.epochs})
        if config.device != 'cpu' or config.rank_weight != 0:
            raise ValueError('P3 control requires original CPU QFL configuration')
        pair_config = PairConfig()
        source_paths = list(p2.config['source_hashes']) + ['tools/kaggle_train_p3.py', 'TileStitchingNet/p3.py']
        source_hashes = {name: audit.runner.sha(ROOT/name) for name in source_paths}
        for name, expected in p2.config['source_hashes'].items():
            if source_hashes[name] != expected:
                raise ValueError('Reused source differs from P2: ' + name)
        packages = {}
        for name in ['numpy', 'scipy', 'torchvision', 'pycocotools']:
            packages[name] = importlib.metadata.version(name)
        signature = dict(schema='p3-v1', cache_sha256=cache_sha, gt_sha256=gt_sha,
            manifest_sha256=audit.runner.sha(manifest_path), reference_sha256=source.fingerprint(),
            p2_reference_sha256=p2.fingerprint(), source_hashes=source_hashes,
            config=asdict(config), pair_config=asdict(pair_config), methods=METHODS,
            smoke=args.smoke, threads=args.threads, torch=str(torch.__version__),
            python=sys.version.split()[0], packages=packages, max_dets=200,
            selection='per-method Tiny_AP50 before_nms every epoch including epoch0',
            formula='QFL on base_logit+r; pair and deployment on base_logit+.25*r',
            pair_normalization='mean across images, within image mean across GTs of mean(weight*softplus)')
        run_path = output/'run_config.json'
        if run_path.is_file() and json.loads(run_path.read_text(encoding='utf-8')) != signature:
            raise ValueError('Resume inputs/code/configuration differ; use a fresh output directory')
        atomic_json(signature, run_path)
        for name in source_paths:
            destination = output/'sources'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT/name).read_bytes())
        samples = OfflineDatasetIO.load_samples(str(cache))
        by_id = {s.image_id: s for s in samples}
        expected = {r['image_id']: r['frozen'] for r in reference['train_refresh']}
        if len(by_id) != len(samples) or set(by_id) != set(expected):
            raise ValueError('Candidate image IDs differ')
        if set(split['train_ids']) & set(split['val_ids']) or set(split['train_ids']) | set(split['val_ids']) != set(by_id):
            raise ValueError('Invalid train/validation split')
        for sample in samples:
            if frozen_fields(sample) != expected[sample.image_id]:
                raise ValueError('Candidate fields changed: ' + sample.image_id)
            if sample.node_features.shape != (len(sample.scores), 76) or not np.isfinite(sample.node_features).all():
                raise ValueError('Invalid ROI features')
        train_ids = split['train_ids'][:8] if args.smoke else split['train_ids']
        val_ids = split['val_ids'][:4] if args.smoke else split['val_ids']
        training, validation = [by_id[i] for i in train_ids], [by_id[i] for i in val_ids]
        atomic_json(dict(train_ids=train_ids, val_ids=val_ids), output/'split.json')
        targets = QualityTargets(gt, config)
        annotation_data = json.loads(gt.read_text(encoding='utf-8'))
        train_pairs = build_pair_dataset(training, annotation_data, targets, pair_config)
        val_pairs = build_pair_dataset(validation, annotation_data, targets, pair_config)
        for name, pair_data in [('train', train_pairs), ('validation', val_pairs)]:
            compressed_json(output/(name+'_pair_geometry.json.gz'), pair_data.pop('geometry'))
            compressed_json(output/(name+'_pairs.json.gz'), pair_data)
            atomic_json({k: v for k, v in pair_data.items() if k != 'pairs'}, output/(name+'_pair_audit.json'))
        report['pairs'] = {name: dict(summary=d['summary'], signature=d['signature'])
                           for name, d in [('train', train_pairs), ('validation', val_pairs)]}
        if train_pairs['summary']['pairs'] == 0:
            raise ValueError('No reliable training pairs; inspect train_pair_audit.json before training')
        batches = [dict(prepare(s, config, False, targets), **pair_tensors(train_pairs['pairs'][s.image_id]))
                   for s in training]
        diagnostics = ValidationDiagnostics(validation, gt, config)
        if not args.smoke:
            check_metrics(diagnostics.base_metrics, read('validation_metrics.json')['Greedy-NMS'], 'P3 baseline')
        atomic_json(audit.clean(diagnostics.base_metrics), output/'baseline_metrics.json')
        atomic_json(read('baseline_reproduction.json')['validation'], output/'p0r_traditional_reference.json')
        compressed_json(output/'baseline_matches.json.gz', diagnostics.base_trace)
        compressed_json(output/'reliable_pairs.json.gz', {
            t: {s.image_id: pairs for s, pairs in zip(validation, grouped)} for t, grouped in diagnostics.pairs.items()})
        p2_history = {r['epoch']: r for r in json.loads(p2.read('A/history.json'))}
        report['input'] = dict(train_images=len(training), validation_images=len(validation),
                              test_evaluated=False, cache_sha256=cache_sha)
        print(f'P3 ready: {len(training)}/{len(validation)} images; fixed alpha=.25; four CPU runs.', flush=True)
        print('Training pair audit:', train_pairs['summary'], flush=True)
        results = {}
        for name, weight in METHODS.items():
            def validate(model, epoch):
                record = diagnostics.assess(model, pair_config.alpha)
                _, _, scores = diagnostics.predict(model, pair_config.alpha)
                record['training_rule_pairs'] = pair_results(validation,
                    [val_pairs['pairs'][s.image_id] for s in validation], scores)
                if epoch == 0:
                    check_metrics(record['metrics'], diagnostics.base_metrics, 'Epoch0 ' + name)
                if name == 'A025' and not args.smoke:
                    previous = p2_history[epoch]['validation']['A_scaled025']['metrics']
                    record['p2_metric_differences'] = audit.metric_diff(record['metrics'], previous)
                    check_metrics(record['metrics'], previous, 'P3 lambda=0 vs P2')
                    payload = torch.load(io.BytesIO(p2.read(f'A/epoch_{epoch:02d}.pth')), map_location='cpu', weights_only=True)
                    if set(payload['model']) != set(model.state_dict()):
                        raise AssertionError('P2 control parameter keys differ')
                    diffs = [float((value - payload['model'][k]).abs().max()) for k, value in model.state_dict().items()]
                    record['p2_max_parameter_abs_diff'] = max(diffs, default=0.)
                    if record['p2_max_parameter_abs_diff'] > 1e-6:
                        raise AssertionError('P3 lambda=0 updates differ from P2; inspect runtime/configuration')
                return {name: record}

            history, selections = train_competition(batches, config, pair_config, weight,
                train_pairs['signature'], output/name, validate)
            selection = selections[name]
            best_path = output/name/(name+'_best.pth')
            model, payload = load_p3_checkpoint(best_path)
            best_detail = diagnostics.detailed(model, pair_config.alpha, output, name+'_best', payload['epoch'])
            check_metrics(best_detail['modes']['before_nms']['metrics'], history[payload['epoch']]['validation'][name]['metrics'], name+' best reload')
            last, _ = load_p3_checkpoint(output/name/f'epoch_{config.epochs:02d}.pth')
            last_detail = diagnostics.detailed(last, pair_config.alpha, output, name+'_last', config.epochs)
            check_metrics(last_detail['modes']['before_nms']['metrics'], history[-1]['validation'][name]['metrics'], name+' last reload')
            values = [r['validation'][name]['metrics']['Tiny_AP50'] for r in history]
            results[name] = dict(pair_weight=weight, alpha=pair_config.alpha,
                best_epoch=selection['epoch'], best_checkpoint=str(best_path.relative_to(output)),
                best_checkpoint_sha256=audit.runner.sha(best_path), best=best_detail['modes'], last=last_detail['modes'],
                best_to_last_tiny_ap50_drop=max(values)-values[-1], min_trained_epoch_tiny_ap50=min(values[1:]))
            atomic_json(results, output/'validation_results.json')
        report['results'] = results
        report['input_files_unchanged'] = (audit.runner.sha(cache) == cache_sha and audit.runner.sha(gt) == gt_sha
            and audit.runner.sha(manifest_path) == signature['manifest_sha256']
            and source.fingerprint() == signature['reference_sha256'] and p2.fingerprint() == signature['p2_reference_sha256'])
        if not report['input_files_unchanged']:
            raise AssertionError('Input files changed during the run')
        report['status'] = 'P3_SMOKE_COMPLETED' if args.smoke else 'P3_TRAINING_COMPLETED'
        report['notes'] = [
            'All four methods train from the same seeded initialization; no P2 best checkpoint fine-tuning.',
            'P2 is a reproduction reference only. Lambda=0 is checked against its parameters and metrics.',
            'Original QFL uses unscaled logits. Pair loss and deployment use alpha=.25; no tanh bounds.',
            'Pair lists are frozen from original-score NMS, built separately for train/validation with no test labels.',
            'Gray candidates retain their original QFL mask; only reliable competition pairs supervise them.',
            'Each GT gets equal weight within each image; confidence weights are not normalized away.',
            'P3 training-rule pairs are more conservative than the unchanged P1 diagnostic pairs; both are reported.',
            'Best/last before/after-NMS traces use the same before-selected checkpoints. Latency=0 is a placeholder.',
            'Model/optimizer/scheduler/RNG/history resume at completed epochs; interrupted epochs replay.',
            'Single seed validation experiment. Freeze settings and replicate seeds before final test evaluation.']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--train-gt', required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--p0r-report')
    parser.add_argument('--p2-report')
    parser.add_argument('--output', default='/kaggle/working/p3_full')
    parser.add_argument('--epochs', type=int, default=18)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if min(args.epochs, args.threads) < 1:
        parser.error('epochs and threads must be positive')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error('Use a new output directory or --resume')
    output.mkdir(parents=True, exist_ok=True)
    report, failed = dict(arguments=vars(args), status='RUNNING'), False
    try:
        execute(args, output, report)
    except Exception:
        failed = True
        report['status'] = 'ERROR'
        report['error'] = traceback.format_exc()
        print(report['error'], flush=True)
    finally:
        atomic_json(audit.clean(report), output/'p3_summary.json')
        with zipfile.ZipFile(output/'p3_reports.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob('*')):
                if path.is_file() and path.suffix in {'.json', '.gz', '.npz', '.py', '.pth'}:
                    archive.write(path, path.relative_to(output))
        print('Status:', report['status'], '\nSend back:', output/'p3_reports.zip', flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
