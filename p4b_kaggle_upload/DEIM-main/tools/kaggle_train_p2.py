"""P2 controlled pointwise training on the frozen P0-R cache."""
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
sys.path.insert(0, str(ROOT/'tools'))

import numpy as np
import torch
import kaggle_audit_p0 as audit
from kaggle_diagnose_p1 import ReportSource, check_metrics, compressed_json, evaluate
from TileStitchingNet.pipeline_deim_tilestitching import OfflineDatasetIO
from TileStitchingNet.p0r import frozen_fields
from TileStitchingNet.stage1 import OfficialTinyEvaluator
from TileStitchingNet.stage2 import QualityConfig, QualityTargets, prepare
from TileStitchingNet.p1 import (CandidateLabels, OfficialMatchTrace, nms_trace, scaled_scores,
    reliable_pairs, pair_results, score_diagnostics, compare_traces)
from TileStitchingNet.p2 import atomic_json, load_p2_checkpoint, train_pointwise


def find_reference(explicit):
    if explicit:
        return Path(explicit)
    folder = ROOT/'TileStitchingNet/reports/p0r_20261116'
    for path in [folder/'p0r_reports.zip', folder/'p0r_reports', folder]:
        if path.is_file() or (path/'p0r_summary.json').is_file():
            return path
    raise FileNotFoundError('Missing bundled P0-R reference; provide --p0r-report')


class ValidationDiagnostics:
    def __init__(self, samples, annotation_path, config):
        self.samples = samples
        self.config = config
        self.evaluator = OfficialTinyEvaluator(annotation_path)
        data = json.loads(Path(annotation_path).read_text(encoding='utf-8'))
        self.tracer = OfficialMatchTrace(data)
        targets = QualityTargets(annotation_path, config)
        labeler = CandidateLabels(data, targets)
        self.labels = [labeler(s) for s in samples]
        self.inputs = [prepare(s, config, False) for s in samples]
        self.base_scores = [s.scores for s in samples]
        baseline = [nms_trace(s.boxes_global, s.scores) for s in samples]
        self.base_keep = [item[0] for item in baseline]
        self.base_edges = [item[1] for item in baseline]
        self.base_metrics = evaluate(self.evaluator, samples, self.base_keep, self.base_scores)
        self.base_trace = self.tracer(samples, self.base_keep, self.base_scores)
        self.pairs = {}
        for threshold in [.5, .75]:
            self.pairs[str(threshold)] = [reliable_pairs(s, label, edges, threshold)[0]
                for s, label, edges in zip(samples, self.labels, self.base_edges)]

    def predict(self, model, alpha):
        model.eval()
        raw_deltas, effective_deltas, scores = [], [], []
        with torch.inference_mode():
            for sample, data in zip(self.samples, self.inputs):
                outputs = model(data)
                raw = outputs['raw_delta'].numpy()
                delta = outputs['delta'].numpy()
                if not np.isfinite(raw).all() or not np.isfinite(delta).all():
                    raise FloatingPointError('Nonfinite validation residual')
                bound = model.residual_bound
                if bound is not None and np.any(np.abs(delta) > bound + 1e-6):
                    raise AssertionError('Residual bound violated')
                score = scaled_scores(sample.scores, delta, alpha)
                if alpha == 1:
                    np.testing.assert_array_equal(score, outputs['scores'].numpy())
                raw_deltas.append(raw)
                effective_deltas.append(delta * alpha)
                scores.append(score)
        return raw_deltas, effective_deltas, scores

    def assess(self, model, alpha):
        _, deltas, scores = self.predict(model, alpha)
        keeps = [nms_trace(s.boxes_global, score)[0] for s, score in zip(self.samples, scores)]
        metrics = evaluate(self.evaluator, self.samples, keeps, scores)
        drift = score_diagnostics(self.samples, self.labels, deltas, scores, self.base_edges)
        bound = model.residual_bound
        all_deltas = np.concatenate(deltas)
        near_bound = None
        if bound is not None and len(all_deltas):
            near_bound = float(np.mean(np.abs(all_deltas) >= .95 * bound))
        return audit.clean(dict(alpha=alpha, metrics=metrics, score_drift=drift,
            near_bound_fraction=near_bound,
            pair_results={t:pair_results(self.samples, pairs, scores) for t, pairs in self.pairs.items()}))

    def detailed(self, model, alpha, output, prefix, epoch):
        marker = output/(prefix+'.json')
        if marker.is_file():
            saved = json.loads(marker.read_text(encoding='utf-8'))
            artifacts = saved.get('artifact_hashes', {})
            if artifacts and all((output/p).is_file() and audit.runner.sha(output/p) == h for p, h in artifacts.items()):
                return saved
        raw, deltas, scores = self.predict(model, alpha)
        predicted = [nms_trace(s.boxes_global, score) for s, score in zip(self.samples, scores)]
        record = dict(epoch=epoch, alpha=alpha, residual_bound=model.residual_bound,
                      modes={}, artifact_hashes={})
        prediction_path = output/(prefix+'_scores.npz')
        np.savez_compressed(prediction_path,
            image_ids=np.asarray([s.image_id for s in self.samples]),
            offsets=np.cumsum([0]+[len(s.scores) for s in self.samples]),
            raw_delta=np.concatenate(raw), effective_delta=np.concatenate(deltas),
            scores=np.concatenate(scores), base=np.concatenate(self.base_scores),
            reason=np.concatenate([label['reason'] for label in self.labels]))
        record['artifact_hashes'][prediction_path.name] = audit.runner.sha(prediction_path)
        for mode in ['before_nms', 'after_nms']:
            keeps = [r[0] for r in predicted] if mode == 'before_nms' else self.base_keep
            edges = [r[1] for r in predicted] if mode == 'before_nms' else self.base_edges
            metrics = evaluate(self.evaluator, self.samples, keeps, scores)
            trace = self.tracer(self.samples, keeps, scores)
            changes = compare_traces(self.samples, self.base_trace, trace, self.base_edges, edges)
            record['modes'][mode] = audit.clean(dict(metrics=metrics, matching_and_budget=changes['summary']))
            artifacts = {
                prefix+'_'+mode+'_matches.json.gz':trace,
                prefix+'_'+mode+'_events.json.gz':changes['per_image'],
                prefix+'_'+mode+'_nms.json.gz':{
                    s.image_id:dict(keep=k.tolist(), edges=e.tolist())
                    for s, k, e in zip(self.samples, keeps, edges)}}
            for filename, contents in artifacts.items():
                compressed_json(output/filename, contents)
                record['artifact_hashes'][filename] = audit.runner.sha(output/filename)
        atomic_json(record, marker)
        print(f'Detailed validation finished: {prefix}', flush=True)
        return record


def execute(args, output, report):
    torch.set_num_threads(args.threads)
    cache = Path(args.train_cache)
    manifest_path = Path(args.manifest or args.train_cache+'.manifest.json')
    annotation_path = Path(args.train_gt)
    reference_path = find_reference(args.p0r_report)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if manifest.get('schema') != 'p0r-roi-v1' or manifest.get('smoke'):
        raise ValueError('P2 needs the full P0-R cache')
    source_paths = [
        'tools/kaggle_train_p2.py', 'TileStitchingNet/p2.py', 'TileStitchingNet/p1.py',
        'tools/kaggle_diagnose_p1.py', 'TileStitchingNet/stage2.py', 'TileStitchingNet/stage1.py',
        'TileStitchingNet/p0r.py', 'TileStitchingNet/datasetBuilder.py',
        'TileStitchingNet/pipeline_deim_tilestitching.py', 'tools/kaggle_audit_p0.py',
        'tools/kaggle_train_stage2.py', 'tools/huicv/evaluation/expand_cocofmt_eval.py']
    packages = {}
    for name in ['numpy', 'scipy', 'torchvision', 'pycocotools']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'unknown'
    with ReportSource(reference_path) as source:
        read = lambda name:json.loads(source.read(name))
        reference = read('p0r_summary.json')
        split = read('split.json')
        if reference.get('status') != 'REBUILD_AND_TRAIN_COMPLETED':
            raise ValueError('P0-R reference did not complete')
        if manifest != read('p0r_train.pt.manifest.json'):
            raise ValueError('Manifest differs from P0-R reference')
        cache_sha = audit.runner.sha(cache)
        gt_sha = audit.runner.sha(annotation_path)
        if cache_sha != manifest['cache_sha256'] or gt_sha != manifest['gt_sha256']:
            raise ValueError('Input cache or annotation hash differs from P0-R')
        checkpoint_bytes = source.read('pointwise_best.pth')
        if hashlib.sha256(checkpoint_bytes).hexdigest() != read('pointwise_completed.json')['checkpoint_sha256']:
            raise ValueError('Reference pointwise checkpoint differs')
        reference_checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=True)
        if reference_checkpoint['local']:
            raise ValueError('Expected pointwise reference')
        original_config = reference_checkpoint['config']
        config = QualityConfig(**{**original_config, 'epochs':2 if args.smoke else args.epochs,
                                  'device':'cpu', 'val_freq':1})
        if config.rank_weight != 0:
            raise ValueError('P2 expects QFL-only reference')
        source_hashes = {p:audit.runner.sha(ROOT/p) for p in source_paths}
        for name in ['TileStitchingNet/stage1.py', 'TileStitchingNet/stage2.py']:
            if source_hashes[name] != reference['feature_source']['source_hashes'][name]:
                raise ValueError('Reference model/evaluator source differs: '+name)
        signature = dict(schema='p2-v1', cache_sha256=cache_sha, gt_sha256=gt_sha,
            manifest_sha256=audit.runner.sha(manifest_path), reference_sha256=source.fingerprint(),
            source_hashes=source_hashes, config=asdict(config), smoke=args.smoke, threads=args.threads,
            torch=str(torch.__version__), python=sys.version.split()[0], packages=packages,
            methods={'A':None, 'B05':.5, 'B10':1.}, A_inference_alphas=[1., .25],
            bounded_formula='a*tanh(raw_delta/a)', selection='per-view Tiny_AP50 before_nms each epoch, including epoch0',
            nms_iou=.5, max_dets=200)
        run_path = output/'run_config.json'
        if run_path.is_file() and json.loads(run_path.read_text(encoding='utf-8')) != signature:
            raise ValueError('Resume input/code/configuration differs; choose a fresh output directory')
        atomic_json(signature, run_path)
        for name in source_paths:
            destination = output/'sources'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT/name).read_bytes())
        samples = OfflineDatasetIO.load_samples(str(cache))
        by_id = {s.image_id:s for s in samples}
        expected = {row['image_id']:row['frozen'] for row in reference['train_refresh']}
        if len(by_id) != len(samples) or set(by_id) != set(expected):
            raise ValueError('Input image IDs differ from P0-R')
        if set(split['train_ids']) & set(split['val_ids']):
            raise ValueError('Train/validation overlap')
        if set(split['train_ids']) | set(split['val_ids']) != set(by_id):
            raise ValueError('Split does not cover input cache')
        for sample in samples:
            if frozen_fields(sample) != expected[sample.image_id]:
                raise ValueError('Candidate fingerprint changed: '+sample.image_id)
            if sample.node_features.shape != (len(sample.scores), 76) or not np.isfinite(sample.node_features).all():
                raise ValueError('Invalid ROI features')
        train_ids = split['train_ids'][:8] if args.smoke else split['train_ids']
        val_ids = split['val_ids'][:4] if args.smoke else split['val_ids']
        training = [by_id[i] for i in train_ids]
        validation = [by_id[i] for i in val_ids]
        atomic_json(dict(train_ids=train_ids, val_ids=val_ids), output/'split.json')
        targets = QualityTargets(annotation_path, config)
        batches = [prepare(s, config, False, targets) for s in training]
        diagnostics = ValidationDiagnostics(validation, annotation_path, config)
        if not args.smoke:
            check_metrics(diagnostics.base_metrics, read('validation_metrics.json')['Greedy-NMS'], 'P2 baseline')
        atomic_json(audit.clean(diagnostics.base_metrics), output/'baseline_metrics.json')
        atomic_json(read('baseline_reproduction.json')['validation'], output/'p0r_traditional_reference.json')
        compressed_json(output/'baseline_matches.json.gz', diagnostics.base_trace)
        compressed_json(output/'reliable_pairs.json.gz', {
            t:{s.image_id:pairs for s,pairs in zip(validation, grouped)} for t,grouped in diagnostics.pairs.items()})
        old_history = {r['epoch']:r for r in read('pointwise_history.json')}
        report['input'] = dict(train_images=len(training), validation_images=len(validation),
                              test_evaluated=False, cache_sha256=cache_sha)
        report['formula_note'] = 'Uses a*tanh(r/a), refining planned a*tanh(r) so derivative at r=0 equals 1.'
        print(f'P2 ready: {len(training)}/{len(validation)} train/validation images, three CPU training runs.', flush=True)
        results = {}
        for name, bound in [('A', None), ('B05', .5), ('B10', 1.)]:
            views = {'A':1., 'A_scaled025':.25} if name == 'A' else {name:1.}

            def validate(model, epoch):
                records = {}
                for view_name, alpha in views.items():
                    record = diagnostics.assess(model, alpha)
                    if epoch == 0:
                        check_metrics(record['metrics'], diagnostics.base_metrics, 'Epoch0 '+view_name)
                    if view_name == 'A' and not args.smoke and config.epochs == original_config['epochs']:
                        old = old_history.get(epoch)
                        if old is not None:
                            record['p0r_metric_differences'] = audit.metric_diff(record['metrics'], old['validation_before_nms'])
                    records[view_name] = record
                return records

            history, selections = train_pointwise(batches, config, bound, output/name, validate)
            for view_name, selection in selections.items():
                best_path = output/name/(view_name+'_best.pth')
                model, payload = load_p2_checkpoint(best_path)
                alpha = selection['alpha']
                best_detail = diagnostics.detailed(model, alpha, output, view_name+'_best', payload['epoch'])
                chosen = history[payload['epoch']]['validation'][view_name]
                check_metrics(best_detail['modes']['before_nms']['metrics'], chosen['metrics'], view_name+' best reload')
                last_path = output/name/f'epoch_{config.epochs:02d}.pth'
                last_model, _ = load_p2_checkpoint(last_path)
                last_detail = diagnostics.detailed(last_model, alpha, output, view_name+'_last', config.epochs)
                check_metrics(last_detail['modes']['before_nms']['metrics'], history[-1]['validation'][view_name]['metrics'], view_name+' last reload')
                values = [r['validation'][view_name]['metrics']['Tiny_AP50'] for r in history]
                results[view_name] = dict(best_epoch=selection['epoch'], alpha=alpha, residual_bound=bound,
                    best_checkpoint=str(best_path.relative_to(output)), best_checkpoint_sha256=audit.runner.sha(best_path),
                    best=best_detail['modes'], last=last_detail['modes'],
                    best_to_last_tiny_ap50_drop=max(values)-values[-1],
                    min_trained_epoch_tiny_ap50=min(values[1:]))
            atomic_json(results, output/'validation_results.json')
        report['results'] = results
        report['input_files_unchanged'] = (
            audit.runner.sha(cache) == cache_sha and audit.runner.sha(annotation_path) == gt_sha
            and source.fingerprint() == signature['reference_sha256'])
        if not report['input_files_unchanged']:
            raise AssertionError('Input files changed during run')
        report['status'] = 'P2_SMOKE_COMPLETED' if args.smoke else 'P2_TRAINING_COMPLETED'
        report['notes'] = [
            'A and A_scaled025 share each epoch weight, with separate before-NMS validation-selected epochs.',
            'All variants have the same epoch0 through final-epoch selection opportunity.',
            'After-NMS metrics use the before-NMS-selected checkpoint; after-NMS is diagnostic.',
            'Per-epoch history includes score drift, quality MAE, within-GT IoU and reliable-pair error rates.',
            'Best and last have official per-GT and budget200 traces, including alpha=.25.',
            'Resume commits optimizer, scheduler and RNG after each completed epoch. Interrupted epochs are replayed.',
            'Primary AP uses maxDets200; unlimited survivor matching is only a budget diagnostic.',
            'A-to-P0-R differences are reported in history for environment/reproduction review.',
            'Candidate extraction historical provenance limitations from P0-R remain recorded in the reference.']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--train-gt', required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--p0r-report')
    parser.add_argument('--output', default='/kaggle/working/p2_full')
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
    report = dict(arguments=vars(args), status='RUNNING')
    failed = False
    try:
        execute(args, output, report)
    except Exception:
        failed = True
        report['status'] = 'ERROR'
        report['error'] = traceback.format_exc()
        print(report['error'], flush=True)
    finally:
        atomic_json(audit.clean(report), output/'p2_summary.json')
        with zipfile.ZipFile(output/'p2_reports.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob('*')):
                if path.is_file() and path.suffix in {'.json', '.gz', '.npz', '.py', '.pth'}:
                    archive.write(path, path.relative_to(output))
        print('Status:', report['status'], '\nSend back:', output/'p2_reports.zip', flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
