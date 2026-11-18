"""P4 first step: paired ROI ablation, annotation audit and fixed gradient probes."""
import argparse
from dataclasses import asdict
import gzip
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
from kaggle_diagnose_p1 import check_metrics, compressed_json
from kaggle_train_p2 import ValidationDiagnostics
from TileStitchingNet.pipeline_deim_tilestitching import OfflineDatasetIO
from TileStitchingNet.p1 import pair_results
from TileStitchingNet.p2 import atomic_json, atomic_save
from TileStitchingNet.p3 import PairConfig, build_pair_dataset, pair_tensors
from TileStitchingNet.p4 import (RoiRescorer, annotation_audit, full_nms_trace,
    select_probe, gradient_probe, train_roi, load_p4_checkpoint)
from TileStitchingNet.stage2 import QualityConfig, QualityTargets, prepare

METHODS = {'A_full': ('full', 0., 'A025'), 'C_full': ('full', .2, 'C020'),
           'A_zero': ('zero', 0., 'A025'), 'C_zero': ('zero', .2, 'C020')}


class P3Reference:
    def __init__(self, path=None):
        folder = ROOT/'TileStitchingNet/reports/p3_20261117'
        if path:
            requested = Path(path)
            candidates = [requested]
            if requested.suffix == '.zip':
                candidates.append(requested.with_suffix(''))
        else:
            candidates = [folder/'p3_reports.zip', folder/'p3_reports', folder]
        self.path = next((p for p in candidates if p.is_file() or (p/'p3_summary.json').is_file()), None)
        if self.path is None:
            raise FileNotFoundError('Missing P3 reference; provide --p3-report ZIP or expanded folder')
        if self.path.name == 'p3_summary.json':
            self.path = self.path.parent
        self.archive = None

    def __enter__(self):
        try:
            if self.path.is_file():
                self.archive = zipfile.ZipFile(self.path)
            self.config = self.json('run_config.json')
            self.summary = self.json('p3_summary.json')
            if self.summary['status'] != 'P3_TRAINING_COMPLETED' or self.config['smoke']:
                raise ValueError('P4 requires completed full P3')
            return self
        except Exception:
            self.__exit__()
            raise

    def __exit__(self, *args):
        if self.archive:
            self.archive.close()

    def read(self, name):
        """Read logical content in both the original and Kaggle-expanded layouts."""
        if self.archive is not None:
            read_bytes = self.archive.read
            missing = KeyError
        else:
            def read_bytes(member):
                return (self.path/member).read_bytes()
            missing = FileNotFoundError

        if not name.endswith('.gz'):
            return read_bytes(name)
        try:
            compressed = read_bytes(name)
        except missing:
            return read_bytes(name.removesuffix('.gz'))
        return gzip.decompress(compressed)

    def json(self, name):
        return json.loads(self.read(name))

    def fingerprint(self):
        names = ['run_config.json', 'p3_summary.json', 'split.json', 'validation_results.json',
                 'baseline_metrics.json', 'p0r_traditional_reference.json']
        for scope in ['train', 'validation']:
            names += [scope+'_pairs.json.gz', scope+'_pair_geometry.json.gz']
        for family in ['A025', 'C020']:
            names += [family+'/history.json']
            names += [f'{family}/epoch_{i:02d}.pth' for i in range(self.config['config']['epochs']+1)]
        h = hashlib.sha256()
        for name in sorted(names):
            h.update(name.encode()); h.update(hashlib.sha256(self.read(name)).digest())
        return h.hexdigest()

    def payload(self, family, epoch):
        raw = self.read(f'{family}/epoch_{epoch:02d}.pth')
        data = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        if data['schema'] != 'p3-v1' or data['epoch'] != epoch or data['config'] != self.config['config']:
            raise ValueError('Invalid P3 checkpoint metadata')
        if data['pair_weight'] != self.config['methods'][family] or data['pair_config'] != self.config['pair_config']:
            raise ValueError('P3 checkpoint supervision differs')
        return data, hashlib.sha256(raw).hexdigest()


def replay_reference(ref, family, name, output, validate):
    """Re-evaluate P3 snapshots; do not train or give the reference more epochs."""
    output.mkdir(parents=True, exist_ok=True)
    history_path = output/'history.json'
    history = json.loads(history_path.read_text(encoding='utf-8')) if history_path.exists() else []
    epochs = ref.config['config']['epochs']
    for epoch in range(epochs+1):
        old, source_sha = ref.payload(family, epoch)
        path = output/f'epoch_{epoch:02d}.pth'
        if epoch < len(history):
            row = history[epoch]
            if row['epoch'] != epoch or row['reference_checkpoint_sha256'] != source_sha or audit.runner.sha(path) != row['checkpoint_sha256']:
                raise ValueError('Cached P3 replay changed')
            continue
        model = RoiRescorer(old['feature_dim'], QualityConfig(**old['config']), 'full')
        model.load_state_dict(old['model'], strict=True)
        views = validate(model, epoch)
        payload = {**old, 'schema': 'p4-v1', 'roi_mode': 'full', 'origin': 'P3_reference',
                   'reference_family': family, 'reference_checkpoint_sha256': source_sha, 'validation': views}
        atomic_save(payload, path)
        record = dict(epoch=epoch, validation=views, reference_checkpoint_sha256=source_sha,
                      checkpoint_sha256=audit.runner.sha(path), origin='P3_reference')
        history.append(record)
        atomic_json(history, history_path)
        print(f'[{name} replay {epoch:02d}/{epochs}] Tiny_AP50={views[name]["metrics"]["Tiny_AP50"]:.4f}', flush=True)
    best = max(history, key=lambda r: r['validation'][name]['metrics']['Tiny_AP50'])
    payload = torch.load(output/f'epoch_{best["epoch"]:02d}.pth', map_location='cpu', weights_only=True)
    payload.update(selected_view=name, selection_metric='Tiny_AP50_before_nms')
    atomic_save(payload, output/(name+'_best.pth'))
    return history, {name: dict(epoch=best['epoch'], metric=best['validation'][name]['metrics']['Tiny_AP50'], alpha=.25)}


def execute(args, output, report):
    torch.set_num_threads(args.threads)
    cache, gt = Path(args.train_cache), Path(args.train_gt)
    manifest_path = Path(args.manifest or args.train_cache+'.manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if manifest.get('schema') != 'p0r-roi-v1' or manifest.get('smoke'):
        raise ValueError('P4 requires full P0-R training cache')
    with P3Reference(args.p3_report) as ref:
        old_config = ref.config
        cache_sha, gt_sha = audit.runner.sha(cache), audit.runner.sha(gt)
        for key, value in [('cache_sha256', cache_sha), ('gt_sha256', gt_sha)]:
            if manifest[key] != value or old_config[key] != value:
                raise ValueError('Input differs from P3: '+key)
        if audit.runner.sha(manifest_path) != old_config['manifest_sha256']:
            raise ValueError('Manifest bytes differ from P3')
        if not args.smoke and args.epochs != old_config['config']['epochs']:
            raise ValueError('P4 epoch budget must equal P3')
        config = QualityConfig(**{**old_config['config'], 'epochs': 2 if args.smoke else args.epochs})
        pair_config = PairConfig(**old_config['pair_config'])
        if config.device != 'cpu' or config.rank_weight != 0 or pair_config.alpha != .25:
            raise ValueError('Unexpected P3 control configuration')
        source_paths = list(old_config['source_hashes']) + ['TileStitchingNet/p4.py', 'tools/kaggle_train_p4.py']
        sources = {p: audit.runner.sha(ROOT/p) for p in source_paths}
        for name, expected in old_config['source_hashes'].items():
            if sources[name] != expected:
                raise ValueError('Reused source changed since P3: '+name)
        packages = {name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'torchvision', 'pycocotools']}
        signature = dict(schema='p4-v1', cache_sha256=cache_sha, gt_sha256=gt_sha,
            manifest_sha256=audit.runner.sha(manifest_path), p3_reference_sha256=ref.fingerprint(),
            source_hashes=sources, config=asdict(config), pair_config=asdict(pair_config),
            methods={k: list(v) for k, v in METHODS.items()}, smoke=args.smoke, threads=args.threads,
            torch=str(torch.__version__), python=sys.version.split()[0], packages=packages,
            roi_columns=[12, 76], input_dimension=78, max_dets=200, nms_iou=.5,
            probe_rule='p4-probe-v1: 4 strata then SHA256(image_id), up to 8 training images',
            probe_epochs=sorted({0, min(10, config.epochs), config.epochs}),
            selection='per-method before-NMS TinyAP50, epoch0 through final; after-NMS uses same weight')
        run_path = output/'run_config.json'
        if run_path.exists() and json.loads(run_path.read_text(encoding='utf-8')) != signature:
            raise ValueError('Resume input/code/runtime/configuration changed; use a fresh output directory')
        atomic_json(signature, run_path)
        for name in source_paths:
            dest = output/'sources'/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT/name).read_bytes())
        split = ref.json('split.json')
        samples = OfflineDatasetIO.load_samples(str(cache))
        by_id = {s.image_id: s for s in samples}
        if len(samples) != len(by_id) or set(split['train_ids']) & set(split['val_ids']) or set(split['train_ids']) | set(split['val_ids']) != set(by_id):
            raise ValueError('Invalid or changed image split')
        old_geometry = {**ref.json('train_pair_geometry.json.gz'), **ref.json('validation_pair_geometry.json.gz')}
        if set(old_geometry) != set(by_id):
            raise ValueError('P3 geometry does not cover current cache')
        for s in samples:
            if s.node_features.shape != (len(s.scores), 76) or not np.isfinite(s.node_features).all():
                raise ValueError('Invalid 76D cache features')
            for attribute, field in [('boxes_global', 'boxes'), ('scores', 'scores'), ('patch_ids', 'patch_ids')]:
                value = getattr(s, attribute)
                if not np.array_equal(value, np.asarray(old_geometry[s.image_id][field], dtype=value.dtype).reshape(value.shape)):
                    raise ValueError('Candidate fields differ from P3: '+s.image_id)
        targets = QualityTargets(gt, config)
        annotation_data = json.loads(gt.read_text(encoding='utf-8'))
        annotation_result = annotation_audit(annotation_data, targets, split, old_geometry)
        atomic_json(audit.clean(annotation_result), output/'annotation_audit.json')
        compressed_json(output/'source_annotations.json.gz', annotation_data)
        report['annotation_audit'] = annotation_result
        if annotation_result['status'] != 'PASS':
            raise ValueError('Annotation audit needs review; see annotation_audit.json')
        train_ids = split['train_ids'][:8] if args.smoke else split['train_ids']
        val_ids = split['val_ids'][:4] if args.smoke else split['val_ids']
        training, validation = [by_id[i] for i in train_ids], [by_id[i] for i in val_ids]
        atomic_json(dict(train_ids=train_ids, val_ids=val_ids), output/'split.json')
        pair_sets = {}
        for scope, selected in [('train', training), ('validation', validation)]:
            pairs = build_pair_dataset(selected, annotation_data, targets, pair_config)
            old_pairs = ref.json(scope+'_pairs.json.gz')
            for s in selected:
                if pairs['pairs'][s.image_id] != old_pairs['pairs'][s.image_id]:
                    raise ValueError('P3 frozen pair list changed; preserve the P3 NumPy/NMS runtime: '+s.image_id)
            if not args.smoke and pairs['signature'] != old_pairs['signature']:
                raise ValueError('Frozen pair signature changed')
            compressed_json(output/(scope+'_pair_geometry.json.gz'), pairs.pop('geometry'))
            compressed_json(output/(scope+'_pairs.json.gz'), pairs)
            atomic_json({k: v for k, v in pairs.items() if k != 'pairs'}, output/(scope+'_pair_audit.json'))
            trace = full_nms_trace(selected, pairs['pairs'])
            compressed_json(output/(scope+'_nms_trace.json.gz'), trace)
            pair_sets[scope] = pairs
        if not pair_sets['train']['summary']['pairs']:
            raise ValueError('No reliable training pairs')
        batches = [dict(prepare(s, config, False, targets), **pair_tensors(pair_sets['train']['pairs'][s.image_id])) for s in training]
        chosen = select_probe(training, batches, pair_sets['train']['pairs'])
        probe_batches = [batches[i] for i in chosen]
        probe_pairs = [pair_sets['train']['pairs'][training[i].image_id] for i in chosen]
        atomic_json(dict(rule=signature['probe_rule'], train_indices=chosen,
            image_ids=[training[i].image_id for i in chosen], candidate_counts=[len(b['base']) for b in probe_batches],
            pair_counts=[len(p) for p in probe_pairs], epochs=signature['probe_epochs'], also_probe_selected_best=True), output/'gradient_probe_selection.json')
        diagnostics = ValidationDiagnostics(validation, gt, config)
        if not args.smoke:
            check_metrics(diagnostics.base_metrics, ref.json('baseline_metrics.json'), 'P4 baseline')
        atomic_json(audit.clean(diagnostics.base_metrics), output/'baseline_metrics.json')
        atomic_json(ref.json('p0r_traditional_reference.json'), output/'p0r_traditional_reference.json')
        compressed_json(output/'baseline_matches.json.gz', diagnostics.base_trace)
        compressed_json(output/'reliable_pairs.json.gz', {
            t: {s.image_id: p for s, p in zip(validation, grouped)} for t, grouped in diagnostics.pairs.items()})
        report['input'] = dict(train_images=len(training), validation_images=len(validation), test_evaluated=False,
            new_training_runs=4 if args.smoke else 2, reused_reference_families=[] if args.smoke else ['A025', 'C020'])
        report['pairs'] = {k: dict(signature=p['signature'], summary=p['summary']) for k, p in pair_sets.items()}
        print('P4 annotation audit PASS; normal/ignored GT:', annotation_result['derived_normal'], annotation_result['derived_ignore'], flush=True)
        print('P4:', report['input'], 'fixed alpha=.25, CPU', flush=True)
        results = {}
        for name, (roi_mode, weight, family) in METHODS.items():
            reference_history = ref.json(family+'/history.json')
            def validate(model, epoch):
                record = diagnostics.assess(model, pair_config.alpha)
                _, _, scores = diagnostics.predict(model, pair_config.alpha)
                record['training_rule_pairs'] = pair_results(validation,
                    [pair_sets['validation']['pairs'][s.image_id] for s in validation], scores)
                if epoch == 0:
                    check_metrics(record['metrics'], diagnostics.base_metrics, 'Epoch0 '+name)
                    if not args.smoke:
                        old, _ = ref.payload(family, 0)
                        if any(not torch.equal(v, old['model'][k]) for k, v in model.state_dict().items()):
                            raise AssertionError('P4 initial parameters differ from P3')
                if roi_mode == 'full' and not args.smoke:
                    previous = reference_history[epoch]['validation'][family]['metrics']
                    record['p3_metric_differences'] = audit.metric_diff(record['metrics'], previous)
                    check_metrics(record['metrics'], previous, 'P4 P3 reference replay')
                if epoch in signature['probe_epochs']:
                    record['gradient_probe'] = gradient_probe(model, probe_batches, probe_pairs, config, pair_config, weight)
                return {name: record}

            if roi_mode == 'full' and not args.smoke:
                history, selections = replay_reference(ref, family, name, output/name, validate)
            else:
                history, selections = train_roi(batches, config, pair_config, weight,
                    pair_sets['train']['signature'], roi_mode, output/name, validate)
            best_path = output/name/(name+'_best.pth')
            model, payload = load_p4_checkpoint(best_path)
            detail = diagnostics.detailed(model, .25, output, name+'_best', payload['epoch'])
            check_metrics(detail['modes']['before_nms']['metrics'], history[payload['epoch']]['validation'][name]['metrics'], name+' best reload')
            atomic_json(audit.clean(dict(epoch=payload['epoch'], roi_mode=roi_mode,
                diagnostics=gradient_probe(model, probe_batches, probe_pairs, config, pair_config, weight))), output/(name+'_best_gradients.json'))
            last_model, _ = load_p4_checkpoint(output/name/f'epoch_{config.epochs:02d}.pth')
            last = diagnostics.detailed(last_model, .25, output, name+'_last', config.epochs)
            check_metrics(last['modes']['before_nms']['metrics'], history[-1]['validation'][name]['metrics'], name+' last reload')
            values = [r['validation'][name]['metrics']['Tiny_AP50'] for r in history]
            results[name] = dict(roi_mode=roi_mode, pair_weight=weight, alpha=.25,
                origin='P3_reference' if roi_mode == 'full' and not args.smoke else 'trained_from_initialization',
                parameter_count=sum(p.numel() for p in model.parameters()), best_epoch=selections[name]['epoch'],
                best_checkpoint=str(best_path.relative_to(output)), best_checkpoint_sha256=audit.runner.sha(best_path),
                best=detail['modes'], last=last['modes'], best_to_last_tiny_ap50_drop=max(values)-values[-1])
            atomic_json(results, output/'validation_results.json')
        report['results'] = results
        comparisons = {}
        for group in ['A', 'C']:
            comparisons[group+'_zero_minus_full'] = audit.metric_diff(results[group+'_zero']['best']['before_nms']['metrics'], results[group+'_full']['best']['before_nms']['metrics'])
        for mode in ['full', 'zero']:
            comparisons['C_minus_A_'+mode] = audit.metric_diff(results['C_'+mode]['best']['before_nms']['metrics'], results['A_'+mode]['best']['before_nms']['metrics'])
        atomic_json(audit.clean(comparisons), output/'ablation_comparisons.json')
        report['comparisons'] = comparisons
        report['input_files_unchanged'] = (audit.runner.sha(cache) == cache_sha and audit.runner.sha(gt) == gt_sha
            and audit.runner.sha(manifest_path) == signature['manifest_sha256'] and ref.fingerprint() == signature['p3_reference_sha256'])
        if not report['input_files_unchanged']:
            raise AssertionError('Inputs changed during run')
        report['status'] = 'P4_SMOKE_COMPLETED' if args.smoke else 'P4_ROI_COMPLETED'
        report['notes'] = ['P4 input-ablation step only; no relation-network D or coordinate changes.',
            'Full-ROI controls are P3 weights re-evaluated at every epoch; smoke trains all four variants on small subsets.',
            'Zero-ROI models mask columns [12:76] inside forward and retain all 78 inputs and the same parameter count.',
            'Train/validation pairs must exactly match P3; complete original-score NMS ordering is exported.',
            'QFL remains unscaled; pair and deployment alpha=.25, weights=0/.2. Only ROI availability changes.',
            'Fixed training-only gradient probe is diagnostic, before clipping, and does not mutate optimizer/grads/RNG.',
            'Gradient subset contributions preserve original denominators; cosine is undefined for zero gradients.',
            'All methods have epoch0 through18 selection opportunity. No test evaluation; single seed.',
            'Latency_ms=0 is a placeholder. Source annotations and candidate geometry are included for later metric audit.']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--train-gt', required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--p3-report')
    parser.add_argument('--output', default='/kaggle/working/p4_full')
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
        atomic_json(audit.clean(report), output/'p4_summary.json')
        with zipfile.ZipFile(output/'p4_reports.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for p in sorted(output.rglob('*')):
                if p.is_file() and p.suffix in {'.json', '.gz', '.npz', '.py', '.pth'}:
                    archive.write(p, p.relative_to(output))
        print('Status:', report['status'], '\nSend back:', output/'p4_reports.zip', flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
