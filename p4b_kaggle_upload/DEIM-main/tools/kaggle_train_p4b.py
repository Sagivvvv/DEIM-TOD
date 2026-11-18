"""P4-B: one fixed-margin training run against frozen full-ROI P4 controls."""
import argparse
from dataclasses import asdict
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
from kaggle_diagnose_p1 import check_metrics, compressed_json
from kaggle_diagnose_p4a import P4Reference, sha_file, sha_bytes
from kaggle_train_p2 import ValidationDiagnostics
from TileStitchingNet.p1 import pair_results
from TileStitchingNet.p2 import atomic_json
from TileStitchingNet.p3 import PairConfig, pair_tensors
from TileStitchingNet.p4 import RoiRescorer, annotation_audit, load_p4_checkpoint
from TileStitchingNet.p4b import (MARGIN, PAIR_WEIGHT, train_margin,
    margin_diagnostics, load_p4b_checkpoint)
from TileStitchingNet.pipeline_deim_tilestitching import OfflineDatasetIO
from TileStitchingNet.stage2 import QualityConfig, QualityTargets, prepare

METHODS = {'A_full': ('softplus', 0.), 'C_full': ('softplus', PAIR_WEIGHT),
           'M010': ('hinge', PAIR_WEIGHT)}


def replay_control(ref, method, output, validate):
    """Same epoch selection budget, no retraining or relabeling of controls."""
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / 'history.json'
    history = json.loads(history_path.read_text(encoding='utf8')) if history_path.exists() else []
    previous = ref.json(method + '/history.json')
    for epoch in range(ref.config['config']['epochs'] + 1):
        name = f'{method}/epoch_{epoch:02d}.pth'
        raw = ref.read(name)
        path = output / f'epoch_{epoch:02d}.pth'
        if epoch < len(history):
            if history[epoch]['epoch'] != epoch or history[epoch]['checkpoint_sha256'] != sha_bytes(raw) or sha_file(path) != sha_bytes(raw):
                raise ValueError('P4 control replay cache changed')
            continue
        payload = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        if payload['schema'] != 'p4-v1' or payload['epoch'] != epoch or payload['config'] != ref.config['config'] or payload['roi_mode'] != 'full':
            raise ValueError('Invalid P4 control checkpoint')
        model = RoiRescorer(payload['feature_dim'], QualityConfig(**payload['config']), 'full')
        model.load_state_dict(payload['model'], strict=True)
        views = validate(model, epoch)
        check_metrics(views[method]['metrics'], previous[epoch]['validation'][method]['metrics'], 'P4 control replay')
        path.write_bytes(raw)
        history.append(dict(epoch=epoch, validation=views, origin='P4_reference', checkpoint_sha256=sha_bytes(raw)))
        atomic_json(history, history_path)
        print(f'[{method} replay {epoch:02d}] TinyAP50={views[method]["metrics"]["Tiny_AP50"]:.4f}', flush=True)
    best = max(history, key=lambda r: r['validation'][method]['metrics']['Tiny_AP50'])
    expected = ref.summary['results'][method]
    if best['epoch'] != expected['best_epoch']:
        raise ValueError('P4 reference selected epoch changed')
    raw = ref.read(expected['best_checkpoint'].replace('\\', '/'))
    if sha_bytes(raw) != expected['best_checkpoint_sha256']:
        raise ValueError('P4 best control hash mismatch')
    (output / (method + '_best.pth')).write_bytes(raw)
    return history, {method: dict(epoch=best['epoch'], metric=best['validation'][method]['metrics']['Tiny_AP50'], alpha=.25)}


def execute(args, output, report):
    torch.set_num_threads(args.threads)
    files = dict(cache=Path(args.train_cache), gt=Path(args.train_gt),
                 manifest=Path(args.manifest or args.train_cache + '.manifest.json'))
    with P4Reference(args.p4_report) as ref:
        previous = ref.config
        if previous['config']['device'] != 'cpu' or previous['config']['rank_weight'] != 0 or previous['pair_config']['alpha'] != .25:
            raise ValueError('Unexpected P4 settings')
        if not args.smoke and args.epochs != previous['config']['epochs']:
            raise ValueError('Use the same epoch budget as P4')
        config = QualityConfig(**{**previous['config'], 'epochs': 2 if args.smoke else args.epochs})
        pair_config = PairConfig(**previous['pair_config'])
        input_hashes = {k: sha_file(p) for k, p in files.items()}
        for k, value in input_hashes.items():
            if previous[k + '_sha256'] != value:
                raise ValueError('Frozen input changed: ' + k)
        sources = {p: sha_file(ROOT / p) for p in previous['source_hashes']}
        if sources != previous['source_hashes']:
            raise ValueError('Frozen P4 source code changed')
        for p in ['TileStitchingNet/p4a.py', 'tools/kaggle_diagnose_p4a.py',
                  'TileStitchingNet/p4b.py', 'tools/kaggle_train_p4b.py', 'tools/kaggle_run_p4b.py']:
            sources[p] = sha_file(ROOT / p)
        names = ['run_config.json', 'p4_summary.json', 'split.json', 'baseline_metrics.json',
                 'p0r_traditional_reference.json']
        for scope in ['train', 'validation']:
            names += [scope + '_pairs.json.gz', scope + '_pair_geometry.json.gz', scope + '_nms_trace.json.gz']
        for method in ['A_full', 'C_full']:
            names += [method + '/history.json', ref.summary['results'][method]['best_checkpoint'].replace('\\', '/')]
            names += [f'{method}/epoch_{e:02d}.pth' for e in range(previous['config']['epochs'] + 1)]
        references = {n: sha_bytes(ref.read(n)) for n in names}
        signature = dict(schema='p4b-v1', inputs=input_hashes, reference_hashes=references,
            source_hashes=sources, config=asdict(config), pair_config=asdict(pair_config),
            margin=MARGIN, methods={k: list(v) for k, v in METHODS.items()}, alpha=.25,
            margin_unit='deployment logit difference; m=.1 fixed before P4-B training, no sweep',
            denominator='all frozen pairs per GT, including inactive pairs; then mean GTs per image',
            smoke=args.smoke, threads=args.threads, torch=str(torch.__version__), numpy=np.__version__,
            python=sys.version.split()[0], packages={p: importlib.metadata.version(p) for p in ['scipy', 'torchvision', 'pycocotools']},
            selection='Each method: maximum validation before-NMS TinyAP50 over epoch0..18; after-NMS uses the same weight')
        run_path = output / 'run_config.json'
        if run_path.exists() and json.loads(run_path.read_text(encoding='utf8')) != signature:
            raise ValueError('Resume inputs/code/runtime/settings changed; use fresh output')
        atomic_json(signature, run_path)
        for name in sources:
            dest = output / 'sources' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT / name).read_bytes())
        split = ref.json('split.json')
        samples = OfflineDatasetIO.load_samples(str(files['cache']))
        by_id = {s.image_id: s for s in samples}
        if len(samples) != len(by_id) or set(split['train_ids']) & set(split['val_ids']) or set(by_id) != set(split['train_ids']) | set(split['val_ids']):
            raise ValueError('Invalid image split')
        geometry = {**ref.json('train_pair_geometry.json.gz'), **ref.json('validation_pair_geometry.json.gz')}
        for s in samples:
            if s.node_features.shape != (len(s.scores), 76) or not np.isfinite(s.node_features).all():
                raise ValueError('Invalid features')
            for attr, field in [('boxes_global', 'boxes'), ('scores', 'scores'), ('patch_ids', 'patch_ids')]:
                value = getattr(s, attr)
                if not np.array_equal(value, np.asarray(geometry[s.image_id][field], dtype=value.dtype).reshape(value.shape)):
                    raise ValueError('Candidate geometry/order changed')
        targets = QualityTargets(files['gt'], config)
        annotations = json.loads(files['gt'].read_text(encoding='utf8'))
        ann_audit = annotation_audit(annotations, targets, split, geometry)
        atomic_json(ann_audit, output / 'annotation_audit.json')
        if ann_audit['status'] != 'PASS':
            raise ValueError('Annotation audit failed')
        compressed_json(output / 'source_annotations.json.gz', annotations)
        ids = dict(train=split['train_ids'][:8] if args.smoke else split['train_ids'],
                   validation=split['val_ids'][:4] if args.smoke else split['val_ids'])
        selected = {scope: [by_id[i] for i in image_ids] for scope, image_ids in ids.items()}
        atomic_json(dict(train_ids=ids['train'], val_ids=ids['validation']), output / 'split.json')
        pairs, batches, signatures = {}, {}, {}
        for scope in ids:
            old = ref.json(scope + '_pairs.json.gz')
            if old['signature'] != ref.summary['pairs'][scope]['signature']:
                raise ValueError('Frozen pair signature changed')
            pairs[scope] = [old['pairs'][i] for i in ids[scope]]
            signatures[scope] = old['signature']
            compressed_json(output / (scope + '_pairs.json.gz'), old)
            compressed_json(output / (scope + '_pair_geometry.json.gz'), ref.json(scope + '_pair_geometry.json.gz'))
            compressed_json(output / (scope + '_nms_trace.json.gz'), ref.json(scope + '_nms_trace.json.gz'))
            batches[scope] = [dict(prepare(s, config, False, targets), **pair_tensors(p))
                              for s, p in zip(selected[scope], pairs[scope])]
        diagnostics = ValidationDiagnostics(selected['validation'], files['gt'], config)
        if not args.smoke:
            check_metrics(diagnostics.base_metrics, ref.json('baseline_metrics.json'), 'P4-B baseline')
        atomic_json(audit.clean(diagnostics.base_metrics), output / 'baseline_metrics.json')
        atomic_json(ref.json('p0r_traditional_reference.json'), output / 'p0r_traditional_reference.json')
        compressed_json(output / 'baseline_matches.json.gz', diagnostics.base_trace)
        report['input'] = dict(train_images=len(ids['train']), validation_images=len(ids['validation']),
                               test_evaluated=False, new_training_runs=3 if args.smoke else 1,
                               reused_controls=[] if args.smoke else ['A_full', 'C_full'])
        report['results'] = {}
        initial = torch.load(io.BytesIO(ref.read('C_full/epoch_00.pth')), map_location='cpu', weights_only=True)['model']
        for method, (objective, weight) in METHODS.items():
            def validate(model, epoch):
                record = diagnostics.assess(model, .25)
                if epoch == 0:
                    check_metrics(record['metrics'], diagnostics.base_metrics, 'Initial baseline')
                    if any(not torch.equal(v, initial[k]) for k, v in model.state_dict().items()):
                        raise AssertionError('Initial weights differ from P4')
                record['margin_diagnostics'] = {}
                for scope in ids:
                    stats, _ = margin_diagnostics(model, batches[scope], pairs[scope], ids[scope])
                    record['margin_diagnostics'][scope] = stats
                _, _, scores = diagnostics.predict(model, .25)
                record['training_rule_pairs'] = pair_results(selected['validation'], pairs['validation'], scores)
                return {method: record}
            folder = output / method
            replay = method != 'M010' and not args.smoke
            if replay:
                history, best = replay_control(ref, method, folder, validate)
                loader = load_p4_checkpoint
            else:
                history, best = train_margin(batches['train'], config, pair_config, weight,
                    signatures['train'], folder, validate, objective=objective)
                loader = load_p4b_checkpoint
            best_path = folder / (method + '_best.pth')
            details = {}
            for tag, path in [('best', best_path), ('last', folder / f'epoch_{config.epochs:02d}.pth')]:
                model, payload = loader(path)
                detail = diagnostics.detailed(model, .25, output, method + '_' + tag, payload['epoch'])
                check_metrics(detail['modes']['before_nms']['metrics'], history[payload['epoch']]['validation'][method]['metrics'], method + ' reload')
                details[tag] = detail['modes']
                for scope in ids:
                    stats, gap_rows = margin_diagnostics(model, batches[scope], pairs[scope], ids[scope])
                    filename = method + '_' + tag + '_' + scope + '_margins.json.gz'
                    compressed_json(output / filename, dict(summary=stats, images=gap_rows))
            report['results'][method] = dict(origin='P4_reference' if replay else 'trained_from_initialization',
                objective=objective if weight else 'qfl_only', margin=MARGIN if objective == 'hinge' else None,
                pair_weight=weight, roi_mode='full', parameter_count=sum(p.numel() for p in model.parameters()),
                best_epoch=best[method]['epoch'], best_checkpoint=best_path.relative_to(output).as_posix(),
                best_checkpoint_sha256=sha_file(best_path), **details)
            atomic_json(report['results'], output / 'validation_results.json')
        report['comparisons'] = {base: audit.metric_diff(report['results']['M010']['best']['before_nms']['metrics'],
            report['results'][base]['best']['before_nms']['metrics']) for base in ['A_full', 'C_full']}
        atomic_json(audit.clean(report['comparisons']), output / 'comparisons.json')
        if any(sha_file(p) != input_hashes[k] for k, p in files.items()) or any(sha_bytes(ref.read(n)) != h for n, h in references.items()):
            raise AssertionError('Inputs changed during run')
        report['status'] = 'P4B_SMOKE_COMPLETED' if args.smoke else 'P4B_TRAINING_COMPLETED'
        report['notes'] = ['Fixed m=.1 in deployment logits, lambda=.2, alpha=.25; full ROI and no coordinate changes.',
            'All frozen reliable pairs retained; active mask depends on current gap, not base_order.',
            'Inactive pairs remain in original per-GT denominators. ReLU subgradient is zero at exact margin.',
            'Hinge differs in both gradient magnitude and saturation from softplus; not an isolated stopping-only ablation.',
            'Full controls replay every P4 epoch and verify metrics. Smoke trains all three variants on subsets.',
            'Pair margin counts use deployment logits; official matching/ranking records use final scores.',
            'Latency_ms is a placeholder. Single seed validation; no test evaluation.']
        report['artifact_hashes'] = {p.relative_to(output).as_posix(): sha_file(p) for p in output.rglob('*')
            if p.is_file() and p.suffix in {'.json', '.gz', '.npz', '.pth', '.py'} and p.name != 'p4b_summary.json'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--train-gt', required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--p4-report')
    parser.add_argument('--output', default='/kaggle/working/p4b_full')
    parser.add_argument('--epochs', type=int, default=18)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if min(args.epochs, args.threads) < 1:
        parser.error('epochs and threads must be positive')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error('Use a fresh output directory or --resume')
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status='RUNNING', arguments=vars(args))
    try:
        execute(args, output, report)
    except Exception:
        report.update(status='ERROR', error=traceback.format_exc())
        raise
    finally:
        atomic_json(audit.clean(report), output / 'p4b_summary.json')
        with zipfile.ZipFile(output / 'p4b_reports.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(output.rglob('*')):
                if p.is_file() and p.suffix in {'.json', '.gz', '.npz', '.py', '.pth'}:
                    z.write(p, p.relative_to(output).as_posix())
        print('Status:', report['status'], '\nSend back:', output / 'p4b_reports.zip', flush=True)


if __name__ == '__main__':
    main()
