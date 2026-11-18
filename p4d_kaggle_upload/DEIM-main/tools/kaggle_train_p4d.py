"""D: one local relation training run against frozen full-ROI P4 controls."""
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
from TileStitchingNet.p3 import PairConfig, pair_tensors, competition_loss
from TileStitchingNet.p4 import RoiRescorer, annotation_audit, load_p4_checkpoint
from kaggle_train_p4b import replay_control
from TileStitchingNet.p4 import train_roi
from TileStitchingNet.p4d import (ARCHITECTURE, train_relation, load_relation_checkpoint, benchmark_postprocessor)
from TileStitchingNet.pipeline_deim_tilestitching import OfflineDatasetIO
from TileStitchingNet.stage2 import QualityConfig, QualityTargets, prepare, quality_loss
from TileStitchingNet.p1 import scaled_scores

METHODS = {'A_full': ('pointwise', 0.), 'C_full': ('pointwise', .2),
           'D_local': ('relation', .2)}


def export_graphs(output, ids, batches, pairs):
    """Archive neighbor order/masks and geometry hashes, including tied kNN order."""
    signatures = {}
    for scope in ids:
        rows, neighbors, masks, shapes = [], [], [], []
        totals = dict(nodes=0, directed_nonself_edges=0, pairs=0,
                      either_direction=0, good_reads_bad=0, bad_reads_good=0)
        for iid, data, plist in zip(ids[scope], batches[scope], pairs[scope]):
            j = data['neighbors'].cpu().numpy()
            mask = data['neighbor_mask'].cpu().numpy()
            geo = data['geometry'].cpu().numpy()
            n, k = j.shape
            assert geo.shape == (n, k, 8) and mask.shape == (n, k)
            assert np.isfinite(geo).all()
            if n:
                assert j.min() >= 0 and j.max() < n
            actual = mask & (j != np.arange(n)[:, None])
            edges = {(i, int(jj)) for i in range(n) for jj in j[i, actual[i]]}
            a = sum((p['good'], p['bad']) in edges for p in plist)
            b = sum((p['bad'], p['good']) in edges for p in plist)
            either = sum((p['good'], p['bad']) in edges or (p['bad'], p['good']) in edges for p in plist)
            values = dict(nodes=n, directed_nonself_edges=len(edges), pairs=len(plist),
                          either_direction=either, good_reads_bad=a, bad_reads_good=b)
            for key in totals:
                totals[key] += values[key]
            rows.append(dict(image_id=iid, shape=[n, k], **values,
                max_nonself_neighbors=int(actual.sum(1).max()) if n else 0,
                neighbors_sha256=sha_bytes(j.tobytes()), geometry_sha256=sha_bytes(geo.tobytes()),
                mask_sha256=sha_bytes(mask.tobytes())))
            neighbors.append(j.reshape(-1)); masks.append(mask.reshape(-1)); shapes.append([n, k])
        logical = sha_bytes(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode())
        marker = output / (scope + '_graph.json')
        if marker.exists() and json.loads(marker.read_text(encoding='utf8'))['signature'] != logical:
            raise ValueError('Graph changed during resume: ' + scope)
        np.savez_compressed(output / (scope + '_graph.npz'), image_ids=np.array(ids[scope]),
            shapes=np.array(shapes, dtype=np.int64), offsets=np.cumsum([0]+[len(j) for j in neighbors]),
            neighbors=np.concatenate(neighbors), neighbor_mask=np.concatenate(masks))
        atomic_json(dict(signature=logical, summary=totals, images=rows,
            note='Existing stage2 candidate_graph; self messages excluded. Geometry reconstructed from frozen boxes/scores and stored neighbor indices.'), marker)
        signatures[scope] = logical
    return signatures


def training_fit(model, samples, batches, pairs, config):
    scores, qfls, ranks = [], [], []
    model.eval()
    with torch.inference_mode():
        for sample, data in zip(samples, batches):
            out = model(data)
            scores.append(scaled_scores(sample.scores, out['delta'].cpu().numpy(), .25))
            if len(data['base']) and (bool(data['valid'].any()) or data['pair_group_count']):
                qfls.append(float(quality_loss(out, data, config)[1]))
                ranks.append(float(competition_loss(out, data, .25)))
    return dict(images=len(samples), eligible_images=len(qfls), qfl=float(np.mean(qfls)),
                softplus_pair=float(np.mean(ranks)), pair_results=pair_results(samples, pairs, scores))


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
                  'TileStitchingNet/p4b.py', 'tools/kaggle_train_p4b.py',
                  'TileStitchingNet/p4d.py', 'tools/kaggle_train_p4d.py', 'tools/kaggle_run_p4d.py']:
            sources[p] = sha_file(ROOT / p)
        names = ['run_config.json', 'p4_summary.json', 'split.json', 'baseline_metrics.json',
                 'p0r_traditional_reference.json']
        for scope in ['train', 'validation']:
            names += [scope + '_pairs.json.gz', scope + '_pair_geometry.json.gz', scope + '_nms_trace.json.gz']
        for method in ['A_full', 'C_full']:
            names += [method + '/history.json', ref.summary['results'][method]['best_checkpoint'].replace('\\', '/')]
            names += [f'{method}/epoch_{e:02d}.pth' for e in range(previous['config']['epochs'] + 1)]
        references = {n: sha_bytes(ref.read(n)) for n in names}
        signature = dict(schema='p4d-v1', inputs=input_hashes, reference_hashes=references,
            source_hashes=sources, config=asdict(config), pair_config=asdict(pair_config),
            architecture=ARCHITECTURE, methods={k: list(v) for k, v in METHODS.items()}, alpha=.25,
            relation='One residual attention block, width32, zero output projection; existing candidate_graph; excludes self messages',
            denominator='original softplus: all frozen pairs per GT, then mean GTs per image',
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
            batches[scope] = [dict(prepare(s, config, True, targets), **pair_tensors(p))
                              for s, p in zip(selected[scope], pairs[scope])]
        graph_signature = export_graphs(output, ids, batches, pairs)
        diagnostics = ValidationDiagnostics(selected['validation'], files['gt'], config)
        # The pointwise controls ignore extra graph keys; D consumes the same label-free graph.
        diagnostics.inputs = [prepare(s, config, True) for s in selected['validation']]
        if not args.smoke:
            check_metrics(diagnostics.base_metrics, ref.json('baseline_metrics.json'), 'D baseline')
        atomic_json(audit.clean(diagnostics.base_metrics), output / 'baseline_metrics.json')
        atomic_json(ref.json('p0r_traditional_reference.json'), output / 'p0r_traditional_reference.json')
        compressed_json(output / 'baseline_matches.json.gz', diagnostics.base_trace)
        report['input'] = dict(train_images=len(ids['train']), validation_images=len(ids['validation']),
                               test_evaluated=False, new_training_runs=3 if args.smoke else 1,
                               reused_controls=[] if args.smoke else ['A_full', 'C_full'])
        report['results'] = {}
        initial = torch.load(io.BytesIO(ref.read('C_full/epoch_00.pth')), map_location='cpu', weights_only=True)['model']
        for method, (architecture, weight) in METHODS.items():
            def validate(model, epoch):
                record = diagnostics.assess(model, .25)
                if epoch == 0:
                    check_metrics(record['metrics'], diagnostics.base_metrics, 'Initial baseline')
                    if any(not torch.equal(model.state_dict()[k], v) for k, v in initial.items()):
                        raise AssertionError('Initial weights differ from P4')
                _, _, scores = diagnostics.predict(model, .25)
                record['training_rule_pairs'] = pair_results(selected['validation'], pairs['validation'], scores)
                return {method: record}
            folder = output / method
            replay = method != 'D_local' and not args.smoke
            if replay:
                history, best = replay_control(ref, method, folder, validate)
                loader = load_p4_checkpoint
            else:
                if method == 'D_local':
                    history, best = train_relation(batches['train'], config, pair_config, weight,
                        signatures['train'], graph_signature['train'], folder, validate, initial_core=initial)
                    loader = load_relation_checkpoint
                else:
                    history, best = train_roi(batches['train'], config, pair_config, weight,
                        signatures['train'], 'full', folder, validate)
                    loader = load_p4_checkpoint
            best_path = folder / (method + '_best.pth')
            details = {}
            for tag, path in [('best', best_path), ('last', folder / f'epoch_{config.epochs:02d}.pth')]:
                model, payload = loader(path)
                detail = diagnostics.detailed(model, .25, output, method + '_' + tag, payload['epoch'])
                check_metrics(detail['modes']['before_nms']['metrics'], history[payload['epoch']]['validation'][method]['metrics'], method + ' reload')
                details[tag] = detail['modes']
                fit = training_fit(model, selected['train'], batches['train'], pairs['train'], config)
                if method == 'D_local':
                    fit['relation_output_weight_norm'] = float(model.relation_output.weight.detach().norm())
                atomic_json(audit.clean(fit), output / (method + '_' + tag + '_training_fit.json'))
            report['results'][method] = dict(origin='P4_reference' if replay else 'trained_from_initialization',
                objective='softplus' if weight else 'qfl_only', architecture=architecture,
                pair_weight=weight, roi_mode='full', parameter_count=sum(p.numel() for p in model.parameters()),
                best_epoch=best[method]['epoch'], best_checkpoint=best_path.relative_to(output).as_posix(),
                best_checkpoint_sha256=sha_file(best_path), **details)
            atomic_json(report['results'], output / 'validation_results.json')
        benchmark_results = {}
        for method in METHODS:
            loader = load_relation_checkpoint if method == 'D_local' else load_p4_checkpoint
            model, _ = loader(output / method / (method + '_best.pth'))
            timing, rows = benchmark_postprocessor(model, selected['validation'], repeats=1 if args.smoke else 3)
            benchmark_results[method] = timing
            compressed_json(output / (method + '_latency_samples.json.gz'), rows)
        atomic_json(benchmark_results, output / 'latency_results.json')
        report['latency'] = benchmark_results
        report['comparisons'] = {base: audit.metric_diff(report['results']['D_local']['best']['before_nms']['metrics'],
            report['results'][base]['best']['before_nms']['metrics']) for base in ['A_full', 'C_full']}
        atomic_json(audit.clean(report['comparisons']), output / 'comparisons.json')
        if any(sha_file(p) != input_hashes[k] for k, p in files.items()) or any(sha_bytes(ref.read(n)) != h for n, h in references.items()):
            raise AssertionError('Inputs changed during run')
        report['status'] = 'P4D_SMOKE_COMPLETED' if args.smoke else 'P4D_TRAINING_COMPLETED'
        report['notes'] = ['D keeps full ROI, QFL and original softplus pair loss; lambda=.2, deployment alpha=.25.',
            'Only D adds one sparse residual relation block. Same candidate_graph; GT never enters graph/model inputs.',
            'Core encoder/head tensors load directly from P4 epoch0. Relation output projection is zero, preserving initial scores.',
            'Parameter count increases; D-C is an architecture comparison, not a parameter-matched capacity ablation.',
            'Controls replay all P4 snapshots and verify metrics; smoke retrains small controls for pipeline checks.',
            'Official metric Latency_ms remains a placeholder; use separate measured latency_results.json.',
            'Timing includes cached-input prepare/graph, score and NMS/top200; excludes detector/ROI extraction and I/O.',
            'Single seed validation; no test evaluation.']
        report['artifact_hashes'] = {p.relative_to(output).as_posix(): sha_file(p) for p in output.rglob('*')
            if p.is_file() and p.suffix in {'.json', '.gz', '.npz', '.pth', '.py'} and p.name != 'p4d_summary.json'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--train-gt', required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--p4-report')
    parser.add_argument('--output', default='/kaggle/working/p4d_full')
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
        atomic_json(audit.clean(report), output / 'p4d_summary.json')
        with zipfile.ZipFile(output / 'p4d_reports.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(output.rglob('*')):
                if p.is_file() and p.suffix in {'.json', '.gz', '.npz', '.py', '.pth'}:
                    z.write(p, p.relative_to(output).as_posix())
        print('Status:', report['status'], '\nSend back:', output / 'p4d_reports.zip', flush=True)


if __name__ == '__main__':
    main()
