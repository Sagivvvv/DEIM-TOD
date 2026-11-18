"""P1: frozen validation-only alpha sweep, suppression, ranking and budget diagnostics."""
import argparse
import gzip
import hashlib
import io
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools'))
import numpy as np
import torch
import kaggle_audit_p0 as audit
from TileStitchingNet.pipeline_deim_tilestitching import OfflineDatasetIO
from TileStitchingNet.stage1 import OfficialTinyEvaluator
from TileStitchingNet.stage2 import QualityConfig, QualityRescorer, QualityTargets, prepare
from TileStitchingNet.p0r import frozen_fields
from TileStitchingNet.p1 import (nms_trace, scaled_scores, CandidateLabels, reliable_pairs,
    pair_results, graph_coverage, score_diagnostics, OfficialMatchTrace, compare_traces)

ALPHAS = [0, .1, .25, .5, 1]


class ReportSource:
    """Kaggle may expand an uploaded ZIP into a directory automatically."""
    def __init__(self, path):
        self.path = Path(path)
        if self.path.name == 'p0r_summary.json': self.path = self.path.parent
        self.zip = None

    def __enter__(self):
        if self.path.is_file() and zipfile.is_zipfile(self.path): self.zip = zipfile.ZipFile(self.path)
        elif not (self.path/'p0r_summary.json').is_file():
            raise FileNotFoundError('Provide p0r_reports.zip or its extracted folder containing p0r_summary.json: '+str(self.path))
        return self

    def __exit__(self, *args):
        if self.zip is not None: self.zip.close()

    def read(self, name):
        return self.zip.read(name) if self.zip is not None else (self.path/name).read_bytes()

    def fingerprint(self):
        if self.path.is_file(): return audit.runner.sha(self.path)
        names = ['p0r_summary.json','split.json','p0r_train.pt.manifest.json','validation_metrics.json']
        for family in ['pointwise','local']:
            names += [family+'_completed.json', family+'_best.pth']
            names += [family+'_snapshots/'+tag+'.pth' for tag in ['epoch_06','epoch_18']]
        h = hashlib.sha256()
        for name in sorted(names):
            h.update(name.encode()); h.update(hashlib.sha256(self.read(name)).digest())
        return h.hexdigest()


def reference_hash(path):
    with ReportSource(path) as source: return source.fingerprint()


def compressed_json(path, data):
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        json.dump(audit.clean(data), f, ensure_ascii=False, allow_nan=False)


def check_metrics(actual, expected, label):
    for key, value in expected.items():
        if key == 'Latency_ms': continue
        if value is None or not np.isfinite(value):
            if actual.get(key) is None or not np.isfinite(actual[key]): continue
            raise ValueError(f'{label}: undefined metric became defined: {key}')
        if key not in actual or not np.isfinite(actual[key]) or abs(actual[key]-value) > 1e-4:
            raise ValueError(f'{label}: metric mismatch {key}: {actual.get(key)} != {value}')


def aggregate_coverage(rows):
    result = {}
    for image_rows in rows:
        for r in image_rows:
            for key in ['all', 'cross_tile' if r['cross_tile'] else 'same_tile', r['suppressed_box_size']]:
                c = result.setdefault(key, dict(edges=0, suppressed_reads_winner=0, winner_reads_suppressed=0, either_direction=0))
                c['edges'] += 1
                for flag in ['suppressed_reads_winner', 'winner_reads_suppressed']: c[flag] += int(r[flag])
                c['either_direction'] += r['suppressed_reads_winner'] or r['winner_reads_suppressed']
    for c in result.values():
        c['either_direction_fraction'] = c['either_direction']/c['edges']
    return result


def evaluate(evaluator, samples, indices, scores):
    return evaluator(samples, [s.boxes_global[k] for s, k in zip(samples, indices)],
                     [sc[k] for sc, k in zip(scores, indices)])


def execute(args, out, report):
    torch.set_num_threads(args.threads)
    cache, manifest_path, gt_path, archive = map(Path, [args.train_cache, args.manifest or args.train_cache+'.manifest.json', args.train_gt, args.p0r_report])
    meta = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if meta.get('schema') != 'p0r-roi-v1' or meta.get('smoke') or not meta.get('candidate_fields_unchanged'):
        raise ValueError('P1 requires the full P0-R cache and its manifest, not the old stage2 cache')
    source_paths = ['tools/kaggle_diagnose_p1.py', 'TileStitchingNet/p1.py', 'TileStitchingNet/stage2.py',
                    'TileStitchingNet/stage1.py', 'TileStitchingNet/datasetBuilder.py',
                    'TileStitchingNet/pipeline_deim_tilestitching.py', 'TileStitchingNet/p0r.py',
                    'tools/kaggle_audit_p0.py', 'tools/kaggle_train_stage2.py',
                    'tools/huicv/evaluation/expand_cocofmt_eval.py']
    packages = {}
    for name in ['numpy','scipy','torchvision','pycocotools']:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = 'unknown'
    signature = dict(schema='p1-v1', cache_sha256=audit.runner.sha(cache), gt_sha256=audit.runner.sha(gt_path),
                     reference_sha256=reference_hash(archive), manifest_sha256=audit.runner.sha(manifest_path),
                     source_hashes={p:audit.runner.sha(ROOT/p) for p in source_paths},
                     smoke=args.smoke, threads=args.threads, torch=str(torch.__version__),
                     python=sys.version.split()[0], packages=packages,
                     alphas=ALPHAS, snapshots=['best', 'epoch_06', 'epoch_18'], nms_iou=.5,
                     max_dets=200, diagnostic_iou=[.5, .75], pair_gap=.1, pair_ambiguity_gap=.05)
    with ReportSource(archive) as z:
        read = lambda n: json.loads(z.read(n))
        reference = read('p0r_summary.json'); split = read('split.json')
        if reference.get('status') != 'REBUILD_AND_TRAIN_COMPLETED' or reference['arguments'].get('smoke'):
            raise ValueError('P0-R reference must be a completed full run')
        if meta != read('p0r_train.pt.manifest.json'): raise ValueError('Manifest differs from P0-R report')
        if signature['cache_sha256'] != meta['cache_sha256'] or signature['gt_sha256'] != meta['gt_sha256']:
            raise ValueError('Cache/GT hash differs from P0-R')
        for p in ['TileStitchingNet/stage2.py', 'TileStitchingNet/stage1.py']:
            if signature['source_hashes'][p] != reference['feature_source']['source_hashes'][p]:
                raise ValueError('Frozen model/evaluator source changed: '+p)
        for family in ['pointwise', 'local']:
            if hashlib.sha256(z.read(f'{family}_best.pth')).hexdigest() != read(f'{family}_completed.json')['checkpoint_sha256']:
                raise ValueError('Best checkpoint hash mismatch')
        entry = Path(args.p0r_entrypoint) if args.p0r_entrypoint else ROOT/'tools/kaggle_p0r_train.py'
        entry_sha = audit.runner.sha(entry) if entry.is_file() else None
        report['p0r_entrypoint_provenance'] = dict(provided=str(entry), sha256=entry_sha,
            expected_sha256=reference['feature_source']['source_hashes']['tools/kaggle_p0r_train.py'],
            verified=entry_sha==reference['feature_source']['source_hashes']['tools/kaggle_p0r_train.py'])
        signature['p0r_entrypoint_sha256'] = entry_sha
        config_path = out/'run_config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != signature:
            raise ValueError('Resume inputs/code/configuration changed; use a fresh output directory')
        audit.write(config_path, signature)
        sources = out/'sources'; sources.mkdir(exist_ok=True)
        for p in source_paths:
            dest = sources/p; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes((ROOT/p).read_bytes())
        if entry.is_file(): (sources/'provided_p0r_entrypoint.py').write_bytes(entry.read_bytes())
        samples = OfflineDatasetIO.load_samples(str(cache))
        by_id = {s.image_id:s for s in samples}
        expected = {r['image_id']:r['frozen'] for r in reference['train_refresh']}
        if len(by_id) != len(samples) or set(by_id) != set(expected): raise ValueError('Image IDs differ from P0-R')
        if set(split['train_ids']) & set(split['val_ids']): raise ValueError('Train/validation overlap')
        if set(split['train_ids']) | set(split['val_ids']) != set(by_id): raise ValueError('Split incomplete')
        for s in samples:
            if frozen_fields(s) != expected[s.image_id]: raise ValueError('Frozen candidate fields differ: '+s.image_id)
            if s.node_features.shape != (len(s.scores), 76) or not np.isfinite(s.node_features).all():
                raise ValueError('Invalid P0-R features')
            if np.any(s.boxes_global[:, 2:] <= s.boxes_global[:, :2]): raise ValueError('Nonpositive candidate size')
        selected_ids = split['val_ids'][:4] if args.smoke else split['val_ids']
        samples = [by_id[i] for i in selected_ids]; del by_id
        audit.write(out/'validation_ids.json', selected_ids)
        report['input'] = dict(validation_images=len(samples), candidates=sum(len(s.scores) for s in samples),
                              cache_sha256=signature['cache_sha256'], split_unchanged=True, test_evaluated=False)
        data = json.loads(gt_path.read_text(encoding='utf-8'))
        cfg = QualityConfig(); labeler = CandidateLabels(data, QualityTargets(gt_path, cfg))
        labels = [labeler(s) for s in samples]
        evaluator = OfficialTinyEvaluator(gt_path)
        tracer = OfficialMatchTrace(data)
        base_scores = [s.scores for s in samples]
        baseline = [nms_trace(s.boxes_global, s.scores) for s in samples]
        base_keep, base_edges = [r[0] for r in baseline], [r[1] for r in baseline]
        base_metrics = evaluate(evaluator, samples, base_keep, base_scores)
        if not args.smoke: check_metrics(base_metrics, read('validation_metrics.json')['Greedy-NMS'], 'baseline')
        base_trace = tracer(samples, base_keep, base_scores)
        compressed_json(out/'baseline_official_matches.json.gz', base_trace)
        compressed_json(out/'baseline_nms_trace.json.gz', {s.image_id:dict(keep=k.tolist(), edges=e.tolist()) for s,k,e in zip(samples,base_keep,base_edges)})
        audit.write(out/'baseline_metrics.json', base_metrics)
        pairs = {}; pair_rows = {}
        for threshold in [.5, .75]:
            rows = [reliable_pairs(s, lab, edges, threshold) for s,lab,edges in zip(samples,labels,base_edges)]
            pairs[str(threshold)] = [r[0] for r in rows]
            pair_rows[str(threshold)] = {s.image_id:dict(pairs=r[0], exclusions=r[1]) for s,r in zip(samples,rows)}
        compressed_json(out/'reliable_conflict_pairs.json.gz', pair_rows)
        graph = [graph_coverage(s, edges, cfg) for s,edges in zip(samples,base_edges)]
        compressed_json(out/'baseline_graph_edges.json.gz', {s.image_id:g for s,g in zip(samples,graph)})
        graph_summary = {'actual_baseline_suppression':aggregate_coverage(graph)}
        for t, pp in pairs.items():
            chosen = [{(p['winner'], p['suppressed']) for p in ps} for ps in pp]
            graph_summary['reliable_pairs_iou'+t] = aggregate_coverage([[r for r in rr if (r['winner'],r['suppressed']) in cc] for rr,cc in zip(graph,chosen)])
        audit.write(out/'graph_coverage.json', graph_summary)
        print(f'P1 preflight passed: {len(samples)} validation images; no training/test evaluation.', flush=True)
        all_results = []
        for family in ['pointwise', 'local']:
            for tag in signature['snapshots']:
                filename = f'{family}_best.pth' if tag=='best' else f'{family}_snapshots/{tag}.pth'
                payload = torch.load(io.BytesIO(z.read(filename)), map_location='cpu', weights_only=True)
                qc = QualityConfig(**{**payload['config'], 'device':'cpu'})
                if payload['local'] != (family=='local'): raise ValueError('Wrong checkpoint model family')
                for name in ['positive_iou','background_iou','ambiguity_gap','ignore_iod']:
                    if getattr(qc,name) != getattr(cfg,name): raise ValueError('Checkpoint label policy differs: '+name)
                model = QualityRescorer(payload['feature_dim'], qc, payload['local']).eval()
                model.load_state_dict(payload['model'], strict=True)
                if qc.neighbors != cfg.neighbors or qc.min_radius != cfg.min_radius:
                    raise ValueError('Checkpoint graph differs from diagnostic graph')
                deltas, scores1 = [], []
                with torch.inference_mode():
                    for s in samples:
                        result = model(prepare(s, qc, model.local))
                        delta = result['delta'].numpy(); sc = result['scores'].numpy()
                        if not np.isfinite(delta).all(): raise ValueError('Nonfinite residual')
                        np.testing.assert_array_equal(scaled_scores(s.scores, delta, 1), sc)
                        deltas.append(delta); scores1.append(sc)
                profile = f'{family}_{tag}'
                # Candidate-level arrays are compact; no large input cache is copied.
                np.savez_compressed(out/(profile+'_scores.npz'), image_ids=np.asarray(selected_ids),
                    offsets=np.cumsum([0]+[len(s.scores) for s in samples]),
                    delta=np.concatenate(deltas), base=np.concatenate(base_scores),
                    full_score=np.concatenate(scores1), quality=np.concatenate([l['quality'] for l in labels]),
                    reason=np.concatenate([l['reason'] for l in labels]))
                drift = score_diagnostics(samples, labels, deltas, scores1, base_edges)
                audit.write(out/(profile+'_score_drift.json'), drift)
                for alpha in ALPHAS:
                    sc = [scaled_scores(s.scores, d, alpha) for s,d in zip(samples,deltas)]
                    traced = [nms_trace(s.boxes_global, a) for s,a in zip(samples,sc)] if alpha else baseline
                    for mode in ['before_nms', 'after_nms']:
                        key = f'{profile}_a{alpha}_{mode}'
                        done = out/(key+'.json')
                        if done.is_file():
                            cached = json.loads(done.read_text())
                            if all((out/p).is_file() and audit.runner.sha(out/p)==h for p,h in cached.get('artifact_hashes',{}).items()):
                                all_results.append(cached); print('Reused', key, flush=True); continue
                        keep = [r[0] for r in traced] if mode=='before_nms' else base_keep
                        edges = [r[1] for r in traced] if mode=='before_nms' else base_edges
                        metrics = base_metrics if alpha==0 else evaluate(evaluator, samples, keep, sc)
                        if not args.smoke and alpha==1:
                            if tag=='best': check_metrics(metrics, read('validation_metrics.json')[family+'/'+mode], key)
                            elif mode=='before_nms': check_metrics(metrics, payload['validation'], key)
                        record = dict(profile=profile, checkpoint_file=filename, checkpoint_sha256=hashlib.sha256(z.read(filename)).hexdigest(),
                            epoch=payload['epoch'], alpha=alpha, mode=mode, metrics=metrics,
                            pair_results={t:pair_results(samples, ps, sc) for t,ps in pairs.items()}, artifact_hashes={})
                        if alpha==1:
                            mt = tracer(samples, keep, sc)
                            compared = compare_traces(samples, base_trace, mt, base_edges, edges)
                            record['matching_and_budget_summary'] = compared['summary']
                            artifacts = {key+'_matches.json.gz':mt, key+'_gt_budget_events.json.gz':compared['per_image'],
                                key+'_nms_trace.json.gz':{s.image_id:dict(keep=k.tolist(), edges=e.tolist()) for s,k,e in zip(samples,keep,edges)}}
                            for path, contents in artifacts.items():
                                compressed_json(out/path, contents); record['artifact_hashes'][path] = audit.runner.sha(out/path)
                        audit.write(done, record); all_results.append(record)
                        print(f'{key}: Tiny_AP50={metrics["Tiny_AP50"]:.4f}, AP={metrics["AP"]:.4f}', flush=True)
        audit.write(out/'validation_alpha_sweep.json', all_results)
        # Selection is descriptive validation-only. No test inference or retraining follows.
        selected = {}
        for family in ['pointwise', 'local']:
            rr = [r for r in all_results if r['profile'].startswith(family+'_')]
            best = max(rr, key=lambda r:r['metrics']['Tiny_AP50'])
            selected[family] = {k:best[k] for k in ['profile','epoch','alpha','mode','metrics']}
        report['validation_selection'] = selected
        report['baseline_metrics'] = base_metrics
        report['input_files_unchanged'] = audit.runner.sha(cache)==signature['cache_sha256'] and z.fingerprint()==signature['reference_sha256']
        if not report['input_files_unchanged']: raise AssertionError('Input changed during diagnosis')
        report['status'] = 'P1_SMOKE_COMPLETED' if args.smoke else 'P1_DIAGNOSIS_COMPLETED'
        report['interpretation_notes'] = [
            'Main AP always uses official maxDets=200. Unlimited-survivor matches only diagnose the budget effect.',
            'GT transitions are unique within each image/area/IoU. Different area/IoU scopes and before-budget events must not be added together.',
            'GT event context associates a matched candidate with another run suppression/budget state; it is not a causal oracle.',
            'Training ignore reasons are exclusive for summaries; conservative pair exclusions are separate from official dtIgnore.',
            'Pair correctness uses strict score margins; ties are separately reported, not treated as strict wins.',
            'Graph tiny/non-tiny groups use sqrt(candidate area)<=20; pair GT groups use sqrt(GT area)<=20.',
            'Default profiles: validation-selected best, epoch6 and epoch18. Alpha0 is the shared original-score baseline.',
            'P1 is validation-only exploratory diagnosis, not a new unbiased test result. Latency is not benchmarked.']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['train-cache','train-gt','p0r-report']: p.add_argument('--'+name, required=True)
    p.add_argument('--manifest'); p.add_argument('--p0r-entrypoint')
    p.add_argument('--output', default='/kaggle/working/p1_full')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    if args.threads<1: p.error('threads must be positive')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()) and not args.resume: p.error('Use a fresh output directory or --resume')
    out.mkdir(parents=True, exist_ok=True)
    report = dict(arguments=vars(args), status='RUNNING'); failed = False
    try: execute(args, out, report)
    except Exception:
        failed = True; report['status']='ERROR'; report['error']=traceback.format_exc(); print(report['error'], flush=True)
    finally:
        audit.write(out/'p1_summary.json', report)
        with zipfile.ZipFile(out/'p1_reports.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.rglob('*')):
                if f.is_file() and f.suffix in {'.json','.gz','.npz','.py'}: z.write(f, f.relative_to(out))
        print('Status:', report['status'], '\nSend back:', out/'p1_reports.zip', flush=True)
    if failed: raise SystemExit(1)


if __name__=='__main__': main()
