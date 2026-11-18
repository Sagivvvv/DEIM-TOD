"""Multi-seed validation/freeze and separate frozen TinyPerson test entry points."""
import argparse
from dataclasses import replace
from contextlib import ExitStack
import gzip
import shutil
import platform
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import traceback
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
import numpy as np
import torch
import kaggle_audit_p0 as audit
from kaggle_diagnose_p1 import check_metrics, compressed_json, evaluate
from kaggle_diagnose_p4a import P4Reference, sha_file, sha_bytes
from kaggle_train_p2 import ValidationDiagnostics
from TileStitchingNet.datasetBuilder import nms_np
from TileStitchingNet.p1 import OfficialMatchTrace, hits, scaled_scores
from TileStitchingNet.p2 import atomic_json
from TileStitchingNet.p3 import pair_tensors, PairConfig
from TileStitchingNet.p4 import load_p4_checkpoint, train_roi
from TileStitchingNet.p5 import (METHODS, SEEDS, PROTOCOL, summarize, choose_family,
    ap_input, paired_intervals, sampling_group)
from TileStitchingNet.pipeline_deim_tilestitching import OfflineDatasetIO
from TileStitchingNet.stage1 import OfficialTinyEvaluator
from TileStitchingNet.stage2 import QualityTargets, QualityConfig, prepare
from TileStitchingNet.trainEval import run_greedy_nms, run_soft_nms, run_wbf, run_sahi_nmm_ios

BASELINES = {
    'Greedy-NMS': lambda b,s: run_greedy_nms(b,s,.5,score_thresh=0),
    'Soft-NMS': lambda b,s: run_soft_nms(b,s,sigma=.5,score_thresh=0),
    'WBF': lambda b,s: run_wbf(b,s,.55,score_thresh=0),
    'SAHI-NMM': lambda b,s: run_sahi_nmm_ios(b,s,.5,score_thresh=0)}
ANCHOR = ROOT/'P5_REFERENCE.json'


def reference_hashes(ref):
    names = ['run_config.json', 'p4_summary.json', 'split.json', 'train_pairs.json.gz',
             'p0r_traditional_reference.json']
    for family in METHODS:
        names.extend([family+'/history.json', ref.summary['results'][family]['best_checkpoint'].replace(chr(92), '/')])
    return {n:sha_bytes(ref.read(n)) for n in names}


def environment(anchor, threads):
    sources = {p:sha_file(ROOT/p) for p in anchor['source_hashes']}
    if sources != anchor['source_hashes']:
        raise ValueError('Source hashes differ from the P5 upload')
    return dict(sources=sources, threads=threads, python=platform.python_version(),
                packages={p:version(p) for p in ('numpy','scipy','torch','torchvision','pycocotools')})


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    atomic_json(audit.clean(data), path)


def lock(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f'Configuration or inputs changed. Use a fresh output folder: {path}')
    write(path, value)


def verified_samples(cache, gt, anchor):
    manifest = Path(str(cache)+'.manifest.json')
    for filename, key in [(cache,'cache_sha256'), (gt,'gt_sha256'), (manifest,'manifest_sha256')]:
        if sha_file(filename) != anchor[key]:
            raise ValueError('Frozen input hash mismatch: '+str(filename))
    meta = read(manifest)
    if (meta.get('smoke') or meta['schema'] != 'p0r-roi-v1' or not meta['candidate_fields_unchanged']
            or any(meta[k] != anchor[k] for k in ('cache_sha256', 'gt_sha256'))):
        raise ValueError('Full P0-R cache required')
    images = {str(im['id']):im for im in read(gt)['images']}
    samples = OfflineDatasetIO.load_samples(str(cache))
    ids = [s.image_id for s in samples]
    if len(ids) != anchor['images'] or len(ids) != len(set(ids)) or not set(ids) <= set(images):
        raise ValueError('Cache and annotated image sets differ')
    return samples


def model_prediction(model, sample):
    with torch.inference_mode():
        data = prepare(sample, model.config, False)
        delta = model(data)['delta'].numpy()
        scores = scaled_scores(sample.scores, delta, PROTOCOL['alpha'])
    keep = nms_np(sample.boxes_global, scores, iou_thresh=.5)
    return sample.boxes_global[keep], scores[keep], keep


def measure(name, samples, gt, annotation, folder, ids, model=None):
    """Evaluate uncapped outputs with official maxDets and save global boxes and TP traces."""
    def predict(s):
        if model is not None:
            return model_prediction(model, s)
        b, sc = BASELINES[name](s.boxes_global, s.scores)
        return b, sc, None
    warmup = [s for s in samples if len(s.scores)][:5]
    for s in warmup:
        predict(s)
    predictions, boxes, scores, times = {}, [], [], []
    for s in samples:
        start = perf_counter()
        b, sc, keep = predict(s)
        top = np.argsort(-sc, kind='stable')[:PROTOCOL['max_dets']]
        _ = b[top], sc[top]
        times.append(1000*(perf_counter()-start))
        boxes.append(b)
        scores.append(sc)
        predictions[s.image_id] = dict(boxes_xyxy=b.tolist(), scores=sc.tolist(), ms=times[-1])
        if keep is not None:
            predictions[s.image_id]['input_candidate_indices'] = keep.tolist()
    metrics = OfficialTinyEvaluator(gt)(samples, boxes, scores, times)
    virtual = [SimpleNamespace(image_id=s.image_id, boxes_global=b) for s,b in zip(samples,boxes)]
    trace = OfficialMatchTrace(annotation)(virtual, [np.arange(len(sc)) for sc in scores], scores)
    compressed_json(folder/(name+'_predictions.json.gz'), predictions)
    compressed_json(folder/(name+'_matches.json.gz'), trace)
    score_by_id = {s.image_id: sc for s, sc in zip(samples, scores)}
    # Small sets for comparisons, sufficient statistics for bootstrap.
    summaries = {iid: {k: (hits(row, 200), hits(row)) for k,row in entries.items()}
                 for iid, entries in trace.items()}
    timing = dict(mean=float(np.mean(times)),median=float(np.median(times)),p95=float(np.percentile(times,95)),
                  warmup_images=len(warmup),timed_images=len(times),passes=1,threads=torch.get_num_threads())
    return metrics, ap_input(trace, score_by_id, ids), summaries, timing


def hit_changes(base, new):
    result = {}
    for key in ('tiny@0.5', 'tiny@0.75', 'all@0.5', 'all@0.75'):
        gains, losses, old_budget, new_budget = [], [], 0, 0
        before_gains, before_losses = [], []
        for iid in base:
            a, full_a = base[iid].get(key,(set(),set()))
            b, full_b = new[iid].get(key,(set(),set()))
            old_budget += len(full_a-a)
            new_budget += len(full_b-b)
            gains.extend((iid, g) for g in sorted(b-a))
            losses.extend((iid, g) for g in sorted(a-b))
            before_gains.extend((iid,g) for g in sorted(full_b-full_a))
            before_losses.extend((iid,g) for g in sorted(full_a-full_b))
        result[key] = dict(gained=len(gains), lost=len(losses), gains=gains, losses=losses,
                           old_budget_loss=old_budget, new_budget_loss=new_budget,
                           gains_before_top200=before_gains, losses_before_top200=before_losses)
    return result


def compare(out, samples, gt, seeds, models, baseline=None, reference_metrics=None, smoke=False):
    folder = out/'predictions'
    folder.mkdir(exist_ok=True)
    annotation = read(gt)
    ids = sorted([s.image_id for s in samples], key=int)
    metrics, ap_rows, matches, timing = {}, {}, {}, {}
    for name in [*BASELINES, *models]:
        model = load_p4_checkpoint(models[name])[0] if name in models else None
        m, ap_rows[name], matches[name], timing[name] = measure(name, samples, gt, annotation, folder, ids, model)
        if reference_metrics and name in reference_metrics:
            check_metrics(m, reference_metrics[name], 'P5 reproduction '+name)
        metrics[name] = m
        print(f'{name}: Tiny AP50 {m["Tiny_AP50"]:.4f}, AP {m["AP"]:.4f}', flush=True)
    if baseline is None:
        baseline = max(BASELINES, key=lambda k: metrics[k]['Tiny_AP50'])
    groups = {str(im['id']):sampling_group(im) for im in annotation['images']}
    write(out/'sampling_groups.json', {iid:groups[iid] for iid in ids})
    paired = paired_intervals(ap_rows, metrics, ids, groups, seeds, baseline, 12 if smoke else 1000)
    write(out/'paired_comparison.json', paired)
    seed_summary = summarize(metrics, seeds, baseline)
    best_observed = max(BASELINES, key=lambda k: metrics[k]['Tiny_AP50'])
    seed_summary['strongest_traditional_on_this_split'] = best_observed
    seed_summary['mean_tiny50_minus_strongest_on_this_split'] = {
        f: seed_summary['families'][f]['Tiny_AP50']['mean']-metrics[best_observed]['Tiny_AP50'] for f in METHODS}
    changes = {}
    for seed in seeds:
        c, a = f'C_full_{seed}', f'A_full_{seed}'
        for x,y in [(c,a),(a,baseline),(c,baseline)]:
            changes[x+'_minus_'+y] = hit_changes(matches[y], matches[x])
    compressed_json(out/'gt_changes_and_failure_cases.json.gz', changes)
    write(out/'metrics.json',metrics)
    write(out/'latency_ms.json',timing)
    write(out/'seed_summary.json', seed_summary)
    return seed_summary


def validate(args, out, anchor):
    cache, gt = Path(args.train_cache), Path(args.train_gt)
    samples = verified_samples(cache, gt, anchor['train'])
    with P4Reference(args.p4_report) as ref:
        cfg = ref.config
        if reference_hashes(ref) != anchor['p4']:
            raise ValueError('P4 reference inputs changed')
        if cfg['config']['epochs'] != anchor['epochs'] or args.threads != cfg['threads']:
            raise ValueError('Use the P4 epoch budget and CPU thread count')
        for k in ('cache_sha256','manifest_sha256','gt_sha256'):
            if cfg[k] != anchor['train'][k]:
                raise ValueError('P4 and P0-R input pins differ')
        split = ref.json('split.json')
        by_id = {s.image_id:s for s in samples}
        tr, va = split['train_ids'], split['val_ids']
        if [len(tr),len(va)] != anchor['split_sizes']:
            raise ValueError('Unexpected split sizes')
        if set(tr)&set(va) or len(set(tr+va))!=len(tr+va) or set(tr+va)!=set(by_id):
            raise ValueError('Invalid original split')
        if args.smoke:
            tr, va = tr[:8], va[:4]
        training, validation = [[by_id[i] for i in ids] for ids in (tr,va)]
        groups = {str(im['id']):sampling_group(im) for im in read(gt)['images']}
        write(out/'split.json', dict(train_ids=tr, val_ids=va, split_seed=42, shared_sampling_groups=sorted({groups[i] for i in tr}&{groups[i] for i in va})))
        base = QualityConfig(**cfg['config'])
        if base.seed != 42 or base.rank_weight or base.device!='cpu' or cfg['pair_config']['alpha']!=PROTOCOL['alpha']:
            raise ValueError('Unexpected P4 configuration')
        pcfg = PairConfig(**cfg['pair_config'])
        pairs = ref.json('train_pairs.json.gz')
        if pairs['signature'] != ref.summary['pairs']['train']['signature']:
            raise ValueError('Frozen pair signature mismatch')
        base = replace(base, epochs=2) if args.smoke else base
        seeds = (43,) if args.smoke else SEEDS
        targets = QualityTargets(gt, base)
        batches = [dict(prepare(s,base,False,targets), **pair_tensors(pairs['pairs'][s.image_id])) for s in training]
        diag = ValidationDiagnostics(validation, gt, base)
        models, frozen_models, expected = {}, {}, {}
        if not args.smoke:
            expected.update(ref.json('p0r_traditional_reference.json'))
        weights = out/'weights'
        weights.mkdir(exist_ok=True)
        for seed in seeds:
            for family, weight in METHODS.items():
                name = f'{family}_{seed}'
                path = weights/(name+'.pth')
                if seed == 42 and not args.smoke:
                    reference = ref.summary['results'][family]
                    raw = ref.read(reference['best_checkpoint'].replace(chr(92),'/'))
                    if sha_bytes(raw)!=reference['best_checkpoint_sha256']:
                        raise ValueError('P4 checkpoint digest mismatch')
                    path.write_bytes(raw)
                    h = ref.json(family+'/history.json')
                    curve = [r['validation'][family]['metrics'] for r in h]
                    origin = 'P4'
                else:
                    def assess(model, epoch):
                        _,_,scores = diag.predict(model, PROTOCOL['alpha'])
                        keep = [nms_np(s.boxes_global, sc, iou_thresh=.5) for s,sc in zip(validation,scores)]
                        metrics = evaluate(diag.evaluator, validation, keep, scores)
                        if epoch==0:
                            check_metrics(metrics, diag.base_metrics, name+' epoch0')
                        return {name:dict(alpha=.25,metrics=audit.clean(metrics))}
                    job = out/'training'/name
                    h,_ = train_roi(batches, replace(base,seed=seed), pcfg, weight,
                                    pairs['signature'],'full',job,assess)
                    curve = [r['validation'][name]['metrics'] for r in h]
                    path.write_bytes((job/(name+'_best.pth')).read_bytes())
                    origin = 'trained'
                epoch = max(range(len(curve)),key=lambda i:curve[i]['Tiny_AP50'])
                if len(curve)!=base.epochs+1:
                    raise ValueError('Incomplete checkpoint selection')
                _, payload = load_p4_checkpoint(path)
                if (payload['epoch'],payload['pair_weight'],payload['config']['seed'],payload['roi_mode'],payload['alpha']) != (epoch,weight,seed,'full',.25):
                    raise ValueError('Frozen metadata mismatch: '+name)
                models[name] = path
                frozen_models[name] = dict(path=path.relative_to(out).as_posix(), sha256=sha_file(path), best_epoch=epoch, origin=origin)
                expected[name] = curve[epoch]
                write(out/(name+'_validation_curve.json'),curve)
        summary = compare(out,validation,gt,seeds,models,reference_metrics=expected,smoke=args.smoke)
        family = None if args.smoke else choose_family(summary)
        freeze = dict(schema='p5-freeze-v1', smoke=args.smoke, protocol=PROTOCOL, seeds=list(seeds),
                      selected_family=family, traditional_reference=summary['reference'], models=frozen_models,
                      run_sha256=sha_file(out/'run_config.json'), seed_summary_sha256=sha_file(out/'seed_summary.json'),
                      test_inputs=anchor['test'])
        write(out/'frozen_config.json',freeze)
        return dict(status='P5_SMOKE_COMPLETED' if args.smoke else 'P5_VALIDATION_FROZEN',
                    new_training_runs=2 if args.smoke else 4, training_images=len(tr),validation_images=len(va),
                    selected_family=family, test_evaluated=False)


def test_frozen(args, out, anchor, env):
    source = Path(args.frozen)
    retained = out/'validation'
    retained.mkdir(exist_ok=True)
    with ExitStack() as stack:
        if source.is_dir():
            get = lambda member: (source/member).read_bytes()
        else:
            get = stack.enter_context(zipfile.ZipFile(source)).read
        def retain(name):
            raw = get(name)
            target = retained/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(raw)
            return raw
        freeze = json.loads(retain('frozen_config.json'))
        run = json.loads(retain('run_config.json'))
        summary = json.loads(retain('seed_summary.json'))
        status = json.loads(retain('status.json'))
        if (freeze['schema'] != 'p5-freeze-v1' or freeze['smoke'] or run['smoke']
                or status['status'] != 'P5_VALIDATION_FROZEN' or run['phase'] != 'validate'
                or freeze['run_sha256'] != sha_file(retained/'run_config.json')
                or freeze['seed_summary_sha256'] != sha_file(retained/'seed_summary.json')
                or run['environment'] != env or run['reference'] != anchor or run['protocol'] != PROTOCOL
                or freeze['test_inputs'] != anchor['test'] or freeze['protocol'] != PROTOCOL
                or freeze['seeds'] != list(SEEDS) or summary['seeds'] != list(SEEDS)):
            raise ValueError('A matching full P5 validation freeze is required')
        val_metrics = json.loads(retain('metrics.json'))
        ref = max(BASELINES,key=lambda name:val_metrics[name]['Tiny_AP50'])
        rebuilt = summarize(val_metrics,SEEDS,ref)
        if (any(summary[k]!=v for k,v in rebuilt.items()) or ref!=freeze['traditional_reference']
                or choose_family(rebuilt)!=freeze['selected_family']):
            raise ValueError('Frozen validation decision failed replay')
        expected = {f'{family}_{seed}' for family in METHODS for seed in SEEDS}
        if set(freeze['models']) != expected:
            raise ValueError('Incomplete seed panel')
        models = {}
        for family, weight in METHODS.items():
            for seed in SEEDS:
                key = f'{family}_{seed}'
                item = freeze['models'][key]
                name = 'weights/'+key+'.pth'
                if item['path']!=name or sha_bytes(retain(name))!=item['sha256']:
                    raise ValueError('Frozen checkpoint hash mismatch: '+key)
                curve = json.loads(retain(key+'_validation_curve.json'))
                best = max(range(len(curve)), key=lambda i:curve[i]['Tiny_AP50'])
                _, ck = load_p4_checkpoint(retained/name)
                if (len(curve)!=anchor['epochs']+1 or item['best_epoch']!=best or ck['epoch']!=best
                        or ck['config']['seed']!=seed or ck['roi_mode']!='full' or ck['alpha']!=PROTOCOL['alpha']
                        or ck['pair_weight']!=weight or curve[best]['Tiny_AP50']!=val_metrics[key]['Tiny_AP50']):
                    raise ValueError('Frozen model disagrees with validation selection: '+key)
                models[key] = retained/name
        for name in ('split.json','sampling_groups.json','paired_comparison.json','latency_ms.json'):
            retain(name)
        # Kaggle sometimes expands gzip members. Restore their portable names
        # and retain validation predictions/GT events in the single final report.
        details = ['gt_changes_and_failure_cases.json.gz']
        details.extend('predictions/'+name+'_'+kind+'.json.gz'
                       for name in [*BASELINES,*models] for kind in ('predictions','matches'))
        for name in details:
            dest = retained/name
            dest.parent.mkdir(exist_ok=True,parents=True)
            if source.is_dir() and not (source/name).is_file():
                with (source/name[:-3]).open('rb') as inp, gzip.open(dest,'wb') as sink:
                    shutil.copyfileobj(inp,sink)
            else:
                dest.write_bytes(get(name))
    # Only now open the labeled test set.
    samples = verified_samples(args.test_cache,args.test_gt,anchor['test'])
    result = compare(out,samples,args.test_gt,SEEDS,models,baseline=ref)
    family = freeze['selected_family']
    intervals = read(out/'paired_comparison.json')
    table = read(out/'metrics.json')
    delta = result['families'][family]['Tiny_AP50_minus_reference']
    band = intervals['percentile_95'][family[0]+'_minus_reference']
    per_baseline = {}
    for name in BASELINES:
        differences = [table[f'{family}_{s}']['Tiny_AP50']-table[name]['Tiny_AP50'] for s in SEEDS]
        per_baseline[name] = dict(seed_deltas=differences,mean_delta=float(np.mean(differences)),
                                  all_three_positive=all(x>0 for x in differences),
                                  mean_extra_ms=float(np.mean([table[f'{family}_{s}']['Latency_ms'] for s in SEEDS]))-table[name]['Latency_ms'])
    final = dict(status='P5_TEST_COMPLETED', test_images=len(samples), selected_on_validation=family,
                 traditional_reference=ref, seed_summary=result, metrics=table,
                 selected_minus_reference_interval=band, all_three_test_gains_positive=all(v>0 for v in delta['values']),
                 selected_vs_each_traditional=per_baseline,
                 positive_lower_interval_endpoint=bool(band and band[0]>0),
                 latency='mean ms per original image, CPU; cached inputs through top200; excludes detector, ROI extraction, I/O and evaluation',
                 timing_ms_by_method={k:v['Latency_ms'] for k,v in table.items()},
                 freeze_sha256=sha_file(retained/'frozen_config.json'),
                 interpretation='All seeds reported; fixed weights, paired image/video sampling. Test split has been inspected in earlier stages.')
    write(out/'final_report.json',final)
    return dict(status=final['status'], test_images=len(samples), selected_family=family,
                test_evaluated=True, traditional_reference=ref)


def package(out, filename):
    temp = out/(filename+'.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for path in sorted(out.rglob('*')):
            name = path.relative_to(out)
            if (path.is_file() and name.parts[0]!='training'
                    and path.suffix not in ('.zip','.tmp') and path.name!='error.json'):
                archive.write(path,name.as_posix(),compress_type=zipfile.ZIP_STORED if path.suffix=='.gz' else zipfile.ZIP_DEFLATED)
    os.replace(temp,out/filename)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('validate','test'))
    parser.add_argument('--output', type=Path, required=True)
    for flag in ('train-cache','train-gt','test-cache','test-gt','p4-report','frozen'):
        parser.add_argument('--'+flag,type=Path)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args(argv)
    out = args.output
    out.mkdir(parents=True,exist_ok=True)
    try:
        anchor = read(ANCHOR)
        if args.threads<1:
            raise ValueError('Positive CPU thread count required')
        torch.set_num_threads(args.threads)
        env = environment(anchor,args.threads)
        contract = dict(phase=args.phase,smoke=args.smoke,environment=env,reference=anchor,protocol=PROTOCOL)
        if args.phase=='test':
            if not all((args.test_gt,args.test_cache,args.frozen)) or args.smoke:
                raise ValueError('Test requires --frozen, --test-cache and --test-gt')
            if args.frozen.is_dir():
                contract['freeze_sha256'] = sha_file(args.frozen/'frozen_config.json')
            else:
                with zipfile.ZipFile(args.frozen) as archive:
                    contract['freeze_sha256'] = sha_bytes(archive.read('frozen_config.json'))
        elif not args.train_gt or not args.train_cache:
            raise ValueError('Validation requires --train-gt and --train-cache')
        lock(out/'run_config.json',contract)
        if args.phase=='test':
            status = test_frozen(args,out,anchor,env)
            filename = 'p5_reports.zip'
        else:
            status = validate(args,out,anchor)
            filename = 'p5_smoke_reports.zip' if args.smoke else 'p5_validation_reports.zip'
        write(out/'status.json',status)
        package(out,filename)
        print(status['status'],out/filename,flush=True)
    except Exception:
        error = dict(status='ERROR',phase=args.phase,smoke=args.smoke,traceback=traceback.format_exc())
        write(out/'error.json',error)
        with zipfile.ZipFile(out/'p5_error.zip','w',zipfile.ZIP_DEFLATED) as report:
            for name in ('error.json','run_config.json'):
                if (out/name).is_file():
                    report.write(out/name,name)
        print('Send back:',out/'p5_error.zip',flush=True)
        raise


if __name__=='__main__':
    main()
