"""Read-only P0 audit. No training, extraction, cache rewriting, or parameter search."""
import argparse
import hashlib
import json
import platform
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
import numpy as np
import torch
import kaggle_train_stage2 as runner
from TileStitchingNet.stage1 import OfficialTinyEvaluator, split_train_validation
from TileStitchingNet.stage2 import QualityConfig, QualityRescorer, QualityPredictor, load_quality_checkpoint
from TileStitchingNet.datasetBuilder import nms_np


def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def write(path,value):
    path.write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def array_hash(array):
    if array is None: return None
    a=np.ascontiguousarray(array)
    h=hashlib.sha256(str((a.shape,a.dtype.str)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


FIELDS=('boxes_global','scores','class_ids','patch_ids','patch_rects','node_features','roi_features')


def fingerprints(samples):
    rows={}
    for s in samples:
        # Sorting row hashes distinguishes candidate reordering from content changes.
        row_hashes=[]
        for i in range(len(s.scores)):
            h=hashlib.sha256()
            for key in FIELDS:
                a=getattr(s,key,None)
                h.update(key.encode())
                h.update(str(array_hash(a[i]) if a is not None else None).encode())
            row_hashes.append(h.hexdigest())
        rows[s.image_id]=dict(count=len(s.scores),size=list(s.img_size),
            ordered={key:array_hash(getattr(s,key,None)) for key in FIELDS},
            row_multiset=hashlib.sha256(''.join(sorted(row_hashes)).encode()).hexdigest(),
            tile_geometry=array_hash(getattr(s,'tile_rects_all',None)))
    return rows


def compare_fingerprints(a,b):
    common=sorted(a.keys() & b.keys())
    return dict(only_left=sorted(a.keys()-b.keys()),only_right=sorted(b.keys()-a.keys()),
        ordered_changed=[i for i in common if a[i]['ordered']!=b[i]['ordered']],
        candidate_content_changed=[i for i in common if a[i]['row_multiset']!=b[i]['row_multiset']],
        geometry_changed=[i for i in common if a[i]['tile_geometry']!=b[i]['tile_geometry']])


def describe(samples):
    def summary(a):
        a=np.asarray(a)
        return dict(n=int(a.size),mean=float(a.mean()),std=float(a.std()),
                    quantiles=np.quantile(a,[0,.1,.5,.9,1]).tolist()) if a.size else dict(n=0)
    nonempty=[s for s in samples if len(s.scores)]
    scores=np.concatenate([s.scores for s in nonempty]) if nonempty else []
    wh=np.concatenate([s.boxes_global[:,2:]-s.boxes_global[:,:2] for s in nonempty]) if nonempty else np.empty((0,2))
    roi=[s.node_features[:,12:] for s in nonempty if s.node_features.shape[1]>12]
    features=np.concatenate(roi) if roi else np.empty((0,0))
    return dict(images=len(samples),candidates=summary([len(s.scores) for s in samples]),scores=summary(scores),
        width=summary(wh[:,0]),height=summary(wh[:,1]),roi_norm=summary(np.linalg.norm(features,axis=1)),
        roi_channel_mean=features.mean(0).tolist() if len(features) else [],
        roi_channel_std=features.std(0).tolist() if len(features) else [])


def evaluate(samples,predictor,evaluator,out,prefix):
    boxes,scores,times=[],[],[]
    h=hashlib.sha256()
    for s in samples:
        b,sc,t=predictor.predict_sample(s)
        boxes.append(b); scores.append(sc); times.append(t)
        h.update(s.image_id.encode()); h.update(array_hash(b).encode()); h.update(array_hash(sc).encode())
    metrics=evaluator(samples,boxes,scores,times)
    write(out/(prefix+'.json'),dict(metrics=metrics,prediction_sha256=h.hexdigest()))
    print(prefix,clean(metrics),flush=True)
    return metrics


def metric_diff(current,old):
    return {k:current[k]-v for k,v in old.items() if k in current and isinstance(v,(float,int))
            and isinstance(current[k],(float,int)) and k!='Latency_ms'}


def execute(args,out,report):
    torch.set_num_threads(args.threads)
    report['environment']=dict(python=sys.version,torch=torch.__version__,platform=platform.platform(),threads=args.threads,device='cpu')
    tracked=['tools/kaggle_audit_p0.py','tools/kaggle_train_stage2.py','TileStitchingNet/stage1.py',
             'TileStitchingNet/stage2.py','TileStitchingNet/trainEval.py','TileStitchingNet/datasetBuilder.py',
             'TileStitchingNet/pipeline_deim_tilestitching.py','tools/huicv/evaluation/expand_cocofmt_eval.py']
    report['source_hashes']={p:runner.sha(ROOT/p) for p in tracked}
    samples_by_split={}; gt_paths={}; manifests={}
    for split,filename in [('train','tinyperson_p3_train.pt'),('test','tinyperson_p3_val.pt')]:
        cache=runner.resolve_file(getattr(args,split+'_cache'),filename,cache=True)
        gt=runner.resolve_file(getattr(args,split+'_gt'),f'tiny_set_{split}_all.json')
        raw=runner.OfflineDatasetIO.load_samples(str(cache))
        if len({s.image_id for s in raw})!=len(raw): raise ValueError('Duplicate image IDs')
        fp=fingerprints(raw)
        write(out/f'{split}_raw_fingerprints.json',fp)
        del raw
        # Explicit audit-only mode may inspect missing provenance; it never licenses training.
        ss,meta=runner.load_checked_cache(cache,gt,smoke=True)
        manifests[split]=meta
        effective=fingerprints(ss)
        write(out/f'{split}_effective_fingerprints.json',effective)
        report[split]=dict(cache=str(cache),cache_sha256=runner.sha(cache),gt=str(gt),gt_sha256=runner.sha(gt),
                           manifest=meta,repair_changes=compare_fingerprints(fp,effective))
        if meta is None: report['warnings'].append(f'{split}: missing manifest; extraction provenance unverified')
        ref=getattr(args,'reference_'+split+'_cache')
        if ref:
            ref_samples=runner.OfflineDatasetIO.load_samples(ref)
            ref_fp=fingerprints(ref_samples)
            report[split]['historical_cache_comparison']=dict(path=ref,sha256=runner.sha(ref),differences=compare_fingerprints(ref_fp,fp))
            del ref_samples
        else: report['warnings'].append(f'{split}: no historical cache supplied; cross-stage candidate equality unverified')
        samples_by_split[split]=ss; gt_paths[split]=gt
    keys=('checkpoint_sha256','conf_thresh','intra_nms','roi_dim','roi_seed','amp','torch_version','torchvision_version','source_hashes')
    report['extraction_signature_mismatches']=[k for k in keys if any(manifests[s] is None or k not in manifests[s] for s in manifests)
        or manifests['train'][k]!=manifests['test'][k]]
    if report['extraction_signature_mismatches']: report['warnings'].append('Train/test extraction signatures incomplete or inconsistent')
    train,val,split_manifest=split_train_validation(samples_by_split['train'],gt_paths['train'])
    test=samples_by_split['test']
    write(out/'split.json',split_manifest)
    report['distributions']={name:describe(ss) for name,ss in [('train',train),('validation',val),('test',test)]}
    stage2=Path(args.stage2_dir) if args.stage2_dir else runner.resolve_file(None,'pointwise_best.pth').parent
    report['stage2_dir']=str(stage2)
    previous={}
    for filename in ['stage2_run.json','stage2_split.json','validation_metrics.json','test_metrics.json']:
        path=stage2/filename
        if path.is_file():
            previous[filename]=json.loads(path.read_text(encoding='utf-8'))
            write(out/('previous_'+filename),previous[filename])
        else: report['warnings'].append('Missing previous '+filename)
    oldsplit=previous.get('stage2_split.json')
    report['split_matches_previous']=None if oldsplit is None else all(set(oldsplit[k])==set(split_manifest[k]) for k in ('train_ids','val_ids'))
    if report['split_matches_previous'] is False: raise ValueError('Split differs from original stage-two run')
    oldrun=previous.get('stage2_run.json',{})
    report['manifest_changes_since_stage2']={}
    for split in ('train','test'):
        oldmeta=oldrun.get(split+'_cache_manifest')
        report['manifest_changes_since_stage2'][split]=None if oldmeta is None else [k for k in keys if manifests[split] is None or oldmeta.get(k)!=manifests[split].get(k)]
    report['source_changes_since_stage2']={k:dict(old=v,current=report['source_hashes'].get(k)) for k,v in oldrun.get('source_hashes',{}).items() if report['source_hashes'].get(k)!=v}
    if args.smoke: train,val,test=train[:8],val[:4],test[:4]
    report['evaluated_ids']={name:[s.image_id for s in ss] for name,ss in [('validation',val),('test',test)]}
    dims={s.node_features.shape[1]+2 for s in train+val+test if len(s.scores)}
    if len(dims)!=1: raise ValueError('Inconsistent feature dimensions')
    feature_dim=dims.pop()
    report['identity_checks']={}
    for local in (False,True):
        torch.manual_seed(42)
        model=QualityRescorer(feature_dim,QualityConfig(),local)
        for mode in ('before_nms','after_nms'):
            predictor=QualityPredictor(model,mode); mismatches=[]
            for s in val+test:
                b,sc,_=predictor.predict_sample(s); keep=nms_np(s.boxes_global,s.scores,.5)
                if not np.array_equal(b,s.boxes_global[keep]) or not np.allclose(sc,s.scores[keep],atol=1e-6,rtol=0): mismatches.append(s.image_id)
            report['identity_checks'][f'{local}/{mode}']=mismatches
    baselines={
        'Greedy-NMS':runner.Baseline(lambda b,s:runner.run_greedy_nms(b,s,.5,score_thresh=0)),
        'Soft-NMS':runner.Baseline(lambda b,s:runner.run_soft_nms(b,s,sigma=.5,score_thresh=0)),
        'WBF':runner.Baseline(lambda b,s:runner.run_wbf(b,s,.55,score_thresh=0)),
        'SAHI-NMM':runner.Baseline(lambda b,s:runner.run_sahi_nmm_ios(b,s,.5,score_thresh=0))}
    models={}
    for name in ('pointwise','local'):
        path=stage2/f'{name}_best.pth'
        if not path.is_file():
            report['warnings'].append('Missing checkpoint '+str(path)); continue
        payload=torch.load(path,map_location='cpu',weights_only=True)
        report[name+'_checkpoint']=dict(path=str(path),sha256=runner.sha(path),epoch=payload['epoch'],config=payload['config'],validation=payload['validation'])
        models[name]=load_quality_checkpoint(path)
    report['metrics']={}; report['metric_differences']={}; report['checkpoint_validation_differences']={}
    for name,ss,gt in [('validation',val,gt_paths['train']),('test',test,gt_paths['test'])]:
        evaluator=OfficialTinyEvaluator(gt,include_split=all(getattr(s,'tile_rects_all',None) is not None for s in ss))
        methods=dict(baselines)
        for model_name,model in models.items():
            for mode in ('before_nms','after_nms'): methods[model_name+'/'+mode]=QualityPredictor(model,mode)
        metrics={}
        for method,predictor in methods.items():
            metrics[method]=evaluate(ss,predictor,evaluator,out,name+'_'+method.replace('/','_'))
        report['metrics'][name]=metrics
        old=previous.get(name+'_metrics.json',{})
        if not args.smoke:
            report['metric_differences'][name]={k:metric_diff(v,old[k]) for k,v in metrics.items() if k in old}
            if name=='validation':
                for model_name in models:
                    report['checkpoint_validation_differences'][model_name]=metric_diff(metrics[model_name+'/before_nms'],report[model_name+'_checkpoint']['validation'])
    if args.reference_stage1_dir:
        ref=Path(args.reference_stage1_dir)/'stage1_metrics.json'
        old=json.loads(ref.read_text(encoding='utf-8')); write(out/'reference_stage1_metrics.json',old)
        mapping={'Greedy-NMS':'Greedy-NMS (IoU=0.50)','Soft-NMS':'Soft-NMS (Gaussian)','WBF':'WBF (IoU=0.55)','SAHI-NMM':'SAHI NMM (IoS=0.50)'}
        if not args.smoke: report['stage1_metric_differences']={k:metric_diff(report['metrics']['test'][k],old[v]) for k,v in mapping.items() if v in old}
    report['warnings'].append('Manifests do not cryptographically bind historical cache payload or ROI projection weights; recorded hashes establish identity from this audit forward.')
    report['cache_files_unchanged']={split:runner.sha(report[split]['cache'])==report[split]['cache_sha256'] for split in ('train','test')}
    inconsistent=any(report['identity_checks'].values()) or bool(report['extraction_signature_mismatches'])
    inconsistent |= any(bool(v) for v in report['manifest_changes_since_stage2'].values()) or not all(report['cache_files_unchanged'].values())
    for section in list(report['metric_differences'].values())+[report['checkpoint_validation_differences']]:
        if any(abs(v)>1e-4 for diffs in section.values() for k,v in diffs.items() if k in ('AP','AP25','AP50','AP75','Tiny_AP25','Tiny_AP50','Tiny_AP75')): inconsistent=True
    report['status']='SMOKE_ONLY' if args.smoke else ('REVIEW_MISMATCHES' if inconsistent else 'COMPLETED_REVIEW_PROVENANCE')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('train-cache','test-cache','train-gt','test-gt','stage2-dir','reference-stage1-dir','reference-train-cache','reference-test-cache'): p.add_argument('--'+key)
    p.add_argument('--output',default='/kaggle/working/p0_audit')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args(argv)
    if args.threads<1: p.error('threads must be positive')
    out=Path(args.output)
    if out.exists() and any(out.iterdir()): p.error('Output directory must be empty; choose a new run directory')
    out.mkdir(parents=True,exist_ok=True)
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),arguments=vars(args),warnings=[],status='RUNNING')
    failed=False
    try: execute(args,out,report)
    except Exception:
        failed=True; report['status']='ERROR'; report['error']=traceback.format_exc(); print(report['error'],flush=True)
    finally:
        write(out/'audit_summary.json',report)
        archive=out/'p0_reports.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.glob('*.json')): z.write(f,f.name)
        print('P0 status:',report['status'],'\nSend back:',archive,flush=True)
    if failed: raise SystemExit(1)


if __name__=='__main__': main()
