"""P0-R: preflight, fixed-candidate ROI refresh, pointwise/local training, zipped reports."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import traceback
import zipfile
import numpy as np
import torch
import torchvision
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'tools'))
import kaggle_audit_p0 as audit
from kaggle_audit_p0_sources import read_reference
from TileStitchingNet.p0r import SharedProjection,extract_roi,load_detector,resolve_image,frozen_fields,replace_roi
from TileStitchingNet.stage2 import QualityConfig,QualityPredictor,train_quality,evaluate_predictor


def refresh(samples,images,model,projection,args,folder,signature):
    folder.mkdir(parents=True,exist_ok=True)
    rows=[]; repeated=False
    for i,s in enumerate(samples):
        # Cache IDs are used only after integer validation to keep shard paths contained.
        name=str(int(s.image_id)); shard=folder/(name+'.npy'); meta=folder/(name+'.json')
        frozen=frozen_fields(s); image_path=images[s.image_id]
        expected=dict(signature=signature,candidates=frozen,image_sha256=audit.runner.sha(image_path))
        reused=False
        if shard.is_file() and meta.is_file():
            saved=json.loads(meta.read_text())
            if saved.get('input')!=expected: raise ValueError('Shard source changed; use a new output directory: '+name)
            if saved.get('input')==expected and saved.get('roi_sha256')==audit.runner.sha(shard):
                roi=np.load(shard,allow_pickle=False); reused=True
        if not reused:
            roi=extract_roi(s,image_path,model,projection,args.extract_device,args.batch_tiles)
            np.save(shard,roi,allow_pickle=False)
            audit.write(meta,dict(input=expected,roi_sha256=audit.runner.sha(shard)))
        repeat_error=None
        if len(s.scores) and not repeated:
            again=extract_roi(s,image_path,model,projection,args.extract_device,args.batch_tiles)
            repeat_error=float(np.abs(roi-again).max())
            if not np.allclose(roi,again,atol=1e-5,rtol=1e-4): raise ValueError('ROI repeatability check failed')
            repeated=True
        replace_roi(s,roi)
        if frozen_fields(s)!=frozen: raise AssertionError('Frozen fields changed')
        rows.append(dict(image_id=s.image_id,candidate_count=len(s.scores),frozen=frozen,original_image=str(image_path),
                         image_sha256=expected['image_sha256'],roi_sha256=audit.runner.sha(shard),reused=reused,repeat_max_abs=repeat_error))
        if i%25==0 or i+1==len(samples): print(f'ROI {folder.name}: {i+1}/{len(samples)} images',flush=True)
    return rows


def execute(args,out,report):
    torch.set_num_threads(args.threads)
    if args.extract_device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('Enable Kaggle GPU or use --extract-device cpu')
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    ref=read_reference(args.p0_report)
    if ref['arguments'].get('smoke'): raise ValueError('P0 reference must be full')
    checkpoint=audit.runner.resolve_file(args.checkpoint,'best_stg2.pth')
    cfg_path=ROOT/args.config
    if not cfg_path.is_file(): raise FileNotFoundError(cfg_path)
    samples={}; metas={}; annotations={}; source_files={}
    for split,filename in [('train','tinyperson_p3_train.pt'),('test','tinyperson_p3_val.pt')]:
        path=audit.runner.resolve_file(getattr(args,split+'_cache'),filename,cache=True)
        gt=audit.runner.resolve_file(getattr(args,split+'_gt'),f'tiny_set_{split}_all.json')
        if audit.runner.sha(path)!=ref[split]['cache_sha256'] or audit.runner.sha(gt)!=ref[split]['gt_sha256']:
            raise ValueError(f'{split}: input differs from frozen P0')
        samples[split],metas[split]=audit.runner.load_checked_cache(path,gt,smoke=True)
        annotations[split]=gt; source_files[split]=path
    train,val,split_manifest=audit.split_train_validation(samples['train'],annotations['train'])
    if [s.image_id for s in val]!=ref['evaluated_ids']['validation']: raise ValueError('Validation split differs from P0')
    if {s.image_id for s in samples['test']}!=set(ref['evaluated_ids']['test']): raise ValueError('Test set differs from P0')
    if args.smoke:
        train,val,test=train[:8],val[:4],samples['test'][:4]
    else: test=samples['test']
    samples={'train':train+val,'test':test}
    image_paths={}; preflight=[]
    for split,ss in samples.items():
        data=json.loads(annotations[split].read_text(encoding='utf-8')); info={str(x['id']):x for x in data['images']}
        image_root=Path(args.image_root) if args.image_root else annotations[split].parent.parent
        image_paths[split]={}
        for s in ss:
            path=resolve_image(image_root,info[s.image_id]['file_name'],split)
            with Image.open(path) as im:
                if tuple(im.size)!=tuple(s.img_size): raise ValueError('Original image size mismatch '+str(path))
            image_paths[split][s.image_id]=path
        preflight.append(dict(split=split,images=len(ss),candidates=sum(len(s.scores) for s in ss)))
    report['preflight']=preflight
    report['legacy_candidate_source']={s:dict(cache_sha256=ref[s]['cache_sha256'],original_manifest=metas[s],
        historical_threshold='UNKNOWN: legacy manifest contradicts observed scores; candidates preserved') for s in samples}
    configs={str(p.relative_to(ROOT)):audit.runner.sha(p) for p in sorted((ROOT/'configs').rglob('*.yml'))}
    signature=dict(schema='p0r-roi-v1',detector_sha256=audit.runner.sha(checkpoint),config=args.config,config_hashes=configs,
        projection_seed=20261112,projection_dim=64,network_size=[640,576],p3_stride=8,
        preprocessing='PIL RGB; crop stored patch rectangle; bilinear resize if needed; /255; FP32; no AMP',
        torch=torch.__version__,torchvision=torchvision.__version__,extract_device=args.extract_device,batch_tiles=args.batch_tiles,
        source_hashes={p:audit.runner.sha(ROOT/p) for p in ['TileStitchingNet/p0r.py','TileStitchingNet/stage2.py','TileStitchingNet/stage1.py','TileStitchingNet/trainEval.py','tools/kaggle_p0r_train.py']})
    # Initialize once on CPU; save exact projection weights, reload the same file for both splits.
    torch.manual_seed(signature['projection_seed'])
    projection=SharedProjection()
    projection_path=out/'roi_projection.pth'
    if not projection_path.exists(): torch.save(projection.state_dict(),projection_path)
    projection.load_state_dict(torch.load(projection_path,map_location='cpu',weights_only=True),strict=True)
    signature['projection_sha256']=audit.runner.sha(projection_path)
    run_config=dict(feature_signature=signature,parent_caches={s:ref[s]['cache_sha256'] for s in samples},split=split_manifest,
                    smoke=args.smoke,epochs=2 if args.smoke else args.epochs,train_device=args.train_device,threads=args.threads)
    run_path=out/'run_config.json'
    if run_path.is_file() and json.loads(run_path.read_text())!=run_config: raise ValueError('Resume configuration or projection differs; use another output directory')
    audit.write(run_path,run_config)
    audit.write(out/'split.json',split_manifest)
    audit.write(out/'roi_distribution_before.json',{name:audit.describe(ss) for name,ss in [('train',train),('validation',val),('test',test)]})
    report['feature_source']=signature
    projection=projection.eval().to(args.extract_device)
    print('Preflight passed:',preflight,'\nLoading frozen detector:',checkpoint,flush=True)
    detector=load_detector(cfg_path,checkpoint,args.extract_device)
    for split,ss in samples.items():
        print('Refreshing ONLY ROI for',split,flush=True)
        report[split+'_refresh']=refresh(ss,image_paths[split],detector,projection,args,out/'shards'/split,signature)
        new_path=out/f'p0r_{split}.pt'
        audit.runner.OfflineDatasetIO.save_samples(ss,str(new_path))
        audit.write(Path(str(new_path)+'.manifest.json'),dict(schema='p0r-roi-v1',parent_cache_sha256=ref[split]['cache_sha256'],
            gt_sha256=ref[split]['gt_sha256'],feature_source=signature,cache_sha256=audit.runner.sha(new_path),
            candidate_fields_unchanged=True,legacy_manifest=metas[split],smoke=args.smoke))
        # Disk round-trip verifies the actual saved candidate/feature payload.
        loaded=audit.runner.OfflineDatasetIO.load_samples(str(new_path))
        for a,b in zip(ss,loaded):
            if frozen_fields(a)!=frozen_fields(b) or not np.array_equal(a.node_features,b.node_features): raise AssertionError('Saved cache changed payload')
        del loaded
    del detector,projection
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    audit.write(out/'roi_distribution_after.json',{name:audit.describe(ss) for name,ss in [('train',train),('validation',val),('test',test)]})
    evaluators={'validation':audit.OfficialTinyEvaluator(annotations['train']), 'test':audit.OfficialTinyEvaluator(annotations['test'])}
    baseline_fns={
        'Greedy-NMS':lambda b,s:audit.runner.run_greedy_nms(b,s,.5,score_thresh=0),
        'Soft-NMS':lambda b,s:audit.runner.run_soft_nms(b,s,sigma=.5,score_thresh=0),
        'WBF':lambda b,s:audit.runner.run_wbf(b,s,.55,score_thresh=0),
        'SAHI-NMM':lambda b,s:audit.runner.run_sahi_nmm_ios(b,s,.5,score_thresh=0)}
    results={}
    for split,ss in [('validation',val),('test',test)]:
        results[split]={}
        for name,fn in baseline_fns.items():
            metrics=evaluate_predictor(ss,audit.runner.Baseline(fn),evaluators[split])
            results[split][name]=metrics
            if not args.smoke:
                diffs=audit.metric_diff(metrics,ref['metrics'][split][name])
                if any(abs(v)>1e-4 for k,v in diffs.items() if k!='Latency_ms'): raise ValueError('Frozen baseline changed '+split+'/'+name)
    audit.write(out/'baseline_reproduction.json',results)
    report['baseline_gate']='SMOKE_METRICS_ONLY' if args.smoke else 'PASSED_SAME_CANDIDATES_SAME_METRICS'
    qcfg=QualityConfig(epochs=2 if args.smoke else args.epochs,val_freq=1,device=args.train_device)
    for name,local in [('pointwise',False),('local',True)]:
        done=out/f'{name}_completed.json'
        if done.is_file():
            recorded=json.loads(done.read_text())
            if recorded['checkpoint_sha256']!=audit.runner.sha(out/f'{name}_best.pth'): raise ValueError('Completed model file changed')
            from TileStitchingNet.stage2 import load_quality_checkpoint
            model=load_quality_checkpoint(out/f'{name}_best.pth',args.train_device)
        else:
            model,history=train_quality(train,val,annotations['train'],evaluators['validation'],qcfg,local,str(out/f'{name}_best.pth'),snapshot_dir=out/(name+'_snapshots'))
            audit.write(out/f'{name}_history.json',history)
            audit.write(done,dict(checkpoint_sha256=audit.runner.sha(out/f'{name}_best.pth')))
        for split,ss in [('validation',val),('test',test)]:
            for mode in ('before_nms','after_nms'):
                results[split][name+'/'+mode]=evaluate_predictor(ss,QualityPredictor(model,mode),evaluators[split])
            audit.write(out/(split+'_metrics.json'),results[split])
    report['metrics']=results
    report['parent_cache_files_unchanged']={s:audit.runner.sha(p)==ref[s]['cache_sha256'] for s,p in source_files.items()}
    if not all(report['parent_cache_files_unchanged'].values()): raise ValueError('Parent cache file changed during run')
    report['status']='SMOKE_COMPLETED' if args.smoke else 'REBUILD_AND_TRAIN_COMPLETED'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('train-cache','test-cache','train-gt','test-gt','checkpoint','image-root'): p.add_argument('--'+key)
    p.add_argument('--p0-report',default=str(ROOT/'TileStitchingNet/reports/p0_20261116'),
                   help='Full P0 ZIP, audit_summary.json, or extracted report directory')
    p.add_argument('--config',default='configs/deim_dfine/dfine_hgnetv2_s_tinyperson_online.yml')
    p.add_argument('--output',default='/kaggle/working/p0r_full')
    p.add_argument('--extract-device',default='cuda:0'); p.add_argument('--train-device',default='cpu')
    p.add_argument('--batch-tiles',type=int,default=4); p.add_argument('--threads',type=int,default=4)
    p.add_argument('--epochs',type=int,default=18); p.add_argument('--smoke',action='store_true'); p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv)
    if min(args.batch_tiles,args.threads,args.epochs)<1: p.error('batch-tiles, threads, epochs must be positive')
    out=Path(args.output)
    if out.exists() and any(out.iterdir()) and not args.resume: p.error('Output exists; choose a new directory or use --resume')
    out.mkdir(parents=True,exist_ok=True)
    report=dict(arguments=vars(args),status='RUNNING'); failed=False
    try: execute(args,out,report)
    except Exception:
        failed=True; report['status']='ERROR'; report['error']=traceback.format_exc(); print(report['error'],flush=True)
    finally:
        audit.write(out/'p0r_summary.json',report)
        with zipfile.ZipFile(out/'p0r_reports.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.glob('*.json')): z.write(f,f.name)
            for f in sorted(out.rglob('*.pth')): z.write(f,str(f.relative_to(out)))
        print('Status:',report['status'],'\nSend back:',out/'p0r_reports.zip',flush=True)
    if failed: raise SystemExit(1)


if __name__=='__main__': main()
