"""P0 source follow-up: inspect frozen cache contents; no training or extraction."""
import argparse
import hashlib
import json
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'tools'))
import numpy as np
import kaggle_audit_p0 as audit


# Frozen facts copied from the reviewed full P0 report, not inferred from current inputs.
# Kept inside this script so Kaggle uploads need not include a separate ZIP asset.
BUNDLED_REFERENCE = {'snapshot_id': 'p0-2026-11-16',
 'source_archive_sha256': '94b4e5db8d12b4ebe1ea343b388d6b243524b77594ed807e849f588d340f7ca4',
 'arguments': {'smoke': False},
 'evaluated_ids': {'validation': ['125',
                                  '570',
                                  '306',
                                  '30',
                                  '686',
                                  '691',
                                  '124',
                                  '59',
                                  '423',
                                  '84',
                                  '162',
                                  '254',
                                  '392',
                                  '62',
                                  '296',
                                  '427',
                                  '650',
                                  '153',
                                  '4',
                                  '528',
                                  '643',
                                  '351',
                                  '618',
                                  '508',
                                  '476',
                                  '234',
                                  '216',
                                  '148',
                                  '662',
                                  '614',
                                  '429',
                                  '144',
                                  '321',
                                  '246',
                                  '769',
                                  '454',
                                  '77',
                                  '628',
                                  '494',
                                  '375',
                                  '87',
                                  '627',
                                  '624',
                                  '683',
                                  '621',
                                  '467',
                                  '773',
                                  '180',
                                  '448',
                                  '583',
                                  '187',
                                  '0',
                                  '557',
                                  '517',
                                  '785',
                                  '123',
                                  '265',
                                  '509',
                                  '173',
                                  '613',
                                  '43',
                                  '490',
                                  '185',
                                  '641',
                                  '564',
                                  '485',
                                  '201',
                                  '81',
                                  '669',
                                  '323',
                                  '220',
                                  '413',
                                  '282',
                                  '237',
                                  '278',
                                  '83',
                                  '26',
                                  '512',
                                  '699',
                                  '434',
                                  '780',
                                  '603',
                                  '681',
                                  '538',
                                  '202',
                                  '702',
                                  '402',
                                  '368',
                                  '319',
                                  '462',
                                  '235',
                                  '679',
                                  '515',
                                  '477',
                                  '19',
                                  '536',
                                  '130',
                                  '651',
                                  '522',
                                  '472',
                                  '567',
                                  '89',
                                  '635',
                                  '674',
                                  '55',
                                  '711',
                                  '370',
                                  '299',
                                  '168',
                                  '793',
                                  '373',
                                  '226',
                                  '525',
                                  '415',
                                  '189',
                                  '596',
                                  '478',
                                  '616',
                                  '272',
                                  '141',
                                  '95',
                                  '82',
                                  '318',
                                  '260',
                                  '195',
                                  '110',
                                  '176',
                                  '103',
                                  '714',
                                  '617',
                                  '386',
                                  '523',
                                  '486',
                                  '441',
                                  '480',
                                  '206',
                                  '553',
                                  '707',
                                  '500',
                                  '213',
                                  '230',
                                  '690',
                                  '152',
                                  '8',
                                  '367',
                                  '499',
                                  '407',
                                  '252',
                                  '257']},
 'train': {'cache_sha256': '4d2138cf0fba10afbeea3c1cf3ccbae072f0ebb647f84c7c1281b819e97acf47',
           'manifest': {'conf_thresh': 0.15}},
 'test': {'cache_sha256': '0e97ef57d22475e3dc34fe7e9692d6891760bc0795c20ab6d38274a27213e571',
          'manifest': {'conf_thresh': 0.15}}}


def similarity(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or not a.size: return None
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return dict(cosine=float(a@b/den) if den else None,
                pearson=float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None,
                l2=float(np.linalg.norm(a-b)))


def resolve_reference_path(path):
    """Accept an archive, a JSON file, or a report directory after upload extraction."""
    path=Path(path)
    # Some upload workflows replace p0_reports.zip with p0_reports/.
    if not path.exists() and path.suffix.lower()=='.zip' and path.with_suffix('').is_dir():
        path=path.with_suffix('')
    if path.is_dir():
        names=sorted(path.rglob('audit_summary.json'))
        if not names:
            names=sorted(path.rglob('p0_reports.zip'))
        if len(names)!=1:
            raise ValueError(f'P0 report directory must identify one audit_summary.json (or p0_reports.zip); found {len(names)} in {path}')
        path=names[0]
    if not path.is_file():
        raise FileNotFoundError(f'P0 reference not found: {path}. Pass p0_reports.zip, audit_summary.json, or its extracted directory.')
    return path


def read_reference(path=None):
    if path is None:
        return json.loads(json.dumps(BUNDLED_REFERENCE))
    path=resolve_reference_path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names=[n for n in z.namelist() if n.endswith('audit_summary.json')]
            if len(names)!=1: raise ValueError('Reference archive must have one audit_summary.json')
            return json.loads(z.read(names[0]))
    return json.loads(path.read_text(encoding='utf-8-sig'))


def inspect_samples(samples, declared_threshold):
    rows=[]; buckets={}; all_roi=[]; valid_scores=[]
    counts=dict(candidates=0,below_008=0,below_015=0,below_declared=0,below_declared_tolerant=0,
                score_column_mismatch=0,roi_quantized_mismatch=0,roi_missing_rows=0,nonfinite_rows=0,nonperson_rows=0)
    # Fixed bins, identical for train/validation/test; size is sqrt(box area).
    size_edges=np.array([0,8,12,20,32,64,np.inf])
    score_edges=np.array([0,.08,.15,.3,.5,.7,1.000001])
    for s in samples:
        n=len(s.scores); nf=s.node_features; scores=s.scores
        if len(nf)!=n or nf.ndim!=2 or nf.shape[1]<12: raise ValueError('Malformed node features: '+s.image_id)
        counts['candidates']+=n
        finite=np.isfinite(scores)&np.isfinite(nf).all(1)
        counts['nonfinite_rows']+=int((~finite).sum())
        counts['nonperson_rows']+=int((s.class_ids!=1).sum())
        below08=int(np.sum(scores<.08)); below15=int(np.sum(scores<.15))
        below=int(np.sum(scores<declared_threshold)) if declared_threshold is not None else 0
        # Large margin distinguishes a substantial discrepancy from FP16 boundary effects.
        below_tol=int(np.sum(scores<declared_threshold-.001)) if declared_threshold is not None else 0
        score_error=np.abs(nf[:,4]-scores)
        score_bad=int(np.sum(~np.isfinite(score_error)|(score_error>1e-6)))
        counts['below_008']+=below08; counts['below_015']+=below15
        counts['below_declared']+=below; counts['below_declared_tolerant']+=below_tol
        counts['score_column_mismatch']+=score_bad
        roi=nf[:,12:]
        roi_bad=0; quant_error=None
        if n and (s.roi_features is None or roi.shape[1]==0):
            counts['roi_missing_rows']+=n
        elif n:
            if s.roi_features.shape!=roi.shape: raise ValueError('ROI array shape mismatch: '+s.image_id)
            # Export stores roi_features in FP16 but node_features in FP32.
            expected=roi.astype(np.float16).astype(np.float32)
            actual=s.roi_features.astype(np.float16).astype(np.float32)
            quant_error=np.abs(expected-actual)
            roi_bad=int(np.sum(~np.isfinite(quant_error).all(1)|(quant_error>0).any(1)))
            counts['roi_quantized_mismatch']+=roi_bad
        rows.append(dict(image_id=s.image_id,n=n,minimum=float(scores.min()) if n else None,
            below_008=below08,below_015=below15,below_declared=below,below_declared_tolerant=below_tol,
            score_column_mismatch=score_bad,score_column_max_error=float(score_error.max()) if n else 0,
            roi_quantized_mismatch=roi_bad,
            roi_quantized_max_error=float(quant_error.max()) if quant_error is not None and quant_error.size else None,
            roi_mean=roi[finite].mean(0).tolist() if finite.any() and roi.shape[1] else None))
        good=finite & (s.class_ids==1)
        valid_scores.extend(scores[good].tolist())
        if not roi.shape[1]: continue
        wh=np.maximum(s.boxes_global[:,2:]-s.boxes_global[:,:2],0)
        scale=np.sqrt(wh.prod(1))
        si=np.searchsorted(size_edges,scale,side='right')-1
        ci=np.searchsorted(score_edges,scores,side='right')-1
        for x,y in sorted(set(zip(si[good].tolist(),ci[good].tolist()))):
            selected=roi[good & (si==x)&(ci==y)]
            key=f'size_bin_{x}/score_bin_{y}'
            item=buckets.setdefault(key,dict(n=0,sum=np.zeros(roi.shape[1],np.float64),sum_sq=np.zeros(roi.shape[1],np.float64)))
            item['n']+=len(selected); item['sum']+=selected.astype(np.float64).sum(0)
            item['sum_sq']+=np.square(selected.astype(np.float64)).sum(0)
        all_roi.append(roi[good])
    strata={}
    for key,b in buckets.items():
        mean=b['sum']/b['n']
        strata[key]=dict(n=b['n'],mean=mean.tolist(),std=np.sqrt(np.maximum(b['sum_sq']/b['n']-mean**2,0)).tolist())
    combined=np.concatenate(all_roi) if all_roi else np.empty((0,0))
    image_means=[r['roi_mean'] for r in rows if r['roi_mean'] is not None]
    observed=np.array(valid_scores)
    return dict(counts=counts,below_015_fraction=counts['below_015']/max(1,counts['candidates']),
        observed_score_minimum=float(observed.min()) if observed.size else None,
        observed_score_quantiles=np.quantile(observed,[0,.1,.5,.9,1]).tolist() if observed.size else [],
        candidate_weighted_roi_mean=combined.mean(0).tolist() if len(combined) else [],
        image_weighted_roi_mean=np.mean(image_means,axis=0).tolist() if image_means else [],
        size_bin_edges=[0,8,12,20,32,64,'inf'],score_bin_edges=score_edges.tolist(),
        strata=strata,per_image=rows)


def compare_strata(a,b):
    rows=[]
    for key in sorted(a['strata'].keys() & b['strata'].keys()):
        x,y=a['strata'][key],b['strata'][key]
        if min(x['n'],y['n'])<100: continue
        rows.append(dict(stratum=key,n_left=x['n'],n_right=y['n'],similarity=similarity(x['mean'],y['mean'])))
    return dict(candidate_weighted=similarity(a['candidate_weighted_roi_mean'],b['candidate_weighted_roi_mean']),
        image_weighted=similarity(a['image_weighted_roi_mean'],b['image_weighted_roi_mean']),matched_strata=rows,
        interpretation='Descriptive distribution check, not proof of identical/different projection weights.')


def execute(args,out,report):
    reference=read_reference(args.p0_report)
    report['reference_mode']='external_file' if args.p0_report else 'bundled_p0_20261116'
    report['reference_report_sha256']=audit.runner.sha(resolve_reference_path(args.p0_report)) if args.p0_report else reference['source_archive_sha256']
    report['reference_snapshot_sha256']=hashlib.sha256(json.dumps(reference,sort_keys=True).encode()).hexdigest()
    report['reference_snapshot']=reference
    report['source_sha256']=audit.runner.sha(Path(__file__))
    groups={}; thresholds={}; paths={}
    for split,fn in [('train','tinyperson_p3_train.pt'),('test','tinyperson_p3_val.pt')]:
        path=audit.runner.resolve_file(getattr(args,split+'_cache'),fn,cache=True)
        actual_hash=audit.runner.sha(path)
        expected_hash=reference[split]['cache_sha256']
        if actual_hash!=expected_hash: raise ValueError(f'{split}: cache differs from P0; refusing mixed-pool analysis')
        ss=audit.runner.OfflineDatasetIO.load_samples(str(path))
        thresholds[split]=reference[split]['manifest'].get('conf_thresh') if reference[split]['manifest'] else None
        groups[split]=ss; paths[split]=path
        report[split+'_input']=dict(path=str(path),sha256=actual_hash,declared_threshold=thresholds[split])
    validation_ids=set(reference['evaluated_ids']['validation'])
    if reference['arguments'].get('smoke'): raise ValueError('Requires a full P0 reference report, not smoke')
    train_ids={s.image_id for s in groups['train']}
    if not validation_ids<=train_ids: raise ValueError('Validation IDs not present in frozen train cache')
    train=[s for s in groups['train'] if s.image_id not in validation_ids]
    val=[s for s in groups['train'] if s.image_id in validation_ids]
    stats={}
    for name,ss in [('train',train),('validation',val),('test',groups['test'])]:
        print('Inspecting',name,'images=',len(ss),flush=True)
        value=inspect_samples(ss,thresholds['train' if name!='test' else 'test'])
        audit.write(out/(name+'_per_image.json'),value.pop('per_image'))
        stats[name]=value
        print(name,value['counts'],flush=True)
    report['content_statistics']=stats
    report['roi_comparisons']={k:compare_strata(stats['train'],stats[k]) for k in ('validation','test')}
    report['cache_files_unchanged']={k:audit.runner.sha(p)==reference[k]['cache_sha256'] for k,p in paths.items()}
    # Optional historical evidence is inventoried, never silently equated with the actual cache exporter.
    report['historical_evidence']=[]
    for p in args.evidence:
        path=Path(p)
        if not path.is_file(): raise FileNotFoundError(path)
        report['historical_evidence'].append(dict(path=str(path),sha256=audit.runner.sha(path),bytes=path.stat().st_size))
    report['projection_identity']='UNVERIFIED: channel statistics and matching seed metadata do not establish shared weights'
    contradiction=any(s['counts']['below_declared_tolerant'] for s in stats.values())
    content_bad=any(s['counts'][k] for s in stats.values() for k in ('score_column_mismatch','roi_quantized_mismatch','nonfinite_rows'))
    report['confirmed_threshold_contradiction']=contradiction
    report['status']='CONTENT_MISMATCH' if content_bad else ('METADATA_CONTRADICTION' if contradiction else 'CONTENT_CHECKED_SOURCE_STILL_UNVERIFIED')
    report['next_action']='Review original export logs/projection state; if unavailable, explicitly select a reproducible new feature source. Do not rewrite legacy manifests or silently filter candidates.'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--p0-report',help='Optional full P0 ZIP/JSON or extracted report directory; defaults to the embedded reviewed 2026-11-16 snapshot')
    p.add_argument('--train-cache'); p.add_argument('--test-cache')
    p.add_argument('--evidence',action='append',default=[],help='Optional original export log/script/projection file; hash inventory only')
    p.add_argument('--output',default='/kaggle/working/p0_sources')
    args=p.parse_args(argv); out=Path(args.output)
    if out.exists() and any(out.iterdir()): p.error('Choose an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),arguments=vars(args),status='RUNNING')
    failed=False
    try: execute(args,out,report)
    except Exception:
        failed=True; report['status']='ERROR'; report['error']=traceback.format_exc(); print(report['error'],flush=True)
    finally:
        audit.write(out/'source_audit.json',report)
        archive=out/'p0_sources_reports.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.glob('*.json')): z.write(f,f.name)
        print('Status:',report['status'],'\nSend back:',archive,flush=True)
    if failed: raise SystemExit(1)


if __name__=='__main__': main()
