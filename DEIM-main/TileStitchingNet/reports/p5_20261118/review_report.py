"""Replay the P5 freeze, AP/GT summaries, and paired baseline comparisons."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from TileStitchingNet.p5 import PROTOCOL, SEEDS, METHODS, choose_family, sampled_ap, summarize


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(z, name):
    raw = z.read(name)
    return json.loads(gzip.decompress(raw) if name.endswith('.gz') else raw)


def rows_and_hits(ids, preds, trace, scope):
    rows, capped, full = [], set(), set()
    for iid in ids:
        row = trace[iid].get(scope, {})
        idx = np.asarray(row.get('indices', []), int)
        valid = ~np.asarray(row.get('ignored', []), bool)
        gt = np.asarray(row.get('matches', []), int)
        scores = np.asarray(preds[iid]['scores'])[idx]
        rows.append((scores[:200][valid[:200]], gt[:200][valid[:200]]>0, len(row.get('eligible_gt_ids', []))))
        full.update((iid,int(k)) for k in gt[valid & (gt>0)])
        capped.update((iid,int(k)) for k in gt[:200][valid[:200] & (gt[:200]>0)])
    return rows, capped, full


def bootstrap(rows, ids, groups):
    # Same paired draws and official image ordering for every method.
    buckets = {}
    for i, iid in enumerate(ids):
        buckets.setdefault(groups[iid], []).append(i)
    buckets = list(buckets.values())
    rng = np.random.default_rng(42)
    deltas = {name:[] for name in ('C-A','A-Greedy-NMS','A-WBF','A-Soft-NMS')}
    for _ in range(1000):
        draw = sorted(i for g in rng.integers(len(buckets),size=len(buckets)) for i in buckets[g])
        ap = {m:sampled_ap(r,draw) for m,r in rows.items() if m!='SAHI-NMM'}
        a,c = [np.mean([ap[f'{m}_{s}'] for s in SEEDS]) for m in METHODS]
        for name,value in zip(deltas,[c-a,a-ap['Greedy-NMS'],a-ap['WBF'],a-ap['Soft-NMS']]):
            deltas[name].append(value)
    return {name:np.percentile(v,[2.5,97.5]).tolist() for name,v in deltas.items()}


def wbf_cases(z, prefix, only_wbf):
    side = 'train' if prefix else 'test'
    annotation = json.loads((ROOT.parent/f'datasets/Tinyperson/annotations/tiny_set_{side}_all.json').read_bytes())
    boxes = {a['id']:a['bbox'] for a in annotation['annotations']}
    with zipfile.ZipFile(HERE.parent/'p0r_20261116/p0r_reports.zip') as old:
        counts = {r['image_id']:r['candidate_count'] for r in read(old,'p0r_summary.json')[side+'_refresh']}
    records = {k:{} for k in sorted(only_wbf)}
    for method in ['Soft-NMS','WBF',*(f'A_full_{s}' for s in SEEDS)]:
        pred = read(z,prefix+'predictions/'+method+'_predictions.json.gz')
        if method=='Soft-NMS':
            # The recorded implementation returns every input box at cutoff 0;
            # only the order and scores change. Confirm coverage for every image.
            assert all(len(p['boxes_xyxy'])==counts[i] for i,p in pred.items())
        for (image,gid), record in records.items():
            x,y,w,h = boxes[gid]
            b = np.array(pred[image]['boxes_xyxy']).reshape(-1,4)
            inter = np.maximum(0,np.minimum(b[:,2:],[x+w,y+h])-np.maximum(b[:,:2],[x,y])).prod(1)
            iou = inter / (w*h+(b[:,2:]-b[:,:2]).prod(1)-inter)
            top = np.argsort(-np.array(pred[image]['scores']),kind='stable')[:200]
            record[method] = [float(iou.max(initial=0)),float(iou[top].max(initial=0))]
    partition = {}
    for (image,gid),rec in records.items():
        assert rec['WBF'][1]>=.5-1e-8
        if rec['Soft-NMS'][0]<.5:
            phase = 'no_raw_candidate_at_IoU50'
        elif max(rec[f'A_full_{s}'][0] for s in SEEDS)<.5:
            phase = 'raw_matchable_but_no_A_survivor'
        elif max(rec[f'A_full_{s}'][1] for s in SEEDS)<.5:
            phase = 'A_survivor_but_below_top200_in_all_A'
        else:
            phase = 'A_in_top200_but_no_official_hit'
        partition.setdefault(phase,[]).append(dict(image=image,gt=gid,bbox_xywh=boxes[gid],
                                                   max_iou_full_and_capped=rec))
    return dict(counts={k:len(v) for k,v in partition.items()},cases=partition)


def analyze(z, prefix):
    metrics, groups, changes = [read(z,prefix+n) for n in ('metrics.json','sampling_groups.json','gt_changes_and_failure_cases.json.gz')]
    ids = sorted(groups,key=int)
    rows, capped, full = {}, {}, {}
    for method, values in metrics.items():
        pred, trace = [read(z,prefix+'predictions/'+method+'_'+suffix+'.json.gz') for suffix in ('predictions','matches')]
        assert set(pred)==set(trace)==set(ids)
        capped[method],full[method] = {},{}
        for key, metric in [('tiny@0.5','Tiny_AP50'),('tiny@0.75','Tiny_AP75'),('all@0.5','AP50'),('all@0.75','AP75')]:
            r,capped[method][key],full[method][key] = rows_and_hits(ids,pred,trace,key)
            assert abs(sampled_ap(r,range(len(ids)))-values[metric])<1e-8, (method,metric)
            if metric=='Tiny_AP50':
                rows[method] = r
    for label, scopes in changes.items():
        new,old = label.split('_minus_')
        for key, record in scopes.items():
            a,b = capped[old][key],capped[new][key]
            fa,fb = full[old][key],full[new][key]
            for field,expected in [('gains',b-a),('losses',a-b),('gains_before_top200',fb-fa),('losses_before_top200',fa-fb)]:
                assert set(map(tuple,record[field]))==expected
            assert (record['gained'],record['lost'],record['old_budget_loss'],record['new_budget_loss'])==(len(b-a),len(a-b),len(fa-a),len(fb-b))
    key = 'tiny@0.5'
    w = capped['WBF'][key]
    a = [capped[f'A_full_{s}'][key] for s in SEEDS]
    w_only = w-set.union(*a)
    a_only = set.intersection(*a)-w
    intervals = bootstrap(rows,ids,groups)
    reported = read(z,prefix+'paired_comparison.json')
    for original, rerun in [('C_minus_A','C-A'),('A_minus_reference','A-Greedy-NMS')]:
        np.testing.assert_allclose(intervals[rerun],reported['percentile_95'][original],rtol=0,atol=1e-8)
    best = read(z,prefix+'seed_summary.json')
    assert all(best[k]==v for k,v in summarize(metrics,SEEDS,'Greedy-NMS').items())
    return dict(images=len(ids),groups=len(set(groups.values())),
                video_frames=sum(v.startswith('video_filename:') for v in groups.values()),
                eligible_tiny=sum(r[2] for r in rows['WBF']),
                hit_counts={m:dict(capped=len(capped[m][key]),full=len(full[m][key])) for m in capped},
                wbf_hit_missed_by_all_A=len(w_only), all_A_hit_missed_by_wbf=len(a_only),
                wbf_only_geometry=wbf_cases(z,prefix,w_only),
                common_A_gains_vs_greedy=len(set.intersection(*a)-capped['Greedy-NMS'][key]),
                common_A_losses_vs_greedy=len(capped['Greedy-NMS'][key]-set.union(*a)),
                seed_summary=best,paired_95=intervals)


def main():
    archive = HERE/'p5_reports.zip'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        freeze = read(z,'validation/frozen_config.json')
        run = read(z,'run_config.json')
        valid = read(z,'validation/run_config.json')
        assert read(z,'status.json')['status']=='P5_TEST_COMPLETED'
        assert read(z,'validation/status.json')['status']=='P5_VALIDATION_FROZEN'
        assert valid['reference']==run['reference']==json.loads((ROOT/'P5_REFERENCE.json').read_bytes())
        assert freeze['protocol']==valid['protocol']==run['protocol']==PROTOCOL
        assert freeze['run_sha256']==sha(z.read('validation/run_config.json'))
        assert freeze['seed_summary_sha256']==sha(z.read('validation/seed_summary.json'))
        assert sha(z.read('validation/frozen_config.json'))==run['freeze_sha256']==read(z,'final_report.json')['freeze_sha256']
        assert freeze['selected_family']==choose_family(read(z,'validation/seed_summary.json'))=='A_full'
        assert valid['environment']==run['environment']
        for name,h in run['environment']['sources'].items():
            assert sha((ROOT/name).read_bytes())==h
        for side in ('train','test'):
            assert sha((ROOT.parent/f'datasets/Tinyperson/annotations/tiny_set_{side}_all.json').read_bytes())==run['reference'][side]['gt_sha256']
        split = read(z,'validation/split.json')
        assert not set(split['train_ids']) & set(split['val_ids'])
        assert set(split['val_ids'])==set(read(z,'validation/sampling_groups.json'))
        with zipfile.ZipFile(HERE.parent/'p4_20261117/p4_reports.zip') as p4:
            for k in ('train_ids','val_ids'):
                assert split[k]==read(p4,'split.json')[k]
            for method in [f'{f}_{s}' for f in METHODS for s in SEEDS]:
                info = freeze['models'][method]
                raw = z.read('validation/'+info['path'])
                assert sha(raw)==info['sha256']
                curve = read(z,'validation/'+method+'_validation_curve.json')
                epoch = max(range(len(curve)),key=lambda i:curve[i]['Tiny_AP50'])
                assert len(curve)==19 and epoch==info['best_epoch']
                assert curve[epoch]['Tiny_AP50']==read(z,'validation/metrics.json')[method]['Tiny_AP50']
                if method.endswith('_42'):
                    family = method[:-3]
                    assert raw==p4.read(f'{family}/{family}_best.pth')
        with zipfile.ZipFile(HERE.parent/'p0r_20261116/p0r_reports.zip') as p0r:
            test = read(z,'metrics.json')
            for name in ('Greedy-NMS','WBF','Soft-NMS','SAHI-NMM'):
                old = read(p0r,'test_metrics.json')[name]
                np.testing.assert_allclose([v for k,v in old.items() if k!='Latency_ms'],
                                           [test[name][k] for k in old if k!='Latency_ms'],rtol=0,atol=1e-8)
        report = dict(sha256=sha(archive.read_bytes()), source_files=len(run['environment']['sources']),
                      validation=analyze(z,'validation/'),test=analyze(z,''))
        (HERE/'review_evidence.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        for scope in ('validation','test'):
            print(scope,{k:v for k,v in report[scope].items() if k not in ('seed_summary','wbf_only_geometry')})
            print('WBF only:',report[scope]['wbf_only_geometry']['counts'])


if __name__=='__main__':
    main()
