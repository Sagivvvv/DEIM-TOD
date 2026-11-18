import copy
import contextlib
import io
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import torch

from .datasetBuilder import nms_np, pairwise_iou_np
from .p0r import replace_roi, frozen_fields
from .p1 import (nms_trace, scaled_scores, CandidateLabels, reliable_pairs,
                 OfficialMatchTrace, compare_traces, hits, graph_coverage)
from .stage1 import OfficialTinyEvaluator, split_train_validation
from .stage2 import QualityConfig, QualityRescorer, prepare, QualityTargets
from .test_stage2 import sample
from .pipeline_deim_tilestitching import OfflineDatasetIO

spec = importlib.util.spec_from_file_location('p1_runner', Path(__file__).resolve().parents[1]/'tools/kaggle_diagnose_p1.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


def annotations(ids):
    return dict(images=[dict(id=i,width=100,height=100,file_name=f'{i}.png') for i in ids],
        categories=[dict(id=1,name='person')], annotations=[
            dict(id=i*10+j+1,image_id=i,category_id=1,bbox=b,area=b[2]*b[3],iscrowd=0,uncertain=ignore)
            for i in ids for j,(b,ignore) in enumerate([([0,0,10,10],0),([60,60,10,10],1)])])


class P1Tests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_nms_actual_edges_and_ties(self):
        # A suppresses B; B overlaps C, but B must never act as a suppressor.
        b = np.array([[0,0,10,10],[3,0,13,10],[6,0,16,10]],np.float32)
        scores = np.array([.9,.8,.7],np.float32)
        keep, edges = nms_trace(b,scores)
        np.testing.assert_array_equal(keep,[0,2]); np.testing.assert_array_equal(edges,[[0,1]])
        for sc in [scores, np.ones(3,np.float32)]:
            np.testing.assert_array_equal(nms_trace(b,sc)[0],nms_np(b,sc,.5))
        self.assertEqual(nms_trace(b[:0],scores[:0])[1].shape,(0,2))

    def test_alpha_identity_and_labels(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'gt.json'; p.write_text(json.dumps(annotations([1])))
            s=sample(); cfg=QualityConfig(); m=QualityRescorer(14,cfg)
            with torch.no_grad(): m.head[-1].bias.fill_(.2)
            result=m(prepare(s,cfg,False))
            d=result['delta'].detach().numpy()
            np.testing.assert_array_equal(scaled_scores(s.scores,d,0),s.scores)
            np.testing.assert_array_equal(scaled_scores(s.scores,d,1),result['scores'].detach().numpy())
            targets=QualityTargets(p,cfg)
            labels=CandidateLabels(annotations([1]),targets)(s)
            q,valid,_=targets(s)
            np.testing.assert_array_equal(labels['valid'],valid)
            np.testing.assert_array_equal(labels['quality'],q)
            self.assertEqual(labels['reason'].tolist(),['positive','positive','negative','ignore_region'])
            _,edges=nms_trace(s.boxes_global,s.scores)
            pairs,ex=reliable_pairs(s,labels,edges,.5)
            self.assertEqual(len(pairs),1); self.assertEqual(pairs[0]['good'],0)
            self.assertTrue(pairs[0]['cross_tile'])
            coverage=graph_coverage(s,edges,cfg)
            self.assertTrue(coverage[0]['suppressed_reads_winner'])
            # Two qualifying different-instance candidates must not get an ordering label.
            labels['gid'][1]=1; labels['best'][:2]=[.95,.6]
            pairs,ex=reliable_pairs(s,labels,edges,.5)
            self.assertFalse(pairs); self.assertEqual(ex['different_gt_positives'],1)

    def test_fractional_gt_rounding_preserves_qfl_threshold(self):
        data=annotations([1]); ann=data['annotations'][0]
        ann['bbox']=[1024.00004,32.,20.00004,12.]
        s=sample(); s.boxes_global[0]=[1024.,32.,1034.,44.]
        old=np.array([ann['bbox']],np.float32)
        old[:,2:]+=old[:,:2]
        self.assertEqual(pairwise_iou_np(s.boxes_global[:1],old)[0,0],.5)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'gt.json'; p.write_text(json.dumps(data))
            targets=QualityTargets(p,QualityConfig())
            labels=CandidateLabels(data,targets)(s)
            self.assertLess(labels['best'][0],.5)
            self.assertEqual(labels['reason'][0],'gray_iou')
            self.assertFalse(labels['valid'][0]); self.assertEqual(labels['quality'][0],0)

    def test_official_occupied_gt_fallback_and_ignore(self):
        s=sample()
        # Candidate 1's best GT is occupied, but a second eligible GT must still match.
        s.boxes_global=np.array([[0,0,10,10],[.5,0,10.5,10],[60,60,70,70]],np.float32)
        s.scores=np.array([.9,.8,.7],np.float32); s.patch_ids=np.array([0,1,0])
        data=annotations([1]); data['annotations'].append(dict(id=99,image_id=1,category_id=1,bbox=[2,0,10,10],area=100,iscrowd=0))
        trace=OfficialMatchTrace(data)([s],[np.arange(3)],[s.scores])
        r=trace['1']['all@0.5']
        self.assertEqual(hits(r,200),{11,99})
        self.assertEqual(r['kinds'],['tp','tp','ignored'])
        # Empty detections still retain all eligible GTs.
        empty=OfficialMatchTrace(data)([s],[np.array([],np.int64)],[s.scores])
        self.assertEqual(set(empty['1']['all@0.5']['eligible_gt_ids']),{11,99})
        self.assertFalse(hits(empty['1']['all@0.5']))

    def test_budget200_unique_gt_and_reordering(self):
        s=sample()
        s.boxes_global=np.tile(np.array([[30,30,40,40]],np.float32),(202,1))
        s.boxes_global[201]=[0,0,10,10]
        s.patch_ids=np.zeros(202,np.int64)
        s.scores=np.linspace(.99,.1,202,dtype=np.float32)
        tracer=OfficialMatchTrace(annotations([1]))
        base=tracer([s],[np.arange(202)],[s.scores])
        self.assertEqual(hits(base['1']['all@0.5']),{11})
        self.assertFalse(hits(base['1']['all@0.5'],200))
        new=s.scores.copy(); new[201]=1
        changed=tracer([s],[np.arange(202)],[new])
        comparison=compare_traces([s],base,changed)
        c=comparison['summary']['all@0.5']
        self.assertEqual(c['recovered_gt'],1)
        self.assertEqual(c['base_budget_lost_gt'],1)
        self.assertEqual(c['new_budget_lost_gt'],0)
        self.assertEqual(c['entered_from_base_below200'],1)
        self.assertEqual(c['recovered_below200'],1)

    def test_cross_tile_rescue_requires_before_nms(self):
        s=sample(); s.boxes_global[1]=[2,0,12,10]
        bk,be=nms_trace(s.boxes_global,s.scores)
        new=s.scores.copy(); new[0]=.95
        nk,ne=nms_trace(s.boxes_global,new)
        tr=OfficialMatchTrace(annotations([1]))
        base=tr([s],[bk],[s.scores]); before=tr([s],[nk],[new]); after=tr([s],[bk],[new])
        c=compare_traces([s],base,before,[be],[ne])['summary']['all@0.75']
        self.assertEqual(c['recovered_gt'],1)
        self.assertEqual(c['recovered_cross_tile_suppression'],1)
        self.assertEqual(compare_traces([s],base,after,[be],[be])['summary']['all@0.75']['recovered_gt'],0)

    def test_full_runner_resume_and_input_guard(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); gt=root/'gt.json'; gt.write_text(json.dumps(annotations(range(1,7))))
            ss=[sample(str(i)) for i in range(1,7)]
            for s in ss: replace_roi(s,np.ones((len(s.scores),64),np.float32))
            cache=root/'p0r_train.pt'; OfflineDatasetIO.save_samples(ss,str(cache))
            _,val,split=split_train_validation(ss,gt)
            cfg=QualityConfig(); ev=OfficialTinyEvaluator(gt)
            keep=[nms_np(s.boxes_global,s.scores,.5) for s in val]
            metrics=runner.evaluate(ev,val,keep,[s.scores for s in val])
            srcs={p:runner.audit.runner.sha(runner.ROOT/p) for p in ['TileStitchingNet/stage2.py','TileStitchingNet/stage1.py','tools/kaggle_p0r_train.py']}
            ref=dict(status='REBUILD_AND_TRAIN_COMPLETED',arguments=dict(smoke=False),feature_source=dict(source_hashes=srcs),
                     train_refresh=[dict(image_id=s.image_id,frozen=frozen_fields(s)) for s in ss])
            meta=dict(schema='p0r-roi-v1',smoke=False,candidate_fields_unchanged=True,
                      cache_sha256=runner.audit.runner.sha(cache),gt_sha256=runner.audit.runner.sha(gt))
            Path(str(cache)+'.manifest.json').write_text(json.dumps(meta))
            archive=root/'p0r_reports.zip'; vm={'Greedy-NMS':metrics}
            with zipfile.ZipFile(archive,'w') as z:
                for family in ['pointwise','local']:
                    model=QualityRescorer(78,cfg,family=='local')
                    for tag,epoch in [('best',1),('epoch_06',6),('epoch_18',18)]:
                        payload=dict(model=model.state_dict(),config=asdict(cfg),local=family=='local',feature_dim=78,epoch=epoch,validation=metrics)
                        path=root/(family+tag+'.pth'); torch.save(payload,path)
                        member=family+'_best.pth' if tag=='best' else family+'_snapshots/'+tag+'.pth'
                        z.write(path,member)
                        if tag=='best': z.writestr(family+'_completed.json',json.dumps(dict(checkpoint_sha256=runner.audit.runner.sha(path))))
                    for mode in ['before_nms','after_nms']: vm[family+'/'+mode]=metrics
                for name,data in [('p0r_summary.json',ref),('split.json',split),('validation_metrics.json',vm),('p0r_train.pt.manifest.json',meta)]:
                    z.writestr(name,json.dumps(data))
            out=root/'p1'
            args=['--train-cache',str(cache),'--train-gt',str(gt),'--p0r-report',str(archive),'--output',str(out),'--threads','1']
            with contextlib.redirect_stdout(io.StringIO()): runner.main(args)
            report=json.loads((out/'p1_summary.json').read_text())
            self.assertEqual(report['status'],'P1_DIAGNOSIS_COMPLETED')
            self.assertFalse(report['input']['test_evaluated'])
            self.assertEqual(len(json.loads((out/'validation_alpha_sweep.json').read_text())),60)
            with contextlib.redirect_stdout(io.StringIO()): runner.main(args+['--resume'])
            self.assertTrue((out/'p1_reports.zip').exists())
            # Same reports work after Kaggle expands the ZIP into a folder.
            expanded=root/'expanded'; expanded.mkdir()
            with zipfile.ZipFile(archive) as z:
                for name in z.namelist():
                    dest=expanded/name; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(z.read(name))
            folder_args=args.copy()
            folder_args[folder_args.index('--p0r-report')+1]=str(expanded)
            folder_args[folder_args.index('--output')+1]=str(root/'smoke')
            with contextlib.redirect_stdout(io.StringIO()): runner.main(folder_args+['--smoke'])
            self.assertEqual(json.loads((root/'smoke/p1_summary.json').read_text())['status'],'P1_SMOKE_COMPLETED')
            # A different GT file must stop before diagnostics instead of silently reusing jobs.
            gt.write_text(gt.read_text()+' ')
            with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()): runner.main(args+['--resume'])
            self.assertEqual(json.loads((out/'p1_summary.json').read_text())['status'],'ERROR')


if __name__=='__main__': unittest.main()
