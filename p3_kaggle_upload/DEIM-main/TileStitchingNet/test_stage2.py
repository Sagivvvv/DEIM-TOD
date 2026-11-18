import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from .datasetBuilder import collect_slice_predictions_from_detector, CroppedGTLabelAssigner, nms_np
from .pipeline_deim_tilestitching import OfflineDatasetIO
from .stage2 import (QualityConfig,QualityTargets,QualityRescorer,QualityPredictor,
                     candidate_graph,prepare,quality_loss,load_quality_checkpoint)


def sample(i='1'):
    b=np.array([[0,0,10,10],[1,0,11,10],[30,30,40,40],[60,60,70,70]],np.float32)
    s=collect_slice_predictions_from_detector(
        [dict(boxes=b,scores=np.array([.7,.8,.6,.5],np.float32),class_ids=np.ones(4,np.int64))],
        np.array([[0,0,100,100]],np.float32),(100,100),conf_thresh=0,intra_nms_iou=1,
        gt_boxes=b[:1],gt_classes=np.ones(1,np.int64),assigner=CroppedGTLabelAssigner(),image_id=i)
    s.tile_rects_all=np.array([[0,0,100,100]],np.float32)
    s.patch_ids=np.array([0,1,0,1])
    return s


class StageTwoTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def annotations(self,path,ids):
        data=dict(images=[dict(id=i,width=100,height=100,file_name=f'{i}.jpg') for i in ids],
                  categories=[dict(id=1,name='person')],annotations=[])
        for i in ids:
            for j,(b,ignore) in enumerate([([0,0,10,10],False),([60,60,10,10],True)]):
                data['annotations'].append(dict(id=i*2+j,image_id=i,category_id=1,bbox=b,area=100,iscrowd=0,uncertain=int(ignore)))
        path.write_text(json.dumps(data),encoding='utf-8')
        return data

    def test_quality_all_duplicates_ignore_and_ambiguity(self):
        p=self.root/'gt.json'
        data=self.annotations(p,[1])
        cfg=QualityConfig()
        q,valid,gid=QualityTargets(p,cfg)(sample())
        self.assertGreater(q[0],q[1])  # lower-scoring box is the better localized one
        self.assertTrue(np.all(q[:2]>.5))  # both duplicate candidates have positive quality
        np.testing.assert_array_equal(valid,[True,True,True,False])
        self.assertEqual(gid[0],gid[1])
        # Two nearly identical normal GTs create ambiguous instance assignment.
        data['annotations'].append(dict(id=99,image_id=1,category_id=1,bbox=[.1,0,10,10],area=100,iscrowd=0))
        p.write_text(json.dumps(data),encoding='utf-8')
        self.assertFalse(QualityTargets(p,cfg)(sample())[1][:2].any())

    def test_identity_coordinates_sparse_graph_and_empty(self):
        s=sample()
        cfg=QualityConfig(neighbors=2)
        idx,geo,mask=candidate_graph(s,cfg)
        self.assertEqual(idx.shape,(4,3))
        self.assertTrue(np.isfinite(geo).all())
        self.assertTrue(mask.any(1).all())
        self.assertTrue(np.any(geo[...,5]==0))  # cross-patch relations exist
        for local in (False,True):
            model=QualityRescorer(14,cfg,local)
            out=model(prepare(s,cfg,local))
            np.testing.assert_array_equal(out['scores'].detach().numpy(),s.scores)
            keep=nms_np(s.boxes_global,s.scores,.5)
            for mode in ('before_nms','after_nms'):
                boxes,scores,_=QualityPredictor(model,mode).predict_sample(s)
                np.testing.assert_array_equal(boxes,s.boxes_global[keep])
                np.testing.assert_array_equal(scores,s.scores[keep])
            empty=copy.deepcopy(s)
            for key in ('node_features','scores','boxes_global','patch_ids'):
                setattr(empty,key,getattr(empty,key)[:0])
            self.assertEqual(QualityPredictor(model).predict_sample(empty)[0].shape,(0,4))

    def test_loss_mask_gradient_and_pair_rank(self):
        cfg=QualityConfig(rank_weight=.1)
        z=torch.tensor([0.,0.,100.,-100.],requires_grad=True)
        data=dict(quality=torch.tensor([.9,.6,0.,1.]),valid=torch.tensor([True,True,False,False]),
                  gt_id=torch.tensor([1,1,-1,-1]),neighbors=torch.tensor([[1],[0],[0],[0]]),neighbor_mask=torch.ones(4,1,dtype=torch.bool))
        loss,qfl,rank=quality_loss(dict(logits=z),data,cfg)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(rank.item(),np.log(2),places=6)
        loss.backward()
        self.assertEqual(z.grad[2:].abs().sum(),0)
        self.assertLess(z.grad[0],z.grad[1])

    def test_real_gradient_path_can_learn_quality_order(self):
        path=self.root/'gt.json'
        self.annotations(path,[1])
        cfg=QualityConfig()
        for local in (False,True):
            torch.manual_seed(42)
            data=prepare(sample(),cfg,local,QualityTargets(path,cfg))
            model=QualityRescorer(14,cfg,local)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
            initial=quality_loss(model(data),data,cfg)[0].item()
            for _ in range(250):
                optimizer.zero_grad()
                loss=quality_loss(model(data),data,cfg)[0]
                loss.backward()
                optimizer.step()
            output=model(data)
            self.assertLess(quality_loss(output,data,cfg)[0].item(),initial*.3)
            self.assertGreater(output['scores'][0].item(),output['scores'][1].item())

    def test_runner_end_to_end_reuses_cache_and_checkpoint(self):
        spec=importlib.util.spec_from_file_location('stage2_runner',Path(__file__).resolve().parents[1]/'tools/kaggle_train_stage2.py')
        runner=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        paths=[]
        manifests={}
        for name,ids in [('train',range(1,7)),('test',range(101,103))]:
            gt=self.root/f'{name}.json'
            self.annotations(gt,ids)
            cache=self.root/f'{name}.pt'
            OfflineDatasetIO.save_samples([sample(str(i)) for i in ids],str(cache))
            manifest=dict(schema='stage1-v1',gt_sha256=runner.sha(gt),checkpoint_sha256='fixture',conf_thresh=.15,
                          intra_nms=.7,roi_dim=0,roi_seed=42,amp=False,torch_version='fixture',torchvision_version='fixture',
                          source_hashes={},max_limit=None)
            Path(str(cache)+'.manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            manifests[cache]=manifest
            paths.extend([f'--{name}-cache',str(cache),f'--{name}-gt',str(gt)])
        before={p:runner.sha(p) for p in self.root.glob('*.pt')}
        runner.main(paths+['--output',str(self.root/'output'),'--epochs','1','--threads','1','--with-ranking'])
        out=self.root/'output/full'
        metrics=json.loads((out/'test_metrics.json').read_text())
        self.assertEqual(len(metrics),10)
        for variant in ('pointwise','local','local_rank'):
            model=load_quality_checkpoint(out/f'{variant}_best.pth')
            boxes,scores,_=QualityPredictor(model).predict_sample(sample())
            self.assertTrue(np.isfinite(scores).all())
            for b in boxes:
                self.assertTrue(np.any(np.all(sample().boxes_global==b,axis=1)))
        self.assertEqual(before,{p:runner.sha(p) for p in self.root.glob('*.pt')})

        # Reproduce an upload containing only the older .pt exports.
        for cache in before:
            runner.cache_manifest_path(cache).unlink()
            samples = OfflineDatasetIO.load_samples(str(cache))
            for cached in samples:
                cached.tile_rects_all = None
                cached.class_ids[-1] = 0
            OfflineDatasetIO.save_samples(samples, str(cache))
        with self.assertRaisesRegex(ValueError, 'Missing stage-one manifest'):
            runner.main(paths+['--output',str(self.root/'output'),'--threads','1'])
        runner.main(paths+['--output',str(self.root/'output'),'--threads','1','--smoke'])
        smoke = self.root/'output/smoke'
        self.assertIsNone(json.loads((smoke/'stage2_run.json').read_text())['train_cache_manifest'])
        metrics = json.loads((smoke/'test_metrics.json').read_text())
        self.assertEqual(len(metrics), 8)
        self.assertNotIn('Split_Recall75', metrics['Greedy-NMS'])

        # A manifest-documented cache can still carry the older export format:
        # placeholder rows and no persisted tile geometry, but consistent patch rects.
        for cache, man in manifests.items():
            samples = OfflineDatasetIO.load_samples(str(cache))
            for cached in samples:
                cached.tile_rects_all = None
                cached.patch_ids = np.zeros(len(cached.boxes_global), np.int64)
                cached.class_ids[-1] = 0
            OfflineDatasetIO.save_samples(samples, str(cache))
            Path(str(cache)+'.manifest.json').write_text(json.dumps(man), encoding='utf-8')
        runner.main(paths+['--output',str(self.root/'output'),'--threads','1','--smoke'])
        run = json.loads((smoke/'stage2_run.json').read_text())
        self.assertEqual(run['train_cache_manifest']['stage2_cache_repairs'],
                         dict(person_rows_removed=6, tile_geometry_rebuilt=6, tile_geometry_unverified=0))
        self.assertEqual(run['test_cache_manifest']['stage2_cache_repairs'],
                         dict(person_rows_removed=2, tile_geometry_rebuilt=2, tile_geometry_unverified=0))
        metrics = json.loads((smoke/'test_metrics.json').read_text())
        self.assertIn('Split_Recall75', metrics['Greedy-NMS'])  # verified rebuild restores subsets


if __name__=='__main__':
    unittest.main()
