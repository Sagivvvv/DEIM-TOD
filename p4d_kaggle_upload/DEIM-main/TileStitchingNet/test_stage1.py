"""Run: python -m unittest TileStitchingNet.test_stage1 -v"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .stage1 import OfficialTinyEvaluator, split_train_validation
from .trainEval import compute_ap_at_iou, train_tile_stitching_net
from .datasetBuilder import SyntheticTiledDatasetGenerator, CroppedGTLabelAssigner
from .pipeline_deim_tilestitching import OfflineDatasetIO


class StageOneTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'gt.json'
        self.data = dict(images=[dict(id=i, width=100, height=100, file_name=f'{i}.jpg') for i in range(1,5)],
                         categories=[dict(id=1,name='person')], annotations=[
                             dict(id=1,image_id=1,category_id=1,bbox=[0,0,10,10],area=100,iscrowd=0),
                             dict(id=2,image_id=1,category_id=1,bbox=[3,0,10,10],area=100,iscrowd=0),
                             dict(id=3,image_id=1,category_id=1,bbox=[50,50,10,10],area=100,iscrowd=0,uncertain=1)])
        self.path.write_text(json.dumps(self.data),encoding='utf-8')
        self.s = SimpleNamespace(image_id='1',tile_rects_all=np.array([[0,0,8,100],[5,0,100,100]]))

    def test_official_matching_ignore_empty_and_cap(self):
        ev = OfficialTinyEvaluator(self.path)
        b = np.array([[50,50,60,60],[0,0,10,10],[1,0,11,10]],dtype=np.float32)
        r = ev([self.s],[b],[np.array([1,.99,.9])])
        self.assertAlmostEqual(r['AP50'],100)
        self.assertEqual(r['Split_Recall75_count'],2)
        self.assertAlmostEqual(r['Split_Recall75'],50)
        r = ev([self.s],[b[:0]],[np.array([])])
        self.assertEqual(r['AP50'],0)
        # 200 high-scoring unmatched boxes exhaust the evaluation budget.
        many = np.concatenate([np.tile([80,80,90,90],(200,1)),b[1:]])
        r = ev([self.s],[many],[np.r_[np.ones(200),.2,.1]])
        self.assertEqual(r['AP50'],0)
        ap, rec = compute_ap_at_iou([b[1:]],[np.array([.99,.9])],[np.array([[0,0,10,10],[3,0,13,10]])])
        self.assertAlmostEqual(ap,100)
        self.assertEqual(rec,100)

    def test_group_split_reproducible(self):
        self.data['images'][0]['video_id']='a'
        self.data['images'][1]['video_id']='a'
        self.path.write_text(json.dumps(self.data),encoding='utf-8')
        samples=[SimpleNamespace(image_id=str(i)) for i in range(1,5)]
        a,b,m=split_train_validation(samples,self.path)
        self.assertFalse(set(m['train_ids']) & set(m['val_ids']))
        self.assertEqual('1' in m['val_ids'],'2' in m['val_ids'])
        self.assertEqual(m,split_train_validation(samples,self.path)[2])

    def test_accumulation_empty_tail_and_cache_geometry(self):
        s=SyntheticTiledDatasetGenerator(seed=42).generate_image_sample('1',CroppedGTLabelAssigner())
        s.tile_rects_all=np.array([[0,0,640,576]],dtype=np.float32)
        cache=str(Path(self.tmp.name)/'features.pt')
        OfflineDatasetIO.save_samples([s],cache)
        np.testing.assert_array_equal(OfflineDatasetIO.load_samples(cache)[0].tile_rects_all,s.tile_rects_all)
        empty=copy.deepcopy(s)
        empty.boxes_global=empty.boxes_global[:0]
        kwargs=dict(epochs=1,batch_size=100000,save_best_path=None,verbose=False)
        a,ha=train_tile_stitching_net([s],**kwargs)
        b,hb=train_tile_stitching_net([s,s,empty],**kwargs)
        self.assertEqual(hb[0]['optimizer_steps'],1)
        self.assertGreater(hb[0]['reg_head_grad_norm'],0)
        self.assertGreater(hb[0]['reg_head_update_norm'],0)
        for k,v in a.state_dict().items():
            torch.testing.assert_close(v,b.state_dict()[k],atol=1e-6,rtol=1e-5)

    def test_training_official_callback_checkpoint(self):
        s=SyntheticTiledDatasetGenerator(seed=42).generate_image_sample('1',CroppedGTLabelAssigner())
        s.tile_rects_all=np.array([[0,0,640,576],[640,0,1280,576]],dtype=np.float32)
        checkpoint=str(Path(self.tmp.name)/'best.pth')
        _,history=train_tile_stitching_net([s], [s], epochs=1, save_best_path=checkpoint,
                                         evaluator=OfficialTinyEvaluator(self.path), verbose=False)
        self.assertTrue(Path(checkpoint).is_file())
        self.assertEqual(history[0]['selection_metric'],'Tiny_AP50')
        self.assertIn('AP25',history[0]['validation_metrics'])


if __name__ == '__main__':
    unittest.main()
