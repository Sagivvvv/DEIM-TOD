import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import asdict
import numpy as np
from PIL import Image
import torch
from torch import nn
from .p0r import SharedProjection,extract_roi,to_network_boxes,replace_roi,frozen_fields
from .test_stage2 import sample

spec=importlib.util.spec_from_file_location('p0r_runner',Path(__file__).resolve().parents[1]/'tools/kaggle_p0r_train.py')
runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


class FakeDetector(nn.Module):
    def backbone(self,x):
        return torch.nn.functional.avg_pool2d(x.mean(1,keepdim=True),8).expand(-1,256,-1,-1)
    def encoder(self,x): return [x]


class P0RTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_crop_coordinate_inverse(self):
        b=np.array([[110,220,310,420]],np.float32)
        rect=np.array([100,200,740,744],np.float32)
        local=to_network_boxes(b,rect)
        restored=local*np.array([1,544/576,1,544/576])+np.array([100,200,100,200])
        np.testing.assert_allclose(restored,b,atol=1e-5)

    def test_roi_refresh_identity_repeat_and_empty(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'image.png'
            Image.fromarray(np.full((100,100,3),120,np.uint8)).save(path)
            s=sample(); before=frozen_fields(s)
            torch.manual_seed(42); projection=SharedProjection().eval()
            a=extract_roi(s,path,FakeDetector(),projection,batch_tiles=1)
            b=extract_roi(s,path,FakeDetector(),projection,batch_tiles=2)
            np.testing.assert_allclose(a,b,atol=1e-5)
            replace_roi(s,a)
            self.assertEqual(before,frozen_fields(s))
            self.assertEqual(s.node_features.shape,(4,76))
            s.scores=s.scores[:0]
            self.assertEqual(extract_roi(s,path,FakeDetector(),projection).shape,(0,64))

    def test_combined_smoke_training_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); argv=[]; reference=dict(arguments=dict(smoke=False),evaluated_ids={},metrics={})
            train_samples=None; train_gt=None
            for split,ids in [('train',range(1,7)),('test',range(101,103))]:
                (root/split).mkdir()
                anns=[]; images=[]; ss=[]
                for i in ids:
                    name=f'{i}.png'; Image.fromarray(np.full((100,100,3),100+i%50,np.uint8)).save(root/split/name)
                    images.append(dict(id=i,width=100,height=100,file_name=name))
                    anns.append(dict(id=i,image_id=i,category_id=1,bbox=[0,0,10,10],area=100,iscrowd=0))
                    ss.append(sample(str(i)))
                gt=root/(split+'.json'); gt.write_text(json.dumps(dict(images=images,annotations=anns,categories=[dict(id=1,name='person')])) )
                cache=root/(split+'.pt'); runner.audit.runner.OfflineDatasetIO.save_samples(ss,str(cache))
                reference[split]=dict(cache_sha256=runner.audit.runner.sha(cache),gt_sha256=runner.audit.runner.sha(gt))
                argv+=['--'+split+'-cache',str(cache),'--'+split+'-gt',str(gt)]
                if split=='train': train_samples,train_gt=ss,gt
                else: reference['evaluated_ids']['test']=[s.image_id for s in ss]
            _,val,_=runner.audit.split_train_validation(train_samples,train_gt)
            reference['evaluated_ids']['validation']=[s.image_id for s in val]
            report_dir=root/'p0_reports'; report_dir.mkdir()
            ref=report_dir/'audit_summary.json'; ref.write_text(json.dumps(reference))
            checkpoint=root/'detector.pth'; checkpoint.write_bytes(b'test fixture only')
            out=root/'output'
            args=argv+['--checkpoint',str(checkpoint),'--image-root',str(root),'--p0-report',str(report_dir),'--extract-device','cpu','--threads','1','--output',str(out),'--smoke']
            with patch.object(runner,'load_detector',return_value=FakeDetector()):
                runner.main(args)
                report=json.loads((out/'p0r_summary.json').read_text())
                self.assertEqual(report['status'],'SMOKE_COMPLETED')
                self.assertTrue(all(report['parent_cache_files_unchanged'].values()))
                self.assertEqual(len(report['metrics']['test']),8)
                self.assertTrue((out/'p0r_reports.zip').exists())
                runner.main(args+['--resume'])
                report=json.loads((out/'p0r_summary.json').read_text())
                self.assertTrue(all(r['reused'] for r in report['train_refresh']))
                self.assertTrue((out/'pointwise_snapshots/last.pth').exists())
                # Exercise full-mode baseline gate on small complete synthetic datasets.
                for split,ss,gt in [('validation',val,train_gt),('test',[sample('101'),sample('102')],root/'test.json')]:
                    evaluator=runner.audit.OfficialTinyEvaluator(gt)
                    fns={'Greedy-NMS':lambda b,s:runner.audit.runner.run_greedy_nms(b,s,.5,score_thresh=0),
                         'Soft-NMS':lambda b,s:runner.audit.runner.run_soft_nms(b,s,sigma=.5,score_thresh=0),
                         'WBF':lambda b,s:runner.audit.runner.run_wbf(b,s,.55,score_thresh=0),
                         'SAHI-NMM':lambda b,s:runner.audit.runner.run_sahi_nmm_ios(b,s,.5,score_thresh=0)}
                    reference['metrics'][split]={n:runner.evaluate_predictor(ss,runner.audit.runner.Baseline(fn),evaluator) for n,fn in fns.items()}
                ref.write_text(json.dumps(reference))
                full_args=argv+['--checkpoint',str(checkpoint),'--image-root',str(root),'--p0-report',str(ref),'--extract-device','cpu','--threads','1','--epochs','1','--output',str(root/'full')]
                runner.main(full_args)
                report=json.loads((root/'full/p0r_summary.json').read_text())
                self.assertEqual(report['baseline_gate'],'PASSED_SAME_CANDIDATES_SAME_METRICS')
                self.assertEqual(report['status'],'REBUILD_AND_TRAIN_COMPLETED')


if __name__=='__main__': unittest.main()
