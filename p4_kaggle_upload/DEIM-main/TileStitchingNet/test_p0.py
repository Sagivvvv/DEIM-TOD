import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
import torch
from .test_stage2 import sample

spec=importlib.util.spec_from_file_location('audit_p0',Path(__file__).resolve().parents[1]/'tools/kaggle_audit_p0.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class P0Tests(unittest.TestCase):
    def test_order_and_content_fingerprints(self):
        a=sample(); b=copy.deepcopy(a)
        for key in audit.FIELDS:
            v=getattr(b,key,None)
            if v is not None: setattr(b,key,v[::-1].copy())
        diff=audit.compare_fingerprints(audit.fingerprints([a]),audit.fingerprints([b]))
        self.assertEqual(diff['ordered_changed'],['1'])
        self.assertEqual(diff['candidate_content_changed'],[])
        b.scores[0]+=.01
        self.assertEqual(audit.compare_fingerprints(audit.fingerprints([a]),audit.fingerprints([b]))['candidate_content_changed'],['1'])

    def test_complete_audit_and_error_package(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); stage2=root/'stage2'; stage2.mkdir()
            argv=[]; ss={}; metas={}; gts={}
            for split,ids in [('train',range(1,7)),('test',range(101,103))]:
                gt=root/f'{split}.json'
                data=dict(images=[dict(id=i,width=100,height=100,file_name=f'{i}.jpg') for i in ids],categories=[dict(id=1,name='person')],
                          annotations=[dict(id=i,image_id=i,category_id=1,bbox=[0,0,10,10],area=100,iscrowd=0) for i in ids])
                gt.write_text(json.dumps(data),encoding='utf-8')
                ss[split]=[sample(str(i)) for i in ids]; gts[split]=gt
                cache=root/f'{split}.pt'
                audit.runner.OfflineDatasetIO.save_samples(ss[split],str(cache))
                meta=dict(schema='stage1-v1',gt_sha256=audit.runner.sha(gt),checkpoint_sha256='test',conf_thresh=.15,
                          intra_nms=.7,roi_dim=0,roi_seed=42,amp=False,torch_version='test',torchvision_version='test',source_hashes={},max_limit=None)
                metas[split]=meta
                Path(str(cache)+'.manifest.json').write_text(json.dumps(meta),encoding='utf-8')
                argv+=['--'+split+'-cache',str(cache),'--'+split+'-gt',str(gt)]
            _,val,split=audit.split_train_validation(ss['train'],gts['train'])
            audit.write(stage2/'stage2_split.json',split)
            cfg=audit.QualityConfig()
            for name,local in [('pointwise',False),('local',True)]:
                m=audit.QualityRescorer(14,cfg,local)
                metrics=audit.evaluate(val,audit.QualityPredictor(m),audit.OfficialTinyEvaluator(gts['train']),root,name)
                from dataclasses import asdict
                torch.save(dict(model=m.state_dict(),config=asdict(cfg),feature_dim=14,local=local,epoch=0,validation=metrics),stage2/f'{name}_best.pth')
            out=root/'audit'
            audit.main(argv+['--stage2-dir',str(stage2),'--output',str(out),'--threads','1'])
            report=json.loads((out/'audit_summary.json').read_text())
            self.assertEqual(report['status'],'COMPLETED_REVIEW_PROVENANCE')
            self.assertTrue(all(report['cache_files_unchanged'].values()))
            self.assertFalse(any(report['identity_checks'].values()))
            self.assertEqual(len(report['metrics']['test']),8)
            with zipfile.ZipFile(out/'p0_reports.zip') as z:
                self.assertIn('audit_summary.json',z.namelist())
            with self.assertRaises(SystemExit):
                audit.main(['--train-cache',str(root/'missing.pt'),'--output',str(root/'error')])
            self.assertEqual(json.loads((root/'error/audit_summary.json').read_text())['status'],'ERROR')
            self.assertTrue((root/'error/p0_reports.zip').exists())


if __name__=='__main__': unittest.main()
