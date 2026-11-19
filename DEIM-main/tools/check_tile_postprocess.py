"""Cross-check the two tiled adapters with the authors' original entry points."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from tile_postprocess_adapters import CarPredictor, tatm, tatm_frame


def scene(image_id, rects, boxes, scores, ids):
    tiles = np.asarray(rects, dtype=np.float32)
    indices = np.asarray(ids, dtype=int)
    return SimpleNamespace(image_id=image_id, img_size=(200,100), tile_rects_all=tiles,
                           boxes_global=np.asarray(boxes,dtype=np.float32).reshape(-1,4),
                           scores=np.asarray(scores,dtype=np.float32), class_ids=np.ones(len(ids),dtype=int),
                           patch_ids=indices, patch_rects=tiles[indices])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, default=Path(__file__).resolve().parents[1]/'output/postprocess_sources')
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    tiles = [[0,0,110,100],[90,0,200,100]]
    samples = [
        scene('a',tiles,[[95,15,100,25],[95,15,100,25],[109,68,117,75],[40,80,46,90]],
              [.65,.55,.3,.65],[0,1,0,0]),
        scene('b',tiles,[[131,50,143,70],[137,50,149,70],[100,10,105,20]], [.9,.9,.48],[1,1,0]),
        scene('empty',tiles,[],[],[]),
    ]
    cfg_file = args.work/'configs/default.yaml'
    cfg_file.parent.mkdir(parents=True,exist_ok=True)
    pred_dir = args.work/'outputs/predictions'
    pred_dir.mkdir(parents=True,exist_ok=True)
    source = args.sources/'ta_tm'
    cfg_file.write_bytes((source/'configs/default.yaml').read_bytes())
    outcomes = {}
    for enabled in (False,True):
        f,cfg = tatm(args.sources,enabled)
        from utils import boundary_distance
        raw_csv = args.work/'raw.csv'
        pd.concat([tatm_frame(s,boundary_distance) for s in samples], ignore_index=True).to_csv(raw_csv,index=False)
        subprocess.run([sys.executable,'-X','utf8',str(source/'src/topology_merge.py'),'--config',str(cfg_file),
                        '--pred',str(raw_csv),'--lambda',str(cfg['lambda_adj'])],check=True)
        golden = pd.read_csv(pred_dir/'tatm_raw.csv')
        for s in samples:
            rows = golden[golden.image_id==s.image_id]
            b,sc = f(s)
            np.testing.assert_allclose(b,rows[['x1','y1','x2','y2']],rtol=0,atol=1e-7)
            np.testing.assert_allclose(sc,rows.score_adj,rtol=0,atol=1e-7)
        outcomes['TA-TM '+str(enabled)] = 'passed author CLI comparison'

    for enabled in (False,True):
        cp = CarPredictor(args.sources,enabled)
        cp.prepare(samples)
        raw, agreements = [], []
        for s in samples:
            df = pd.DataFrame(s.boxes_global,columns=['x0','y0','x1','y1'])
            df['image_id'],df['score'],df['view_id'],df['class_id'] = s.image_id,s.scores,s.patch_ids,s.class_ids
            raw.append(df)
            views = [cp.views.View(i,*map(int,r),1.,'tile',int(r[2]-r[0])) for i,r in enumerate(s.tile_rects_all)]
            im = SimpleNamespace(image_id=s.image_id,width=200,height=100)
            with patch.object(cp.views,'make_views',return_value=views):
                a, ncov = cp.features.agr_only(df,[im],cp.cfg)
            if s.image_id=='a':
                assert ncov[0]==2 and a[0]==1 and ncov[-1]==1
            if s.image_id=='b':
                assert ncov[-1]==2 and a[-1]==.5
            agreements.extend(a)
        df = pd.concat(raw,ignore_index=True)
        golden = cp.car.apply_car(df,np.asarray(agreements),cp.cfg) if enabled else cp.car.baseline_nms(df,cp.cfg)
        actual_split = cp.predict_split(samples)
        for s in samples:
            b,sc = cp(s)
            rows = golden[golden.image_id==s.image_id]
            # Global pandas sorting may permute ties across images. Compare
            # membership and adjusted scores; per-image sorting is used in both benchmark arms.
            assert len(b) == len(rows)
            def ordered(bb, ss):
                return sorted(tuple(r)+ (float(v),) for r,v in zip(bb,ss))
            np.testing.assert_allclose(ordered(b,sc),ordered(rows[['x0','y0','x1','y1']].to_numpy(),rows.score),rtol=0,atol=1e-7)
        for s,(b,sc) in zip(samples,actual_split):
            expected = golden[golden.image_id==s.image_id]
            np.testing.assert_array_equal(b,expected[['x0','y0','x1','y1']].to_numpy())
            np.testing.assert_array_equal(sc,expected.score.to_numpy())
        outcomes['CAR '+str(enabled)] = 'passed author split-level comparison'
    (args.work/'checks.json').write_text(json.dumps(outcomes,indent=2),encoding='utf8')
    print(outcomes)


if __name__ == '__main__':
    main()
