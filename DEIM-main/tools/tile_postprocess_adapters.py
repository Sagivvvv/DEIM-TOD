"""Input adapters for upstream post-processing code. The detector is never run."""
import ctypes
import hashlib
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml


TA_COMMIT = '5536f1867feb4d24bc7574b954afdd2f1afada75'
CAR_SHA256 = '6e2eaa2f4c24b903765f33fe210b57281229709559c6c9baefe1c428a8fdc1a0'


def checked_repo(root, name, sha):
    path = root/name
    revision = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != sha:
        raise ValueError(f'Unexpected {name} revision {revision}')
    subprocess.run(['git', '-C', str(path), 'diff', '--quiet', 'HEAD', '--'], check=True)
    return path


def frame_output(rows, columns, score):
    if len(rows) == 0:
        return np.empty((0,4), np.float32), np.empty(0, np.float32)
    frame = pd.DataFrame(rows)
    return frame[columns].to_numpy(dtype=float), frame[score].to_numpy(dtype=float)


def tatm_frame(s, boundary_distance):
    tiles = np.asarray(s.tile_rects_all)
    xorig, yorig = np.unique(tiles[:,0]), np.unique(tiles[:,1])
    rows = []
    for b, sc, cls, tid, rect in zip(s.boxes_global,s.scores,s.class_ids,s.patch_ids,s.patch_rects):
        tx, ty, right, bottom = map(float, rect)
        x1, y1, x2, y2 = map(float, b)
        w, h = right-tx, bottom-ty
        # Keep the shared DEIM geometry, including inferred object extent beyond
        # a source tile. The author's clipped local distance is zero at that edge.
        bd = max(0., boundary_distance(x1-tx,y1-ty,x2-tx,y2-ty,w,h))
        rows.append(dict(image_id=s.image_id, image_path=s.image_id, tile_id=int(tid),
                         tile_x=tx,tile_y=ty,tile_w=w,tile_h=h,
                         grid_row=int(np.searchsorted(yorig,ty)),grid_col=int(np.searchsorted(xorig,tx)),
                         x1=x1,y1=y1,x2=x2,y2=y2,score=float(sc),class_id=int(cls),
                         boundary_distance=round(bd,2)))
    return pd.DataFrame(rows)


def tatm(root, enabled):
    path = checked_repo(root, 'ta_tm', TA_COMMIT)
    sys.path.insert(0, str(path/'src'))
    from topology_merge import tatm_image
    from utils import boundary_distance
    cfg = yaml.safe_load((path/'configs/default.yaml').read_text(encoding='utf8'))
    topo, tile = cfg['topology'], cfg['tiling']
    params = dict(tau=tile['boundary_tau'], lambda_adj=topo['lambda_adj'] if enabled else 0.,
                  mu_edge=topo['mu_edge'], use_edge_continuity=topo['use_edge_continuity'],
                  iou_threshold=tile['iou_threshold'], out_method='tatm')

    def predict(s):
        rows = tatm_image(tatm_frame(s, boundary_distance), **params)
        return frame_output(rows,['x1','y1','x2','y2'],'score_adj')

    return predict, dict(source='mohammad-AJP/PCB-Defect-Detection', commit=TA_COMMIT, **params,
                         input='full-precision shared P0-R boxes and scores; pixel edge distance rounded to 2 decimals')


class CarPredictor:
    def __init__(self, root, enabled):
        import zipfile
        archive = root/'CAR_code.zip'
        if hashlib.sha256(archive.read_bytes()).hexdigest() != CAR_SHA256:
            raise ValueError('Wrong CAR source archive')
        path = root/'car/code'
        with zipfile.ZipFile(archive) as z:
            for member in ('src/car.py','src/fuse.py','src/views.py','src/features.py','configs/frozen_test.yaml'):
                if z.read('code/'+member) != (path/member).read_bytes():
                    raise ValueError('Modified CAR dependency: '+member)
        sys.path.insert(0,str(path))
        from src import car, features, views
        self.car, self.features, self.views = car, features, views
        self.cfg = yaml.safe_load((path/'configs/frozen_test.yaml').read_text(encoding='utf8'))
        self.enabled = enabled
        self.divisor = 1.
        self.params = dict(source='Zenodo 23956349 v1.3.0', archive_sha256=CAR_SHA256,
                           enabled=enabled, lambda_agr=self.cfg['car']['lambda_agr'], features=self.cfg['features'],
                           iou=self.cfg['fuse']['final_nms_iou'], input='actual P0-R tile grid',
                           internal_cap='full candidate count; official evaluator maxDets=200')

    def frame(self, s):
        df = pd.DataFrame(s.boxes_global, columns=['x0','y0','x1','y1'])
        df['score'], df['class_id'], df['image_id'] = s.scores, s.class_ids, s.image_id
        df['view_id'] = s.patch_ids
        return df

    def agreement(self, s, df):
        views = [self.views.View(i,*map(int,r),1.,'tile',int(r[2]-r[0]))
                 for i,r in enumerate(s.tile_rects_all)]
        im = SimpleNamespace(image_id=s.image_id,width=s.img_size[0],height=s.img_size[1])
        with patch.object(self.views, 'make_views', return_value=views):
            agreement, _ = self.features.agr_only(df,[im],self.cfg)
        return agreement

    def raw(self, s):
        df = self.frame(s)
        if self.enabled and len(df):
            agreement = self.agreement(s,df)
            df['score'] = self.car.car_scores(s.scores, agreement, self.cfg['car']['lambda_agr'])
        return df

    def predict_split(self, samples):
        # Use the original split-wide entry point for published metrics, including
        # pandas' tie ordering and the upstream batched torchvision NMS path.
        frames = [self.frame(s) for s in samples]
        df = pd.concat(frames, ignore_index=True)
        cfg = dict(self.cfg, eval=dict(max_dets=max(1,max(len(f) for f in frames))))
        if self.enabled:
            agr = np.concatenate([self.agreement(s,f) for s,f in zip(samples,frames)])
            merged = self.car.apply_car(df,agr,cfg)
        else:
            merged = self.car.baseline_nms(df,cfg)
        by_id = dict(tuple(merged.groupby('image_id',sort=False)))
        return [frame_output(by_id.get(s.image_id,[]),['x0','y0','x1','y1'],'score') for s in samples]

    def prepare(self, samples):
        # apply_car normalizes over its entire input split. Cache the divisor,
        # not a different maximum for each image (which would change global AP).
        if self.enabled:
            self.divisor = max(float(self.raw(s)['score'].max()) for s in samples if len(s.scores))
            self.params['split_max_rescored'] = self.divisor

    def __call__(self, s):
        df = self.raw(s)
        if self.enabled:
            df['score'] = np.clip(df.score.to_numpy()/self.divisor,0.,1.).astype('float32')
        merged = self.car.class_wise_nms(df, self.params['iou'], max(1,len(df)))
        return frame_output(merged,['x0','y0','x1','y1'],'score')


def load_method(name, root):
    if name in ('tatm-off','tatm'):
        return tatm(root, name=='tatm')
    if name in ('car-off','car'):
        method = CarPredictor(root,name=='car')
        return method,method.params
    if name=='cma-third-party':
        cp = CarPredictor(root,False)
        from src.fuse import class_wise_cma_nms
        def run(s):
            df = cp.raw(s)
            df[['view_x0','view_y0','view_x1','view_y1']] = s.patch_rects
            merged = class_wise_cma_nms(df,max(1,len(df)),t=.65,theta=10.,t_mt=.9)
            return frame_output(merged,['x0','y0','x1','y1'],'score')
        return run,dict(source='CAR authors, third-party CMA-NMS',archive_sha256=CAR_SHA256,
                        entry='class_wise_cma_nms',iou=.65,margin_px=10.,ios=.9,
                        input='P0-R tiles only; no DAHI region selection or full-image pass')
    if name in ('wbf','nmw-third-party'):
        commit = '651884c5c7df13a920d286404b47c6fa86944782'
        path = checked_repo(root,'wbf',commit)
        sys.path[:0] = [str(path),str(root/'deps')]
        from ensemble_boxes import weighted_boxes_fusion, non_maximum_weighted
        merge = weighted_boxes_fusion if name=='wbf' else non_maximum_weighted

        def run(s):
            if not len(s.scores):
                return s.boxes_global.copy(), s.scores.copy()
            scale = np.array([*s.img_size,*s.img_size], dtype=np.float64)
            b,sc,_ = merge([s.boxes_global/scale],[s.scores],[s.class_ids],weights=[1.],iou_thr=.55,skip_box_thr=0.)
            return b*scale,sc
        return run, dict(source='ZFTurbo/Weighted-Boxes-Fusion',commit=commit,entry=merge.__name__,
                         iou=.55,weights=[1],skip_box_thr=0,conf_type='avg' if name=='wbf' else 'NMW default')
    if name=='sahi-nmm':
        commit = '68e61981e1f79c56299c208df1d5bac03a01811d'
        path = checked_repo(root,'sahi',commit)
        sys.path[:0] = [str(path),str(root/'deps')]
        from sahi.prediction import ObjectPrediction
        from sahi.postprocess.combine import NMMPostprocess
        nmm = NMMPostprocess(match_threshold=.5, match_metric='IOS', class_agnostic=False)

        def run(s):
            if not len(s.scores):
                return s.boxes_global.copy(), s.scores.copy()
            # SAHI rejects negative endpoints and clips negative origins. Shift
            # the whole coordinate frame together to preserve IOS and unions.
            origin = np.minimum(s.boxes_global[:,:2].min(axis=0),0.).astype(float)
            shift = np.tile(origin,2)
            rows = [ObjectPrediction(bbox=b.tolist(),category_id=1,category_name='person',score=float(sc))
                    for b,sc in zip(s.boxes_global-shift,s.scores)]
            result = nmm(rows)
            return np.asarray([p.bbox.to_xyxy() for p in result])+shift,np.asarray([p.score.value for p in result])
        return run,dict(source='obss/sahi',commit=commit,entry='NMMPostprocess',IOS=.5,class_agnostic=False,
                        geometry='common translation for negative coordinates, inverted after merging')
    if name=='soft-nms':
        commit = 'b8e69bdf8df2ad53025c9d198ded909b50471d4f'
        checked_repo(root,'soft',commit)
        sys.path.insert(0,str(root/'compiled'))
        from original_soft_nms import cpu_soft_nms

        def run(s):
            data = np.column_stack((s.boxes_global,s.scores)).astype('float32')
            data[:,2:4] -= 1.
            indices = cpu_soft_nms(data, sigma=.5, Nt=.3, threshold=.001, method=2)
            output = data[indices]
            output[:,2:4] += 1.
            return output[:,:4],output[:,4]
        return run,dict(source='bharatsingh430/soft-nms',commit=commit,entry='cpu_soft_nms',
                        method=2,sigma=.5,threshold=.001,Nt=.3)
    if name in ('cluster-nms','cluster-diou'):
        import torch
        commit = 'cec038bc1057f0cd532752413b24924fde427f09'
        path = checked_repo(root,'cluster',commit)
        sys.path.insert(0,str(path))
        from utils.detection import detection
        kind = 'cluster_nms' if name=='cluster-nms' else 'cluster_diounms'

        def run(s):
            n = len(s.scores)
            if not n:
                return s.boxes_global.copy(),s.scores.copy()
            # Upstream contains CUDA constants and 21 VOC classes. Provide an
            # empty background and 19 inactive classes; bypass only SSD decoding.
            with torch.inference_mode(), torch.device('cuda'):
                scale = float(max(s.img_size))
                boxes = torch.as_tensor(s.boxes_global,device='cuda')/scale
                conf = torch.zeros((1,n,21))
                conf[0,:,1] = torch.as_tensor(s.scores,device='cuda')
                detector = detection.Detect(21,0,n,.01,.45,[.1,.2],kind,.5)
                with patch.object(detection,'decode',return_value=boxes):
                    result = detector.forward(torch.zeros((1,n,4)),conf,torch.zeros((n,4)))[0,1]
                result = result[result[:,0]>0].cpu().numpy()
            return result[:,1:]*scale,result[:,0]
        return run,dict(source='Zzh-tju/DIoU-SSD-pytorch',commit=commit,entry=kind,device='CUDA',
                        iou=.45,beta=.5,top_k='all',input_score_floor=.01,active_class=1,
                        geometry='uniform scaling by longer image edge')
    if name=='diou-nms':
        return darknet_diou(root)
    raise ValueError('Unknown method: '+name)


def darknet_diou(root):
    commit = '257ba175cb626e6d2b10da25f49332c71696550a'
    checked_repo(root,'diou',commit)
    class Box(ctypes.Structure):
        _fields_ = [(k,ctypes.c_float) for k in ('x','y','w','h')]
    float_ptr = ctypes.POINTER(ctypes.c_float)
    class Detection(ctypes.Structure):
        _fields_ = [('bbox',Box),('classes',ctypes.c_int),('prob',float_ptr),('mask',float_ptr),
                    ('objectness',ctypes.c_float),('sort_class',ctypes.c_int)]
    lib = ctypes.CDLL(str(root/'compiled'/('darknet_box.dll' if os.name=='nt' else 'libdarknet_box.so')))
    suppress = lib.diounms_sort
    suppress.argtypes = [ctypes.POINTER(Detection),ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_char_p,ctypes.c_float]
    suppress.restype = None

    def run(s):
        scores = np.array(s.scores, dtype='float32',copy=True)
        dets = (Detection*len(scores))()
        for i,(x1,y1,x2,y2) in enumerate(s.boxes_global):
            dets[i] = Detection(Box((x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1),1,
                                scores[i:i+1].ctypes.data_as(float_ptr),None,1.,0)
        suppress(dets,len(dets),1,.5,b'diounms',1.)
        # qsort permutes structs, but each prob still points into the original
        # candidate array. Follow the author's sorted order for tied outputs.
        keep = [(ctypes.addressof(d.prob.contents)-scores.ctypes.data)//scores.itemsize
                for d in dets if d.prob[0]>0]
        return s.boxes_global[keep],s.scores[keep]
    return run,dict(source='Zzh-tju/DIoU-darknet',commit=commit,entry='diounms_sort',kind='diounms',iou=.5,beta=1.)
