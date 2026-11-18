"""Official output-only evaluation and reproducible stage-one diagnostics."""
import contextlib
import copy
import importlib
import io
import json
import sys
from pathlib import Path

import numpy as np


def official_classes():
    # DEIM may install faster_coco_eval aliases. The official subclass needs native COCOeval.
    mod = sys.modules.get('pycocotools.cocoeval')
    if mod is not None and 'faster_coco_eval' in str(getattr(mod, '__file__', '')):
        for name in list(sys.modules):
            if name == 'pycocotools' or name.startswith('pycocotools.') or name.startswith('huicv.evaluation.expand_cocofmt_eval'):
                del sys.modules[name]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
    return (importlib.import_module('pycocotools.coco').COCO,
            importlib.import_module('huicv.evaluation.expand_cocofmt_eval').COCOExpandEval)


class OfficialTinyEvaluator:
    """Consumes final global boxes; never merges or runs NMS a second time.

    AP = all-area AP50:95 (auxiliary). Tiny_AP50 is the checkpoint criterion.
    Official AP25/50/75 are reported for both all and tiny areas, maxDets=200.
    """
    def __init__(self, annotation_path, *, include_split=True):
        with open(annotation_path, encoding='utf-8') as f:
            self.data = json.load(f)
        self.images = {int(x['id']): x for x in self.data['images']}
        self.include_split = include_split

    def __call__(self, samples, boxes, scores, latencies_ms=None):
        COCO, Eval = official_classes()
        ids = [int(s.image_id) for s in samples]
        if len(ids) != len(set(ids)) or not set(ids) <= self.images.keys():
            raise ValueError('Evaluation requires unique original-image IDs present in GT')
        if not (len(ids) == len(boxes) == len(scores)):
            raise ValueError('Prediction/sample lengths differ')
        results = []
        for iid, bb, ss in zip(ids, boxes, scores):
            if len(bb) != len(ss) or not np.isfinite(bb).all() or not np.isfinite(ss).all():
                raise ValueError('Invalid detection arrays')
            for b, score in zip(bb, ss):
                x, y, x2, y2 = map(float, b)
                if x2 > x and y2 > y:
                    results.append(dict(image_id=iid, category_id=1, bbox=[x,y,x2-x,y2-y], score=float(score)))
        thresholds = np.r_[.25, np.linspace(.5, .95, 10)]
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO()
            gt.dataset = copy.deepcopy(self.data)
            gt.createIndex()
            if results:
                dt = gt.loadRes(results)
            else:
                dt = COCO()
                dt.dataset = dict(images=copy.deepcopy(self.data['images']), categories=copy.deepcopy(self.data['categories']), annotations=[])
                dt.createIndex()
            ev = Eval(gt, dt, 'bbox', ignore_uncertain=True, use_ignore_attr=True,
                      use_iod_for_ignore=True, iod_th_of_iou_f='lambda iou: (2*iou)/(1+iou)',
                      cocofmt_param=dict(evaluate_standard='tiny', iouThrs=thresholds.tolist(), maxDets=[200]))
            ev.params.imgIds = ids
            ev.params.catIds = [1]
            ev.evaluate()
            ev.accumulate()
        p = ev.eval['precision']
        def ap(ts, area):
            a = ev.params.areaRngLbl.index(area)
            v = p[np.asarray(ts), :, :, a, 0]
            v = v[v >= 0]
            return float(v.mean()*100) if v.size else float('nan')
        out = {'AP': ap(range(1,11), 'all')}
        for area, prefix in [('all',''), ('tiny','Tiny_')]:
            for t in [.25,.5,.75]:
                out[f'{prefix}AP{int(t*100)}'] = ap([int(np.flatnonzero(np.isclose(thresholds,t))[0])], area)
        # Subsets use fixed GT geometry, and hits come from official one-to-one matches.
        nested, split = set(), set()
        by_image = {i: [] for i in ids}
        for a in self.data['annotations']:
            if a['image_id'] in by_image and a['category_id'] == 1 and not any(a.get(k,0) for k in ('ignore','uncertain','iscrowd')):
                by_image[a['image_id']].append(a)
        for s in samples:
            anns = by_image[int(s.image_id)]
            g = np.array([a['bbox'] for a in anns], dtype=float).reshape(-1,4)
            g[:,2:] += g[:,:2]
            areas = np.prod(g[:,2:]-g[:,:2],axis=1)
            tiles = getattr(s, 'tile_rects_all', None)
            if self.include_split and tiles is None:
                raise ValueError('Missing full tile geometry; rebuild stage-one cache')
            for k, a in enumerate(anns):
                inter = np.prod(np.maximum(0, np.minimum(g[k,2:],g[:,2:])-np.maximum(g[k,:2],g[:,:2])),axis=1)
                if np.any((areas > areas[k]) & (inter / max(areas[k],1e-8) > .7)):
                    nested.add(a['id'])
                if self.include_split:
                    vis = np.prod(np.maximum(0,np.minimum(g[k,2:],tiles[:,2:])-np.maximum(g[k,:2],tiles[:,:2])),axis=1)
                    # Visible in >=2 tiles, with at least one genuinely truncated view.
                    if np.sum(vis > 0) >= 2 and np.any((vis > 0) & (vis < areas[k]-1e-5)):
                        split.add(a['id'])
        subsets = [('Nested_Recall50', nested, .5)]
        if self.include_split:
            subsets.append(('Split_Recall75', split, .75))
        for name, subset, t in subsets:
            hits = set()
            ti = int(np.flatnonzero(np.isclose(thresholds,t))[0])
            for e in ev.evalImgs:
                if e is not None and e['aRng'] == ev.params.areaRng[0]:
                    hits.update(int(x) for x in e['dtMatches'][ti][~e['dtIgnore'][ti].astype(bool)] if x > 0)
            out[name] = 100*len(hits & subset)/len(subset) if subset else float('nan')
            out[name+'_count'] = len(subset)
        out['Latency_ms'] = float(np.mean(latencies_ms)) if latencies_ms else 0.
        return out


def split_train_validation(samples, annotation_path, fraction=.2, seed=42):
    with open(annotation_path, encoding='utf-8') as f:
        images = {str(x['id']): x for x in json.load(f)['images']}
    groups = {}
    for s in samples:
        im = images[s.image_id]
        # Explicit sequence metadata when available; otherwise group by original image.
        key = str(im.get('video_id', im.get('sequence_id', 'image:'+s.image_id)))
        groups.setdefault(key, []).append(s)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError('At least two independent groups required for validation')
    np.random.default_rng(seed).shuffle(keys)
    nval = min(len(keys)-1, max(1, round(len(keys)*fraction)))
    valkeys = set(keys[:nval])
    train = [s for k in keys if k not in valkeys for s in groups[k]]
    val = [s for k in keys if k in valkeys for s in groups[k]]
    return train, val, dict(seed=seed, validation_fraction=fraction, groups={k:[s.image_id for s in v] for k,v in groups.items()},
                           train_ids=[s.image_id for s in train], val_ids=[s.image_id for s in val])


def regression_diagnostics(outputs, sample):
    mask = sample.is_leader.astype(bool)
    if not mask.any():
        return {}
    g = sample.target_reg[mask]
    def iou(b):
        inter = np.prod(np.maximum(0,np.minimum(b[:,2:],g[:,2:])-np.maximum(b[:,:2],g[:,:2])),axis=1)
        area = lambda x: np.prod(np.maximum(0,x[:,2:]-x[:,:2]),axis=1)
        return inter/np.maximum(area(b)+area(g)-inter,1e-7)
    raw = iou(sample.boxes_global[mask])
    env = iou(outputs['env_boxes'].detach().cpu().numpy()[mask])
    refined = iou(outputs['pred_boxes'].detach().cpu().numpy()[mask])
    return dict(raw_iou=float(raw.mean()), env_iou=float(env.mean()), refined_iou=float(refined.mean()),
                improved_fraction=float(np.mean(refined > raw+1e-4)), degraded_fraction=float(np.mean(refined < raw-1e-4)),
                clamp_saturation=float((outputs['raw_deltas'].detach().abs() >= .35).float().mean().item()))
