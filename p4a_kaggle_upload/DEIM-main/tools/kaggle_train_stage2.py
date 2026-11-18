"""Stage two: reuse stage-one caches, train quality scorers, keep boxes fixed.

Run from any directory: python tools/kaggle_train_stage2.py --help
Only load trusted local .pt caches produced by your stage-one run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

# Also supports pasting this file into a fresh Kaggle Notebook cell.
REPO_SRC = '/kaggle/input/datasets/xyjxyjxyj55/deim-tod/DEIM-main'
ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path(REPO_SRC)
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from TileStitchingNet.pipeline_deim_tilestitching import OfflineDatasetIO
from TileStitchingNet.stage1 import OfficialTinyEvaluator, split_train_validation
from TileStitchingNet.stage2 import QualityConfig, QualityPredictor, train_quality, evaluate_predictor
from TileStitchingNet.trainEval import run_greedy_nms, run_soft_nms, run_wbf, run_sahi_nmm_ios


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def cache_manifest_path(path):
    return path.with_name(path.name + '.manifest.json')


# Stage-one extraction grid; tile geometry is deterministic given the image size.
TILE_W, TILE_H, TILE_OVERLAP = 640, 576, 50


def reconstruct_tile_geometry(sample):
    """Rebuild the stage-one tile grid for caches saved before tile geometry persisted.

    A rebuild is accepted only when every stored candidate patch rectangle equals
    the grid rectangle indexed by its patch ID; otherwise None keeps Split Recall
    disabled instead of guessing wrong geometry.
    """
    from engine.data.dataset.tinyperson_tiling import generate_tile_corners
    corners = np.array(generate_tile_corners(*sample.img_size, tile_w=TILE_W, tile_h=TILE_H,
                                             overlap=TILE_OVERLAP), dtype=np.float32)
    ids = sample.patch_ids
    if len(ids) and (ids.min() < 0 or ids.max() >= len(corners)
                     or not np.array_equal(sample.patch_rects, corners[ids])):
        return None
    return corners


def resolve_file(explicit, filename, *, cache=False):
    if explicit:
        p=Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    preferred=Path('/kaggle/working')/filename
    if preferred.is_file() and (not cache or cache_manifest_path(preferred).is_file()):
        return preferred
    found=[]
    for base in [Path('/kaggle/input'),ROOT/'datasets',ROOT.parent/'datasets']:
        if base.exists():
            found.extend(base.rglob(filename))
    found=sorted(set(p.resolve() for p in found))
    if cache:
        documented = [path for path in found if cache_manifest_path(path).is_file()]
        if documented:
            found = documented
        elif preferred.is_file():
            return preferred
    if len(found)!=1:
        raise ValueError(f'Expected one {filename}; found {len(found)}. Set its explicit CLI path: {found}')
    return found[0]


def load_checked_cache(path, annotations, *, smoke=False):
    sidecar = cache_manifest_path(path)
    manifest = None
    if sidecar.is_file():
        manifest = json.loads(sidecar.read_text(encoding='utf-8'))
        if manifest.get('schema') != 'stage1-v1' or manifest.get('gt_sha256') != sha(annotations):
            raise ValueError(f'Cache schema or GT hash mismatch: {path}')
    elif smoke:
        print(f'[smoke] Cache has no manifest: {path}. Extraction provenance is unverified.', flush=True)
    else:
        raise ValueError(f'Missing stage-one manifest: {sidecar}. For a full run, use the matching '
                         'manifest from the stage-one export alongside this .pt. '
                         'Use --smoke to check a cache without a manifest.')
    samples=OfflineDatasetIO.load_samples(str(path))
    with open(annotations,encoding='utf-8') as f:
        images={str(i['id']):i for i in json.load(f)['images']}
    ids=[s.image_id for s in samples]
    if not ids or len(ids)!=len(set(ids)) or not set(ids)<=images.keys():
        raise ValueError('Cache has duplicate/unknown original image IDs')
    if manifest is not None and not manifest.get('max_limit') and set(ids)!=images.keys():
        raise ValueError('Full cache is missing annotated original images; do not silently evaluate a subset')
    repaired = dict(person_rows_removed=0, tile_geometry_rebuilt=0, tile_geometry_unverified=0)
    for s in samples:
        im=images[s.image_id]
        if tuple(s.img_size)!=(im['width'],im['height']):
            raise ValueError(f'Image size mismatch: {s.image_id}')
        # Old exports may contain the placeholder class, with or without a manifest.
        # Keep only person rows in the candidate arrays consumed by stage two.
        person = s.class_ids == 1
        if not person.all():
            for field in ('boxes_global','scores','class_ids','patch_ids','patch_rects',
                          'boundary_dists','is_boundary_risk','node_features','roi_features'):
                value = getattr(s, field)
                if value is not None:
                    setattr(s, field, value[person])
            repaired['person_rows_removed'] += int((~person).sum())
        # Old exports may predate persisted tile geometry; rebuild the deterministic
        # extraction grid when the stored patch rectangles confirm it.
        if getattr(s,'tile_rects_all',None) is None:
            tiles = reconstruct_tile_geometry(s)
            if tiles is None:
                repaired['tile_geometry_unverified'] += 1
            else:
                s.tile_rects_all = tiles
                repaired['tile_geometry_rebuilt'] += 1
        if not np.isfinite(s.node_features).all() or not np.isfinite(s.boxes_global).all() or not np.isfinite(s.scores).all():
            raise ValueError(f'Nonfinite cached candidates: {s.image_id}')
        if np.any(s.scores < 0) or np.any(s.scores > 1) or np.any(s.boxes_global[:,2:] <= s.boxes_global[:,:2]):
            raise ValueError(f'Invalid probability or box geometry: {s.image_id}')
        # Stage two rebuilds supervision from original GT; old leader/edge labels are unused.
        s.target_edge=None
    if any(repaired.values()):
        print(f'[cache repair] {path.name}: {repaired}', flush=True)
        if manifest is not None:
            manifest = {**manifest, 'stage2_cache_repairs': repaired}
    return samples,manifest


class Baseline:
    def __init__(self,fn):
        self.fn=fn
    def predict_sample(self,s):
        start=time.perf_counter()
        b,sc=self.fn(s.boxes_global,s.scores)
        return b,sc,(time.perf_counter()-start)*1000


def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('train-cache','test-cache','train-gt','test-gt','split-manifest'):
        parser.add_argument('--'+flag)
    parser.add_argument('--output',default='/kaggle/working/stage2')
    parser.add_argument('--epochs',type=int,default=18)
    parser.add_argument('--val-freq',type=int,default=1)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--with-ranking',action='store_true',help='Additional local+pair-ranking ablation; defaults off')
    parser.add_argument('--smoke',action='store_true',help='Two epochs on 8 train/4 val/4 test images, separate smoke output')
    args=parser.parse_args(argv)
    if args.epochs<1 or args.val_freq<1 or args.threads<1:
        parser.error('epochs, val-freq and threads must be positive')
    torch.set_num_threads(args.threads)
    out=Path(args.output)/('smoke' if args.smoke else 'full')
    out.mkdir(parents=True,exist_ok=True)
    train_gt=resolve_file(args.train_gt,'tiny_set_train_all.json')
    test_gt=resolve_file(args.test_gt,'tiny_set_test_all.json')
    train_path=resolve_file(args.train_cache,'tinyperson_p3_train.pt',cache=True)
    test_path=resolve_file(args.test_cache,'tinyperson_p3_val.pt',cache=True)
    train_all,train_meta=load_checked_cache(train_path,train_gt,smoke=args.smoke)
    test,test_meta=load_checked_cache(test_path,test_gt,smoke=args.smoke)
    documented_pair = train_meta is not None and test_meta is not None
    if documented_pair:
        for key in ('checkpoint_sha256','conf_thresh','intra_nms','roi_dim','roi_seed','amp','torch_version','torchvision_version','source_hashes'):
            if key not in train_meta or train_meta[key]!=test_meta.get(key):
                raise ValueError(f'Train/test extraction signature mismatch: {key}')
    feature_dims = {s.node_features.shape[1] for samples in (train_all, test)
                    for s in samples if len(s.boxes_global)}
    if len(feature_dims) != 1:
        raise ValueError(f'Expected one shared nonempty candidate feature width, found {sorted(feature_dims)}')
    if not args.smoke and (train_meta.get('max_limit') or test_meta.get('max_limit')):
        raise ValueError('Full experiment requires full caches (max_limit=None)')
    train,val,split=split_train_validation(train_all,train_gt,fraction=.2,seed=42)
    old_split=Path(args.split_manifest) if args.split_manifest else Path('/kaggle/working/stage1_split.json')
    if old_split.is_file() and (documented_pair or args.split_manifest):
        old=json.loads(old_split.read_text(encoding='utf-8'))
        if any(set(old[k])!=set(split[k]) for k in ('train_ids','val_ids')):
            raise ValueError('Stage-one split differs; resolve before comparing experiments')
    elif args.split_manifest:
        raise FileNotFoundError(old_split)
    write_json(out/'stage2_split.json',split)
    if args.smoke:
        train,val,test=train[:8],val[:4],test[:4]
    print(f'Stage2 train/val/test images: {len(train)}/{len(val)}/{len(test)}; fixed coordinates; device={args.device}',flush=True)
    cfg=QualityConfig(epochs=2 if args.smoke else args.epochs,val_freq=args.val_freq,device=args.device)
    from dataclasses import asdict
    files=['TileStitchingNet/stage2.py','TileStitchingNet/stage1.py','tools/kaggle_train_stage2.py']
    write_json(out/'stage2_run.json',dict(config=asdict(cfg),cli=vars(args),train_cache=str(train_path),test_cache=str(test_path),
               train_cache_manifest=train_meta,test_cache_manifest=test_meta,
               source_hashes={p:sha(ROOT/p) for p in files},
               actual_train_ids=[s.image_id for s in train],actual_val_ids=[s.image_id for s in val],actual_test_ids=[s.image_id for s in test]))
    val_eval = OfficialTinyEvaluator(train_gt, include_split=all(s.tile_rects_all is not None for s in val))
    test_eval = OfficialTinyEvaluator(test_gt, include_split=all(s.tile_rects_all is not None for s in test))
    baselines={
        'Greedy-NMS':Baseline(lambda b,s:run_greedy_nms(b,s,iou_thresh=.5,score_thresh=0)),
        'Soft-NMS':Baseline(lambda b,s:run_soft_nms(b,s,sigma=.5,score_thresh=0)),
        'WBF':Baseline(lambda b,s:run_wbf(b,s,iou_thresh=.55,score_thresh=0)),
        'SAHI-NMM':Baseline(lambda b,s:run_sahi_nmm_ios(b,s,ios_thresh=.5,score_thresh=0))}
    validation={name:evaluate_predictor(val,p,val_eval) for name,p in baselines.items()}
    models={}
    variants=[('pointwise',False,0.),('local',True,0.)]
    if args.with_ranking:
        variants.append(('local_rank',True,.1))
    for name,local,rank_weight in variants:
        variant_cfg=QualityConfig(**{**asdict(cfg),'rank_weight':rank_weight})
        model,history=train_quality(train,val,train_gt,val_eval,variant_cfg,local,str(out/f'{name}_best.pth'))
        models[name]=model
        write_json(out/f'{name}_history.json',history)
        for mode in ('before_nms','after_nms'):
            validation[f'{name}/{mode}']=evaluate_predictor(val,QualityPredictor(model,mode),val_eval)
        write_json(out/'validation_metrics.json',validation)
    # Test is only evaluated after training/selection of all predefined variants.
    results={name:evaluate_predictor(test,p,test_eval) for name,p in baselines.items()}
    for name,model in models.items():
        for mode in ('before_nms','after_nms'):
            results[f'{name}/{mode}']=evaluate_predictor(test,QualityPredictor(model,mode),test_eval)
            write_json(out/'test_metrics.json',results)
    print('\nMethod | AP50:95(all) | AP50(all) | AP75(all) | Tiny AP50 | Tiny AP75 | ms')
    for name,m in results.items():
        print(f"{name:24} | {m['AP']:.2f} | {m['AP50']:.2f} | {m['AP75']:.2f} | {m['Tiny_AP50']:.2f} | {m['Tiny_AP75']:.2f} | {m['Latency_ms']:.2f}")
    print(f'Artifacts: {out}. Stage-one weights/caches unchanged.',flush=True)


if __name__=='__main__':
    main(None if '__file__' in globals() else [])
