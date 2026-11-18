"""Refresh ROI only, preserving legacy candidate identity and non-ROI features."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.ops import roi_align


def array_hash(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256(str((a.shape,a.dtype.str)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def frozen_fields(s):
    keys=('boxes_global','scores','class_ids','patch_ids','patch_rects','boundary_dists','is_boundary_risk')
    return dict(image_id=s.image_id,img_size=list(s.img_size),
                arrays={k:array_hash(getattr(s,k)) for k in keys},node_prefix=array_hash(s.node_features[:,:12]))


def replace_roi(s,roi):
    if roi.shape!=(len(s.scores),64) or not np.isfinite(roi).all():
        raise ValueError('Invalid refreshed ROI: '+s.image_id)
    before=frozen_fields(s)
    s.roi_features=roi.astype(np.float32)
    s.node_features=np.concatenate([s.node_features[:,:12],s.roi_features],axis=1)
    if frozen_fields(s)!=before: raise AssertionError('Candidate mutation')


def to_network_boxes(boxes,rect):
    rect=np.asarray(rect,np.float32)
    wh=rect[2:]-rect[:2]
    if np.any(wh<=0): raise ValueError('Invalid patch rectangle')
    return ((boxes-np.tile(rect[:2],2))*np.tile(np.array([640,576],np.float32)/wh,2)).astype(np.float32)


class SharedProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(256,64),nn.LayerNorm(64),nn.ReLU())
        nn.init.orthogonal_(self.proj[0].weight); nn.init.zeros_(self.proj[0].bias)

    def forward(self,p3,rois):
        pooled=roi_align(p3.float(),rois.float(),(1,1),spatial_scale=1/8,sampling_ratio=2,aligned=False)
        return self.proj(pooled.flatten(1))


def load_detector(config,checkpoint,device):
    from engine.core import YAMLConfig
    cfg=YAMLConfig(str(config))
    if 'HGNetv2' in cfg.yaml_cfg: cfg.yaml_cfg['HGNetv2']['pretrained']=False
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    state=payload.get('ema',{}).get('module',payload.get('model',payload))
    model=cfg.model
    model.load_state_dict(state,strict=True)
    return model.eval().to(device)


def resolve_image(root,filename,split):
    root=Path(root); filename=Path(filename)
    options=[root/split/filename,root/filename,root/split/filename.name,root/filename.name]
    for p in options:
        if p.is_file(): return p
    matches=list((root/split).rglob(filename.name)) if (root/split).is_dir() else []
    if len(matches)!=1: raise FileNotFoundError(f'Original image {filename}, split={split}, root={root}: {len(matches)} matches')
    return matches[0]


@torch.inference_mode()
def extract_roi(sample,image_path,detector,projection,device='cpu',batch_tiles=4):
    n=len(sample.scores)
    if not n: return np.empty((0,64),np.float32)
    ids=np.unique(sample.patch_ids)
    rects=[]; rows=[]
    for pid in ids:
        indices=np.flatnonzero(sample.patch_ids==pid)
        rect=sample.patch_rects[indices[0]]
        if not np.all(sample.patch_rects[indices]==rect): raise ValueError('Inconsistent patch ID rectangle')
        if not np.allclose(rect,np.round(rect),atol=1e-4,rtol=0): raise ValueError('Fractional crop rectangle')
        rects.append(tuple(int(x) for x in np.round(rect))); rows.append(indices)
    result=np.empty((n,64),np.float32)
    with Image.open(image_path) as im:
        if im.size!=tuple(sample.img_size): raise ValueError('Original image dimensions differ from cache')
        im=im.convert('RGB')
        for start in range(0,len(ids),batch_tiles):
            tensors=[]; boxes=[]; dest=[]
            for j,rect in enumerate(rects[start:start+batch_tiles]):
                if min(rect[:2])<0 or rect[2]>im.width or rect[3]>im.height: raise ValueError('Patch outside original image')
                crop=im.crop(rect)
                if crop.size!=(640,576): crop=crop.resize((640,576),Image.Resampling.BILINEAR)
                tensors.append(torch.from_numpy(np.array(crop,copy=True)).permute(2,0,1).float()/255.)
                indices=rows[start+j]
                local=to_network_boxes(sample.boxes_global[indices],rect)
                boxes.append(np.column_stack([np.full(len(indices),j,np.float32),local])); dest.extend(indices.tolist())
            batch=torch.stack(tensors).to(device)
            encoded=detector.encoder(detector.backbone(batch))
            p3=encoded[0] if isinstance(encoded,(list,tuple)) else encoded['p3']
            if tuple(p3.shape[1:])!=(256,72,80): raise ValueError(f'Unexpected P3 shape {p3.shape}; requires stride8, 256 channels')
            rois=torch.from_numpy(np.concatenate(boxes).astype(np.float32)).to(device)
            feat=projection(p3,rois).cpu().numpy()
            result[np.asarray(dest)]=feat
    if not np.isfinite(result).all(): raise FloatingPointError('Nonfinite ROI')
    return result
