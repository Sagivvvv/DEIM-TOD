"""Coordinate-preserving quality rescoring, independent of stage-one stitching."""
import copy
import json
import time
from pathlib import Path
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial import cKDTree

from .datasetBuilder import box_area_np, nms_np, pairwise_iou_np


@dataclass
class QualityConfig:
    hidden: int = 64
    neighbors: int = 16
    min_radius: float = 32.0
    positive_iou: float = .5
    background_iou: float = .3
    ambiguity_gap: float = .05
    ignore_iod: float = .5
    rank_weight: float = 0.0
    rank_gap: float = .1
    epochs: int = 18
    lr: float = 1e-3
    batch_tokens: int = 1000
    val_freq: int = 1
    seed: int = 42
    device: str = 'cpu'


class QualityTargets:
    """All reliable duplicates get full-box IoU; never select a single leader.

    Foreground >= .5, background < .3; gray/ambiguous/ignored regions have no loss.
    Ignore IOD is a conservative training rule, not the official evaluator rule.
    """
    def __init__(self, annotation_path, config):
        self.config = config
        with open(annotation_path, encoding='utf-8') as f:
            data = json.load(f)
        self.by_image = {str(x['id']): ([], []) for x in data['images']}
        for a in data['annotations']:
            if a['category_id'] != 1:
                continue
            x,y,w,h = a['bbox']
            if w <= 0 or h <= 0:
                continue
            ignored = any(a.get(k,0) for k in ('ignore','uncertain','iscrowd'))
            self.by_image[str(a['image_id'])][int(ignored)].append([x,y,x+w,y+h])

    def __call__(self, sample):
        cfg = self.config
        normal, ignored = [np.asarray(x,np.float32).reshape(-1,4) for x in self.by_image[sample.image_id]]
        b = sample.boxes_global
        n = len(b)
        q, valid, gid = np.zeros(n,np.float32), np.ones(n,bool), np.full(n,-1,np.int64)
        # Bounded candidate x GT working memory, independent of candidate graph size.
        for start in range(0,n,512):
            end = min(n,start+512)
            bb = b[start:end]
            best = np.zeros(len(bb),np.float32)
            second = best.copy()
            ids = np.full(len(bb),-1,np.int64)
            if len(normal):
                ov = pairwise_iou_np(bb,normal)
                ids = ov.argmax(1)
                best = ov[np.arange(len(bb)),ids]
                if len(normal)>1:
                    second = np.partition(ov,-2,axis=1)[:,-2]
            fg = best >= cfg.positive_iou
            ambiguous = (second >= cfg.positive_iou) & (best-second < cfg.ambiguity_gap)
            keep = ((best < cfg.background_iou) | fg) & ~ambiguous
            if len(ignored):
                inter = np.prod(np.maximum(0,np.minimum(bb[:,None,2:],ignored[None,:,2:])-
                                           np.maximum(bb[:,None,:2],ignored[None,:,:2])),axis=-1)
                iod = inter / np.maximum(box_area_np(bb)[:,None],1e-7)
                # Reliable normal positives take precedence over ignored regions.
                keep &= fg | (iod.max(1) < cfg.ignore_iod)
            q[start:end] = np.where(fg,best,0)
            valid[start:end] = keep
            gid[start:end] = np.where(fg & keep,ids,-1)
        return q, valid, gid


def candidate_graph(sample, cfg):
    """kNN via a spatial tree; O(NK) relation tensors, same- and cross-tile edges."""
    b = sample.boxes_global
    n = len(b)
    k = min(cfg.neighbors+1,n)
    if not n:
        return np.empty((0,0),np.int64), np.empty((0,0,8),np.float32), np.empty((0,0),bool)
    wh = np.maximum(b[:,2:]-b[:,:2],1.)
    centers = (b[:,:2]+b[:,2:])*.5
    distance, idx = cKDTree(centers).query(centers,k=k)
    idx, distance = idx.reshape(n,k), distance.reshape(n,k)
    mask = distance <= np.maximum(cfg.min_radius,2*wh.max(1))[:,None]
    # At least one zero-distance neighbor always exists (self or a coincident box).
    dxdy = (centers[idx]-centers[:,None]) / wh[:,None]
    ratio = np.log(wh[idx]/wh[:,None])
    inter = np.prod(np.maximum(0,np.minimum(b[:,None,2:],b[idx,2:])-np.maximum(b[:,None,:2],b[idx,:2])),axis=-1)
    area = box_area_np(b)
    iou = inter / np.maximum(area[:,None]+area[idx]-inter,1e-7)
    same = (sample.patch_ids[:,None] == sample.patch_ids[idx]).astype(np.float32)
    diff = sample.scores[idx]-sample.scores[:,None]
    dist = np.log1p(distance / np.sqrt(np.prod(wh,axis=1))[:,None])
    geo = np.concatenate([dxdy,ratio,iou[...,None],same[...,None],diff[...,None],dist[...,None]],axis=-1)
    return idx.astype(np.int64),geo.clip(-10,10).astype(np.float32),mask


def prepare(sample, cfg, local=True, targets=None):
    nf = sample.node_features.copy()
    nf[:,6] = 0  # Tile IDs are arbitrary; relation geometry carries same/cross identity.
    nf = np.concatenate([nf,np.log1p(np.maximum(sample.boxes_global[:,2:]-sample.boxes_global[:,:2],0))],axis=1)
    arrays = dict(features=nf.astype(np.float32), base=sample.scores.astype(np.float32))
    if local:
        arrays['neighbors'],arrays['geometry'],arrays['neighbor_mask'] = candidate_graph(sample,cfg)
    if targets is not None:
        arrays['quality'],arrays['valid'],arrays['gt_id'] = targets(sample)
    return {k:torch.from_numpy(v).to(cfg.device) for k,v in arrays.items()}


class QualityRescorer(nn.Module):
    def __init__(self, feature_dim, cfg, local=False):
        super().__init__()
        self.config, self.local, self.feature_dim = cfg, local, feature_dim
        d = cfg.hidden
        self.encoder = nn.Sequential(nn.Linear(feature_dim,d),nn.LayerNorm(d),nn.ReLU(),nn.Linear(d,d),nn.ReLU())
        if local:
            self.attention = nn.Sequential(nn.Linear(2*d+8,d),nn.ReLU(),nn.Linear(d,1))
            self.message = nn.Sequential(nn.Linear(d+8,d),nn.ReLU())
        self.head = nn.Sequential(nn.Linear(d*(2 if local else 1),d),nn.ReLU(),nn.Linear(d,1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, data):
        h = self.encoder(data['features'])
        if self.local and len(h):
            j,geo = data['neighbors'],data['geometry']
            hi = h[:,None].expand(-1,j.shape[1],-1)
            a = self.attention(torch.cat([hi,h[j],geo],-1)).squeeze(-1)
            a = a.masked_fill(~data['neighbor_mask'],-torch.inf).softmax(1)
            message = (a[...,None]*self.message(torch.cat([h[j],geo],-1))).sum(1)
            h = torch.cat([h,message],-1)
        elif self.local:
            h = torch.cat([h,h],-1)
        delta = self.head(h).squeeze(-1)
        base = data['base'].clamp(1e-6,1-1e-6)
        base_logits = torch.logit(base)
        logits = base_logits+delta
        # Identity at zero initialization, including float32 roundoff, with live gradients.
        scores = (base+(logits.sigmoid()-base_logits.sigmoid())).clamp(0,1)
        return dict(logits=logits,scores=scores,delta=delta)


def quality_loss(outputs, data, cfg):
    valid, q = data['valid'],data['quality']
    z = outputs['logits'].float()
    if not valid.any():
        zero = z.sum()*0
        return zero,zero,zero
    zv,qv = z[valid],q[valid]
    qfl = (F.binary_cross_entropy_with_logits(zv,qv,reduction='none') * (zv.sigmoid()-qv).abs().square()).sum()
    qfl = qfl / max(1,int((qv>0).sum()))
    rank = z.sum()*0
    if cfg.rank_weight and 'neighbors' in data:
        j = data['neighbors']
        pairs = (data['neighbor_mask'] & valid[:,None] & valid[j] & (data['gt_id'][:,None] >= 0)
                 & (data['gt_id'][:,None] == data['gt_id'][j]) & (q[:,None]-q[j] >= cfg.rank_gap))
        if pairs.any():
            rank = F.softplus((z[j]-z[:,None])[pairs]).mean()
    return qfl+cfg.rank_weight*rank,qfl,rank


class QualityPredictor:
    def __init__(self, model, mode='before_nms', nms_iou=.5):
        if mode not in ('before_nms','after_nms'):
            raise ValueError(mode)
        self.model,self.mode,self.nms_iou = model,mode,nms_iou

    def predict_sample(self,sample):
        self.model.eval()
        if len(sample.boxes_global) == 0:
            return sample.boxes_global.copy(),sample.scores.copy(),0.0
        if self.model.config.device.startswith('cuda'):
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            data = prepare(sample,self.model.config,self.model.local)
            scores = self.model(data)['scores'].cpu().numpy()
        # after_nms uses the *original* NMS survivor set; all nodes can supply context.
        keep = nms_np(sample.boxes_global,sample.scores if self.mode=='after_nms' else scores,self.nms_iou)
        latency = (time.perf_counter()-start)*1000
        return sample.boxes_global[keep].copy(),scores[keep],latency


def load_quality_checkpoint(path, device='cpu'):
    payload = torch.load(path,map_location=device,weights_only=True)
    cfg = QualityConfig(**{**payload['config'],'device':device})
    model = QualityRescorer(payload['feature_dim'],cfg,payload['local']).to(device)
    model.load_state_dict(payload['model'])
    return model.eval()


def evaluate_predictor(samples,predictor,evaluator):
    rows = [predictor.predict_sample(s) for s in samples]
    return evaluator(samples,[r[0] for r in rows],[r[1] for r in rows],[r[2] for r in rows])


def quality_diagnostics(model,samples,targets):
    """Validation-only quality calibration and within-GT ranking; no detection AP claim."""
    model.eval()
    errors,chosen_base,chosen_new,oracle=[],[],[],[]
    with torch.no_grad():
        for s in samples:
            if not len(s.boxes_global):
                continue
            q,valid,gid=targets(s)
            scores=model(prepare(s,model.config,model.local))['scores'].cpu().numpy()
            positive=valid & (q>0)
            errors.extend(np.abs(scores[positive]-q[positive]).tolist())
            for g in np.unique(gid[gid>=0]):
                indices=np.flatnonzero(gid==g)
                chosen_base.append(float(q[indices[np.argmax(s.scores[indices])]]))
                chosen_new.append(float(q[indices[np.argmax(scores[indices])]]))
                oracle.append(float(q[indices].max()))
    mean=lambda x:float(np.mean(x)) if x else None
    return dict(positive_mae=mean(errors),gt_best_by_base=mean(chosen_base),gt_best_by_new=mean(chosen_new),
                gt_oracle_quality=mean(oracle),reliable_gt_groups=len(oracle),positive_candidates=len(errors))


def train_quality(train_samples,val_samples,annotation_path,evaluator,cfg,local,checkpoint_path,snapshot_dir=None):
    if not train_samples or not val_samples or not any(len(s.boxes_global) for s in train_samples):
        raise ValueError('Nonempty train/validation sets required')
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    feature_dim = next(s.node_features.shape[1]+2 for s in train_samples if len(s.boxes_global))
    model = QualityRescorer(feature_dim,cfg,local).to(cfg.device)
    target_builder = QualityTargets(annotation_path,cfg)
    batches = [prepare(s,cfg,local,target_builder) for s in train_samples]
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[max(1,int(cfg.epochs*.65)),max(2,int(cfg.epochs*.85))],gamma=.35)
    history = []
    best, best_epoch, best_state = -float('inf'),0,None
    stats = dict(positive=sum(int(((d['quality']>0)&d['valid']).sum()) for d in batches),
                 negative=sum(int(((d['quality']==0)&d['valid']).sum()) for d in batches),
                 ignored=sum(int((~d['valid']).sum()) for d in batches))
    if stats['positive'] == 0:
        raise ValueError('No reliable positive candidates in training')
    print('Quality label counts:',stats,flush=True)

    def validate(epoch,record):
        nonlocal best,best_state,best_epoch
        # Select once using before-NMS validation; after-NMS is a controlled
        # diagnostic with the SAME checkpoint, not another test-set selection.
        metrics = evaluate_predictor(val_samples,QualityPredictor(model),evaluator)
        value = metrics['Tiny_AP50']
        if not np.isfinite(value):
            raise ValueError('Tiny_AP50 undefined on validation')
        record['validation_before_nms'] = metrics
        record['quality_diagnostics'] = quality_diagnostics(model,val_samples,target_builder)
        if snapshot_dir is not None:
            snapshot_root=Path(snapshot_dir)
            snapshot_root.mkdir(parents=True,exist_ok=True)
            payload=dict(model=model.state_dict(),config=asdict(cfg),local=local,feature_dim=model.feature_dim,
                         epoch=epoch,validation=metrics,label_counts=stats)
            torch.save(payload,snapshot_root/'last.pth')
            if epoch in {0,1,2,6,12,cfg.epochs}:
                torch.save(payload,snapshot_root/f'epoch_{epoch:02d}.pth')
        if value > best:
            best,best_epoch,best_state = value,epoch,copy.deepcopy(model.state_dict())
            torch.save(dict(model=best_state,config=asdict(cfg),local=local,feature_dim=model.feature_dim,
                            epoch=epoch,selection_metric='Tiny_AP50',validation=metrics,label_counts=stats),checkpoint_path)
        print(f"[Quality {'local' if local else 'pointwise'} {epoch:02d}] loss={record.get('loss',0):.5f} "
              f"Val_Tiny_AP50={value:.2f} best={best:.2f}@{best_epoch}",flush=True)
        print('  validation quality:',record['quality_diagnostics'],flush=True)
    record = dict(epoch=0,label_counts=stats)
    validate(0,record)  # Baseline-preserving initialization is eligible for selection.
    history.append(record)
    for epoch in range(1,cfg.epochs+1):
        model.train()
        optimizer.zero_grad()
        count=tokens=steps=0
        losses=[]
        def flush(count):
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(count)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            optimizer.zero_grad()
        for i in rng.permutation(len(batches)):
            data = batches[i]
            if not len(data['base']) or not data['valid'].any():
                continue
            out = model(data)
            loss,qfl,rank = quality_loss(out,data,cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite quality loss')
            loss.backward()
            losses.append([loss.item(),qfl.item(),rank.item()])
            count+=1
            tokens+=len(data['base'])
            if tokens>=cfg.batch_tokens:
                flush(count)
                steps+=1
                count=tokens=0
        if count:
            flush(count)
            steps+=1
        scheduler.step()
        means=np.mean(losses,axis=0)
        record=dict(epoch=epoch,loss=float(means[0]),qfl=float(means[1]),rank=float(means[2]),optimizer_steps=steps)
        if epoch%cfg.val_freq==0 or epoch==cfg.epochs:
            validate(epoch,record)
        history.append(record)
    model.load_state_dict(best_state)
    return model,history
