"""Frozen NMS competition labels and instance-balanced supervision for P3.

QFL keeps its original labels/logits. Pair supervision uses the deployed scaled
logits; gray candidates are supervised only through a reliable competition.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import torch
from torch.nn import functional as F

from .p1 import CandidateLabels, nms_trace
from .p2 import BoundedQualityRescorer, atomic_json, atomic_save
from .stage2 import QualityConfig, quality_loss


@dataclass(frozen=True)
class PairConfig:
    threshold: float = .5
    nms_iou: float = .5
    gap: float = .1
    ambiguity_gap: float = .05
    ambiguity_floor: float = .3
    ignore_iod: float = .5
    boundary_band: float = .05
    boundary_floor: float = .25
    alpha: float = .25


def competition_pairs(sample, labels, edges, cfg):
    """One label per actual frozen suppression edge, independent of the network.

    Both good and bad may be reliable positives for the same GT. A gray bad
    candidate must have the same nearest GT as good; positives for a different
    GT are always protected. Ignore overlap is excluded even for QFL positives.
    """
    q, gid = labels['best'], labels['gid']
    rows, excluded = [], Counter()
    for winner, suppressed in edges:
        i, j = int(winner), int(suppressed)
        ids = [i, j]
        if np.any(labels['iod'][ids] >= cfg.ignore_iod):
            excluded['ignore_overlap'] += 1
            continue
        if np.any((labels['second'][ids] >= cfg.ambiguity_floor) &
                  (q[ids] - labels['second'][ids] < cfg.ambiguity_gap)):
            excluded['ambiguous_assignment'] += 1
            continue
        fi, fj = q[i] >= cfg.threshold, q[j] >= cfg.threshold
        if not (fi or fj):
            excluded['neither_qualified'] += 1
            continue
        if fi and fj and gid[i] != gid[j]:
            excluded['different_gt_positives'] += 1
            continue
        good, bad = (i, j) if q[i] > q[j] else (j, i)
        if q[bad] >= cfg.ambiguity_floor and gid[good] != gid[bad]:
            excluded['different_gt_gray'] += 1
            continue
        gap = float(q[good]) - float(q[bad])
        if gap + 1e-7 < cfg.gap:
            excluded['small_quality_gap'] += 1
            continue
        crossing = not (fi and fj)
        # Multiplicative confidence remains effective even in a one-pair group:
        # loss divides by pair count, NOT by the sum of confidence weights.
        margin = min(float(q[good]) - cfg.threshold, cfg.threshold - float(q[bad]))
        weight = (cfg.boundary_floor + (1 - cfg.boundary_floor) *
                  min(1., max(0., margin / cfg.boundary_band))) if crossing else 1.
        gi = int(gid[good])
        gt = labels['normal_boxes'][gi]
        base_margin = float(sample.scores[good] - sample.scores[bad])
        rows.append(dict(winner=i, suppressed=j, good=good, bad=bad,
            gt_id=int(labels['normal_ids'][gi]), quality_good=float(q[good]), quality_bad=float(q[bad]),
            threshold=cfg.threshold, kind='threshold_crossing' if crossing else 'same_gt_quality',
            cross_tile=bool(sample.patch_ids[i] != sample.patch_ids[j]),
            gt_size='tiny' if float(np.sqrt(np.prod(gt[2:] - gt[:2]))) <= 20 else 'non_tiny',
            weight=weight, base_order='correct' if base_margin > 0 else 'wrong' if base_margin < 0 else 'tie',
            good_reason=str(labels['reason'][good]), bad_reason=str(labels['reason'][bad]),
            good_qfl_valid=bool(labels['valid'][good]), bad_qfl_valid=bool(labels['valid'][bad])))
    return rows, dict(excluded)


def build_pair_dataset(samples, annotation_data, targets, cfg):
    labeler = CandidateLabels(annotation_data, targets)
    rows, summaries, cases, geometry = {}, [], [], {}
    exclusions, included = Counter(), Counter()
    for sample in samples:
        labels = labeler(sample)
        geometry[sample.image_id] = dict(boxes=sample.boxes_global.tolist(), scores=sample.scores.tolist(),
            patch_ids=sample.patch_ids.tolist(), normal_ids=labels['normal_ids'].tolist(),
            normal_boxes=labels['normal_boxes'].tolist(),
            ignored_boxes=np.asarray(targets.by_image[sample.image_id][1], np.float32).reshape(-1, 4).tolist())
        _, edges = nms_trace(sample.boxes_global, sample.scores, cfg.nms_iou)
        pairs, rejected = competition_pairs(sample, labels, edges, cfg)
        if len(pairs) + sum(rejected.values()) != len(edges):
            raise AssertionError('Suppression edge accounting differs')
        rows[sample.image_id] = pairs
        exclusions.update(rejected)
        summaries.append(dict(image_id=sample.image_id, edges=len(edges), pairs=len(pairs),
            gt_groups=len({p['gt_id'] for p in pairs}), excluded=rejected))
        for pair in pairs:
            included.update([pair['kind'], pair['base_order'], 'bad_' + pair['bad_reason'],
                             'cross_tile' if pair['cross_tile'] else 'same_tile'])
            # Keep a bounded, stratified sample for annotation review.
            stratum = (pair['kind'], pair['base_order'], pair['bad_reason'])
            if sum(tuple(c['stratum']) == stratum for c in cases) < 10:
                gi = int(np.flatnonzero(labels['normal_ids'] == pair['gt_id'])[0])
                cases.append(dict(image_id=sample.image_id, stratum=list(stratum), **pair,
                    good_box=sample.boxes_global[pair['good']].tolist(),
                    bad_box=sample.boxes_global[pair['bad']].tolist(),
                    gt_box=labels['normal_boxes'][gi].tolist()))
    payload = dict(config=asdict(cfg), pairs=rows)
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                          allow_nan=False).encode()).hexdigest()
    return dict(**payload, signature=signature, summary=dict(images=len(samples),
        images_with_pairs=sum(bool(row['pairs']) for row in summaries),
        edges=sum(row['edges'] for row in summaries), pairs=sum(row['pairs'] for row in summaries),
        gt_groups=sum(row['gt_groups'] for row in summaries), included=dict(included), excluded=dict(exclusions)),
        per_image=summaries, audit_cases=cases, geometry=geometry)


def pair_tensors(pairs):
    groups = {g: k for k, g in enumerate(sorted({p['gt_id'] for p in pairs}))}
    return dict(pair_good=torch.tensor([p['good'] for p in pairs], dtype=torch.long),
                pair_bad=torch.tensor([p['bad'] for p in pairs], dtype=torch.long),
                pair_group=torch.tensor([groups[p['gt_id']] for p in pairs], dtype=torch.long),
                pair_confidence=torch.tensor([p['weight'] for p in pairs], dtype=torch.float32),
                pair_group_count=len(groups))


def competition_loss(outputs, data, alpha=.25):
    """Mean over GT groups of mean(weight * softplus(z_bad - z_good))."""
    delta = outputs['delta'].float()
    if data['pair_group_count'] == 0:
        return delta.sum() * 0
    logits = torch.logit(data['base'].float().clamp(1e-6, 1-1e-6)) + alpha * delta
    loss = data['pair_confidence'] * F.softplus(logits[data['pair_bad']] - logits[data['pair_good']])
    groups = data['pair_group']
    sums = loss.new_zeros(data['pair_group_count']).scatter_add(0, groups, loss)
    counts = torch.bincount(groups, minlength=data['pair_group_count']).to(loss.dtype)
    return (sums / counts.clamp_min(1)).mean()


def load_p3_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema') != 'p3-v1':
        raise ValueError('Expected a P3 checkpoint')
    model = BoundedQualityRescorer(payload['feature_dim'], QualityConfig(**payload['config']))
    model.load_state_dict(payload['model'], strict=True)
    return model.eval(), payload


def train_competition(batches, config, pair_config, pair_weight, pair_signature, output_dir, validate):
    """Same updates/order as stage2.train_quality; validation may expose multiple views.

    A completed epoch atomically commits optimizer, scheduler, RNG, history and
    best-epoch metadata. A partial epoch is replayed from the previous commit.
    """
    from pathlib import Path
    if config.rank_weight != 0 or pair_weight < 0 or not np.isfinite(pair_weight):
        raise ValueError('P3 requires original QFL with a nonnegative finite pair weight')
    if config.device != 'cpu':
        raise ValueError('P3 controlled experiment uses CPU training')
    if not batches or not any(len(batch['base']) for batch in batches):
        raise ValueError('No training candidates')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    feature_dim = batches[0]['features'].shape[1]
    model = BoundedQualityRescorer(feature_dim, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [max(1, int(config.epochs*.65)), max(2, int(config.epochs*.85))], gamma=.35)
    counts = dict(
        positive=sum(int(((d['quality'] > 0) & d['valid']).sum()) for d in batches),
        negative=sum(int(((d['quality'] == 0) & d['valid']).sum()) for d in batches),
        ignored=sum(int((~d['valid']).sum()) for d in batches))
    if counts['positive'] == 0:
        raise ValueError('No reliable positive training candidates')
    history = []
    best = {}
    resume_path = output_dir/'resume.pth'
    start_epoch = 0
    if resume_path.is_file():
        saved = torch.load(resume_path, map_location='cpu', weights_only=True)
        if saved['config'] != asdict(config) or saved['pair_config'] != asdict(pair_config) or saved['pair_weight'] != pair_weight or saved['pair_signature'] != pair_signature:
            raise ValueError('Resume training configuration differs')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'])
        history, best = saved['history'], saved['best']
        start_epoch = saved['epoch'] + 1
        print(f'Resuming {output_dir.name} after epoch {saved["epoch"]}', flush=True)

    def commit(epoch, record):
        views = validate(model, epoch)
        record['validation'] = views
        for name, view in views.items():
            value = view['metrics']['Tiny_AP50']
            if value is None or not np.isfinite(value):
                raise ValueError('Validation Tiny_AP50 is undefined')
            if name not in best or value > best[name]['metric']:
                best[name] = dict(epoch=epoch, metric=value, alpha=view['alpha'])
        history.append(record)
        payload = dict(schema='p3-v1', model=model.state_dict(), config=asdict(config),
                       feature_dim=feature_dim, residual_bound=None, epoch=epoch,
                       pair_config=asdict(pair_config), pair_weight=pair_weight, pair_signature=pair_signature,
                       alpha=pair_config.alpha, formula='QFL: base+r; pair/deploy: base+alpha*r',
                       label_counts=counts, validation=views)
        atomic_save(payload, output_dir/f'epoch_{epoch:02d}.pth')
        state = dict(payload, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                     history=history, best=best)
        atomic_save(state, resume_path)
        atomic_json(history, output_dir/'history.json')
        values = ' '.join(f'{name}={view["metrics"]["Tiny_AP50"]:.4f}' for name, view in views.items())
        print(f'[{output_dir.name} {epoch:02d}/{config.epochs}] loss={record.get("loss", 0):.6f} '
              f'qfl={record.get("qfl", 0):.6f} pair={record.get("pair", 0):.6f} '
              f'weighted_pair={record.get("weighted_pair", 0):.6f} {values}', flush=True)

    if start_epoch == 0:
        commit(0, dict(epoch=0, label_counts=counts))
        start_epoch = 1

    def optimizer_step(image_count):
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(image_count)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        optimizer.zero_grad()

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        image_count = tokens = steps = 0
        losses = []
        for index in rng.permutation(len(batches)):
            data = batches[index]
            if not len(data['base']) or (not data['valid'].any() and not (pair_weight and data['pair_group_count'])):
                continue
            outputs = model(data)
            _, qfl, _ = quality_loss(outputs, data, config)
            rank = competition_loss(outputs, data, pair_config.alpha)
            # Preserve the exact original QFL graph when lambda=0.
            loss = qfl if pair_weight == 0 else qfl + pair_weight * rank
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            losses.append([loss.item(), qfl.item(), rank.item(), pair_weight * rank.item()])
            image_count += 1
            tokens += len(data['base'])
            if tokens >= config.batch_tokens:
                optimizer_step(image_count)
                steps += 1
                image_count = tokens = 0
        if image_count:
            optimizer_step(image_count)
            steps += 1
        scheduler.step()
        means = np.mean(losses, axis=0)
        commit(epoch, dict(epoch=epoch, loss=float(means[0]), qfl=float(means[1]),
                           pair=float(means[2]), weighted_pair=float(means[3]), optimizer_steps=steps))

    for name, selection in best.items():
        payload = torch.load(output_dir/f'epoch_{selection["epoch"]:02d}.pth', map_location='cpu', weights_only=True)
        payload.update(alpha=selection['alpha'], selected_view=name, selection_metric='Tiny_AP50_before_nms')
        atomic_save(payload, output_dir/(name+'_best.pth'))
    atomic_json(history, output_dir/'history.json')
    return history, best
