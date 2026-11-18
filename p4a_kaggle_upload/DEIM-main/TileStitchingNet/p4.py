"""P4 input ablation and non-mutating, training-only gradient diagnostics."""
from collections import Counter
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .p1 import nms_trace
from .p2 import BoundedQualityRescorer, atomic_json, atomic_save
from .p3 import competition_loss
from .stage2 import QualityConfig, quality_loss


class RoiRescorer(BoundedQualityRescorer):
    """Zero ROI at every forward, including reload/inference; keep 78 inputs."""
    def __init__(self, feature_dim, config, roi_mode):
        if feature_dim != 78 or roi_mode not in ('full', 'zero'):
            raise ValueError('P4 expects 12 geometry/metadata + 64 ROI + 2 log-size features')
        super().__init__(feature_dim, config)
        self.roi_mode = roi_mode

    def forward(self, data):
        if self.roi_mode == 'zero':
            features = data['features'].clone()
            features[:, 12:76] = 0
            data = {**data, 'features': features}
        return super().forward(data)


def load_p4_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema') != 'p4-v1':
        raise ValueError('Expected P4 checkpoint with explicit ROI mode')
    model = RoiRescorer(payload['feature_dim'], QualityConfig(**payload['config']), payload['roi_mode'])
    model.load_state_dict(payload['model'], strict=True)
    return model.eval(), payload


def annotation_audit(data, targets, split, previous_geometry):
    """Explain zero ignored GT using raw annotations; never silently fix labels."""
    ids = set(split['train_ids']) | set(split['val_ids'])
    scopes = {'all_source_images': {str(i['id']) for i in data['images']},
              'cache_images': ids, 'training': set(split['train_ids']), 'validation': set(split['val_ids'])}
    counts = {name: {} for name in scopes}
    expected = {iid: [[], []] for iid in ids}
    normal_ids = {iid: [] for iid in ids}
    errors, unhandled = [], []
    seen = set()
    for ann in data['annotations']:
        iid, category = str(ann['image_id']), str(ann['category_id'])
        if ann['id'] in seen:
            errors.append('duplicate annotation ID: '+str(ann['id']))
        seen.add(ann['id'])
        flags = {k: ann.get(k, 0) for k in ('ignore', 'uncertain', 'iscrowd')}
        if any(v not in (0, 1, False, True) for v in flags.values()):
            errors.append('non-binary ignore flag: '+str(ann['id']))
        ignored = any(flags.values())
        x, y, w, h = ann['bbox']
        valid_box = bool(np.isfinite([x, y, w, h]).all() and w > 0 and h > 0)
        for name, image_ids in scopes.items():
            if iid in image_ids:
                row = counts[name].setdefault(category, Counter())
                row['annotations'] += 1
                row['valid_boxes'] += int(valid_box)
                row['invalid_boxes'] += int(not valid_box)
                row['flagged_any'] += int(ignored)
                row['valid_flagged_any'] += int(ignored and valid_box)
                for flag, value in flags.items():
                    row[flag] += int(bool(value))
        if iid not in scopes['all_source_images']:
            errors.append('annotation references unknown image: '+iid)
        if iid not in ids:
            continue
        if not np.isfinite([x, y, w, h]).all():
            errors.append('nonfinite bbox: '+str(ann['id']))
        if valid_box and ignored and ann['category_id'] != 1:
            unhandled.append(dict(id=ann['id'], image_id=iid, category_id=ann['category_id'], flags=flags))
        if valid_box and ann['category_id'] == 1:
            expected[iid][int(ignored)].append([x, y, x+w, y+h])
            if not ignored:
                normal_ids[iid].append(ann['id'])
    for image in data['images']:
        if str(image['id']) in ids:
            for key in ('ignore_regions', 'ignore_bboxes'):
                if image.get(key):
                    errors.append(f'image-level {key} requires explicit interpretation: {image["id"]}')
    if unhandled:
        errors.append('valid flagged non-person annotations require category-semantics review')
    derived_normal = derived_ignore = 0
    for iid in ids:
        if iid not in targets.by_image or iid not in previous_geometry:
            errors.append('missing image in targets/reference: '+iid)
            continue
        actual = targets.by_image[iid]
        old = previous_geometry[iid]
        for j, key in [(0, 'normal_boxes'), (1, 'ignored_boxes')]:
            raw = np.asarray(expected[iid][j], np.float32).reshape(-1, 4)
            derived = np.asarray(actual[j], np.float32).reshape(-1, 4)
            previous = np.asarray(old[key], np.float32).reshape(-1, 4)
            if not np.array_equal(raw, derived) or not np.array_equal(raw, previous):
                errors.append('raw/builder/P3 geometry mismatch: '+iid+'/'+key)
        if old['normal_ids'] != normal_ids[iid]:
            errors.append('normal annotation ID/order mismatch: '+iid)
        derived_normal += len(actual[0])
        derived_ignore += len(actual[1])
    return dict(status='PASS' if not errors else 'REVIEW_REQUIRED', categories=data.get('categories', []),
        counts=counts, derived_normal=derived_normal, derived_ignore=derived_ignore,
        ignored_zero_explanation='raw in-scope valid person annotations also contain zero ignore/uncertain/crowd flags'
            if derived_ignore == 0 and not errors else None,
        errors=errors, unhandled_flagged_annotations=unhandled,
        limitation='Checks this annotation file and its P3 derivation, not completeness of external annotations.')


def full_nms_trace(samples, pairs, threshold=.5):
    records = {}
    for sample in samples:
        keep, edges = nms_trace(sample.boxes_global, sample.scores, threshold)
        edge_set = {tuple(map(int, e)) for e in edges}
        if any((p['winner'], p['suppressed']) not in edge_set for p in pairs[sample.image_id]):
            raise ValueError('Frozen pair/NMS ordering differs: '+sample.image_id)
        records[sample.image_id] = dict(initial_order=sample.scores.argsort()[::-1].tolist(),
            keep=keep.tolist(), edges=edges.tolist())
    return records


def select_probe(samples, batches, pairs, limit=8):
    """Fixed training-only selection, no validation scores or mutable RNG used."""
    eligible = [i for i, d in enumerate(batches) if len(d['base']) and bool(d['valid'].any())]
    order = sorted(eligible, key=lambda i: hashlib.sha256(('p4-probe-v1:'+samples[i].image_id).encode()).hexdigest())
    chosen = []
    for kind, base_order in [('threshold_crossing', 'wrong'), ('same_gt_quality', 'wrong'),
                             ('threshold_crossing', 'correct'), ('same_gt_quality', 'correct')]:
        candidate = next((i for i in order if i not in chosen and any(
            p['kind'] == kind and p['base_order'] == base_order for p in pairs[samples[i].image_id])), None)
        if candidate is not None:
            chosen.append(candidate)
    chosen = (chosen + [i for i in order if i not in chosen])[:limit]
    if not chosen:
        raise ValueError('No valid training images for gradient probe')
    return chosen


def pair_components(outputs, data, pairs, alpha):
    """Subsets retain original group denominators, so contributions add up."""
    z = torch.logit(data['base'].float().clamp(1e-6, 1-1e-6)) + alpha*outputs['delta'].float()
    keys = ['same_gt_quality', 'threshold_crossing', 'base_correct', 'base_wrong', 'base_tie']
    if not pairs:
        return {key: z.sum()*0 for key in ['all']+keys}
    terms = data['pair_confidence']*F.softplus(z[data['pair_bad']]-z[data['pair_good']])
    groups, size = data['pair_group'], data['pair_group_count']
    counts = torch.bincount(groups, minlength=size).to(terms.dtype)
    normalized = terms/counts[groups]/size
    out = {'all': normalized.sum()}
    for key in keys:
        field, value = ('base_order', key[5:]) if key.startswith('base_') else ('kind', key)
        mask = torch.tensor([p[field] == value for p in pairs], dtype=torch.bool, device=z.device)
        out[key] = normalized[mask].sum()
    return out


def gradient_probe(model, batches, pairs_by_batch, config, pair_config, pair_weight):
    """Parameter gradients before optimizer normalization/clipping, no backward()."""
    training = model.training
    rng = torch.get_rng_state()
    model.eval()
    try:
        with torch.enable_grad():
            qfls, components = [], []
            for data, pairs in zip(batches, pairs_by_batch):
                out = model(data)
                qfls.append(quality_loss(out, data, config)[1])
                components.append(pair_components(out, data, pairs, pair_config.alpha))
            qfl = torch.stack(qfls).mean()
            losses = {key: torch.stack([c[key] for c in components]).mean() for key in components[0]}
            params = [p for p in model.parameters() if p.requires_grad]
            qg = torch.autograd.grad(qfl, params, retain_graph=True, allow_unused=True)
            qnorm2 = sum(float(g.double().square().sum()) for g in qg if g is not None)
            qnorm = qnorm2**.5
            records = {}
            for key, loss in losses.items():
                pg = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                norm = sum(float(g.double().square().sum()) for g in pg if g is not None)**.5
                dot = sum(float(a.double().mul(b.double()).sum()) for a, b in zip(qg, pg) if a is not None and b is not None)
                records[key] = dict(loss=float(loss.detach()), raw_gradient_norm=norm,
                    weighted_gradient_norm=pair_weight*norm,
                    weighted_to_qfl_norm=pair_weight*norm/qnorm if qnorm > 1e-12 else None,
                    cosine_to_qfl=float(np.clip(dot/(qnorm*norm), -1, 1)) if min(qnorm, norm) > 1e-12 else None,
                    weighted_cosine_to_qfl=float(np.clip(dot/(qnorm*norm), -1, 1))
                        if pair_weight > 0 and min(qnorm, norm) > 1e-12 else None)
            return dict(images=len(batches), pairs=sum(len(p) for p in pairs_by_batch),
                qfl_loss=float(qfl.detach()), qfl_gradient_norm=qnorm, pair_weight=pair_weight,
                pair_components=records, note='Fixed stratified training probe, not a full-training gradient estimate; before clipping.')
    finally:
        model.train(training)
        torch.set_rng_state(rng)


def train_roi(batches, config, pair_config, pair_weight, pair_signature, roi_mode, output_dir, validate):
    """Same updates/order as stage2.train_quality; validation may expose multiple views.

    A completed epoch atomically commits optimizer, scheduler, RNG, history and
    best-epoch metadata. A partial epoch is replayed from the previous commit.
    """
    from pathlib import Path
    if config.rank_weight != 0 or pair_weight < 0 or not np.isfinite(pair_weight):
        raise ValueError('P4 requires original QFL with a nonnegative finite pair weight')
    if config.device != 'cpu':
        raise ValueError('P4 controlled experiment uses CPU training')
    if not batches or not any(len(batch['base']) for batch in batches):
        raise ValueError('No training candidates')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    feature_dim = batches[0]['features'].shape[1]
    model = RoiRescorer(feature_dim, config, roi_mode)
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
        if saved['config'] != asdict(config) or saved['pair_config'] != asdict(pair_config) or saved['pair_weight'] != pair_weight or saved['pair_signature'] != pair_signature or saved['roi_mode'] != roi_mode:
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
        payload = dict(schema='p4-v1', roi_mode=roi_mode, origin='trained_from_initialization', model=model.state_dict(), config=asdict(config),
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
