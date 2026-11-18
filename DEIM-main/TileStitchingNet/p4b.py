"""P4-B: finite deployment-logit margin, with the frozen P3 pair denominators."""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .p2 import atomic_json, atomic_save
from .p3 import competition_loss
from .p4 import RoiRescorer
from .stage2 import QualityConfig, quality_loss

MARGIN = .1
PAIR_WEIGHT = .2


def deployment_gap(outputs, data, alpha=.25):
    z = torch.logit(data['base'].float().clamp(1e-6, 1-1e-6)) + alpha * outputs['delta'].float()
    return z[data['pair_good']] - z[data['pair_bad']]


def margin_loss(outputs, data, alpha=.25, margin=MARGIN):
    """Mean GT-group mean w*relu(margin-gap); inactive pairs stay in denominator."""
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError('margin must be finite and positive')
    if not data['pair_group_count']:
        return outputs['delta'].sum() * 0
    terms = data['pair_confidence'] * F.relu(margin - deployment_gap(outputs, data, alpha))
    groups, size = data['pair_group'], data['pair_group_count']
    counts = torch.bincount(groups, minlength=size).to(terms.dtype)
    return (terms.new_zeros(size).scatter_add(0, groups, terms) / counts.clamp_min(1)).mean()


def margin_diagnostics(model, batches, pair_lists, image_ids, alpha=.25, margin=MARGIN):
    """Inference-only per-pair statistics, no altered training labels or gradients."""
    if not (len(batches) == len(pair_lists) == len(image_ids)):
        raise ValueError('Pair diagnostics inputs are not aligned')
    keys = ['all', 'base_correct', 'base_wrong', 'base_tie', 'same_gt_quality', 'threshold_crossing']
    counts = {k: dict(pairs=0, active=0, satisfied=0, correct=0, wrong=0, tie=0,
                      base_margin_satisfied=0, base_satisfied_now_active=0) for k in keys}
    gaps = {k: [] for k in keys}
    rows, eligible, soft_sum, hinge_sum = [], 0, 0., 0.
    mode = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for image_id, data, pairs in zip(image_ids, batches, pair_lists):
                output = model(data)
                gap = deployment_gap(output, data, alpha).cpu().numpy()
                base_z = torch.logit(data['base'].float().clamp(1e-6, 1-1e-6))
                base_gap = (base_z[data['pair_good']] - base_z[data['pair_bad']]).cpu().numpy()
                if not np.isfinite(gap).all():
                    raise FloatingPointError('Nonfinite deployment gap')
                # Same image eligibility as the nonzero-pair P4/P4-B trainer.
                if len(data['base']) and (bool(data['valid'].any()) or data['pair_group_count']):
                    eligible += 1
                    soft_sum += float(competition_loss(output, data, alpha))
                    hinge_sum += float(margin_loss(output, data, alpha, margin))
                rows.append(dict(image_id=image_id, gap=gap.tolist(), base_gap=base_gap.tolist()))
                for i, pair in enumerate(pairs):
                    d, base = float(gap[i]), float(base_gap[i])
                    for key in ['all', 'base_' + pair['base_order'], pair['kind']]:
                        row = counts[key]
                        row['pairs'] += 1
                        row['active'] += int(d < margin)
                        row['satisfied'] += int(d >= margin)
                        row['correct'] += int(d > 0)
                        row['wrong'] += int(d < 0)
                        row['tie'] += int(d == 0)
                        row['base_margin_satisfied'] += int(base >= margin)
                        row['base_satisfied_now_active'] += int(base >= margin and d < margin)
                        gaps[key].append(d)
    finally:
        model.train(mode)
    for key, row in counts.items():
        row['active_fraction'] = row['active'] / row['pairs'] if row['pairs'] else None
        row['gap_quantiles'] = np.quantile(gaps[key], [0, .1, .25, .5, .75, .9, 1]).tolist() if gaps[key] else []
    return dict(margin=margin, alpha=alpha, groups=counts, eligible_images=eligible,
                softplus_loss=soft_sum / eligible if eligible else None,
                hinge_loss=hinge_sum / eligible if eligible else None,
                note='Counts are unweighted pair counts; losses use original per-image/GT denominators.'), rows


def load_p4b_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema') != 'p4b-v1' or payload['roi_mode'] != 'full':
        raise ValueError('Expected full-ROI P4-B checkpoint')
    model = RoiRescorer(payload['feature_dim'], QualityConfig(**payload['config']), 'full')
    model.load_state_dict(payload['model'], strict=True)
    return model.eval(), payload


def train_margin(batches, config, pair_config, pair_weight, pair_signature, output_dir, validate,
                 margin=MARGIN, objective="hinge"):
    """Same updates/order as stage2.train_quality; validation may expose multiple views.

    A completed epoch atomically commits optimizer, scheduler, RNG, history and
    best-epoch metadata. A partial epoch is replayed from the previous commit.
    """
    roi_mode = "full"
    if objective not in ("hinge", "softplus") or not np.isfinite(margin) or margin <= 0:
        raise ValueError("Invalid P4-B objective/margin")
    if config.rank_weight != 0 or pair_weight < 0 or not np.isfinite(pair_weight):
        raise ValueError('P4-B requires original QFL with a nonnegative finite pair weight')
    if config.device != 'cpu':
        raise ValueError('P4-B controlled experiment uses CPU training')
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
        if saved.get('objective') != objective or saved.get('margin') != margin or saved['config'] != asdict(config) or saved['pair_config'] != asdict(pair_config) or saved['pair_weight'] != pair_weight or saved['pair_signature'] != pair_signature or saved['roi_mode'] != roi_mode:
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
        payload = dict(schema='p4b-v1', objective=objective, margin=margin, roi_mode=roi_mode, origin='trained_from_initialization', model=model.state_dict(), config=asdict(config),
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
            rank = (margin_loss(outputs, data, pair_config.alpha, margin) if objective == "hinge"
                    else competition_loss(outputs, data, pair_config.alpha))
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
