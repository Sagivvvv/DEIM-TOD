"""P2 pointwise models and epoch-resumable training using the existing QFL."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .stage2 import QualityConfig, QualityRescorer, quality_loss


class BoundedQualityRescorer(nn.Module):
    def __init__(self, feature_dim, config, residual_bound=None):
        super().__init__()
        if residual_bound is not None and (not np.isfinite(residual_bound) or residual_bound <= 0):
            raise ValueError('residual_bound must be positive and finite')
        self.config = config
        self.feature_dim = feature_dim
        self.local = False
        self.residual_bound = residual_bound
        self.rescorer = QualityRescorer(feature_dim, config, local=False)

    def forward(self, data):
        result = self.rescorer(data)
        raw_delta = result['delta']
        if self.residual_bound is not None:
            bound = self.residual_bound
            # Unit derivative at zero preserves the initial residual gradient.
            delta = bound * torch.tanh(raw_delta / bound)
            base = data['base'].clamp(1e-6, 1-1e-6)
            base_logits = torch.logit(base)
            logits = base_logits + delta
            # Use the original operation grouping for bitwise zero identity.
            scores = (base + (logits.sigmoid() - base_logits.sigmoid())).clamp(0, 1)
            result = dict(logits=logits, scores=scores, delta=delta)
        return {**result, 'raw_delta': raw_delta}


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def load_p2_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema') != 'p2-v1':
        raise ValueError('Expected a P2 checkpoint')
    config = QualityConfig(**payload['config'])
    model = BoundedQualityRescorer(payload['feature_dim'], config, payload['residual_bound'])
    model.load_state_dict(payload['model'], strict=True)
    return model.eval(), payload


def train_pointwise(batches, config, bound, output_dir, validate):
    """Same updates/order as stage2.train_quality; validation may expose multiple views.

    A completed epoch atomically commits optimizer, scheduler, RNG, history and
    best-epoch metadata. A partial epoch is replayed from the previous commit.
    """
    if config.device != 'cpu':
        raise ValueError('P2 controlled experiment uses CPU training')
    if not batches or not any(len(batch['base']) for batch in batches):
        raise ValueError('No training candidates')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    feature_dim = batches[0]['features'].shape[1]
    model = BoundedQualityRescorer(feature_dim, config, bound)
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
        if saved['config'] != asdict(config) or saved['residual_bound'] != bound:
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
        payload = dict(schema='p2-v1', model=model.state_dict(), config=asdict(config),
                       feature_dim=feature_dim, residual_bound=bound, epoch=epoch,
                       formula='raw' if bound is None else 'a*tanh(raw_delta/a)',
                       label_counts=counts, validation=views)
        atomic_save(payload, output_dir/f'epoch_{epoch:02d}.pth')
        state = dict(payload, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                     history=history, best=best)
        atomic_save(state, resume_path)
        atomic_json(history, output_dir/'history.json')
        values = ' '.join(f'{name}={view["metrics"]["Tiny_AP50"]:.4f}' for name, view in views.items())
        print(f'[{output_dir.name} {epoch:02d}/{config.epochs}] loss={record.get("loss", 0):.6f} {values}', flush=True)

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
            if not len(data['base']) or not data['valid'].any():
                continue
            outputs = model(data)
            loss, qfl, rank = quality_loss(outputs, data, config)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            losses.append([loss.item(), qfl.item(), rank.item()])
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
                           rank=float(means[2]), optimizer_steps=steps))

    for name, selection in best.items():
        payload = torch.load(output_dir/f'epoch_{selection["epoch"]:02d}.pth', map_location='cpu', weights_only=True)
        payload.update(alpha=selection['alpha'], selected_view=name, selection_metric='Tiny_AP50_before_nms')
        atomic_save(payload, output_dir/(name+'_best.pth'))
    atomic_json(history, output_dir/'history.json')
    return history, best
