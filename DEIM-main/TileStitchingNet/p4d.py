"""D: a single residual local relation block on the full-ROI pointwise model."""
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .p1 import nms_trace, scaled_scores
from .p2 import atomic_json, atomic_save
from .p3 import competition_loss
from .p4 import RoiRescorer
from .stage2 import QualityConfig, quality_loss, prepare

ARCHITECTURE = 'local-residual-32-v1'


def install_initial_core(model, state):
    """Load epoch0 tensors without cross-PyTorch initialization rounding."""
    expected = {k for k in model.state_dict() if k.startswith('rescorer.')}
    if set(state) != expected:
        raise ValueError('Initial core keys differ from P4')
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu()
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError('Nonfinite initial core')
        digest.update(key.encode())
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.numpy().tobytes())
    model.load_state_dict(state, strict=False)
    if any(not torch.equal(v, model.state_dict()[k]) for k, v in state.items()):
        raise AssertionError('Initial core did not load exactly')
    return digest.hexdigest()


class RelationRescorer(RoiRescorer):
    """Existing 78D encoder/head plus sparse attention over existing graph edges.

    a_ij = softmax(MLP([h_i,h_j,g_ij]));
    h'_i = h_i + W_o sum_j a_ij ReLU(W_v[h_j-h_i,g_ij]).
    Self edges are excluded; rows without other neighbors have exactly zero
    relation contribution. W_o starts at zero. Node/edge inputs contain no GT.
    """
    def __init__(self, feature_dim, config):
        super().__init__(feature_dim, config, 'full')
        self.local = True
        d, width = config.hidden, 32
        self.relation_attention = nn.Sequential(nn.Linear(2*d+8, width), nn.ReLU(), nn.Linear(width, 1))
        self.relation_value = nn.Sequential(nn.Linear(d+8, width), nn.ReLU())
        self.relation_output = nn.Linear(width, d, bias=False)
        nn.init.zeros_(self.relation_output.weight)

    def forward(self, data):
        if not len(data['base']):
            return super().forward(data)
        h = self.rescorer.encoder(data['features'])
        j, geometry = data['neighbors'], data['geometry']
        valid = data['neighbor_mask'] & (j != torch.arange(len(h), device=h.device)[:, None])
        hi = h[:, None].expand(-1, j.shape[1], -1)
        hj = h[j]
        logits = self.relation_attention(torch.cat([hi, hj, geometry], -1)).squeeze(-1)
        attention = logits.masked_fill(~valid, -1e9).softmax(1) * valid.to(logits.dtype)
        attention = attention / attention.sum(1, keepdim=True).clamp_min(1e-12)
        message = (attention[..., None] * self.relation_value(torch.cat([hj-hi, geometry], -1))).sum(1)
        h = h + self.relation_output(message)
        delta = self.rescorer.head(h).squeeze(-1)
        base = data['base'].clamp(1e-6, 1-1e-6)
        base_logits = torch.logit(base)
        logits = base_logits + delta
        scores = (base + (logits.sigmoid()-base_logits.sigmoid())).clamp(0, 1)
        return dict(logits=logits, scores=scores, delta=delta, raw_delta=delta)


def load_relation_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema') != 'p4d-v1' or payload['architecture'] != ARCHITECTURE:
        raise ValueError('Expected a D relation checkpoint')
    model = RelationRescorer(payload['feature_dim'], QualityConfig(**payload['config']))
    model.load_state_dict(payload['model'], strict=True)
    return model.eval(), payload


def benchmark_postprocessor(model, samples, repeats=3):
    """CPU wall time from cached candidates/features through graph, scores, NMS/top200.

    Excludes detector/ROI extraction, disk I/O and GT evaluation. Data preparation
    and graph building are timed afresh every repetition, not retrieved from a
    validation tensor cache. Warmup uses at most eight nonempty validation images.
    """
    if repeats < 1 or model.config.device != 'cpu':
        raise ValueError('CPU benchmark and positive repeats required')
    previous_mode = model.training
    model.eval()
    rows = []
    def predict(sample):
        t0 = time.perf_counter()
        data = prepare(sample, model.config, model.local)
        t1 = time.perf_counter()
        output = model(data)
        scores = scaled_scores(sample.scores, output['delta'].cpu().numpy(), .25)
        t2 = time.perf_counter()
        keep = nms_trace(sample.boxes_global, scores)[0][:200]
        t3 = time.perf_counter()
        return dict(prepare_ms=(t1-t0)*1000, forward_ms=(t2-t1)*1000,
                    nms_ms=(t3-t2)*1000, total_ms=(t3-t0)*1000, output_boxes=len(keep))
    try:
        with torch.inference_mode():
            for sample in [s for s in samples if len(s.scores)][:8]:
                predict(sample)
            for repetition in range(repeats):
                for sample in samples:
                    rows.append(dict(image_id=sample.image_id, repeat=repetition,
                                     candidates=len(sample.scores), **predict(sample)))
    finally:
        model.train(previous_mode)
    summary = {key: dict(mean=float(np.mean([r[key] for r in rows])),
                         median=float(np.median([r[key] for r in rows])),
                         p95=float(np.quantile([r[key] for r in rows], .95)))
               for key in ['prepare_ms', 'forward_ms', 'nms_ms', 'total_ms']}
    return dict(images=len(samples), repeats=repeats, threads=torch.get_num_threads(),
                local=model.local, measurements=summary,
                scope='Cached candidates -> prepare/graph -> score -> NMS/top200; excludes detector and I/O.'), rows


def train_relation(batches, config, pair_config, pair_weight, pair_signature, graph_signature, output_dir, validate,
                   relation=True, initial_core=None):
    """Same updates/order as stage2.train_quality; validation may expose multiple views.

    A completed epoch atomically commits optimizer, scheduler, RNG, history and
    best-epoch metadata. A partial epoch is replayed from the previous commit.
    """
    roi_mode = "full"
    architecture = ARCHITECTURE if relation else "pointwise-control-v1"
    if config.rank_weight != 0 or pair_weight < 0 or not np.isfinite(pair_weight):
        raise ValueError('D requires original QFL with a nonnegative finite pair weight')
    if config.device != 'cpu':
        raise ValueError('D controlled experiment uses CPU training')
    if not batches or not any(len(batch['base']) for batch in batches):
        raise ValueError('No training candidates')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    feature_dim = batches[0]['features'].shape[1]
    model = RelationRescorer(feature_dim, config) if relation else RoiRescorer(feature_dim, config, roi_mode)
    initial_core_hash = install_initial_core(model, initial_core) if initial_core is not None else None
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
        if saved.get('initial_core_hash') != initial_core_hash or saved.get('architecture') != architecture or saved.get('graph_signature') != graph_signature or saved['config'] != asdict(config) or saved['pair_config'] != asdict(pair_config) or saved['pair_weight'] != pair_weight or saved['pair_signature'] != pair_signature or saved['roi_mode'] != roi_mode:
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
        payload = dict(schema='p4d-v1', architecture=architecture, graph_signature=graph_signature, initial_core_hash=initial_core_hash, roi_mode=roi_mode, origin='trained_from_initialization', model=model.state_dict(), config=asdict(config),
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
