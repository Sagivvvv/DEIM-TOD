"""Streaming, read-only training gradient census for frozen P4 models."""
import numpy as np
import torch

from .p4 import pair_components
from .stage2 import quality_loss

COMPONENTS = ('all', 'same_gt_quality', 'threshold_crossing',
              'base_correct', 'base_wrong', 'base_tie')


def vector_gradient(loss, params):
    if not loss.requires_grad:
        return torch.zeros(sum(p.numel() for p in params), dtype=torch.float64)
    gradients = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1).double().cpu()
                      for p, g in zip(params, gradients)])


def gradient_statistics(qg, pg, weight):
    qnorm, pnorm = float(qg.norm()), float(pg.norm())
    cosine = float(torch.dot(qg, pg) / (qnorm * pnorm)) if min(qnorm, pnorm) > 1e-12 else None
    if cosine is not None:
        cosine = float(np.clip(cosine, -1, 1))
    return dict(raw_gradient_norm=pnorm, weighted_gradient_norm=weight * pnorm,
                weighted_to_qfl_norm=weight * pnorm / qnorm if qnorm > 1e-12 else None,
                cosine_to_qfl=cosine, weighted_cosine_to_qfl=cosine if weight > 0 else None)


class Accumulator:
    def __init__(self, size):
        self.n = self.pairs = self.groups = 0
        self.gradients = {k: torch.zeros(size, dtype=torch.float64) for k in ('qfl',) + COMPONENTS}
        self.losses = {k: 0. for k in self.gradients}
        self.norm_sums = {k: 0. for k in self.gradients}
        self.cosines = {k: [] for k in COMPONENTS}

    def add(self, gradients, losses, pairs, groups, weight):
        self.n += 1
        self.pairs += pairs
        self.groups += groups
        for key, gradient in gradients.items():
            self.gradients[key] += gradient
            self.losses[key] += losses[key]
            self.norm_sums[key] += float(gradient.norm())
        for key in COMPONENTS:
            value = gradient_statistics(gradients['qfl'], gradients[key], weight)['cosine_to_qfl']
            if value is not None:
                self.cosines[key].append(value)

    def finish(self, weight):
        if not self.n:
            return dict(images=0, pairs=0, gt_groups=0, diagnostics=None)
        gradients = {k: v / self.n for k, v in self.gradients.items()}
        records = {}
        for key in COMPONENTS:
            values = self.cosines[key]
            records[key] = dict(loss=self.losses[key] / self.n,
                **gradient_statistics(gradients['qfl'], gradients[key], weight),
                per_image_cosine=dict(defined=len(values), negative=sum(v < 0 for v in values),
                    negative_fraction=sum(v < 0 for v in values) / len(values) if values else None,
                    quantiles=np.quantile(values, [0, .1, .25, .5, .75, .9, 1]).tolist() if values else []),
                vector_coherence=float(self.gradients[key].norm()) / self.norm_sums[key]
                    if self.norm_sums[key] > 1e-12 else None)
        return dict(images=self.n, pairs=self.pairs, gt_groups=self.groups,
                    qfl_loss=self.losses['qfl'] / self.n,
                    qfl_gradient_norm=float(gradients['qfl'].norm()), pair_weight=weight,
                    pair_components=records)


def gradient_census(model, batches, pair_lists, image_ids, config, pair_config,
                    pair_weight, original_probe_ids=(), progress=None):
    """Exact equal-image gradient mean on eligible training images at fixed weights.

    Pair terms keep P3's per-image/per-GT denominators. No averaging of cosine
    values replaces vector aggregation. Graphs are released after every image.
    This is NOT the accumulated Adam update across an epoch of changing weights.
    """
    if not (len(batches) == len(pair_lists) == len(image_ids)) or len(set(image_ids)) != len(image_ids):
        raise ValueError('Aligned unique image IDs and batches required')
    if config.device != 'cpu' or pair_weight < 0 or not np.isfinite(pair_weight):
        raise ValueError('CPU and finite nonnegative pair weight required')
    if not set(original_probe_ids) <= set(image_ids):
        raise ValueError('Original probe must be inside the supplied training split')
    params = [p for p in model.parameters() if p.requires_grad]
    size = sum(p.numel() for p in params)
    accumulators = {k: Accumulator(size) for k in ('all_training', 'pair_images', 'original_probe')}
    records, skipped = [], []
    training, rng = model.training, torch.get_rng_state().clone()
    model.eval()
    try:
        with torch.enable_grad():
            for index, (data, pairs, image_id) in enumerate(zip(batches, pair_lists, image_ids)):
                if not len(data['base']) or not bool(data['valid'].any()):
                    skipped.append(image_id)
                    continue
                out = model(data)
                terms = {'qfl': quality_loss(out, data, config)[1],
                         **pair_components(out, data, pairs, pair_config.alpha)}
                losses = {k: float(v.detach()) for k, v in terms.items()}
                if not all(np.isfinite(v) for v in losses.values()):
                    raise FloatingPointError('Nonfinite diagnostic loss: ' + image_id)
                gradients = {k: vector_gradient(v, params) for k, v in terms.items()}
                if not all(bool(torch.isfinite(v).all()) for v in gradients.values()):
                    raise FloatingPointError('Nonfinite diagnostic gradient: ' + image_id)
                groups = int(data['pair_group_count'])
                selected = ['all_training']
                if pairs:
                    selected.append('pair_images')
                if image_id in original_probe_ids:
                    selected.append('original_probe')
                for key in selected:
                    accumulators[key].add(gradients, losses, len(pairs), groups, pair_weight)
                records.append(dict(image_id=image_id, candidates=len(data['base']), pairs=len(pairs),
                    gt_groups=groups, losses=losses, qfl_gradient_norm=float(gradients['qfl'].norm()),
                    pair_counts={key: sum(p['base_order'] == key for p in pairs) for key in ('correct', 'wrong', 'tie')},
                    components={k: gradient_statistics(gradients['qfl'], gradients[k], pair_weight) for k in COMPONENTS}))
                del out, terms, gradients
                if progress and ((index + 1) % 50 == 0 or index + 1 == len(batches)):
                    progress(index + 1, len(batches))
    finally:
        model.train(training)
        torch.set_rng_state(rng)
    if not accumulators['all_training'].n:
        raise ValueError('No training images eligible under the original training rule')
    if set(original_probe_ids) & set(skipped):
        raise ValueError('Original P4 probe unexpectedly contains skipped images')
    summary = dict(cohorts={k: v.finish(pair_weight) for k, v in accumulators.items()},
                   skipped_image_ids=skipped, requested_images=len(image_ids),
                   interpretation='Frozen-weight, equal-image raw parameter gradient means before clipping; '
                   'original GT normalization within each image. Not Adam updates or causal proof. '
                   'Per-image cosine fractions exclude undefined (zero-norm) cases. '
                   'Vector coherence is norm(sum gradients)/sum(norm gradients).')
    return summary, records


def check_original_probe(actual, expected, rtol=3e-4, atol=2e-6):
    """Cross-check streaming sums against P4's jointly differentiated probe."""
    for key in ('images', 'pairs'):
        if actual[key] != expected[key]:
            raise ValueError('P4 probe count mismatch: ' + key)
    checks = [('qfl_loss', actual['qfl_loss'], expected['qfl_loss']),
              ('qfl_gradient_norm', actual['qfl_gradient_norm'], expected['qfl_gradient_norm'])]
    for component in COMPONENTS:
        for key in ('loss', 'raw_gradient_norm', 'weighted_to_qfl_norm', 'cosine_to_qfl'):
            checks.append((component + '/' + key, actual['pair_components'][component][key],
                           expected['pair_components'][component][key]))
    for label, a, b in checks:
        if a is None or b is None:
            if a is not None or b is not None:
                raise ValueError('P4 probe undefined value mismatch: ' + label)
        elif not np.isclose(a, b, rtol=rtol, atol=atol):
            raise ValueError(f'P4 probe mismatch {label}: {a} vs {b}')
    return dict(status='PASS', rtol=rtol, atol=atol, scalar_checks=len(checks))
