"""Frozen P5 comparison protocol, paired seed statistics and held-checkpoint AP intervals."""
from pathlib import Path
import re

import numpy as np

SEEDS = (42, 43, 44)
METHODS = {'A_full': 0., 'C_full': .2}
METRICS = ('Tiny_AP50', 'Tiny_AP75', 'AP', 'AP50', 'AP75')
PROTOCOL = dict(split_seed=42, training_seeds=list(SEEDS), pair_weights=METHODS,
    alpha=.25, nms_iou=.5, max_dets=200, epochs=18,
    select_epoch='first maximum validation before-NMS Tiny_AP50, epochs 0..18',
    # A is the simpler supervision. A tie or an inconclusive result retains A.
    require_all_three_positive=True, secondary_tolerance_points=.10,
    select_family='C if all 3 paired Tiny_AP50 gains > 0 and mean C-A for Tiny_AP75 and AP >= -0.10 points, else A',
    traditional='highest validation Tiny_AP50 among four fixed methods; first declared method wins exact ties',
    test='evaluate all six preselected checkpoints individually; report seed means, no best test seed or ensemble',
    interval='1000 paired cluster bootstrap draws, fixed checkpoints, Tiny_AP50, RNG seed 42')


def describe(values):
    x = np.asarray(values, float)
    return dict(mean=float(x.mean()), sample_sd=float(x.std(ddof=1)) if len(x) > 1 else None,
                values=x.tolist())


def summarize(results, seeds, traditional):
    families, paired = {}, {}
    for family in METHODS:
        rows = [results[f'{family}_{s}'] for s in seeds]
        families[family] = {key: describe([m[key] for m in rows]) for key in METRICS}
        families[family]['Tiny_AP50_minus_reference'] = describe([m['Tiny_AP50']-results[traditional]['Tiny_AP50'] for m in rows])
    for key in METRICS:
        paired[key] = describe([results[f'C_full_{s}'][key]-results[f'A_full_{s}'][key] for s in seeds])
    return dict(seeds=list(seeds), reference=traditional, families=families, paired_C_minus_A=paired)


def choose_family(summary):
    d = summary['paired_C_minus_A']
    passed = len(d['Tiny_AP50']['values']) == len(SEEDS) and all(x > 0 for x in d['Tiny_AP50']['values']) and all(
        d[k]['mean'] >= -PROTOCOL['secondary_tolerance_points'] for k in ('Tiny_AP75', 'AP'))
    return 'C_full' if passed else 'A_full'


def sampling_group(image):
    for field in ('video_id', 'sequence_id'):
        if image.get(field) is not None:
            return f'{field}:{image[field]}'
    match = re.fullmatch(r'(.+_V\d+)_I\d+', Path(image['file_name']).stem, re.I)
    return 'video_filename:'+match[1] if match else 'image:'+str(image['id'])


def ap_input(trace, scores, ids):
    rows = []
    for iid in ids:
        row = trace[iid].get('tiny@0.5', {})
        idx = np.asarray(row.get('indices', [])[:200], dtype=int)
        keep = ~np.asarray(row.get('ignored', [])[:200], dtype=bool)
        tp = np.asarray(row.get('matches', [])[:200], dtype=int) > 0
        rows.append((scores[iid][idx][keep], tp[keep], len(row.get('eligible_gt_ids', []))))
    return rows


def sampled_ap(rows, indices):
    scores, hits, eligible = zip(*(rows[i] for i in indices))
    if not sum(eligible):
        return np.nan
    tp = np.concatenate(hits)[np.argsort(-np.concatenate(scores), kind='stable')].cumsum()
    precision = tp / np.arange(1, len(tp)+1)
    envelope = np.r_[np.maximum.accumulate(precision[::-1])[::-1], 0.]
    return float(100 * envelope[np.searchsorted(tp / sum(eligible), np.linspace(0, 1, 101))].mean())


def paired_intervals(inputs, metrics, ids, groups, seeds, reference, repetitions=1000):
    buckets = {}
    for i, iid in enumerate(ids):
        buckets.setdefault(groups[iid], []).append(i)
    members = list(buckets.values())
    keys = [reference] + [f'{f}_{s}' for f in METHODS for s in seeds]
    for name in keys:
        if not np.isclose(sampled_ap(inputs[name], range(len(ids))), metrics[name]['Tiny_AP50'], rtol=0, atol=1e-5):
            raise ValueError('Bootstrap AP does not reproduce official Tiny_AP50 for '+name)
    rng = np.random.default_rng(42)
    deltas = {k: [] for k in ['C_minus_A', 'A_minus_reference', 'C_minus_reference']}
    for _ in range(repetitions):
        indices = [i for g in rng.integers(len(members), size=len(members)) for i in members[g]]
        indices.sort()  # Keep official image ordering when detections tie in score.
        values = {k: sampled_ap(inputs[k], indices) for k in keys}
        if not all(np.isfinite(v) for v in values.values()):
            continue
        a, c = [np.mean([values[f'{family}_{s}'] for s in seeds]) for family in METHODS]
        for k, value in zip(deltas, [c-a, a-values[reference], c-values[reference]]):
            deltas[k].append(value)
    return dict(metric='Tiny_AP50 points, fixed weights averaged across seeds', groups=len(members),
                draws=repetitions, valid_draws=len(deltas['C_minus_A']),
                percentile_95={k: np.percentile(v, [2.5, 97.5]).tolist() if v else None for k, v in deltas.items()})
