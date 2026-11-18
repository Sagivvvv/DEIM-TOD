"""Review P4-D using saved scores and evaluator traces. Never load pickle weights."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np


here = Path(__file__).resolve().parent
archive = here / 'p4d_reports.zip'
if not archive.exists():
    shutil.copyfile(Path.home() / 'Downloads/p4d_reports.zip', archive)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
evidence = {'report_sha256': sha(archive.read_bytes())}


def ap_rows(trace, scores, ids, key):
    result = []
    for image in ids:
        row = trace[image].get(key, {})
        indices = np.array(row.get('indices', [])[:200], dtype=int)
        include = ~np.array(row.get('ignored', [])[:200], dtype=bool)
        tp = np.array(row.get('matches', [])[:200], dtype=int) > 0
        result.append((scores[image][indices][include], tp[include], len(row.get('eligible_gt_ids', []))))
    return result


def sampled_ap(rows, draws):
    scores, tp, gt = zip(*(rows[i] for i in draws))
    tp = np.concatenate(tp)[np.argsort(-np.concatenate(scores), kind='stable')]
    cumulative = tp.cumsum()
    recall = cumulative / sum(gt)
    precision = cumulative / np.arange(1, len(tp) + 1)
    envelope = np.r_[np.maximum.accumulate(precision[::-1])[::-1], 0.]
    return 100 * envelope[np.searchsorted(recall, np.linspace(0., 1., 101))].mean()


with zipfile.ZipFile(archive) as z, zipfile.ZipFile(here.parent / 'p4_20261117/p4_reports.zip') as old:
    def read(name):
        raw = z.read(name)
        return json.loads(gzip.decompress(raw) if name.endswith('.gz') else raw)

    assert z.testzip() is None
    summary, cfg = read('p4d_summary.json'), read('run_config.json')
    assert summary['status'] == 'P4D_TRAINING_COMPLETED' and not cfg['smoke']
    for name, digest in summary['artifact_hashes'].items():
        assert sha(z.read(name)) == digest, name
    for name, digest in cfg['source_hashes'].items():
        assert sha((here.parents[2] / name).read_bytes()) == digest == sha(z.read('sources/' + name)), name
    for name, digest in cfg['reference_hashes'].items():
        raw = old.read(name)
        assert sha(gzip.decompress(raw) if name.endswith('.gz') else raw) == digest, name
    for method, result in summary['results'].items():
        history = read(method + '/history.json')
        assert [r['epoch'] for r in history] == list(range(19))
        assert max(history, key=lambda r: r['validation'][method]['metrics']['Tiny_AP50'])['epoch'] == result['best_epoch']
        if method != 'D_local':
            previous = json.loads(old.read(method + '/history.json'))
            for row in history:
                epoch = row['epoch']
                name = f'{method}/epoch_{epoch:02d}.pth'
                assert z.read(name) == old.read(name)
                assert row['validation'][method]['metrics'] == previous[epoch]['validation'][method]['metrics']
        latency = read(method + '_latency_samples.json.gz')
        assert len({(r['image_id'], r['repeat']) for r in latency}) == len(latency) == 447
        for key, values in summary['latency'][method]['measurements'].items():
            raw = [r[key] for r in latency]
            np.testing.assert_allclose([values['mean'], values['median'], values['p95']], [np.mean(raw), np.median(raw), np.percentile(raw, 95)])
    ids = sorted(read('split.json')['val_ids'], key=int)
    selected_methods = ['C_full', 'D_local']
    traces = {m: read(m + '_best_before_nms_matches.json.gz') for m in selected_methods}
    evidence['paired_hits'] = {}
    for key in ['all@0.5', 'all@0.75', 'tiny@0.5', 'tiny@0.75']:
        def hitset(m, budget):
            result = set()
            for i in ids:
                row = traces[m][i].get(key, {})
                result.update((i, gt) for gt, ignore in zip(row.get('matches', [])[:budget], row.get('ignored', [])[:budget]) if gt and not ignore)
            return result
        a, b = (hitset(m, 200) for m in selected_methods)
        evidence['paired_hits'][key] = dict(gained=len(b-a), lost=len(a-b), C=len(a), D=len(b))
        for m in selected_methods:
            counts = summary['results'][m]['best']['before_nms']['matching_and_budget'][key]
            assert counts['new_hit_gt'] == len(hitset(m, 200))
            assert counts['new_budget_lost_gt'] == len(hitset(m, None))-len(hitset(m, 200))
    evidence['graph'] = {}
    for scope in ['train', 'validation']:
        graph = np.load(io.BytesIO(z.read(scope + '_graph.npz')), allow_pickle=False)
        pairs = read(scope + '_pairs.json.gz')['pairs']
        covered = total = 0
        for i, image_id in enumerate(graph['image_ids']):
            shape = tuple(graph['shapes'][i]); sl = slice(*graph['offsets'][i:i+2])
            j = graph['neighbors'][sl].reshape(shape); mask = graph['neighbor_mask'][sl].reshape(shape)
            for p in pairs[str(image_id)]:
                g, b = p['good'], p['bad']; total += 1
                covered += int(g != b and (b in j[g, mask[g]] or g in j[b, mask[b]]))
        stats = read(scope + '_graph.json')['summary']
        assert (total, covered) == (stats['pairs'], stats['either_direction'])
        evidence['graph'][scope] = dict(total=total, covered=covered)
    scores = {}
    for m in selected_methods:
        data = np.load(io.BytesIO(z.read(m + '_best_scores.npz')), allow_pickle=False)
        scores[m] = {str(i): data['scores'][a:b] for i, a, b in zip(data['image_ids'], data['offsets'], data['offsets'][1:])}
    draws = np.random.default_rng(42).integers(len(ids), size=(2000, len(ids)))
    evidence['paired_image_bootstrap'] = {'repetitions': len(draws), 'seed': 42, 'fixed_checkpoints': True}
    for key, metric in [('tiny@0.5', 'Tiny_AP50'), ('tiny@0.75', 'Tiny_AP75')]:
        rows = [ap_rows(traces[m], scores[m], ids, key) for m in selected_methods]
        for r, m in zip(rows, selected_methods):
            assert abs(sampled_ap(r, range(len(ids))) - summary['results'][m]['best']['before_nms']['metrics'][metric]) < 1e-6
        delta = [sampled_ap(rows[1], indices)-sampled_ap(rows[0], indices) for indices in draws]
        evidence['paired_image_bootstrap'][metric] = np.percentile(delta, [2.5, 97.5]).tolist()
    evidence.update(source_checks=len(cfg['source_hashes']), artifact_checks=len(summary['artifact_hashes']),
                    control_checkpoints=38, status='PASS')
(here / 'review_evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf8')
print(json.dumps(evidence, indent=2))
