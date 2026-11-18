"""Audit returned P4 artifacts; does not recompute official AP or training gradients."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = Path(r'C:\Users\19333\Downloads\p4_reports.zip')
ARCHIVE = HERE / 'p4_reports.zip'
if not ARCHIVE.exists():
    shutil.copy2(SOURCE, ARCHIVE)
sha = lambda b: hashlib.sha256(b).hexdigest()
evidence = {'zip_sha256': sha(ARCHIVE.read_bytes()), 'limitations': [
    'Official AP and parameter gradients are reported values, not independently recomputed.',
    'Single seed, 149 validation images; no new test evaluation.']}
with zipfile.ZipFile(ARCHIVE) as z:
    assert z.testzip() is None
    read = lambda n: json.loads(gzip.decompress(z.read(n)) if n.endswith('.gz') else z.read(n))
    s, cfg = read('p4_summary.json'), read('run_config.json')
    assert s['status'] == 'P4_ROI_COMPLETED'
    assert s['input_files_unchanged'] and not s['input']['test_evaluated']
    for path, digest in cfg['source_hashes'].items():
        assert sha((ROOT / path).read_bytes()) == digest, path
    evidence['source_hashes_verified'] = len(cfg['source_hashes'])
    split = read('split.json')
    assert len(split['train_ids']) == 597 and len(split['val_ids']) == 149
    assert not set(split['train_ids']) & set(split['val_ids'])
    assert set(read('gradient_probe_selection.json')['image_ids']) <= set(split['train_ids'])
    evidence['methods'] = {}
    hashes = 0
    for method, result in s['results'].items():
        history = read(method + '/history.json')
        assert [r['epoch'] for r in history] == list(range(19))
        best = max(history, key=lambda r: r['validation'][method]['metrics']['Tiny_AP50'])
        assert best['epoch'] == result['best_epoch']
        assert best['validation'][method]['metrics'] == result['best']['before_nms']['metrics']
        assert sha(z.read(result['best_checkpoint'])) == result['best_checkpoint_sha256']
        for tag in ['best', 'last']:
            detail = read(method + '_' + tag + '.json')
            for path, digest in detail['artifact_hashes'].items():
                assert sha(z.read(path)) == digest, path
                hashes += 1
        evidence['methods'][method] = {
            'epoch': result['best_epoch'], 'parameters': result['parameter_count'],
            'best': result['best']['before_nms']['metrics'],
            'last': result['last']['before_nms']['metrics'],
            'best_gradient_probe': read(method + '_best_gradients.json')}
    evidence['artifact_hashes_verified'] = hashes
    with zipfile.ZipFile(HERE.parent / 'p3_20261117' / 'p3_reports.zip') as prior:
        for current, old in [('A_full', 'A025'), ('C_full', 'C020')]:
            previous = json.loads(prior.read(old + '/history.json'))
            for a, b in zip(read(current + '/history.json'), previous):
                assert a['validation'][current]['metrics'] == b['validation'][old]['metrics']
        for part in ['train', 'validation']:
            name = part + '_pairs.json.gz'
            assert read(name) == json.loads(gzip.decompress(prior.read(name)))
    evidence['p3_reference_metrics_epochs_verified'] = 38
    evidence['p3_pair_lists_identical'] = True
    source = read('source_annotations.json.gz')
    assert len(source['annotations']) == 18433
    assert all(a['category_id'] == 1 and not any(a.get(k, False) for k in ['ignore', 'uncertain', 'iscrowd']) for a in source['annotations'])
    evidence['annotation_audit'] = s['annotation_audit']
    evidence['nms_replays'] = {}
    for part in ['train', 'validation']:
        geometries = read(part + '_pair_geometry.json.gz')
        traces = read(part + '_nms_trace.json.gz')
        pairs = read(part + '_pairs.json.gz')['pairs']
        count = 0
        for image_id, g in geometries.items():
            b = np.asarray(g['boxes'], dtype=np.float32).reshape(-1, 4)
            scores = np.asarray(g['scores'])
            trace = traces[image_id]
            order = np.asarray(trace['initial_order'], dtype=np.int64)
            assert sorted(order.tolist()) == list(range(len(b)))
            assert np.all(np.diff(scores[order]) <= 0)
            area = np.maximum(b[:, 2:] - b[:, :2], 0).prod(axis=1)
            keep, edges = [], []
            while len(order):
                i = int(order[0]); keep.append(i); rest = order[1:]
                wh = np.maximum(np.minimum(b[i, 2:], b[rest, 2:]) - np.maximum(b[i, :2], b[rest, :2]), 0)
                inter = wh.prod(axis=1)
                overlap = inter / np.maximum(area[i] + area[rest] - inter, 1e-8)
                removed = overlap > .5
                edges.extend([[i, int(j)] for j in rest[removed]])
                order = rest[~removed]
            assert keep == trace['keep'], image_id
            assert edges == trace['edges'], image_id
            edge_set = set(map(tuple, edges))
            assert all((p['winner'], p['suppressed']) in edge_set for p in pairs[image_id])
            count += len(edges)
        evidence['nms_replays'][part] = {'images': len(traces), 'edges': count}
    evidence['comparisons'] = read('ablation_comparisons.json')
evidence['status'] = 'PASS'
(HERE / 'review_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps({k: v for k, v in evidence.items() if k not in ['methods', 'annotation_audit', 'comparisons']}, ensure_ascii=False, indent=2))
