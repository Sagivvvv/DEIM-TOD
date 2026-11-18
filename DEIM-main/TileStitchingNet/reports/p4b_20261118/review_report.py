"""Audit returned P4-B metrics, provenance, frozen pairs and margin arithmetic."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARCHIVE = HERE / 'p4b_reports.zip'
if not ARCHIVE.exists():
    shutil.copy2(Path(r'C:\Users\19333\Downloads\p4b_reports.zip'), ARCHIVE)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
evidence = dict(zip_sha256=sha(ARCHIVE.read_bytes()),
    limitations=['Official AP is taken from returned evaluation, not rerun locally.',
                 'Single seed and validation split; no new test evaluation.'])
with zipfile.ZipFile(ARCHIVE) as z, zipfile.ZipFile(HERE.parent / 'p4_20261117/p4_reports.zip') as old:
    assert z.testzip() is None
    def read(name):
        return json.loads(gzip.decompress(z.read(name)) if name.endswith('.gz') else z.read(name))
    summary, config = read('p4b_summary.json'), read('run_config.json')
    assert summary['status'] == 'P4B_TRAINING_COMPLETED' and not config['smoke']
    assert config['margin'] == .1 and config['alpha'] == .25
    for name, digest in summary['artifact_hashes'].items():
        assert sha(z.read(name)) == digest, name
    for name, digest in config['source_hashes'].items():
        assert sha((ROOT / name).read_bytes()) == digest
        assert sha(z.read('sources/' + name)) == digest
    for name, digest in config['reference_hashes'].items():
        raw = old.read(name)
        assert sha(gzip.decompress(raw) if name.endswith('.gz') else raw) == digest
    assert read('split.json') == json.loads(old.read('split.json'))
    histories = {}
    for method, result in summary['results'].items():
        h = read(method + '/history.json')
        histories[method] = h
        assert [r['epoch'] for r in h] == list(range(19))
        best = max(h, key=lambda r: r['validation'][method]['metrics']['Tiny_AP50'])
        assert best['epoch'] == result['best_epoch'] == 10
        assert best['validation'][method]['metrics'] == result['best']['before_nms']['metrics']
        assert sha(z.read(result['best_checkpoint'])) == result['best_checkpoint_sha256']
        if method != 'M010':
            previous = json.loads(old.read(method + '/history.json'))
            for a, b in zip(h, previous):
                assert a['validation'][method]['metrics'] == b['validation'][method]['metrics']
                name = f"{method}/epoch_{a['epoch']:02d}.pth"
                assert z.read(name) == old.read(name)
    pairs = {scope: read(scope + '_pairs.json.gz')['pairs'] for scope in ['train', 'validation']}
    for scope in pairs:
        assert read(scope + '_pairs.json.gz') == json.loads(gzip.decompress(old.read(scope + '_pairs.json.gz')))
    verified_pairs = 0
    for method in summary['results']:
        for tag, epoch in [('best', 10), ('last', 18)]:
            for scope in pairs:
                detail = read(f'{method}_{tag}_{scope}_margins.json.gz')
                stats = detail['summary']
                assert stats == histories[method][epoch]['validation'][method]['margin_diagnostics'][scope]
                active = correct = repaired = broken = satisfied = 0
                hinge_sum = soft_sum = 0.
                for row in detail['images']:
                    ps = pairs[scope][row['image_id']]
                    gap = np.asarray(row['gap'], dtype=np.float64)
                    assert len(gap) == len(ps)
                    verified_pairs += len(ps)
                    active += int((gap < .1).sum()); satisfied += int((gap >= .1).sum())
                    correct += int((gap > 0).sum())
                    repaired += sum(p['base_order'] == 'wrong' and g > 0 for p, g in zip(ps, gap))
                    broken += sum(p['base_order'] == 'correct' and g < 0 for p, g in zip(ps, gap))
                    groups = {}
                    for p, g in zip(ps, gap):
                        groups.setdefault(p['gt_id'], []).append((p['weight'] * max(0, .1-g), p['weight'] * np.logaddexp(0, -g)))
                    if groups:
                        means = np.array([np.mean(v, axis=0) for v in groups.values()]).mean(axis=0)
                        hinge_sum += means[0]; soft_sum += means[1]
                counts = stats['groups']
                assert (active, satisfied, correct) == tuple(counts['all'][k] for k in ['active', 'satisfied', 'correct'])
                assert repaired == counts['base_wrong']['correct'] and broken == counts['base_correct']['wrong']
                assert np.isclose(hinge_sum / stats['eligible_images'], stats['hinge_loss'], rtol=1e-5, atol=1e-7)
                assert np.isclose(soft_sum / stats['eligible_images'], stats['softplus_loss'], rtol=1e-5, atol=1e-7)
    traces = {m: read(m + '_best_before_nms_matches.json.gz') for m in summary['results']}
    changes = {}
    for method in ['A_full', 'C_full']:
        changes[method] = {}
        for metric in ['all@0.5', 'tiny@0.5', 'tiny@0.75']:
            def hits(trace):
                return {(i, g) for i, r in trace.items() if metric in r
                        for g, ignored in zip(r[metric]['matches'][:200], r[metric]['ignored'][:200]) if g > 0 and not ignored}
            before, after = hits(traces[method]), hits(traces['M010'])
            assert len(after) == summary['results']['M010']['best']['before_nms']['matching_and_budget'][metric]['new_hit_gt']
            changes[method][metric] = dict(added=len(after-before), lost=len(before-after), old=len(before), new=len(after))
    evidence.update(status='PASS', source_hashes_verified=len(config['source_hashes']),
        artifact_hashes_verified=len(summary['artifact_hashes']), frozen_control_snapshots_verified=38,
        margin_pair_records_verified=verified_pairs, selected_epoch=10,
        direct_top200_gt_changes=changes, results=summary['results'], comparisons=summary['comparisons'])
(HERE / 'review_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps({k: v for k, v in evidence.items() if k not in ['results', 'comparisons']}, ensure_ascii=False, indent=2))
