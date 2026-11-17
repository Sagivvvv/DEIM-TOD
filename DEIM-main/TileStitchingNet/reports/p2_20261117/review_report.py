"""Audit the returned P2 archive without rerunning training or official AP."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch


def sha(data):
    return hashlib.sha256(data).hexdigest()


def hits(row, budget=None):
    return {g for g, ignored in zip(row.get('matches', [])[:budget],
                                   row.get('ignored', [])[:budget]) if g > 0 and not ignored}


def main(path):
    target = Path(__file__).resolve().parent
    root = target.parents[2]
    old_path = target.parent / 'p0r_20261116/p0r_reports.zip'
    checks = {}
    with zipfile.ZipFile(path) as z, zipfile.ZipFile(old_path) as old:
        read = lambda n: json.loads(z.read(n))
        gz = lambda n: json.loads(gzip.decompress(z.read(n)))
        load = lambda archive, n: torch.load(io.BytesIO(archive.read(n)), map_location='cpu', weights_only=True)
        cfg, summary, results, split = map(read, ['run_config.json', 'p2_summary.json', 'validation_results.json', 'split.json'])
        assert z.testzip() is None
        checks['zip_crc'] = 'pass'
        assert len(split['train_ids']) == 597 and len(split['val_ids']) == 149
        assert not set(split['train_ids']) & set(split['val_ids'])
        old_split = json.loads(old.read('split.json'))
        assert all(split[k] == old_split[k] for k in ['train_ids', 'val_ids'])
        checks['split_same_as_p0r'] = 'pass'
        names = ['p0r_summary.json', 'split.json', 'p0r_train.pt.manifest.json', 'validation_metrics.json']
        for family in ['pointwise', 'local']:
            names += [family + '_completed.json', family + '_best.pth']
            names += [family + '_snapshots/' + tag + '.pth' for tag in ['epoch_06', 'epoch_18']]
        content_hash = hashlib.sha256()
        for name in sorted(names):
            content_hash.update(name.encode())
            content_hash.update(hashlib.sha256(old.read(name)).digest())
        assert cfg['reference_sha256'] in [sha(old_path.read_bytes()), content_hash.hexdigest()]
        checks['reference_zip_or_directory_fingerprint'] = 'pass'
        for name, expected in cfg['source_hashes'].items():
            assert sha(z.read('sources/' + name)) == expected, name
            assert sha((root / name).read_bytes()) == expected, name
        checks['packaged_and_local_sources'] = len(cfg['source_hashes'])
        assert summary['status'] == 'P2_TRAINING_COMPLETED'
        assert not summary['input']['test_evaluated'] and summary['input_files_unchanged']
        assert results == summary['results']
        checks['completion_summary'] = 'pass'
        histories = {m: read(m + '/history.json') for m in ['A', 'B05', 'B10']}
        baseline = histories['A'][0]['validation']['A']['metrics']
        for history in histories.values():
            assert [r['epoch'] for r in history] == list(range(19))
            for d in history[0]['validation'].values():
                assert d['metrics'] == baseline
        assert all(not any(r['validation']['A'].get('p0r_metric_differences', {}).values()) for r in histories['A'])
        checks['A_recorded_p0r_metric_differences'] = 'all zero'
        for epoch in [0, 1, 2, 6, 12, 18]:
            a = load(z, f'A/epoch_{epoch:02d}.pth')['model']
            b = load(old, f'pointwise_snapshots/epoch_{epoch:02d}.pth')['model']
            assert {k.removeprefix('rescorer.') for k in a} == set(b)
            assert all(torch.equal(t, b[k.removeprefix('rescorer.')]) for k, t in a.items()), epoch
        checks['A_tensor_exact_p0r_epochs'] = [0, 1, 2, 6, 12, 18]
        print('Sources, split, A reproduction verified.', flush=True)
        base_trace = gz('baseline_matches.json.gz')
        raw_checks, artifact_count, modes_checked = {}, 0, 0
        for view, result in results.items():
            model = 'A' if view.startswith('A') else view
            history = histories[model]
            best = max(history, key=lambda r: r['validation'][view]['metrics']['Tiny_AP50'])
            assert best['epoch'] == result['best_epoch']
            assert sha(z.read(result['best_checkpoint'])) == result['best_checkpoint_sha256']
            best_payload = load(z, result['best_checkpoint'])
            epoch_payload = load(z, f'{model}/epoch_{best["epoch"]:02d}.pth')
            assert all(torch.equal(t, epoch_payload['model'][k]) for k, t in best_payload['model'].items())
            for tag, epoch in [('best', best['epoch']), ('last', 18)]:
                prefix = view + '_' + tag
                d = read(prefix + '.json')
                assert d['epoch'] == epoch and d['modes'] == result[tag]
                assert d['modes']['before_nms']['metrics'] == history[epoch]['validation'][view]['metrics']
                for name, expected in d['artifact_hashes'].items():
                    assert sha(z.read(name)) == expected, name
                    artifact_count += 1
                a = np.load(io.BytesIO(z.read(prefix + '_scores.npz')))
                raw, effective, bound, alpha = a['raw_delta'], a['effective_delta'], d['residual_bound'], d['alpha']
                reconstructed = raw if bound is None else bound * np.tanh(raw / bound)
                assert np.max(np.abs(reconstructed * alpha - effective)) < 1e-6
                base = torch.tensor(a['base'])
                logits = torch.logit(base.clamp(1e-6, 1-1e-6))
                expected = (base + (torch.sigmoid(logits + torch.tensor(effective)) - torch.sigmoid(logits))).clamp(0, 1).numpy()
                assert np.max(np.abs(expected - a['scores'])) < 1e-6
                near = None if bound is None else float(np.mean(np.abs(effective) >= .95 * bound))
                assert near == history[epoch]['validation'][view]['near_bound_fraction']
                raw_checks[prefix] = dict(epoch=epoch, near_bound_fraction=near,
                    max_abs_effective_delta=float(np.max(np.abs(effective))),
                    gray_mean_score=float(a['scores'][a['reason'] == 'gray_iou'].mean()))
                for mode, report in d['modes'].items():
                    trace = gz(prefix + '_' + mode + '_matches.json.gz')
                    nms = gz(prefix + '_' + mode + '_nms.json.gz')
                    tallies = {}
                    for si, iid in enumerate(a['image_ids']):
                        keep, edges = nms[iid]['keep'], nms[iid]['edges']
                        survivors = set(keep)
                        losers = [e[1] for e in edges]
                        count = int(a['offsets'][si+1] - a['offsets'][si])
                        assert len(survivors) == len(keep) and len(set(losers)) == len(losers)
                        assert not survivors & set(losers) and set(keep + losers) == set(range(count))
                        assert all(e[0] in survivors for e in edges)
                        for scope, row in trace[iid].items():
                            br = base_trace[iid].get(scope, {})
                            bh, bf, nh, nf = hits(br, 200), hits(br), hits(row, 200), hits(row)
                            assert set(row.get('indices', [])) == survivors
                            if mode == 'after_nms':
                                assert survivors == set(br.get('indices', []))
                            cur = dict(base_hit_gt=len(bh), new_hit_gt=len(nh), base_full_hit_gt=len(bf),
                                new_full_hit_gt=len(nf), recovered_gt=len(nh-bh), lost_gt=len(bh-nh),
                                base_budget_lost_gt=len(bf-bh), new_budget_lost_gt=len(nf-nh))
                            total = tallies.setdefault(scope, {k: 0 for k in cur})
                            for k, value in cur.items():
                                total[k] += value
                    for scope, total in tallies.items():
                        for k, value in total.items():
                            assert value == report['matching_and_budget'][scope][k], (prefix, mode, scope, k)
                    modes_checked += 1
            print('Verified', view, flush=True)
        checks.update(artifacts_sha256_verified=artifact_count, score_formulas_best_last='pass',
            best_checkpoint_tensor_and_selection='pass', independently_recounted_match_and_nms_runs=modes_checked)
        evidence = dict(archive_sha256=sha(path.read_bytes()), checks=checks, configuration=cfg,
            input=summary['input'], baseline=baseline, results=results, raw_score_checks=raw_checks,
            diagnostics={v: {t: histories['A' if v.startswith('A') else v][ep]['validation'][v]
                for t, ep in [('best', r['best_epoch']), ('last', 18)]} for v, r in results.items()},
            limitations=[
                'No candidate cache or full GT attached; official AP was not re-evaluated locally. Sources, checkpoint tensors, score formulas, NMS partitions and saved matching sets were verified.',
                'One seed and validation split; method and epoch selection use the same validation set. No new test evaluation.',
                'Bounded saturation is observed but is not proven to be the sole cause of performance differences.',
                'Latency_ms=0 is a placeholder; no runtime comparison supported.'],
            decision='P2 accepted as a completed controlled experiment. Use unbounded pointwise training with inference alpha=.25 before NMS as P3 control. Preserve bounds .5/1 as ablations. P3 adds reliable conflict-pair supervision; coordinates/features/graph stay fixed.')
        archive = target / 'p2_reports.zip'
        if archive.exists():
            assert sha(archive.read_bytes()) == evidence['archive_sha256']
        else:
            archive.write_bytes(path.read_bytes())
        (target / 'review_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
