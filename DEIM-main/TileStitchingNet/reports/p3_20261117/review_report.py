"""Audit returned P3 sources, labels, scores, checkpoints and match records."""
from collections import Counter
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


def iou(a, b):
    overlap = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                         np.maximum(a[:, None, :2], b[None, :, :2])).prod(2)
    aa = np.maximum(0, a[:, 2:] - a[:, :2]).prod(1)
    bb = np.maximum(0, b[:, 2:] - b[:, :2]).prod(1)
    return overlap / (aa[:, None] + bb[None, :] - overlap + 1e-7)


def nms(boxes, scores):
    order = scores.argsort()[::-1]
    keeps, edges = [], []
    while len(order):
        k, rest = int(order[0]), order[1:]
        keeps.append(k)
        overlaps = iou(boxes[k:k+1], boxes[rest])[0]
        edges.extend((k, int(j)) for j in rest[overlaps > .5])
        order = rest[overlaps <= .5]
    return keeps, edges


def hits(row, n=None):
    return {g for g, ignore in zip(row.get('matches', [])[:n], row.get('ignored', [])[:n]) if g > 0 and not ignore}


def audit_pairs(geometry, saved, full_nms=None):
    exclusions, inclusions = Counter(), Counter()
    counts = dict(images=len(geometry), images_with_pairs=0, edges=0, pairs=0, gt_groups=0)
    qfl_reasons, normal_count, ignored_count = {}, 0, 0
    local_replay_missing_pairs = 0
    for iid, row in geometry.items():
        b = np.asarray(row['boxes'], np.float32).reshape(-1, 4)
        g = np.asarray(row['normal_boxes'], np.float32).reshape(-1, 4)
        ig = np.asarray(row['ignored_boxes'], np.float32).reshape(-1, 4)
        scores = np.asarray(row['scores'], np.float32)
        normal_count += len(g)
        ignored_count += len(ig)
        ov = iou(b, g)
        q = ov.max(1) if len(g) else np.zeros(len(b), np.float32)
        second = np.partition(ov, -2, axis=1)[:, -2] if len(g) > 1 else np.zeros(len(b), np.float32)
        gid = ov.argmax(1) if len(g) else np.full(len(b), -1)
        iod = np.zeros(len(b), np.float32)
        if len(ig):
            inter = np.maximum(0, np.minimum(b[:, None, 2:], ig[None, :, 2:]) - np.maximum(b[:, None, :2], ig[None, :, :2])).prod(2)
            iod = (inter / np.maximum((b[:, 2:] - b[:, :2]).prod(1)[:, None], 1e-7)).max(1)
        reasons = np.where(q >= .5, 'positive', 'negative').astype('<U24')
        reasons[(q >= .3) & (q < .5)] = 'gray_iou'
        reasons[(q < .5) & (iod >= .5)] = 'ignore_region'
        reasons[(second >= .5) & (q - second < .05)] = 'ambiguous_gt'
        qfl_reasons[iid] = reasons
        actual = {(p['winner'], p['suppressed']): p for p in saved['pairs'][iid]}
        assert len(actual) == len(saved['pairs'][iid])
        if full_nms is None:
            # Training export contains accepted edges, not the full tie order.
            # Audit every returned label; do not fabricate the missing ordering.
            edges = list(actual)
            _, locally_replayed = nms(b, scores)
            local_replay_missing_pairs += len(set(actual) - set(locally_replayed))
        else:
            edges = [tuple(p) for p in full_nms[iid]['edges']]
            remaining = set(range(len(b)))
            for winner in full_nms[iid]['keep']:
                assert winner in remaining and scores[winner] == max(scores[list(remaining)])
                rest = np.array(sorted(remaining - {winner}), dtype=np.int64)
                suppressed = set(rest[iou(b[winner:winner+1], b[rest])[0] > .5])
                assert suppressed == {j for w, j in edges if w == winner}
                remaining -= suppressed | {winner}
            assert not remaining
        retained = set()
        gt_groups = set()
        for winner, loser in edges:
            assert scores[winner] >= scores[loser]
            assert iou(b[winner:winner+1], b[loser:loser+1])[0, 0] > .5
            idx = [winner, loser]
            if np.any(iod[idx] >= .5):
                reason = 'ignore_overlap'
            elif np.any((second[idx] >= .3) & (q[idx] - second[idx] < .05)):
                reason = 'ambiguous_assignment'
            elif max(q[idx]) < .5:
                reason = 'neither_qualified'
            elif min(q[idx]) >= .5 and gid[winner] != gid[loser]:
                reason = 'different_gt_positives'
            else:
                good, bad = (winner, loser) if q[winner] > q[loser] else (loser, winner)
                if q[bad] >= .3 and gid[good] != gid[bad]:
                    reason = 'different_gt_gray'
                elif float(q[good]) - float(q[bad]) + 1e-7 < .1:
                    reason = 'small_quality_gap'
                else:
                    reason = None
            if reason:
                exclusions[reason] += 1
                assert (winner, loser) not in actual
                continue
            if (winner, loser) not in actual:
                raise AssertionError((iid, winner, loser, float(scores[winner]), float(scores[loser]),
                    [(w, l, float(scores[w])) for w, l in actual if l == loser or w == winner],
                    float(q[winner]), float(q[loser])))
            pair = actual[(winner, loser)]
            retained.add((winner, loser))
            assert (pair['good'], pair['bad']) == (good, bad)
            assert pair['gt_id'] == row['normal_ids'][gid[good]]
            assert pair['quality_good'] == float(q[good]) and pair['quality_bad'] == float(q[bad])
            crossing = q[bad] < .5
            expected_weight = .25 + .75 * np.clip(min(float(q[good])-.5, .5-float(q[bad]))/.05, 0, 1) if crossing else 1.
            assert abs(pair['weight'] - expected_weight) < 1e-12
            assert pair['kind'] == ('threshold_crossing' if crossing else 'same_gt_quality')
            for side, index in [('good', good), ('bad', bad)]:
                assert pair[side+'_reason'] == reasons[index]
                assert pair[side+'_qfl_valid'] == (reasons[index] in ['positive', 'negative'])
            margin = scores[good] - scores[bad]
            assert pair['base_order'] == ('correct' if margin > 0 else 'wrong' if margin < 0 else 'tie')
            assert pair['cross_tile'] == (row['patch_ids'][good] != row['patch_ids'][bad])
            inclusions.update([pair['kind'], pair['base_order'], 'bad_'+pair['bad_reason'], 'cross_tile' if pair['cross_tile'] else 'same_tile'])
            gt_groups.add(pair['gt_id'])
        assert retained == set(actual)
        counts['edges'] += len(edges)
        counts['pairs'] += len(actual)
        counts['gt_groups'] += len(gt_groups)
        counts['images_with_pairs'] += bool(actual)
    if full_nms is None:
        # Counts/exclusions agree with the per-image audit, but only validation
        # exclusions can be independently reclassified from a full NMS trace.
        counts['edges'] = sum(r['edges'] for r in saved['per_image'])
        for r in saved['per_image']:
            assert r['pairs'] == len(saved['pairs'][r['image_id']])
            assert r['pairs'] + sum(r['excluded'].values()) == r['edges']
            exclusions.update(r['excluded'])
    counts.update(included=dict(inclusions), excluded=dict(exclusions))
    assert counts == saved['summary']
    signature = sha(json.dumps(dict(config=saved['config'], pairs=saved['pairs']), sort_keys=True,
                               separators=(',', ':'), allow_nan=False).encode())
    assert signature == saved['signature']
    return dict(summary=counts, normal_gt=normal_count, ignored_gt=ignored_count,
                full_nms_replayed=full_nms is not None,
                local_default_sort_missing_pairs=local_replay_missing_pairs), qfl_reasons


def main(path):
    target = Path(__file__).resolve().parent
    root = target.parents[2]
    checks, analyses = {}, {}
    with zipfile.ZipFile(path) as z, zipfile.ZipFile(target.parent/'p2_20261117/p2_reports.zip') as p2:
        read = lambda n: json.loads(z.read(n))
        gz = lambda n: json.loads(gzip.decompress(z.read(n)))
        load = lambda archive, n: torch.load(io.BytesIO(archive.read(n)), map_location='cpu', weights_only=True)
        cfg, summary, results, split = map(read, ['run_config.json', 'p3_summary.json', 'validation_results.json', 'split.json'])
        assert z.testzip() is None
        assert summary['status'] == 'P3_TRAINING_COMPLETED' and summary['input_files_unchanged'] and not summary['input']['test_evaluated']
        assert results == summary['results']
        assert split == json.loads(p2.read('split.json'))
        assert len(split['train_ids']) == 597 and len(split['val_ids']) == 149
        assert not set(split['train_ids']) & set(split['val_ids'])
        for name, expected in cfg['source_hashes'].items():
            assert sha(z.read('sources/'+name)) == expected
            assert sha((root/name).read_bytes()) == expected
        checks.update(zip_crc='pass', split='same as P2, disjoint', source_files=len(cfg['source_hashes']))
        h2 = json.loads(p2.read('A/history.json'))
        h3 = read('A025/history.json')
        for r in h3:
            e = r['epoch']
            assert r['validation']['A025']['metrics'] == h2[e]['validation']['A_scaled025']['metrics']
            assert r['validation']['A025']['p2_max_parameter_abs_diff'] == 0
            a = load(z, f'A025/epoch_{e:02d}.pth')['model']
            b = load(p2, f'A/epoch_{e:02d}.pth')['model']
            assert set(a) == set(b) and all(torch.equal(v, b[k]) for k, v in a.items())
        checks['A025_exact_P2_parameter_epochs'] = len(h3)
        pair_datasets, geometries = {}, {}
        for family, id_key in [('train', 'train_ids'), ('validation', 'val_ids')]:
            geometry, pair_data = gz(family+'_pair_geometry.json.gz'), gz(family+'_pairs.json.gz')
            assert set(geometry) == set(pair_data['pairs']) == set(split[id_key])
            analysis, reasons = audit_pairs(geometry, pair_data,
                gz('A025_best_after_nms_nms.json.gz') if family == 'validation' else None)
            analyses[family+'_labels'] = analysis
            pair_datasets[family], geometries[family] = pair_data, geometry
            print('Independent pair geometry/NMS audit:', family, analysis['summary']['pairs'], flush=True)
        checks['reconstructed_pair_labels'] = 'all returned train/validation labels valid; full validation NMS/exclusions replayed'
        base_trace = gz('baseline_matches.json.gz')
        A_best_trace = gz('A025_best_before_nms_matches.json.gz')
        val_geometry, val_pairs = geometries['validation'], pair_datasets['validation']['pairs']
        artifact_count = modes_checked = 0
        detailed, trajectories = {}, {}
        for name, result in results.items():
            history = read(name+'/history.json')
            assert [r['epoch'] for r in history] == list(range(19))
            best = max(history, key=lambda r: r['validation'][name]['metrics']['Tiny_AP50'])
            assert best['epoch'] == result['best_epoch']
            assert sha(z.read(result['best_checkpoint'])) == result['best_checkpoint_sha256']
            bp = load(z, result['best_checkpoint'])
            ep = load(z, f'{name}/epoch_{best["epoch"]:02d}.pth')
            assert all(torch.equal(v, ep['model'][k]) for k, v in bp['model'].items())
            for r in history[1:]:
                assert abs(r['weighted_pair'] - cfg['methods'][name]*r['pair']) < 1e-7
                assert abs(r['loss'] - r['qfl'] - r['weighted_pair']) < 1e-7
            trajectories[name] = history
            for tag, epoch in [('best', best['epoch']), ('last', 18)]:
                prefix = name+'_'+tag
                d = read(prefix+'.json')
                assert d['epoch'] == epoch and d['modes'] == result[tag]
                assert d['modes']['before_nms']['metrics'] == history[epoch]['validation'][name]['metrics']
                for filename, expected in d['artifact_hashes'].items():
                    assert sha(z.read(filename)) == expected
                    artifact_count += 1
                a = np.load(io.BytesIO(z.read(prefix+'_scores.npz')))
                assert np.array_equal(a['effective_delta'], .25*a['raw_delta'])
                base = torch.tensor(a['base']); delta = torch.tensor(a['effective_delta'])
                logits = torch.logit(base.clamp(1e-6, 1-1e-6))
                expected = (base + ((logits+delta).sigmoid()-logits.sigmoid())).clamp(0, 1).numpy()
                assert np.max(np.abs(expected-a['scores'])) < 1e-6
                pair_counts, image_losses = Counter(), []
                for si, iid in enumerate(a['image_ids']):
                    start, end = map(int, a['offsets'][si:si+2])
                    g = val_geometry[iid]
                    np.testing.assert_array_equal(a['base'][start:end], np.asarray(g['scores'], np.float32))
                    scores = a['scores'][start:end]
                    group_losses = {}
                    for p in val_pairs[iid]:
                        bmargin = a['base'][start+p['good']] - a['base'][start+p['bad']]
                        nmargin = scores[p['good']] - scores[p['bad']]
                        pair_counts.update(pairs=1, base_correct=int(bmargin > 0), new_correct=int(nmargin > 0),
                            broken_correct=int(bmargin > 0 and nmargin <= 0), recovered_wrong=int(bmargin < 0 and nmargin > 0))
                        effective_logits = logits + delta
                        gap = float(effective_logits[start+p['bad']] - effective_logits[start+p['good']])
                        group_losses.setdefault(p['gt_id'], []).append(p['weight']*float(np.logaddexp(0., gap)))
                    image_losses.append(float(np.mean([np.mean(v) for v in group_losses.values()])) if group_losses else 0.)
                reported = history[epoch]['validation'][name]['training_rule_pairs']['all']
                assert all(v == reported[k] for k, v in pair_counts.items())
                detailed[prefix] = dict(epoch=epoch, validation_pair_loss=float(np.mean(image_losses)),
                                       validation_pair_counts=dict(pair_counts), modes=d['modes'])
                for mode, mr in d['modes'].items():
                    traces, nms_records = gz(prefix+'_'+mode+'_matches.json.gz'), gz(prefix+'_'+mode+'_nms.json.gz')
                    totals, vs_A = {}, {}
                    for iid, scopes in traces.items():
                        nr, geo = nms_records[iid], val_geometry[iid]
                        keep, losers = set(nr['keep']), [e[1] for e in nr['edges']]
                        assert len(keep) == len(nr['keep']) and len(losers) == len(set(losers))
                        assert not keep & set(losers) and keep | set(losers) == set(range(len(geo['scores'])))
                        assert all(e[0] in keep for e in nr['edges'])
                        for scope, row in scopes.items():
                            br = base_trace[iid].get(scope, {})
                            bh, bf, nh, nf = hits(br, 200), hits(br), hits(row, 200), hits(row)
                            assert set(row['indices']) == keep
                            if mode == 'after_nms': assert set(br['indices']) == keep
                            cur = dict(base_hit_gt=len(bh), new_hit_gt=len(nh), base_full_hit_gt=len(bf), new_full_hit_gt=len(nf),
                                recovered_gt=len(nh-bh), lost_gt=len(bh-nh), base_budget_lost_gt=len(bf-bh), new_budget_lost_gt=len(nf-nh))
                            totals.setdefault(scope, Counter()).update(cur)
                            ah = hits(A_best_trace[iid].get(scope, {}), 200)
                            vs_A.setdefault(scope, Counter()).update(recovered_gt=len(nh-ah), lost_gt=len(ah-nh))
                    for scope, counts in totals.items():
                        assert all(mr['matching_and_budget'][scope][k] == v for k, v in counts.items())
                    detailed[prefix].setdefault('vs_A025_best', {})[mode] = vs_A
                    modes_checked += 1
            print('Scores, pairs, selected weights and match records:', name, flush=True)
        checks.update(artifact_hashes=artifact_count, independently_recounted_match_runs=modes_checked,
                      score_formulas='pass', checkpoint_selection='pass', loss_components='pass')
        evidence = dict(archive_sha256=sha(path.read_bytes()), checks=checks, configuration=cfg,
            pair_audit=analyses, results=results, detailed=detailed, histories=trajectories,
            limitations=['Single seed, same validation split used for model/epoch selection; no new test evaluation.',
                'Official AP values not recomputed locally; sources, scores, label geometry and saved official match sets audited.',
                'Returned train/validation label geometry contains zero ignored GT; cannot validate original annotation completeness from these derived files.',
                'Training full NMS trace was not exported; accepted-edge geometry/labels verified, but training exclusion counts only reconciled with per-image audits. Local NumPy tie order differs from Kaggle.',
                'Loss magnitudes are not parameter-gradient magnitudes; no claim that pair weight is the sole bottleneck.'],
            decision='P3 execution and label audit accepted; C020 is a provisional candidate with only +0.08026 TinyAP50 over A025, not established stable improvement. Keep A025 control; next P4 should start with low-cost ROI ablation, delaying graph growth and fusion.')
        destination = target/'p3_reports.zip'
        if destination.exists(): assert sha(destination.read_bytes()) == evidence['archive_sha256']
        else: destination.write_bytes(path.read_bytes())
        (target/'review_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps(checks, indent=2))
        print('Validation pair losses:', {k: v['validation_pair_loss'] for k, v in detailed.items()})


if __name__ == '__main__':
    main(Path(sys.argv[1]))
