"""Validation-only diagnostics; candidate indices always refer to the frozen cache."""
import contextlib
import copy
import io
from collections import Counter

import numpy as np
import torch

from .datasetBuilder import box_area_np, pairwise_iou_np
from .stage1 import official_classes
from .stage2 import candidate_graph


def nms_trace(boxes, scores, threshold=.5):
    """Exactly the existing nms_np ordering, including its tie convention."""
    order = scores.argsort()[::-1]
    keep, edges = [], []
    while len(order):
        winner = int(order[0]); keep.append(winner)
        rest = order[1:]
        if not len(rest): break
        ov = pairwise_iou_np(boxes[winner:winner+1], boxes[rest])[0]
        edges.extend((winner, int(j)) for j in rest[ov > threshold])
        order = rest[ov <= threshold]
    return np.asarray(keep, np.int64), np.asarray(edges, np.int64).reshape(-1, 2)


def scaled_scores(base, delta, alpha):
    # Same float32 correction as QualityRescorer; alpha=0 is bitwise identity.
    if alpha == 0: return base.copy()
    b = torch.from_numpy(base.astype(np.float32)).clamp(1e-6, 1-1e-6)
    z = torch.logit(b)
    return (b + ((z + float(alpha)*torch.from_numpy(delta)).sigmoid()-z.sigmoid())).clamp(0, 1).numpy()


def describe(values):
    a = np.asarray(values)
    return dict(n=int(a.size), mean=float(a.mean()),
                quantiles=np.quantile(a, [0, .01, .1, .5, .9, .99, 1]).tolist()) if a.size else dict(n=0)


class CandidateLabels:
    """Explain the existing training mask, without treating all ignored rows as background."""
    def __init__(self, data, targets):
        self.targets = targets
        self.normal_ids = {iid: [] for iid in targets.by_image}
        for a in data['annotations']:
            _, _, w, h = a['bbox']
            if a['category_id'] == 1 and not (w <= 0 or h <= 0) and not any(a.get(k, 0) for k in ('ignore', 'uncertain', 'iscrowd')):
                self.normal_ids[str(a['image_id'])].append(a['id'])

    def __call__(self, sample):
        cfg = self.targets.config
        quality, valid, training_gid = self.targets(sample)
        # Reuse the training corners: rounding xywh before adding xy+wh can
        # move an otherwise identical box across an IoU or IOD threshold.
        g, ig = [np.asarray(x, np.float32).reshape(-1, 4) for x in self.targets.by_image[sample.image_id]]
        normal_ids = self.normal_ids[sample.image_id]
        if len(normal_ids) != len(g): raise ValueError('Annotation IDs differ from training geometry: '+sample.image_id)
        b = sample.boxes_global; n = len(b)
        ov = pairwise_iou_np(b, g)
        best = ov.max(1) if len(g) else np.zeros(n, np.float32)
        second = np.partition(ov, -2, axis=1)[:, -2] if len(g) > 1 else np.zeros(n, np.float32)
        gid = ov.argmax(1) if len(g) else np.full(n, -1, np.int64)
        iod = np.zeros(n, np.float32)
        if len(ig):
            inter = np.prod(np.maximum(0, np.minimum(b[:, None, 2:], ig[None, :, 2:])-
                                         np.maximum(b[:, None, :2], ig[None, :, :2])), axis=-1)
            iod = (inter/np.maximum(box_area_np(b)[:, None], 1e-7)).max(1)
        fg = best >= cfg.positive_iou
        ambiguous = (second >= cfg.positive_iou) & (best-second < cfg.ambiguity_gap)
        ignore_region = (~fg) & (iod >= cfg.ignore_iod)
        gray = (best >= cfg.background_iou) & ~fg
        reason = np.where(fg, 'positive', 'negative').astype('<U24')
        reason[gray] = 'gray_iou'
        reason[ignore_region] = 'ignore_region'
        reason[ambiguous] = 'ambiguous_gt'
        if not np.array_equal(valid, ~(gray | ignore_region | ambiguous)):
            raise AssertionError('Diagnostic reason masks differ from QFL: '+sample.image_id)
        return dict(best=best, second=second, gid=gid, training_gid=training_gid,
                    iod=iod, reason=reason, valid=valid, quality=quality,
                    normal_ids=np.asarray(normal_ids), normal_boxes=g,
                    flags=dict(gray_iou=gray, ignore_region=ignore_region, ambiguous_gt=ambiguous))


def reliable_pairs(sample, labels, edges, threshold):
    """Conservative diagnostics on real baseline suppression edges, not new training labels.

    Include same-GT quality gaps and qualified/unqualified competition. Exclude
    different-instance positives, ambiguous membership and uncertain/ignore overlaps.
    """
    pairs, excluded = [], Counter()
    q, gid = labels['best'], labels['gid']
    for winner, loser in edges:
        i, j = int(winner), int(loser)
        if ((labels['second'][[i, j]] >= threshold) &
                (q[[i, j]]-labels['second'][[i, j]] < .05)).any():
            excluded['ambiguous_gt'] += 1; continue
        # Conservative for diagnostic ranking pairs, independent of the QFL mask.
        if np.any(labels['iod'][[i, j]] >= .5):
            excluded['ignore_overlap'] += 1; continue
        fi, fj = q[i] >= threshold, q[j] >= threshold
        if not (fi or fj):
            excluded['neither_qualified'] += 1; continue
        if fi and fj and gid[i] != gid[j]:
            excluded['different_gt_positives'] += 1; continue
        if abs(float(q[i]-q[j])) < .1:
            excluded['small_quality_gap'] += 1; continue
        good, bad = (i, j) if q[i] > q[j] else (j, i)
        gi = int(gid[good]); gt = labels['normal_boxes'][gi]
        size = float(np.sqrt(np.prod(gt[2:]-gt[:2])))
        pairs.append(dict(winner=i, suppressed=j, good=good, bad=bad,
                          gt_id=int(labels['normal_ids'][gi]), quality_good=float(q[good]), quality_bad=float(q[bad]),
                          threshold=threshold, kind='same_gt_quality' if fi and fj else 'threshold_crossing',
                          cross_tile=bool(sample.patch_ids[i] != sample.patch_ids[j]),
                          gt_size='tiny' if size <= 20 else 'non_tiny'))
    return pairs, dict(excluded)


def pair_results(samples, pairs_by_image, new_scores):
    groups = {}
    for s, pairs, scores in zip(samples, pairs_by_image, new_scores):
        for p in pairs:
            b = float(s.scores[p['good']]-s.scores[p['bad']])
            a = float(scores[p['good']]-scores[p['bad']])
            for group in ['all', 'cross_tile' if p['cross_tile'] else 'same_tile', p['kind'], p['gt_size']]:
                c = groups.setdefault(group, Counter())
                c['pairs'] += 1; c['base_correct'] += b > 0; c['new_correct'] += a > 0
                c['base_ties'] += b == 0; c['new_ties'] += a == 0
                c['broken_correct'] += b > 0 and a <= 0
                c['recovered_wrong'] += b < 0 and a > 0
    for c in groups.values():
        c['new_win_rate'] = c['new_correct']/c['pairs']
        c['broken_correct_rate'] = c['broken_correct']/c['base_correct'] if c['base_correct'] else None
    return groups


def graph_coverage(sample, edges, cfg):
    idx, _, mask = candidate_graph(sample, cfg)
    adj = [set(row[m].tolist()) for row, m in zip(idx, mask)]
    rows = []
    for winner, loser in edges:
        w, l = int(winner), int(loser)
        size = float(np.sqrt(max(0, np.prod(sample.boxes_global[l, 2:]-sample.boxes_global[l, :2]))))
        rows.append(dict(winner=w, suppressed=l, cross_tile=bool(sample.patch_ids[w] != sample.patch_ids[l]),
                         suppressed_box_size='tiny' if size <= 20 else 'non_tiny',
                         suppressed_reads_winner=w in adj[l], winner_reads_suppressed=l in adj[w]))
    return rows


def score_diagnostics(samples, labels, deltas, scores, base_edges):
    groups = {}
    for reason in ['all', 'positive', 'negative', 'gray_iou', 'ignore_region', 'ambiguous_gt']:
        ds, old, new = [], [], []
        for s, lab, delta, sc in zip(samples, labels, deltas, scores):
            mask = np.ones(len(sc), bool) if reason == 'all' else lab['reason'] == reason
            ds.extend(delta[mask]); old.extend(s.scores[mask]); new.extend(sc[mask])
        old, new = np.asarray(old), np.asarray(new)
        groups[reason] = dict(delta=describe(ds), base_score=describe(old), new_score=describe(new),
                              score_change=describe(new-old), saturation_count=int(((new <= 0)|(new >= 1)).sum()))
    reverse = Counter(); mae = []; base_q = []; new_q = []; oracle = []
    for s, lab, sc, edges in zip(samples, labels, scores, base_edges):
        mask = lab['valid'] & (lab['quality'] > 0)
        mae.extend(np.abs(sc[mask]-lab['quality'][mask]))
        for g in np.unique(lab['training_gid'][mask]):
            ix = np.flatnonzero(mask & (lab['training_gid'] == g))
            base_q.append(lab['quality'][ix[np.argmax(s.scores[ix])]])
            new_q.append(lab['quality'][ix[np.argmax(sc[ix])]])
            oracle.append(lab['quality'][ix].max())
        for w, l in edges:
            old, new = float(s.scores[w]-s.scores[l]), float(sc[w]-sc[l])
            reverse['edges'] += 1; reverse['strict_reversals'] += old*new < 0
            reverse['base_ties'] += old == 0; reverse['new_ties'] += new == 0
    mean = lambda x: float(np.mean(x)) if len(x) else None
    return dict(groups=groups, baseline_suppression_edge_order=reverse,
                quality=dict(positive_mae=mean(mae), gt_best_by_base=mean(base_q), gt_best_by_new=mean(new_q),
                             gt_oracle_quality=mean(oracle), reliable_gt_groups=len(oracle)))


class OfficialMatchTrace:
    """Official greedy matches at IoU .50/.75, including uncertain/ignore/IOD.

    Compute all survivors for budget diagnostics. Top-200 uses the same score
    prefix; later detections cannot change an earlier official greedy match.
    Main AP is evaluated separately at maxDets=200 by OfficialTinyEvaluator.
    """
    def __init__(self, annotation_data):
        self.data = annotation_data

    def __call__(self, samples, kept_indices, scores):
        COCO, Eval = official_classes()
        predictions, mapping = [], {}
        for s, keep, sc in zip(samples, kept_indices, scores):
            for k in keep:
                x, y, x2, y2 = map(float, s.boxes_global[k])
                if x2 <= x or y2 <= y: raise ValueError('Nonpositive candidate box in trace')
                predictions.append(dict(image_id=int(s.image_id), category_id=1,
                                        bbox=[x, y, x2-x, y2-y], score=float(sc[k])))
                mapping[len(predictions)] = int(k)
        limit = max(200, max((len(k) for k in kept_indices), default=0))
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(); gt.dataset = copy.deepcopy(self.data); gt.createIndex()
            if predictions:
                dt = gt.loadRes(predictions)
            else:
                dt = COCO(); dt.dataset = dict(images=copy.deepcopy(self.data['images']),
                    categories=copy.deepcopy(self.data['categories']), annotations=[]); dt.createIndex()
            ev = Eval(gt, dt, 'bbox', ignore_uncertain=True, use_ignore_attr=True, use_iod_for_ignore=True,
                      iod_th_of_iou_f='lambda iou: (2*iou)/(1+iou)',
                      cocofmt_param=dict(evaluate_standard='tiny', iouThrs=[.5, .75], maxDets=[limit]))
            ev.params.imgIds = [int(s.image_id) for s in samples]; ev.params.catIds = [1]
            ev.params.areaRng = ev.params.areaRng[:2]; ev.params.areaRngLbl = ['all', 'tiny']
            ev.evaluate()
        by_id = {s.image_id: s for s in samples}
        traces = {s.image_id: {} for s in samples}
        for e in ev.evalImgs:
            if e is None: continue
            iid = str(e['image_id']); sample = by_id[iid]
            area = ev.params.areaRngLbl[ev.params.areaRng.index(e['aRng'])]
            ix = np.asarray([mapping[k] for k in e['dtIds']], np.int64)
            # Confirm prefix assumption against the evaluator's actual capped call.
            capped = ev.evaluateImg(int(iid), 1, e['aRng'], 200)
            if not np.array_equal(e['dtMatches'][:, :200], capped['dtMatches']) or not np.array_equal(e['dtIgnore'][:, :200], capped['dtIgnore']):
                raise AssertionError('Official budget-prefix matching changed')
            eligible = [g for g, ignore in zip(e['gtIds'], e['gtIgnore']) if not ignore]
            gb = np.asarray([gt.anns[g]['bbox'] for g in eligible], np.float32).reshape(-1, 4)
            gb[:, 2:] += gb[:, :2]
            ov = pairwise_iou_np(sample.boxes_global[ix], gb)
            best = ov.max(1) if len(gb) else np.zeros(len(ix))
            for ti, threshold in enumerate([.5, .75]):
                matches = e['dtMatches'][ti].astype(np.int64)
                ignore = e['dtIgnore'][ti].astype(bool)
                kinds = np.where(best > 0, 'localization_fp', 'background_fp').astype('<U24')
                kinds[best >= threshold] = 'duplicate_fp'
                kinds[matches > 0] = 'tp'; kinds[ignore] = 'ignored'
                traces[iid][f'{area}@{threshold}'] = dict(indices=ix.tolist(), matches=matches.tolist(),
                    ignored=ignore.tolist(), kinds=kinds.tolist(), eligible_gt_ids=list(map(int, eligible)))
        return traces


def hits(row, budget=None):
    return {g for g, ignored in zip(row.get('matches', [])[:budget], row.get('ignored', [])[:budget]) if g > 0 and not ignored}


def compare_traces(samples, baseline, changed, base_edges=None, changed_edges=None):
    """Use distinct annotation-ID sets per image; categories are not summed as causal gains."""
    totals, records = {}, []
    for si, s in enumerate(samples):
        iid = s.image_id
        for area in ['all', 'tiny']:
            for threshold in [.5, .75]:
                key = f'{area}@{threshold}'
                b = baseline[iid].get(key, {}); a = changed[iid].get(key, {})
                bh, ah, bf, af = hits(b, 200), hits(a, 200), hits(b), hits(a)
                bi, ai = b.get('indices', []), a.get('indices', [])
                btop, atop = set(bi[:200]), set(ai[:200]); bs, aset = set(bi), set(ai)
                row = dict(image_id=iid, scope=key, recovered_gt=sorted(ah-bh), lost_gt=sorted(bh-ah),
                    before_budget_recovered_gt=sorted(af-bf), before_budget_lost_gt=sorted(bf-af),
                    base_budget_lost_gt=sorted(bf-bh), new_budget_lost_gt=sorted(af-ah),
                    entered_top200=sorted(atop-btop), left_top200=sorted(btop-atop),
                    entered_from_base_below200=sorted((atop-btop) & bs),
                    entered_from_base_suppressed=sorted((atop-btop)-bs),
                    new_survivors=sorted(aset-bs), removed_survivors=sorted(bs-aset))
                contexts = []
                for event, gt_set, matched_row, other_row, edge_list in [
                    ('recovered', ah-bh, a, b, base_edges),
                    ('lost', bh-ah, b, a, changed_edges)]:
                    parents = {} if edge_list is None else {int(l): int(w) for w, l in edge_list[si]}
                    for g in sorted(gt_set):
                        k = next(k for k, m, ig in zip(matched_row['indices'][:200], matched_row['matches'][:200], matched_row['ignored'][:200]) if m == g and not ig)
                        parent = parents.get(k)
                        if parent is not None:
                            cause = 'cross_tile_suppression' if s.patch_ids[k] != s.patch_ids[parent] else 'same_tile_suppression'
                        elif k in other_row.get('indices', [])[200:]: cause = 'below200'
                        else: cause = 'matching_or_survivor_change'
                        contexts.append(dict(event=event, gt_id=g, matched_candidate=k,
                                             other_run_suppressor=parent, association=cause))
                row['gt_event_context'] = contexts
                c = totals.setdefault(key, Counter())
                for field, value in row.items():
                    if isinstance(value, list) and field != 'gt_event_context': c[field] += len(value)
                for event in contexts: c[event['event']+'_'+event['association']] += 1
                c['base_hit_gt'] += len(bh); c['new_hit_gt'] += len(ah)
                c['base_full_hit_gt'] += len(bf); c['new_full_hit_gt'] += len(af)
                c['eligible_gt'] += len(b.get('eligible_gt_ids', []))
                for kind, count in Counter(a.get('kinds', [])[:200]).items(): c['new_'+kind] += count
                for kind, count in Counter(b.get('kinds', [])[:200]).items(): c['base_'+kind] += count
                for side, rr in [('base', b), ('new', a)]:
                    # Cross/same evidence is on suppression edges, not guessed GT provenance.
                    c[side+'_detections_top200'] += min(200, len(rr.get('indices', [])))
                records.append(row)
    return dict(summary=totals, per_image=records)
