"""Evaluate one whole-image post-processor on the official validation candidates."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torchvision.ops import nms

from official_tile_candidates import CACHE, GT, OUTPUT, ROOT, SPLIT, load_candidates, run_dir

import sys
sys.path.insert(0, str(ROOT))

from TileStitchingNet import stage1
from TileStitchingNet.p1 import OfficialMatchTrace, hits
from TileStitchingNet.p5 import ap_input, sampled_ap, sampling_group
from TileStitchingNet.stage1 import OfficialTinyEvaluator
from tile_postprocess_adapters import load_method


def dump(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(tmp, 'wt', encoding='utf8') as f:
        json.dump(obj, f, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, path)


def input_fingerprints(samples):
    fingerprints = {}
    for s in samples:
        digest = hashlib.sha256()
        for v in (s.img_size, s.boxes_global, s.scores, s.class_ids, s.patch_ids, s.tile_rects_all):
            digest.update(np.ascontiguousarray(v).tobytes())
        fingerprints[s.image_id] = digest.hexdigest()
    return fingerprints


def whole_image_nms(sample):
    boxes, scores = sample.boxes_global, sample.scores
    keep = nms(torch.from_numpy(boxes), torch.from_numpy(scores), .5).numpy()
    return boxes[keep], scores[keep]


def boundary_counts(samples):
    tile = image = 0
    for sample in samples:
        boxes, rects = sample.boxes_global, sample.patch_rects
        tile += int(((boxes[:, :2] < rects[:, :2]) | (boxes[:, 2:] > rects[:, 2:])).any(axis=1).sum())
        image += int(((boxes[:, :2] < 0) | (boxes[:, 2:] > sample.img_size)).any(axis=1).sum())
    return dict(outside_tile=tile, outside_image=image)


def bootstrap(base, current, ids, groups):
    by_group = {}
    for i, iid in enumerate(ids):
        by_group.setdefault(groups[iid], []).append(i)
    members = list(by_group.values())
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(1000):
        ix = sorted(i for g in rng.integers(len(members), size=len(members)) for i in members[g])
        diff = sampled_ap(current, ix) - sampled_ap(base, ix)
        if np.isfinite(diff):
            deltas.append(diff)
    return np.percentile(deltas, [2.5, 97.5]).tolist()


def requested_metrics(evaluator, samples, boxes, scores):
    # Area bins are evaluated independently. Keep the two bins actually read by
    # OfficialTinyEvaluator. The upstream IoU, matching and accumulate code is unchanged.
    coco, original = stage1.official_classes()
    class ReportedAreas(original):
        def evaluate(self):
            p = self.params
            indices = [p.areaRngLbl.index(k) for k in ('all','tiny')]
            p.areaRng = [p.areaRng[i] for i in indices]
            p.areaRngLbl = ['all','tiny']
            return super().evaluate()
    with patch.object(stage1,'official_classes',return_value=(coco,ReportedAreas)):
        return evaluator(samples,boxes,scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True)
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--gt', type=Path, default=GT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--sources', type=Path, default=ROOT/'output/postprocess_sources')
    args = parser.parse_args()
    torch.set_num_threads(4)
    samples, anchor = load_candidates(args.cache, args.gt)
    out = run_dir(args.output, anchor['cache_sha256'])
    out.mkdir(parents=True, exist_ok=True)
    ids = [s.image_id for s in samples]
    if args.method == 'nms':
        predict = whole_image_nms
        config = dict(source='torchvision.ops.nms', iou=.5, scope='whole image',
                      score_thresh=anchor['score_threshold'])
    else:
        base_file = out/'nms.json'
        if not base_file.is_file():
            raise ValueError('Run the whole-image NMS baseline for this candidate cache first')
        base_info = json.loads(base_file.read_text(encoding='utf8'))
        if base_info['inputs'] != anchor or base_info['split'] != SPLIT:
            raise ValueError('Baseline input mismatch')
        predict, config = load_method(args.method, args.sources)
    setup_start = perf_counter()
    if hasattr(predict, 'prepare'):
        predict.prepare(samples)
    setup_ms = (perf_counter()-setup_start)*1000
    predictions, timings = [], []
    for s in samples[:5]:
        predict(s)
    for rep in range(3):
        elapsed = []
        for s in samples:
            start = perf_counter()
            boxes, scores = predict(s)
            elapsed.append(1000 * (perf_counter() - start))
            if len(boxes) != len(scores) or boxes.shape != (len(scores), 4):
                raise ValueError(f'Invalid output shape for {s.image_id}')
            if not np.isfinite(boxes).all() or not np.isfinite(scores).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
                raise ValueError(f'Invalid output for {s.image_id}')
            if rep == 0:
                predictions.append((boxes, scores))
        timings.append(elapsed)
    grouped_ms = None
    if hasattr(predict,'predict_split'):
        start = perf_counter()
        predictions = predict.predict_split(samples)
        grouped_ms = (perf_counter()-start)*1000
    boxes, scores = zip(*predictions)
    evaluator = OfficialTinyEvaluator(args.gt)
    metrics = requested_metrics(evaluator, samples, boxes, scores)
    metrics.pop('Latency_ms')
    virtual = [SimpleNamespace(image_id=i, boxes_global=b) for i, b in zip(ids, boxes)]
    trace = OfficialMatchTrace(evaluator.data)(virtual, [np.arange(len(sc)) for sc in scores], scores)
    ap_rows = ap_input(trace, dict(zip(ids, scores)), ids)
    if abs(sampled_ap(ap_rows, range(len(ids))) - metrics['Tiny_AP50']) > 1e-5:
        raise ValueError('Bootstrap inputs do not reproduce official AP')
    result = dict(method=args.method, split=SPLIT, images=len(ids), candidates=sum(len(s.scores) for s in samples),
                  output_boxes=sum(map(len, scores)), config=config, inputs=anchor, metrics=metrics,
                  environment=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                                   threads=torch.get_num_threads(), platform=platform.platform()),
                  latency_ms=dict(median=float(np.median(timings)), p95=float(np.percentile(timings, 95)),
                                  passes=len(timings), warmup_images=min(5,len(ids)), split_prepare_ms=setup_ms,
                                  original_split_run_ms=grouped_ms),
                  tiny50_hits={str(k):sum(len(hits(t.get('tiny@0.5', {}), k)) for t in trace.values()) for k in (200, None)})
    if args.method == 'nms':
        result['boundary_counts'] = boundary_counts(samples)
        result['delta_nms'] = 0.
    else:
        with gzip.open(out/'nms.ap.json.gz', 'rt', encoding='utf8') as f:
            saved = json.load(f)
        if saved['ids'] != ids:
            raise ValueError('Baseline input mismatch')
        base = [(np.asarray(a), np.asarray(b, dtype=bool), c) for a,b,c in saved['rows']]
        groups = {i:sampling_group(evaluator.images[int(i)]) for i in ids}
        result['delta_nms'] = metrics['Tiny_AP50'] - base_info['metrics']['Tiny_AP50']
        result['ci95'] = bootstrap(base, ap_rows, ids, groups)
    prefix = args.method
    dump(out/(prefix+'.predictions.json.gz'), {i:dict(boxes_xyxy=b.tolist(), scores=sc.tolist())
                                               for i,(b,sc) in zip(ids,predictions)})
    dump(out/(prefix+'.timing.json.gz'), dict(ids=ids, milliseconds=timings))
    dump(out/(prefix+'.ap.json.gz'), dict(ids=ids, rows=[(a.tolist(),b.tolist(),c) for a,b,c in ap_rows]))
    if args.method == 'nms':
        dump(out/'input_fingerprints.json', input_fingerprints(samples))
    dump(out/(prefix+'.json'), result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
