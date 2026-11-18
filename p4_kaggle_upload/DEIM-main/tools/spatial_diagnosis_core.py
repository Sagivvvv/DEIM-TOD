"""Geometry, replay and official TinyPerson evaluation for spatial diagnosis.

This module deliberately has no engine imports. Analysis can run on CPU in a
fresh process with native pycocotools, without importing the training registry.
"""

import contextlib
import copy
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.special import expit


SCHEMA_VERSION = 1
PHASE_GRIDS = ("p00", "p10", "p01", "p11")


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def axis_starts(length, tile, step):
    """Two half-stride phases with equal counts and equal real-pixel totals."""
    if min(length, tile, step) <= 0 or step > tile:
        raise ValueError("Require positive dimensions and 0 < step <= tile")
    if length <= tile:
        start = -((tile - length) // 2)
        return [start], [start]
    shift = step // 2
    intervals = (length - tile + shift + step - 1) // step
    excess = intervals * step + tile - length
    start = -((excess - shift) // 2)
    return tuple(
        [start - phase * shift + index * step for index in range(intervals + 1)]
        for phase in (0, 1)
    )


def phase_corners(width, height, tile_w, tile_h, overlap, grid):
    xs = axis_starts(width, tile_w, tile_w - overlap)[int(grid[1])]
    ys = axis_starts(height, tile_h, tile_h - overlap)[int(grid[2])]
    return [(x, y, x + tile_w, y + tile_h) for y in ys for x in xs]


def valid_rectangles(corners, width, height):
    """Real image rectangles in local tile coordinates."""
    corners = np.asarray(corners, dtype=np.float32)
    sizes = corners[:, 2:] - corners[:, :2]
    return np.concatenate(
        (np.maximum(-corners[:, :2], 0),
         np.minimum(np.array([width, height]) - corners[:, :2], sizes)), axis=1
    )


def xywh_to_xyxy(boxes):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def xyxy_to_xywh(boxes):
    boxes = np.asarray(boxes).reshape(-1, 4).copy()
    boxes[:, 2:] -= boxes[:, :2]
    return boxes


def official_box_iou(annotations, boxes):
    """Use the exact bbox IoU arithmetic used by official normal-GT matching."""
    from pycocotools import mask

    if not len(annotations) or not len(boxes):
        return np.zeros((len(annotations), len(boxes)))
    gt_xywh = [a["bbox"] for a in annotations]
    return mask.iou(xyxy_to_xywh(boxes).tolist(), gt_xywh, [0] * len(gt_xywh)).T


def box_relations(left, right):
    """Pairwise IoU, intersection / left area, intersection / right area."""
    left = np.asarray(left, dtype=np.float32).reshape(-1, 4)
    right = np.asarray(right, dtype=np.float32).reshape(-1, 4)
    lo = np.maximum(left[:, None, :2], right[None, :, :2])
    hi = np.minimum(left[:, None, 2:], right[None, :, 2:])
    wh = np.maximum(hi - lo, 0)
    intersection = wh[..., 0] * wh[..., 1]
    left_area = np.prod(np.maximum(left[:, 2:] - left[:, :2], 0), axis=1)
    right_area = np.prod(np.maximum(right[:, 2:] - right[:, :2], 0), axis=1)
    union = left_area[:, None] + right_area[None, :] - intersection
    return (
        intersection / np.maximum(union, 1e-12),
        intersection / np.maximum(left_area[:, None], 1e-12),
        intersection / np.maximum(right_area[None, :], 1e-12),
    )


def maximum_matches(edges):
    """One candidate can match at most one normal GT."""
    matching = maximum_bipartite_matching(csr_matrix(edges), perm_type="column")
    return int(np.count_nonzero(matching >= 0))


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def load_candidates(cache):
    """Flatten postprocessor candidates, preserving the tile/rank identity."""
    local = cache["boxes_tile"].reshape(-1, 4)
    tile_count, top_k = cache["scores"].shape
    tile_indices = np.repeat(np.arange(tile_count), top_k)
    offsets = cache["corners"][tile_indices, :2].astype(np.float32)
    boxes = local + np.tile(offsets, (1, 2))
    centers = (local[:, :2] + local[:, 2:]) / 2
    valid = cache["valid_rects"][tile_indices]
    valid_center = np.all((centers >= valid[:, :2]) & (centers < valid[:, 2:]), axis=1)
    return {
        "boxes": boxes,
        "scores": cache["scores"].ravel(),
        "labels": cache["labels"].ravel(),
        "tile_indices": tile_indices,
        "ranks": np.tile(np.arange(top_k), tile_count),
        "query_ids": cache["query_ids"].ravel(),
        "valid_center": valid_center,
    }


def merge_candidates(candidates, grid, score_threshold, nms_threshold, max_det):
    """Replay the repository's class-agnostic CPU torchvision NMS exactly."""
    import torch
    from torchvision.ops import nms

    scores = candidates["scores"]
    valid = np.ones(len(scores), dtype=bool)
    if grid != "baseline":
        valid &= candidates["valid_center"]
    before = np.flatnonzero(valid & (scores > score_threshold))
    keep = nms(
        torch.from_numpy(candidates["boxes"][before].copy()),
        torch.from_numpy(scores[before].copy()), nms_threshold,
    ).numpy()
    after_nms = before[keep]
    after_cap = after_nms
    if max_det and 0 < max_det < len(after_nms):
        order = torch.argsort(torch.from_numpy(scores[after_nms]), descending=True).numpy()
        after_cap = after_nms[order[:max_det]]
    return before, after_nms, after_cap, valid


def suppression_trace(candidates, before, after_nms, after_cap, valid, nms_threshold):
    """Identify the first retained box that suppressed each discarded box.

    Native NMS determines retention, including score ties. Its retained order
    determines the first overlapping suppressor without imposing a new tie rule.
    """
    scores = candidates["scores"]
    status = np.full(len(scores), "score", dtype="<U16")
    status[~valid] = "padding_center"
    status[before] = "nms"
    status[after_nms] = "max_det"
    status[after_cap] = "kept"
    suppressor = np.full(len(scores), -1, dtype=np.int64)
    removed = np.flatnonzero(status == "nms")
    if len(removed):
        overlaps = box_relations(candidates["boxes"][removed], candidates["boxes"][after_nms])[0]
        allowed = (overlaps > nms_threshold) & (scores[after_nms][None, :] >= scores[removed][:, None])
        if not np.all(allowed.any(axis=1)):
            raise RuntimeError("Could not reconstruct native NMS suppression")
        suppressor[removed] = after_nms[allowed.argmax(axis=1)]
    return status, suppressor


def normal_annotations(dataset, image_id):
    return [
        annotation for annotation in dataset["annotations_by_image"].get(image_id, [])
        if not any(annotation.get(key, False) for key in ("ignore", "uncertain", "iscrowd"))
        and 1 <= annotation["area"] <= 1e10
    ]


def prepare_dataset(dataset):
    """Index GT and explicitly exclude images carrying dense annotations."""
    by_image = {}
    dense = {image["id"] for image in dataset["images"] if image.get("in_dense_image", False)}
    for annotation in dataset["annotations"]:
        if annotation["id"] <= 0:
            raise ValueError("Official matching requires positive GT annotation IDs")
        if min(annotation["bbox"][2:]) <= 0:
            raise ValueError(f"Non-positive GT box for annotation {annotation['id']}")
        by_image.setdefault(annotation["image_id"], []).append(annotation)
        if annotation.get("in_dense_image", False):
            dense.add(annotation["image_id"])
    dataset["annotations_by_image"] = by_image
    dataset["images_by_id"] = {image["id"]: image for image in dataset["images"]}
    dataset["dense_image_ids"] = sorted(dense)
    return dataset


def query_evidence(cache, category_id):
    """All decoder boxes and the requested class score before PostProcessor topK."""
    normalized = cache["pred_boxes"].reshape(-1, 4)
    tile_count, query_count = cache["pred_boxes"].shape[:2]
    tiles = np.repeat(np.arange(tile_count), query_count)
    corners = cache["corners"][tiles].astype(np.float32)
    size = corners[:, 2:] - corners[:, :2]
    center = normalized[:, :2] * size
    half_size = normalized[:, 2:] / 2
    local = np.concatenate(((normalized[:, :2] - half_size) * size,
                            (normalized[:, :2] + half_size) * size), axis=1)
    valid = cache["valid_rects"][tiles]
    selected = np.zeros((tile_count, query_count), dtype=bool)
    tile, rank = np.nonzero(cache["labels"] == category_id)
    selected[tile, cache["query_ids"][tile, rank]] = True
    return {
        "boxes": local + np.tile(corners[:, :2], (1, 2)),
        "scores": expit(cache["pred_logits"][:, :, category_id]).ravel(),
        "tile_indices": tiles,
        "query_ids": np.tile(np.arange(query_count), tile_count),
        "selected": selected.ravel(),
        "valid_center": np.all((center >= valid[:, :2]) & (center < valid[:, 2:]), axis=1),
    }


def fragment_mask(gt_box, query_boxes, corners, image_wh, seam_distance):
    """GT crosses a source tile's internal edge, prediction ends near that edge."""
    width, height = image_wh
    seam = np.zeros(len(query_boxes), dtype=bool)
    for coordinate, limit in ((0, width), (1, height)):
        for side in (coordinate, coordinate + 2):
            edge = corners[:, side]
            internal = (edge > 0) & (edge < limit)
            crosses = (gt_box[coordinate] < edge) & (gt_box[coordinate + 2] > edge)
            near = np.abs(query_boxes[:, side] - edge) <= seam_distance
            seam |= internal & crosses & near
    return seam


def gt_tile_geometry(gt_box, corners, image_wh):
    """Count intersecting tiles cut through the GT and find its best tile margin."""
    wh = np.minimum(corners[:, 2:], gt_box[2:]) - np.maximum(corners[:, :2], gt_box[:2])
    intersects = (wh > 0).all(axis=1)
    interior = np.concatenate((corners[:, :2] > 0, corners[:, 2:] < image_wh), axis=1)
    margins = np.concatenate((gt_box[:2] - corners[:, :2], corners[:, 2:] - gt_box[2:]), axis=1)
    cut_tiles = np.count_nonzero(intersects & ((margins < 0) & interior).any(axis=1))
    return int(cut_tiles), float(margins.min(axis=1).max())


def classify_evidence(gt_index, gt_box, threshold, raw_iou, query_iou, query_iog,
                      query_iop, candidates, queries, valid, query_valid, cache,
                      category_id, args, image_wh):
    """Diagnostic labels refer to an explicit score floor and IoU threshold."""
    same_class = candidates["labels"] == category_id
    full_saved = (raw_iou[gt_index] >= threshold) & same_class & valid
    full_queries = (query_iou[gt_index] >= threshold) & query_valid
    floor = args.evidence_floor
    saved_evidence = full_saved & (candidates["scores"] > floor)
    query_evidence_mask = queries["scores"] > floor
    missing_full_queries = full_queries & query_evidence_mask & ~queries["selected"]
    fragment = (
        query_valid & query_evidence_mask & (query_iou[gt_index] < threshold)
        & (query_iop[gt_index] >= args.fragment_iop)
        & (query_iog[gt_index] >= args.fragment_min_iog)
        & (query_iog[gt_index] <= args.fragment_max_iog)
    )
    possible = np.flatnonzero(fragment)
    if len(possible):
        fragment[possible] &= fragment_mask(
            gt_box, queries["boxes"][possible],
            cache["corners"][queries["tile_indices"][possible]], image_wh, args.seam_distance,
        )
    practical = full_saved & (candidates["scores"] > args.score_threshold)
    overlap = query_valid & query_evidence_mask & (query_iou[gt_index] > 0)
    if practical.any():
        label = "full_candidate"
    elif saved_evidence.any():
        label = "low_score_full"
    elif missing_full_queries.any():
        label = "pretopk_full"
    elif fragment.any():
        label = "seam_fragment"
    elif overlap.any():
        label = "other_overlap"
    else:
        label = "no_evidence_at_floor"
    query_indices = np.flatnonzero(overlap)
    best_query = int(query_indices[np.argmax(query_iou[gt_index, query_indices])]) if len(query_indices) else -1
    overlap_indices = np.flatnonzero(query_valid & (query_iou[gt_index] > 0))
    best_any = int(overlap_indices[np.argmax(query_iou[gt_index, overlap_indices])]) if len(overlap_indices) else -1
    fragment_indices = np.flatnonzero(fragment)
    best_fragment = -1
    if len(fragment_indices):
        best_fragment = int(fragment_indices[np.argmax(query_iog[gt_index, fragment_indices])])
    details = {
        "evidence_label": label,
        "saved_full_below_evidence_floor": bool((full_saved & ~saved_evidence).any()),
        "pretopk_excluded_full_at_floor": bool(missing_full_queries.any()),
        "best_query_index": best_query,
        "best_query_iou": float(query_iou[gt_index, best_query]) if best_query >= 0 else 0.0,
        "best_query_score": float(queries["scores"][best_query]) if best_query >= 0 else None,
        "best_query_any_score_index": best_any,
        "best_query_any_score_iou": float(query_iou[gt_index, best_any]) if best_any >= 0 else 0.0,
        "best_query_any_score_value": float(queries["scores"][best_any]) if best_any >= 0 else None,
        "fragment_query_index": best_fragment,
        "fragment_iop": float(query_iop[gt_index, best_fragment]) if best_fragment >= 0 else None,
        "fragment_iog": float(query_iog[gt_index, best_fragment]) if best_fragment >= 0 else None,
    }
    for cutoff in args.evidence_floors:
        suffix = str(cutoff)
        details[f"saved_full_score_gt_{suffix}"] = bool((full_saved & (candidates["scores"] > cutoff)).any())
        details[f"pretopk_full_score_gt_{suffix}"] = bool((full_queries & (queries["scores"] > cutoff)).any())
    return details


def official_evaluate(dataset, image_ids, results, thresholds, max_dets, log_path):
    """Reuse native COCOExpandEval, including ignore/uncertain/IOD and AP."""
    from pycocotools.coco import COCO
    from pycocotools import cocoeval
    from huicv.evaluation.expand_cocofmt_eval import COCOExpandEval

    if "faster_coco_eval" in str(cocoeval.__file__):
        raise RuntimeError("Run analyze in a fresh process with native pycocotools")
    selected = set(image_ids)
    with Path(log_path).open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        gt = COCO()
        gt.dataset = {
            "info": copy.deepcopy(dataset.get("info", {})),
            "images": [copy.deepcopy(dataset["images_by_id"][image_id]) for image_id in image_ids],
            "categories": copy.deepcopy(dataset["categories"]),
            "annotations": [copy.deepcopy(a) for a in dataset["annotations"] if a["image_id"] in selected],
        }
        gt.createIndex()
        if results:
            detections = gt.loadRes(results)
        else:
            detections = COCO()
            detections.dataset = {"images": gt.dataset["images"], "categories": gt.dataset["categories"], "annotations": []}
            detections.createIndex()
        evaluator = COCOExpandEval(
            gt, detections, "bbox", ignore_uncertain=True, use_ignore_attr=True,
            use_iod_for_ignore=True, iod_th_of_iou_f="lambda iou: (2*iou)/(1+iou)",
            cocofmt_param={"evaluate_standard": "tiny", "iouThrs": thresholds, "maxDets": [max_dets]},
        )
        evaluator.params.imgIds = image_ids
        evaluator.evaluate()
        evaluator.accumulate()
    matches = {}
    detection_counts = {}
    all_range = evaluator.params.areaRng[0]
    for record in evaluator.evalImgs:
        if record is None or list(record["aRng"]) != list(all_range):
            continue
        for threshold_index, threshold in enumerate(thresholds):
            for gt_index, gt_id in enumerate(record["gtIds"]):
                if record["gtIgnore"][gt_index]:
                    continue
                detection_id = int(record["gtMatches"][threshold_index, gt_index])
                candidate = int(detections.anns[detection_id]["candidate_index"]) if detection_id else -1
                matches[(record["image_id"], float(threshold), gt_id)] = candidate
            matched = record["dtMatches"][threshold_index] > 0
            ignored = record["dtIgnore"][threshold_index].astype(bool)
            detection_counts[(record["image_id"], float(threshold))] = {
                "official_tp": int(np.count_nonzero(matched & ~ignored)),
                "official_fp": int(np.count_nonzero(~matched & ~ignored)),
                "official_ignored": int(np.count_nonzero(ignored)),
            }
    precision = evaluator.eval["precision"]
    ap = {}
    for threshold_index, threshold in enumerate(thresholds):
        ap[str(threshold)] = {}
        for area_index, area in enumerate(evaluator.params.areaRngLbl):
            values = precision[threshold_index, :, :, area_index, 0]
            values = values[values >= 0]
            ap[str(threshold)][area] = float(values.mean()) if len(values) else None
    return matches, detection_counts, ap
