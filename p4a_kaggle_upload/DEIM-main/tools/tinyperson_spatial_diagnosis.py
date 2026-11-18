#!/usr/bin/env python
"""Fixed-detector candidate export and offline TinyPerson spatial diagnosis.

Run with ``python -X utf8 tools/tinyperson_spatial_diagnosis.py --help``.
See SPATIAL_DIAGNOSIS.md for full commands, definitions and output schemas.
"""

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
from collections import Counter
from pathlib import Path
import sys
import time

import numpy as np

from spatial_diagnosis_core import (
    PHASE_GRIDS, SCHEMA_VERSION, atomic_json, box_relations, classify_evidence,
    gt_tile_geometry, load_candidates, maximum_matches, merge_candidates, normal_annotations,
    official_box_iou, official_evaluate, phase_corners, prepare_dataset, query_evidence, read_json,
    safe_ratio, suppression_trace, valid_rectangles, xywh_to_xyxy, xyxy_to_xywh,
)


REPO = Path(__file__).resolve().parents[1]
DATA = REPO.parent / "datasets" / "Tinyperson"
DEFAULT_CACHE = REPO / "output" / "spatial_diagnosis"
KAGGLE_INPUT = Path("/kaggle/input")


def resolve_checkpoint(checkpoint):
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint.resolve()

    relative_path = "model_pth/best_stg1.pth"
    mounted = sorted({
        path.resolve()
        for depth in range(5)
        for path in KAGGLE_INPUT.glob("*/" * depth + relative_path)
        if path.is_file()
    })
    if len(mounted) > 1:
        paths = "\n".join(str(path) for path in mounted)
        raise ValueError(f"Multiple Kaggle checkpoints found. Select one with --checkpoint:\n{paths}")
    if mounted:
        return mounted[0]

    for path in (REPO.parent / relative_path, REPO / "output/baseline/best_stg1.pt"):
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"Checkpoint not found. Attach a Kaggle input containing {relative_path} "
        "or set --checkpoint to the full path of best_stg1.pt or best_stg1.pth."
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_cache(path, signature):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(arrays.pop("metadata")))
    if metadata["signature"] != signature or metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Cache signature/schema mismatch at {path}")
    tiles, top_k = arrays["scores"].shape
    if arrays["boxes_tile"].shape != (tiles, top_k, 4):
        raise ValueError(f"Invalid cached candidate shape at {path}")
    if arrays["corners"].shape != (tiles, 4) or len(metadata["tile_ids"]) != tiles:
        raise ValueError(f"Invalid cached tile metadata at {path}")
    return arrays, metadata


class CsvSink:
    def __init__(self, path, empty_fields):
        self.stream = Path(path).open("w", newline="", encoding="utf-8-sig")
        self.writer = None
        self.empty_fields = empty_fields

    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)

    def close(self):
        if self.writer is None:
            csv.writer(self.stream).writerow(self.empty_fields)
        self.stream.close()


def get_tiling_module():
    path = REPO / "engine/data/dataset/tinyperson_tiling.py"
    spec = importlib.util.spec_from_file_location("_diagnosis_tiling", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_images(dataset, args):
    available = sorted(set(dataset["images_by_id"]) - set(dataset["dense_image_ids"]))
    if args.image_ids:
        missing = set(args.image_ids) - set(available)
        if missing:
            raise ValueError(f"Unavailable or dense image IDs: {sorted(missing)}")
        available = sorted(set(args.image_ids))
    if args.limit is not None:
        available = available[:args.limit]
    if not available:
        raise ValueError("No images selected")
    return available


def baseline_index(dataset, path, evaluator):
    tiles = read_json(path)["images"]
    by_image = {}
    if len({tile["id"] for tile in tiles}) != len(tiles):
        raise ValueError("Duplicate baseline tile IDs")
    for tile in tiles:
        by_image.setdefault(int(tile["ori_image_id"]), []).append(tile)
    tiling = get_tiling_module()
    for image_id, image in dataset["images_by_id"].items():
        expected = tiling.generate_tile_corners(
            image["width"], image["height"], evaluator["tile_w"],
            evaluator["tile_h"], evaluator["overlap"],
        )
        group = sorted(by_image.get(image_id, []), key=lambda item: item["id"])
        if [tuple(tile["corner"]) for tile in group] != expected:
            raise ValueError(f"Baseline tile corners disagree with configuration for image {image_id}")
        by_image[image_id] = group
    return by_image


def build_detector(args):
    import torch
    import torchvision

    sys.path.insert(0, str(REPO))
    from engine.core import YAMLConfig

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = YAMLConfig(str(args.config))
    config.yaml_cfg["HGNetv2"]["pretrained"] = False
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(checkpoint.get("ema"), dict) and "module" in checkpoint["ema"]:
        state, weight_source = checkpoint["ema"]["module"], "ema.module"
    else:
        state, weight_source = checkpoint["model"], "model"
    model = config.model
    model.load_state_dict(state, strict=True)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    postprocessor = config.postprocessor.eval().to(device)
    if not postprocessor.use_focal_loss or postprocessor.remap_mscoco_category:
        raise ValueError("This TinyPerson tool requires focal scores and direct category IDs")
    return config, model.eval().to(device), postprocessor, device, {
        "weight_source": weight_source, "checkpoint_epoch": checkpoint.get("last_epoch"),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
    }


def infer_group(model, postprocessor, device, preprocess, crops, tile_count, batch_size):
    import torch

    output_arrays = {key: [] for key in ("boxes_tile", "scores", "labels", "query_ids", "pred_boxes", "pred_logits")}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, tile_count, batch_size):
            stop = min(start + batch_size, tile_count)
            images = crops(start, stop)
            batch = torch.stack([preprocess(image) for image in images]).to(device)
            crop_wh = torch.tensor([image.size for image in images], device=device, dtype=torch.float32)
            outputs = model(batch)
            predictions = postprocessor(outputs, crop_wh)
            flat_scores = outputs["pred_logits"].sigmoid().flatten(1)
            selected_scores, selected_indices = torch.topk(flat_scores, postprocessor.num_top_queries, dim=-1)
            query_ids = selected_indices // postprocessor.num_classes
            for index, prediction in enumerate(predictions):
                # Assert provenance follows the actual PostProcessor topK operation.
                torch.testing.assert_close(prediction["scores"], selected_scores[index], rtol=0, atol=0)
                torch.testing.assert_close(prediction["labels"], selected_indices[index] % postprocessor.num_classes)
                for key, tensor in (
                    ("boxes_tile", prediction["boxes"]), ("scores", prediction["scores"]),
                    ("labels", prediction["labels"]), ("query_ids", query_ids[index]),
                    ("pred_boxes", outputs["pred_boxes"][index]),
                    ("pred_logits", outputs["pred_logits"][index]),
                ):
                    output_arrays[key].append(tensor.detach().cpu().numpy())
            for image in images:
                image.close()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    arrays = {key: np.stack(value) for key, value in output_arrays.items()}
    for key in ("boxes_tile", "scores", "pred_boxes", "pred_logits"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"Non-finite model output in {key}")
    return arrays, time.perf_counter() - started


def dump(args):
    from PIL import Image
    from torchvision.transforms.v2 import Compose, Resize

    args.checkpoint = resolve_checkpoint(args.checkpoint)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    dataset = prepare_dataset(read_json(args.gt))
    if len(dataset["categories"]) != 1:
        raise ValueError("This diagnostic runner expects the single-class TinyPerson GT")
    image_ids = select_images(dataset, args)
    config, model, postprocessor, device, runtime = build_detector(args)
    from engine.data.transforms._transforms import ConvertPILImage

    evaluator = config.yaml_cfg["evaluator"]
    tile_w, tile_h, overlap = [int(evaluator[key]) for key in ("tile_w", "tile_h", "overlap")]
    if not 0 <= overlap < min(tile_w, tile_h):
        raise ValueError("Require 0 <= overlap < tile dimensions")
    eval_hw = list(config.yaml_cfg["eval_spatial_size"])
    ops = config.yaml_cfg["val_dataloader"]["dataset"]["transforms"]["ops"]
    if ([op["type"] for op in ops] != ["Resize", "ConvertPILImage"]
            or list(ops[0]["size"]) != eval_hw
            or ops[1].get("dtype", "float32") != "float32" or not ops[1].get("scale", True)):
        raise ValueError("Expected validation Resize + float32 /255 conversion at eval_spatial_size")
    resize_kwargs, conversion_kwargs = ({key: value for key, value in op.items() if key != "type"} for op in ops)
    preprocess = Compose([Resize(**resize_kwargs), ConvertPILImage(**conversion_kwargs)])
    groups = baseline_index(dataset, args.tile_json, evaluator) if "baseline" in args.grids else {}
    category_id = int(dataset["categories"][0]["id"])
    if not 0 <= category_id < postprocessor.num_classes:
        raise ValueError("GT category ID is outside model class logits")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256_file(args.checkpoint),
        "config": str(args.config.resolve()), "resolved_config": config.yaml_cfg,
        "gt": str(args.gt.resolve()), "gt_sha256": sha256_file(args.gt),
        "image_root": str(args.image_root.resolve()), "tile_root": str(args.tile_root.resolve()),
        "tile_json": str(args.tile_json.resolve()),
        "tile_json_sha256": sha256_file(args.tile_json) if groups else None,
        "image_ids": image_ids, "dense_excluded": dataset["dense_image_ids"],
        "grids": args.grids, "category_id": category_id,
        "tile_wh": [tile_w, tile_h], "overlap": overlap, "eval_hw": eval_hw,
        "phase_step_xy": [tile_w - overlap, tile_h - overlap],
        "phase_shift_xy": [(tile_w - overlap) // 2, (tile_h - overlap) // 2],
        "phase_fill_rgb": [0, 0, 0], "phase_validity": "center inside real pixels",
        "seed": args.seed, "batch_size": args.batch_size, "runtime": runtime,
        "baseline_evaluator": evaluator,
        "tool_sha256": {name: sha256_file(REPO / "tools" / name) for name in
                         ("tinyperson_spatial_diagnosis.py", "spatial_diagnosis_core.py")},
    }
    manifest["signature"] = canonical_hash(manifest)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        if read_json(manifest_path)["signature"] != manifest["signature"]:
            raise ValueError("Output contains a different run. Choose a new --out directory")
    else:
        atomic_json(manifest_path, manifest)
    completed = 0
    for number, image_id in enumerate(image_ids, 1):
        started = time.perf_counter()
        info = dataset["images_by_id"][image_id]
        source = None
        tile_count = 0
        resumed_grids = 0
        try:
            for grid in args.grids:
                destination = args.out / "candidates" / grid / f"{image_id}.npz"
                if destination.exists():
                    _, meta = load_cache(destination, manifest["signature"])
                    if meta["image_id"] != image_id or meta["grid"] != grid:
                        raise ValueError(f"Cache identity mismatch at {destination}")
                    tile_count += len(meta["tile_ids"])
                    resumed_grids += 1
                    completed += 1
                    continue
                if grid == "baseline":
                    group = groups[image_id]
                    corners = [tile["corner"] for tile in group]
                    tile_ids = [tile["id"] for tile in group]
                    tile_names = [tile["file_name"] for tile in group]

                    def crops(start, stop):
                        images = []
                        for tile in group[start:stop]:
                            with Image.open(args.tile_root / tile["file_name"]) as image:
                                converted = image.convert("RGB")
                            if converted.size != (tile["width"], tile["height"]):
                                raise ValueError(f"Baseline tile size mismatch: {tile['file_name']}")
                            if converted.size != tuple(np.subtract(tile["corner"][2:], tile["corner"][:2])):
                                raise ValueError(f"Baseline corner size mismatch: {tile['id']}")
                            images.append(converted)
                        return images
                else:
                    if source is None:
                        with Image.open(args.image_root / info["file_name"]) as image:
                            source = image.convert("RGB")
                        if source.size != (info["width"], info["height"]):
                            raise ValueError(f"Original image dimensions disagree with GT: {image_id}")
                    corners = phase_corners(info["width"], info["height"], tile_w, tile_h, overlap, grid)
                    tile_ids = list(range(len(corners)))
                    tile_names = []

                    def crops(start, stop):
                        return [source.crop(tuple(corner)) for corner in corners[start:stop]]
                arrays, elapsed = infer_group(model, postprocessor, device, preprocess, crops, len(corners), args.batch_size)
                corners = np.asarray(corners, dtype=np.int32)
                valid = valid_rectangles(corners, info["width"], info["height"])
                sizes = corners[:, 2:] - corners[:, :2]
                tile_to_image = np.repeat(np.eye(3, dtype=np.float32)[None], len(corners), axis=0)
                tile_to_image[:, :2, 2] = corners[:, :2]
                network_to_tile = np.repeat(np.eye(3, dtype=np.float32)[None], len(corners), axis=0)
                network_to_tile[:, 0, 0] = sizes[:, 0] / eval_hw[1]
                network_to_tile[:, 1, 1] = sizes[:, 1] / eval_hw[0]
                metadata = {
                    "schema_version": SCHEMA_VERSION, "signature": manifest["signature"],
                    "image_id": image_id, "grid": grid, "file_name": info["file_name"],
                    "image_wh": [info["width"], info["height"]], "tile_ids": tile_ids,
                    "tile_file_names": tile_names, "inference_seconds": elapsed,
                    "real_pixel_total": int(np.prod(valid[:, 2:] - valid[:, :2], axis=1).sum()),
                    "input_pixel_total": int(len(corners) * eval_hw[0] * eval_hw[1]),
                }
                save_npz(destination, metadata=np.array(json.dumps(metadata)), corners=corners,
                         valid_rects=valid, tile_to_image=tile_to_image,
                         network_to_tile=network_to_tile, **arrays)
                tile_count += len(corners)
                completed += 1
        finally:
            if source is not None:
                source.close()
        elapsed = time.perf_counter() - started
        resumed = f", {resumed_grids} cached grids" if resumed_grids else ""
        print(f"[{number}/{len(image_ids)}] {image_id}: {len(args.grids)} grids, "
              f"{tile_count} tiles, {elapsed:.2f}s{resumed}", flush=True)
    atomic_json(args.out / "dump_complete.json", {"signature": manifest["signature"], "groups": completed})
    print(f"Candidate export complete: {args.out}", flush=True)


def coco_results(image_id, candidates, indices, category_id):
    indices = indices[candidates["labels"][indices] == category_id]
    boxes = xyxy_to_xywh(candidates["boxes"][indices])
    results = []
    for index, box in zip(indices, boxes):
        results.append({"image_id": image_id, "category_id": category_id,
                        "bbox": box.tolist(), "score": float(candidates["scores"][index]),
                        "candidate_index": int(index)})
    return results


def person_eval_indices(candidates, indices, category_id, max_dets):
    indices = indices[candidates["labels"][indices] == category_id]
    order = np.argsort(-candidates["scores"][indices], kind="stable")
    return indices[order[:max_dets]]


def best_index(ious, indices):
    return int(indices[np.argmax(ious[indices])]) if len(indices) else -1


def analyze_image(args, dataset, cache, metadata, matches, official_counts, gt_sink, paired):
    image_id, grid = metadata["image_id"], metadata["grid"]
    category_id = args.category_id
    candidates = load_candidates(cache)
    before, after_nms, after_cap, valid = merge_candidates(
        candidates, grid, args.score_threshold, args.nms_threshold, args.max_det,
    )
    final = person_eval_indices(candidates, after_cap, category_id, args.eval_max_dets)
    status, suppressor = suppression_trace(
        candidates, before, after_nms, after_cap, valid, args.nms_threshold,
    )
    codes = {"score": 0, "padding_center": 1, "nms": 2, "max_det": 3, "kept": 4}
    save_npz(args.report_dir / "replay" / grid / f"{image_id}.npz",
             status=np.array([codes[value] for value in status], dtype=np.uint8),
             suppressor_index=suppressor, before_nms=before, after_nms=after_nms,
             after_max_det=after_cap, official_person_indices=final,
             metadata=np.array(json.dumps({"image_id": image_id, "grid": grid,
                                          "analysis_signature": args.analysis_signature, "status_codes": codes})))
    normal = normal_annotations(dataset, image_id)
    gt_boxes = xywh_to_xyxy([a["bbox"] for a in normal])
    gt_labels = np.array([a["category_id"] for a in normal], dtype=np.int64)
    raw_iou = official_box_iou(normal, candidates["boxes"])
    same_class = gt_labels[:, None] == candidates["labels"][None, :]
    queries = query_evidence(cache, category_id)
    _, query_iog, query_iop = box_relations(gt_boxes, queries["boxes"])
    query_iou = official_box_iou(normal, queries["boxes"])
    query_valid = queries["valid_center"] if grid != "baseline" else np.ones(len(queries["scores"]), dtype=bool)
    person_before = before[candidates["labels"][before] == category_id]
    geometry = [gt_tile_geometry(box, cache["corners"], metadata["image_wh"]) for box in gt_boxes]
    width, height = metadata["image_wh"]
    stats = []
    for threshold in args.ious:
        cutoff = min(threshold, 1 - 1e-10)  # COCOExpandEval's inclusive matching threshold.
        edges = (raw_iou >= cutoff) & same_class
        pre_has = edges[:, before].any(axis=1)
        nms_has = edges[:, after_nms].any(axis=1)
        cap_has = edges[:, after_cap].any(axis=1)
        final_has = edges[:, final].any(axis=1)
        oracle_pre = maximum_matches(edges[:, before])
        oracle_nms = maximum_matches(edges[:, after_nms])
        oracle_cap = maximum_matches(edges[:, after_cap])
        oracle_final = maximum_matches(edges[:, final])
        actual = official_counts.get((image_id, threshold), {"official_tp": 0, "official_fp": 0, "official_ignored": 0})
        if oracle_final < actual["official_tp"]:
            raise RuntimeError(f"Official TP exceeds one-to-one matching for {image_id}/{grid} at IoU {threshold}")
        selection_budget = min(args.eval_max_dets, args.max_det) if args.max_det and args.max_det > 0 else args.eval_max_dets
        row = {
            "image_id": image_id, "grid": grid, "iou": threshold, "normal_gt": len(normal),
            "tiles": len(cache["corners"]), "input_pixel_total": metadata["input_pixel_total"],
            "real_pixel_total": metadata["real_pixel_total"], "inference_seconds": metadata["inference_seconds"],
            "pre_candidate_gt": int(pre_has.sum()), "nms_candidate_gt": int(nms_has.sum()),
            "final_candidate_gt": int(final_has.sum()),
            "direct_nms_lost_gt": int(np.count_nonzero(pre_has & ~nms_has)),
            "direct_final_lost_gt": int(np.count_nonzero(pre_has & ~final_has)),
            "oracle_pre_tp": oracle_pre, "oracle_nms_tp": oracle_nms,
            "oracle_max_det_tp": oracle_cap, "oracle_final_tp": oracle_final,
            "nms_assignment_loss": oracle_pre - oracle_nms,
            "max_det_assignment_loss": oracle_nms - oracle_cap,
            "eval_cap_assignment_loss": oracle_cap - oracle_final,
            "greedy_assignment_gap": oracle_final - actual["official_tp"],
            "budgeted_selection_gap": min(oracle_pre, selection_budget) - actual["official_tp"],
            "person_count_after_merge": int(np.count_nonzero(candidates["labels"][after_cap] == category_id)),
            "person_count_in_eval": len(final), **actual,
        }
        evidence_counts = Counter()
        final_indices = set(final.tolist())
        assigned_gt = {matches.get((image_id, threshold, a["id"]), -1): a["id"] for a in normal}
        assigned_gt.pop(-1, None)
        for gt_index, annotation in enumerate(normal):
            gt_id = annotation["id"]
            matched = matches.get((image_id, threshold, gt_id), -1)
            if matched >= 0 and matched not in final_indices:
                raise RuntimeError(f"Official match missing from replay for {image_id}/{grid}/{gt_id}")
            best_pre = best_index(raw_iou[gt_index], person_before)
            qualified_final = final[edges[gt_index, final]]
            best_final = best_index(raw_iou[gt_index], qualified_final)
            if matched >= 0:
                outcome = "matched"
            elif not pre_has[gt_index]:
                outcome = "no_qualified_before_nms"
            elif not nms_has[gt_index]:
                outcome = "nms_removed_all_qualified"
            elif not cap_has[gt_index]:
                outcome = "merge_cap_removed_all_qualified"
            elif not final_has[gt_index]:
                outcome = "eval_cap_removed_all_qualified"
            else:
                outcome = "unmatched_with_qualified_final"
            details = classify_evidence(
                gt_index, gt_boxes[gt_index], cutoff, raw_iou, query_iou, query_iog,
                query_iop, candidates, queries, valid, query_valid, cache,
                category_id, args, metadata["image_wh"],
            )
            if (details["evidence_label"] == "full_candidate") != bool(pre_has[gt_index]):
                raise RuntimeError(f"Inconsistent pre-merge match for {image_id}/{grid}/{gt_id}")
            evidence_counts[details["evidence_label"]] += 1
            gt_box = gt_boxes[gt_index]
            border_distance = float(min(gt_box[0], gt_box[1], width - gt_box[2], height - gt_box[3]))
            cut_tiles, best_margin = geometry[gt_index]
            gt_row = {
                "image_id": image_id, "grid": grid, "gt_id": gt_id, "iou": threshold,
                "category_id": annotation["category_id"], "gt_bbox_xywh": json.dumps(annotation["bbox"]),
                "size_sqrt_area": float(np.sqrt(annotation["area"])), "image_border_distance": border_distance,
                "gt_cut_tile_count": cut_tiles, "gt_best_tile_margin_px": best_margin,
                "has_pre_candidate": bool(pre_has[gt_index]), "has_nms_candidate": bool(nms_has[gt_index]),
                "has_after_merge_cap_candidate": bool(cap_has[gt_index]),
                "has_final_candidate": bool(final_has[gt_index]), "official_tp": matched >= 0, "outcome": outcome,
                "matched_candidate_index": matched, "best_pre_candidate_index": best_pre,
                "best_pre_iou": float(raw_iou[gt_index, best_pre]) if best_pre >= 0 else 0.0,
                "best_pre_status": str(status[best_pre]) if best_pre >= 0 else "",
                "best_pre_suppressor_index": int(suppressor[best_pre]) if best_pre >= 0 else -1,
                "qualified_final_count": len(qualified_final), "best_final_candidate_index": best_final,
                "best_final_candidate_assigned_gt": assigned_gt.get(best_final),
                **details,
            }
            gt_sink.write(gt_row)
            if grid in PHASE_GRIDS:
                paired.setdefault((image_id, gt_id, threshold), {})[grid] = {
                    "candidate": bool(pre_has[gt_index]), "tp": matched >= 0,
                    "box": candidates["boxes"][matched].tolist() if matched >= 0 else None,
                    "score": float(candidates["scores"][matched]) if matched >= 0 else None,
                    "sqrt_area": float(np.sqrt(annotation["area"])), "border_distance": border_distance,
                    "cut_tiles": cut_tiles, "best_margin": best_margin,
                }
        for label in ("full_candidate", "low_score_full", "pretopk_full", "seam_fragment", "other_overlap", "no_evidence_at_floor"):
            row[f"evidence_{label}"] = evidence_counts[label]
        stats.append(row)
    return stats


def paired_reports(args, dataset, paired, image_stats):
    gt_sink = CsvSink(args.report_dir / "grid_gt_stability.csv", ["image_id", "gt_id", "iou"])
    image_sink = CsvSink(args.report_dir / "grid_image_stability.csv", ["image_id", "iou"])
    aggregate, count_ranges = {}, {}
    for (image_id, gt_id, threshold), groups in sorted(paired.items()):
        if not all(grid in groups for grid in PHASE_GRIDS):
            continue
        candidate_pattern = "".join(str(int(groups[grid]["candidate"])) for grid in PHASE_GRIDS)
        tp_pattern = "".join(str(int(groups[grid]["tp"])) for grid in PHASE_GRIDS)
        cut_pattern = "".join(str(int(groups[grid]["cut_tiles"] > 0)) for grid in PHASE_GRIDS)
        pair_ious, pair_distances = [], []
        pair_details = {}
        for first, second in itertools.combinations(PHASE_GRIDS, 2):
            a, b = groups[first], groups[second]
            pair_iou, pair_distance = None, None
            if a["tp"] and b["tp"]:
                pair_iou = float(box_relations([a["box"]], [b["box"]])[0][0, 0])
                center_a = (np.array(a["box"][:2]) + a["box"][2:]) / 2
                center_b = (np.array(b["box"][:2]) + b["box"][2:]) / 2
                pair_distance = float(np.linalg.norm(center_a - center_b))
                pair_ious.append(pair_iou)
                pair_distances.append(pair_distance)
            pair_details[f"box_iou_{first}_{second}"] = pair_iou
            pair_details[f"center_distance_px_{first}_{second}"] = pair_distance
        info = dataset["images_by_id"][image_id]
        distinct = info["width"] > args.tile_wh[0] and info["height"] > args.tile_wh[1]
        first = groups[PHASE_GRIDS[0]]
        matched_scores = [groups[grid]["score"] for grid in PHASE_GRIDS if groups[grid]["tp"]]
        row = {
            "image_id": image_id, "gt_id": gt_id, "iou": threshold,
            "four_distinct_grids": distinct, "image_border_distance": first["border_distance"],
            "candidate_pattern_p00_p10_p01_p11": candidate_pattern,
            "tp_pattern_p00_p10_p01_p11": tp_pattern,
            "cut_pattern_p00_p10_p01_p11": cut_pattern,
            "candidate_hit_count": candidate_pattern.count("1"), "tp_hit_count": tp_pattern.count("1"),
            "matched_phase_pairs": len(pair_ious),
            "pair_mean_box_iou": float(np.mean(pair_ious)) if pair_ious else None,
            "pair_min_box_iou": min(pair_ious) if pair_ious else None,
            "pair_max_center_distance_px": max(pair_distances) if pair_distances else None,
            "pair_max_center_distance_normalized": max(pair_distances) / first["sqrt_area"] if pair_distances else None,
            "matched_score_range": max(matched_scores) - min(matched_scores) if len(matched_scores) >= 2 else None,
            **{f"best_tile_margin_px_{grid}": groups[grid]["best_margin"] for grid in PHASE_GRIDS},
            **pair_details,
        }
        gt_sink.write(row)
        key = str(threshold)
        tally = aggregate.setdefault(key, {"all_gt": 0, "four_distinct_gt": 0,
                                          "candidate_hits_0_to_4": [0] * 5, "tp_hits_0_to_4": [0] * 5,
                                          "distinct_candidate_hits_0_to_4": [0] * 5,
                                          "distinct_tp_hits_0_to_4": [0] * 5,
                                          "matched_phase_pairs": 0, "pair_iou_sum": 0.0,
                                          "pair_center_distance_sum": 0.0, "distinct_strata": {}})
        tally["all_gt"] += 1
        tally["candidate_hits_0_to_4"][candidate_pattern.count("1")] += 1
        tally["tp_hits_0_to_4"][tp_pattern.count("1")] += 1
        if distinct:
            tally["four_distinct_gt"] += 1
            tally["distinct_candidate_hits_0_to_4"][candidate_pattern.count("1")] += 1
            tally["distinct_tp_hits_0_to_4"][tp_pattern.count("1")] += 1
            tally["matched_phase_pairs"] += len(pair_ious)
            tally["pair_iou_sum"] += sum(pair_ious)
            tally["pair_center_distance_sum"] += sum(pair_distances)
            border = "near_image_border" if first["border_distance"] <= args.border_distance else "image_interior"
            seam = "crosses_internal_edge" if "1" in cut_pattern else "no_crossed_edge"
            for group in (border, seam):
                stratum = tally["distinct_strata"].setdefault(group, {"gt": 0, "candidate_flips": 0, "tp_flips": 0})
                stratum["gt"] += 1
                stratum["candidate_flips"] += int(0 < candidate_pattern.count("1") < 4)
                stratum["tp_flips"] += int(0 < tp_pattern.count("1") < 4)
    for (image_id, threshold), groups in sorted(image_stats.items()):
        if not all(grid in groups for grid in PHASE_GRIDS):
            continue
        rows = [groups[grid] for grid in PHASE_GRIDS]
        if (len({row["tiles"] for row in rows}) != 1
                or len({row["real_pixel_total"] for row in rows}) != 1
                or len({row["input_pixel_total"] for row in rows}) != 1):
            raise RuntimeError(f"Four-phase compute/coverage budget mismatch for image {image_id}")
        counts = [row["person_count_after_merge"] for row in rows]
        tps = [row["official_tp"] for row in rows]
        info = dataset["images_by_id"][image_id]
        distinct = info["width"] > args.tile_wh[0] and info["height"] > args.tile_wh[1]
        row = {"image_id": image_id, "iou": threshold, "four_distinct_grids": distinct,
               "tiles_each_phase": rows[0]["tiles"],
               "real_pixel_total_each_phase": rows[0]["real_pixel_total"],
               "count_range": max(counts) - min(counts), "count_std": float(np.std(counts)),
               "tp_range": max(tps) - min(tps), "tp_std": float(np.std(tps))}
        row.update({f"count_{grid}": count for grid, count in zip(PHASE_GRIDS, counts)})
        row.update({f"tp_{grid}": count for grid, count in zip(PHASE_GRIDS, tps)})
        image_sink.write(row)
        if distinct:
            count_ranges.setdefault(str(threshold), []).append((row["count_range"], row["tp_range"]))
    gt_sink.close()
    image_sink.close()
    for threshold, tally in aggregate.items():
        tally["distinct_mean_pair_box_iou"] = safe_ratio(tally.pop("pair_iou_sum"), tally["matched_phase_pairs"])
        tally["distinct_mean_pair_center_distance_px"] = safe_ratio(
            tally.pop("pair_center_distance_sum"), tally["matched_phase_pairs"],
        )
        for stratum in tally["distinct_strata"].values():
            stratum["candidate_flip_rate"] = safe_ratio(stratum["candidate_flips"], stratum["gt"])
            stratum["tp_flip_rate"] = safe_ratio(stratum["tp_flips"], stratum["gt"])
        tally["distinct_candidate_flip_rate"] = safe_ratio(sum(tally["distinct_candidate_hits_0_to_4"][1:4]), tally["four_distinct_gt"])
        tally["distinct_tp_flip_rate"] = safe_ratio(sum(tally["distinct_tp_hits_0_to_4"][1:4]), tally["four_distinct_gt"])
    image_summary = {}
    for threshold, values in count_ranges.items():
        ranges = np.asarray(values)
        image_summary[threshold] = {
            "four_distinct_images": len(values),
            "mean_count_range": float(ranges[:, 0].mean()), "p95_count_range": float(np.quantile(ranges[:, 0], 0.95)),
            "mean_tp_range": float(ranges[:, 1].mean()), "p95_tp_range": float(np.quantile(ranges[:, 1], 0.95)),
        }
    return {"thresholds": aggregate, "image_counts": image_summary, "primary_iou": args.primary_iou}


def analyze(args):
    args.cache = args.cache.resolve()
    manifest = read_json(args.cache / "manifest.json")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported cache schema")
    if canonical_hash({key: value for key, value in manifest.items() if key != "signature"}) != manifest["signature"]:
        raise ValueError("Candidate manifest signature does not match its contents")
    args.gt = (args.gt or Path(manifest["gt"])).resolve()
    if sha256_file(args.gt) != manifest["gt_sha256"]:
        raise ValueError("GT file changed since candidate export")
    dataset = prepare_dataset(read_json(args.gt))
    evaluator = manifest["baseline_evaluator"]
    args.category_id = manifest["category_id"]
    args.tile_wh = manifest["tile_wh"]
    args.grids = args.grids or manifest["grids"]
    if not set(args.grids).issubset(manifest["grids"]):
        raise ValueError("Requested grids are absent from manifest")
    args.score_threshold = evaluator["score_th"] if args.score_threshold is None else args.score_threshold
    args.nms_threshold = evaluator["merge_nms_th"] if args.nms_threshold is None else args.nms_threshold
    args.max_det = evaluator.get("max_det") if args.max_det is None else args.max_det
    if args.eval_max_dets is None:
        configured = evaluator.get("eval_max_dets", 200)
        if configured is None:
            configured = 200
        if isinstance(configured, list):
            if len(configured) != 1:
                raise ValueError("Choose one --eval-max-dets for diagnosis")
            configured = configured[0]
        args.eval_max_dets = configured
    if args.eval_max_dets <= 0 or not 0 <= args.nms_threshold <= 1 or not 0 <= args.score_threshold <= 1:
        raise ValueError("Invalid detection budget or score/NMS thresholds")
    if args.evidence_floor > args.score_threshold:
        raise ValueError("--evidence-floor must be <= the operating score threshold")
    args.report_dir = (args.report_dir or args.cache / "reports").resolve()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    parameters = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    parameters["cache_signature"] = manifest["signature"]
    args.analysis_signature = canonical_hash(parameters)
    parameters["analysis_signature"] = args.analysis_signature
    atomic_json(args.report_dir / "analysis_parameters.json", parameters)
    gt_sink = CsvSink(args.report_dir / "gt_diagnosis.csv", ["image_id", "grid", "gt_id", "iou"])
    image_sink = CsvSink(args.report_dir / "image_grid_stats.csv", ["image_id", "grid", "iou"])
    paired, image_stats, summaries = {}, {}, {}
    try:
        for grid in args.grids:
            results = []
            # First pass collects final detections. Official evaluation runs once per grid.
            for image_id in manifest["image_ids"]:
                path = args.cache / "candidates" / grid / f"{image_id}.npz"
                if not path.exists():
                    raise FileNotFoundError(f"Incomplete export. Resume dump to create {path}")
                cache, metadata = load_cache(path, manifest["signature"])
                if metadata["image_id"] != image_id or metadata["grid"] != grid:
                    raise ValueError(f"Cache identity mismatch at {path}")
                candidates = load_candidates(cache)
                final = merge_candidates(candidates, grid, args.score_threshold, args.nms_threshold, args.max_det)[2]
                results.extend(coco_results(image_id, candidates, final, args.category_id))
            atomic_json(args.report_dir / f"detections_{grid}.json", results)
            matches, official_counts, ap = official_evaluate(
                dataset, manifest["image_ids"], results, args.ious, args.eval_max_dets,
                args.report_dir / f"official_eval_{grid}.log",
            )
            del results
            totals = {str(threshold): Counter() for threshold in args.ious}
            for number, image_id in enumerate(manifest["image_ids"], 1):
                cache, metadata = load_cache(args.cache / "candidates" / grid / f"{image_id}.npz", manifest["signature"])
                stats = analyze_image(args, dataset, cache, metadata, matches, official_counts, gt_sink, paired)
                for row in stats:
                    image_sink.write(row)
                    image_stats.setdefault((image_id, row["iou"]), {})[grid] = row
                    excluded = {"image_id", "grid", "iou", "inference_seconds"}
                    totals[str(row["iou"])].update({key: value for key, value in row.items()
                                                   if key not in excluded and isinstance(value, int)})
                if number % 25 == 0 or number == len(manifest["image_ids"]):
                    print(f"Analyzed {grid}: {number}/{len(manifest['image_ids'])} images", flush=True)
            summary = {"official_ap": ap, "thresholds": {}}
            for threshold, tally in totals.items():
                values = dict(tally)
                values["direct_nms_lost_rate"] = safe_ratio(tally["direct_nms_lost_gt"], tally["pre_candidate_gt"])
                values["direct_final_lost_rate"] = safe_ratio(tally["direct_final_lost_gt"], tally["pre_candidate_gt"])
                values["official_recall"] = safe_ratio(tally["official_tp"], tally["normal_gt"])
                values["oracle_pre_recall"] = safe_ratio(tally["oracle_pre_tp"], tally["normal_gt"])
                values["budgeted_selection_gap_recall"] = safe_ratio(tally["budgeted_selection_gap"], tally["normal_gt"])
                summary["thresholds"][threshold] = values
            summaries[grid] = summary
    finally:
        gt_sink.close()
        image_sink.close()
    stability = paired_reports(args, dataset, paired, image_stats)
    summary = {
        "schema_version": SCHEMA_VERSION, "cache_signature": manifest["signature"],
        "analysis_signature": args.analysis_signature, "images": len(manifest["image_ids"]),
        "grids": summaries, "stability": stability,
        "oracle_scope": "Fixed candidate boxes, same-class one-to-one matching, chosen IoU and per-image output budget",
    }
    atomic_json(args.report_dir / "summary.json", summary)
    print(f"Analysis complete: {args.report_dir}", flush=True)
    for grid, report in summaries.items():
        primary = report["thresholds"][str(args.primary_iou)]
        print(f"{grid} @ IoU {args.primary_iou:g}: TP={primary['official_tp']}, "
              f"NMS lost GT={primary['direct_nms_lost_gt']}, "
              f"budgeted selection gap={primary['budgeted_selection_gap']}", flush=True)


def parser():
    main = argparse.ArgumentParser(description=__doc__)
    commands = main.add_subparsers(dest="command", required=True)
    export = commands.add_parser("dump", help="Export fixed-model predictions before filtering/merging")
    export.add_argument("--config", type=Path, default=REPO / "configs/deim_dfine/dfine_hgnetv2_s_tinyperson_online.yml")
    export.add_argument("--checkpoint", type=Path, help="Checkpoint path. Defaults to the mounted or local best_stg1")
    export.add_argument("--gt", type=Path, default=DATA / "annotations/tiny_set_test_all.json")
    export.add_argument("--image-root", type=Path, default=DATA / "test")
    export.add_argument("--tile-root", type=Path, default=DATA / "test_tiles")
    export.add_argument("--tile-json", type=Path, default=DATA / "annotations/tiny_set_test_tiles.json")
    export.add_argument("--out", type=Path, default=DEFAULT_CACHE)
    export.add_argument("--grids", nargs="+", choices=("baseline",) + PHASE_GRIDS, default=["baseline", *PHASE_GRIDS])
    export.add_argument("--image-ids", nargs="+", type=int)
    export.add_argument("--limit", type=int)
    export.add_argument("--batch-size", type=int, default=8, help="Maximum tiles per inference batch (default: 8)")
    export.add_argument("--device", default="auto", help="auto, cpu, cuda:0, etc.")
    export.add_argument("--seed", type=int, default=0)
    analysis = commands.add_parser("analyze", help="Replay merge and diagnose all three questions on CPU")
    analysis.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    analysis.add_argument("--gt", type=Path, help="Optional relocated copy of the same GT file")
    analysis.add_argument("--report-dir", type=Path)
    analysis.add_argument("--grids", nargs="+", choices=("baseline",) + PHASE_GRIDS)
    analysis.add_argument("--ious", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    analysis.add_argument("--primary-iou", type=float, default=0.5)
    analysis.add_argument("--score-threshold", type=float)
    analysis.add_argument("--nms-threshold", type=float)
    analysis.add_argument("--max-det", type=int, help="Override configured merge cap. 0 keeps all NMS survivors")
    analysis.add_argument("--eval-max-dets", type=int)
    analysis.add_argument("--evidence-floor", type=float, default=0.001)
    analysis.add_argument("--evidence-floors", nargs="+", type=float, default=[0.001, 0.01])
    analysis.add_argument("--fragment-iop", type=float, default=0.7)
    analysis.add_argument("--fragment-min-iog", type=float, default=0.1)
    analysis.add_argument("--fragment-max-iog", type=float, default=0.8)
    analysis.add_argument("--seam-distance", type=float, default=4.0)
    analysis.add_argument("--border-distance", type=float, default=16.0)
    return main


def main():
    arguments = parser().parse_args()
    if arguments.command == "dump":
        if arguments.batch_size <= 0 or (arguments.limit is not None and arguments.limit <= 0):
            raise ValueError("Batch size and limit must be positive")
        if len(set(arguments.grids)) != len(arguments.grids):
            raise ValueError("Grid names must be unique")
        dump(arguments)
    else:
        arguments.ious = sorted(set(arguments.ious))
        arguments.evidence_floors = sorted(set(arguments.evidence_floors))
        if not all(0 < value <= 1 for value in arguments.ious) or arguments.primary_iou not in arguments.ious:
            raise ValueError("Require IoUs in (0,1] and primary IoU included in --ious")
        if not all(0 <= value <= 1 for value in [arguments.evidence_floor, *arguments.evidence_floors,
                                               arguments.fragment_iop, arguments.fragment_min_iog,
                                               arguments.fragment_max_iog]):
            raise ValueError("Evidence scores and overlap ratios must be in [0,1]")
        if (arguments.fragment_min_iog > arguments.fragment_max_iog
                or not arguments.seam_distance >= 0 or not arguments.border_distance >= 0):
            raise ValueError("Invalid fragment interval or boundary distance")
        if arguments.grids and len(set(arguments.grids)) != len(arguments.grids):
            raise ValueError("Grid names must be unique")
        analyze(arguments)


if __name__ == "__main__":
    main()
