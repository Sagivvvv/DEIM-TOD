"""Read and write the candidate pool captured before TinyPerson whole-image NMS."""

import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT.parent / "datasets/Tinyperson"
GT = DATASET / "annotations/tiny_set_test_all.json"
OUTPUT = ROOT / "output/postprocess_benchmark"
SPLIT = "official_786_global_nms"
CACHE = OUTPUT / "official_786_pre_nms.npz"
REPORTS = ROOT / "TileStitchingNet/reports/postprocess_official_786_global_nms"
SCHEMA = "tinyperson-validation-pre-nms-v1"
GT_SHA256 = "8c7703d7058a75de37c45d4b0023769ad08ee14a6767ad4939c53e4763d1f8b1"
IMAGE_COUNT = 786


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def official_images(gt):
    if sha256(gt) != GT_SHA256:
        raise ValueError("Expected the official 786-image tiny_set_test_all.json")
    data = json.loads(gt.read_text(encoding="utf8"))
    return sorted(data["images"], key=lambda image: image["id"])


def run_dir(output, cache_sha):
    return output / SPLIT / cache_sha[:16]


def save_candidates(path, rows, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temp = Path(stream.name)
    try:
        np.savez_compressed(
            temp,
            metadata=json.dumps(metadata),
            image_ids=np.array([row["image_id"] for row in rows]),
            box_counts=np.array([len(row["scores"]) for row in rows]),
            tile_counts=np.array([len(row["tiles"]) for row in rows]),
            boxes=np.concatenate([row["boxes"] for row in rows]),
            scores=np.concatenate([row["scores"] for row in rows]),
            patch_ids=np.concatenate([row["patch_ids"] for row in rows]),
            tiles=np.concatenate([row["tiles"] for row in rows]),
        )
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def load_candidates(cache, gt):
    images = official_images(gt)
    if cache.suffix.lower() != ".npz":
        raise ValueError("Expected a new pre-NMS .npz export; legacy .pt caches cannot be used")
    if not cache.is_file():
        raise FileNotFoundError(f"Export the validation candidates first: {cache}")
    with cache.open("rb") as stream:
        cache_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as arrays:
            metadata = json.loads(arrays["metadata"].item())
            if metadata.get("schema") != SCHEMA or metadata.get("gt_sha256") != GT_SHA256:
                raise ValueError("Cache is not a TinyPerson official pre-NMS export")
            if arrays["image_ids"].tolist() != [im["id"] for im in images]:
                raise ValueError("Cache image IDs differ from the official split")
            box_counts = arrays["box_counts"]
            tile_counts = arrays["tile_counts"]
            boxes, scores, patch_ids, tiles = (arrays[k] for k in ("boxes", "scores", "patch_ids", "tiles"))
    if (box_counts.shape != (IMAGE_COUNT,) or tile_counts.shape != (IMAGE_COUNT,)
            or (box_counts < 0).any() or (tile_counts <= 0).any()
            or int(box_counts.sum()) != len(scores) or int(tile_counts.sum()) != len(tiles)
            or boxes.shape != (len(scores), 4) or scores.ndim != 1 or patch_ids.shape != scores.shape
            or boxes.dtype != np.float32 or scores.dtype != np.float32
            or tiles.shape != (len(tiles), 4)):
        raise ValueError("Invalid pre-NMS cache array shapes")
    if (not np.isfinite(boxes).all() or not np.isfinite(scores).all()
            or (boxes[:, 2:] < boxes[:, :2]).any()
            or (scores <= metadata["score_threshold"]).any() or (scores > 1).any()):
        raise ValueError("Invalid candidate coordinates or confidence scores")
    samples = []
    box_start = tile_start = 0
    for im, box_count, tile_count in zip(images, box_counts, tile_counts):
        box_end, tile_end = box_start + box_count, tile_start + tile_count
        window = slice(box_start, box_end)
        patch = patch_ids[window]
        rects = tiles[tile_start:tile_end]
        size = (im["width"], im["height"])
        if ((patch < 0).any() or (patch >= tile_count).any()
                or (rects[:, :2] < 0).any() or (rects[:, 2:] > size).any()
                or (rects[:, 2:] <= rects[:, :2]).any()):
            raise ValueError(f"Invalid tile mapping for image {im['id']}")
        samples.append(SimpleNamespace(
            image_id=str(im["id"]), img_size=size, boxes_global=boxes[window],
            scores=scores[window], class_ids=np.ones(box_count, dtype=np.int64),
            patch_ids=patch, patch_rects=rects[patch], tile_rects_all=rects,
        ))
        box_start, tile_start = box_end, tile_end
    return samples, dict(metadata, cache_sha256=cache_sha)
