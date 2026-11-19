"""Export DEIM validation detections from all official TinyPerson tiles before NMS."""

import argparse
from collections import defaultdict
import hashlib
from io import BytesIO
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import torch

from official_tile_candidates import CACHE, DATASET, GT_SHA256, ROOT, SCHEMA, official_images, save_candidates, sha256

sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig
from torchvision.ops import nms


CONFIG = ROOT / "configs/deim_dfine/dfine_hgnetv2_s_tinyperson_online.yml"


def load_weights(model, checkpoint, use_ema):
    if checkpoint.suffix.lower() == ".zip":
        with ZipFile(checkpoint) as archive:
            members = [name for name in archive.namelist() if Path(name).suffix.lower() in {".pt", ".pth"}]
            if len(members) != 1:
                raise ValueError("Checkpoint ZIP must contain exactly one .pt or .pth file")
            raw = archive.read(members[0])
        checkpoint_sha = hashlib.sha256(raw).hexdigest()
        payload = torch.load(BytesIO(raw), map_location="cpu", weights_only=False)
    else:
        checkpoint_sha = sha256(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    key = "ema.module" if use_ema and "ema" in payload else "model"
    state = payload["ema"]["module"] if key == "ema.module" else payload.get("model", payload)
    model.load_state_dict(state, strict=True)
    return dict(checkpoint_sha256=checkpoint_sha, weights=key, epoch=payload.get("last_epoch"))


def configure(args, gt):
    config = YAMLConfig(str(args.config))
    cfg = config.yaml_cfg
    cfg["evaluator"]["merge_gt_file"] = str(gt)
    if cfg["evaluator"]["type"] != "TinyPersonMergeEvaluator":
        raise ValueError("Use the TinyPerson validation configuration")
    if "HGNetv2" in cfg:
        cfg["HGNetv2"]["pretrained"] = False
    loader = cfg["val_dataloader"]
    dataset = loader["dataset"]
    dataset["ann_file"] = str(gt.parent / "tiny_set_test_tiles.json")
    dataset["img_folder"] = str(args.dataset / "test_tiles")
    loader.pop("batch_size", None)
    loader.update(total_batch_size=args.batch_size, num_workers=args.workers, shuffle=False, drop_last=False)
    if not args.workers and "persistent_workers" in loader:
        loader["persistent_workers"] = False
    return config


def tile_geometry(evaluator, loader):
    dataset_tiles = loader.dataset.coco.imgs
    if set(loader.dataset.ids) != set(evaluator.block_info):
        raise ValueError("Validation tile IDs do not cover the official image grid")
    tiles = defaultdict(list)
    indices = {}
    for tile_id, (rect, image_id) in evaluator.block_info.items():
        tile = dataset_tiles[tile_id]
        if (tuple(tile["corner"]) != rect or tile["ori_image_id"] != image_id
                or (tile["width"], tile["height"]) != (rect[2] - rect[0], rect[3] - rect[1])):
            raise ValueError(f"Validation tile geometry differs from the official grid: {tile_id}")
        indices[tile_id] = len(tiles[image_id])
        tiles[image_id].append(rect)
    return tiles, indices


@torch.no_grad()
def collect(model, postprocessor, loader, evaluator, indices, device, smoke_batches):
    patch_ids = defaultdict(list)
    for batch, (samples, targets) in enumerate(loader, start=1):
        sizes = torch.stack([target["orig_size"] for target in targets]).to(device)
        predictions = postprocessor(model(samples.to(device)), sizes)
        for target, prediction in zip(targets, predictions):
            tile_id = int(target["image_id"].item())
            _, image_id = evaluator.block_info[tile_id]
            before = len(patch_ids[image_id])
            evaluator.update({tile_id: prediction})
            if image_id in evaluator.merged:
                count = len(evaluator.merged[image_id][1]) - before
                patch_ids[image_id].extend([indices[tile_id]] * count)
        if batch % 100 == 0 or batch == len(loader) or batch == smoke_batches:
            print(f"Inference {batch}/{len(loader)} batches, {len(evaluator._seen_tiles)} tiles", flush=True)
        if batch == smoke_batches:
            break
    return patch_ids


def candidate_rows(images, tiles, merged, patch_ids):
    rows = []
    nonperson = 0
    for image in images:
        iid = image["id"]
        boxes, scores, labels = merged.get(iid, (torch.empty((0, 4)), torch.empty(0), torch.empty(0)))
        person = labels.numpy() == 1
        nonperson += int((~person).sum())
        rows.append(dict(image_id=iid, tiles=np.asarray(tiles[iid], dtype=np.int32),
                         boxes=boxes.numpy()[person], scores=scores.numpy()[person],
                         patch_ids=np.asarray(patch_ids[iid], dtype=np.int32)[person]))
    return rows, nonperson


def parity_check(evaluator, rows):
    training = evaluator._build_predictions()
    matches = 0
    for row in rows:
        if row["image_id"] not in training:
            continue
        reference = training[row["image_id"]]
        person = reference["labels"] == 1
        boxes, scores = torch.from_numpy(row["boxes"]), torch.from_numpy(row["scores"])
        keep = nms(boxes, scores, evaluator.merge_nms_th)
        if not (torch.equal(boxes[keep], reference["boxes"][person])
                and torch.equal(scores[keep], reference["scores"][person])):
            matches += 1
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Validation .pth/.pt or a ZIP containing it.")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=CACHE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke-batches", type=int, default=0, help="Check initial batches without writing a cache.")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".npz" or args.batch_size < 1 or min(args.workers, args.smoke_batches) < 0:
        parser.error("Expected an .npz output, a positive batch size and nonnegative worker/smoke counts")
    gt = args.dataset / "annotations/tiny_set_test_all.json"
    images = official_images(gt)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    config = configure(args, gt)
    model = config.model
    weights = load_weights(model, args.checkpoint, config.yaml_cfg.get("use_ema", False))
    print(f"Checkpoint: {weights}, device: {device}", flush=True)
    model = model.eval().to(device)
    postprocessor = config.postprocessor.eval().to(device)
    loader, evaluator = config.val_dataloader, config.evaluator
    if evaluator.merge_nms_th != .5 or evaluator.max_det or evaluator.eval_max_dets != 200:
        raise ValueError("Expected validation IoU 0.50 and official maxDets=200")
    tiles, indices = tile_geometry(evaluator, loader)
    patch_ids = collect(model, postprocessor, loader, evaluator, indices, device, args.smoke_batches)
    rows, nonperson = candidate_rows(images, tiles, evaluator.merged, patch_ids)
    if args.smoke_batches or nonperson:
        mismatches = parity_check(evaluator, rows)
    else:
        mismatches = 0
    count = sum(len(row["scores"]) for row in rows)
    print(f"Collected {count:,} person boxes, {nonperson:,} other labels; NMS parity differences: {mismatches} images",
          flush=True)
    if args.smoke_batches:
        return
    if evaluator._seen_tiles != set(evaluator.block_info) or not count:
        raise ValueError("Inference did not produce a complete official candidate pool")
    metadata = dict(schema=SCHEMA, gt_sha256=GT_SHA256, **weights, score_threshold=evaluator.score_th,
                    tile_count=len(indices), nonperson_candidates=nonperson, nms_parity_differences=mismatches,
                    precision="float32", device=str(device), torch_version=str(torch.__version__),
                    config_sha256=sha256(args.config), tile_gt_sha256=sha256(Path(loader.dataset.ann_file)))
    save_candidates(args.output, rows, metadata)
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
