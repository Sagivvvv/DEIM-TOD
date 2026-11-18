"""
Step 4.2 & Section 5: Offline Training Loop, Fast Inference Pipeline,
Baseline Post-Processors (Greedy-NMS, Soft-NMS, SAHI-NMM, WBF),
COCO-Style AP/AP50/AP75 Evaluator, and Ablation Study Runner.

Module: tile_stitching_postprocess/train_eval.py
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    try:
        from .datasetBuilder import (
            CroppedGTLabelAssigner,
            SyntheticTiledDatasetGenerator,
            TileSample,
            box_area_np,
            nms_np,
            pairwise_ios_np,
            pairwise_iou_np,
        )
    except ImportError:
        from .dataset_builder import (
            CroppedGTLabelAssigner,
            SyntheticTiledDatasetGenerator,
            TileSample,
            box_area_np,
            nms_np,
            pairwise_ios_np,
            pairwise_iou_np,
        )
    from .losses import TileStitchingLoss
    from .model import TORCH_BACKEND, TileStitchingNet, torch
except ImportError:
    try:
        from datasetBuilder import (
            CroppedGTLabelAssigner,
            SyntheticTiledDatasetGenerator,
            TileSample,
            box_area_np,
            nms_np,
            pairwise_ios_np,
            pairwise_iou_np,
        )
    except ImportError:
        from dataset_builder import (
            CroppedGTLabelAssigner,
            SyntheticTiledDatasetGenerator,
            TileSample,
            box_area_np,
            nms_np,
            pairwise_ios_np,
            pairwise_iou_np,
        )
    from losses import TileStitchingLoss
    from model import TORCH_BACKEND, TileStitchingNet, torch


# ============================================================================
# 1. Traditional Handcrafted Post-Processing Baselines
#    (Greedy-NMS, Soft-NMS, SAHI-NMM, WBF)
# ============================================================================

def run_greedy_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.5,
    score_thresh: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Baseline 1: Standard Greedy NMS."""
    keep_mask = scores >= score_thresh
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    keep = nms_np(boxes, scores, iou_thresh=iou_thresh)
    return boxes[keep], scores[keep]


def run_soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    sigma: float = 0.5,
    iou_thresh: float = 0.3,
    score_thresh: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Baseline 2: Gaussian Soft-NMS."""
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    dets = boxes.astype(np.float32).copy()
    sc = scores.astype(np.float32).copy()
    n = len(dets)

    for i in range(n):
        max_pos = i + int(np.argmax(sc[i:]))
        dets[[i, max_pos]] = dets[[max_pos, i]]
        sc[[i, max_pos]] = sc[[max_pos, i]]

        if i + 1 >= n:
            break
        ious = pairwise_iou_np(dets[i:i+1], dets[i+1:])[0]
        decay = np.where(ious > iou_thresh, np.exp(-(ious ** 2) / sigma), 1.0)
        sc[i+1:] *= decay

    keep = sc >= score_thresh
    order = np.argsort(sc[keep])[::-1]
    return dets[keep][order], sc[keep][order]


def run_sahi_nmm_ios(
    boxes: np.ndarray,
    scores: np.ndarray,
    ios_thresh: float = 0.5,
    score_thresh: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Baseline 3: SAHI Non-Maximum Merging (NMM) based on IoS (Intersection over Smaller Area).
    Demonstrates two classic failure modes documented in the plan:
      1) Cross-cutline truncated fragments with near-0 IoS fail to merge into a full box.
      2) Nested same-class targets ("adult holding child") have IoS > 0.85 and get falsely
         merged/swallowed into the outer adult box, causing severe missed detections.
    """
    keep_mask = scores >= score_thresh
    boxes = boxes[keep_mask].copy()
    scores = scores[keep_mask].copy()
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    order = scores.argsort()[::-1]
    suppressed = np.zeros(len(boxes), dtype=bool)
    merged_boxes = []
    merged_scores = []

    for idx in order:
        if suppressed[idx]:
            continue
        suppressed[idx] = True
        curr_box = boxes[idx].copy()
        curr_score = scores[idx]

        # Find all remaining un-suppressed boxes with IoS >= ios_thresh
        rem_indices = [j for j in order if not suppressed[j]]
        if rem_indices:
            rem_boxes = boxes[rem_indices]
            ios_vals = pairwise_ios_np(curr_box[None, :], rem_boxes)[0]
            to_merge = [rem_indices[k] for k, v in enumerate(ios_vals) if v >= ios_thresh]

            if to_merge:
                cluster_idx = [idx] + to_merge
                cluster_boxes = boxes[cluster_idx]
                cluster_scores = scores[cluster_idx]
                # SAHI confidence-weighted coordinate union/merge
                x1 = np.min(cluster_boxes[:, 0])
                y1 = np.min(cluster_boxes[:, 1])
                x2 = np.max(cluster_boxes[:, 2])
                y2 = np.max(cluster_boxes[:, 3])
                curr_box = np.array([x1, y1, x2, y2], dtype=np.float32)
                curr_score = float(np.max(cluster_scores))
                for m_i in to_merge:
                    suppressed[m_i] = True

        merged_boxes.append(curr_box)
        merged_scores.append(curr_score)

    return np.array(merged_boxes, dtype=np.float32), np.array(merged_scores, dtype=np.float32)


def run_wbf(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.55,
    score_thresh: float = 0.18,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Baseline 4: Weighted Boxes Fusion (WBF).
    Clusters boxes by standard IoU and averages coordinates weighted by confidence scores.
    Because standard IoU between cross-tile fragments is ~0, WBF cannot stitch split fragments,
    and when fragments partially overlap, weighted averaging shrinks the box rather than taking
    the outer bounding envelope.
    """
    keep_mask = scores >= score_thresh
    boxes = boxes[keep_mask].copy()
    scores = scores[keep_mask].copy()
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    order = scores.argsort()[::-1]
    clusters: List[List[int]] = []
    cluster_rep_boxes: List[np.ndarray] = []

    for idx in order:
        b = boxes[idx]
        if not cluster_rep_boxes:
            clusters.append([idx])
            cluster_rep_boxes.append(b.copy())
            continue

        rep_arr = np.array(cluster_rep_boxes, dtype=np.float32)
        ious = pairwise_iou_np(b[None, :], rep_arr)[0]
        best_c = int(np.argmax(ious))
        if ious[best_c] >= iou_thresh:
            clusters[best_c].append(idx)
            c_idx = clusters[best_c]
            w = scores[c_idx][:, None]
            cluster_rep_boxes[best_c] = np.sum(boxes[c_idx] * w, axis=0) / (np.sum(w) + 1e-7)
        else:
            clusters.append([idx])
            cluster_rep_boxes.append(b.copy())

    out_boxes = np.array(cluster_rep_boxes, dtype=np.float32)
    out_scores = np.array([float(np.max(scores[c])) for c in clusters], dtype=np.float32)
    sort_idx = np.argsort(out_scores)[::-1]
    return out_boxes[sort_idx], out_scores[sort_idx]


# ============================================================================
# 2. Step 4.2: TileStitchingNet Inference Wrapper
# ============================================================================

class TileStitchingPredictor:
    """
    Step 4.2 Fast Inference Pipeline for TileStitchingNet:
      1. Single forward pass through TileStitchingNet (< 3ms per image).
      2. Directly filter out suppressed boxes with s_i < score_thresh (default 0.05~0.25).
      3. Optional lightweight high-threshold fallback NMS (iou_thresh = 0.85).
    """

    def __init__(
        self,
        model: TileStitchingNet,
        score_thresh: float = 0.25,
        fallback_nms_iou: float = 0.85,
    ):
        self.model = model
        self.score_thresh = score_thresh
        self.fallback_nms_iou = fallback_nms_iou

    def predict_sample(self, sample: TileSample) -> Tuple[np.ndarray, np.ndarray, float]:
        self.model.eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            nf = torch.from_numpy(sample.node_features)
            bg = torch.from_numpy(sample.boxes_global)
            outputs = self.model(nf, bg)
            pred_boxes = outputs["pred_boxes"].numpy()
            pred_scores = outputs["pred_scores"].numpy()

            keep = pred_scores >= self.score_thresh
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]

            if len(pred_boxes) > 0 and self.fallback_nms_iou < 1.0:
                nms_keep = nms_np(pred_boxes, pred_scores, iou_thresh=self.fallback_nms_iou)
                pred_boxes = pred_boxes[nms_keep]
                pred_scores = pred_scores[nms_keep]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return pred_boxes, pred_scores, latency_ms


# ============================================================================
# 3. COCO-Style Detection Evaluator (AP, AP50, AP75, Nested Recall, Split AP75)
# ============================================================================

def compute_ap_at_iou(
    all_pred_boxes: List[np.ndarray],
    all_pred_scores: List[np.ndarray],
    all_gt_boxes: List[np.ndarray],
    iou_thresh: float = 0.5,
) -> Tuple[float, float]:
    """Compute VOC/COCO 101-point interpolated Average Precision and Recall at `iou_thresh`."""
    tp_list = []
    fp_list = []
    score_list = []
    total_gts = sum(len(g) for g in all_gt_boxes)
    if total_gts == 0:
        return 0.0, 0.0

    for p_boxes, p_scores, g_boxes in zip(all_pred_boxes, all_pred_scores, all_gt_boxes):
        if len(p_boxes) == 0:
            continue
        order = np.argsort(p_scores)[::-1]
        p_boxes = p_boxes[order]
        p_scores = p_scores[order]

        matched_gt = np.zeros(len(g_boxes), dtype=bool)
        if len(g_boxes) == 0:
            tp_list.append(np.zeros(len(p_boxes), dtype=np.float32))
            fp_list.append(np.ones(len(p_boxes), dtype=np.float32))
            score_list.append(p_scores)
            continue

        ious = pairwise_iou_np(p_boxes, g_boxes)
        tps = np.zeros(len(p_boxes), dtype=np.float32)
        fps = np.zeros(len(p_boxes), dtype=np.float32)

        for i in range(len(p_boxes)):
            available = np.where(~matched_gt, ious[i], -1.0)
            best_g = int(np.argmax(available))
            if ious[i, best_g] >= iou_thresh and not matched_gt[best_g]:
                matched_gt[best_g] = True
                tps[i] = 1.0
            else:
                fps[i] = 1.0

        tp_list.append(tps)
        fp_list.append(fps)
        score_list.append(p_scores)

    if not score_list:
        return 0.0, 0.0

    all_scores = np.concatenate(score_list)
    all_tps = np.concatenate(tp_list)
    all_fps = np.concatenate(fp_list)

    global_order = np.argsort(all_scores)[::-1]
    all_tps = all_tps[global_order]
    all_fps = all_fps[global_order]

    cum_tp = np.cumsum(all_tps)
    cum_fp = np.cumsum(all_fps)

    recalls = cum_tp / float(total_gts)
    precisions = cum_tp / np.maximum(1e-7, cum_tp + cum_fp)

    # 101-point COCO interpolation
    ap = 0.0
    for r_thresh in np.linspace(0.0, 1.0, 101):
        p_at_r = precisions[recalls >= r_thresh]
        ap += float(np.max(p_at_r)) if len(p_at_r) > 0 else 0.0
    ap /= 101.0
    max_recall = float(recalls[-1]) if len(recalls) > 0 else 0.0
    return ap * 100.0, max_recall * 100.0


def evaluate_postprocessor_on_dataset(
    samples: List[TileSample],
    pred_boxes_list: List[np.ndarray],
    pred_scores_list: List[np.ndarray],
    latencies_ms: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Evaluate predictions across AP (0.50:0.95), AP50, AP75, Nested-Target Recall@50,
    and Cross-Cutline Split Target AP75.
    """
    gt_boxes_all = [s.gt_boxes for s in samples]

    ap50, rec50 = compute_ap_at_iou(pred_boxes_list, pred_scores_list, gt_boxes_all, iou_thresh=0.50)
    ap75, rec75 = compute_ap_at_iou(pred_boxes_list, pred_scores_list, gt_boxes_all, iou_thresh=0.75)

    iou_thresholds = np.linspace(0.50, 0.95, 10)
    aps = [
        compute_ap_at_iou(pred_boxes_list, pred_scores_list, gt_boxes_all, iou_thresh=float(t))[0]
        for t in iou_thresholds
    ]
    m_ap = float(np.mean(aps))

    # Subgroup 1: Nested small targets ("child" inside "adult": GT box with IoS > 0.75 against another larger GT box)
    nested_gts_list = []
    # Subgroup 2: Cross-cutline split GTs (GT boxes matched by >= 2 fragments from different patches)
    split_gts_list = []

    for s in samples:
        g = s.gt_boxes
        ios_mat = pairwise_ios_np(g, g)
        np.fill_diagonal(ios_mat, 0.0)
        areas = box_area_np(g)
        is_nested_small = np.zeros(len(g), dtype=bool)
        for idx_a in range(len(g)):
            for idx_b in range(len(g)):
                if idx_a != idx_b and ios_mat[idx_a, idx_b] > 0.70 and areas[idx_a] < areas[idx_b]:
                    is_nested_small[idx_a] = True
        nested_gts_list.append(g[is_nested_small])

        is_split_gt = np.zeros(len(g), dtype=bool)
        if s.gt_id is not None:
            for k in range(len(g)):
                p_set = np.unique(s.patch_ids[s.gt_id == k])
                if len(p_set) >= 2:
                    is_split_gt[k] = True
        split_gts_list.append(g[is_split_gt])

    # Compute Nested Small Target Recall@IoU=0.50
    nested_hits = 0
    nested_total = 0
    for p_boxes, n_gts in zip(pred_boxes_list, nested_gts_list):
        nested_total += len(n_gts)
        if len(n_gts) > 0 and len(p_boxes) > 0:
            ious = pairwise_iou_np(n_gts, p_boxes)
            nested_hits += int(np.sum(np.max(ious, axis=1) >= 0.50))
    nested_recall = (100.0 * nested_hits / max(1, nested_total))

    # Compute Split Target Recall@IoU=0.75 (high-IoU synthesis accuracy on truncated objects)
    split_hits75 = 0
    split_total = 0
    for p_boxes, sp_gts in zip(pred_boxes_list, split_gts_list):
        split_total += len(sp_gts)
        if len(sp_gts) > 0 and len(p_boxes) > 0:
            ious = pairwise_iou_np(sp_gts, p_boxes)
            split_hits75 += int(np.sum(np.max(ious, axis=1) >= 0.75))
    split_rec75 = (100.0 * split_hits75 / max(1, split_total))

    avg_lat = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    return {
        "AP": round(m_ap, 2),
        "AP50": round(ap50, 2),
        "AP75": round(ap75, 2),
        "Nested_Recall50": round(nested_recall, 2),
        "Split_Recall75": round(split_rec75, 2),
        "Latency_ms": round(avg_lat, 3),
    }


# ============================================================================
# 4. Offline Training Loop for TileStitchingNet
# ============================================================================

def train_tile_stitching_net(
    train_samples: List[TileSample],
    val_samples: Optional[List[TileSample]] = None,
    roi_dim: int = 0,
    d_model: int = 64,
    num_layers: int = 2,
    use_cutline_dist: bool = True,
    use_iou1d: bool = True,
    use_soft_envelope: bool = True,
    epochs: int = 18,
    lr: float = 3e-3,
    batch_size: int = 1000,
    lambda_stitch: float = 2.0,
    lambda_edge: float = 1.0,
    val_freq: int = 1,
    save_best_path: Optional[str] = "best_tilestitching.pth",
    seed: int = 42,
    verbose: bool = True,
    evaluator=None,
    selection_metric: str = "Tiny_AP50",
    prediction_score_thresh: float = 0.0,
) -> Tuple[TileStitchingNet, List[Dict[str, float]]]:
    """
    Offline training pipeline for TileStitchingNet on exported slice predictions.
    Supports optional RoI appearance features (e.g. C_roi = 64) and dynamic validation
    to checkpoint the best model weights based on Val AP.
    """
    if not train_samples or not any(len(s.boxes_global) for s in train_samples):
        raise ValueError("Training set has no candidates")
    try:
        from .stage1 import regression_diagnostics
    except ImportError:
        from stage1 import regression_diagnostics
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Automatically detect roi_dim from candidate node_features if not explicitly passed
    if roi_dim == 0 and len(train_samples) > 0 and train_samples[0].node_features is not None:
        detected_roi = train_samples[0].node_features.shape[1] - 12
        if detected_roi > 0:
            roi_dim = detected_roi

    model = TileStitchingNet(
        in_dim=12,
        roi_dim=roi_dim,
        d_model=d_model,
        num_layers=num_layers,
        use_cutline_dist=use_cutline_dist,
        use_iou1d=use_iou1d,
        use_soft_envelope=use_soft_envelope,
    )
    criterion = TileStitchingLoss(
        lambda_stitch=lambda_stitch,
        lambda_edge=lambda_edge,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    history: List[Dict[str, float]] = []
    best_val_ap = -1.0
    best_epoch = -1
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        ep_losses = []
        ep_rescore = []
        ep_stitch = []
        ep_edge = []
        ep_iou = []
        ep_diag = []
        grad_norms, update_norms = [], []
        optimizer_steps = 0

        # Cosine / step learning rate decay
        # 中3修复：torch.optim.AdamW 实际读取 param_groups[i]['lr']，给实例挂 .lr 属性无效(全程恒定 LR)；
        # 这里同时写 param_groups(torch 后端) 与 .lr(model.py 的 numpy 回退后端)，两者皆生效。
        if epoch == int(epochs * 0.65):
            new_lr = lr * 0.35
        elif epoch == int(epochs * 0.85):
            new_lr = lr * 0.15
        else:
            new_lr = None
        if new_lr is not None:
            optimizer.lr = new_lr  # type: ignore[attr-defined]  # numpy 回退后端
            for _pg in getattr(optimizer, "param_groups", []):   # torch AdamW
                _pg["lr"] = new_lr

        perm = np.random.permutation(len(train_samples))
        accum_tokens = 0
        accum_images = 0
        optimizer.zero_grad()

        def flush_gradients(count):
            # Each image loss is already normalized; average images in this window.
            params = list(model.box_refine_head.parameters())
            before = [p.detach().clone() for p in params]
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(count)
            grad_norms.append(float(sum(p.grad.detach().square().sum().item() for p in params if p.grad is not None)**.5))
            optimizer.step()
            update_norms.append(float(sum((p.detach()-b).square().sum().item() for p,b in zip(params,before))**.5))
            optimizer.zero_grad()


        for step_i, idx in enumerate(perm):
            s = train_samples[int(idx)]
            n_boxes = len(s.boxes_global)
            if n_boxes == 0:
                continue

            nf = torch.from_numpy(s.node_features)
            bg = torch.from_numpy(s.boxes_global)
            t_score = torch.from_numpy(s.target_score)
            t_reg = torch.from_numpy(s.target_reg)
            t_edge = torch.from_numpy(s.target_edge)
            t_leader = torch.from_numpy(s.is_leader.astype(np.float32)) > 0.5

            outputs = model(nf, bg)
            loss_dict = criterion(
                model_outputs=outputs,
                target_score=t_score,
                target_reg=t_reg,
                target_edge=t_edge,
                is_leader=t_leader,
            )

            loss_total = loss_dict["loss_total"]
            loss_total.backward()
            accum_tokens += n_boxes
            accum_images += 1
            ep_diag.append(regression_diagnostics(outputs, s))

            ep_losses.append(float(loss_total.item()))
            ep_rescore.append(float(loss_dict["loss_rescore"].item()))
            ep_stitch.append(float(loss_dict["loss_stitch"].item()))
            ep_edge.append(float(loss_dict["loss_edge"].item()))
            ep_iou.append(float(loss_dict["mean_leader_iou"].item()))

            if accum_tokens >= batch_size:
                flush_gradients(accum_images)
                optimizer_steps += 1
                accum_tokens = accum_images = 0

        # Flush independently of the final sample, which can have zero candidates.
        if accum_images:
            flush_gradients(accum_images)
            optimizer_steps += 1

        record = {
            "epoch": epoch,
            "loss_total": round(float(np.mean(ep_losses)), 4),
            "loss_rescore": round(float(np.mean(ep_rescore)), 4),
            "loss_stitch": round(float(np.mean(ep_stitch)), 4),
            "loss_edge": round(float(np.mean(ep_edge)), 4),
            "mean_leader_iou": round(float(np.mean(ep_iou)), 4),
        }

        record.update({"reg_"+k: float(np.mean([d[k] for d in ep_diag if k in d]))
                       for k in {k for d in ep_diag for k in d}})
        record.update(reg_head_grad_norm=float(np.mean(grad_norms)),
                      reg_head_update_norm=float(np.mean(update_norms)), optimizer_steps=optimizer_steps)

        # 动态验证集评估与最优模型保存
        val_metric_str = ""
        if val_samples is not None and len(val_samples) > 0 and (epoch % val_freq == 0 or epoch == epochs):
            model.eval()
            pred_temp = TileStitchingPredictor(model=model, score_thresh=prediction_score_thresh, fallback_nms_iou=0.85)
            pb_list, ps_list = [], []
            for vs in val_samples:
                b, sc, _ = pred_temp.predict_sample(vs)
                pb_list.append(b)
                ps_list.append(sc)
            val_eval = (evaluator or evaluate_postprocessor_on_dataset)(val_samples, pb_list, ps_list)
            cur_ap = val_eval[selection_metric if evaluator else "AP"]
            if not np.isfinite(cur_ap):
                raise ValueError("Checkpoint metric is undefined; validation has no eligible GT")
            record["validation_metrics"] = val_eval
            record["selection_metric"] = selection_metric if evaluator else "AP"
            record["val_ap"] = cur_ap
            record["val_ap50"] = val_eval["AP50"]
            record["val_ap75"] = val_eval["AP75"]
            val_metric_str = f" | Val_{record['selection_metric']}={cur_ap:.2f} (AP50={val_eval['AP50']:.2f}, AP75={val_eval['AP75']:.2f})"
            if cur_ap > best_val_ap:
                best_val_ap = cur_ap
                best_epoch = epoch
                if hasattr(model, "state_dict"):
                    best_weights = {k: v.clone() if hasattr(v, "clone") else v.copy() for k, v in model.state_dict().items()}
                if save_best_path:
                    torch.save(best_weights if best_weights is not None else model, save_best_path)

        history.append(record)
        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs or "val_ap" in record):
            print(
                f"[Epoch {epoch:02d}/{epochs:02d}] "
                f"L_total={record['loss_total']:.4f} | "
                f"L_rescore={record['loss_rescore']:.4f} | "
                f"L_stitch={record['loss_stitch']:.4f} | "
                f"L_edge={record['loss_edge']:.4f} | "
                f"Leader_IoU={record['mean_leader_iou']:.4f}"
                f"{val_metric_str}"
                f" | Reg(raw/env/ref)={record.get('reg_raw_iou',0):.4f}/{record.get('reg_env_iou',0):.4f}/{record.get('reg_refined_iou',0):.4f}"
                f" | Saturation={record.get('reg_clamp_saturation',0):.3f} Grad={record['reg_head_grad_norm']:.4g}"
            )

    if best_weights is not None and hasattr(model, "load_state_dict"):
        model.load_state_dict(best_weights)
        if verbose:
            print(f"\n>>> [Best Checkpoint] 已成功保存并加载最佳模型轮次 Epoch {best_epoch:02d} (最优 Val_AP = {best_val_ap:.2f}) -> {save_best_path}")

    return model, history


# ============================================================================
# 5. Full Benchmark & Ablation Suite (Section 5 Checklist)
# ============================================================================

def run_full_benchmark_and_ablations(
    output_json_path: Optional[str] = None,
) -> Dict[str, object]:
    """
    Executes the complete Section 5 Experimental Verification Checklist:
      1. Main Comparison Table: Standard NMS, Soft-NMS, WBF, SAHI (NMM-IoS), and Ours (TileStitchingNet).
      2. Ablation Studies:
         - w/o Cut-line Distance Encoding & 1D Cut-Projection IoU_1D
         - IoS Label Assignment vs Cropped-GT Standard IoU Label Assignment
         - 1-Layer Attention vs 2-Layer Attention
         - w/o Soft Bounding Envelope Operator
    """
    print("=" * 80)
    print(f"Running BALR-Attention TileStitchingNet Benchmark (Backend: {TORCH_BACKEND})")
    print("=" * 80)

    # 1. Build training and validation datasets with Cropped-GT Label Assigner
    cropped_assigner = CroppedGTLabelAssigner(local_iou_thresh=0.5, matching_metric="cropped_iou")
    ios_assigner = CroppedGTLabelAssigner(local_iou_thresh=0.5, matching_metric="ios")

    train_gen = SyntheticTiledDatasetGenerator(seed=101)
    val_gen = SyntheticTiledDatasetGenerator(seed=202)

    train_samples = train_gen.generate_dataset(num_samples=28, assigner=cropped_assigner)
    val_samples = val_gen.generate_dataset(num_samples=16, assigner=cropped_assigner)

    # Also build IoS-assigned training dataset for Ablation 2
    train_gen_ios = SyntheticTiledDatasetGenerator(seed=101)
    train_samples_ios = train_gen_ios.generate_dataset(num_samples=28, assigner=ios_assigner)

    # 2. Evaluate Handcrafted Baselines on Val Set
    main_results: Dict[str, Dict[str, float]] = {}

    for name, fn in [
        ("Standard Greedy-NMS", lambda b, s: run_greedy_nms(b, s, iou_thresh=0.50)),
        ("Soft-NMS (Gaussian)", lambda b, s: run_soft_nms(b, s, sigma=0.50)),
        ("WBF (Weighted Boxes Fusion)", lambda b, s: run_wbf(b, s, iou_thresh=0.55)),
        ("SAHI (NMM-IoS)", lambda b, s: run_sahi_nmm_ios(b, s, ios_thresh=0.50)),
    ]:
        p_boxes_all, p_scores_all, lats = [], [], []
        for s in val_samples:
            t0 = time.perf_counter()
            pb, ps = fn(s.boxes_global, s.scores)
            lats.append((time.perf_counter() - t0) * 1000.0)
            p_boxes_all.append(pb)
            p_scores_all.append(ps)
        main_results[name] = evaluate_postprocessor_on_dataset(val_samples, p_boxes_all, p_scores_all, lats)

    # 3. Train Full Proposed Model (Ours: 2-Layer BALR-Attention + Soft Envelope + Cropped-GT)
    print("\n--- Training Full Model: Ours (TileStitchingNet) ---")
    full_model, full_history = train_tile_stitching_net(
        train_samples=train_samples,
        val_samples=val_samples,
        d_model=64,
        num_layers=2,
        use_cutline_dist=True,
        use_iou1d=True,
        use_soft_envelope=True,
        epochs=18,
        lr=3.5e-3,
        seed=42,
        verbose=True,
    )
    predictor_full = TileStitchingPredictor(full_model, score_thresh=0.25, fallback_nms_iou=0.85)
    pb_full, ps_full, lat_full = [], [], []
    for s in val_samples:
        b, sc, l_ms = predictor_full.predict_sample(s)
        pb_full.append(b)
        ps_full.append(sc)
        lat_full.append(l_ms)
    main_results["Ours (TileStitchingNet)"] = evaluate_postprocessor_on_dataset(
        val_samples, pb_full, ps_full, lat_full
    )

    # 4. Run Section 5 Ablation Studies
    ablation_results: Dict[str, Dict[str, float]] = {}

    # Ablation A: w/o Cut-line Distance Encoding & 1D Cut-Projection IoU_1D
    print("\n--- Ablation 1: w/o Cut-line Distance & IoU_1D ---")
    model_no_prior, _ = train_tile_stitching_net(
        train_samples=train_samples,
        d_model=64,
        num_layers=2,
        use_cutline_dist=False,
        use_iou1d=False,
        use_soft_envelope=True,
        epochs=18,
        lr=3.5e-3,
        seed=42,
        verbose=False,
    )
    pred_no_prior = TileStitchingPredictor(model_no_prior, score_thresh=0.25, fallback_nms_iou=0.85)
    pb_a, ps_a, lat_a = zip(*[pred_no_prior.predict_sample(s) for s in val_samples])
    ablation_results["w/o Cutline Dist & IoU_1D"] = evaluate_postprocessor_on_dataset(
        val_samples, list(pb_a), list(ps_a), list(lat_a)
    )

    # Ablation B: IoS Label Assignment vs Cropped-GT Standard IoU Label Assignment
    print("--- Ablation 2: IoS Label Assignment (instead of Cropped-GT IoU) ---")
    model_ios, _ = train_tile_stitching_net(
        train_samples=train_samples_ios,
        d_model=64,
        num_layers=2,
        use_cutline_dist=True,
        use_iou1d=True,
        use_soft_envelope=True,
        epochs=18,
        lr=3.5e-3,
        seed=42,
        verbose=False,
    )
    pred_ios = TileStitchingPredictor(model_ios, score_thresh=0.25, fallback_nms_iou=0.85)
    pb_b, ps_b, lat_b = zip(*[pred_ios.predict_sample(s) for s in val_samples])
    ablation_results["IoS Label Assignment (w/o Cropped-GT)"] = evaluate_postprocessor_on_dataset(
        val_samples, list(pb_b), list(ps_b), list(lat_b)
    )

    # Ablation C: 1-Layer Attention vs 2-Layer Attention
    print("--- Ablation 3: 1-Layer BALR-Attention (w/o 2-Hop Message Passing) ---")
    model_1layer, _ = train_tile_stitching_net(
        train_samples=train_samples,
        d_model=64,
        num_layers=1,
        use_cutline_dist=True,
        use_iou1d=True,
        use_soft_envelope=True,
        epochs=18,
        lr=3.5e-3,
        seed=42,
        verbose=False,
    )
    pred_1layer = TileStitchingPredictor(model_1layer, score_thresh=0.25, fallback_nms_iou=0.85)
    pb_c, ps_c, lat_c = zip(*[pred_1layer.predict_sample(s) for s in val_samples])
    ablation_results["1-Layer BALR-Attention"] = evaluate_postprocessor_on_dataset(
        val_samples, list(pb_c), list(ps_c), list(lat_c)
    )

    # Ablation D: w/o Soft Bounding Envelope Operator
    print("--- Ablation 4: w/o Soft Bounding Envelope Operator ---")
    model_no_env, _ = train_tile_stitching_net(
        train_samples=train_samples,
        d_model=64,
        num_layers=2,
        use_cutline_dist=True,
        use_iou1d=True,
        use_soft_envelope=False,
        epochs=18,
        lr=3.5e-3,
        seed=42,
        verbose=False,
    )
    pred_no_env = TileStitchingPredictor(model_no_env, score_thresh=0.25, fallback_nms_iou=0.85)
    pb_d, ps_d, lat_d = zip(*[pred_no_env.predict_sample(s) for s in val_samples])
    ablation_results["w/o Soft Bounding Envelope"] = evaluate_postprocessor_on_dataset(
        val_samples, list(pb_d), list(ps_d), list(lat_d)
    )

    ablation_results["Full Model (Ours: 2-Layer + Soft Env + Cropped-GT)"] = main_results["Ours (TileStitchingNet)"]

    # Print formatted tables
    print("\n" + "=" * 88)
    print("TABLE 1: Main Comparison against Handcrafted Post-Processing Baselines")
    print("=" * 88)
    header = f"{'Method':<32} | {'AP':>6} | {'AP50':>6} | {'AP75':>6} | {'NestedRec':>9} | {'SplitR75':>8} | {'Time(ms)':>8}"
    print(header)
    print("-" * len(header))
    for k, v in main_results.items():
        print(
            f"{k:<32} | {v['AP']:>6.2f} | {v['AP50']:>6.2f} | {v['AP75']:>6.2f} | "
            f"{v['Nested_Recall50']:>8.2f}% | {v['Split_Recall75']:>7.2f}% | {v['Latency_ms']:>8.3f}"
        )

    print("\n" + "=" * 88)
    print("TABLE 2: Section 5 Ablation Study Results")
    print("=" * 88)
    header_ab = f"{'Ablation Variant':<50} | {'AP':>6} | {'AP50':>6} | {'AP75':>6} | {'NestedRec':>9} | {'SplitR75':>8}"
    print(header_ab)
    print("-" * len(header_ab))
    for k, v in ablation_results.items():
        print(
            f"{k:<50} | {v['AP']:>6.2f} | {v['AP50']:>6.2f} | {v['AP75']:>6.2f} | "
            f"{v['Nested_Recall50']:>8.2f}% | {v['Split_Recall75']:>7.2f}%"
        )

    summary = {
        "backend": TORCH_BACKEND,
        "main_comparison": main_results,
        "ablation_study": ablation_results,
        "training_history": full_history,
    }
    if output_json_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    run_full_benchmark_and_ablations()
