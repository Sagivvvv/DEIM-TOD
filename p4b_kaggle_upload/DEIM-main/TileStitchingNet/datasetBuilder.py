"""
Step 1 & Step 2: Offline Tile Data Preparation, Boundary Distance Encoding,
and Cropped-GT Anti-Nested-Kill Label Assignment.

Module: tile_stitching_postprocess/dataset_builder.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object


# ============================================================================
# 1. Core Box Geometry Utilities (NumPy & PyTorch Compatible)
# ============================================================================

def box_area_np(boxes: np.ndarray) -> np.ndarray:
    """Compute area of boxes in [x1, y1, x2, y2] format. Shape: [N, 4] -> [N]."""
    w = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    h = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return w * h


def pairwise_iou_np(boxes1: np.ndarray, boxes2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Compute standard pairwise IoU between boxes1 [N, 4] and boxes2 [M, 4].
    Strictly uses Standard IoU: Area(A n B) / Area(A u B).
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    wh = np.maximum(0.0, rb - lt)                              # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]                          # [N, M]

    area1 = box_area_np(boxes1)[:, None]                       # [N, 1]
    area2 = box_area_np(boxes2)[None, :]                       # [1, M]
    union = area1 + area2 - inter
    return inter / (union + eps)


def pairwise_ios_np(boxes1: np.ndarray, boxes2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Compute pairwise IoS (Intersection over Smaller Area) between boxes1 [N, 4] and boxes2 [M, 4].
    Provided ONLY for SAHI-NMM baseline comparison and Ablation Study (IoS vs Cropped-GT IoU).
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.maximum(0.0, rb - lt)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = box_area_np(boxes1)[:, None]
    area2 = box_area_np(boxes2)[None, :]
    smaller_area = np.minimum(area1, area2)
    return inter / (smaller_area + eps)


def intersect_boxes_np(boxes1: np.ndarray, boxes2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Intersect boxes1 [N, 4] with boxes2 [M, 4] (e.g., patch regions with GT boxes).
    Returns:
        cropped_boxes: [N, M, 4] in [x1, y1, x2, y2]
        valid_mask: [N, M] boolean indicating non-empty intersection (w > 0 and h > 0)
    """
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    wh = np.maximum(0.0, rb - lt)
    valid_mask = (wh[:, :, 0] > 1e-4) & (wh[:, :, 1] > 1e-4)
    cropped = np.concatenate([lt, lt + wh], axis=-1)           # [N, M, 4]
    return cropped, valid_mask


def nms_np(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.7) -> np.ndarray:
    """Standard Greedy NMS in NumPy. Returns kept indices."""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = pairwise_iou_np(boxes[i:i+1], boxes[order[1:]])[0]
        inds = np.where(ious <= iou_thresh)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int64)


# ============================================================================
# 2. Step 1: Tile Grid Generation & Input Feature Extraction
# ============================================================================

@dataclass
class TileSample:
    """Container for a single high-resolution image's offline tile predictions and labels."""
    image_id: str
    img_size: Tuple[int, int]             # (W_img, H_img)
    boxes_global: np.ndarray              # [N, 4] float32 in global pixel coords (x1, y1, x2, y2)
    scores: np.ndarray                    # [N] float32 in (0, 1)
    class_ids: np.ndarray                 # [N] int64
    patch_ids: np.ndarray                 # [N] int64 in {0, ..., P-1}
    patch_rects: np.ndarray               # [N, 4] float32 patch region R_{p_i} in global coords
    boundary_dists: np.ndarray            # [N, 4] float32 (d_left, d_right, d_top, d_bottom) in [0, 1]
    is_boundary_risk: np.ndarray          # [N] float32 in {0.0, 1.0}
    node_features: np.ndarray             # [N, 12 + C_roi] float32 input feature vector
    roi_features: Optional[np.ndarray] = None  # [N, C_roi] optional appearance features
    # Ground Truth Supervision (Populated in Step 2)
    gt_boxes: Optional[np.ndarray] = None      # [M, 4] full GT boxes
    gt_classes: Optional[np.ndarray] = None    # [M] int64
    gt_id: Optional[np.ndarray] = None         # [N] int64 assigned GT index (-1 for background)
    is_leader: Optional[np.ndarray] = None     # [N] bool leader indicator
    target_reg: Optional[np.ndarray] = None    # [N, 4] regression targets T_i^reg
    target_score: Optional[np.ndarray] = None  # [N] rescoring targets T_i^score
    target_edge: Optional[np.ndarray] = None   # [N, N] stitching edge matrix Y_ij^edge


def generate_tile_grid(
    img_w: int,
    img_h: int,
    patch_w: int = 640,
    patch_h: int = 576,
    overlap_ratio: float = 0.0,
) -> np.ndarray:
    """
    Generate sliding-window tile coordinates [P, 4] in (x1, y1, x2, y2) global pixel format.
    """
    stride_x = max(1, int(round(patch_w * (1.0 - overlap_ratio))))
    stride_y = max(1, int(round(patch_h * (1.0 - overlap_ratio))))

    xs = list(range(0, max(1, img_w - patch_w + 1), stride_x))
    if len(xs) == 0 or (xs[-1] + patch_w < img_w):
        xs.append(max(0, img_w - patch_w))

    ys = list(range(0, max(1, img_h - patch_h + 1), stride_y))
    if len(ys) == 0 or (ys[-1] + patch_h < img_h):
        ys.append(max(0, img_h - patch_h))

    patches = []
    for y1 in ys:
        for x1 in xs:
            x2 = min(img_w, x1 + patch_w)
            y2 = min(img_h, y1 + patch_h)
            patches.append([float(x1), float(y1), float(x2), float(y2)])
    return np.array(patches, dtype=np.float32)


def compute_cutline_boundary_distances(
    boxes_global: np.ndarray,
    patch_rects: np.ndarray,
    img_size: Tuple[int, int],
    tau_d_pixels: float = 5.0,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Step 1 Core Prior: Compute normalized distances to the 4 patch cut-lines
    (d_left, d_right, d_top, d_bottom) and the high-risk truncation indicator I_bound.

    Formula:
        d_left   = x_{1,i}^{local} / W_patch
        d_right  = (W_patch - x_{2,i}^{local}) / W_patch
        d_top    = y_{1,i}^{local} / H_patch
        d_bottom = (H_patch - y_{2,i}^{local}) / H_patch

    Outer Physical Boundary Rule:
        If a patch edge coincides with the outer boundary of the full image
        (rather than an internal cut seam), force that direction's distance to 1.0
        so it cannot trigger cross-tile truncation.
    """
    if len(boxes_global) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    img_w, img_h = float(img_size[0]), float(img_size[1])

    px1, py1, px2, py2 = (
        patch_rects[:, 0],
        patch_rects[:, 1],
        patch_rects[:, 2],
        patch_rects[:, 3],
    )
    w_patch = np.maximum(eps, px2 - px1)
    h_patch = np.maximum(eps, py2 - py1)

    # Local coordinates inside the patch
    x1_local = np.clip(boxes_global[:, 0] - px1, 0.0, w_patch)
    x2_local = np.clip(boxes_global[:, 2] - px1, 0.0, w_patch)
    y1_local = np.clip(boxes_global[:, 1] - py1, 0.0, h_patch)
    y2_local = np.clip(boxes_global[:, 3] - py1, 0.0, h_patch)

    # Normalized distances in [0, 1]
    d_left = x1_local / w_patch
    d_right = (w_patch - x2_local) / w_patch
    d_top = y1_local / h_patch
    d_bottom = (h_patch - y2_local) / h_patch

    # Pixel distances for physical threshold comparison
    px_left = x1_local
    px_right = w_patch - x2_local
    px_top = y1_local
    px_bottom = h_patch - y2_local

    # Outer physical image boundary check: force to 1.0 (and pixel dist to +inf)
    is_outer_left = px1 <= 0.5
    is_outer_right = px2 >= (img_w - 0.5)
    is_outer_top = py1 <= 0.5
    is_outer_bottom = py2 >= (img_h - 0.5)

    d_left = np.where(is_outer_left, 1.0, d_left)
    d_right = np.where(is_outer_right, 1.0, d_right)
    d_top = np.where(is_outer_top, 1.0, d_top)
    d_bottom = np.where(is_outer_bottom, 1.0, d_bottom)

    px_left = np.where(is_outer_left, 1e6, px_left)
    px_right = np.where(is_outer_right, 1e6, px_right)
    px_top = np.where(is_outer_top, 1e6, px_top)
    px_bottom = np.where(is_outer_bottom, 1e6, px_bottom)

    boundary_dists = np.stack([d_left, d_right, d_top, d_bottom], axis=-1).astype(np.float32)

    # High-risk truncation flag I_bound = I(min(d_left, d_right, d_top, d_bottom) < tau_d)
    min_px_dist = np.minimum(np.minimum(px_left, px_right), np.minimum(px_top, px_bottom))
    tau_norm = tau_d_pixels / np.maximum(w_patch, h_patch)
    min_norm_dist = np.min(boundary_dists, axis=-1)
    is_boundary_risk = ((min_px_dist < tau_d_pixels) | (min_norm_dist < tau_norm)).astype(np.float32)

    return boundary_dists, is_boundary_risk


def build_candidate_feature_vector(
    boxes_global: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    patch_ids: np.ndarray,
    patch_rects: np.ndarray,
    img_size: Tuple[int, int],
    roi_features: Optional[np.ndarray] = None,
    tau_d_pixels: float = 5.0,
    coord_mode: str = "xyxy",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the Step 1 single-box input feature vector b_i of dimension D_in = 12 + C_roi:
        - [0:4]   Global geometric features (4D): normalized (x1, y1, x2, y2) or (xc, yc, w, h)
        - [4:6]   Base detector output info (2D): confidence s_i in (0, 1), class ID c_i
        - [6:7]   Patch ID (1D): patch_id in {0, ..., P-1} (used only for graph masking, excluded from Linear)
        - [7:11]  Cut-line boundary distance encoding (4D): (d_left, d_right, d_top, d_bottom)
        - [11:12] Boundary truncation high-risk flag (1D): I_bound in {0, 1}
        - [12:]   Optional lightweight RoI appearance features (C_roi = 64D)
    """
    n_boxes = len(boxes_global)
    c_roi = 0 if roi_features is None else roi_features.shape[1]
    if n_boxes == 0:
        return (
            np.zeros((0, 12 + c_roi), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    img_w, img_h = float(img_size[0]), float(img_size[1])
    x1 = boxes_global[:, 0] / max(1.0, img_w)
    y1 = boxes_global[:, 1] / max(1.0, img_h)
    x2 = boxes_global[:, 2] / max(1.0, img_w)
    y2 = boxes_global[:, 3] / max(1.0, img_h)

    if coord_mode == "xywh":
        w = x2 - x1
        h = y2 - y1
        xc = x1 + 0.5 * w
        yc = y1 + 0.5 * h
        geom_global = np.stack([xc, yc, w, h], axis=-1).astype(np.float32)
    else:
        geom_global = np.stack([x1, y1, x2, y2], axis=-1).astype(np.float32)

    det_info = np.stack([scores.astype(np.float32), class_ids.astype(np.float32)], axis=-1)
    patch_col = patch_ids.astype(np.float32)[:, None]

    boundary_dists, is_boundary_risk = compute_cutline_boundary_distances(
        boxes_global=boxes_global,
        patch_rects=patch_rects,
        img_size=img_size,
        tau_d_pixels=tau_d_pixels,
    )

    parts = [
        geom_global,                    # 4D [0:4]
        det_info,                       # 2D [4:6]
        patch_col,                      # 1D [6:7] (patch_id)
        boundary_dists,                 # 4D [7:11] (d_left, d_right, d_top, d_bottom)
        is_boundary_risk[:, None],      # 1D [11:12] (I_bound)
    ]
    if roi_features is not None:
        parts.append(roi_features.astype(np.float32))

    node_features = np.concatenate(parts, axis=-1).astype(np.float32)
    return node_features, boundary_dists, is_boundary_risk


# ============================================================================
# 3. Step 2: Cropped-GT Anti-Nested-Kill Label Assignment
# ============================================================================

class CroppedGTLabelAssigner:
    """
    Step 2: Cropped-GT Label Assignment Algorithm.
    Prevents false suppression/merging of nested same-class objects (e.g., "adult holding child")
    while accurately linking multi-tile fragments (2~4 tiles) of the same object.
    """

    def __init__(
        self,
        local_iou_thresh: float = 0.5,
        use_dynamic_iou_score: bool = False,
        matching_metric: str = "cropped_iou",  # "cropped_iou" (Ours) or "ios" (Ablation baseline)
    ):
        assert matching_metric in ("cropped_iou", "ios"), f"Unsupported metric: {matching_metric}"
        self.local_iou_thresh = local_iou_thresh
        self.use_dynamic_iou_score = use_dynamic_iou_score
        self.matching_metric = matching_metric

    def assign(
        self,
        boxes_global: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        patch_ids: np.ndarray,
        patch_rects: np.ndarray,
        gt_boxes: np.ndarray,
        gt_classes: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        n_preds = len(boxes_global)
        m_gts = len(gt_boxes)

        if gt_classes is None:
            gt_classes = np.zeros((m_gts,), dtype=np.int64)

        gt_id = np.full((n_preds,), -1, dtype=np.int64)
        is_leader = np.zeros((n_preds,), dtype=bool)
        target_reg = np.copy(boxes_global).astype(np.float32)
        target_score = np.zeros((n_preds,), dtype=np.float32)
        target_edge = np.zeros((n_preds, n_preds), dtype=np.float32)
        matched_local_ious = np.zeros((n_preds,), dtype=np.float32)

        if n_preds == 0 or m_gts == 0:
            return {
                "gt_id": gt_id,
                "is_leader": is_leader,
                "target_reg": target_reg,
                "target_score": target_score,
                "target_edge": target_edge,
                "local_ious": matched_local_ious,
            }

        # --------------------------------------------------------------------
        # Step 2.1: Compute Cropped-GT IoU to determine unique gt_id[i]
        # For candidate b_i from patch p_i: G_k^{(p_i)} = Intersect(G_k, R_{p_i})
        # --------------------------------------------------------------------
        cropped_gts, valid_crop = intersect_boxes_np(patch_rects, gt_boxes)  # [N, M, 4], [N, M]

        b_expanded = boxes_global[:, None, :]  # [N, 1, 4]
        lt = np.maximum(b_expanded[:, :, :2], cropped_gts[:, :, :2])
        rb = np.minimum(b_expanded[:, :, 2:], cropped_gts[:, :, 2:])
        wh = np.maximum(0.0, rb - lt)
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

        area_b = box_area_np(boxes_global)[:, None]  # [N, 1]
        crop_w = np.maximum(0.0, cropped_gts[:, :, 2] - cropped_gts[:, :, 0])
        crop_h = np.maximum(0.0, cropped_gts[:, :, 3] - cropped_gts[:, :, 1])
        area_crop = crop_w * crop_h  # [N, M]

        if self.matching_metric == "cropped_iou":
            # Standard IoU with Cropped-GT (Strictly NO IoS!)
            union = area_b + area_crop - inter
            sim_matrix = np.where(valid_crop & (union > 1e-7), inter / (union + 1e-7), 0.0)
        else:
            # Ablation baseline: IoS with Cropped-GT (when a small object is nested inside a larger
            # GT, IoS with both GTs is ~1.0; IoS-based assignment prefers the larger enclosing GT
            # and swallows the nested small object as a redundant fragment of the outer GT)
            # IoS / IoA over candidate box area: Area(b_i n G_k) / Area(b_i)
            # Both the adult box and the nested child box have IoS ~ 1.0 with the adult GT,
            # causing the child box to be assigned to the adult GT and suppressed as a non-leader!
            ios_mat = np.where(valid_crop & (area_b > 1e-7), inter / (area_b + 1e-7), 0.0)
            gt_area_bonus = 0.05 * (area_crop / (np.max(area_crop) + 1.0))
            sim_matrix = np.where(ios_mat >= self.local_iou_thresh, ios_mat + gt_area_bonus, ios_mat)

        class_match = (class_ids[:, None] == gt_classes[None, :])
        sim_matrix = np.where(class_match, sim_matrix, -1.0)

        best_gt_idx = np.argmax(sim_matrix, axis=1)             # [N]
        best_sim = sim_matrix[np.arange(n_preds), best_gt_idx]  # [N]

        pos_mask = best_sim >= self.local_iou_thresh
        gt_id[pos_mask] = best_gt_idx[pos_mask]
        matched_local_ious = np.clip(best_sim, 0.0, 1.0).astype(np.float32)

        # --------------------------------------------------------------------
        # Step 2.2: Build Same-Source Stitching Edge Matrix Y_{ij}^{edge}
        # Y_{ij}^{edge} = 1 iff:
        #   1) gt_id[i] == gt_id[j] and gt_id[i] != -1
        #   2) patch_id[i] != patch_id[j] (strictly across different patches)
        # --------------------------------------------------------------------
        same_gt = (gt_id[:, None] == gt_id[None, :]) & (gt_id[:, None] != -1)
        diff_patch = patch_ids[:, None] != patch_ids[None, :]
        target_edge = (same_gt & diff_patch).astype(np.float32)

        # --------------------------------------------------------------------
        # Step 2.3: Leader Selection & Target Assignment
        # For each GT G_k, S_k = {i | gt_id[i] == k}.
        # Quality score q_i = s_i * IoU(b_i, G_k)
        # --------------------------------------------------------------------
        global_iou_matrix = pairwise_iou_np(boxes_global, gt_boxes)  # [N, M]

        for k in range(m_gts):
            subset_indices = np.where(gt_id == k)[0]
            if len(subset_indices) == 0:
                continue

            ious_k = global_iou_matrix[subset_indices, k]
            quality_scores = scores[subset_indices] * (ious_k + 1e-4 * matched_local_ious[subset_indices])
            leader_local_idx = int(np.argmax(quality_scores))
            leader_idx = subset_indices[leader_local_idx]

            is_leader[leader_idx] = True
            target_reg[subset_indices] = gt_boxes[k].astype(np.float32)

            if self.use_dynamic_iou_score:
                target_score[leader_idx] = float(np.clip(ious_k[leader_local_idx], 0.5, 1.0))
            else:
                target_score[leader_idx] = 1.0

        return {
            "gt_id": gt_id,
            "is_leader": is_leader,
            "target_reg": target_reg,
            "target_score": target_score,
            "target_edge": target_edge,
            "local_ious": matched_local_ious,
        }


# ============================================================================
# 4. Offline Slice Exporter & Realistic High-Resolution Benchmark Generator
# ============================================================================

def collect_slice_predictions_from_detector(
    patch_predictions: List[Dict[str, np.ndarray]],
    patch_rects: np.ndarray,
    img_size: Tuple[int, int],
    conf_thresh: float = 0.1,
    intra_nms_iou: float = 0.7,
    tau_d_pixels: float = 5.0,
    gt_boxes: Optional[np.ndarray] = None,
    gt_classes: Optional[np.ndarray] = None,
    assigner: Optional[CroppedGTLabelAssigner] = None,
    image_id: str = "img_000",
) -> TileSample:
    all_boxes = []
    all_scores = []
    all_classes = []
    all_patch_ids = []
    all_patch_rects = []
    all_rois = []
    has_roi = False

    for p_idx, pred in enumerate(patch_predictions):
        p_rect = patch_rects[p_idx]  # [px1, py1, px2, py2]
        boxes_local = np.asarray(pred["boxes"], dtype=np.float32)
        scores_p = np.asarray(pred["scores"], dtype=np.float32)
        classes_p = np.asarray(pred.get("class_ids", np.zeros(len(boxes_local))), dtype=np.int64)
        roi_p = pred.get("roi_features", None)

        keep_conf = scores_p >= conf_thresh
        boxes_local = boxes_local[keep_conf]
        scores_p = scores_p[keep_conf]
        classes_p = classes_p[keep_conf]
        if roi_p is not None:
            roi_p = np.asarray(roi_p, dtype=np.float32)[keep_conf]
            has_roi = True

        if len(boxes_local) == 0:
            continue

        kept_indices = []
        for cls_val in np.unique(classes_p):
            cls_mask = np.where(classes_p == cls_val)[0]
            nms_keep = nms_np(boxes_local[cls_mask], scores_p[cls_mask], iou_thresh=intra_nms_iou)
            kept_indices.extend(cls_mask[nms_keep].tolist())
        kept_indices = np.array(sorted(kept_indices), dtype=np.int64)

        boxes_local = boxes_local[kept_indices]
        scores_p = scores_p[kept_indices]
        classes_p = classes_p[kept_indices]
        if roi_p is not None:
            roi_p = roi_p[kept_indices]

        boxes_glob = boxes_local.copy()
        boxes_glob[:, [0, 2]] += p_rect[0]
        boxes_glob[:, [1, 3]] += p_rect[1]

        all_boxes.append(boxes_glob)
        all_scores.append(scores_p)
        all_classes.append(classes_p)
        all_patch_ids.append(np.full((len(boxes_glob),), p_idx, dtype=np.int64))
        all_patch_rects.append(np.tile(p_rect[None, :], (len(boxes_glob), 1)))
        if roi_p is not None:
            all_rois.append(roi_p)

    if len(all_boxes) > 0:
        boxes_global = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        class_ids = np.concatenate(all_classes, axis=0)
        patch_ids = np.concatenate(all_patch_ids, axis=0)
        patch_rects_per_box = np.concatenate(all_patch_rects, axis=0)
        roi_features = np.concatenate(all_rois, axis=0) if (has_roi and len(all_rois) > 0) else None
    else:
        boxes_global = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        class_ids = np.zeros((0,), dtype=np.int64)
        patch_ids = np.zeros((0,), dtype=np.int64)
        patch_rects_per_box = np.zeros((0, 4), dtype=np.float32)
        roi_features = None

    node_features, boundary_dists, is_boundary_risk = build_candidate_feature_vector(
        boxes_global=boxes_global,
        scores=scores,
        class_ids=class_ids,
        patch_ids=patch_ids,
        patch_rects=patch_rects_per_box,
        img_size=img_size,
        roi_features=roi_features,
        tau_d_pixels=tau_d_pixels,
    )

    sample = TileSample(
        image_id=image_id,
        img_size=img_size,
        boxes_global=boxes_global,
        scores=scores,
        class_ids=class_ids,
        patch_ids=patch_ids,
        patch_rects=patch_rects_per_box,
        boundary_dists=boundary_dists,
        is_boundary_risk=is_boundary_risk,
        node_features=node_features,
        roi_features=roi_features,
        gt_boxes=gt_boxes,
        gt_classes=gt_classes,
    )

    if gt_boxes is not None:
        if assigner is None:
            assigner = CroppedGTLabelAssigner(local_iou_thresh=0.5)
        labels = assigner.assign(
            boxes_global=boxes_global,
            scores=scores,
            class_ids=class_ids,
            patch_ids=patch_ids,
            patch_rects=patch_rects_per_box,
            gt_boxes=gt_boxes,
            gt_classes=gt_classes,
        )
        sample.gt_id = labels["gt_id"]
        sample.is_leader = labels["is_leader"]
        sample.target_reg = labels["target_reg"]
        sample.target_score = labels["target_score"]
        sample.target_edge = labels["target_edge"]

    return sample


class SyntheticTiledDatasetGenerator:
    """
    Generates realistic high-resolution tile-and-merge detection scenarios (640x576 patches)
    covering all pain points and ablation test cases in the engineering plan:
      1. Cross-cutline truncation:
         - 50-50 splits across vertical/horizontal seams
         - Asymmetric splits (70-30 / 85-15) across seams
         - 3~4 Patch Cross-Intersection / Chain splits straddling the 4-tile corner
      2. Crowded adjacent distinct targets near cutlines (tests 1D Cut-Projection IoU_1D
         and Cutline Boundary Distance encoding so adjacent distinct targets across a seam
         are NOT falsely merged).
      3. Nested same-class objects ("adult holding child"):
         - Independent small target inside a larger same-class target (IoS = 1.0, Standard IoU < 0.25)
      4. Redundant overlapping boxes and seam-adjacent background false positives.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (1280, 1152),
        patch_size: Tuple[int, int] = (640, 576),
        overlap_ratio: float = 0.0,
        seed: int = 42,
    ):
        self.img_size = img_size
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.rng = np.random.default_rng(seed)
        self.patch_rects = generate_tile_grid(
            img_size[0], img_size[1], patch_size[0], patch_size[1], overlap_ratio
        )

    def generate_image_sample(
        self,
        image_id: str,
        assigner: Optional[CroppedGTLabelAssigner] = None,
    ) -> TileSample:
        vx = float(self.patch_size[0])  # 640.0 vertical cutline
        hy = float(self.patch_size[1])  # 576.0 horizontal cutline

        gt_boxes_list = []
        gt_classes_list = []
        oblique_chain_indices = set()
        intact_near_seam_indices = set()

        # 1. Three Oblique / Cross 3~4-Tile Targets straddling (vx=640, hy=576)
        # Leader is in Patch 0 (Top-Left), bridge fragments in Patch 1/2, diagonal fragment in Patch 3 (Bottom-Right).
        for c_idx, (ox, oy) in enumerate([(-6.0, -6.0), (4.0, -5.0), (-4.0, 5.0)]):
            bw = self.rng.uniform(58.0, 74.0)
            bh = self.rng.uniform(58.0, 74.0)
            cx = vx + ox
            cy = hy + oy
            oblique_chain_indices.add(len(gt_boxes_list))
            gt_boxes_list.append([cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5])
            gt_classes_list.append(0)

        # 2. Three 50-50 Split Targets straddling vertical (x=640) cutline, paired with an INTACT
        # nearby small target 6~10px from the seam (I_bound == 0, but <16px across seam so w/o Cutline Dist
        # confuses it with a truncated fragment!)
        for y_pos in [170.0, 360.0, 810.0]:
            bw = self.rng.uniform(48.0, 64.0)
            bh = self.rng.uniform(38.0, 52.0)
            cy = y_pos + self.rng.uniform(-12.0, 12.0)
            ratio_l = self.rng.uniform(0.46, 0.54)
            gt_boxes_list.append([vx - bw * ratio_l, cy - bh * 0.5, vx + bw * (1.0 - ratio_l), cy + bh * 0.5])
            gt_classes_list.append(0)

            # Intact small target 6.5~10px to the right of seam (not cut by seam -> d_left > tau_d -> I_bound = 0)
            nb_w = self.rng.uniform(26.0, 36.0)
            nb_h = self.rng.uniform(26.0, 36.0)
            nb_x1 = vx + self.rng.uniform(6.5, 9.5)
            nb_y1 = cy - bh * 0.5 + self.rng.uniform(-4.0, 4.0)
            intact_near_seam_indices.add(len(gt_boxes_list))
            gt_boxes_list.append([nb_x1, nb_y1, nb_x1 + nb_w, nb_y1 + nb_h])
            gt_classes_list.append(0)

        # 3. Three Asymmetric Split Targets ("One Large, One Small" 82-18 / 72-28 split) across horizontal cutline (y=576)
        for x_pos in [210.0, 430.0, 940.0]:
            bw = self.rng.uniform(44.0, 62.0)
            bh = self.rng.uniform(50.0, 68.0)
            cx = x_pos + self.rng.uniform(-18.0, 18.0)
            ratio_t = self.rng.uniform(0.68, 0.74)
            gt_boxes_list.append([cx - bw * 0.5, hy - bh * ratio_t, cx + bw * 0.5, hy + bh * (1.0 - ratio_t)])
            gt_classes_list.append(0)

        # 4. Three Nested Same-Class Target Pairs ("Adult Holding Child")
        nested_centers = [(280.0, 250.0), (960.0, 270.0), (310.0, 860.0)]
        for acx_base, acy_base in nested_centers:
            acx = acx_base + self.rng.uniform(-18.0, 18.0)
            acy = acy_base + self.rng.uniform(-18.0, 18.0)
            aw = self.rng.uniform(66.0, 86.0)
            ah = self.rng.uniform(78.0, 102.0)
            adult_box = [acx - aw * 0.5, acy - ah * 0.5, acx + aw * 0.5, acy + ah * 0.5]

            cw = aw * 0.40
            ch = ah * 0.38
            ccx = acx + self.rng.uniform(-aw * 0.08, aw * 0.08)
            ccy = acy + self.rng.uniform(-ah * 0.08, ah * 0.08)
            child_box = [ccx - cw * 0.5, ccy - ch * 0.5, ccx + cw * 0.5, ccy + ch * 0.5]

            gt_boxes_list.append(adult_box)
            gt_classes_list.append(0)
            gt_boxes_list.append(child_box)
            gt_classes_list.append(0)

        # 5. Two regular interior small targets
        for icx, icy in [(150.0, 440.0), (1080.0, 440.0)]:
            bw = self.rng.uniform(28.0, 44.0)
            bh = self.rng.uniform(28.0, 44.0)
            gt_boxes_list.append([icx - bw * 0.5, icy - bh * 0.5, icx + bw * 0.5, icy + bh * 0.5])
            gt_classes_list.append(0)

        gt_boxes = np.array(gt_boxes_list, dtype=np.float32)
        gt_classes = np.array(gt_classes_list, dtype=np.int64)

        patch_predictions = []
        for p_idx, p_rect in enumerate(self.patch_rects):
            px1, py1, px2, py2 = p_rect
            p_boxes = []
            p_scores = []
            p_classes = []

            cropped, valid = intersect_boxes_np(p_rect[None, :], gt_boxes)
            cropped = cropped[0]
            valid = valid[0]

            for k in range(len(gt_boxes)):
                if not valid[k]:
                    continue
                c_box = cropped[k].copy()

                # For Oblique 3~4-Tile Targets (k in oblique_chain_indices):
                # Simulate realistic oblique appearance across tiles:
                # - Patch 0 (Top-Left): upper-left torso [x1, y1, vx, hy - 8]
                # - Patch 1 (Top-Right bridge): central bridge [vx, y1 + 8, vx + 16, hy]
                # - Patch 3 (Bottom-Right extremity): lower-right legs [vx + 6, hy, x2, y2]
                # Notice Patch 0 and Patch 3 do NOT overlap in 1D projection (IoU_1D = 0) and only connect
                # via 2-hop message passing through the bridge fragment in Patch 1!
                if k in oblique_chain_indices:
                    gx1, gy1, gx2, gy2 = gt_boxes[k]
                    if p_idx == 0:
                        c_box = np.array([gx1, gy1, vx, hy - 5.0], dtype=np.float32)
                    elif p_idx == 1:
                        c_box = np.array([vx, gy1 + 6.0, min(gx2, vx + 16.0), hy], dtype=np.float32)
                    elif p_idx == 2:
                        continue  # Oblique target passes through Patch 0 -> Patch 1 -> Patch 3
                    elif p_idx == 3:
                        c_box = np.array([vx + 5.0, hy, gx2, gy2], dtype=np.float32)

                cw = c_box[2] - c_box[0]
                ch = c_box[3] - c_box[1]
                if cw < 3.5 or ch < 3.5:
                    continue

                full_area = max(1.0, (gt_boxes[k, 2] - gt_boxes[k, 0]) * (gt_boxes[k, 3] - gt_boxes[k, 1]))
                vis_ratio = np.clip((cw * ch) / full_area, 0.08, 1.0)

                lx1 = (c_box[0] - px1) + self.rng.normal(0.0, 0.5)
                ly1 = (c_box[1] - py1) + self.rng.normal(0.0, 0.5)
                lx2 = (c_box[2] - px1) + self.rng.normal(0.0, 0.5)
                ly2 = (c_box[3] - py1) + self.rng.normal(0.0, 0.5)

                if k in intact_near_seam_indices:
                    # Keep intact object >= 6.5px from seam so I_bound == 0
                    lx1 = max(6.5, lx1)
                else:
                    if c_box[0] <= px1 + 0.5:
                        lx1 = max(0.0, self.rng.uniform(0.0, 0.8))
                    if c_box[1] <= py1 + 0.5:
                        ly1 = max(0.0, self.rng.uniform(0.0, 0.8))
                    if c_box[2] >= px2 - 0.5:
                        lx2 = min(px2 - px1, (px2 - px1) - self.rng.uniform(0.0, 0.8))
                    if c_box[3] >= py2 - 0.5:
                        ly2 = min(py2 - py1, (py2 - py1) - self.rng.uniform(0.0, 0.8))

                if lx2 - lx1 < 3.0 or ly2 - ly1 < 3.0:
                    continue

                if k in oblique_chain_indices:
                    base_score = 0.93 if p_idx == 0 else (0.76 if p_idx == 1 else 0.72)
                else:
                    base_score = float(np.clip(0.44 + 0.50 * (vis_ratio ** 0.40) + self.rng.normal(0.0, 0.02), 0.18, 0.96))

                p_boxes.append([lx1, ly1, lx2, ly2])
                p_scores.append(base_score)
                p_classes.append(gt_classes[k])

                if vis_ratio > 0.45 and self.rng.random() < 0.35:
                    dup_box = [
                        max(0.0, lx1 + self.rng.normal(0.0, 1.8)),
                        max(0.0, ly1 + self.rng.normal(0.0, 1.8)),
                        min(px2 - px1, lx2 + self.rng.normal(0.0, 1.8)),
                        min(py2 - py1, ly2 + self.rng.normal(0.0, 1.8)),
                    ]
                    if dup_box[2] - dup_box[0] > 4.0 and dup_box[3] - dup_box[1] > 4.0:
                        p_boxes.append(dup_box)
                        p_scores.append(float(np.clip(base_score * self.rng.uniform(0.42, 0.65), 0.12, 0.70)))
                        p_classes.append(gt_classes[k])

            # Background false positive box
            fw = self.rng.uniform(18.0, 34.0)
            fh = self.rng.uniform(18.0, 34.0)
            fx1 = self.rng.uniform(25.0, max(30.0, (px2 - px1) - fw - 25.0))
            fy1 = self.rng.uniform(25.0, max(30.0, (py2 - py1) - fh - 25.0))
            p_boxes.append([fx1, fy1, fx1 + fw, fy1 + fh])
            p_scores.append(float(self.rng.uniform(0.11, 0.24)))
            p_classes.append(0)

            patch_predictions.append({
                "boxes": np.array(p_boxes, dtype=np.float32) if p_boxes else np.zeros((0, 4), dtype=np.float32),
                "scores": np.array(p_scores, dtype=np.float32) if p_scores else np.zeros((0,), dtype=np.float32),
                "class_ids": np.array(p_classes, dtype=np.int64) if p_classes else np.zeros((0,), dtype=np.int64),
            })

        return collect_slice_predictions_from_detector(
            patch_predictions=patch_predictions,
            patch_rects=self.patch_rects,
            img_size=self.img_size,
            conf_thresh=0.1,
            intra_nms_iou=0.7,
            tau_d_pixels=5.0,
            gt_boxes=gt_boxes,
            gt_classes=gt_classes,
            assigner=assigner,
            image_id=image_id,
        )

    def generate_dataset(
        self,
        num_samples: int = 40,
        assigner: Optional[CroppedGTLabelAssigner] = None,
        save_npz_path: Optional[str] = None,
    ) -> List[TileSample]:
        samples = [
            self.generate_image_sample(f"sample_{idx:04d}", assigner=assigner)
            for idx in range(num_samples)
        ]
        if save_npz_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_npz_path)), exist_ok=True)
            serialized = {}
            for i, s in enumerate(samples):
                serialized[f"{i}_boxes_global"] = s.boxes_global
                serialized[f"{i}_scores"] = s.scores
                serialized[f"{i}_class_ids"] = s.class_ids
                serialized[f"{i}_patch_ids"] = s.patch_ids
                serialized[f"{i}_node_features"] = s.node_features
                serialized[f"{i}_gt_boxes"] = s.gt_boxes
                serialized[f"{i}_target_reg"] = s.target_reg
                serialized[f"{i}_target_score"] = s.target_score
                serialized[f"{i}_target_edge"] = s.target_edge
            np.savez_compressed(save_npz_path, **serialized)
        return samples


class OfflineTileDataset(Dataset):
    """PyTorch Dataset wrapper for offline exported TileSample instances."""

    def __init__(self, samples: List[TileSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[TileSample, Dict[str, "torch.Tensor"]]:
        s = self.samples[idx]
        if not TORCH_AVAILABLE:
            return s
        item = {
            "image_id": s.image_id,
            "img_size": torch.tensor(s.img_size, dtype=torch.float32),
            "boxes_global": torch.from_numpy(s.boxes_global).float(),
            "scores": torch.from_numpy(s.scores).float(),
            "class_ids": torch.from_numpy(s.class_ids).long(),
            "patch_ids": torch.from_numpy(s.patch_ids).long(),
            "boundary_dists": torch.from_numpy(s.boundary_dists).float(),
            "is_boundary_risk": torch.from_numpy(s.is_boundary_risk).float(),
            "node_features": torch.from_numpy(s.node_features).float(),
        }
        if s.gt_boxes is not None:
            item["gt_boxes"] = torch.from_numpy(s.gt_boxes).float()
            item["gt_id"] = torch.from_numpy(s.gt_id).long()
            item["is_leader"] = torch.from_numpy(s.is_leader).bool()
            item["target_reg"] = torch.from_numpy(s.target_reg).float()
            item["target_score"] = torch.from_numpy(s.target_score).float()
            item["target_edge"] = torch.from_numpy(s.target_edge).float()
        return item