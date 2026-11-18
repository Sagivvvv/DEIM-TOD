"""
Step 4.1: Multi-Task Loss Functions for TileStitchingNet
  L_total = L_rescore + lambda_stitch * L_stitch + lambda_edge * L_edge
  Recommended hyperparameters: lambda_stitch = 2.0, lambda_edge = 1.0

Includes:
  - L_rescore: Focal Loss over all N candidate boxes (replaces NMS)
  - L_stitch:  CIoU Loss + Normalized L1 Loss on positive Leader boxes (T_i^score == 1)
  - L_edge:    Binary Cross-Entropy / Edge Focal Loss on active M_stitch edges

Module: tile_stitching_postprocess/losses.py
"""

from __future__ import annotations

import math
from typing import Dict, Optional

try:
    import torch
    import torch.nn as nn
    Tensor = torch.Tensor
except ImportError:
    try:
        from .model import Tensor, nn, torch
    except ImportError:
        from model import Tensor, nn, torch


def sigmoid_focal_loss(
    probs: "torch.Tensor",
    targets: "torch.Tensor",
    alpha: float = 0.25,
    gamma: float = 2.0,
    eps: float = 1e-6,
) -> "torch.Tensor":
    """
    Focal Loss on probabilities `probs` in (0, 1) and `targets` in [0, 1].
    Supports both binary labels {0, 1} and continuous dynamic IoU targets [0, 1].
    """
    if probs.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32)

    p = torch.clamp(probs, min=eps, max=1.0 - eps)
    t = targets.float()

    # Standard binary / continuous cross-entropy terms
    ce_pos = -torch.log(p)
    ce_neg = -torch.log(1.0 - p)

    # Focal modulation weight |t - p|^gamma
    mod_pos = (1.0 - p) ** gamma
    mod_neg = p ** gamma

    loss_pos = alpha * mod_pos * ce_pos * t
    loss_neg = (1.0 - alpha) * mod_neg * ce_neg * (1.0 - t)
    loss = loss_pos + loss_neg

    num_pos = float((t > 0.5).sum().item())
    norm_factor = max(1.0, num_pos)
    return loss.sum() / norm_factor


def quality_focal_loss(
    logits: "torch.Tensor",
    targets: "torch.Tensor",
    beta: float = 2.0,
) -> "torch.Tensor":
    """
    ④ Quality Focal Loss (Generalized Focal Loss, NeurIPS'20) on raw logits.
        QFL(σ, t) = |t - σ|^β * BCEWithLogits(logits, t),  σ = sigmoid(logits)
    - 极小点在 σ = t，故 rescore 输出可被解释为 IoU 质量分（修复原 sigmoid_focal_loss
      “双 focal 项加权和”极小点≠目标的问题：t=0.8 时原损失最低点在 p≈0.527）。
    - BCEWithLogits 形式数值稳定，规避原手写 log(p)/log(1-p) 在 FP16 下溢致 NaN。
    正样本(含软标签 t>0)归一化。
    """
    if logits.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32)

    t = targets.float()
    sigma = torch.sigmoid(logits)
    # 数值稳定 BCE from logits: max(x,0) - x*t + log(1+exp(-|x|))
    bce = (
        torch.clamp(logits, min=0.0)
        - logits * t
        + torch.log(1.0 + torch.exp(-torch.abs(logits)))
    )
    qfl = (torch.abs(t - sigma) ** beta) * bce

    num_pos = float((t > 0.0).sum().item())
    norm_factor = max(1.0, num_pos)
    return qfl.sum() / norm_factor


def ciou_and_l1_loss(
    pred_boxes: "torch.Tensor",
    target_boxes: "torch.Tensor",
    img_scale: float = 1280.0,
    eps: float = 1e-6,
) -> Dict[str, "torch.Tensor"]:
    """
    Compute Complete IoU (CIoU) Loss + normalized L1 Loss between predicted full boxes B_i
    and un-cropped ground-truth boxes T_i^reg for positive Leader samples.

    Args:
        pred_boxes:   [P, 4] predicted boxes (x1, y1, x2, y2)
        target_boxes: [P, 4] ground-truth full boxes (x1, y1, x2, y2)
        img_scale:    normalization factor for L1 coordinate loss
    """
    if pred_boxes.shape[0] == 0:
        zero = torch.tensor(0.0, dtype=torch.float32)
        return {"loss_ciou": zero, "loss_l1": zero, "loss_stitch": zero, "mean_iou": zero}

    # Ensure valid box ordering x2 > x1, y2 > y1
    px1 = pred_boxes[:, 0]
    py1 = pred_boxes[:, 1]
    px2 = torch.maximum(px1 + 1.0, pred_boxes[:, 2])
    py2 = torch.maximum(py1 + 1.0, pred_boxes[:, 3])

    gx1 = target_boxes[:, 0]
    gy1 = target_boxes[:, 1]
    gx2 = torch.maximum(gx1 + 1.0, target_boxes[:, 2])
    gy2 = torch.maximum(gy1 + 1.0, target_boxes[:, 3])

    pw = torch.clamp(px2 - px1, min=eps)
    ph = torch.clamp(py2 - py1, min=eps)
    pxc = px1 + 0.5 * pw
    pyc = py1 + 0.5 * ph

    gw = torch.clamp(gx2 - gx1, min=eps)
    gh = torch.clamp(gy2 - gy1, min=eps)
    gxc = gx1 + 0.5 * gw
    gyc = gy1 + 0.5 * gh

    # Intersection
    ix1 = torch.maximum(px1, gx1)
    iy1 = torch.maximum(py1, gy1)
    ix2 = torch.minimum(px2, gx2)
    iy2 = torch.minimum(py2, gy2)
    iw = torch.clamp(ix2 - ix1, min=0.0)
    ih = torch.clamp(iy2 - iy1, min=0.0)
    inter = iw * ih

    union = pw * ph + gw * gh - inter + eps
    iou = inter / union

    # Smallest enclosing box diagonal c^2
    cx1 = torch.minimum(px1, gx1)
    cy1 = torch.minimum(py1, gy1)
    cx2 = torch.maximum(px2, gx2)
    cy2 = torch.maximum(py2, gy2)
    cw = torch.clamp(cx2 - cx1, min=eps)
    ch = torch.clamp(cy2 - cy1, min=eps)
    c2 = cw * cw + ch * ch + eps

    # Center distance squared rho^2
    rho2 = (pxc - gxc) ** 2 + (pyc - gyc) ** 2

    # Aspect ratio consistency term v and trade-off parameter alpha (detached per CIoU paper)
    atan_diff = torch.atan(gw / gh) - torch.atan(pw / ph)
    v = (4.0 / (math.pi ** 2)) * (atan_diff ** 2)
    v_det = v.detach()
    iou_det = iou.detach()
    alpha = v_det / (1.0 - iou_det + v_det + eps)

    ciou = iou - (rho2 / c2) - alpha * v
    loss_ciou = (1.0 - ciou).mean()

    # Normalized L1 loss (normalized by GT box size so small and large targets get balanced gradients)
    norm_wh = torch.stack([gw, gh, gw, gh], dim=-1) + 1.0
    l1_diff = torch.abs(pred_boxes - target_boxes) / norm_wh
    loss_l1 = l1_diff.mean()

    loss_stitch = loss_ciou + 0.5 * loss_l1
    return {
        "loss_ciou": loss_ciou,
        "loss_l1": loss_l1,
        "loss_stitch": loss_stitch,
        "mean_iou": iou.detach().mean(),
    }


def edge_affinity_bce_loss(
    edge_probs: "torch.Tensor",
    target_edge: "torch.Tensor",
    m_stitch: "torch.Tensor",
    pos_weight: float = 2.5,
    eps: float = 1e-6,
) -> "torch.Tensor":
    """
    Compute Binary Cross-Entropy Loss on the active cross-patch stitching candidate set:
        E_stitch = { (i, j) | M_stitch[i, j] == 1 }
    """
    mask_f = m_stitch.float()
    num_active = float(mask_f.sum().item())
    if num_active < 1.0:
        return torch.tensor(0.0, dtype=torch.float32)

    p = torch.clamp(edge_probs, min=eps, max=1.0 - eps)
    y = target_edge.float()

    bce = -(pos_weight * y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))
    masked_bce = bce * mask_f
    return masked_bce.sum() / num_active


class TileStitchingLoss(nn.Module):
    """
    Step 4.1 Total Multi-Task Loss:
        L_total = L_rescore + lambda_stitch * L_stitch + lambda_edge * L_edge
    Default hyperparameters:
        lambda_stitch = 2.0
        lambda_edge   = 1.0
    """

    def __init__(
        self,
        lambda_stitch: float = 2.0,
        lambda_edge: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.lambda_stitch = lambda_stitch
        self.lambda_edge = lambda_edge
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self,
        model_outputs: Dict[str, "torch.Tensor"],
        target_score: "torch.Tensor",
        target_reg: "torch.Tensor",
        target_edge: "torch.Tensor",
        is_leader: Optional["torch.Tensor"] = None,
    ) -> Dict[str, "torch.Tensor"]:
        """
        Compute L_total and individual loss components.
        Also supervises the intermediate Soft Bounding Envelope (via L_edge and L_stitch)
        so gradients flow through both the affinity head and the box refinement head.
        """
        pred_scores = model_outputs["pred_scores"]
        pred_boxes = model_outputs["pred_boxes"]
        env_boxes = model_outputs["env_boxes"]
        edge_probs = model_outputs["edge_probs"]
        m_stitch = model_outputs["m_stitch"]
        # ④ rescore 改用 logits 版 QFL：优先取模型已输出的 rescore_logits；
        # 若缺失(旧路径/空框兜底)则由 pred_scores 反推 logits，保证兼容。
        rescore_logits = model_outputs.get("rescore_logits", None)
        if rescore_logits is None:
            p_clamp = torch.clamp(pred_scores, min=1e-6, max=1.0 - 1e-6)
            rescore_logits = torch.log(p_clamp / (1.0 - p_clamp))

        # 1. L_rescore: Quality Focal Loss over all N candidate boxes
        #    (logits 版, 数值稳定, 极小点=目标 -> 输出可解释为 IoU 质量分)
        loss_rescore = quality_focal_loss(
            logits=rescore_logits,
            targets=target_score,
            beta=self.focal_gamma,
        )

        # 2. L_stitch: CIoU + L1 Loss on positive Leader boxes (T_i^score == 1)
        if is_leader is None:
            leader_mask = target_score > 0.5
        else:
            leader_mask = is_leader.bool()

        num_leaders = int(leader_mask.sum().item()) if leader_mask.numel() > 0 else 0
        if num_leaders > 0:
            leader_pred_boxes = pred_boxes[leader_mask]
            leader_env_boxes = env_boxes[leader_mask]
            leader_target_reg = target_reg[leader_mask]

            stitch_metrics = ciou_and_l1_loss(leader_pred_boxes, leader_target_reg)
            env_metrics = ciou_and_l1_loss(leader_env_boxes, leader_target_reg)
            # Combine final refined box loss with auxiliary envelope alignment loss
            loss_stitch = stitch_metrics["loss_stitch"] + 0.3 * env_metrics["loss_stitch"]
            mean_leader_iou = stitch_metrics["mean_iou"]
        else:
            loss_stitch = torch.tensor(0.0, dtype=torch.float32)
            mean_leader_iou = torch.tensor(0.0, dtype=torch.float32)

        # 3. L_edge: BCE Loss on active cross-patch stitching edges M_stitch == 1
        loss_edge = edge_affinity_bce_loss(
            edge_probs=edge_probs,
            target_edge=target_edge,
            m_stitch=m_stitch,
        )

        loss_total = (
            loss_rescore
            + self.lambda_stitch * loss_stitch
            + self.lambda_edge * loss_edge
        )

        return {
            "loss_total": loss_total,
            "loss_rescore": loss_rescore,
            "loss_stitch": loss_stitch,
            "loss_edge": loss_edge,
            "mean_leader_iou": mean_leader_iou,
        }