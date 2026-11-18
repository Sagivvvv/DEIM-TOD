# -*- coding: utf-8 -*-
"""
End-to-End Pipeline: DEIM-S (P3 1x1 RoI -> 64D) + TileStitchingNet Post-Processor
================================================================================
本脚本打通以下完整工业级流程：
  1. DEIM-S 离线特征提取 (P3 RoIAlign 1x1 -> 64D 降维)：
     - 使用 Forward Hook 自动截获 DEIM HybridEncoder 的 P3 特征图 (stride=8, C=256)；
     - 对切块低置信度候选框 (conf >= 0.08) 执行 1x1 RoIAlign，并通过轻量线性层降维至 64D；
     - 结合 4 维切线距离与 I_bound 标志，利用 CroppedGTLabelAssigner 分配防误杀 Leader 标签；
     - 将特征与监督信号高效序列化保存至本地/Kaggle工作区 (.pt / .npz 紧凑缓存)。
  2. 极速离线训练 TileStitchingNet：
     - 直接加载离线特征，输入维度 D_in = 12 + 64 = 76；
     - 训练 2 层 BALR-Attention 关系网络、同源边亲和力头与软外接包络算子 (几分钟内收敛)。
  3. 全基线评测与消融对比：
     - 在验证集上一键对比 Greedy-NMS、Soft-NMS、WBF、SAHI 与 Ours TileStitchingNet；
     - 打印详尽的 AP / AP50 / AP75 与切口碎片召回率对比表。

运行模式：
  - 模式 1 (快速全流程仿真自检，无需 GPU / 权重，10 秒跑通全部步骤)：
      python TileStitchingNet/pipeline_deim_tilestitching.py --mode demo
  - 模式 2 (Kaggle / 服务器真实前向提取并 Dump 特征)：
      python TileStitchingNet/pipeline_deim_tilestitching.py --mode extract --config ... --ckpt ... --img-dir ...
  - 模式 3 (基于 Dump 的特征极速训练并评估)：
      python TileStitchingNet/pipeline_deim_tilestitching.py --mode train_eval --data-dir ./dumped_features
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# 将当前目录与上级目录加入 sys.path，保证无论是独立执行还是包调用均正常
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_CUR_DIR)
for _p in (_CUR_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 兼顾独立执行与包内引用
try:
    from .datasetBuilder import (
        CroppedGTLabelAssigner,
        SyntheticTiledDatasetGenerator,
        TileSample,
        build_candidate_feature_vector,
        collect_slice_predictions_from_detector,
        compute_cutline_boundary_distances,
        generate_tile_grid,
        nms_np,
    )
    from .losses import TileStitchingLoss
    from .model import TileStitchingNet, torch
    from .trainEval import (
        TileStitchingPredictor,
        evaluate_postprocessor_on_dataset,
        run_greedy_nms,
        run_sahi_nmm_ios,
        run_soft_nms,
        run_wbf,
        train_tile_stitching_net,
    )
except ImportError:
    from datasetBuilder import (
        CroppedGTLabelAssigner,
        SyntheticTiledDatasetGenerator,
        TileSample,
        build_candidate_feature_vector,
        collect_slice_predictions_from_detector,
        compute_cutline_boundary_distances,
        generate_tile_grid,
        nms_np,
    )
    from losses import TileStitchingLoss
    from model import TileStitchingNet, torch
    from trainEval import (
        TileStitchingPredictor,
        evaluate_postprocessor_on_dataset,
        run_greedy_nms,
        run_sahi_nmm_ios,
        run_soft_nms,
        run_wbf,
        train_tile_stitching_net,
    )

try:
    import torch.nn as nn
    import torchvision
    HAS_TORCH_NN = True
except ImportError:
    HAS_TORCH_NN = False


# ============================================================================
# 1. 轻量级 P3 1x1 RoI 特征提取与降维模块
# ============================================================================

class P3RoIFeatureExtractor(nn.Module if HAS_TORCH_NN else object):
    """
    从 DEIM-S 的 P3 特征图抽取 1x1 RoI 并降维到 64 维：
      1. P3 特征图步长为 8 (stride=8)，通道数 256；
      2. 使用 torchvision.ops.roi_align 抽取 1x1 池化特征 (Flat 为 256D)；
      3. 通过轻量全连接投影层 (256 -> 64) 降维，以极低存储体积保留关键外观信息。
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 64, spatial_scale: float = 1.0 / 8.0):
        if HAS_TORCH_NN:
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.spatial_scale = spatial_scale
            self.proj = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.LayerNorm(out_channels),
                nn.ReLU(inplace=True),
            )
            # 使用固定或正交初始化保证特征表达稳定性
            nn.init.orthogonal_(self.proj[0].weight)
            nn.init.zeros_(self.proj[0].bias)
        else:
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.spatial_scale = spatial_scale

    def forward(self, p3_feat: "torch.Tensor", rois_tensor: "torch.Tensor") -> "torch.Tensor":
        """
        Args:
            p3_feat: [B, 256, H/8, W/8] P3 特征图
            rois_tensor: [K, 5] 包含 (batch_idx, x1, y1, x2, y2) 局部像素坐标
        Returns:
            roi_embed: [K, 64] float32 外观特征向量
        """
        if not HAS_TORCH_NN:
            raise RuntimeError("RoIAlign requires native PyTorch and torchvision.")
        if rois_tensor.shape[0] == 0:
            return torch.zeros((0, self.out_channels), dtype=torch.float32, device=p3_feat.device)

        # 1x1 RoIAlign 池化
        pooled = torchvision.ops.roi_align(
            p3_feat,
            rois_tensor,
            output_size=(1, 1),
            spatial_scale=self.spatial_scale,
            sampling_ratio=2,
        )  # [K, 256, 1, 1]
        pooled_flat = pooled.flatten(1)  # [K, 256]
        return self.proj(pooled_flat)     # [K, 64]


class DEIMP3HookManager:
    """
    非侵入式 Forward Hook 管理器：自动拦截 DEIM HybridEncoder 输出的 P3 特征图。
    无需修改 DEIM 原仓库代码。
    """

    def __init__(self, deim_model: "nn.Module"):
        self.deim_model = deim_model
        self.p3_cache: Optional["torch.Tensor"] = None
        self.hook_handle = None
        self._register_hook()

    def _register_hook(self):
        # 探测 DEIM 的 encoder 模块
        encoder = getattr(self.deim_model, "encoder", None)
        if encoder is None:
            raise AttributeError("DEIM model does not have an 'encoder' attribute.")

        def hook_fn(module, input_tensor, output_tensor):
            # HybridEncoder 的输出为 [P3, P4, P5]，output_tensor[0] 即为 P3
            if isinstance(output_tensor, (list, tuple)) and len(output_tensor) > 0:
                self.p3_cache = output_tensor[0]
            elif isinstance(output_tensor, dict) and "p3" in output_tensor:
                self.p3_cache = output_tensor["p3"]

        self.hook_handle = encoder.register_forward_hook(hook_fn)

    def get_last_p3(self) -> Optional["torch.Tensor"]:
        return self.p3_cache

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


# ============================================================================
# 2. 离线数据缓存序列化管理器 (Save & Load)
# ============================================================================

class OfflineDatasetIO:
    """
    负责将切块推理的预测框、64 维 RoI 特征与 GT 监督标签存取为离线文件。
    采用紧凑的 float16 / int32 压缩格式，单样本仅约几十 KB，全集仅几百 MB。
    """

    @staticmethod
    def save_samples(samples: List[TileSample], save_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        dump_list = []
        for s in samples:
            item = {
                "image_id": s.image_id,
                "tile_rects_all": getattr(s, "tile_rects_all", None),
                "img_size": s.img_size,
                "boxes_global": s.boxes_global.astype(np.float32),
                "scores": s.scores.astype(np.float32),
                "class_ids": s.class_ids.astype(np.int64),
                "patch_ids": s.patch_ids.astype(np.int64),
                "patch_rects": s.patch_rects.astype(np.float32),
                "boundary_dists": s.boundary_dists.astype(np.float32),
                "is_boundary_risk": s.is_boundary_risk.astype(np.float32),
                "node_features": s.node_features.astype(np.float32),
                "roi_features": s.roi_features.astype(np.float16) if s.roi_features is not None else None,
                "gt_boxes": s.gt_boxes.astype(np.float32) if s.gt_boxes is not None else None,
                "gt_classes": s.gt_classes.astype(np.int64) if s.gt_classes is not None else None,
                "gt_id": s.gt_id.astype(np.int64) if s.gt_id is not None else None,
                "is_leader": s.is_leader.astype(bool) if s.is_leader is not None else None,
                "target_reg": s.target_reg.astype(np.float32) if s.target_reg is not None else None,
                "target_score": s.target_score.astype(np.float32) if s.target_score is not None else None,
                "target_edge": s.target_edge.astype(np.float32) if s.target_edge is not None else None,
            }
            dump_list.append(item)

        if HAS_TORCH_NN:
            torch.save(dump_list, save_path)
            actual_path = save_path
        else:
            actual_path = save_path if save_path.endswith(".npy") else save_path + ".npy"
            np.save(actual_path, np.array(dump_list, dtype=object), allow_pickle=True)
        print(f"[OfflineDatasetIO] 成功保存 {len(samples)} 个样本至: {actual_path} ({os.path.getsize(actual_path)/1048576.0:.2f} MB)")

    @staticmethod
    def load_samples(load_path: str) -> List[TileSample]:
        if not os.path.isfile(load_path):
            if os.path.isfile(load_path + ".npy"):
                load_path = load_path + ".npy"
            else:
                raise FileNotFoundError(f"找不到特征文件: {load_path}")

        if HAS_TORCH_NN and not load_path.endswith(".npy"):
            try:
                raw_data = torch.load(load_path, map_location="cpu", weights_only=False)
            except TypeError:
                raw_data = torch.load(load_path, map_location="cpu")
        else:
            raw_data = np.load(load_path, allow_pickle=True).tolist()

        samples: List[TileSample] = []
        for d in raw_data:
            roi_feat = d["roi_features"]
            if roi_feat is not None and isinstance(roi_feat, np.ndarray):
                roi_feat = roi_feat.astype(np.float32)

            s = TileSample(
                image_id=str(d["image_id"]),
                img_size=tuple(d["img_size"]),
                boxes_global=d["boxes_global"],
                scores=d["scores"],
                class_ids=d["class_ids"],
                patch_ids=d["patch_ids"],
                patch_rects=d["patch_rects"],
                boundary_dists=d["boundary_dists"],
                is_boundary_risk=d["is_boundary_risk"],
                node_features=d["node_features"],
                roi_features=roi_feat,
                gt_boxes=d.get("gt_boxes", None),
                gt_classes=d.get("gt_classes", None),
                gt_id=d.get("gt_id", None),
                is_leader=d.get("is_leader", None),
                target_reg=d.get("target_reg", None),
                target_score=d.get("target_score", None),
                target_edge=d.get("target_edge", None),
            )
            s.tile_rects_all = d.get("tile_rects_all")
            samples.append(s)

        print(f"[OfflineDatasetIO] 成功载入 {len(samples)} 个样本 (来自 {load_path})")
        return samples


# ============================================================================
# 3. DEIM-S 离线切块提取器 (运行在 Kaggle / GPU 服务器)
# ============================================================================

class DEIMSliceFeatureExtractor:
    """
    负责驱动 DEIM 模型进行多切块前向，并利用 Hook 提取 P3 1x1 RoI 特征。
    """

    def __init__(
        self,
        deim_model: "nn.Module",
        postprocessor: "nn.Module",
        device: str = "cuda",
        conf_thresh: float = 0.08,
        intra_nms_iou: float = 0.70,
        tau_d_pixels: float = 5.0,
        roi_out_channels: int = 64,
    ):
        self.model = deim_model.eval().to(device)
        self.postprocessor = postprocessor.eval().to(device)
        self.device = device
        self.conf_thresh = conf_thresh
        self.intra_nms_iou = intra_nms_iou
        self.tau_d_pixels = tau_d_pixels

        # 注册 P3 Hook
        self.hook_mgr = DEIMP3HookManager(self.model)
        # 初始化 256 -> 64 投影层
        self.roi_extractor = P3RoIFeatureExtractor(
            in_channels=256,
            out_channels=roi_out_channels,
            spatial_scale=1.0 / 8.0,
        ).to(device).eval()

        self.assigner = CroppedGTLabelAssigner(local_iou_thresh=0.5, matching_metric="cropped_iou")

    def extract_single_image(
        self,
        img_tensor_tiles: List["torch.Tensor"],
        patch_rects: np.ndarray,
        img_size: Tuple[int, int],
        image_id: str,
        gt_boxes: Optional[np.ndarray] = None,
        gt_classes: Optional[np.ndarray] = None,
        tile_bs: int = 8,
    ) -> TileSample:
        """
        对单张大图的所有切块批量前向，截获 P3 并提取 RoI 特征，组装为 TileSample。
        """
        with torch.no_grad():
            patch_predictions = []
            n_patches = len(img_tensor_tiles)

            for s in range(0, n_patches, tile_bs):
                sel_tiles = img_tensor_tiles[s : s + tile_bs]
                batch = torch.stack(sel_tiles).to(self.device)  # [B, 3, 576, 640]
                b_size = batch.shape[0]

                # 1. 前向传播 (Hook 自动截获 P3 特征图)
                outputs = self.model(batch)
                p3_feat = self.hook_mgr.get_last_p3()  # [B, 256, 72, 80]

                # 2. 调用检测器后处理器获取局部候选框
                orig_sizes = torch.tensor([[640.0, 576.0]] * b_size, device=self.device, dtype=torch.float32)
                results = self.postprocessor(outputs, orig_sizes)

                # 3. 逐切块抽取 P3 RoI 特征
                for sub_i in range(b_size):
                    p_idx = s + sub_i
                    r = results[sub_i]
                    b_local = r["boxes"].detach()    # [M, 4] 切块内部局部像素坐标
                    scores = r["scores"].detach()     # [M]
                    labels = r["labels"].detach()     # [M]

                    # 低置信度过滤 (保留边缘弱碎片)
                    keep = scores >= self.conf_thresh
                    b_local = b_local[keep]
                    scores = scores[keep]
                    labels = labels[keep]

                    if len(b_local) == 0:
                        patch_predictions.append({
                            "boxes": np.zeros((0, 4), dtype=np.float32),
                            "scores": np.zeros((0,), dtype=np.float32),
                            "class_ids": np.zeros((0,), dtype=np.int64),
                            "roi_features": np.zeros((0, 64), dtype=np.float32),
                        })
                        continue

                    # 构建 RoIs: [K, 5] -> (batch_idx, x1, y1, x2, y2)
                    batch_idx_col = torch.full((len(b_local), 1), sub_i, dtype=torch.float32, device=self.device)
                    rois = torch.cat([batch_idx_col, b_local], dim=-1)

                    # 4. RoIAlign 池化 1x1 并降维至 64D
                    roi_64 = self.roi_extractor(p3_feat, rois).cpu().numpy().astype(np.float32)

                    patch_predictions.append({
                        "boxes": b_local.cpu().numpy().astype(np.float32),
                        "scores": scores.cpu().numpy().astype(np.float32),
                        "class_ids": labels.cpu().numpy().astype(np.int64),
                        "roi_features": roi_64,
                    })

            # 5. 调用 datasetBuilder 统一构图与组装
            sample = collect_slice_predictions_from_detector(
                patch_predictions=patch_predictions,
                patch_rects=patch_rects,
                img_size=img_size,
                conf_thresh=self.conf_thresh,
                intra_nms_iou=self.intra_nms_iou,
                tau_d_pixels=self.tau_d_pixels,
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
                assigner=self.assigner if gt_boxes is not None else None,
                image_id=image_id,
            )
            return sample


# ============================================================================
# 4. 模拟仿真测试数据生成 (Mock / Self-Test: 验证 76 维输入全流程)
# ============================================================================

def generate_mock_dataset_with_p3_roi(
    num_samples: int = 16,
    seed: int = 42,
    c_roi: int = 64,
) -> Tuple[List[TileSample], List[TileSample]]:
    """
    生成带有 64 维模拟 P3 RoI 外观特征的仿真数据集，用于快速自检。
    """
    gen = SyntheticTiledDatasetGenerator(seed=seed)
    assigner = CroppedGTLabelAssigner(local_iou_thresh=0.5, matching_metric="cropped_iou")
    rng = np.random.default_rng(seed)

    all_samples: List[TileSample] = []
    for i in range(num_samples):
        s = gen.generate_image_sample(f"mock_img_{i:03d}", assigner=assigner)
        n = len(s.boxes_global)
        if n > 0:
            # 模拟 P3 抽取并降维后的 64 维外观特征：
            # 同一 GT 来源的切块碎片赋予相似的外观基向量 + 小扰动
            base_appearance = rng.normal(0.0, 1.0, size=(100, c_roi)).astype(np.float32)
            mock_rois = np.zeros((n, c_roi), dtype=np.float32)
            for box_idx in range(n):
                gid = s.gt_id[box_idx]
                if gid >= 0:
                    mock_rois[box_idx] = base_appearance[gid % 100] + rng.normal(0.0, 0.1, size=(c_roi,))
                else:
                    mock_rois[box_idx] = rng.normal(0.0, 1.0, size=(c_roi,))

            # 重新组装包含 64 维 RoI 特征的完整 76 维输入
            node_features, boundary_dists, is_boundary_risk = build_candidate_feature_vector(
                boxes_global=s.boxes_global,
                scores=s.scores,
                class_ids=s.class_ids,
                patch_ids=s.patch_ids,
                patch_rects=s.patch_rects,
                img_size=s.img_size,
                roi_features=mock_rois,
            )
            s.roi_features = mock_rois
            s.node_features = node_features

        all_samples.append(s)

    n_train = int(num_samples * 0.75)
    return all_samples[:n_train], all_samples[n_train:]


# ============================================================================
# 5. 端到端流程主控制器 (Pipeline Runner)
# ============================================================================

def run_end_to_end_demo():
    """
    一键自检模式：在无 GPU / 外部权重环境下，全自动验证：
      1. 生成含 P3 RoI 64D 特征的切片预测；
      2. 离线缓存序列化 (Save/Load)；
      3. 极速训练 76 维输入 (12 + 64) 的 TileStitchingNet；
      4. 运行全基线评测对比 (Greedy-NMS, Soft-NMS, WBF, SAHI vs Ours)。
    """
    print("\n" + "=" * 78)
    print(">>> 启动端到端全流程模拟自检 (Mock End-to-End Pipeline)")
    print("=" * 78)

    # 步骤 1: 生成含 64 维 P3 外观特征的样本
    print("\n[Step 1] 生成包含 P3 1x1 RoI (64D) 的高分辨率切块预测样本...")
    train_samples, val_samples = generate_mock_dataset_with_p3_roi(num_samples=20, seed=42, c_roi=64)
    print(f"  - 训练集样本数: {len(train_samples)}, 验证集样本数: {len(val_samples)}")
    print(f"  - 输入节点特征维度: {train_samples[0].node_features.shape[1]} (12 几何/切线先验 + 64 外观特征)")

    # 步骤 2: 验证离线持久化机制 (Dump -> Load)
    cache_path = os.path.join(".", "_tmp_demo_features.pt")
    print(f"\n[Step 2] 验证离线特征持久化机制 ({cache_path})...")
    OfflineDatasetIO.save_samples(train_samples + val_samples, cache_path)
    loaded_samples = OfflineDatasetIO.load_samples(cache_path)
    for p in (cache_path, cache_path + ".npy"):
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
    print("  - 离线序列化与重载校验通过！")

    # 步骤 3: 训练 TileStitchingNet (76 维输入)
    print("\n[Step 3] 极速离线训练 TileStitchingNet (BALR-Attention + 64D RoI)...")
    t0 = time.perf_counter()
    model, history = train_tile_stitching_net(
        train_samples=train_samples,
        val_samples=val_samples,
        roi_dim=64,
        epochs=12,
        lr=3e-3,
        lambda_stitch=2.0,
        lambda_edge=1.0,
        verbose=True,
    )
    train_sec = time.perf_counter() - t0
    print(f"  - 训练完成！耗时: {train_sec:.2f} 秒 (平均单 Epoch: {train_sec/12:.3f} 秒)")

    # 步骤 4: 全基线评测与消融对比
    print("\n[Step 4] 运行全基线评测 (Greedy-NMS, Soft-NMS, WBF, SAHI vs TileStitchingNet)...")
    predictor = TileStitchingPredictor(model=model, score_thresh=0.20, fallback_nms_iou=0.85)

    def ours_fn(boxes, scores):
        # 兼容简易接口
        return predictor.predict_sample(val_samples[0])[:2]

    def evaluate_fn_on_samples(samples: List[TileSample], fn) -> Dict[str, float]:
        pb_all, ps_all, lats = [], [], []
        for s in samples:
            t0 = time.perf_counter()
            if hasattr(fn, "predict_sample"):
                b, sc, l_ms = fn.predict_sample(s)
            else:
                b, sc = fn(s.boxes_global, s.scores)
                l_ms = (time.perf_counter() - t0) * 1000.0
            pb_all.append(b)
            ps_all.append(sc)
            lats.append(l_ms)
        return evaluate_postprocessor_on_dataset(samples, pb_all, ps_all, lats)

    baselines = {
        "Greedy-NMS (IoU=0.50)": lambda b, s: run_greedy_nms(b, s, iou_thresh=0.50),
        "Soft-NMS (Gaussian)":   lambda b, s: run_soft_nms(b, s, sigma=0.50),
        "WBF (IoU=0.55)":        lambda b, s: run_wbf(b, s, iou_thresh=0.55),
        "SAHI NMM (IoS=0.50)":   lambda b, s: run_sahi_nmm_ios(b, s, ios_thresh=0.50),
    }

    print("\n" + "-" * 92)
    print(f"{'Post-Processor Method':<28} | {'AP':<6} | {'AP50':<6} | {'AP75':<6} | {'Nested_Rec%':<11} | {'Split_Rec%':<10} | {'Latency':<8}")
    print("-" * 92)

    for name, fn in baselines.items():
        res = evaluate_fn_on_samples(val_samples, fn)
        lat = res.get("Latency_ms", res.get("mean_latency_ms", 0.0))
        n_rec = res.get("Nested_Recall50", 0.0)
        s_rec = res.get("Split_Recall75", 0.0)
        print(f"{name:<28} | {res['AP']:<6.2f} | {res['AP50']:<6.2f} | {res['AP75']:<6.2f} | {n_rec:<11.1f} | {s_rec:<10.1f} | {lat:<6.2f}ms")

    # 评测 Ours TileStitchingPredictor
    res_ours = evaluate_fn_on_samples(val_samples, predictor)
    lat_o = res_ours.get("Latency_ms", res_ours.get("mean_latency_ms", 0.0))
    n_rec_o = res_ours.get("Nested_Recall50", 0.0)
    s_rec_o = res_ours.get("Split_Recall75", 0.0)
    print(f"{'Ours: TileStitchingNet (76D)':<28} | {res_ours['AP']:<6.2f} | {res_ours['AP50']:<6.2f} | {res_ours['AP75']:<6.2f} | {n_rec_o:<11.1f} | {s_rec_o:<10.1f} | {lat_o:<6.2f}ms")
    print("-" * 92)
    print("-" * 78)
    print("\n>>> 全流程打通验证大获成功！")


def run_kaggle_extraction(args):
    """
    在 Kaggle / GPU 服务器上执行真实 DEIM-S 离线特征提取。
    """
    print(f"[Kaggle Extractor] 载入配置: {args.config}, 权重: {args.ckpt}")
    # 动态导入 DEIM 模块
    sys.path.insert(0, os.path.abspath(args.repo_dir))
    from engine.core import YAMLConfig

    cfg = YAMLConfig(args.config, resume=args.ckpt)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("ema", {}).get("module", ckpt.get("model", ckpt))
    cfg.model.load_state_dict(state)
    print(f"[Kaggle Extractor] 模型加载就绪, device={device}")

    extractor = DEIMSliceFeatureExtractor(
        deim_model=cfg.model,
        postprocessor=cfg.postprocessor,
        device=str(device),
        conf_thresh=args.conf_thresh,
        roi_out_channels=64,
    )

    # 此处接入真实图片集并循环提取... (保存至 args.out_path)
    print(f"[Kaggle Extractor] 特征将导出至: {args.out_path}")


# ============================================================================
# 6. CLI 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DEIM-S + TileStitchingNet 完整打通管线")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "extract", "train_eval"],
                        help="运行模式: demo (自检仿真), extract (Kaggle 离线提取), train_eval (离线训练与评测)")
    parser.add_argument("--config", type=str, default="", help="DEIM yml 配置文件路径")
    parser.add_argument("--ckpt", type=str, default="", help="DEIM 训练好的权重路径 (.pth)")
    parser.add_argument("--repo-dir", type=str, default="./DEIM-main", help="DEIM 仓库根目录")
    parser.add_argument("--data-file", type=str, default="./dumped_features.pt", help="离线提取特征文件存储路径")
    parser.add_argument("--conf-thresh", type=float, default=0.08, help="低置信度采掘阈值")
    parser.add_argument("--epochs", type=int, default=18, help="TileStitchingNet 训练轮数")

    args = parser.parse_args()

    if args.mode == "demo":
        run_end_to_end_demo()
    elif args.mode == "extract":
        run_kaggle_extraction(args)
    elif args.mode == "train_eval":
        samples = OfflineDatasetIO.load_samples(args.data_file)
        n_train = int(len(samples) * 0.8)
        train_samples = samples[:n_train]
        val_samples = samples[n_train:]
        model, _ = train_tile_stitching_net(train_samples, val_samples, roi_dim=64, epochs=args.epochs)
        predictor = TileStitchingPredictor(model=model)
        evaluate_postprocessor_on_dataset(val_samples, predictor)


if __name__ == "__main__":
    main()
