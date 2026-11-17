# 第一阶段 Kaggle 运行说明

本阶段修正评测、验证划分、梯度累积和诊断；未重构匹配标签、图结构或损失目标。

## 运行

1. 将更新后的整个 `DEIM-main` 上传到 Kaggle 代码 Dataset 并发布新版本；Notebook 选择新版本。
2. 重启 Notebook Session，使用更新后的 `tools/kaggle_dump_p3_features.py`（仍可复制到 Cell）。确认 `REPO_SRC`、`DATASET_SRC`、`CKPT_PATH` 指向正确数据。建议明确指定检测器权重路径。
3. 首次先设 `MAX_IMAGES_LIMIT=10`、`VAL_FREQ=1` 跑完整流程；确认输出后设 `MAX_IMAGES_LIMIT=None` 重跑全量。检查训练与测试整图均存在。
4. `USE_CACHED_FEATURES=True` 可以保留：旧缓存缺少匹配的 `.manifest.json` 会自动失效，首次需重新提取训练和测试特征。之后一起保存/上传 `.pt` 与 `.pt.manifest.json`。

脚本现在每次同步输入仓库代码，不再执行字符串“自愈补丁”。重启 Session 防止 Python 已导入模块仍驻留旧版本。

## 口径

- 候选池暂保留 `CONF_THRESH=0.15`、块内 NMS=0.70，统一只取 person 类别 1。所有基线及 Ours 使用同一缓存。输出阈值统一为 0；官方评测按分数取 maxDets=200。
- **这不是之前 score>0.01 的完整检测器 baseline**。本阶段结果只能解释固定当前候选池下的比较。降低提取阈值及移除块内 NMS 要单独实验，当前网络的稠密二次复杂度仍存在。
- 官方 `COCOExpandEval` 直接接收最终整图预测，不再追加 NMS；保留 ignore/uncertain/IOD。报告 all/tiny 的 AP25、AP50、AP75；表中 AP 是额外计算的 all-area AP50:95，不是官方 AP25。
- 训练原图按固定 seed=42 划分约 80%/20%；若 JSON 有 video_id/sequence_id，整组划分，否则按原图 ID。没有序列元数据时，不保证相邻视频帧隔离。
- 最佳轮次根据内部验证集 `Tiny_AP50` 选择；测试集不传入训练函数。旧实验已看过测试结果，因此不能把既有测试集称为从未接触过的盲测集。
- Split 子集固定为：GT 在至少两个切块有正面积交集，且至少一个视图截断 GT。使用全部切块几何，包括没有候选的块；召回从全体 GT 的官方一对一匹配中读取。它表示“存在截断视图”，并不表示所有视图均不完整。
- 每图损失已归一化，累积窗口再按实际非空图数平均。Token 数只控制更新频率，不把一张图拆成多个图。

## 输出

- `stage1_run.json`：运行参数和训练/模型源码哈希。
- `stage1_split.json`：划分种子、分组和训练/验证 image ID。
- `stage1_history.json`：各轮损失、完整验证指标、原始/包络/精修 IoU、提升/退化比例、偏移截断比例、回归头梯度和参数更新范数。IoU 诊断固定使用标签的 leader 与目标，不随预测重新匹配；按图平均，不能替代全 GT 召回。
- `stage1_metrics.json`：五种方法的最终官方指标、子集大小及延迟。
- `best_tilestitching.pth`：内部验证集选出的模型。
- `.pt.manifest.json`：标注/权重/源码/配置哈希、提取阈值、ROI 固定种子与运行版本；不匹配则重提取。

当前 Stitcher 仍在 CPU 训练和推理。GPU 用于检测器特征提取。训练标签中的 ignore/uncertain 忽略带、逐框 argmax、多种边与实例唯一输出属于后续阶段；本次官方评测的忽略规则已经正确处理，但训练标签机制未重构。

## 本地验证

在 `DEIM-main` 目录执行：

```text
python -B -X utf8 -m unittest TileStitchingNet.test_stage1 -v
```

测试覆盖官方匹配、uncertain、空预测、maxDets、几何子集、分组隔离、缓存几何往返、梯度平均/空尾更新，以及官方验证回调与 checkpoint 保存。未在本地完成 TinyPerson 全量训练；真实性能和 Kaggle 显存/耗时需在实际环境验证。
