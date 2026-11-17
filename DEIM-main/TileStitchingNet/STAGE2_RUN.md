# 第二阶段：固定坐标的质量重评分

## 第一阶段结果与本次方向

786 张测试图上，Ours 的 all-area AP50:95=20.44，Greedy-NMS=20.44，Soft-NMS=20.85；Tiny AP50 分别为49.70、49.62、50.20。
Ours 的 Nested Recall50 有提升（36.6 vs Greedy 29.0），但整体指标尚未超过 Soft-NMS。
最佳 Epoch16 的固定 leader 诊断为 raw/env/refined IoU=0.7532/0.7393/0.7454，梯度非零、截断比例为零。
这支持先隔离重评分能力：当前包络平均损害定位，精修只恢复一部分；不能由此断言每个框都变差。
日志 `Leader_IoU` 与 `Reg_refined_iou` 的均值口径不同：前者包含无 leader 图的零值，后者跳过这些图，不能直接相减。

## 实现

新增 `stage2.py`，独立于原模型；第一阶段模型、提取脚本和缓存不改。

默认两个可学习对照：

- `pointwise`：逐框 MLP，使用原有几何、检测分数、边界和 ROI 特征，另加像素宽高的 log1p；去掉任意编号的 patch ID。
- `local`：在相同输入上加入局部关系聚合。空间树找最近16个邻居（加一个查询位置），按 max(32px, 2×自身最大边长)限制中心距离；同时允许同块与跨块邻居。边特征为相对中心、宽高比、IoU、同块标记、分数差、中心距离。空间树查询后只构造 N×K 边张量，没有 N×N 注意力。

两者均预测残差：`z_i = logit(s_detector_i) + delta_i`，`s_i = sigmoid(z_i)`。
输出头零初始化，初始分数等于输入分数（端点0/1先截断到[1e-6,1-1e-6]）；坐标始终原样保留。

训练目标重新读取完整训练 GT，不使用缓存里的 leader、旧 target_score 或 target_edge：

- 对普通 GT 的最大完整框 IoU >=0.5：q=该 IoU；同一 GT 的多个可靠候选都监督为正。
- 最大 IoU <0.3：q=0；[0.3,0.5)候选不监督。
- 若第二个普通 GT 的 IoU也>=0.5，且最高两个 IoU差<0.05，忽略归属模糊的候选。
- ignore/uncertain/crowd 区域与候选交集除候选面积>=0.5时，忽略非可靠普通正样本。普通可靠正样本优先。这是训练忽略带，**不是**官方评测 IOD 阈值的复刻。

损失 `L_QFL = sum_valid |sigmoid(z)-q|² BCEWithLogits(z,q) / max(1,N_positive)`。
采用原生 FP32 BCEWithLogits，图内按正候选数归一化，累积窗口按有效图数平均。
不要求训练匹配一对一：质量预测允许同一实例的多个候选为正，重复输出仍由 NMS 处理。

可选 `--with-ranking` 增加第三个 `local_rank` 对照。只对局部边中归属于同一普通 GT、两端有效且质量差>=0.1的有向候选对计算 `mean softplus(z_low-z_high)`，权重0.1。默认关闭，避免同时增加多个目标而无法归因。QFL及该排序项本身不宣称为新损失。

## 如何运行

上传更新后的整个 `DEIM-main`，重启 Kaggle Session。将第一阶段下列文件保存在 working，或上传为 Dataset：

```text
tinyperson_p3_train.pt
tinyperson_p3_train.pt.manifest.json
tinyperson_p3_val.pt
tinyperson_p3_val.pt.manifest.json
stage1_split.json                 # 建议一起保留以核对划分
```

入口是 **`tools/kaggle_train_stage2.py`**，不要再运行特征提取脚本。
它核验 GT 哈希、提取设置、训练/测试投影及源码签名一致性，复用已有候选和 ROI。训练监督在内存重新计算，无需提取器和检测器权重。

Notebook 中先运行（把 repo 改为你实际的新版代码挂载路径）：

```python
import subprocess, sys
repo = '/kaggle/input/datasets/xyjxyjxyj55/deim-tod/DEIM-main'
subprocess.run([sys.executable, repo + '/tools/kaggle_train_stage2.py', '--smoke'], check=True)
```

自动查找优先选有配套 manifest 的缓存。若找到多份合格文件，可用下面的参数指定路径。
旧导出格式的缓存（**无论是否带 manifest**）会被自动修复：移除占位非 person 行；缺少完整切块坐标时，按第一阶段切块网格（640×576、最小重叠50）由原图尺寸确定性重建，且仅当与每个候选保存的 patch 矩形逐一对齐时才采纳；无法核验的样本跳过 Split Recall，仍计算 AP。修复计数写入 `stage2_run.json` 的 manifest 字段（`stage2_cache_repairs`），控制台也会打印。试跑产物中的空 manifest 表示无法核验提取来源。

路径参数可追加到上述列表：

```text
--train-cache /kaggle/input/你的缓存目录/tinyperson_p3_train.pt
--test-cache /kaggle/input/你的缓存目录/tinyperson_p3_val.pt
--train-gt /kaggle/input/你的标注目录/tiny_set_train_all.json
--test-gt /kaggle/input/你的标注目录/tiny_set_test_all.json
--split-manifest /kaggle/input/你的缓存目录/stage1_split.json
```

正式对照请使用同次第一阶段导出的两份 `.pt` 和配套 manifest，再去掉 `--smoke`。若只漏上传了 manifest，从第一阶段 Notebook 的 Output 补传即可。默认训练2个模型、各18轮，CPU 4线程；可设 `--device cuda:0`，该阶段原生支持 GPU，但小图未必比CPU快。所有基线在同一运行内重测延迟，跨硬件延迟不可直接比较。
若直接将脚本粘贴到 Cell，会按默认全量参数运行；建议使用上述 subprocess 方式保留显式参数。
需要原生 pycocotools、scipy、torch；原第一阶段环境通常已具备。

## 评测与产出

划分与第一阶段相同：seed42，训练集内80%/20%，不使用测试集选轮次。完整缓存应得到597/149/786。
每个模型按验证 `Tiny_AP50` 选一次最优；初始 Epoch0（Greedy-NMS基线分数）也可入选。
若最终选择Epoch0，说明该次训练未提升此验证指标，不能宣称学到了有效改进。

同一最佳权重分别评测：

- `before_nms`：新分数决定NMS保留哪些原始框，再排序；NMS IoU=0.5。
- `after_nms`：保持原检测分数NMS=0.5的保留集合，只重排这些框。关系模型仍可观察原始全部候选作为上下文，但不能恢复被原NMS删除的框。

两者均由官方评测取每图前200。after_nms的变化包含排名、200框预算和跨图置信度校准；不能全部归因为预算回收。
四种传统基线使用同一候选池和输出阈值0；与第一阶段一样保留候选提取阈值0.15和块内NMS=0.7。
全量结果保存到 `/kaggle/working/stage2/full/`，小规模结果在独立的 `smoke/`：

- `stage2_run.json`、`stage2_split.json`：代码/缓存签名、参数、真实使用的image ID；旧格式缓存触发修复时，manifest 字段附修复计数。
- `pointwise_best.pth`、`local_best.pth`：模型、配置、验证指标、最佳轮次。
- `*_history.json`：Epoch0起的损失、验证指标和正/负/忽略标签统计；另有正候选质量MAE、同一GT内按原分数/新分数选出的框IoU和理想最大IoU。该诊断只覆盖可靠正候选，未执行NMS或预算截断，不等同于检测召回。
- `validation_metrics.json`、`test_metrics.json`：所有基线及两种推理位置的官方指标。

加载模型可用 `load_quality_checkpoint(path, device='cpu')`，再构造 `QualityPredictor(model, mode='before_nms')`。

## 判断下一步

先看是否有训练轮次超过Epoch0，再看是否同时超过验证集Greedy及Soft-NMS；固定设置后核对测试Tiny AP50、Tiny AP75、all AP与子集召回。
pointwise有效而local无额外收益，优先排查邻居覆盖率和关系特征；after_nms有效、before_nms退化，说明新分数适合排序但尚不适合决定NMS赢家。
若这一步证明质量信号有效，再分别加入坐标精修和互补边融合。当前仍是“可学习重评分+NMS”对照，不是已经完成的无NMS合并网络。

验证命令：`python -B -X utf8 -m unittest TileStitchingNet.test_stage1 TileStitchingNet.test_stage2 -v`。
本地测试含短程质量排序学习、零初始化等价性、忽略标签、梯度屏蔽、稀疏边、空图、缓存复用及完整训练/评测/权重重载；不替代Kaggle真实数据全量实验。
