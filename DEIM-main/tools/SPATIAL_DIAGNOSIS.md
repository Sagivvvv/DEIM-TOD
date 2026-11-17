# DFINE-DEIM-S 的 TinyPerson 空间诊断

工具默认使用 `configs/deim_dfine/dfine_hgnetv2_s_tinyperson_online.yml`。`dump` 优先严格加载 checkpoint 的 `ema.module`，以验证配置做 FP32 推理。`analyze` 在 CPU 上从缓存回放合并，复用 TinyPerson 的官方 GT 匹配和 AP 计算。运行环境需要项目依赖、SciPy 和原生 `pycocotools`。Windows 上使用 `-X utf8` 读取配置。

Kaggle 上将含 `model_pth/best_stg1.pth` 的权重数据集作为 Notebook Input，`dump` 会自动找到挂载文件。沿用此前的 Notebook 初始化单元格时，将权重赋值改为 `WEIGHTS = find_input("model_pth/best_stg1.pth")`。`--checkpoint` 始终优先使用指定的文件。本地默认依次查找仓库同级 `model_pth/best_stg1.pth` 和仓库内 `output/baseline/best_stg1.pt`。

在 PowerShell 中先跑四张图。这些图包含两种常见分辨率、没有正常 GT 的图和一个方向无法移动网格的短图。

```powershell
cd D:\Python_Project\DEIM-TOD\DEIM-main
conda activate deim
python -X utf8 tools/tinyperson_spatial_diagnosis.py dump --out output/spatial_diagnosis_smoke --image-ids 794 798 799 886 --batch-size 8
python -X utf8 tools/tinyperson_spatial_diagnosis.py analyze --cache output/spatial_diagnosis_smoke
```

全量命令默认读取同级 `datasets/Tinyperson` 下的整图、离线切块及其标注。`--checkpoint`、`--gt`、`--image-root`、`--tile-root` 和 `--tile-json` 可以指定其他位置。`--device` 默认自动选择 GPU 或 CPU。

```powershell
python -X utf8 tools/tinyperson_spatial_diagnosis.py dump --batch-size 8
python -X utf8 tools/tinyperson_spatial_diagnosis.py analyze
```

默认缓存写入 `output/spatial_diagnosis`。每张图每种网格以原子替换提交一个 NPZ。中途停止后，使用相同命令会复用已完成的分组。运行清单保存权重及数据索引哈希、配置、图像 ID、网格参数和运行环境。使用新权重、图像子集或导出参数时，用 `--out` 指定新目录。可以只跑 `--grids baseline` 检查前两个问题。四偏移配对需要同一缓存包含 `p00 p10 p01 p11`。移动缓存到另一台机器时，可用 `analyze --gt` 指向哈希相同的整图标注。

`dump` 默认每批最多推理 8 个切块。每张原图完成后输出一行，显示进度、网格数、总切块数和本次耗时。续跑时还显示复用的网格数。

分数、NMS 和碎片阈值可以只在离线阶段修改。单独保存不同分析条件的报告，便于比较。

```powershell
python -X utf8 tools/tinyperson_spatial_diagnosis.py analyze --grids baseline --nms-threshold 0.6 --report-dir output/spatial_nms06
python -X utf8 tools/tinyperson_spatial_diagnosis.py analyze --evidence-floor 0.01 --report-dir output/spatial_evidence01
```

**合并损失。** 基线按现有切块索引读取物理小图，核验其网格与仓库评估器一致。按配置以 `score > 0.01` 构造合并前集合，回放跨类别 NMS、可选合并数量上限和每图 `person` 前 200 的评估上限。`direct_nms_lost_gt` 数的是合并前有合格候选、NMS 后所有合格候选都消失的目标。`nms_assignment_loss`、`max_det_assignment_loss` 和 `eval_cap_assignment_loss` 是各阶段同类一对一最大匹配数的差。`greedy_assignment_gap` 用最终候选的最大匹配数减官方实际 TP。`budgeted_selection_gap` 将合并前最大匹配按每图预算截断后减实际 TP，只衡量固定候选集合下的选择空间。指标默认输出 IoU 0.25、0.50、0.75，终端主指标为 0.50。

**碎片和候选证据。** 正常 GT 的 `evidence_label` 依次检查合格框、低分整框、TopK 未选中的整框、疑似切边碎片、其他重叠和指定下限下无证据。碎片要求预测框至少 70% 落在 GT 内，覆盖 GT 的 10%–80%，IoU 低于当前阈值，且 GT 跨过来源块的一条内部裁切边，预测框对应边在 4 px 内。这里使用 query 的 `person` sigmoid 分数，默认下限为 `>0.001`。报告另存 `0.001` 和 `0.01` 两档是否存在整框，以及不限分数时的最佳重叠，便于识别低分证据。碎片标签适用于人工复核。官方匹配使用正常 GT，保留 `ignore`、`uncertain`、`iscrowd` 和 IOD 的原有规则。带密集标记的整图在采集时排除。

**四偏移。** 实验窗口为 `640×576`，横纵步长为 `590/526`，相位差为 `295/263`。所有窗口保持固定像素尺度，超出原图的部分补黑。四组完整覆盖同一张图，使用同一数量的窗口，且窗口内真实像素面积之和一致。四组统一过滤中心落在补边的预测。补边位置与内部裁切边随相位变化，所以报告分别按原图边缘和内部切边分层。若一轴短于或等于窗口，该轴两种相位共用一个居中窗口。此类图保留在逐图结果中，主汇总使用四个不同网格的样本。每图计数以合并后、官方评估上限之前的 `person` 预测数为准，并附官方 TP 数。

| 文件 | 内容 |
| --- | --- |
| `manifest.json` / `dump_complete.json` | 采集配置、运行签名、完整导出的分组数 |
| `candidates/<grid>/<image_id>.npz` | 裁块坐标与真实区域、两个坐标变换、PostProcessor 输出、每条候选的 query ID、TopK 前全部 query 框与 logits |
| `reports/analysis_parameters.json` / `summary.json` | 离线参数、各网格 AP/召回/损失、四偏移命中和计数汇总 |
| `reports/gt_diagnosis.csv` | 每个 GT、网格和 IoU 的候选证据、丢失阶段、匹配候选和切边证据 |
| `reports/image_grid_stats.csv` | 每张图、网格和 IoU 的合并损失、检测计数和计算预算 |
| `reports/grid_gt_stability.csv` | 同一 GT 的四组命中模式、六个相位对的框间 IoU 和中心位移 |
| `reports/grid_image_stability.csv` | 同一张图的四组计数、TP 数、极差和标准差 |
| `reports/replay/<grid>/<image_id>.npz` | 各阶段的候选索引、去留状态和 NMS 抑制者索引 |
| `reports/detections_<grid>.json` | 可供官方 TinyPerson 评估的整图 `person` 结果，附候选索引 |

CSV 和回放文件中的 `candidate_index` 是同图同网格下按 `[tile, rank]` 展平的下标。可用 `tile_row, rank = divmod(candidate_index, scores.shape[1])` 定位切块，再用 `metadata.tile_ids[tile_row]` 获取来源 ID。`query_index` 以同样方式用 `pred_boxes.shape[1]` 展平 `[tile, query]`。`boxes_tile` 为切块像素 `xyxy`，`pred_boxes` 为归一化 `cxcywh`，`valid_rects` 为切块内真实像素区域。对一个 `outcome=nms_removed_all_qualified` 的目标，用 `best_pre_candidate_index` 和 `best_pre_suppressor_index` 可直接找到被删框及抑制框。
