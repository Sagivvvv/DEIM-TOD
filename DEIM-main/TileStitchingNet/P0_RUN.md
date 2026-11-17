# P0：输入核验与基线复现

入口：`tools/kaggle_audit_p0.py`。只读缓存/权重，不训练，不提取特征，不改输入文件；仅在独立输出目录保存审计产物。

## 准备

上传新版整个DEIM-main；保留或挂载：

- 第一阶段导出的 `tinyperson_p3_train.pt`、`tinyperson_p3_val.pt`，以及各自 `.pt.manifest.json`。
- 第二阶段 `full/` 目录：`pointwise_best.pth`、`local_best.pth`、`stage2_run.json`、`stage2_split.json`、`validation_metrics.json`、`test_metrics.json`。
- 完整训练/测试GT JSON。

审计允许读取没有manifest的缓存，但会明确标记来源未核验；这不表示可以据此通过正式实验的来源检查。有manifest但GT哈希不匹配时直接报错并打包诊断。

## Kaggle运行

在新Cell执行以下代码，repo改为新版代码实际挂载路径：

```python
import subprocess, sys
repo = '/kaggle/input/datasets/xyjxyjxyj55/deim-tod/DEIM-main'
subprocess.run([
    sys.executable, repo + '/tools/kaggle_audit_p0.py',
    '--smoke', '--output', '/kaggle/working/p0_smoke'
], check=True)
```

脚本自动查找缓存、GT及 `pointwise_best.pth` 所在的第二阶段目录。若同名文件存在多份，会要求显式指定路径，不静默任选。
先小规模核验后，运行全量（去掉smoke，换一个空输出目录）：

```python
subprocess.run([
    sys.executable, repo + '/tools/kaggle_audit_p0.py',
    '--output', '/kaggle/working/p0_full'
], check=True)
```

smoke仍读取全量缓存、核验全部指纹，但评测仅取4张验证/4张测试图；不会把这些指标和历史全量AP作比较。完整结论必须使用全量审计。

路径冲突或未找到文件时，在命令列表里添加所需参数及实际路径：

```text
--train-cache /kaggle/input/.../tinyperson_p3_train.pt
--test-cache /kaggle/input/.../tinyperson_p3_val.pt
--train-gt /kaggle/input/.../tiny_set_train_all.json
--test-gt /kaggle/input/.../tiny_set_test_all.json
--stage2-dir /kaggle/input/.../stage2/full
```

`--stage2-dir` 是同时包含两个最佳权重和第二阶段报告的目录。
运行默认CPU、4线程，可用 `--threads` 调整；延迟不会与历史值作硬性一致检查。
输出目录必须为空，重复运行请使用新目录，避免覆盖历史证据。

若保留了第一阶段当时实际使用的缓存/报告，可选添加：

```text
--reference-stage1-dir /kaggle/input/.../第一阶段输出目录
--reference-train-cache /kaggle/input/.../第一阶段当时的训练缓存.pt
--reference-test-cache /kaggle/input/.../第一阶段当时的测试缓存.pt
```

参考目录中应包含 `stage1_metrics.json`。不提供历史缓存时仍可完成当前池内复现，但无法据此证明两阶段实际候选相同，更不能确认SAHI基线变化的原因。

## 核验内容

1. 当前缓存文件SHA256、标注SHA256、manifest与训练/测试提取签名。
2. 原始缓存、内存修复后的逐图候选指纹：有序数组指纹与不受候选顺序影响的行集合指纹，另记完整切块几何。
3. 可选历史缓存逐图对比：候选内容变化、仅顺序变化、几何变化分别记录。
4. 训练/验证划分与原stage2清单比较；候选数、尺寸、分数、实际输入ROI通道统计。
5. pointwise/local零初始化，before/after NMS在验证/测试上的原始框与分数等价检查。
6. 当前4个传统基线、两个最佳权重的before/after模式；验证集和测试集分别执行官方评测。
7. 与历史validation/test指标比较，并与checkpoint内保存的验证指标交叉检查；原始AP差值容差1e-4（百分点），不按延迟判失败。
8. 输入缓存运行前后文件哈希相同；模型和输入源码哈希进入报告。

注意：candidate行集合指纹包括ROI与node_features，并非只比较框；缺失历史投影权重时，无法由manifest或ROI分布统计证明投影基一定相同。报告会保留这一证据边界。

## 发回什么

下载并发回 **`/kaggle/working/p0_full/p0_reports.zip`**。
里面包含 `audit_summary.json`、逐图指纹、划分、历史报告副本和各方法当前评测结果；不打包权重、图像或缓存。

主状态含义：

- `SMOKE_ONLY`：小规模流程完成，不代表全量复现。
- `REVIEW_MISMATCHES`：初始化、提取签名或数值复现等发现不一致，需要先分析。
- `COMPLETED_REVIEW_PROVENANCE`：已执行的数值检查未发现不一致；仍需人工审阅warnings、缺失证据和历史来源，不能自动视为P0全部通过。
- `ERROR`：未跑完；诊断包仍会生成，直接发回即可。

在报告分析完成前，不启动P1之后的新训练。P0工具产出不等于P0验收完成。
