# P0补充：实际缓存内容与来源核验

目的：量化manifest与分数的矛盾，排查缓存字段错位/序列化问题，观察ROI差异在尺寸/分数分层后是否仍存在。
不训练、不执行检测器、不重建/重写缓存、不改变manifest，也不扫描alpha或调参。

## Kaggle运行

上传更新后的代码。脚本现在默认内置已审阅的2026-11-16 P0固定快照（两份缓存SHA256、149张验证图ID、原manifest声明的阈值），不再依赖Kaggle上传额外的ZIP。
该快照来自已收到的真实P0报告，不从当前缓存重新猜测；原manifest的0.15仍是待核验的声明，并非已证实的真实提取阈值。
继续挂载此前两份缓存；本次不需要GT、检测器权重或第二阶段模型权重。

```python
import subprocess, sys
repo = '/kaggle/input/datasets/xyjxyjxyj55/deim-tod/DEIM-main'
subprocess.run([
    sys.executable, repo + '/tools/kaggle_audit_p0_sources.py',
    '--train-cache', '/kaggle/input/datasets/xyjxyjxyj55/sample/trainvalsample/tinyperson_p3_train.pt',
    '--test-cache', '/kaggle/input/datasets/xyjxyjxyj55/sample/trainvalsample/tinyperson_p3_val.pt',
    '--output', '/kaggle/working/p0_sources_v2'
], check=True)
```

请删除旧Cell中的`--p0-report`及其路径。只有需要显式使用另一份P0参考时才传该参数；指定不存在的文件仍会报错，不会静默切换候选池。
所有路径按实际挂载位置调整。输出目录必须为空，重复运行请换新目录。
全量缓存内容检查不运行官方AP，默认一次检查全部图。CPU即可，不需要GPU。

## 核查项目

- 校验缓存文件SHA256与已审阅P0报告完全一致，避免混入另一批缓存。
- 按图统计score<0.08、score<0.15、score<manifest阈值及明显偏离阈值的数量；实际阈值未知时不凭最小值补造历史参数。
- 对比`scores`与`node_features[:,4]`；不一致可能涉及字段错位或后续修改。
- 对比`roi_features`与`node_features[:,12:]`。前者历史保存为FP16，按FP16量化后逐行比较，避免正常量化误差产生假阳性。
- 记录空ROI、非有限值、非person行。
- 分别按候选加权、每图等权，以及固定尺寸/分数区间计算ROI通道均值并比较。每个分层至少双方各100候选才进行比较。
- 复用P0验证集ID，确保train/validation划分不变。运行后再次验证缓存文件哈希未变。

有真实历史日志、提取脚本或投影state_dict时，可追加重复的`--evidence /路径/文件`参数。它只记录路径、哈希和大小，不会把某个文件的存在自动当成“此缓存确由它生成”的证据。需要查看文件内容时应另行提供该历史文件。

## 回传

将 **`/kaggle/working/p0_sources_v2/p0_sources_reports.zip`** 发回。
其中含source_audit.json和训练/验证/测试逐图统计；不含完整特征、缓存或权重。报错也会尽量生成诊断包。

预期可能出现`METADATA_CONTRADICTION`：这不是程序崩溃，而是确认实际分数低于声明阈值。`CONTENT_MISMATCH`表示分数字段、ROI对应关系或数值本身存在额外问题；需要先查具体图。

## 结果后的决定

1. 若字段错位/ROI数组与模型输入不一致，先追查其生成步骤，不直接修改缓存掩盖问题。
2. 若内容自洽、但元数据不可信，保留旧实验及哈希，补写独立的“观测事实与未知来源”记录；不把0.08猜测写成已证实的提取阈值。
3. ROI分层相近或差异都不能直接证明投影权重身份。若有可靠原始日志和投影状态，优先比对这些直接证据。
4. 若直接来源证据不存在，不继续无限重复统计。明确选择：建立可追溯的同源特征缓存，或重新训练不使用ROI的独立对照。旧模型输入ROI置零不等于正确的无ROI训练消融。
5. 下一步需要重新提取时，尽量保留候选坐标与分数，仅补可追溯特征以减少变量；是否可行需结合原图、检测器权重与坐标映射核实。若候选池改变，必须重跑基线。

本脚本只完成缓存内部核验，不能单独闭合历史来源链。获得结果后再决定P0正式验收和P1的适用范围。
