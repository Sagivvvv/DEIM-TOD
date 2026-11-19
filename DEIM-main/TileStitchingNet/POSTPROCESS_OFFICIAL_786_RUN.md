# TinyPerson 官方 786 张整图后处理对比

运行入口为 `tools/benchmark_tile_postprocess.py`。每次执行一个方法，读取与项目 val 入口对应的整图标注 `tiny_set_test_all.json`、`p0r_test.pt` 和同目录的 `p0r_test.pt.manifest.json`。脚本在加载候选前对照 `P5_REFERENCE.json` 的 test 段校验三份文件。每次评测均覆盖 786 个原图 ID、174,146 个相同的原始候选。

## 本机运行

在项目根目录打开 PowerShell，用下面的入口逐项续跑。入口固定使用官方 786 张整图标注和本机 Downloads 中的候选缓存，跳过已完成的方法，结束后生成 `reports/postprocess_official_786/RESULTS.md`。按 Ctrl+C 中断后，可用同一命令继续；中断的方法会从头开始。

```powershell
& D:/anaconda3/envs/deim/python.exe -B -X utf8 DEIM-main/tools/run_official_786.py
```

只测一个方法可在命令末尾加 `--method car`。只看完成状态可加 `--list`。

### 手动调用底层评测

在 `DEIM-TOD` 根目录打开 PowerShell。作者代码已经放入 `DEIM-main/output/postprocess_sources`。首先运行普通 NMS，随后将 `--method` 改为下表对应名称逐项运行。

```powershell
$officialCache = Join-Path $env:USERPROFILE 'Downloads/p0r_test.pt'
& D:/anaconda3/envs/deim/python.exe -B -X utf8 DEIM-main/tools/benchmark_tile_postprocess.py --method nms --cache $officialCache --gt datasets/Tinyperson/annotations/tiny_set_test_all.json
```

| 名称 | 固定配置与出处 |
|---|---|
| `nms` | 项目 P5 普通 Greedy-NMS，原始分数，IoU=0.50。 |
| `tatm-off`、`tatm` | 同一作者实现，最终 IoU=0.45，像素边界带 16，λ=0 / 0.2，μ=0。使用同一批全精度 DEIM 框，按作者原函数保留评分舍入。缓存中预测框可能伸出来源切块，计算边界距离时以 0 为下限，输入全局框原样传给两组。 [TA-TM 源码](https://github.com/mohammad-AJP/PCB-Defect-Detection/tree/5536f1867feb4d24bc7574b954afdd2f1afada75) |
| `car-off`、`car` | 作者 NMS IoU=0.60；开启时 λ=5、投票 IoU=0.75、边缘余量系数 0.5。逐图提供含空切块的真实 DEIM 网格，采用作者 `agr_only` 计算资格与投票，作者实现允许符合资格的来源视图计一票。在整个测试池按一个最大值归一化。这里评测切块视图的重评分模块；作者基准的视图生成器另可提供整图视图。 [CAR v1.3.0 归档](https://zenodo.org/records/23956349) |
| `diou-nms` | 编译作者 Darknet 的 `box.c::diounms_sort`，IoU=0.50、β=1。 [DIoU 源码](https://github.com/Zzh-tju/DIoU-darknet/tree/257ba175cb626e6d2b10da25f49332c71696550a) |
| `cluster-nms`、`cluster-diou` | 实际调用作者 `Detect.forward`。输入已解码的候选，阈值 0.45，距离指数 0.5，top-k 覆盖本图全部候选。源码含 CUDA 和 21 类常量，保留一个 person 通道，其他通道置零。报告 GPU 耗时，包含传入数据与取回结果。Cluster-DIoU 行对应 Zheng 等上游实现。 [Cluster 源码](https://github.com/Zzh-tju/DIoU-SSD-pytorch/tree/cec038bc1057f0cd532752413b24924fde427f09) |
| `soft-nms` | 编译作者 Cython Gaussian 分支，σ=0.5，内部剪枝 0.001。端点转换适配其加 1 的面积定义。 [Soft-NMS 源码](https://github.com/bharatsingh430/soft-nms/tree/b8e69bdf8df2ad53025c9d198ded909b50471d4f) |
| `wbf`、`nmw-third-party` | 同一检测器，每图一个列表，权重 1，IoU=0.55。按整图宽高归一化，保留源码对越界框的裁剪行为。该库由 WBF 作者维护，NMW 列入第三方附表。 [ensemble-boxes 源码](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/tree/651884c5c7df13a920d286404b47c6fa86944782) |
| `sahi-nmm` | SAHI v0.11.36，`NMMPostprocess`，同类 IOS=0.5。对含负坐标的图，将整图坐标系作共同平移，合并后逆变换，以保留框尺寸与相对位置。 [SAHI 源码](https://github.com/obss/sahi/tree/68e61981e1f79c56299c208df1d5bac03a01811d) |
| `cma-third-party` | CAR 作者归档中的 `class_wise_cma_nms`，仅传切块视图，按该入口默认参数运行（IoU=0.65、边距 10 px、IOS=0.9）。列入第三方附表；DAHI 作者的整套选区和整图推理另需源码和输入。 [第三方实现出处](https://zenodo.org/records/23956349) |

Group-Evidence 原作者完整实现待取得。CAR 归档中明确标为 spatial-only 的 Xiao 方法对照缺少原论文的语义门控，正式比较中保留待测。 [Group-Evidence 论文](https://arxiv.org/html/2509.10779v2)

## 输出与复核

`output/postprocess_benchmark/official_786` 下每个方法各有结果 JSON、逐图框与分数、三轮逐图耗时、bootstrap 输入。指标采用 TinyPerson 的 ignore、uncertain、IOD 与 maxDets=200，另报完整保留集合的 tiny GT 命中数。latency 的中位数和 P95 从三轮逐图测量汇总；CAR 获取整池归一化常数的额外准备时间单列为 `split_prepare_ms`。CPU 与 CUDA 的耗时需结合设备列阅读。区间由同一序列分组配对抽样 1,000 次获得，单位为 Tiny AP50 百分点。

```powershell
& D:/anaconda3/envs/deim/python.exe -B -X utf8 DEIM-main/tools/check_tile_postprocess.py --work DEIM-main/output/postprocess_benchmark/checks
& D:/anaconda3/envs/deim/python.exe -B -X utf8 DEIM-main/tools/report_tile_postprocess.py --gt datasets/Tinyperson/annotations/tiny_set_test_all.json --output DEIM-main/TileStitchingNet/reports/postprocess_official_786
```

其他机器可将按照上表提交号检出的仓库放入 `--sources` 下的 `ta_tm`、`diou`、`cluster`、`soft`、`wbf`、`sahi` 目录。CAR 代码包保存为 `CAR_code.zip`，解压到 `car`；脚本核验归档 SHA256 及实际使用的源文件。独立运行 `tools/build_postprocess_extensions.py <sources>` 编译 Soft-NMS 和 Darknet，需 Python、Cython、NumPy、setuptools 与 C 编译器。CPU 评测需要项目的 PyTorch、torchvision、pandas、PyYAML、原生 pycocotools；SAHI 与 WBF 另需对应上游依赖。当前机器的额外包位于 `--sources/deps`，各方法直接导入固定提交的作者代码。
