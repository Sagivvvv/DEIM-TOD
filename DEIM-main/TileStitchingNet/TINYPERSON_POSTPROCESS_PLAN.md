# TinyPerson 切块后处理对比实施计划

2026-11-19 更新。按用户确认的口径，本轮正式对比只使用项目 val 入口对应的官方 786 张整图。每个方法使用提前固定的默认配置，逐个运行。历史参考数值来自 P5 报告。

## 1. 实验问题与基线

固定 DEIM 已经在每张切块上给出的 person 检测框和原始置信度，只改变跨块后处理，测量相对普通 Greedy-NMS 的精度和耗时差异。首轮使用项目已有的 P0-R 候选池，沿用 P5 的原分数基线 `run_greedy_nms(boxes, scores, iou_thresh=0.5, score_thresh=0)`。所有方法跳过本项目训练得到的重评分、关系模型和坐标修正。每张整图独立处理。

首轮交付顺序为普通 NMS 回归、TA-TM 和 CAR、经典方法、最后处理需要额外输入或等待作者实现的条目。经典 Cluster-DIoU 也能回答一个独立问题，即在同一候选池中把 Cluster-NMS 的 IoU 换成作者实现的距离判据，收益有多少。

## 2. 源码、输入与纳入条件

“作者源码”指目标论文作者发布或明确指向的可运行实现。下表记录截至本计划日期找到的材料。“待取得”表示本次未核实到相应作者实现。正式结果以完成源码锁定和适配验收的方法为准。

| 方法 | 已核实的来源 | TinyPerson 接入决定 |
|---|---|---|
| 普通 Greedy-NMS | 本仓库 `tools/kaggle_train_p5.py` 的 `BASELINES['Greedy-NMS']` 和 `TileStitchingNet/trainEval.py` | 主参照。保留 P5 并列分数顺序和输入精度。 |
| CAR，2026 | Cross-view agreement rescoring，作者 Zenodo **v1.3.0**，2026-11-05。含 `CAR_code.zip`、`CAR_repro.zip` 和结果归档。 [作者归档](https://zenodo.org/records/23956349) | 首批。读入全部切块视图、视图范围、逐视图候选及分数，保留能覆盖目标但没有检出的视图。按归档原顺序运行 agreement、rescoring 与所选 NMS 合并流程。配对运行同一流程的 CAR off/on；两者均报告相对项目普通 NMS 的差值。 |
| TA-TM，2026 | PCB 论文明确链接作者仓库，`src/topology_merge.py::tatm_image` 包含邻块评分与最终 NMS。 [论文](https://arxiv.org/abs/2605.24726) · [源码](https://github.com/mohammad-AJP/PCB-Defect-Detection/blob/main/src/topology_merge.py) | 首批。由真实切块矩形计算原点、实际宽高、网格行列、本地像素框和边界距离，映射到作者的逐图表格，读出调整后的分数。在普通 NMS 外，加跑同一作者 NMS、同阈值、关闭评分调整的对照。 |
| Cluster-DIoU-NMS，所指 2026 ASAHI | 2026 预印本描述了该模块，2023 ASAHI 论文第 3.3 节已用于切块检测。Zheng 等公开了更早的 `cluster_diounms`。本次未核实到 2026 论文作者的具体源码。 [2026 预印本](https://arxiv.org/abs/2604.19233) · [2023 论文](https://www.mdpi.com/2072-4292/15/5/1249) · [Zheng 等源码](https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/detection/detection.py) | 先跑并标记为“Cluster-DIoU-NMS（Zheng 等，上游实现）”。记录距离项指数、迭代停止条件和 top-k。2026 文章保留独立来源状态，待得到对应实现再核对；相同执行路径只计一次。 |
| CMA-NMS，DAHI | 论文 2025-07-30 在线，刊载于 2026 年卷。CMA-NMS 利用整图与局部推理结果及裁剪边界。本次未核实到作者模块。 [论文全文及作者联系方式](https://accedacris.ulpgc.es/bitstream/10553/144493/1/2507_DAHI_paper_PatternRecognition.pdf) | 待取得。若作者入口接受公共切块输入，按支持的模式运行并标明几何设置。若要求整图分支，统一新增同一 DEIM 的整图预测并保存，同池重跑普通 NMS。若依赖 DAHI 选区，另列完整流程及其同输入 NMS 对照。 |
| Group-Evidence，2025 | 文中使用高置信基线路径、低置信切块、空间 DBSCAN、ResNet-18 语义特征、重评分与最终 CB-NMS。本次未核实到对应作者代码。 [论文 v2 和作者联系方式](https://arxiv.org/html/2509.10779v2) | 待取得。同步索取权重标识、裁剪预处理和两分支构造方法。拿到后核对现有分数下限、整图分支与原图像素可用性；如需新增候选，配同池普通 NMS；记录裁剪与特征提取时间。P0-R 的 DEIM ROI 向量不能代替指定的 ResNet-18 表征。 |
| DIoU-NMS | Zheng 等的 Darknet 仓库提供 `diounms`。 [作者仓库](https://github.com/Zzh-tju/DIoU-darknet) | 调用作者的贪心 DIoU 分支，记录框格式、beta 和阈值。老 C 代码需要时在独立环境调用编译产物。 |
| Cluster-NMS | Zheng 等的 PyTorch 代码提供 `cluster_nms`。 [作者源码](https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/detection/detection.py) | 与 Cluster-DIoU 共用排序、候选预算和运行环境，选择无加权分支。先比对与 Greedy 的保留框，允许实测精度持平，单独量耗时。 |
| NMW | 原方法来自 Zhou 等的 CAD。WBF 作者提供独立的 `non_maximum_weighted`，本次未核实到 CAD 作者的对应模块。 [CAD 论文](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w14/html/Zhou_CAD_Scale_Invariant_ICCV_2017_paper.html) · [公开第三方实现](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) | 正文作者源码比较保留待取得；可在附表直接运行并明确标记“NMW（ensemble-boxes 第三方实现）”。 |
| Soft-NMS | Bodla、Singh 等的 Cython `cpu_soft_nms`。 [作者源码](https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx) | 预定 Gaussian，sigma=0.5，显式选 Gaussian 分支，记录输出剪枝阈值。作者函数就地重排并改分，在副本上跑并读取更新后的框和分数。核对源码的闭区间像素端点和单像素框。 |
| WBF | Solovyev 等的 `weighted_boxes_fusion`。 [作者源码](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) | 主行采用一个检测器的一份列表，每张图汇集所有切块候选，按整图宽高归一化 xyxy，单源权重 1，记录置信度聚合方式，输出再还原像素。每块单列会改变多源置信度含义，如另做应单列。 |
| SAHI-NMM | SAHI 项目的 `NMMPostprocess` 与 `GreedyNMMPostprocess` 是不同入口。 [官方源码](https://github.com/obss/sahi/blob/main/sahi/postprocess/combine.py) · [API](https://obss.github.io/sahi/api/) | 正式行显式调用 `NMMPostprocess`，同类匹配，预定 IOS=0.5；输入整图坐标、零位移，避免二次平移。项目历史同名方法由本地函数实现，作为旧结果参照。 |

## 3. 冻结输入与验证普通 NMS

1. 读取 `p0r_test.pt`、`p0r_test.pt.manifest.json`、`tiny_set_test_all.json`。复用 P5 的 `verified_samples` 和 `OfflineDatasetIO.load_samples`。每次运行均核对官方 786 个整图 ID 和三份输入文件的 SHA256。
2. 对每张图保存 image_id、宽高、稳定的候选行号、`boxes_global`、`scores`、`class_ids`、`patch_ids`、`patch_rects` 和包括空切块的 `tile_rects_all`。检查坐标范围、有效面积、原分数、person 类和行对应关系。所有后处理只读上述字段及各自声明的额外输入。GT 及缓存中派生的监督字段只进评测。
3. 官方测试池共有 174,146 个候选。P0-R 保留了旧候选池，旧 manifest 的 0.15 和实测最低分约 0.08008 不一致，准确提取阈值未知。主实验使用实存候选；块内已有筛选与 NMS 是共同上游条件。
4. 首先在官方 786 张上重跑普通 NMS，对照历史逐图框、分数及官方指标，量化运行环境对并列分数排序的影响。历史 Tiny AP50 约 **49.97737**；主表的差值均以当前环境、当前整图池实跑的普通 NMS 为参照。

仓库入口可从 [P5 的基线和数据校验](../tools/kaggle_train_p5.py)、[整图评测器](stage1.py)、[既有结果及来源说明](RESEARCH_PLAN.md) 和 [P5 审阅证据](reports/p5_20261118/review_evidence.json) 查阅。

## 4. 接入作者代码的验收

- 每个方法登记论文、仓库或发布 DOI、commit SHA 或压缩包 SHA256、源码入口、许可证、运行环境、完整参数、输入字段和输出坐标定义。主工程写薄适配器，仅处理格式、坐标、批次和设备。旧依赖可用独立进程或环境；如果必须抽取嵌在检测器里的函数体，保存上游原文件、最小 diff，并用原入口的同输入输出逐项对齐。
- CAR 固定上述版本，先取小体积的代码包、复核包和 README，按作者说明选最小参考样例或必要的单份结果。复核 agreement 和前后分数，再映射 TinyPerson 的真实 view ID、覆盖范围、候选和缺失投票。原归档若需要当前池没有的整图视图、低分候选或其他字段，就补同条件缓存并配套 NMS。不能用“没有候选行”推断“没有该切块”。特别检查同一视图内多个框最多贡献该视图一票、仅一个合格视图和多个支持视图三个用例。
- TA-TM 使用全网格推导行列，按作者 `infer_tiles.py` 与 `utils` 的像素坐标约定计算 `boundary_distance`。缓存中的归一化边距不能直接填入像素列。以作者配置的边缘带 16 px、邻块系数 0.2、连续性项关闭起步；最终 NMS 阈值以锁定脚本实际解析值为准，并记入结果。用同一小表分别从作者 CLI 和适配器跑一次，比较输出框及 `score_adj`。 [作者输入构造](https://github.com/mohammad-AJP/PCB-Defect-Detection/blob/main/src/infer_tiles.py) · [默认配置](https://github.com/mohammad-AJP/PCB-Defect-Detection/blob/main/configs/default.yaml)
- 经典方法共享整图浮点坐标。Darknet 对应中心宽高、WBF/NMW 对应归一化坐标、Soft-NMS 对应闭区间语义，各写往返检查；端点变换保留小数。核查 Cluster 原入口在 NMS 前的 top-k，将其配置为足以覆盖每图全部候选，保留原抑制核心。记录任何无法关闭的内部截断。相同输入按原作者入口与适配器比较框、数量、分数及同分次序，预先约定数值误差；避免用指标接近代替框级核对。
- 合成样例至少覆盖空图、单框、同分框、同一目标跨边界两次检出、近邻两个不同人、嵌套小框、三个框链式重叠、超过 200 个候选。再选从官方 786 张按 ID 固定抽取的小样本复跑。空图也进入评测分母；适配器报错的图片不能跳过。对上游没有现成参考输出的方法，用同批输入跑其原入口作为核对依据。

## 5. 参数、指标与报告

**默认主表。** 普通 NMS 固定 IoU=0.50；各对手运行预先登记的作者推荐设置或已注明的项目选择。公开列出匹配度量、阈值、分数聚合、输出剪枝、top-k、设备、输入池和额外特征。配对开关实验固定所有其他参数。官方 786 张只用于最终测量，禁用选参和阈值扫描。

**精度。** 直接向 `stage1.OfficialTinyEvaluator` 传最终整图框和分数，复用官方 ignore、uncertain、IOD 规则及每图 maxDets=200。保留完整输出供诊断，评测阶段按得分取前 200，禁止在评测器外再叠一轮全局 NMS。主指标 Tiny AP50，辅助 Tiny AP25、Tiny AP75、全尺寸 AP50:95。每行给相对同池普通 NMS 的差值，单位为百分点。按 P5 的视频/序列分组配对 bootstrap，固定 seed=42，1,000 次，报告 Tiny AP50 差值的 95% 区间。

**切块诊断与耗时。** 报告 `Split_Recall75` 及子集 GT 数、每图输入/输出框数、tiny/IoU=0.50 在截断前和前 200 的命中数，区分误删和排名变化。相同硬件与线程数，预热后官方 786 张整图至少重复 3 次，给每整图中位数和 P95。计入格式转换、建图、融合和必要的特征提取，排除公用检测器、磁盘加载和 GT 评测；GPU 同步计时。额外整图或选区推理另报耗时及端到端总耗时。CPU、GPU 和预计算特征的结果注明条件。

结果表字段建议为 **方法及实现来源｜候选池｜额外输入｜参数｜Tiny AP25/50/75｜全尺寸 AP｜ΔTiny AP50 [95% CI]｜Split Recall75｜ms/图（中位/P95）**。历史自写 Soft-NMS、WBF、SAHI-NMM 的分数不得填入作者代码行。另附来源状态表，记录未测原因和出处，指标留空。

## 6. 缺源码时如何推进

先查论文正文和附件、勘误、作者主页、项目仓库及 Zenodo 等归档。仍缺时，可向通讯作者索取**单独的后处理函数、默认配置、一份输入和预期输出**；涉及外观分支再索取权重及预处理说明。即使无法共享源码，作者也可在允许共享的样例或共同候选上运行并返回预测。对这种结果注明“作者代跑”、输入指纹、参数和可核验范围，仅在覆盖完整目标划分且按同一协议评测时报告完整数值。对外联系和数据发送另行安排。

有可信第三方公开实现时显式标明来源，放附表；只拿到论文公式时保留“未实测”及原因，继续其余方法。当前计划不安排 AI 从论文补写算法，也不使用其他数据集的论文 AP 代填 TinyPerson。若后来收到源码，在已冻结协议上追加评测。

## 7. 后续编码与交付顺序

| 顺序 | 编码内容与产物 | 进入下一步的依据 |
|---|---|---|
| 1 | 独立 benchmark 入口读取官方 786 张 P0-R 候选；输出候选指纹、输入检查及普通 NMS 结果 | 与历史记录逐图比对，量化同分排序差异 |
| 2 | 建立来源锁定清单，先接 TA-TM、CAR 及其 off/on 对照，再接经典方法；每法输出固定小样本的适配核对报告 | 原入口能运行、框与分数对齐、所需输入齐备；阻塞只影响该方法 |
| 3 | 用源码示例和合成用例核对适配，锁定参数、额外输入与主表或附表身份 | 运行配置冻结 |
| 4 | 对就绪方法运行 786 图测试；使用配对 NMS 作差并计算区间 | 逐图预测覆盖全部 ID，能从保存的预测重新评测出表中数值 |
| 5 | 输出 Kaggle 可执行入口与挂载说明、锁定清单、逐图整图预测、运行配置、CSV 汇总、计时记录、方法状态及可回传的报告包 | 每个请求的方法都有结果或具体待办；作者实现、上游实现、第三方实现可追溯 |

正式入口只接受 P0-R 的官方测试缓存、manifest 和对应整图 GT。新增视图或语义特征时，再挂载原图、相应权重和检测器配置。
