# TinyPerson 在线切块训练/评测适配 DEIM —— 会话交接文档

> 本文件是给**完全没有本次对话记忆**的新 AI 的自包含交接说明。读完即可无缝续接。
> 生成时间上下文：任务主体代码已写完，剩验证/微调。原始批准计划文件（只读、禁止编辑）：
> `C:\Users\19333\AppData\Roaming\QoderCN\SharedClientCache\cache\plans\TinyPerson_Online_Tile_Eval_8d182bdf.md`

---

## 1. 【任务目标】最初要解决的问题、期望的最终效果

在 **DEIM-main**（D-FINE / DEIM 目标检测框架）里，把 **TinyPerson 微小行人数据集**的**训练 + 验证 + 官方评测**打通成一条完整可跑的流水线。

核心诉求（逐条，均来自用户原始需求）：

1. **训练集已物理切块，保持不动**：`datasets/Tinyperson/train` 已经是 640×576 的小图 + 对应标注 `tiny_set_train_tiles.json`，训练直接用，不重新生成/改写。
2. **测试集是未切分整图**：`datasets/Tinyperson/test` 是大图（3840×2160、1920×1080 等）。要求在**验证/评测时在线切块**（不预先生成、不改写测试集文件）。
3. **在线切块规则**：
   - 切块尺寸 **640×576（宽×高）**，横/纵**名义重叠 50**（步长 590 / 526）。
   - 某方向不能整除步长时，**最后一块平移使边缘与原图对齐**（该块实际重叠可能 > 50）。
   - 原图该方向长度 ≤ 切块尺寸时**不补黑边**，用实际范围 `[0, length]`、偏移记 0。
   - 每块送 DEIM 推理 → 收集同一原图所有块预测 → **映射回原图坐标** → merge → 按 TinyPerson 官方流程评估。
   - **确保每张原图全部块预测完成后才合并评估**；**merge 只执行一次**。
4. **评测规则严格以官方 `evaluate_tiny.py` 实际实现为准，不替换/简化，注释里标注参数出处**：
   - sea 与 earth 目标合并为同一类 `person`；
   - 只有 normal 标注作正样本；
   - 含 dense 标注的图按 TinyPerson 规则排除评测；
   - `ignore` 与 `uncertain` 都按忽略区域处理，匹配它们的检测**既不算 TP 也不算 FP**；
   - 保留评测代码**实际使用**的 IoU/IOD 阈值与 AP/MR 计算方式。
5. **缺文件可提出**；用户明确：**“测试计划在别的平台自己跑，把代码写好写完整就行”**（本机无需运行，只需代码完整正确）。

**期望最终效果**：一条 `configs/deim_dfine/dfine_hgnetv2_s_tinyperson_online.yml` 配置，`torchrun train.py -c <该配置>` 即可训练（物理切块）+ 每轮在线切块验证并按官方规则出 AP（可选 MR）。

---

## 2. 【背景与约束】项目结构、技术栈、已定方案与决策、不要动的东西

### 2.1 项目/技术栈
- 框架：**DEIM-main**（RT-DETR/D-FINE 血统），模型 D-FINE-S（HGNetv2-B0 + HybridEncoder + DFINETransformer）。
- 配置系统：**YAML + `@register()` + `create()`**（`engine/core/workspace.py`）；`__include__` 组合；`__inject__`/`__share__` 依赖注入。
- 数据后端：`torchvision.datasets.CocoDetection`；DEIM 用 **`faster_coco_eval`**（在 `engine/data/dataset/coco_dataset.py` 顶部 `faster_coco_eval.init_as_pycocotools()` 把 `sys.modules['pycocotools.*']` 换成编译版影子）。
- 评测工具：**huicv**（在 `DEIM-main/tools/huicv/...`）——官方 `evaluate_tiny.py` 使用的 `COCOExpandEval`（tiny 口径 AP）与 `cityperson_eval`（MR）。

### 2.2 数据现状（已核实）
- 目录：`datasets/Tinyperson/{annotations, train, test}`；`train/{labeled_images, pure_bg_images}`、`test/{labeled_images, pure_bg_images}`。
- `annotations/tiny_set_test_all.json`（**整图 GT**，评测基准，minified 单行）：
  - 786 张整图；单类 `person`（`category_id=1`，sea/earth 已在 GT 里合并）。
  - 标注字段含：`ignore`/`uncertain`/`logo`/`in_dense_image`/`iscrowd`/`area`/`bbox`(xywh)。示例含 `uncertain:true`。dense 图已在该 GT 里排除（`in_dense_image:false`）。
  - images 字段：`{"file_name": "labeled_images/xxx.jpg", "height": 2160, "width": 3840, "id": 794}` —— **带 `labeled_images/` 前缀**，故 `img_folder` 指向 `test/`（不是 `test/labeled_images/`）。
- `annotations/tiny_set_train_tiles.json`（**物理切块训练标注**，pretty 多行）：
  - 切块 640×576；`file_name` 形如 `labeled_images/bb_V0032_I0001640__r000_c000.jpg`；单类 `person`；**无 `corner` 字段** → 训练 `corner_mode=False` 直接读块。

### 2.3 已确定的关键方案与决策（用户拍板）
1. **官方规则源不改**：`evaluate_tiny.py` 及 huicv 依赖（`expand_cocofmt_eval.py`、`cityperson_eval.py`）**一律不动**，作为规则唯一真源，被**进程内复用**。
2. **AP 进程内复用 `COCOExpandEval`**：评测器构造时**临时**把 `sys.modules['pycocotools*']` 从 faster 影子**恢复为原生 pycocotools** 再 import（原生纯 Python `accumulate` 才能配合被重写的 `evaluate/_prepare/evaluateImg` 与 IOD 逻辑）。→ **运行环境需 `pip install pycocotools`**（requirements.txt 里没有）。
3. **MR 默认关闭**：新增 `evaluator.eval_mr`，**默认 False**（训练每轮与 `--test-only` 都只算 AP，不自动跑 MR），保留手动开关；开启时懒加载 `cityperson_eval`（依赖 matplotlib/scipy）。
4. **merge 只做一次**：DEIM 侧用 **torchvision NMS(th=0.5)** 合并到整图，直接喂官方 AP/MR；**不调用** `evaluate_tiny.merge_det_result`（规避其 `mini_maskrcnn_benchmark` 编译依赖）。
5. **切块索引数据集与评测器共用** `build_tile_index`（同参数 + 同一个整图 COCO 对象），保证 `tile_id ↔ (corner, 整图id)` 两侧完全一致。
6. **旧离线切块配置保留、不删除、不引用**：`configs/dataset/tinyperson_crop_detection.yml`、`configs/deim_dfine/dfine_hgnetv2_s_tinyperson_crop.yml`（语义被本方案取代）。
7. **yaml_config 无需改**：`implementedEvaluators` 已含 `TinyPersonMergeEvaluator`。

### 2.4 约束/不要动
- 不改 DEIM 核心链（det_engine/det_solver/yaml_config/workspace）。
- 不改训练数据/标注（用现有物理切块）。
- 不改官方 huicv 评测脚本。
- 计划 `.md` 文件禁止编辑。

---

## 3. 【已完成的工作】改了哪些文件、目的、关键代码

共 **6 个文件**（3 新建代码 + 1 重写 + 1 导出 + 2 配置；合计新建5/改2）。全部**语法自检通过**（编辑器的 py 语法检查报告 no syntax errors），但**尚未在真实 Python 环境编译/运行**（见 §4、§6）。

### 3.1 新建 `engine/data/dataset/tinyperson_tiling.py`（共享切块工具）
- 目的：数据集与评测器共用，保证 tile_id↔corner↔整图 映射一致。数学等价 huicv `SplitImage(use_fix_size=True)`，但自包含（规避其 `np.int` 弃用）。
- 关键函数：
  - `_axis_positions(length, tile_len, step)`：单方向覆盖 `[0,length]` 的区间列表；`length<=tile_len` 时返回 `[(0,length)]`（不补边）；末块 `end>=length` 时左移到 `start=length-tile_len, end=length`；去重相邻同 `start`。
  - `generate_tile_corners(width, height, tile_w=640, tile_h=576, overlap=50)`：`step_x=tile_w-overlap`、`step_y=tile_h-overlap`；行优先（y 外层 x 内层）返回 `[(x1,y1,x2,y2),...]`。
  - `build_tile_index(coco, tile_w, tile_h, overlap)`：按 `sorted(int(i) for i in coco.getImgIds())` 顺序，用 `coco.imgs[id]['width'/'height'/'file_name']` 生成块，`tile_id` 从 1 连续编号，返回 `[(tile_id, orig_id, file_name, (x1,y1,x2,y2)), ...]`。
- 已手工验算：3840×2160 → X 7 块（0/590/1180/1770/2360/2950/3200）、Y 5 块 → 35 块；1920×1080 → 4×2=8 块。覆盖完整，重叠加区由 NMS 处理。

### 3.2 新建 `engine/data/dataset/coco_dataset_tinyperson_tile.py`（在线切块数据集）
- `@register() class TinyPersonTileDataset(CocoDetection)`；`__inject__=['transforms']`、`__share__=['remap_mscoco_category']`。
- `__init__(img_folder, ann_file, transforms, tile_w=640, tile_h=576, overlap=50, return_masks=False, remap_mscoco_category=False)`：
  - `super().__init__(..., corner_mode=False)` → `self.coco` = **整图 COCO**（faster 版），`self.ids` = 整图 id 列表；
  - `self.tiles = build_tile_index(self.coco, ...)`；`__len__ = len(self.tiles)`。
- 重写 `load_item(idx)`：取 `(tile_id, orig_id, _fn, (x1,y1,x2,y2))`；`image=self._load_image(orig_id).crop((x1,y1,x2,y2))`；`self.prepare(image, {'image_id': tile_id, 'annotations': []})` 生成**空框** target（`orig_size` = 裁后切块尺寸 (x2-x1, y2-y1)）；补 `target['idx']=torch.tensor([idx])`；空 boxes 走 `convert_to_tv_tensor(..., spatial_size=image.size[::-1])`。
- 复用父类 `__getitem__`（`load_item` → `self._transforms`）。
- 关键正确性：`get_coco_api_from_dataset(val_dataloader.dataset)` 返回 `dataset.coco`（整图 COCO）注入评测器 `coco_gt`，评测器用同参 `build_tile_index(coco_gt)` 复现同一映射。

### 3.3 重写 `engine/data/dataset/coco_eval_tinyperson.py`（在线 merge + 官方 AP/MR）
> 旧版是 faster_coco_eval 标准 COCO + 离线切块 json（靠 `corner`/`ori_file_name` 字段），**已整体替换**为下述在线方案（不再依赖那些 json 字段，规避“ori_file_name 遗漏 → AP=0”老坑）。

- `__all__=['TinyPersonMergeEvaluator']`；顶部导入（**已移除未使用的 numpy**）：`contextlib, importlib, json, os, sys, tempfile, torch, torchvision, from ...core import register, from ...misc import dist_utils, from .tinyperson_tiling import build_tile_index`。
- `_tools_dir()`：`os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..','tools'))` → `DEIM-main/tools`。
- `_load_official_evaluators()` → 返回 `(原生 COCO, COCOExpandEval)`：
  1. **无条件**把 `sys.modules` 里 `pycocotools` 与 `pycocotools.*` 条目弹出并缓存（faster 影子常是 `ModuleType` 且无 `__file__`，故不按 `__file__` 过滤，直接全清）；同样清 `huicv.evaluation.expand_cocofmt_eval*` 缓存。
  2. `importlib.import_module('pycocotools.coco'/'pycocotools.cocoeval')`；失败→还原影子并抛 `ImportError`（提示 `pip install pycocotools`）；再校验 `cocoeval.__file__` 不含 `faster_coco_eval`，否则还原并抛 `RuntimeError`。
  3. 把 `_tools_dir()` 加入 `sys.path`，`importlib.import_module('huicv.evaluation.expand_cocofmt_eval')`（此刻 `COCOExpandEval` 绑定**原生** COCOeval）。成功后**保留**原生绑定（其它组件用 `from faster_coco_eval import ...` 不受影响）。
- `class TinyPersonMergeEvaluator`（保留注册名）`__init__(self, coco_gt, iou_types, merge_gt_file, tile_w=640, tile_h=576, overlap=50, merge_nms_th=0.5, score_th=0.01, max_det=1000, eval_mr=False, mr_ious=(0.25,0.5,0.75), mr_sizes=('tiny1','tiny2','tiny3','tiny','small','All'))`：
  - `self._COCO, self._COCOExpandEval = _load_official_evaluators()`；
  - `self.gt_coco = self._COCO(merge_gt_file)`（整图 GT，官方评估基准）；
  - `self.img_ids = sorted(int(i) for i in self.gt_coco.getImgIds())`（含全部整图，无预测计 FN）；
  - `self.block_info = {int(tile_id): (tuple(int(v) for v in corner), int(orig_id))}` ← `build_tile_index(coco_gt, tile_w, tile_h, overlap)`；
  - `self.coco_eval={}`、`self._last_results=[]`、`_reset_buffer()`（`self.merged={}`、`self._seen_tiles=set()`）。
  - `cleanup()`：重置 buffer、`coco_eval={}`、`_last_results=[]`。
  - `update(predictions)`：`{tile_id:{boxes(切块xyxy),scores,labels}}` → 跳过非本 tile/已见 tile（卡内 padding 去重）→ `score_th` 过滤 → `boxes[:,0::2]+=x1; boxes[:,1::2]+=y1` 映射回整图 → 按 `orig_id` `torch.cat` 累积（`.detach().to('cpu',...)`）。
  - `synchronize_between_processes()`：DDP `dist_utils.all_gather(list of (orig_id, boxes_np, scores_np, labels_np))` 跨卡合并（跨卡重复框由后续 NMS 自然去除）。
  - `_build_predictions()`：每 orig_id → `torchvision.ops.nms(boxes,scores,merge_nms_th=0.5)` → 按分数截断 `max_det` → `{orig_id:pred}`（**merge 只做一次**）。
  - `_prepare_for_coco_detection()`：转 `[{image_id:orig_id, category_id:labels[k], bbox:xywh, score}]`（`convert_to_xywh`）。
  - `accumulate()`：
    - `predictions=_build_predictions(); results=_prepare_for_coco_detection(predictions); self._last_results=results`；
    - devnull 内 `coco_dt = gt_coco.loadRes(results) if results else self._COCO()`；
    - **官方 AP**（注释标 evaluate_tiny.py 行号出处）：
      ```
      coco_eval = self._COCOExpandEval(
        self.gt_coco, coco_dt, 'bbox',
        ignore_uncertain=True,                              # L233
        use_ignore_attr=True,                               # L135
        use_iod_for_ignore=True,                            # L136/L233
        iod_th_of_iou_f="lambda iou: (2*iou)/(1+iou)",      # L137
        cocofmt_param=dict(evaluate_standard='tiny',        # L139/L233
                           iouThrs=[0.25,0.5,0.75],         # L140
                           maxDets=[200]))                  # L141
      coco_eval.params.imgIds = list(self.img_ids)
      devnull 内 coco_eval.evaluate(); coco_eval.accumulate()  # accumulate 填 .eval(可pickle)
      self.coco_eval['bbox']=coco_eval
      ```
  - `summarize()`：遍历 `coco_eval` → `print("IoU metric: bbox")` + `coco_eval.summarize()`（官方 tiny AP/AR 表，内部设 `.stats`；`stats[0]=AP@0.25/all` 供选 best）；`if self.eval_mr: self._run_mr(self._last_results)`。
  - `_run_mr(results)`：**懒加载** `from huicv.deps.Cityscapes.cityperson_eval import cityperson_eval`（try/except 缺 matplotlib/scipy → 打印提示跳过）；results 写临时 json（`tempfile.NamedTemporaryFile('w',suffix='.json',delete=False)`）；调用 `cityperson_eval(tmp.name, merge_gt_file, CUT_WH=(1,1), ignore_uncertain=True, use_iod_for_ignore=True, use_citypersons_standard=False, iou_ths=list(mr_ious), setup_labels=list(mr_sizes))`（出处 evaluate_tiny.py L167-171, L204-238）。
  - `convert_to_xywh(boxes)` 辅助函数保留。

### 3.4 改 `engine/data/dataset/__init__.py`（导出）
- 在 coco_dataset 导入块之后新增：`from .coco_dataset_tinyperson_tile import TinyPersonTileDataset`（`TinyPersonMergeEvaluator` 原已导出）。无循环依赖。

### 3.5 新建 `configs/dataset/tinyperson_online_detection.yml`
- `evaluator`（type=TinyPersonMergeEvaluator）：`iou_types:['bbox']`、`merge_gt_file: datasets/Tinyperson/annotations/tiny_set_test_all.json`、`tile_w:640 tile_h:576 overlap:50`、`merge_nms_th:0.5`、`score_th:0.01`、`max_det:1000`、`eval_mr:False`。
- `num_classes: 2`（单类 person，0 占位）、`remap_mscoco_category: False`。
- `train_dataloader.dataset`：`CocoDetection, corner_mode:False, img_folder: datasets/Tinyperson/train, ann_file: datasets/Tinyperson/annotations/tiny_set_train_tiles.json`；`transforms:{type:Compose, ops: ~}`（ops 由 base/dataloader.yml 后置 include 填充）。
- `val_dataloader.dataset`：`type: TinyPersonTileDataset, img_folder: datasets/Tinyperson/test, ann_file: .../tiny_set_test_all.json, tile_w/tile_h/overlap`；`transforms ops: ~`；`collate_fn: BatchImageCollateFunction`（val 不设 scales → 关多尺度）。
- 路径默认**仓库相对**（相对 DEIM-main 根，运行期可用 `-u` 覆盖为绝对路径，如 `/root/autodl-tmp/Tinyperson/...`）。

### 3.6 新建 `configs/deim_dfine/dfine_hgnetv2_s_tinyperson_online.yml`
- `__include__`: `['../dataset/tinyperson_online_detection.yml', '../runtime.yml', '../base/dataloader.yml', '../base/optimizer.yml', '../base/dfine_hgnetv2.yml']`。
- `output_dir: ./output/dfine_hgnetv2_s_tinyperson_online`。
- D-FINE-S 结构（与 crop 版/官方一致）：HGNetv2 `B0`/`return_idx:[1,2,3]`/`freeze_at:-1`/`freeze_norm:False`/`use_lab:True`；DFINETransformer `num_layers:3, eval_idx:-1`；HybridEncoder `in_channels:[256,512,1024], hidden_dim:256, depth_mult:0.34, expansion:0.5`；AdamW 三组 params。
- `epoches: 36`（切块迭代量大，起点 36，可调）。
- `train_dataloader`：`total_batch_size:8, num_workers:8, dataset.transforms.policy.epoch:30, collate_fn.scales:[640,640,736,800,800,800], stop_epoch:30, ema_restart_decay:0.9999`（全部 32 对齐；stop_epoch≠0）。
- `val_dataloader`：`total_batch_size:16, num_workers:8`。
- 分辨率三要素（继承 base）：`Resize=[800,800]=base_size=800=eval_spatial_size=[800,800]` 一致。

### 3.7 未改动但已核对
- `engine/core/yaml_config.py`：`implementedEvaluators` 已含 `TinyPersonMergeEvaluator`；`evaluator` 属性 `create('evaluator', global_cfg, coco_gt=get_coco_api_from_dataset(val_dataloader.dataset))`。**无需改**。
- 评测时序/依赖：`det_engine.evaluate()`、`det_solver`（详见 §2.3/§3.3 与 §8）。

---

## 4. 【当前/已完成进度】哪些已验证、哪些还没跑

- ✅ **全部代码与配置写完**（§3 六文件）。
- ✅ **接口静态核对通过**：
  - det_engine/det_solver 对评测器的要求已满足：`accumulate()` 内 `coco_eval.accumulate()` 填 `.eval`（可 pickle，供 `torch.save`）；`summarize()` 内 `coco_eval.summarize()` 填 `.stats`（numpy，供 `stats.tolist()` 与选 best）；`self.coco_eval['bbox']` 存在。
  - COCOExpandEval 不重写 `accumulate`（继承原生纯 Python），已用读码确认。
  - 官方 AP 参数逐一对照 `evaluate_tiny.py` 行号。
  - `Resize = register()(T.Resize)`（torchvision v2）→ 普通 tensor `orig_size` **不被改写**（关键：切块预测经 postprocessor 还原到切块像素坐标）。
  - `get_coco_api_from_dataset` 对 `CocoDetection` 子类返回 `dataset.coco`（整图 COCO）。
  - 训练标签流：`remap_mscoco_category=False` → `labels=category_id=1` → 预测 `category_id` 与整图 GT `person(1)` 匹配。
  - 数据字段/几何已用 PowerShell/StreamReader 核实（§2.2）；`include` 文件均存在。
- ❌ **尚未在真实 Python 环境编译/运行**：本机 `python` 是 Windows Store 桩（`exit=9009`，`C:\Users\19333\AppData\Local\Microsoft\WindowsApps\python.exe`），无法 `py_compile`/跑训练；IDE 语言服务器 `GetProblems` 一直“not ready”。用户已声明**自行在 AutoDL/GPU 平台测试**。

---

## 5. 【剩余待办】按优先级

> 代码层面基本收尾，剩余以**验证/调优**为主（多数在 AutoDL 上做）。

1. **（高）冒烟验证 `--test-only`**：在 AutoDL 装 `pycocotools`（原生）后，用训练好的权重跑一次纯验证，确认：在线切块数与 tile_id 映射一致、坐标回映正确、官方 AP 表正常打印、`.stats`/`.eval` 被 det_solver 正常保存。涉及：`configs/deim_dfine/dfine_hgnetv2_s_tinyperson_online.yml`、`coco_eval_tinyperson.py`。
2. **（高）小规模训练冒烟**：跑 1–2 epoch，确认 train（物理切块 corner_mode=False）与 val（在线切块）都能取数、不 OOM、DDP `all_gather` 正常。
3. **（中）AP 数值与官方 `evaluate_tiny.py` 对齐校验**：同一份 `latest_result.json` 分别用本评测器与官方脚本（`--merge-gt tiny_set_test_all.json --metric ap`）对拍，确认一致。
4. **（中）路径核对**：确认运行环境数据实际位置；若在 `/root/autodl-tmp/...`，用 `-u` 覆盖 `train_dataloader.dataset.img_folder/ann_file`、`val_dataloader.dataset.img_folder/ann_file`、`evaluator.merge_gt_file`。
5. **（低）MR 试跑**：`eval_mr:True` + 装 `matplotlib scipy`，验证 `cityperson_eval` 出 MR。
6. **（低）调参**：`epoches`（36→48~60）、`val total_batch_size`（在线切块迭代量大，可调大加速）、必要时 `-u PostProcessor.num_top_queries` 降内存。

---

## 6. 【已知的坑/注意事项】

- **原生 pycocotools 必需**：`requirements.txt` 无 pycocotools。`_load_official_evaluators()` 会剥离 faster 影子并 import 原生版；缺原生会抛清晰 `ImportError`（提示 `pip install pycocotools`）。若 import 到的仍是 faster 影子（`__file__` 含 `faster_coco_eval`）抛 `RuntimeError`。
- **faster 影子常是 `ModuleType` 无 `__file__`**：所以清影子按**模块名前缀全清**，不要只按 `__file__` 过滤。
- **`corner_mode`**：训练物理切块 json **无 `corner` 字段**，必须 `corner_mode:False`；误设 True 会 `self.coco.imgs[id]['corner']` KeyError。
- **tile 参数两侧必须一致**：`val_dataloader.dataset.{tile_w,tile_h,overlap}` 与 `evaluator.{tile_w,tile_h,overlap}` 必须相同，否则 `build_tile_index` 映射错位。
- **`merge_gt_file` 必须=val 数据集 `ann_file`**：同一整图 GT 文件（`tiny_set_test_all.json`），保证 `img_ids`/`orig_id` 一致。
- **imgIds 含全部整图**：无预测整图计 FN，否则 AP 虚高（已实现 `coco_eval.params.imgIds = self.img_ids`）。
- **`orig_size` 不被 Resize 改写**：整条链依赖“切块预测 = 归一化框 × 切块 orig_size”；若改用会写 `orig_size` 的自定义 Resize 会全错。
- **分辨率三要素一致**：`Resize.size = collate_fn.base_size = eval_spatial_size`（均 800，继承 base 保证）；不一致会 FLOPs 统计/`.deploy()` 崩溃。
- **多尺度尺寸必须 32 整除**：`scales:[640,640,736,800,800,800]` 均满足；否则 HybridEncoder FPN 上采样拼接崩。
- **`stop_epoch` 绝不能为 0**：0 会在 epoch0 触发加载未生成的 `best_stg1.pth` → FileNotFoundError。这里设 30（<epoches=36）。关多尺度应 `base_size_repeat: ~` 而非 stop_epoch=0。
- **DDP padding 重复**：`update` 内卡内 `tile_id` 去重；跨卡重复由 NMS(th=0.5) 自然去除（同块预测完全相同）。
- **MR 懒加载**：`cityperson_eval` 需 matplotlib/scipy；缺失时 `_run_mr` try/except 打印提示跳过，不影响 AP。`cityperson_eval` 会在 CWD 写 `results.txt`（官方行为）。
- **内存**：在线切块后每轮验证迭代量大（~万级块），`score_th=0.01` 过滤 + `max_det=1000` 控内存；必要时降 `num_top_queries`。
- **PowerShell 编码坑**：读大 JSON 用 `System.IO.StreamReader`/`File::ReadAllText`，别用 `Get-Content`（UTF-8 大文件陷阱）。
- **旧离线 evaluator 记忆已过时**：历史 `TinyPersonMergeEvaluator`（依赖 `corner`/`ori_file_name`/`fname2intid`）已被本在线方案**整体替换**，勿据此改回。
- **相对路径**：配置默认 `datasets/Tinyperson/...` 相对 **DEIM-main 根**，需从该根运行或 `-u` 覆盖绝对路径。

---

## 7. 【关键文件清单】（绝对路径）

### 本次新增/修改（续接主战场）
- `c:\Users\19333\Downloads\Dome-DETR-master\DEIM-main\engine\data\dataset\tinyperson_tiling.py`（新建）
- `c:\Users\19333\Downloads\Dome-DETR-master\DEIM-main\engine\data\dataset\coco_dataset_tinyperson_tile.py`（新建）
- `c:\Users\19333\Downloads\Dome-DETR-master\DEIM-main\engine\data\dataset\coco_eval_tinyperson.py`（重写）
- `c:\Users\19333\Downloads\Dome-DETR-master\DEIM-main\engine\data\dataset\__init__.py`（改：导出）
- `c:\Users\19333\Downloads\Dome-DETR-master\DEIM-main\configs\dataset\tinyperson_online_detection.yml`（新建）
- `c:\Users\19333\Downloads\Dome-DETR-master\DEIM-main\configs\deim_dfine\dfine_hgnetv2_s_tinyperson_online.yml`（新建）

### 需参考/核对（未改）
- `...\DEIM-main\engine\data\dataset\coco_dataset.py`（`CocoDetection`、`corner_mode`、`ConvertCocoPolysToMask`、顶部 `init_as_pycocotools()`）
- `...\DEIM-main\engine\data\dataset\coco_eval.py`（标准 `CocoEvaluator`，对照 `.stats`/`.eval` 接口）
- `...\DEIM-main\engine\data\dataset\coco_utils.py`（`get_coco_api_from_dataset`）
- `...\DEIM-main\engine\core\yaml_config.py`（`implementedEvaluators`、evaluator 注入）
- `...\DEIM-main\engine\core\workspace.py`（`register/create/extract_schema`）
- `...\DEIM-main\engine\core\yaml_utils.py`（`__include__` 合并语义）
- `...\DEIM-main\engine\solver\det_engine.py`（`evaluate()` 时序、`orig_size`→postprocessor）
- `...\DEIM-main\engine\solver\det_solver.py`（`stats[0]` 选 best、`torch.save(...eval)`、`stop_epoch` 两阶段）
- `...\DEIM-main\engine\data\transforms\_transforms.py`（`Resize=T.Resize` 不改 orig_size）
- `...\DEIM-main\configs\base\dataloader.yml`（Resize/base_size/ops/collate）
- `...\DEIM-main\configs\base\dfine_hgnetv2.yml`（`eval_spatial_size=[800,800]`）
- `...\DEIM-main\configs\runtime.yml`、`...\configs\base\optimizer.yml`（include 目标）

### 官方评测源（复用、禁止改）
- `...\DEIM-main\tools\huicv\evaluation\evaluate_tiny.py`（规则真源；AP L126-149/L233、MR L167-171/L204-238、merge_nms_th=0.5 L221）
- `...\DEIM-main\tools\huicv\evaluation\expand_cocofmt_eval.py`（`COCOExpandEval`/`ExpandParam` tiny 口径 L12-31、summarize L298-343）
- `...\DEIM-main\tools\huicv\deps\Cityscapes\cityperson_eval.py`（MR）

### 数据
- `...\DEIM-main\datasets\Tinyperson\annotations\tiny_set_test_all.json`（整图 GT / merge_gt / val ann）
- `...\DEIM-main\datasets\Tinyperson\annotations\tiny_set_train_tiles.json`（训练物理切块）
- `...\DEIM-main\datasets\Tinyperson\train\{labeled_images,pure_bg_images}`、`...\test\{labeled_images,pure_bg_images}`

### 计划（只读）
- `C:\Users\19333\AppData\Roaming\QoderCN\SharedClientCache\cache\plans\TinyPerson_Online_Tile_Eval_8d182bdf.md`（禁止编辑）

---

## 8. 【新账号接手后的第一个动作】

**建议第一步：静态复核 + 触发一次真实编译检查（在装好依赖的环境里）。** 因为本机无可用 Python（Windows Store 桩），代码语法/接口虽已人工核对，但**从未在解释器里跑过**。

具体：
1. 先完整读一遍本文件 + `coco_eval_tinyperson.py`、`coco_dataset_tinyperson_tile.py`、`tinyperson_tiling.py`、两个 yml（§7 路径）。
2. 在有 GPU + 依赖的 AutoDL/等价环境：
   ```bash
   pip install pycocotools            # 关键：原生 pycocotools
   # （可选 MR）pip install matplotlib scipy
   python -m py_compile engine/data/dataset/tinyperson_tiling.py \
       engine/data/dataset/coco_dataset_tinyperson_tile.py \
       engine/data/dataset/coco_eval_tinyperson.py engine/data/dataset/__init__.py
   ```
3. 冒烟 `--test-only`（需一个权重；没有先用已存在的 ckpt 或临时训练产物）：
   ```bash
   torchrun --master_port: nproc_per_node=1 train.py -c configs/deim_dfine/dfine_hgnetv2_s_tinyperson_online.yml --test-only -r <ckpt.pth> [-u 覆盖绝对路径]
   ```
   重点看：`[TinyPersonTileDataset] N 整图 -> M 切块` 打印、官方 tiny AP/AR 表、`.eval/.stats` 是否被 det_solver 正常保存、有无 `pip install pycocotools` 类报错。
4. 若报“仍是 faster 影子/缺原生 pycocotools”→ 检查环境是否装了**原生** pycocotools；若报坐标/AP 异常→ 核对 §6 中 tile 参数两侧一致、`merge_gt_file`=`val.ann_file`、`orig_size` 未被自定义 Resize 改写。

> 若新账号同样没有可运行环境：则改为逐行静态审阅 §3 的代码（重点 §3.3 `accumulate` 的官方参数对齐与 `_load_official_evaluators` 影子剥离逻辑），并把运行验证留给用户。
