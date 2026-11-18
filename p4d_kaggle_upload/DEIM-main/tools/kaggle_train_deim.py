# -*- coding: utf-8 -*-
"""
Kaggle 训练脚本：DEIM-main 在 AI-TOD-v2 / VisDrone 上的训练（Script 类型 Kernel，非 Notebook）

使用方法：
  1. Kaggle -> Create -> New Kernel -> 选择 "Script"（不要选 Notebook），把本文件内容整体粘贴进去；
  2. Kernel 设置中打开 Internet（pip 安装依赖、下载 HGNetv2 骨干预训练权重需要）；
  3. 修改下方第 0 节常量（DATASET / MODEL / EPOCHS ...），Run All 即可。
  4. 换另一个数据集训练时：改 DATASET 后在新会话中再跑一遍（/kaggle/working 每次会话重置）。

脚本自动完成：
  a. 将只读区 /kaggle/input/.../deim-tod/DEIM-main 复制到可写区 /kaggle/working/DEIM-main；
  b. pip 安装 faster-coco-eval / faster-coco-eval-aitod / calflops；
  c. 给 _transforms.py 打 torchvision 兼容补丁（新版 forward() 调 transform() 而非 _transform()，
     否则处理第一个 batch 就 NotImplementedError）；
  d. VisDrone：复用 Dome-DETR tools/dataset/visdrone2coco.py 的逻辑，把 VisDrone txt 标注
     转成 COCO 格式 json（写到 /kaggle/working/visdrone_coco/）；AITOD 本身已是 COCO json，无需转换；
  e. 在 working 副本中生成 Kaggle 专用 yml（数据集路径 + RESOLUTION 分辨率的 DEIM 训练配置，
     并按 USE_MOSAIC / USE_MIXUP / MULTI_SCALE 开关组装增强链）；
  f. 按 GPU 数量用 torchrun 启动训练，输出（checkpoint/日志）在
     /kaggle/working/DEIM-main/output/deim_hgnetv2_<m>_<ds>_kaggle/。
"""

import os
import sys
import json
import glob
import shutil
import subprocess
import urllib.request

from PIL import Image

# ==================== 0. 用户配置 ====================
DATASET = "aitod"          # "aitod" | "visdrone" | "both"(同一会话顺序训练，注意时长)
MODEL = "s"                # "s" | "m" | "l"
EPOCHS = 132               # DEIM 官方配方总 epoch（120 + 4n）；超时可减小
BATCH_PER_GPU = 4          # 每卡训练 batch（S@800 用 4；M 建议 2，L 建议 1）
NUM_WORKERS = 2            # 每个进程的 dataloader worker 数
PRETRAINED_BACKBONE = True # 下载 HGNetv2 stage1 预训练权重（需 Internet）

# ---- 分辨率 / 数据增强（默认值针对 AITOD、VisDrone 这类小目标数据集 + Kaggle T4 显存）----
RESOLUTION = 800           # 训练/评估分辨率：eval_spatial_size = base_size = Resize 三者会同步用它。
                           # 显存吃紧可降到 640（DEIM 原生分辨率，更省显存/更快，但极小目标 AP 可能略降）
USE_MOSAIC = False         # 四图拼接会把每张子图有效分辨率腰斩，对微小目标不利，默认关（对齐 Dome-DETR）。
                           # 关闭时脚本自动把 mosaic_prob 置 0，避免 ZoomOut/IoUCrop 被随机跳过
USE_MIXUP = False          # 批内图像混合，同样会稀释微小目标，默认关
MULTI_SCALE = True         # 多尺度：训练时每个 batch 从 MULTI_SCALE_SIZES 随机取一尺寸 resize；放大到 800 时 T4 易 OOM。
                           # 关闭用 base_size_repeat=~（None）实现，绝不能用 stop_epoch=0（那是两阶段边界）
MULTI_SCALE_SIZES = [640, 736, 800, 800, 800, 800, 864, 960]  # 8 项：800 出现 4 次→被选中概率 50%，640/736/864/960 各 12.5%。
                           # 仅 MULTI_SCALE=True 生效；运行时注入 working 副本的 generate_scales（见 patch_multiscale）。
                           # 【硬性约束】每个尺寸必须能被 32 整除（本列表已满足：640/736/800/864/960 = 20/23/25/27/30 ×32），
                           # 否则 FPN 把 stride32 上采样2倍与 stride16 拼接时 H/W 对不上会崩（如 720：Expected 46 got 45）。
                           # ⚠ 864/960 是大图（960²=800² 的 1.44 倍、864²=1.17 倍），T4+AI-TOD 密集小目标下极易 OOM；
                           #   若报 torch.OutOfMemoryError(exitcode 1)，把 BATCH_PER_GPU 降到 2，或删掉 960/864。
NO_AUG_EPOCHS = 12         # 两阶段训练末段关增强的微调 epoch 数；EPOCHS<=此值时自动退化为单阶段

INPUT_DEIM = "/kaggle/input/datasets/xyjxyjxyj55/deim-tod/DEIM-main"
INPUT_AITOD = "/kaggle/input/datasets/ngtininh/ai-tod-v2"
INPUT_VISDRONE = "/kaggle/input/datasets/banuprasadb/visdrone-dataset"

WORK = "/kaggle/working"
REPO = os.path.join(WORK, "DEIM-main")
VISDRONE_COCO_DIR = os.path.join(WORK, "visdrone_coco")

BACKBONE_TAG = {"s": "B0", "m": "B2", "l": "B4"}
BB_URL = ("https://github.com/Peterande/storage/releases/download/dfinev1.0/"
          "PPHGNetV2_%s_stage1.pth")


def run(cmd, cwd=None):
    print("[CMD]", " ".join(cmd) if isinstance(cmd, (list, tuple)) else cmd, flush=True)
    subprocess.run(cmd, cwd=cwd, check=True, shell=isinstance(cmd, str))


# ==================== 1. 复制代码到 Working 并装依赖 ====================
def setup_repo():
    if not os.path.isdir(REPO):
        print("copy %s -> %s ..." % (INPUT_DEIM, REPO), flush=True)
        shutil.copytree(INPUT_DEIM, REPO, dirs_exist_ok=True)
    try:
        import faster_coco_eval  # noqa: F401
        import faster_coco_eval_aitod  # noqa: F401
        import calflops  # noqa: F401
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q",
             "faster-coco-eval>=1.6.5", "faster-coco-eval-aitod", "calflops"])


def patch_transforms():
    """给 engine/data/transforms/_transforms.py 的自定义 Transform 补 transform() 兼容方法（幂等）。

    新版 torchvision(>=0.17) 的 Transform.forward() 调用 transform()，而 DEIM 的
    ConvertBoxes / ConvertPILImage / PadToSize 只重写了旧接口 _transform()，会在处理第一个
    batch 时抛 NotImplementedError。仓库里 mixup.py 已有同样的 shim，这里给 _transforms.py 补齐。
    """
    path = os.path.join(REPO, "engine", "data", "transforms", "_transforms.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if "def transform(self, inpt" in src:
        print("transforms already patched, skip", flush=True)
        return
    old = "    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:"
    shim = (
        "    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:\n"
        "        # 兼容新版 torchvision：forward() 调 transform() 而非 _transform()\n"
        "        if isinstance(inpt, self._transformed_types):\n"
        "            return self._transform(inpt, params)\n"
        "        return inpt\n"
        "\n"
    )
    n = src.count(old)
    if n == 0:
        print("WARN: 未找到 _transform 签名，shim 未注入；若训练报 NotImplementedError "
              "请手动检查 _transforms.py", flush=True)
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace(old, shim + old))
    print("patched %d custom Transform(s) with transform() shim" % n, flush=True)


def patch_multiscale(sizes):
    """把 working 副本 engine/data/dataloader.py 的 generate_scales 短路为返回自定义尺度列表（幂等）。

    DEIM 原生 generate_scales 只按 base_size 生成 0.75x~1.25x、32 对齐的对称列表，无法表达
    [640,640,720,800] 这种自定义（且靠重复元素调权重）的组合，故在函数体首行注入 return 短路它。
    仅当 MULTI_SCALE=True（base_size_repeat!=None）时 generate_scales 才会被调用，故不影响关闭态。
    """
    path = os.path.join(REPO, "engine", "data", "dataloader.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    marker = "# KAGGLE_CUSTOM_MULTISCALE"
    if marker in src:
        print("generate_scales already patched, skip", flush=True)
        return
    anchor = "def generate_scales(base_size, base_size_repeat):"
    if anchor not in src:
        raise RuntimeError("未找到 generate_scales 定义，无法注入自定义多尺度")
    inject = "%s\n    %s\n    return %s" % (anchor, marker, list(sizes))
    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace(anchor, inject, 1))
    print("patched generate_scales -> %s" % (list(sizes),), flush=True)


def download_backbone():
    if not PRETRAINED_BACKBONE:
        return
    # 先单进程下载好，规避 hgnetv2.py 中非 rank0 分支 torch.load(目录) 的问题
    tag = BACKBONE_TAG[MODEL]
    dst_dir = os.path.join(REPO, "weight", "hgnetv2")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "PPHGNetV2_%s_stage1.pth" % tag)
    if not os.path.exists(dst):
        print("downloading backbone:", BB_URL % tag, flush=True)
        urllib.request.urlretrieve(BB_URL % tag, dst)
    else:
        print("backbone exists:", dst, flush=True)


# ==================== 2. 数据集路径探测 / VisDrone 转 COCO ====================
def _pick(cands, what):
    for p in cands:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError("cannot locate %s, candidates: %s" % (what, cands))


def find_aitod_paths():
    root = INPUT_AITOD
    aitod = os.path.join(root, "aitod") if os.path.isdir(os.path.join(root, "aitod")) else root
    ann_dir = os.path.join(aitod, "annotations")
    img_root = os.path.join(aitod, "images")
    anns = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
    train_ann = next((p for p in anns if "trainval" in os.path.basename(p)),
                     next((p for p in anns if "train" in os.path.basename(p)), None))
    val_ann = next((p for p in anns if "test" in os.path.basename(p)),
                   next((p for p in anns if "val" in os.path.basename(p)), None))
    train_img = _pick([os.path.join(img_root, d) for d in ("trainval", "train")], "aitod train images")
    val_img = _pick([os.path.join(img_root, d) for d in ("test", "val")], "aitod val images")
    _check_coco(train_ann, train_img, "aitod-train")
    _check_coco(val_ann, val_img, "aitod-val")
    return train_img, train_ann, val_img, val_ann


def _check_coco(ann_file, img_folder, tag):
    with open(ann_file) as f:
        meta = json.load(f)
    n_img, n_ann = len(meta["images"]), len(meta["annotations"])
    max_cat = max(c["id"] for c in meta["categories"])
    f0 = os.path.join(img_folder, meta["images"][0]["file_name"])
    if not os.path.exists(f0):
        raise FileNotFoundError("[%s] first image not found under %s (got %s)" % (tag, img_folder, f0))
    print("[%s] images=%d anns=%d max_category_id=%d ok" % (tag, n_img, n_ann, max_cat), flush=True)


def find_visdrone_root():
    cands = [os.path.join(INPUT_VISDRONE, "VisDrone_Dataset"), INPUT_VISDRONE]
    cands += sorted(glob.glob(os.path.join(INPUT_VISDRONE, "*", "VisDrone2019-DET-train"), recursive=False))
    cands = [os.path.dirname(p) if p.endswith("VisDrone2019-DET-train") else p for p in cands]
    for p in cands:
        if os.path.isdir(os.path.join(p, "VisDrone2019-DET-train", "annotations")):
            return p
    raise FileNotFoundError("VisDrone root not found under %s" % INPUT_VISDRONE)


VISDRONE_CATEGORIES = [
    {"id": 1, "name": "pedestrian", "supercategory": "nark"},
    {"id": 2, "name": "people", "supercategory": "nark"},
    {"id": 3, "name": "bicycle", "supercategory": "nark"},
    {"id": 4, "name": "car", "supercategory": "nark"},
    {"id": 5, "name": "van", "supercategory": "nark"},
    {"id": 6, "name": "truck", "supercategory": "nark"},
    {"id": 7, "name": "tricycle", "supercategory": "nark"},
    {"id": 8, "name": "awning-tricycle", "supercategory": "nark"},
    {"id": 9, "name": "bus", "supercategory": "nark"},
    {"id": 10, "name": "motor", "supercategory": "nark"},
]


def visdrone2coco(src_root, out_dir):
    """与 Dome-DETR tools/dataset/visdrone2coco.py 同逻辑：
    丢弃 category 11(others)，保留 category 0(ignored regions) 并置 iscrowd/ignore=1，
    供 VisdroneCocoEvaluator 在评测时屏蔽 ignore 区域。"""
    os.makedirs(out_dir, exist_ok=True)
    outs = {}
    for mode in ("train", "val"):
        dst = os.path.join(out_dir, "VisDrone2019-DET_%s_coco.json" % mode)
        outs[mode] = dst
        if os.path.exists(dst):
            print("skip converted:", dst, flush=True)
            continue
        ann_dir = os.path.join(src_root, "VisDrone2019-DET-%s" % mode, "annotations")
        img_dir = os.path.join(src_root, "VisDrone2019-DET-%s" % mode, "images")
        images, annotations = [], []
        id_num = 0
        names = sorted(f for f in os.listdir(ann_dir) if f.endswith(".txt"))
        for idx, fn in enumerate(names):
            name = fn[:-4]
            with Image.open(os.path.join(img_dir, name + ".jpg")) as im:
                w, h = im.size
            images.append({"file_name": name + ".jpg", "height": h, "width": w, "id": idx})
            with open(os.path.join(ann_dir, fn)) as f:
                for line in f:
                    line = line.strip().rstrip(",")
                    if not line:
                        continue
                    v = [int(x) for x in line.split(",")]
                    cat = v[5]
                    if cat == 11:          # others: 直接丢弃
                        continue
                    bbox = [v[0], v[1], v[2], v[3]]
                    annotations.append({
                        "image_id": idx, "bbox": bbox, "category_id": cat, "id": id_num,
                        "iscrowd": int(cat == 0), "ignore": int(cat == 0),
                        "segmentation": [], "area": bbox[2] * bbox[3],
                    })
                    id_num += 1
            if (idx + 1) % 1000 == 0:
                print("  %s: %d/%d" % (mode, idx + 1, len(names)), flush=True)
        with open(dst, "w") as fo:
            json.dump({"images": images, "annotations": annotations,
                       "categories": VISDRONE_CATEGORIES}, fo)
        print("[%s] images=%d anns=%d -> %s" % (mode, len(images), len(annotations), dst), flush=True)
    return outs


def prepare_visdrone_paths():
    src = find_visdrone_root()
    outs = visdrone2coco(src, VISDRONE_COCO_DIR)
    train_img = os.path.join(src, "VisDrone2019-DET-train", "images")
    val_img = os.path.join(src, "VisDrone2019-DET-val", "images")
    return train_img, outs["train"], val_img, outs["val"]


# ==================== 3. 生成 Kaggle 专用 yml ====================
DATASET_YML = """task: detection

evaluator:
  type: __EVALUATOR__
  iou_types: ['bbox', ]

num_classes: __NUM_CLASSES__
remap_mscoco_category: False

eval_spatial_size: [__RES__, __RES__] # h w，必须与 base_size / Resize 一致，否则 deploy/eval 崩溃

train_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: __TRAIN_IMG__
    ann_file: __TRAIN_ANN__
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: True
  num_workers: 4
  drop_last: True
  collate_fn:
    type: BatchImageCollateFunction

val_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: __VAL_IMG__
    ann_file: __VAL_ANN__
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: False
  num_workers: 4
  drop_last: False
  collate_fn:
    type: BatchImageCollateFunction
"""


def _train_ops_yaml(res, use_mosaic):
    """按分辨率 / 是否用 Mosaic 生成 DEIM 签名的训练增强链（YAML 片段，已含缩进）。"""
    lines = []
    if use_mosaic:
        # Mosaic output_size = 最终尺寸 / 2（对齐 DEIM 官方 320->640 的比例）
        lines.append(
            "        - {type: Mosaic, output_size: %d, rotation_range: 10, "
            "translation_range: [0.1, 0.1], scaling_range: [0.5, 1.5], probability: 1.0, "
            "fill_value: 0, use_cache: False, max_cached_images: 50, random_pop: True}" % (res // 2))
    lines += [
        "        - {type: RandomPhotometricDistort, p: 0.5}",
        "        - {type: RandomZoomOut, fill: 0}",
        "        - {type: RandomIoUCrop, p: 0.8}",
        "        - {type: SanitizeBoundingBoxes, min_size: 1}",
        "        - {type: RandomHorizontalFlip}",
        "        - {type: Resize, size: [%d, %d], }" % (res, res),
        "        - {type: SanitizeBoundingBoxes, min_size: 1}",
        "        - {type: ConvertPILImage, dtype: 'float32', scale: True}",
        "        - {type: ConvertBoxes, fmt: 'cxcywh', normalize: True}",
    ]
    return "\n".join(lines)


def _build_model_yml(model, ds, res, use_mosaic, multi_scale):
    """生成模型/训练 yml。

    include 直接复用官方 deim_hgnetv2_<m>_coco.yml（已含 dfine 模型 + base/deim.yml + DEIM-S 配方：
    optimizer / epoches132 / flat_epoch64 / policy.epoch[4,64,120] / stop_epoch120），再用本文件覆盖数据集与增强。
    关键不变量：
    - eval_spatial_size / base_size / Resize 三者都 = res，避免 deploy 模式 pos_embed 尺寸不匹配；
    - 关 Mosaic 时 mosaic_prob 必须置 0，且 policy.ops 去掉 'Mosaic'；
    - 关多尺度用 base_size_repeat: ~（None），不动 stop_epoch（它是两阶段边界，设 0 会去找 best_stg1.pth）。
    """
    policy_ops = ("['Mosaic', 'RandomPhotometricDistort', 'RandomZoomOut', 'RandomIoUCrop']"
                  if use_mosaic else
                  "['RandomPhotometricDistort', 'RandomZoomOut', 'RandomIoUCrop']")
    return """__include__: [
  './deim_hgnetv2_%(model)s_coco.yml',
  '../dataset/%(ds)s_detection_kaggle.yml',
]

output_dir: ./output/deim_hgnetv2_%(model)s_%(ds)s_kaggle

HGNetv2:
  local_model_dir: ./weight/hgnetv2/

train_dataloader:
  dataset:
    transforms:
      ops:
%(train_ops)s
      mosaic_prob: %(mosaic_prob)s
      policy:
        ops: %(policy_ops)s
  collate_fn:
    base_size: %(res)d
    base_size_repeat: %(bsr)s

val_dataloader:
  dataset:
    transforms:
      ops:
        - {type: Resize, size: [%(res)d, %(res)d], }
        - {type: ConvertPILImage, dtype: 'float32', scale: True}
""" % {
        "model": model, "ds": ds, "res": res,
        "train_ops": _train_ops_yaml(res, use_mosaic),
        "mosaic_prob": "0.5" if use_mosaic else "0",
        "policy_ops": policy_ops,
        "bsr": "2" if multi_scale else "~",
    }

DS_META = {
    "aitod": {"evaluator": "AitodCocoEvaluator", "num_classes": 9},    # 8 类, category_id=1~8
    "visdrone": {"evaluator": "VisdroneCocoEvaluator", "num_classes": 12},  # 10 类, category_id=1~10
}


def write_configs(ds, paths):
    train_img, train_ann, val_img, val_ann = paths
    ds_yml = (DATASET_YML
              .replace("__EVALUATOR__", DS_META[ds]["evaluator"])
              .replace("__NUM_CLASSES__", str(DS_META[ds]["num_classes"]))
              .replace("__RES__", str(RESOLUTION))
              .replace("__TRAIN_IMG__", train_img)
              .replace("__TRAIN_ANN__", train_ann)
              .replace("__VAL_IMG__", val_img)
              .replace("__VAL_ANN__", val_ann))
    ds_path = os.path.join(REPO, "configs", "dataset", "%s_detection_kaggle.yml" % ds)
    with open(ds_path, "w") as f:
        f.write(ds_yml)

    model_yml = _build_model_yml(MODEL, ds, RESOLUTION, USE_MOSAIC, MULTI_SCALE)
    model_path = os.path.join(REPO, "configs", "deim_dfine",
                              "deim_hgnetv2_%s_%s_kaggle.yml" % (MODEL, ds))
    with open(model_path, "w") as f:
        f.write(model_yml)
    print("written:", ds_path, flush=True)
    print("written:", model_path, flush=True)
    return model_path


# ==================== 4. 启动训练 ====================
def train(cfg_rel):
    import torch
    n_gpu = torch.cuda.device_count() or 1
    print("GPUs =", n_gpu, flush=True)

    # 两阶段训练：末段 NO_AUG_EPOCHS 个 epoch 关增强微调；到达 stop_epoch 时 det_solver 会加载
    # best_stg1.pth 并重启 EMA。若 EPOCHS 太小则退化为单阶段——令 stop_epoch=EPOCHS，使循环内
    # epoch==stop_epoch 永不成立，从而不依赖 best_stg1.pth（否则会 FileNotFoundError）。
    if EPOCHS > NO_AUG_EPOCHS:
        stop, no_aug = EPOCHS - NO_AUG_EPOCHS, NO_AUG_EPOCHS
    else:
        stop, no_aug = EPOCHS, 0
    flat = 4 + stop // 2
    if flat >= stop:                 # EPOCHS 过小时防止 flat 越过 stop，保证 policy.epoch=[4,flat,stop] 有序
        flat = max(stop - 1, 1)
    updates = [
        "train_dataloader.total_batch_size=%d" % (BATCH_PER_GPU * n_gpu),
        "val_dataloader.total_batch_size=%d" % (2 * BATCH_PER_GPU * n_gpu),
        "train_dataloader.num_workers=%d" % NUM_WORKERS,
        "val_dataloader.num_workers=%d" % NUM_WORKERS,
        "epoches=%d" % EPOCHS,
        "flat_epoch=%d" % flat,
        "no_aug_epoch=%d" % no_aug,
        "train_dataloader.dataset.transforms.policy.epoch=[4,%d,%d]" % (flat, stop),
        "train_dataloader.collate_fn.stop_epoch=%d" % stop,
        "train_dataloader.collate_fn.mixup_prob=%s" % ("0.5" if USE_MIXUP else "0"),
        "train_dataloader.collate_fn.mixup_epochs=[4,%d]" % flat,
    ]
    if not PRETRAINED_BACKBONE:
        updates.append("HGNetv2.pretrained=False")

    # 缓解显存碎片：新版 torch 用 PYTORCH_ALLOC_CONF，旧别名 PYTORCH_CUDA_ALLOC_CONF 一并设
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cmd = [sys.executable, "-m", "torch.distributed.run",
           "--nproc_per_node", str(n_gpu), "--master_port", "29500",
           "train.py", "-c", cfg_rel, "--use-amp", "--seed", "0",
           "-u"] + updates
    run(cmd, cwd=REPO)


def main():
    setup_repo()
    patch_transforms()                 # 兼容新版 torchvision，否则第一个 batch 就 NotImplementedError
    if MULTI_SCALE:
        patch_multiscale(MULTI_SCALE_SIZES)   # 注入自定义多尺度 [640,640,720,800]
    download_backbone()
    datasets = ["aitod", "visdrone"] if DATASET == "both" else [DATASET]
    for ds in datasets:
        paths = find_aitod_paths() if ds == "aitod" else prepare_visdrone_paths()
        cfg = write_configs(ds, paths)
        train(os.path.relpath(cfg, REPO))


if __name__ == "__main__":
    main()
