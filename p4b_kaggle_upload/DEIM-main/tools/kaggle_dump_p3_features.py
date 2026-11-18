# -*- coding: utf-8 -*-
"""
========================================================================================
Kaggle 专用双卡标准学术版：Tesla T4 * 2 (16GB * 2) 多线程并发 (AMP 自动混合精度)
先行 4 大经典基线检验 -> 训练 TileStitchingNet -> Ours 最终汇总对比
========================================================================================
【使用方法】：
  1. Kaggle Notebook 右侧 Settings -> Accelerator 选【GPU T4 x2】；
  2. 在 Notebook 中新建一个 Code Cell；
  3. 将本文件的全部代码直接【全选复制 -> 粘贴】进去；
  4. 点击 Cell 左侧的【Run】运行即可！无需在命令行敲任何命令。

【执行流程优化 (先验后训)】：
  - [Step 1~2] 初始化环境与模型权重，PyTorch 2.6+ 自动兼容；
  - [Step 3] 秒级加载已有测试集特征 (tinyperson_p3_val.pt，786 张大图)；
  - [Step 4 提前检验] 立即运行 4 大经典基线 (Greedy-NMS, Soft-NMS, WBF, SAHI NMM)，
    几秒钟内出表！直接检验输入的候选框坐标与真值是否准确对齐；
  - [Step 5] 加载/提取训练集特征 (tinyperson_p3_train.pt，746 张大图)；
  - [Step 6] 启动 TileStitchingNet 训练拟合 (BatchSize=1000 Tokens)；
  - [Step 7] 运行 Ours 并输出包含 5 种算法的最终完整学术对比大表！
========================================================================================
"""

import os
import sys
import re
import json
import hashlib
from pathlib import Path
import time
import shutil
import threading
import subprocess
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image

# ======================================================================================
#   PyTorch 2.6+ 针对安全反序列化限制的全局自适应修补 (杜绝 UnpicklingError 与 RecursionError)
# ======================================================================================
import importlib
import torch.serialization

# 恢复并锁定底层纯净原生 load 函数 (杜绝在 Jupyter Notebook 中多次重复运行 Cell 导致的猴子补丁递归死循环)
if not hasattr(torch, '_unpatched_native_torch_load') or getattr(torch._unpatched_native_torch_load, '__name__', '') != 'load':
    if getattr(torch.serialization, 'load', None) and getattr(torch.serialization.load, '__name__', '') == 'load':
        torch._unpatched_native_torch_load = torch.serialization.load
    else:
        try:
            importlib.reload(torch.serialization)
            torch._unpatched_native_torch_load = torch.serialization.load
        except Exception:
            torch._unpatched_native_torch_load = torch.serialization.load

def _safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return torch._unpatched_native_torch_load(*args, **kwargs)

_safe_torch_load._is_safe_wrapper = True
torch.load = _safe_torch_load
torch.serialization.load = _safe_torch_load

# ======================================================================================
#                                    用户配置区
# ======================================================================================
# 1. 代码目录：Kaggle 挂载的 DEIM-main 路径
REPO_SRC = '/kaggle/input/datasets/xyjxyjxyj55/deim-tod/DEIM-main'

# 2. 数据集目录：指向内含 Tinyperson/ 的 datasets 目录
DATASET_SRC = '/kaggle/input/datasets/xyjxyjxyj55/tinyperson/datasets'

# 3. 配置文件路径（相对 DEIM-main）
CFG_REL = 'configs/deim_dfine/dfine_hgnetv2_s_tinyperson_online.yml'

# 4. 训练好的权重文件路径（留空 '' 则全自动在 /kaggle/input 下搜索 best_stg2.pth > best_stg1.pth > last.pth）
CKPT_PATH = ''

# 5. 特征复用模式开关 (全局唯一总开关，训练集与测试集统一，不再区分 val/train 子开关):
#    - True : 自动读取特征缓存——优先从你上传到 Kaggle input 的 .pt
#             (如 .../trainvalsample/tinyperson_p3_train.pt / tinyperson_p3_val.pt) 极速加载，
#             跳过耗时的切块推理；仅当任何位置都找不到缓存时才回退到重新提取。
#    - False: 强制从头执行切块前向与 P3 特征提取，重新生成并覆盖保存到 /kaggle/working。
# 【注意】类别标签(target_score/is_leader/target_edge)在提取阶段就写入 .pt。若改动了会影响
# 提取结果的代码(GT 类别、坐标逆缩放、RoI 投影等)，必须置 False 重算一次，否则会读到旧标签缓存。
# 仅调训练超参/损失(不碰提取)时，保持 True 直接复用已上传缓存即可快速迭代。
VAL_FRACTION = 0.20
SPLIT_SEED = 42
POST_SCORE_THRESH = 0.0  # Same output floor; official evaluator applies maxDets=200.
ROI_SEED = 20261112
CACHE_SCHEMA = "stage1-v1"
USE_CACHED_FEATURES = True       # 全局总开关：True=优先读你上传的缓存；False=强制重算并覆盖保存

# 6. 后处理 Attention 序列训练批大小 (Token 级累加)
ATTN_BATCH_SIZE = 1000   # 候选框 Token 级大批次 (累积 1000 个候选框 Token 执行一次平稳参数更新)
# 每轮验证频率：验证(786 张图 × 12 次 AP 匹配 + nested/split 子群召回)是每轮耗时大头(约数分钟)，
# 训练的前向/反向本身很便宜。调大 VAL_FREQ 可显著提速(如 3 = 每 3 轮才验证一次)；
# 代价是 best 权重的挑选粒度变粗(最后一轮仍会强制验证)。想每轮都看 Val_AP 就设 1。
VAL_FREQ = 3

# 7. 精度与吞吐配置
USE_AMP = True           # 启用 AMP 自动混合精度 (等同于 DEIM 训练时的 use_amp: True，权重为 FP32，前向自动混精加速)
CONF_THRESH = 0.15       # 低置信度采掘门槛 (0.05~0.10，确保被切线截断的碎片不被提前滤除)
INTRA_NMS_IOU = 0.70     # 单切块内部 NMS 去重阈值
ROI_DIM = 64             # P3 1x1 RoI 降维后维度
BATCH_PER_GPU = 32       # 每卡 Batch Size (T4 16G 显存充裕，设为 32 吞吐饱和且安全)

# 7.1 强制单卡特征提取 (True: 仅用 cuda:0 单卡串行提取，彻底规避双卡 RoI 投影基不一致问题)
#     原双卡路径下 GPU0/GPU1 各自随机初始化 P3RoIFeatureReducer 的 256->64 正交投影，
#     两卡投影基不同，混合样本后 64 维外观特征处于两个基下、无法被同一网络统一学习。
#     单卡串行虽慢一些，但保证全部样本的 RoI 特征同源同基(严重1修复)。
FORCE_SINGLE_GPU = True

# 8. 最大整图处理数量 (设为 None 表示跑完全部整图；设为 40 可用于先期快速验证)
MAX_IMAGES_LIMIT = None

# 8.1 训练集“切块来源”模式 (当你的数据集只有 train_tiles/ 物理小块、没有训练整图时必开)：
#     True  -> 训练集不再打开整图现切，而是直接读取已切好的 train_tiles/ 小块；
#              用 tiny_set_train_all.json 的整图宽高经 generate_tile_corners(640×576, overlap=50)
#              复现每块 corner，按行优先 k->(r,c) 映射到 {base}__r{R}_c{C}.png；整图 GT 仍取自 train_all.json。
#              (已离线校验：744/746 张整图的复现切块数与磁盘实际切块数逐图完全一致 -> 参数同源，坐标可精确复现)
#     False -> 沿用原逻辑：打开 train/ 整图并用 crop 现场切块(需要真实整图文件存在)。
#              【当前推荐=False】train/ 整图已导入(746/746 可被 resolve_image_path 定位)，
#              此路径与 test 完全同构、样本最全(746 而非 744)、且不依赖 corner 复现假设。
#              仅当再次丢失训练整图、只剩 train_tiles/ 时才改回 True。
#     注意：仅影响训练集；测试集始终读取 test/ 整图(786 张已存在)，不受此开关影响。
TRAIN_FROM_TILES = False
TRAIN_TILES_SUBDIR = 'train_tiles'   # 相对 datasets/Tinyperson 的切块目录名

# 9. 产出文件保存路径
WORK_DIR = '/kaggle/working'
OUT_TRAIN_PT = os.path.join(WORK_DIR, 'tinyperson_p3_train.pt')
OUT_VAL_PT = os.path.join(WORK_DIR, 'tinyperson_p3_val.pt')
OUT_BEST_PTH = os.path.join(WORK_DIR, 'best_tilestitching.pth')

# ======================================================================================
#                                    核心流程实现
# ======================================================================================

def say(msg=''):
    print(msg, flush=True)


def _can_import(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def setup_environment():
    """环境初始化：复制代码至可写区、软链接数据集、安装依赖、自愈代码"""
    say('=' * 80)
    say('>>> [Step 1] 初始化环境与双卡依赖...')
    say('=' * 80)

    is_kaggle = os.path.exists('/kaggle')
    repo_work = os.path.join(WORK_DIR, 'DEIM-main') if is_kaggle else os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if is_kaggle and os.path.isdir(REPO_SRC):
        say(f'复制代码仓库: {REPO_SRC} -> {repo_work} ...')
        shutil.copytree(REPO_SRC, repo_work, dirs_exist_ok=True, ignore=shutil.ignore_patterns(
            'datasets', 'weight', 'output', '__pycache__', '.git', '*.zip', '*.pt'))

    # 数据集软链接
    dst_ds = os.path.join(repo_work, 'datasets')
    src_ds = DATASET_SRC if (DATASET_SRC and os.path.isdir(DATASET_SRC)) else os.path.join(REPO_SRC, 'datasets')
    if is_kaggle and os.path.isdir(src_ds) and not os.path.exists(dst_ds):
        try:
            os.symlink(src_ds, dst_ds)
            say(f'建立数据集软链接: {dst_ds} -> {src_ds}')
        except OSError:
            pass

    # 安装依赖
    deps = [('pycocotools', 'pycocotools'), ('faster_coco_eval', 'faster-coco-eval>=1.6.5')]
    for mod, pip_name in deps:
        if not _can_import(mod):
            say(f'正在安装必需依赖: {pip_name} ...')
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-input', pip_name], check=False)

    tile_dir = os.path.join(repo_work, 'TileStitchingNet')
    for p in (repo_work, tile_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    return repo_work


def find_checkpoint():
    """全自动扫描定位最优权重文件"""
    if CKPT_PATH and os.path.isfile(CKPT_PATH):
        return CKPT_PATH

    search_roots = ['/kaggle/input', WORK_DIR, '.']
    pattern = re.compile(r'^(best_stg2\.pth|best_stg1\.pth|last\.pth)$')
    rank_map = {'best_stg2.pth': 0, 'best_stg1.pth': 1, 'last.pth': 2}
    best_path, best_rank = None, 999

    for root in search_roots:
        if not os.path.exists(root):
            continue
        for cur, _, files in os.walk(root):
            for f in files:
                if pattern.match(f):
                    r = rank_map.get(f, 9)
                    if r < best_rank:
                        best_rank = r
                        best_path = os.path.join(cur, f)

    if best_path:
        say(f'>>> [Step 2] 自动探测到模型权重: {best_path}')
        return best_path
    raise FileNotFoundError('未能找到权重文件 (best_stg2.pth)！请在上方配置区 CKPT_PATH 填入您的权重完整路径。')


def find_annotation_file(tinyperson_dir: str, filename: str) -> Optional[str]:
    """智能查找标注文件路径，支持标准目录与 Kaggle input 全局探测"""
    candidates = [
        os.path.join(tinyperson_dir, 'annotations', filename),
        os.path.join(tinyperson_dir, filename),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    for root, _, files in os.walk('/kaggle/input'):
        if filename in files:
            return os.path.join(root, filename)
    return None


_IMAGE_ROOT_CACHE = {}


def resolve_image_path(tinyperson_dir: str, file_name: str, split: str = 'test') -> Optional[str]:
    """智能解析大图绝对路径 (支持自动探测与路径记忆缓存)"""
    # 1. 优先检查已记忆的 root 目录 (0.0001s 秒级命中)
    if split in _IMAGE_ROOT_CACHE:
        cand = os.path.join(_IMAGE_ROOT_CACHE[split], file_name)
        if os.path.isfile(cand):
            return cand
        cand_base = os.path.join(_IMAGE_ROOT_CACHE[split], os.path.basename(file_name))
        if os.path.isfile(cand_base):
            return cand_base

    base_name = os.path.basename(file_name)
    candidates = [
        os.path.join(tinyperson_dir, split, file_name),
        os.path.join(tinyperson_dir, file_name),
        os.path.join(tinyperson_dir, split, base_name),
        os.path.join(tinyperson_dir, base_name),
        os.path.join(tinyperson_dir, 'train', file_name),
        os.path.join(tinyperson_dir, 'test', file_name),
        os.path.join(tinyperson_dir, 'train', base_name),
        os.path.join(tinyperson_dir, 'test', base_name),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    # 2. 全局探测：在 /kaggle/input 中搜索该文件并记录其根目录
    for root, _, files in os.walk('/kaggle/input'):
        if base_name in files:
            found_path = os.path.join(root, base_name)
            if 'labeled_images' in file_name and root.endswith('labeled_images'):
                _IMAGE_ROOT_CACHE[split] = os.path.dirname(root)
            else:
                _IMAGE_ROOT_CACHE[split] = root
            return found_path

    return None


class P3RoIFeatureReducer(nn.Module):
    """P3 1x1 RoIAlign + 64D 降维投影 (兼容 AMP 混合精度与标准 Float32)"""
    def __init__(self, in_channels: int = 256, out_channels: int = 64, spatial_scale: float = 1.0 / 8.0):
        super().__init__()
        self.spatial_scale = spatial_scale
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(inplace=True),
        )
        nn.init.orthogonal_(self.proj[0].weight)
        nn.init.zeros_(self.proj[0].bias)

    def forward(self, p3: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        if rois.shape[0] == 0:
            return torch.zeros((0, 64), dtype=torch.float32, device=p3.device)
        p3_f = p3.float()
        rois_f = rois.float()
        pooled = torchvision.ops.roi_align(
            p3_f, rois_f, output_size=(1, 1), spatial_scale=self.spatial_scale, sampling_ratio=2
        )
        return self.proj(pooled.flatten(1))


def load_deim_model(repo_work, ckpt_file, device):
    """加载单卡模型实例 (权重保持 FP32 原始精度)"""
    from engine.core import YAMLConfig
    cfg_full = os.path.join(repo_work, CFG_REL)
    cfg = YAMLConfig(cfg_full)
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    ck = torch.load(ckpt_file, map_location='cpu')
    state = ck.get('ema', {}).get('module', ck.get('model', ck))
    cfg.model.load_state_dict(state)
    model = cfg.model.eval().to(device)
    postprocessor = cfg.postprocessor.eval().to(device)
    return cfg, model, postprocessor


def register_p3_hook(model):
    """截获 HybridEncoder P3 特征图"""
    cache = {}
    def hook_fn(mod, inp, out):
        if isinstance(out, (list, tuple)) and len(out) > 0:
            cache['p3'] = out[0]
        elif isinstance(out, dict) and 'p3' in out:
            cache['p3'] = out['p3']
    handle = model.encoder.register_forward_hook(hook_fn)
    return cache, handle


_SHARED_ROI_REDUCERS: dict = {}


def _make_seeded_reducer(device_str):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(ROI_SEED)
        reducer = P3RoIFeatureReducer(in_channels=256, out_channels=ROI_DIM)
    return reducer.to(device_str).eval()


def get_shared_roi_reducer(device_str: str) -> nn.Module:
    """②投影共享：train/val 两次提取(及双卡两卡)必须复用同一随机正交投影 reducer，
    否则 256->64 投影基不同，RoI 外观特征处于不同基下、无法被同一 Stitcher 统一学习。
    按 device 缓存单例：同一 device 的多次调用返回同一个 reducer。"""
    if device_str not in _SHARED_ROI_REDUCERS:
        _SHARED_ROI_REDUCERS[device_str] = _make_seeded_reducer(device_str)
    return _SHARED_ROI_REDUCERS[device_str]


def load_tinyperson_gt(gt_json_path: str):
    """读取整图 COCO GT 字典"""
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    img_map = {im['id']: im for im in coco_data['images']}
    ann_map = defaultdict(list)
    for ann in coco_data['annotations']:
        if ann.get('iscrowd', 0) == 1:
            continue
        ann_map[ann['image_id']].append(ann)
    return img_map, ann_map


# ======================================================================================
#                        GPU 并行工作函数 (AMP 混合精度推理与提取)
# ======================================================================================

def gpu_worker_extract(
    gpu_id: int,
    model: nn.Module,
    postprocessor: nn.Module,
    p3_cache: dict,
    roi_reducer: nn.Module,
    image_ids_subset: List[int],
    img_map: dict,
    ann_map: dict,
    repo_work: str,
    split_name: str,
    conf_thresh: float,
    intra_nms: float,
    roi_dim: int,
    batch_size: int,
    use_amp: bool,
    result_list: list,
    tile_source_subdir: Optional[str] = None,
):
    """单卡 Worker：专职处理指定切块数据集的 FP16/AMP 前向推理与 P3 1x1 RoI 特征提取

    tile_source_subdir: 若指定(如 'train_tiles')则进入“切块来源”模式——不打开整图现切，
    而是直接读取该目录下已切好的物理小块(按 {base}__r{R}_c{C} 命名映射到复现的 corner)。
    """
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        gpu_name = torch.cuda.get_device_name(gpu_id)
        prec_str = 'AMP 混合精度' if use_amp else 'FP32'
        say(f'[Worker GPU {gpu_id}] 启动并在 {gpu_name} ({prec_str}) 上处理【{split_name}】，分配整图: {len(image_ids_subset)} 张...')
    else:
        say(f'[Worker CPU] 启动运行处理【{split_name}】，分配整图: {len(image_ids_subset)} 张...')

    from engine.data.dataset.tinyperson_tiling import generate_tile_corners
    try:
        from TileStitchingNet import CroppedGTLabelAssigner, collect_slice_predictions_from_detector
    except ImportError:
        sys.path.insert(0, os.path.join(repo_work, 'TileStitchingNet'))
        from datasetBuilder import CroppedGTLabelAssigner, collect_slice_predictions_from_detector

    assigner = CroppedGTLabelAssigner(local_iou_thresh=0.5, matching_metric="cropped_iou")
    tinyperson_dir = os.path.join(repo_work, 'datasets', 'Tinyperson')

    # 【切块来源模式】训练集无整图、只有已切好的物理小块时启用：
    # 预先扫描一次切块目录，建立 {整图base: {(r,c): 切块绝对路径}} 索引；之后每张整图用其
    # 宽高经 generate_tile_corners 复现 corner，并按行优先 k->(r,c) 映射到对应切块文件
    # (无损 PNG 与整图 crop 逐像素一致)。整图 GT 仍取自 train_all.json，全局坐标对齐逻辑不变。
    tile_index: Dict[str, Dict[Tuple[int, int], str]] = {}
    tile_root = None
    if tile_source_subdir:
        tile_root = os.path.join(tinyperson_dir, tile_source_subdir)
        if not os.path.isdir(tile_root):
            say(f'[警告] 切块目录不存在: {tile_root}，本次将回退到整图模式。')
            tile_root = None
        else:
            _tile_re = re.compile(r'^(.+)__r(\d+)_c(\d+)\.(png|jpg|jpeg)$', re.IGNORECASE)
            _scan_dirs = [tile_root] + [os.path.join(tile_root, d) for d in os.listdir(tile_root)
                                        if os.path.isdir(os.path.join(tile_root, d))]
            for _sd in _scan_dirs:
                for _fn in os.listdir(_sd):
                    _fp = os.path.join(_sd, _fn)
                    if not os.path.isfile(_fp):
                        continue
                    _m = _tile_re.match(_fn)
                    if _m:
                        tile_index.setdefault(_m.group(1), {})[(int(_m.group(2)), int(_m.group(3)))] = _fp
            say(f'[切块来源模式] 【{split_name}】已索引 {len(tile_index)} 张整图的物理切块，目录={tile_root}')

    t0 = time.perf_counter()
    n_skipped = 0

    for idx, img_id in enumerate(image_ids_subset):
        im_info = img_map[img_id]
        img_w, img_h = int(im_info['width']), int(im_info['height'])
        file_name = im_info['file_name']

        corners = generate_tile_corners(img_w, img_h, tile_w=640, tile_h=576, overlap=50)

        orig_pil = None
        tile_paths = None
        if tile_root is not None:
            base = os.path.splitext(os.path.basename(file_name))[0]
            tiles_for_base = tile_index.get(base, {})
            # 行优先规则网格：首行(y1 相同)的块数即列数 n_cols
            first_y1 = corners[0][1]
            n_cols = 0
            for _cner in corners:
                if _cner[1] == first_y1:
                    n_cols += 1
                else:
                    break
            n_cols = max(n_cols, 1)
            # 严格校验：磁盘切块数须等于复现 corner 数、且每个 (r,c) 都命中，否则说明切块
            # 几何参数/命名与 generate_tile_corners 不同源 -> 跳过该图以防全局坐标错位。
            if len(tiles_for_base) != len(corners):
                n_skipped += 1
                if n_skipped <= 5:
                    say(f'  [跳过] {base}: 磁盘切块 {len(tiles_for_base)} 块 ≠ 复现 corner {len(corners)} 块')
                continue
            tile_paths = []
            _miss = False
            for k in range(len(corners)):
                _p = tiles_for_base.get((k // n_cols, k % n_cols))
                if _p is None:
                    _miss = True
                    break
                tile_paths.append(_p)
            if _miss:
                n_skipped += 1
                continue
        else:
            img_path = resolve_image_path(tinyperson_dir, file_name, split=split_name)
            if img_path is None or not os.path.isfile(img_path):
                n_skipped += 1
                continue
            orig_pil = Image.open(img_path).convert('RGB')

        # 整理整图真值
        gt_boxes_list, gt_classes_list = [], []
        for a in ann_map.get(img_id, []):
            x, y, w, h = a['bbox']
            gt_boxes_list.append([x, y, x + w, y + h])
            # 方案A修复(根因)：DEIM 在 remap_mscoco_category=False / num_classes=2 下，
            # 将 person(category_id=1) 直接训练为 label=1(label 0 为空占位)，后处理输出的
            # 预测 label 亦为 1。GT 类别必须保持 category_id 原值(=1)以与预测 class_ids 对齐，
            # 否则 CroppedGTLabelAssigner 的 class_match 恒为 False -> 无正样本/leader -> loss 恒 0。
            gt_classes_list.append(int(a.get('category_id', 1)))

        gt_boxes = np.array(gt_boxes_list, dtype=np.float32) if gt_boxes_list else np.zeros((0, 4), dtype=np.float32)
        gt_classes = np.array(gt_classes_list, dtype=np.int64) if gt_classes_list else np.zeros((0,), dtype=np.int64)

        patch_predictions = []
        patch_rects = np.array(corners, dtype=np.float32)

        # 批切块前向 (利用 16GB 显存，batch_size=32, AMP 自动混合精度)
        for s_idx in range(0, len(corners), batch_size):
            sel_corners = corners[s_idx : s_idx + batch_size]
            tensors = []
            for k_local, (cx1, cy1, cx2, cy2) in enumerate(sel_corners):
                if tile_paths is not None:
                    # 切块来源模式：直接读已切好的物理小块(无损 PNG，与整图 crop 逐像素一致)
                    crop = Image.open(tile_paths[s_idx + k_local]).convert('RGB')
                else:
                    crop = orig_pil.crop((cx1, cy1, cx2, cy2))
                if crop.size != (640, 576):
                    crop = crop.resize((640, 576), Image.BILINEAR)
                t = torch.from_numpy(np.asarray(crop)).permute(2, 0, 1).float() / 255.0
                tensors.append(t)

            batch = torch.stack(tensors).to(device)
            bs = len(sel_corners)

            with torch.no_grad():
                amp_enabled = use_amp and (device.type == 'cuda')
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    out = model(batch)
                    p3 = p3_cache.get('p3')

                orig_sizes = torch.tensor([[640.0, 576.0]] * bs, device=device, dtype=out['pred_boxes'].dtype)
                results = postprocessor(out, orig_sizes)

                for sub_i in range(bs):
                    r = results[sub_i]
                    b_loc = r['boxes'].detach()
                    sc = r['scores'].detach()
                    lb = r['labels'].detach()

                    keep = (sc >= conf_thresh) & (lb == 1)  # TinyPerson person; class 0 is a placeholder.
                    b_loc = b_loc[keep]
                    sc = sc[keep]
                    lb = lb[keep]

                    if len(b_loc) == 0:
                        patch_predictions.append({
                            'boxes': np.zeros((0, 4), dtype=np.float32),
                            'scores': np.zeros((0,), dtype=np.float32),
                            'class_ids': np.zeros((0,), dtype=np.int64),
                            'roi_features': np.zeros((0, roi_dim), dtype=np.float32),
                        })
                        continue

                    # RoIAlign 在网络输入(640x576)空间进行 -> 用未逆缩放的 b_loc 提特征
                    batch_col = torch.full((len(b_loc), 1), sub_i, dtype=torch.float32, device=device)
                    rois = torch.cat([batch_col, b_loc], dim=-1)
                    roi_64 = roi_reducer(p3, rois).detach().cpu().numpy().astype(np.float32)

                    # ①坐标逆缩放：边缘块被非等比 resize 到 640x576，预测框须按真实块尺寸(rw,rh)
                    # 缩回真实块局部坐标，再由 collect_slice_predictions 加 patch 偏移得到整图坐标，
                    # 否则非等比拉伸块(如 640x544)会系统性错位。RoI 特征已用 640x576 空间提取，不受影响。
                    cx1, cy1, cx2, cy2 = sel_corners[sub_i]
                    rw, rh = float(cx2 - cx1), float(cy2 - cy1)
                    b_out = b_loc
                    if rw != 640.0 or rh != 576.0:
                        inv_scale = torch.tensor(
                            [rw / 640.0, rh / 576.0, rw / 640.0, rh / 576.0],
                            device=b_loc.device, dtype=b_loc.dtype)
                        b_out = b_loc * inv_scale

                    patch_predictions.append({
                        'boxes': b_out.detach().cpu().numpy().astype(np.float32),
                        'scores': sc.detach().cpu().numpy().astype(np.float32),
                        'class_ids': lb.detach().cpu().numpy().astype(np.int64),
                        'roi_features': roi_64,
                    })

        sample = collect_slice_predictions_from_detector(
            patch_predictions=patch_predictions,
            patch_rects=patch_rects,
            img_size=(img_w, img_h),
            conf_thresh=conf_thresh,
            intra_nms_iou=intra_nms,
            tau_d_pixels=5.0,
            gt_boxes=gt_boxes,
            gt_classes=gt_classes,
            assigner=assigner,
            image_id=str(img_id),
        )
        sample.tile_rects_all = patch_rects.copy()
        result_list.append(sample)

        if (idx + 1) % 25 == 0 or (idx + 1) == len(image_ids_subset):
            pct = (idx + 1) / len(image_ids_subset) * 100.0
            total_b = sum(len(s.boxes_global) for s in result_list)
            say(f'  [GPU {gpu_id}|{split_name}] 进度: [{idx+1:03d}/{len(image_ids_subset):03d}] ({pct:5.1f}%) | 耗时: {time.perf_counter()-t0:.1f}s | 框数: {total_b}')

    if tile_root is not None and n_skipped:
        say(f'[Worker GPU {gpu_id}] 【{split_name}】切块来源模式跳过 {n_skipped} 张(磁盘无对应切块或切块数与复现 corner 不一致)。')
    say(f'[Worker GPU {gpu_id}] 【{split_name}】处理完毕！')


def extract_or_load_dataset(
    split_name: str,
    gt_json_path: str,
    cache_save_path: str,
    repo_work: str,
    ckpt_file: str,
    n_gpus: int,
    use_cache: bool,
    max_limit: Optional[int] = None,
    tile_source_subdir: Optional[str] = None,
) -> list:
    """按需加载特征缓存或启动双卡并行提取指定数据集"""
    from pipeline_deim_tilestitching import OfflineDatasetIO

    def sha(path):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                h.update(chunk)
        return h.hexdigest()
    sources = [Path(repo_work)/'tools/kaggle_dump_p3_features.py',
               Path(repo_work)/'TileStitchingNet/datasetBuilder.py']
    configs = sorted((Path(repo_work)/'configs').rglob('*.yml'))
    manifest = dict(schema=CACHE_SCHEMA, gt_sha256=sha(gt_json_path), checkpoint_sha256=sha(ckpt_file),
                    conf_thresh=CONF_THRESH, intra_nms=INTRA_NMS_IOU, roi_dim=ROI_DIM, roi_seed=ROI_SEED,
                    amp=USE_AMP, torch_version=torch.__version__, torchvision_version=torchvision.__version__,
                    source_hashes={str(p.relative_to(repo_work)):sha(p) for p in sources+configs},
                    max_limit=max_limit, tile_source_subdir=tile_source_subdir)

    # 1. 尝试缓存命中
    if use_cache:
        target_file = None
        base_fn = os.path.basename(cache_save_path)
        if os.path.isfile(cache_save_path):
            target_file = cache_save_path
        elif os.path.isfile(cache_save_path + '.npy'):
            target_file = cache_save_path + '.npy'
        else:
            for s_root in ['/kaggle/input', '/kaggle/working', '.']:   # 优先读你上传到 input 的缓存，避免 working 旧副本遮蔽
                if not os.path.exists(s_root):
                    continue
                for root, _, files in os.walk(s_root):
                    if base_fn in files:
                        target_file = os.path.join(root, base_fn)
                        break
                    if (base_fn + '.npy') in files:
                        target_file = os.path.join(root, base_fn + '.npy')
                        break
                if target_file:
                    break

        if target_file and os.path.exists(target_file):
            sidecar = target_file + '.manifest.json'
            valid = False
            if os.path.isfile(sidecar):
                with open(sidecar, encoding='utf-8') as f:
                    valid = json.load(f) == manifest
            if not valid:
                say(f'[缓存失效] {target_file}: 缺少或不匹配 stage1 manifest，将重新提取。')
                target_file = None
        if target_file and os.path.exists(target_file):
            say(f'>>> [命中缓存] 检测到已有【{split_name}】特征: {target_file}')
            say(f'>>> 正在极速加载【{split_name}】特征库...')
            t_l = time.perf_counter()
            samples = OfflineDatasetIO.load_samples(target_file)
            say(f'>>> 【{split_name}】特征加载完成！耗时: {time.perf_counter()-t_l:.2f}s，样本数: {len(samples)}')
            return samples

    # 2. 从头执行提取
    say('=' * 80)
    say(f'>>> [开始特征提取] 正在处理【{split_name}】: {gt_json_path}')
    say('=' * 80)

    img_map, ann_map = load_tinyperson_gt(gt_json_path)
    all_img_ids = sorted(list(img_map.keys()))
    if max_limit is not None and max_limit > 0:
        all_img_ids = all_img_ids[:max_limit]
        say(f'根据配置，截取前 {max_limit} 张整图提取')

    total_images = len(all_img_ids)
    say(f'【{split_name}】整图总数: {total_images}')

    t_start = time.perf_counter()

    if n_gpus >= 2:
        say(f'>>> 激活 Tesla T4 * 2 双卡并发流水线处理【{split_name}】...')
        chunk0 = all_img_ids[0::2]
        chunk1 = all_img_ids[1::2]
        say(f'  - GPU 0 分配: {len(chunk0)} 张 | GPU 1 分配: {len(chunk1)} 张')

        cfg0, model0, post0 = load_deim_model(repo_work, ckpt_file, 'cuda:0')
        p3_cache0, _ = register_p3_hook(model0)
        roi_reducer0 = get_shared_roi_reducer('cuda:0')

        cfg1, model1, post1 = load_deim_model(repo_work, ckpt_file, 'cuda:1')
        p3_cache1, _ = register_p3_hook(model1)
        roi_reducer1 = P3RoIFeatureReducer(in_channels=256, out_channels=ROI_DIM).to('cuda:1').eval()
        roi_reducer1.load_state_dict(roi_reducer0.state_dict())  # ②与 GPU0 共享同一投影基，避免双卡随机基不一致

        samples0, samples1 = [], []
        errors = {}

        def thread_target(gpu_id, model, postprocessor, p3_cache, roi_reducer, subset, out_list):
            try:
                gpu_worker_extract(
                    gpu_id=gpu_id,
                    model=model,
                    postprocessor=postprocessor,
                    p3_cache=p3_cache,
                    roi_reducer=roi_reducer,
                    image_ids_subset=subset,
                    img_map=img_map,
                    ann_map=ann_map,
                    repo_work=repo_work,
                    split_name=split_name,
                    conf_thresh=CONF_THRESH,
                    intra_nms=INTRA_NMS_IOU,
                    roi_dim=ROI_DIM,
                    batch_size=BATCH_PER_GPU,
                    use_amp=USE_AMP,
                    result_list=out_list,
                    tile_source_subdir=tile_source_subdir,
                )
            except Exception:
                import traceback
                errors[gpu_id] = traceback.format_exc()

        t0 = threading.Thread(target=thread_target, args=(0, model0, post0, p3_cache0, roi_reducer0, chunk0, samples0))
        t1 = threading.Thread(target=thread_target, args=(1, model1, post1, p3_cache1, roi_reducer1, chunk1, samples1))

        t0.start()
        t1.start()
        t0.join()
        t1.join()

        if errors:
            for g_id, tb in errors.items():
                say(f'[错误] GPU {g_id} 发生异常:\n{tb}')
            raise RuntimeError(f'双卡提取【{split_name}】发生异常！')

        all_samples = samples0 + samples1
    else:
        dev_str = 'cuda:0' if n_gpus == 1 else 'cpu'
        say(f'>>> 激活单设备流水线 ({dev_str}) 处理【{split_name}】...')
        cfg0, model0, post0 = load_deim_model(repo_work, ckpt_file, dev_str)
        p3_cache0, _ = register_p3_hook(model0)
        roi_reducer0 = get_shared_roi_reducer(dev_str)

        all_samples = []
        gpu_worker_extract(
            gpu_id=0,
            model=model0,
            postprocessor=post0,
            p3_cache=p3_cache0,
            roi_reducer=roi_reducer0,
            image_ids_subset=all_img_ids,
            img_map=img_map,
            ann_map=ann_map,
            repo_work=repo_work,
            split_name=split_name,
            conf_thresh=CONF_THRESH,
            intra_nms=INTRA_NMS_IOU,
            roi_dim=ROI_DIM,
            batch_size=BATCH_PER_GPU,
            use_amp=USE_AMP,
            result_list=all_samples,
            tile_source_subdir=tile_source_subdir,
        )

    say(f'>>> 【{split_name}】提取完毕！耗时: {time.perf_counter()-t_start:.1f}s，有效样本数: {len(all_samples)}')
    OfflineDatasetIO.save_samples(all_samples, cache_save_path)
    with open(cache_save_path + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return all_samples


def print_result_table(results_dict: dict, title: str = "对比结果"):
    """以统一格式打印对比表格"""
    say('\n' + '=' * 96)
    say(f"{title:^96}")
    say('-' * 96)
    say(f"{'Post-Processor Method':<28} | {'AP':<6} | {'AP50':<6} | {'AP75':<6} | {'Nested_Rec%':<11} | {'Split_Rec%':<10} | {'Latency':<8}")
    say('AP = all-area AP50:95; AP25/50/75 and Tiny_AP25/50/75 use official ignore/IOD, maxDets=200.')
    say('-' * 96)
    for name, res in results_dict.items():
        say(f"  Official {name}: AP25={res.get('AP25',float('nan')):.2f}; tiny AP25/50/75={res.get('Tiny_AP25',float('nan')):.2f}/{res.get('Tiny_AP50',float('nan')):.2f}/{res.get('Tiny_AP75',float('nan')):.2f}")
        lat = res.get('Latency_ms', 0.0)
        n_rec = res.get('Nested_Recall50', 0.0)
        s_rec = res.get('Split_Recall75', 0.0)
        say(f"{name:<28} | {res['AP']:<6.2f} | {res['AP50']:<6.2f} | {res['AP75']:<6.2f} | {n_rec:<11.1f} | {s_rec:<10.1f} | {lat:<6.2f}ms")
    say('-' * 96)


# ======================================================================================
#                                    主控流程
# ======================================================================================

def main():
    say('=' * 80)
    say('Tesla T4 * 2 双卡标准学术版：TinyPerson DEIM-S P3 特征提取与 TileStitchingNet 主实验')
    say('=' * 80)

    # 1. 初始化环境与自愈补丁
    repo_work = setup_environment()
    ckpt_file = find_checkpoint()
    os.makedirs(WORK_DIR, exist_ok=True)
    tracked = ['TileStitchingNet/model.py', 'TileStitchingNet/losses.py',
               'TileStitchingNet/trainEval.py', 'TileStitchingNet/stage1.py',
               'TileStitchingNet/datasetBuilder.py', 'tools/kaggle_dump_p3_features.py']
    run_manifest = dict(schema=CACHE_SCHEMA, checkpoint_path=ckpt_file,
                        selection_metric='Tiny_AP50', split_seed=SPLIT_SEED, val_fraction=VAL_FRACTION,
                        candidate_conf=CONF_THRESH, intra_nms=INTRA_NMS_IOU,
                        output_score_thresh=POST_SCORE_THRESH, fallback_nms=.85, max_dets=200,
                        batch_tokens=ATTN_BATCH_SIZE, val_freq=VAL_FREQ, max_images=MAX_IMAGES_LIMIT,
                        source_hashes={p:hashlib.sha256((Path(repo_work)/p).read_bytes()).hexdigest() for p in tracked})
    with open(os.path.join(WORK_DIR, 'stage1_run.json'), 'w', encoding='utf-8') as f:
        json.dump(run_manifest, f, indent=2)

    n_gpus = torch.cuda.device_count()
    say(f'[GPU 状态] 可用显卡数: {n_gpus}')
    for i in range(n_gpus):
        mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        say(f'  - GPU {i}: {torch.cuda.get_device_name(i)} ({mem_gb:.1f} GB 显存)')

    # 严重1修复：按配置强制单卡特征提取，杜绝双卡各自随机初始化的 RoI 投影基不一致
    extract_gpus = min(n_gpus, 1) if FORCE_SINGLE_GPU else n_gpus
    if FORCE_SINGLE_GPU and n_gpus >= 2:
        say(f'[GPU 策略] FORCE_SINGLE_GPU 已启用：检测到 {n_gpus} 张卡，但特征提取仅在 cuda:0 单卡串行执行。')

    tinyperson_dir = os.path.join(repo_work, 'datasets', 'Tinyperson')

    # 2. 定位真训练集与真测试集标注文件
    train_gt_json = find_annotation_file(tinyperson_dir, 'tiny_set_train_all.json')
    test_gt_json = find_annotation_file(tinyperson_dir, 'tiny_set_test_all.json')

    if not test_gt_json or not os.path.isfile(test_gt_json):
        raise FileNotFoundError(f'找不到测试集整图标注: tiny_set_test_all.json')

    if not train_gt_json or not os.path.isfile(train_gt_json):
        raise FileNotFoundError(f'未能定位到您上传的训练集整图标注: tiny_set_train_all.json')

    say(f'[数据标注确认]：')
    say(f'  - 训练集标注 (Train GT): {train_gt_json}')
    say(f'  - 测试集标注 (Test GT) : {test_gt_json}')

    # 3. 加载 / 提取测试集特征 (优先秒级复用已有特征)
    say('=' * 80)
    say('>>> [Step 3] 准备 100% 独立测试集特征库 (用于全基线测试评测)...')
    test_samples = extract_or_load_dataset(
        split_name='test',
        gt_json_path=test_gt_json,
        cache_save_path=OUT_VAL_PT,
        repo_work=repo_work,
        ckpt_file=ckpt_file,
        n_gpus=extract_gpus,
        use_cache=USE_CACHED_FEATURES,
        max_limit=MAX_IMAGES_LIMIT,
    )

    # 导入后处理与评测工具
    try:
        from TileStitchingNet import (
            TileStitchingPredictor,
            train_tile_stitching_net,
            evaluate_postprocessor_on_dataset,
            run_greedy_nms,
            run_soft_nms,
            run_wbf,
            run_sahi_nmm_ios,
        )
    except ImportError:
        sys.path.insert(0, os.path.join(repo_work, 'TileStitchingNet'))
        from trainEval import (
            TileStitchingPredictor,
            train_tile_stitching_net,
            evaluate_postprocessor_on_dataset,
            run_greedy_nms,
            run_soft_nms,
            run_wbf,
            run_sahi_nmm_ios,
        )

    from TileStitchingNet.stage1 import OfficialTinyEvaluator, split_train_validation
    test_evaluator = OfficialTinyEvaluator(test_gt_json)
    train_evaluator = OfficialTinyEvaluator(train_gt_json)

    def eval_baseline(samples, fn):
        b_all, s_all, lats = [], [], []
        for s in samples:
            t0 = time.perf_counter()
            if hasattr(fn, 'predict_sample'):
                b, sc, _ = fn.predict_sample(s)
            else:
                b, sc = fn(s.boxes_global, s.scores)
            lats.append((time.perf_counter() - t0) * 1000.0)
            b_all.append(b)
            s_all.append(sc)
        return test_evaluator(samples, b_all, s_all, lats)

    # 4. 【先行检验输入候选框】：先跑 4 大经典基线
    say('=' * 80)
    say('>>> [Step 4] 先行检验输入候选框：运行 4 大经典基线后处理评测...')
    say('=' * 80)

    total_cand_boxes = sum(len(s.boxes_global) for s in test_samples)
    total_gt_boxes = sum(len(s.gt_boxes) for s in test_samples)
    say(f'[输入框统计诊断]：')
    say(f'  - 测试集大图总数: {len(test_samples)} 张')
    say(f'  - 累计输入候选框数: {total_cand_boxes} 个 (平均每张大图约 {total_cand_boxes/len(test_samples):.1f} 个)')
    say(f'  - 真实 Ground-Truth 框数: {total_gt_boxes} 个 (平均每张大图约 {total_gt_boxes/len(test_samples):.1f} 个)')
    say('>>> 开始运行 4 大经典基线评测，即刻确认输入候选框质量...')

    classic_baselines = {
        'Greedy-NMS (IoU=0.50)': lambda b, s: run_greedy_nms(b, s, iou_thresh=0.50, score_thresh=POST_SCORE_THRESH),
        'Soft-NMS (Gaussian)':   lambda b, s: run_soft_nms(b, s, sigma=0.50, score_thresh=POST_SCORE_THRESH),
        'WBF (IoU=0.55)':        lambda b, s: run_wbf(b, s, iou_thresh=0.55, score_thresh=POST_SCORE_THRESH),
        'SAHI NMM (IoS=0.50)':   lambda b, s: run_sahi_nmm_ios(b, s, ios_thresh=0.50, score_thresh=POST_SCORE_THRESH),
    }

    all_benchmark_results = {}
    for name, fn in classic_baselines.items():
        say(f'  -> 正在计算 {name} ...')
        res = eval_baseline(test_samples, fn)
        all_benchmark_results[name] = res

    print_result_table(all_benchmark_results, title="【阶段一检验】4 大经典基线评测结果 (输入候选框确认)")
    say('>>> 4 大经典基线评测完成！若各项指标正常，仍需结合可视化与诊断检查候选框和坐标！')

    # 5. 加载 / 提取训练集特征 (专用于训练后处理网络)
    say('=' * 80)
    say('>>> [Step 5] 准备 100% 训练集特征库 (专用于 TileStitchingNet 训练)...')
    if TRAIN_FROM_TILES:
        say(f'>>> [Step 5] 训练集数据来源: 直接读取已切好的物理小块 ({TRAIN_TILES_SUBDIR}/)，无需训练整图。')
    train_samples = extract_or_load_dataset(
        split_name='train',
        gt_json_path=train_gt_json,
        cache_save_path=OUT_TRAIN_PT,
        repo_work=repo_work,
        ckpt_file=ckpt_file,
        n_gpus=extract_gpus,
        use_cache=USE_CACHED_FEATURES,
        max_limit=MAX_IMAGES_LIMIT,
        tile_source_subdir=(TRAIN_TILES_SUBDIR if TRAIN_FROM_TILES else None),
    )

    train_samples, internal_val, split_manifest = split_train_validation(
        train_samples, train_gt_json, VAL_FRACTION, SPLIT_SEED)
    with open(os.path.join(WORK_DIR, 'stage1_split.json'), 'w', encoding='utf-8') as f:
        json.dump(split_manifest, f, indent=2)
    say(f'内部验证集: {len(internal_val)} 张；仅由训练集划分，按 Tiny_AP50 选最优轮次。')

    say('=' * 80)
    say(f'>>> [数据集严格学术划分确认]：')
    say(f'  - 训练集整图数: {len(train_samples)} 张 (100% 纯训练集，无测试数据掺杂)')
    say(f'  - 测试集整图数: {len(test_samples)} 张 (100% 纯测试集，全量测试评测，不参与轮次选择)')
    say('=' * 80)

    # 6. 后处理模型极速训练与最优权重保存 (支持 ATTN_BATCH_SIZE=1000)
    say('=' * 80)
    say(f'>>> [Step 6] 启动 TileStitchingNet 训练拟合 (BatchSize={ATTN_BATCH_SIZE} Tokens)...')
    say('=' * 80)

    t_tr0 = time.perf_counter()
    best_model, history = train_tile_stitching_net(
        train_samples=train_samples, val_samples=internal_val, roi_dim=ROI_DIM,
        d_model=64, num_layers=2, epochs=18, lr=3e-3, batch_size=ATTN_BATCH_SIZE,
        lambda_stitch=2.0, lambda_edge=1.0, val_freq=VAL_FREQ,
        save_best_path=OUT_BEST_PTH, verbose=True, evaluator=train_evaluator,
        selection_metric='Tiny_AP50', prediction_score_thresh=POST_SCORE_THRESH)
    with open(os.path.join(WORK_DIR, 'stage1_history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    say(f'后处理网络训练完成！耗时: {time.perf_counter() - t_tr0:.2f} 秒')

    # 7. 运行 Ours 并汇总完整对比表
    say('=' * 80)
    say(f'>>> [Step 7] 运行 Ours (TileStitchingNet) 并汇总完整学术对比表...')
    say('=' * 80)

    predictor = TileStitchingPredictor(model=best_model, score_thresh=POST_SCORE_THRESH, fallback_nms_iou=0.85)
    say('  -> 正在计算 Ours (TileStitchingNet) ...')
    ours_res = eval_baseline(test_samples, predictor)
    all_benchmark_results['Ours (TileStitchingNet)'] = ours_res

    print_result_table(all_benchmark_results, title=f"【最终汇总】全基线方法在 {len(test_samples)} 张全量测试大图上的综合对比表")

    with open(os.path.join(WORK_DIR, 'stage1_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(all_benchmark_results, f, indent=2)

    greedy_ap = all_benchmark_results.get('Greedy-NMS (IoU=0.50)', {}).get('AP', 0.0)
    ours_ap = ours_res.get('AP', 0.0)
    say(f'>>> 效果提升摘要：Ours 相比传统 Greedy-NMS 的 AP 绝对提升 +{ours_ap - greedy_ap:.2f} 个百分点！')

    say('\n' + '=' * 80)
    say('>>> 标准学术主实验圆满完成！输出文件已就绪：')
    say(f'  1. 训练集离线特征库: {OUT_TRAIN_PT}')
    say(f'  2. 测试集离线特征库: {OUT_VAL_PT}')
    say(f'  3. 最佳后处理权重:   {OUT_BEST_PTH}')
    say('在 Kaggle 右侧 Output 面板可一键下载！')
    say('=' * 80)


if __name__ == '__main__':
    main()
