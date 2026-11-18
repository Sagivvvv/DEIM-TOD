# -*- coding: utf-8 -*-
"""
Kaggle 双卡 DDP 训练启动器：TinyPerson 离线预切块 DEIM / D-FINE-S
====================================================================================
仓库代码已含全部修复，本脚本只做四件事：复制代码 → 装依赖 → 下骨干权重 → torchrun 训练。

运行前（Kaggle Script/Notebook）：
  1. Settings -> Internet 打开（需下载骨干权重）；
  2. Accelerator 选 GPU T4 x2；
  3. Input 挂载**两个** dataset（代码与数据集分开上传）：
     - 代码：含 DEIM-main 的 dataset（路径见 REPO_SRC，须含 train.py）；
     - 数据集：单独上传的 TinyPerson dataset（路径见 DATASET_SRC，指向其**内层 datasets/ 目录**），
       内含 Tinyperson/{annotations, train_tiles, test_tiles}（离线预切块；train/test_tiles 与
       tiny_set_train_tiles.json / tiny_set_test_tiles.json，test_tiles 由 tile_tinyperson_test.py 生成）。
     若仍用旧的一体式上传（代码 dataset 内自带 datasets/），把 DATASET_SRC 留空('')即自动回退。
直接 Run 即可。

产出：/kaggle/working/DEIM-main/output/dfine_hgnetv2_s_tinyperson_online/
  best_stg2.pth(最终) / best_stg1.pth / last.pth(续训用)

排障：
  - 显存 OOM        -> 调低 BATCH_PER_GPU（探测表：每卡16覆盖全5尺度，最大档9.65GB<13.11GB）
  - 12h 被中断      -> 把 RESUME_FROM 填成 last.pth 路径，重 Run 续训
  - 骨干下载失败    -> Kaggle 会自动回退在线下载（需 Internet）；或改 PRETRAINED_BACKBONE=False 随机初始化
"""

import os
import sys
import shutil
import subprocess
import urllib.request

# ==================== 用户配置（通常只改这里）====================
# 代码 dataset：挂载后 DEIM-main 的完整路径（须含 train.py）
REPO_SRC = '/kaggle/input/datasets/xyjxyjxyj55/deim-tod/DEIM-main'
# 数据集 dataset：单独上传的 TinyPerson，指向其**内层 datasets/ 目录**（下面就是 Tinyperson/）。
# 例：数据集挂载在 /kaggle/input/datasets/xyjxyjxyj55/tinyperson/datasets/Tinyperson，则本项填到 .../datasets。
# 代码与数据集分开上传时必填；留空('')则回退到 REPO_SRC/datasets（旧的一体式上传）。
DATASET_SRC = '/kaggle/input/datasets/xyjxyjxyj55/tinyperson/datasets'
CFG_REL = 'configs/deim_dfine/dfine_hgnetv2_s_tinyperson_online.yml'  # 离线预切块 S 配置（文件名保留 _online，相对 REPO）

BATCH_PER_GPU = 16        # 每卡训练批大小（全局 = 此值 × 卡数）
VAL_BATCH_PER_GPU = 16    # 每卡验证批大小（val bs16 → 1.51GB，吞吐已饱和）
NUM_WORKERS = 4           # 每 rank 的 dataloader worker（2 rank × 4 = 8）

PRETRAINED_BACKBONE = True   # True=预下载 HGNetv2-B0 骨干；False=随机初始化(不推荐)
BACKBONE_TAG = 'B0'          # S->B0 / M->B4 / L->B6
RESUME_FROM = ''             # 续训：填 last.pth 的完整路径；留空=全新训练
EVAL_FREQ = 5                # 每 N 轮验证一次（1=每轮；设 5=每 5 轮 merge 评测，省验证时间）
EVAL_START = 40              # 从第几轮起才按 EVAL_FREQ 验证（省时：早于此轮不评测）。末轮与阶段切换前一轮(stop_epoch-1)始终强制评测、不受此项影响；设 0=从头评测
EVAL_MAX_DETS = 500          # 评测每图最大检测数：覆盖 yml 官方默认 200（500=消融放宽，结果不再与官方基准直接对比）

# ==================== 派生常量 ====================
WORK = '/kaggle/working'
REPO = os.path.join(WORK, 'DEIM-main')
OUT_DIR = os.path.join(REPO, 'output', 'dfine_hgnetv2_s_tinyperson_online')
BB_URL = 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_%s_stage1.pth'


def say(msg=''):
    print(msg, flush=True)


def run(cmd, cwd=None, check=True):
    say('[CMD] ' + ' '.join(cmd))
    subprocess.run(cmd, cwd=cwd, check=check)


def _can_import(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


# ---------- 1. 复制代码到可写区（Input 只读）+ 挂数据集 ----------
def setup_repo():
    if not (os.path.isdir(REPO_SRC) and os.path.isfile(os.path.join(REPO_SRC, 'train.py'))):
        raise FileNotFoundError('REPO_SRC 无效（目录需含 train.py）：%s' % REPO_SRC)
    if os.path.isdir(REPO):
        say('repo 已存在，跳过复制：%s' % REPO)
    else:
        say('复制代码 %s -> %s（排除 datasets/weight 等大目录）...' % (REPO_SRC, REPO))
        shutil.copytree(REPO_SRC, REPO, dirs_exist_ok=True, ignore=shutil.ignore_patterns(
            'datasets', 'weight', 'output', '__pycache__', '.git', '.idea',
            '_tmp_*', '*.log', '*.zip'))
    # 数据集 symlink 指向只读区（配置用相对路径 datasets/Tinyperson/...）
    # 分开上传：数据集在独立的 DATASET_SRC（其下含 Tinyperson/）；一体上传：回退 REPO_SRC/datasets
    dst_ds = os.path.join(REPO, 'datasets')
    if DATASET_SRC and os.path.isdir(os.path.join(DATASET_SRC, 'Tinyperson')):
        src_ds = DATASET_SRC
    else:
        src_ds = os.path.join(REPO_SRC, 'datasets')
    if not os.path.isdir(src_ds):
        raise FileNotFoundError(
            '数据集目录无效：%s（应含 Tinyperson/）。分开上传请核对 DATASET_SRC；'
            '一体上传请确认 REPO_SRC/datasets 存在。' % src_ds)
    if not os.path.exists(dst_ds):
        os.symlink(src_ds, dst_ds)
        say('symlink datasets -> %s' % src_ds)
    ds = os.path.join(REPO, 'datasets', 'Tinyperson')
    if not os.path.isdir(ds):
        raise FileNotFoundError('找不到数据集 datasets/Tinyperson（应含 annotations/ train_tiles/ test_tiles/）')
    say('数据集就绪：%s' % ds)


# ---------- 2. 装依赖（pip 包不在 zip 内，需现装）----------
def install_deps():
    # 只装 TinyPerson 训练所需；不装 faster-coco-eval-aitod（TinyPerson 用不到）
    want = [('pycocotools', 'pycocotools', True),
            ('faster_coco_eval', 'faster-coco-eval>=1.6.5', True),
            ('calflops', 'calflops', False)]  # calflops 仅供 FLOPs 打印，可选
    for mod, pip_name, _ in want:
        try:
            __import__(mod)
            say('  [ok] %s' % mod)
            continue
        except ImportError:
            pass
        say('  安装 %s ...' % pip_name)
        run([sys.executable, '-m', 'pip', 'install', '--no-input', pip_name], check=False)
    missing = [m for m, _, req in want if req and not _can_import(m)]
    if missing:
        raise RuntimeError('必需依赖装不上：%s（请把上面 pip 输出发回）' % missing)


# ---------- 3. 预下载骨干权重（单进程，规避 DDP 两 rank 竞争）----------
def download_backbone():
    dst_dir = os.path.join(REPO, 'weight', 'hgnetv2')
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, 'PPHGNetV2_%s_stage1.pth' % BACKBONE_TAG)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        say('  骨干已存在：%s' % dst)
        return
    url = BB_URL % BACKBONE_TAG
    say('  下载骨干：%s' % url)
    urllib.request.urlretrieve(url, dst)
    say('  完成（%.1f MB）' % (os.path.getsize(dst) / 1048576.0))


# ---------- 4. 启动 DDP 训练 ----------
def train():
    import torch
    n_gpu = torch.cuda.device_count() or 1
    say('GPU=%d，每卡 train/val bs=%d/%d' % (n_gpu, BATCH_PER_GPU, VAL_BATCH_PER_GPU))
    updates = [
        # total_batch_size 是全局批；engine 自动 // 卡数 = 每卡 bs
        'train_dataloader.total_batch_size=%d' % (BATCH_PER_GPU * n_gpu),
        'val_dataloader.total_batch_size=%d' % (VAL_BATCH_PER_GPU * n_gpu),
        'train_dataloader.num_workers=%d' % NUM_WORKERS,
        'val_dataloader.num_workers=%d' % NUM_WORKERS,
        # 覆盖 base 里指向 Kaggle 不存在路径的 local_model_dir
        'HGNetv2.local_model_dir=weight/hgnetv2/',
    ]
    if not PRETRAINED_BACKBONE:
        updates.append('HGNetv2.pretrained=False')
    updates.append('eval_freq=%d' % EVAL_FREQ)   # 每 N 轮验证一次（默认每轮）
    updates.append('eval_start=%d' % EVAL_START)  # 从第几轮起才验证（末轮/阶段切换前强制）
    updates.append('evaluator.eval_max_dets=%d' % EVAL_MAX_DETS)  # 覆盖 yml 评测 maxDets（官方默认 200）
    os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')  # 减显存碎片
    # NCCL watchdog 兜底:阶段切换时 rank0 独家写 checkpoint/log 较久,默认 heartbeat 480s 可能误报卡死;
    # 放宽到 1800s(与 collective 超时对齐)。根因已由 det_solver.py 阶段切换 barrier 修复,此处仅兜底。
    os.environ.setdefault('TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC', '1800')
    cmd = [sys.executable, '-m', 'torch.distributed.run',
           '--nproc_per_node', str(n_gpu), '--master_port', '29500',
           # --use-amp 必传：train.py 只注入非 None 的 arg，不传则 use_amp=False → FP32 → bs16 OOM
           'train.py', '-c', CFG_REL, '--use-amp', '--seed', '0']
    if RESUME_FROM:
        cmd += ['-r', RESUME_FROM]
    cmd += ['-u'] + updates
    run(cmd, cwd=REPO)


def main():
    say('=' * 68)
    say('TinyPerson / DEIM-D-FINE-S   双卡 DDP 训练')
    say('=' * 68)
    setup_repo()
    install_deps()
    if RESUME_FROM:
        if not os.path.exists(RESUME_FROM):
            raise FileNotFoundError('RESUME_FROM 不存在：%s' % RESUME_FROM)
        say('续训模式：-r %s（train.py 自动置 pretrained=False，跳过骨干下载）' % RESUME_FROM)
    elif PRETRAINED_BACKBONE:
        download_backbone()
    train()
    say('\n训练结束，产出在：%s' % OUT_DIR)


if __name__ == '__main__':
    main()
