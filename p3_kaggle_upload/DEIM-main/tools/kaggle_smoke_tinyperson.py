# -*- coding: utf-8 -*-
"""
Kaggle 冒烟测试：TinyPerson 在线切块 DEIM/D-FINE 流水线（阶段 1-10）

用途
----
把本文件作为 **Kaggle Script**（Create -> New Kernel -> Script）整体粘贴运行，
它会自动完成环境准备 + 全链路冒烟，并把完整报告写到
    /kaggle/working/SMOKE_REPORT.txt
把该文件内容（或 Kaggle 的完整 Log）发回即可据此定最终配置。

运行前提
--------
1. 已把 DEIM-main（含 datasets/Tinyperson，约 804MB）作为 Kaggle Dataset 上传并挂载到本 Kernel；
   若数据集是单独上传的，脚本也会自动在 /kaggle/input/* 下查找并链接进来。
2. Kernel 设置里打开 **Internet**（pip 装 pycocotools / faster-coco-eval 需要）。
3. Kernel 加速器选 GPU T4 x2 或 GPU P100（本脚本只用 1 张卡，不做分布式）。

脚本做了什么
------------
a. 定位只读区 /kaggle/input/.../DEIM-main，复制到可写区 /kaggle/working/DEIM-main；
b. pip 安装缺失依赖（pycocotools、faster-coco-eval、calflops）；
c. 幂等给 _transforms.py 打 torchvision>=0.17 兼容补丁（仓库里已打过，这里是安全网）；
d. 校验注解 JSON 与磁盘图片一一对应（含 ZIP 解压导致的 CP437 文件名乱码检测与修复）；
e. 依次跑阶段 1-9 冒烟（配置 / 切块数学 / 数据集 / 模型前向 / 官方评测器 /
   真实 evaluate() / solver 契约 / 真实 train_one_epoch）；
f. 阶段 10 探测本机 GPU 的安全 batch_size 与多尺度组合；
g. 全过程 stdout 同步落盘 SMOKE_REPORT.txt。
"""

import os
import sys
import gc
import json
import time
import glob
import shutil
import pickle
import platform
import tempfile
import traceback
import subprocess
from datetime import datetime

# ==================== 0. 用户可调参数 ====================
CFG_REL = 'configs/deim_dfine/dfine_hgnetv2_s_tinyperson_online.yml'

SMOKE_TILES = 320        # 阶段 7 evaluate() 截断的切块数（够出官方 AP 表即可）
TRAIN_SAMPLES = 64       # 阶段 9 train_one_epoch 截断的训练样本数
NUM_WORKERS = 4          # Kaggle GPU Notebook 通常 4 CPU
PROBE_BS = [16, 12, 8, 6, 4]   # 阶段 10 要扫的 batch_size（按显存从大到小）
USE_PRETRAINED = False   # 冒烟阶段不下载骨干权重（省时且不需要网络）；要测真训练改 True
COPY_DATASET = False     # False=用 symlink 指向只读区（快、省空间）；
                         # True=完整复制数据集到 working（仅在需要修复文件名乱码时才需要，约 800MB）

# 已按你的 Kaggle 路径预填；换环境可改这里，或留空则自动 BFS 搜索。
# 仓库   : /kaggle/input/datasets/xyjxyjxyj55/deim-tod/DEIM-main
# 数据集 : 就在仓库内 DEIM-main/datasets/Tinyperson（setup_repo 会自动 symlink 进来，无需单独挂载）
REPO_SRC = '/kaggle/input/datasets/xyjxyjxyj55/deim-tod/DEIM-main'

WORK = '/kaggle/working'
REPO = os.path.join(WORK, 'DEIM-main')
REPORT = os.path.join(WORK, 'SMOKE_REPORT.txt')

RESULTS = {}
_CTX = {}        # 跨阶段共享对象（cfg / model / evaluator ...）


# ==================== 输出：同时写屏幕与报告文件 ====================
class Tee(object):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass

    def fileno(self):
        # 某些库/子进程会对 sys.stdout 调 fileno()；返回第一个真实流的 fd，
        # 避免 AttributeError（真实输出仍走该 fd，报告文件由 write() 负责）。
        for st in self.streams:
            try:
                return st.fileno()
            except (AttributeError, OSError, ValueError):
                continue
        raise OSError('Tee: 底层没有可用的 fileno()')

    def isatty(self):
        for st in self.streams:
            try:
                return st.isatty()
            except (AttributeError, OSError, ValueError):
                continue
        return False


def say(msg=''):
    print(msg)


def stage(n, name):
    say('\n' + '=' * 78)
    say('STAGE %s: %s' % (n, name))
    say('=' * 78)


def done(key, ok, note=''):
    RESULTS[key] = 'PASS' if ok else 'FAIL'
    say('>>> STAGE %s : %s %s' % (key, RESULTS[key], note))


def run_cmd(cmd):
    """运行子进程并把输出实时转写到屏幕+报告。

    不能用 subprocess.run(stdout=sys.stdout)：sys.stdout 已被换成 Tee 对象，
    而 subprocess 需要真实文件描述符(会调 .fileno())。改用 Popen+PIPE 逐行读，
    再交给 say() 同时写屏幕与报告文件（pip / probe 的输出因此也会进报告）。
    """
    say('[CMD] ' + (cmd if isinstance(cmd, str) else ' '.join(cmd)))
    p = subprocess.Popen(cmd, shell=isinstance(cmd, str),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        say('  | ' + line.rstrip())
    p.wait()
    say('[CMD exit] %s' % p.returncode)
    return p


# ==================== 1. 定位并复制仓库 ====================
def _looks_like_deim(path):
    return (os.path.isfile(os.path.join(path, 'train.py'))
            and os.path.isdir(os.path.join(path, 'engine'))
            and os.path.isdir(os.path.join(path, 'configs')))


def _bfs_find_deim(base, max_depth=6):
    """从 base 广度优先找含 train.py+engine/+configs/ 的目录。

    只遍历目录树（scandir 每次只列一层，图片等文件不入队），命中即返回、
    不再深入其内部，因此不会去走 800MB 的图片目录，速度很快。
    Kaggle 数据集可能挂在很深的层级（如
    /kaggle/input/datasets/<user>/<slug>/DEIM-main），故用 BFS+深度上限，
    而不是固定层数的 glob。
    """
    from collections import deque
    skip = {'__pycache__', '.git', '.ipynb_checkpoints'}
    q = deque([(base, 0)])
    while q:
        dp, depth = q.popleft()
        if _looks_like_deim(dp):
            return dp
        if depth >= max_depth:
            continue
        try:
            with os.scandir(dp) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError:
            continue
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False) and e.name not in skip:
                    q.append((e.path, depth + 1))
            except OSError:
                continue
    return None


def find_repo():
    """定位 DEIM-main：优先用 REPO_SRC，否则在 /kaggle/input 下深度搜索。"""
    # 1) 用户显式指定（最可靠）
    if REPO_SRC:
        if _looks_like_deim(REPO_SRC):
            say('  使用 REPO_SRC 指定路径: %s' % REPO_SRC)
            return REPO_SRC
        say('  !! REPO_SRC=%s 不像 DEIM-main（缺 train.py/engine/configs），改用自动搜索' % REPO_SRC)
    # 2) 自动搜索（BFS，可命中任意深度）
    for base in ('/kaggle/input', '/kaggle'):
        if os.path.isdir(base):
            hit = _bfs_find_deim(base)
            if hit:
                return hit
    # 3) 当前目录本身
    if _looks_like_deim(os.getcwd()):
        return os.getcwd()
    # 4) 都没找到：打印 /kaggle/input 结构帮助排查
    say('  !! 自动定位失败。/kaggle/input 下的目录结构（前 40 行）：')
    n = 0
    for dp, dns, _fns in os.walk('/kaggle/input'):
        if dp.count('/') - '/kaggle/input'.count('/') > 4:
            dns[:] = []
            continue
        say('     %s' % dp)
        n += 1
        if n >= 40:
            say('     ... (truncated)')
            break
    raise FileNotFoundError(
        '找不到 DEIM-main（需含 train.py + engine/ + configs/）。请把它的完整路径'
        '填到脚本顶部 REPO_SRC，例如 '
        "REPO_SRC = '/kaggle/input/datasets/xyjxyjxyj55/deim-tod/DEIM-main'")


def setup_repo():
    stage('0a', 'locate DEIM-main & copy code to writable area')
    src = find_repo()
    say('  source (read-only) : %s' % src)
    say('  COPY_DATASET       : %s' % COPY_DATASET)
    if os.path.abspath(src) != os.path.abspath(REPO):
        if os.path.isdir(REPO):
            say('  %s 已存在，跳过复制' % REPO)
        else:
            t0 = time.time()
            if COPY_DATASET:
                say('  copying FULL repo incl. datasets (~800MB, 较慢) ...')
                shutil.copytree(src, REPO, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns(
                                    '__pycache__', '.git', '.idea', 'output',
                                    '_tmp_*', '*.log', '*.zip'))
            else:
                # 只复制代码（约 3MB），数据集用 symlink 指向只读区，避开 800MB 拷贝耗时
                say('  copying code only (datasets excluded) ...')
                shutil.copytree(src, REPO, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns(
                                    'datasets', '__pycache__', '.git', '.idea',
                                    'output', 'weight', '_tmp_*', '*.log', '*.zip'))
                # 把只读区的数据集链进来（配置里用的是相对路径 datasets/Tinyperson/...）
                src_ds = os.path.join(src, 'datasets')
                dst_ds = os.path.join(REPO, 'datasets')
                if os.path.isdir(src_ds) and not os.path.exists(dst_ds):
                    os.symlink(src_ds, dst_ds)
                    say('  symlinked datasets -> %s' % src_ds)
            say('  copied in %.1fs' % (time.time() - t0))
    say('  repo (writable)    : %s' % REPO)
    _CTX['repo_src'] = src
    return REPO


def ensure_dataset():
    """确保 REPO/datasets/Tinyperson 存在；若数据集单独上传则链接进来。"""
    stage('0b', 'locate TinyPerson dataset')
    dst = os.path.join(REPO, 'datasets', 'Tinyperson')
    if os.path.isdir(dst):
        say('  found in repo copy: %s' % dst)
        _report_ds_size(dst)
        return dst

    say('  仓库副本内无 datasets/Tinyperson，到 /kaggle/input 下找 ...')
    found = None
    for pat in ('/kaggle/input/*/*Tinyperson*', '/kaggle/input/*/*tiny*person*',
                '/kaggle/input/*/*/*Tinyperson*'):
        for p in sorted(glob.glob(pat)):
            if os.path.isdir(os.path.join(p, 'annotations')):
                found = p
                break
        if found:
            break
    if not found:
        raise FileNotFoundError(
            '找不到 TinyPerson 数据集（需含 annotations/ 与 train/ test/）。'
            '请把 datasets/Tinyperson 一起打包上传。')
    say('  found external    : %s' % found)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(found, dst)
    say('  symlinked -> %s' % dst)
    _report_ds_size(dst)
    return dst


def _report_ds_size(root):
    """报告数据集规模，便于确认上传完整。"""
    for sub in ('train', 'test', 'annotations'):
        p = os.path.join(root, sub)
        if not os.path.isdir(p):
            say('    [WARN] 缺少子目录: %s' % sub)
            continue
        n, sz = 0, 0
        for dp, _dn, fns in os.walk(p):
            for fn in fns:
                n += 1
                try:
                    sz += os.path.getsize(os.path.join(dp, fn))
                except OSError:
                    pass
        say('    %-12s %6d files  %8.1f MB' % (sub, n, sz / 1048576.0))


def install_deps():
    stage('0c', 'install missing python deps')
    # (真正被 import 的模块名, pip 包名, 是否必需)
    # 注意：coco_dataset.py 顶部 `import faster_coco_eval` 用的是标准包 faster-coco-eval，
    #       不是 faster-coco-eval-aitod（后者仅 AITOD 路径/kaggle_train_deim.py 用，
    #       engine/ 里无任何模块 import 它，TinyPerson 冒烟与训练都不需要，
    #       故不装，省掉它在 py3.13 无 wheel 时的源码编译风险）。
    want = [('pycocotools', 'pycocotools', True),
            ('faster_coco_eval', 'faster-coco-eval>=1.6.5', True),
            ('calflops', 'calflops', False)]
    need = []
    for mod, pip_name, req in want:
        try:
            __import__(mod)
            say('  [ok] %-24s already installed' % mod)
        except ImportError:
            say('  [MISSING] %-22s -> pip install %-24s %s'
                % (mod, pip_name, '(REQUIRED)' if req else '(optional)'))
            need.append((mod, pip_name, req))
    # 逐个安装：必需的先装、各自单独装，避免某个 C++ 包在 py3.13 编译失败拖累其它包
    for mod, pip_name, req in sorted(need, key=lambda x: not x[2]):
        say('  --- installing %s (%s) ---' % (pip_name, 'REQUIRED' if req else 'optional'))
        run_cmd([sys.executable, '-m', 'pip', 'install', '--no-input', pip_name])
    # 复核：按「真正 import 的模块名」判断
    missing_req = []
    for mod, pip_name, req in want:
        try:
            m = __import__(mod)
            say('  [ok] %-24s %s' % (mod, getattr(m, '__version__', '')))
        except ImportError:
            if req:
                say('  [STILL MISSING - REQUIRED] %s (pip: %s)' % (mod, pip_name))
                missing_req.append(mod)
            else:
                say('  [STILL MISSING - optional] %s (pip: %s) 冒烟可继续' % (mod, pip_name))
    if missing_req:
        raise RuntimeError(
            '必需依赖装不上：%s。多半是该 C++ 包在 Python 3.13 无 wheel 且源码编译失败；'
            '请把上方 pip 输出发回。' % missing_req)


def patch_transforms():
    """幂等：给自定义 Transform 补 transform() 兼容方法（torchvision>=0.17）。"""
    stage('0d', 'torchvision Transform compatibility shim')
    path = os.path.join(REPO, 'engine', 'data', 'transforms', '_transforms.py')
    with open(path, encoding='utf-8') as f:
        src = f.read()
    n_have = src.count('def transform(self, inpt')
    if n_have >= 3:
        say('  already patched (%d transform() shim found), skip' % n_have)
        return
    old = '    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:'
    shim = ('    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:\n'
            '        # 兼容新版 torchvision：forward() 调 transform() 而非 _transform()\n'
            '        if isinstance(inpt, self._transformed_types):\n'
            '            return self._transform(inpt, params)\n'
            '        return inpt\n'
            '\n')
    n = src.count(old)
    if n == 0:
        say('  WARN: 未找到 _transform 签名，未注入；若后续 NotImplementedError 请检查该文件')
        return
    with open(path, 'w', encoding='utf-8') as f:
        f.write(src.replace(old, shim + old))
    say('  patched %d custom Transform(s)' % n)


def patch_optional_evaluators():
    """幂等：把 dataset/__init__.py 对 coco_eval_aitod / coco_eval_visdrone 的硬导入改成容错。

    这两个评测器分别依赖 faster_coco_eval_aitod / faster_coco_eval；缺包时其顶层 import 抛
    ModuleNotFoundError，而 engine/data/dataset/__init__.py 无条件导入它们，会连带整个
    `import engine` 失败（Kaggle py3.13 上 faster_coco_eval_aitod 未预装）。TinyPerson 只用
    TinyPersonMergeEvaluator，故把这两个改成 try/except 跳过即可，无需安装/编译该包。
    """
    stage('0d2', 'make AITOD/VisDrone evaluator imports optional')
    path = os.path.join(REPO, 'engine', 'data', 'dataset', '__init__.py')
    with open(path, encoding='utf-8') as f:
        src = f.read()
    if 'AitodCocoEvaluator = None' in src:
        say('  already patched, skip')
    else:
        old_a = 'from .coco_eval_aitod import AitodCocoEvaluator'
        old_v = 'from .coco_eval_visdrone import VisdroneCocoEvaluator'
        new_a = ('try:\n'
                 '    from .coco_eval_aitod import AitodCocoEvaluator\n'
                 'except ImportError:\n'
                 '    AitodCocoEvaluator = None  # 缺 faster_coco_eval_aitod；TinyPerson 不需要')
        new_v = ('try:\n'
                 '    from .coco_eval_visdrone import VisdroneCocoEvaluator\n'
                 'except ImportError:\n'
                 '    VisdroneCocoEvaluator = None  # 缺依赖时跳过；TinyPerson 不需要')
        if old_a in src:
            src = src.replace(old_a, new_a)
            if old_v in src:
                src = src.replace(old_v, new_v)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(src)
            say('  patched: coco_eval_aitod / coco_eval_visdrone 改为容错导入')
        else:
            say('  WARN: 未找到预期导入行(文件可能已改)，跳过')
    # 清理可能已缓存的 engine 模块，确保 patch 后重新导入生效(Notebook 重复运行场景)
    purged = [m for m in list(sys.modules) if m == 'engine' or m.startswith('engine.')]
    for m in purged:
        del sys.modules[m]
    say('  cleared %d cached engine.* module(s) from sys.modules' % len(purged))


# ==================== 2. 数据集文件一致性（含 CP437 乱码修复）====================
def cp437_mojibake(name):
    try:
        return name.encode('utf-8').decode('cp437')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None


def index_disk(root):
    idx = {}
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(exts):
                ap = os.path.join(dp, fn)
                idx[os.path.relpath(ap, root).replace(os.sep, '/')] = ap
    return idx


def check_files(ann_rel, img_rel, tag, do_fix=True):
    stage('0e-%s' % tag, 'annotation <-> disk filename consistency (%s)' % tag)
    ann_file = os.path.join(REPO, ann_rel)
    img_root = os.path.join(REPO, img_rel)
    if not os.path.isfile(ann_file):
        say('  !! 注解不存在: %s' % ann_file)
        return False
    with open(ann_file, encoding='utf-8') as f:
        ann = json.load(f)
    images = ann.get('images', [])
    disk = index_disk(img_root)
    say('  ann_file     : %s' % ann_rel)
    say('  images/anns  : %d / %d' % (len(images), len(ann.get('annotations', []))))
    say('  disk files   : %d' % len(disk))

    found, mojibake, missing = 0, [], []
    for im in images:
        fn = im.get('file_name', '')
        if fn in disk or fn.replace('\\', '/').lstrip('./') in disk:
            found += 1
            continue
        bad = cp437_mojibake(fn)
        key = bad.replace('\\', '/') if bad else None
        if key and key in disk:
            mojibake.append((fn, key, disk[key]))
        else:
            missing.append(fn)

    say('  direct hit   : %d / %d' % (found, len(images)))
    say('  CP437 broken : %d' % len(mojibake))
    say('  truly missing: %d' % len(missing))
    for fn, bad, _ in mojibake[:10]:
        say('    want : %s' % fn)
        say('    disk : %s' % bad)
    for fn in missing[:10]:
        say('    MISSING: %s' % fn)

    if mojibake and do_fix:
        # /kaggle/input 只读；只有已复制到 working（COPY_DATASET=True）的文件才能改名
        fixed, failed = 0, 0
        for fn, bad, ap in mojibake:
            target = os.path.join(img_root, fn.replace('/', os.sep))
            try:
                if os.path.exists(target):
                    failed += 1
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                os.rename(ap, target)
                fixed += 1
            except OSError:
                failed += 1
        say('  renamed      : %d ok, %d failed' % (fixed, failed))
        if failed and fixed == 0:
            say('  !! 需重命名但失败：/kaggle/input 是只读区，symlink 模式下改不了名。')
            say('     两种解法（二选一）：')
            say('     (a) 推荐：在本地先跑 python tools/check_dataset_files.py --fix 修好，再重新打包上传；')
            say('     (b) 把本脚本顶部的 COPY_DATASET 改为 True（会把 ~800MB 数据集复制到可写区）。')
        if fixed:
            disk = index_disk(img_root)
            still = [im['file_name'] for im in images
                     if im.get('file_name', '') not in disk]
            say('  复核仍未命中 : %d' % len(still))
    elif mojibake:
        say('  (do_fix=False，未尝试重命名)')
    ok = (len(missing) == 0 and len(mojibake) == 0)
    say('  => %s' % ('OK' if ok else 'PROBLEM'))
    return ok


# ==================== 3. 阶段 1-9 冒烟 ====================
def stage1_config():
    stage(1, 'merged YAML config sanity (resolution / scales / epochs)')
    try:
        from engine.core import YAMLConfig
        cfg = YAMLConfig(CFG_REL,
                         HGNetv2={'pretrained': USE_PRETRAINED},
                         train_dataloader={'num_workers': NUM_WORKERS},
                         val_dataloader={'num_workers': NUM_WORKERS})
        y = cfg.yaml_cfg
        _CTX['cfg'] = cfg

        ess = list(y.get('eval_spatial_size'))
        say('  eval_spatial_size      : %s' % ess)
        assert ess == [576, 640], 'eval_spatial_size 应为 [576,640]，实际 %s' % ess

        def resize_size(side):
            ops = y[side + '_dataloader']['dataset']['transforms']['ops']
            for o in ops:
                if o.get('type') == 'Resize':
                    return list(o['size'])
            return None

        tr, va = resize_size('train'), resize_size('val')
        say('  train Resize.size      : %s' % tr)
        say('  val   Resize.size      : %s' % va)
        assert va == ess, 'val Resize %s 必须 == eval_spatial_size %s' % (va, ess)
        assert tr == [576, 640], 'train Resize 应为 [576,640]，实际 %s' % tr

        scales = [list(s) for s in y['train_dataloader']['collate_fn']['scales']]
        say('  collate_fn.scales      : %s' % scales)
        for s in scales:
            assert len(s) == 2, 'scales 必须是 [h,w] 对，实际 %s' % s
            assert s[0] % 32 == 0 and s[1] % 32 == 0, '%s 未 32 对齐' % s
        say('  全部尺度 32 对齐       : OK')

        se = y['train_dataloader']['collate_fn']['stop_epoch']
        ep = y.get('epoches')
        say('  stop_epoch / epoches   : %s / %s' % (se, ep))
        assert se != 0, 'stop_epoch 不能为 0（两阶段会去找 best_stg1.pth 而 FileNotFoundError）'
        assert 0 < se < ep, 'stop_epoch(%s) 应在 (0, epoches=%s) 之间' % (se, ep)

        say('  use_amp / use_ema      : %s / %s' % (y.get('use_amp'), y.get('use_ema')))
        say('  train total_batch_size : %s' % y['train_dataloader'].get('total_batch_size'))
        say('  val   total_batch_size : %s' % y['val_dataloader'].get('total_batch_size'))
        say('  num_classes            : %s' % y.get('num_classes'))

        ev = y['evaluator']
        say('  evaluator              : %s' % ev['type'])
        say('    merge_gt_file        : %s' % ev['merge_gt_file'])
        say('    tile/overlap         : %sx%s overlap=%s' % (ev['tile_w'], ev['tile_h'], ev['overlap']))
        assert y['val_dataloader']['dataset']['ann_file'] == ev['merge_gt_file'], \
            'val ann_file 必须 == evaluator.merge_gt_file'
        assert (y['val_dataloader']['dataset']['tile_w'] == ev['tile_w']
                and y['val_dataloader']['dataset']['tile_h'] == ev['tile_h']
                and y['val_dataloader']['dataset']['overlap'] == ev['overlap']), \
            'val 数据集与评测器的 tile/overlap 参数必须一致'
        say('  val ann == merge_gt    : OK')
        say('  tile params 两侧一致    : OK')
        done('1_config', True)
    except Exception:
        traceback.print_exc()
        done('1_config', False)


def stage2_tiling():
    stage(2, 'tiling math: generate_tile_corners / build_tile_index')
    try:
        from engine.data.dataset.tinyperson_tiling import (
            _axis_starts, generate_tile_corners, build_tile_index)

        say('  X(3840, min_ovlp=50) : %s' % (_axis_starts(3840, 640, 50),))
        say('  Y(2160, min_ovlp=50) : %s' % (_axis_starts(2160, 576, 50),))

        cases = [(3840, 2160, 35), (1920, 1080, 8), (640, 576, 1), (300, 200, 1)]
        for w, h, expect in cases:
            c = generate_tile_corners(w, h, 640, 576, 50)
            say('  %5dx%-5d -> %3d tiles (expect %d) %s'
                % (w, h, len(c), expect, 'OK' if len(c) == expect else 'MISMATCH'))
            assert len(c) == expect, '%dx%d 切块数 %d != %d' % (w, h, len(c), expect)
            for (x1, y1, x2, y2) in c:
                assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h, \
                    '越界块 %s on %dx%d' % ((x1, y1, x2, y2), w, h)
            # 全覆盖检查：每个采样点都落在某块内
            for px in range(0, w, max(w // 17, 1)):
                for py in range(0, h, max(h // 17, 1)):
                    assert any(x1 <= px < x2 and y1 <= py < y2 for (x1, y1, x2, y2) in c), \
                        '点 (%d,%d) 未被覆盖 on %dx%d' % (px, py, w, h)
        say('  所有块无越界 + 采样点全覆盖 : OK')

        small = generate_tile_corners(300, 200, 640, 576, 50)
        say('  小图 300x200 -> %s （不补黑边，取实际范围）' % small)
        assert small == [(0, 0, 300, 200)], '小图应取实际范围，实际 %s' % small

        # build_tile_index：用真实整图 COCO 验证 tile_id 连续
        cfg = _CTX['cfg']
        vds = cfg.val_dataloader.dataset
        _CTX['vds'] = vds
        tiles = vds.tiles
        say('  build_tile_index: %d 整图 -> %d 切块' % (len(vds.ids), len(tiles)))
        ids = [t[0] for t in tiles]
        assert ids == list(range(1, len(tiles) + 1)), 'tile_id 应从 1 连续编号'
        say('  tile_id 连续 1..%d : OK' % len(tiles))
        say('  首块 : %s' % (tiles[0],))
        say('  末块 : %s' % (tiles[-1],))
        _CTX['n_tiles'] = len(tiles)
        done('2_tiling', True)
    except Exception:
        traceback.print_exc()
        done('2_tiling', False)


def _peek(loader, tag):
    it = iter(loader)
    imgs, tgts = next(it)
    say('  %s batch images : %s %s' % (tag, tuple(imgs.shape), imgs.dtype))
    say('  %s target[0] keys: %s' % (tag, sorted(tgts[0].keys())))
    return imgs, tgts


def stage3_val_dataset():
    stage(3, 'val dataset (online tiling) + batch shape')
    try:
        cfg = _CTX['cfg']
        vds = _CTX.get('vds') or cfg.val_dataloader.dataset
        say('  len(vds) = %d 切块' % len(vds))
        img, tgt = vds[0]
        say('  vds[0] image : %s %s' % (type(img).__name__, getattr(img, 'size', None)))
        say('  vds[0] target: %s' % {k: (tuple(v.shape) if hasattr(v, 'shape') else v)
                                       for k, v in sorted(tgt.items())})
        assert list(tgt['orig_size']) == [640, 576], \
            'orig_size 应为切块 (w,h)=(640,576)，实际 %s' % list(tgt['orig_size'])
        assert tuple(tgt['boxes'].shape)[1:] == (4,), 'boxes 末维应为 4'
        say('  orig_size=(640,576) 空 boxes : OK（切块不带 GT，评测用整图 GT 单独算）')

        imgs, tgts = _peek(cfg.val_dataloader, 'val')
        assert imgs.shape[1:] == (3, 576, 640), 'val batch 应为 (B,3,576,640)，实际 %s' % (imgs.shape,)
        say('  val batch = (B,3,576,640) 非方形 : OK')
        done('3_val_ds', True)
    except Exception:
        traceback.print_exc()
        done('3_val_ds', False)


def stage4_train_dataset():
    stage(4, 'train dataset (physical tiles) + multi-scale non-square')
    try:
        import torch
        cfg = _CTX['cfg']

        # 先建 loader（会构建完整数据集），记录真实规模后**立即截断**，
        # 使阶段 4/9 复用同一个 loader：避免重建 dataset、也避开已 fork worker 持旧数据的问题。
        # DataLoader 的 sampler 每次 __iter__ 都动态读 len(dataset)，故截断后 len(loader) 会同步变小。
        loader = cfg.train_dataloader
        tds = loader.dataset
        full = len(tds.ids)
        _CTX['train_full_n'] = full
        _CTX['train_loader'] = loader
        say('  len(tds) = %d 物理切块小图' % full)

        img, tgt = tds[0]
        say('  tds[0] target: %s' % {k: (tuple(v.shape) if hasattr(v, 'shape') else v)
                                      for k, v in sorted(tgt.items())})
        assert tuple(tgt['boxes'].shape)[1:] == (4,), 'train boxes 末维应为 4'
        b = tgt['boxes']
        if b.numel():
            say('  boxes 范围: x[%.4f,%.4f] y[%.4f,%.4f] (归一化 cxcywh)'
                % (b[:, 0].min(), b[:, 0].max(), b[:, 1].min(), b[:, 1].max()))
            assert float(b.max()) <= 1.5 and float(b.min()) >= -0.5, \
                'boxes 应已归一化到 ~[0,1]，实际范围 %s' % b
            say('  boxes 已归一化 cxcywh : OK')

        collate = loader.collate_fn
        scales = {tuple(s) for s in collate.scales}
        say('  scales(去重) : %s' % sorted(scales))
        say('  base_size / stop_epoch : %s / %s'
            % (getattr(collate, 'base_size', None), collate.stop_epoch))

        # 截断到 TRAIN_SAMPLES，仅用于冒烟（真实训练不受影响，这里改的是内存里的 dataset）
        tds.ids = list(tds.ids[:TRAIN_SAMPLES])
        say('  ids 截断用于冒烟: %d -> %d (%d batches @ bs=%d)'
            % (full, len(tds.ids), len(loader), loader.batch_size))

        # epoch=0 应开多尺度；多跑几个 batch 看是否出现多种非方形尺寸
        loader.set_epoch(0)      # DataLoaderWrapper.set_epoch 会同时设 dataset 与 collate_fn
        seen = {}
        for i, (imgs, _t) in enumerate(loader):
            seen[tuple(imgs.shape[2:])] = seen.get(tuple(imgs.shape[2:]), 0) + 1
            if i >= 11:
                break
        say('  epoch=0 前 %d 个 batch 的形状分布 : %s' % (min(12, len(loader)), seen))
        for hw in seen:
            assert len(hw) == 2 and hw[0] % 32 == 0 and hw[1] % 32 == 0, '形状 %s 未 32 对齐' % (hw,)
            assert hw in scales, '形状 %s 不在 scales 集合内' % (hw,)
        say('  全部落在 scales 且 32 对齐 : OK')
        if len(seen) > 1:
            say('  多尺度确实在生效（出现 %d 种尺寸）: OK' % len(seen))
        else:
            say('  WARN: 只出现 1 种尺寸，多尺度可能未生效（检查 epoch < stop_epoch）')

        # 关掉多尺度后应恒定为原生尺寸（epoch >= stop_epoch）
        loader.set_epoch(999)
        imgs, _t = next(iter(loader))
        say('  epoch=999(>stop_epoch) batch: %s  应恒为 (B,3,576,640)' % (tuple(imgs.shape),))
        assert tuple(imgs.shape[2:]) == (576, 640), \
            '关多尺度后应恒为 576x640，实际 %s' % (tuple(imgs.shape[2:]),)
        loader.set_epoch(0)
        done('4_train_ds', True)
    except Exception:
        traceback.print_exc()
        done('4_train_ds', False)


def stage5_model():
    stage(5, 'model forward @ non-square 576x640 (train & eval mode)')
    try:
        import torch
        cfg = _CTX['cfg']
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = cfg.model.to(dev)
        post = cfg.postprocessor
        n_param = sum(p.numel() for p in model.parameters())
        say('  device      : %s' % dev)
        say('  params      : %.3f M' % (n_param / 1e6))

        x = torch.randn(2, 3, 576, 640, device=dev)
        tgt = [{'labels': torch.zeros(2, dtype=torch.long, device=dev),
                'boxes': torch.tensor([[0.5, 0.5, 0.05, 0.08], [0.3, 0.4, 0.04, 0.06]],
                                      device=dev),
                'orig_size': torch.tensor([576, 640], device=dev)} for _ in range(2)]

        model.train()
        out = model(x, targets=tgt)
        say('  train mode keys : %s' % sorted(out.keys()))
        say('  pred_logits %s  pred_boxes %s'
            % (tuple(out['pred_logits'].shape), tuple(out['pred_boxes'].shape)))
        assert tuple(out['pred_logits'].shape)[:2] == (2, 300)

        model.eval()
        with torch.no_grad():
            out = model(x)
            res = post(out, torch.stack([t['orig_size'] for t in tgt]))
        say('  eval mode  : pred_logits %s' % (tuple(out['pred_logits'].shape),))
        say('  postproc[0]: boxes %s scores %s labels %s'
            % (tuple(res[0]['boxes'].shape), tuple(res[0]['scores'].shape),
               tuple(res[0]['labels'].shape)))
        bx = res[0]['boxes']
        if bx.numel():
            say('  boxes 像素坐标范围: x[%.1f,%.1f] y[%.1f,%.1f] (切块 640x576 内)'
                % (bx[:, 0].min(), bx[:, 0].max(), bx[:, 1].min(), bx[:, 1].max()))
        _CTX['model'] = model
        _CTX['device'] = dev
        _CTX['post'] = post
        done('5_model_fwd', True)
    except Exception:
        traceback.print_exc()
        done('5_model_fwd', False)


def stage6_evaluator():
    stage(6, 'TinyPersonMergeEvaluator: native pycocotools + huicv COCOExpandEval + tile map')
    try:
        from engine.data.dataset.coco_eval_tinyperson import (
            _load_official_evaluators, _tools_dir)
        say('  _tools_dir() = %s' % _tools_dir())
        assert os.path.isdir(_tools_dir()), 'tools 目录不存在，无法 import huicv'

        COCO_cls, Expand_cls = _load_official_evaluators()
        import pycocotools.cocoeval as _ce
        say('  COCO           : %s' % COCO_cls)
        say('  COCOExpandEval : %s.%s' % (Expand_cls.__module__, Expand_cls.__name__))
        say('  cocoeval file  : %s' % _ce.__file__)
        assert 'faster_coco_eval' not in str(_ce.__file__), \
            'cocoeval 仍是 faster_coco_eval 影子，官方重写逻辑会失效'
        base = Expand_cls.__mro__[1]
        say('  ExpandEval base: %s.%s' % (base.__module__, base.__name__))
        assert 'faster_coco_eval' not in str(getattr(base, '__file__', '')), '基类非原生 COCOeval'
        say('  [OK] 原生 pycocotools 绑定成功，官方 COCOExpandEval 可进程内复用')

        cfg = _CTX['cfg']
        t0 = time.time()
        ev = cfg.evaluator
        say('  evaluator built in %.1fs : %s' % (time.time() - t0, type(ev).__name__))
        say('    gt imgIds      : %d' % len(ev.img_ids))
        say('    gt annotations : %d' % len(ev.gt_coco.dataset['annotations']))
        say('    gt categories  : %s' % ev.gt_coco.dataset['categories'])
        say('    block_info     : %d tiles' % len(ev.block_info))
        say('    tile/overlap   : %dx%s overlap=%s' % (ev.tile_w, ev.tile_h, ev.overlap))
        say('    nms/score/maxd : %s / %s / %s' % (ev.merge_nms_th, ev.score_th, ev.max_det))

        vds = _CTX.get('vds') or cfg.val_dataloader.dataset
        assert len(ev.block_info) == len(vds.tiles), \
            'block_info(%d) != dataset.tiles(%d)' % (len(ev.block_info), len(vds.tiles))
        bad = 0
        for (tid, oid, _fn, corner) in vds.tiles:
            got = ev.block_info.get(int(tid))
            if got is None or got != (tuple(int(v) for v in corner), int(oid)):
                bad += 1
                if bad <= 3:
                    say('    MISMATCH tile %s: ds=%s ev=%s' % (tid, (corner, oid), got))
        assert bad == 0, 'tile_id 映射两侧不一致，共 %d 处' % bad
        say('  [OK] tile_id <-> (corner, orig_id) 映射两侧 %d 条全部一致' % len(vds.tiles))
        _CTX['ev'] = ev
        _CTX['val_loader'] = cfg.val_dataloader
        done('6_evaluator', True)
    except Exception:
        traceback.print_exc()
        done('6_evaluator', False)


def stage7_evaluate():
    stage(7, 'real det_engine.evaluate() on %d tiles -> merge -> official tiny AP' % SMOKE_TILES)
    try:
        import torch
        from engine.solver.det_engine import evaluate as det_evaluate
        vds = _CTX['vds']
        ev = _CTX['ev']
        full = len(vds.tiles)
        vds.tiles = vds.tiles[:SMOKE_TILES]
        say('  truncated val tiles: %d -> %d (%d batches)'
            % (full, len(vds.tiles), (len(vds.tiles) + 15) // 16))

        model = _CTX['model']
        dev = _CTX['device']
        model.eval()
        t0 = time.time()
        test_stats, coco_evaluator = det_evaluate(
            model, _CTX['cfg'].criterion, _CTX['post'],
            _CTX['val_loader'], ev, dev)
        el = time.time() - t0
        say('\n  [TIME] evaluate(%d tiles) = %.1fs (%.1f tiles/s)'
            % (SMOKE_TILES, el, SMOKE_TILES / el))
        say('  [EST ] 全量 %d 切块 ≈ %.1f 分钟（按同吞吐线性外推，未含 accumulate）'
            % (full, full / max(SMOKE_TILES / el, 1e-6) / 60))
        say('  test_stats keys : %s' % sorted(test_stats.keys()))
        for k, v in test_stats.items():
            say('    %-14s len=%-3d first=%.6f' % (k, len(v), float(v[0])))
        _CTX['test_stats'] = test_stats
        vds.tiles = vds.tiles  # 保持截断态供阶段 8 用
        done('7_evaluate', True)
    except Exception:
        traceback.print_exc()
        done('7_evaluate', False)


def stage8_solver_contract():
    stage(8, 'det_solver contract: coco_eval["bbox"].eval picklable + .stats tolist')
    try:
        import torch
        import numpy as np
        ev = _CTX['ev']
        assert ev.coco_eval, 'coco_eval 为空，accumulate 未填充'
        ce = ev.coco_eval['bbox']
        say('  coco_eval["bbox"] : %s' % type(ce).__name__)

        evd = ce.eval
        say('  .eval type/keys   : %s %s' % (type(evd).__name__, sorted(evd.keys())))
        blob = pickle.dumps(evd)
        say('  .eval picklable   : OK (%.2f MB)' % (len(blob) / 1048576.0))
        tmp = os.path.join(tempfile.gettempdir(), '_kaggle_eval.pth')
        torch.save(evd, tmp)
        back = torch.load(tmp, weights_only=False)
        say('  torch.save/load   : OK keys=%s' % sorted(back.keys()))
        os.remove(tmp)

        st = ce.stats
        assert isinstance(st, np.ndarray), '.stats 应为 ndarray，实际 %s' % type(st)
        lst = st.tolist()
        say('  .stats            : ndarray len=%d' % len(st))
        say('  .stats.tolist()   : %s' % ['%.6f' % x for x in lst])
        say('  AP(stats[0])      : %.6f  <-- det_solver 用它选 best' % lst[0])
        assert all(isinstance(x, float) for x in lst), '.stats 应全为 float 才能 JSON 序列化'
        say('  JSON serializable : OK')
        say('  NOTE: 随机权重(pretrained=%s)，AP≈0 属正常，本阶段只验证契约' % USE_PRETRAINED)
        done('8_solver_contract', True)
    except Exception:
        traceback.print_exc()
        done('8_solver_contract', False)


def stage9_train():
    stage(9, 'real train_one_epoch on %d physical tiles (loss/backward/EMA/LR)' % TRAIN_SAMPLES)
    try:
        import torch
        from engine.solver.det_engine import train_one_epoch
        cfg = _CTX['cfg']
        dev = _CTX['device']
        model = _CTX['model']
        criterion = cfg.criterion.to(dev)
        optimizer = cfg.optimizer
        ema, scaler = cfg.ema, cfg.scaler
        say('  optimizer   : %s lr=%s groups=%d'
            % (type(optimizer).__name__, optimizer.param_groups[0]['lr'], len(optimizer.param_groups)))
        say('  ema / scaler: %s / %s' % (type(ema).__name__,
                                          type(scaler).__name__ if scaler else None))
        say('  clip_max_norm: %s' % cfg.yaml_cfg.get('clip_max_norm'))

        # 复用阶段 4 已截断并建好的 loader（ids 已限为 TRAIN_SAMPLES）
        loader = _CTX.get('train_loader') or cfg.train_dataloader
        tds = loader.dataset
        full = _CTX.get('train_full_n', len(tds.ids))
        loader.set_epoch(0)
        say('  train loader: %d 截断样本 -> %d batches (bs=%d, num_workers=%d)'
            % (len(tds.ids), len(loader), loader.batch_size, loader.num_workers))

        probe = 'decoder.dec_score_head.0.weight'
        before = dict(model.named_parameters())[probe].detach().clone()

        if dev == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        train_stats = train_one_epoch(
            False, cfg.lr_scheduler, model, criterion, loader, optimizer, dev,
            epoch=0, max_norm=cfg.yaml_cfg.get('clip_max_norm', 0.1),
            print_freq=1, ema=ema, scaler=scaler,
            lr_warmup_scheduler=cfg.lr_warmup_scheduler, writer=None)
        el = time.time() - t0
        if dev == 'cuda':
            torch.cuda.synchronize()
            pk = torch.cuda.max_memory_allocated() / 1024 ** 3
            say('  [VRAM] peak allocated = %.2f GB' % pk)
        say('  [TIME] train_one_epoch(%d) = %.1fs (%.2f s/iter)'
            % (TRAIN_SAMPLES, el, el / max(len(loader), 1)))

        tl = train_stats.get('loss')
        say('  total loss  = %.4f  (finite=%s)' % (tl, tl == tl and tl < 1e4))
        say('  loss_bbox   = %.4f' % train_stats.get('loss_bbox', float('nan')))
        say('  loss_giou   = %.4f' % train_stats.get('loss_giou', float('nan')))
        say('  loss_vfl    = %.4f' % train_stats.get('loss_vfl', float('nan')))
        say('  loss_fgl    = %.4f' % train_stats.get('loss_fgl', float('nan')))
        say('  lr          = %.8f' % train_stats.get('lr', float('nan')))

        after = dict(model.named_parameters())[probe].detach()
        changed = not torch.equal(before, after)
        say('  param updated by backward : %s (max|delta|=%.3e)'
            % (changed, (after - before).abs().max().item()))
        gok = True
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all():
                gok = False
                say('    NON-FINITE grad: %s' % n)
                break
        say('  all grads finite          : %s' % gok)

        # 外推整轮训练耗时（用截断前/后的真实规模）
        bs = loader.batch_size
        iters = max(full // bs, 1)
        sec_it = el / max(len(loader), 1)
        epochs = cfg.yaml_cfg.get('epoches', 36)
        say('\n  [EST] 全量训练 1 epoch ≈ %d iters x %.2f s ≈ %.1f 分钟'
            % (iters, sec_it, iters * sec_it / 60))
        say('  [EST] %d epochs ≈ %.1f 小时（不含每轮验证）'
            % (epochs, iters * sec_it * epochs / 3600))
        say('  [EST] 若每轮都跑全量验证，需再加上 阶段7 外推的单轮验证耗时 x %d' % epochs)
        ok = (tl == tl and tl < 1e4 and changed and gok)
        done('9_train', ok)
    except torch.cuda.OutOfMemoryError:
        traceback.print_exc()
        say('\n  !! CUDA OOM：把 train_dataloader.total_batch_size 调小，或删掉最大的尺度档')
        done('9_train', False, '(OOM)')
    except Exception:
        traceback.print_exc()
        done('9_train', False)


def stage10_probe():
    stage(10, 'VRAM budget probe -> recommended batch_size for THIS GPU')
    try:
        cmd = [sys.executable, os.path.join(REPO, 'tools', 'probe_vram_budget.py'),
               '-c', CFG_REL, '--bs', ','.join(str(b) for b in PROBE_BS), '--amp']
        say('  运行: %s' % ' '.join(cmd))
        rc = run_cmd(cmd)
        say('  probe exit code = %s' % getattr(rc, 'returncode', None))
        done('10_probe', True)
    except Exception:
        traceback.print_exc()
        done('10_probe', False)


# ==================== main ====================
def main():
    os.makedirs(WORK, exist_ok=True)
    _rep = open(REPORT, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, _rep)
    sys.stderr = Tee(sys.__stderr__, _rep)

    say('#' * 78)
    say('# TinyPerson / DEIM 冒烟测试报告')
    say('# 时间   : %s' % datetime.now().isoformat())
    say('# 平台   : %s' % platform.platform())
    say('# python : %s' % sys.version.replace('\n', ' '))
    say('#' * 78)

    t_start = time.time()
    try:
        import torch
        say('torch    : %s | cuda(build) %s | available %s'
            % (torch.__version__, torch.version.cuda, torch.cuda.is_available()))
        if torch.cuda.is_available():
            say('GPU count: %d' % torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                say('  [%d] %s  VRAM=%.2f GB  sm_%d%d'
                    % (i, p.name, p.total_memory / 1024 ** 3, p.major, p.minor))
        import torchvision
        say('torchvision: %s' % torchvision.__version__)
    except Exception:
        traceback.print_exc()

    interrupted = False
    try:
        setup_repo()
        ensure_dataset()
        install_deps()
        patch_transforms()
        patch_optional_evaluators()

        ok_files = True
        ok_files &= check_files('datasets/Tinyperson/annotations/tiny_set_test_all.json',
                                'datasets/Tinyperson/test', 'test')
        ok_files &= check_files('datasets/Tinyperson/annotations/tiny_set_train_tiles.json',
                                'datasets/Tinyperson/train', 'train')
        RESULTS['0_files'] = 'PASS' if ok_files else 'FAIL'
        say('>>> STAGE 0_files : %s' % RESULTS['0_files'])

        os.chdir(REPO)
        sys.path.insert(0, REPO)
        say('\n工作目录已切到: %s' % os.getcwd())
        say('配置文件      : %s' % CFG_REL)

        # 缓解显存碎片
        os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

        stage1_config()
        if '1_config' in RESULTS and RESULTS['1_config'] == 'PASS':
            stage2_tiling()
            stage3_val_dataset()
            stage4_train_dataset()
            stage5_model()
            stage6_evaluator()
            if RESULTS.get('6_evaluator') == 'PASS':
                stage7_evaluate()
                stage8_solver_contract()
            stage9_train()
        stage10_probe()
    except Exception:
        interrupted = True
        say('\n!!! 冒烟流程异常中断 !!!')
        traceback.print_exc()

    el = time.time() - t_start
    say('\n' + '#' * 78)
    say('# 汇总（总耗时 %.1f 分钟）' % (el / 60))
    say('#' * 78)
    for k in sorted(RESULTS, key=lambda s: (len(s), s)):
        say('  %-24s %s' % (k, RESULTS[k]))
    required = ['0_files', '1_config', '2_tiling', '3_val_ds', '4_train_ds',
                '5_model_fwd', '6_evaluator', '7_evaluate', '8_solver_contract',
                '9_train', '10_probe']
    missing = [k for k in required if k not in RESULTS]
    failed = [k for k, v in RESULTS.items() if v != 'PASS']
    # 关键：RESULTS 为空(流程在最前面就中断)时 all([]) 会是 True，必须显式排除这种假成功
    all_pass = (not interrupted) and (not missing) and (not failed)
    say('\nALL_PASS = %s' % all_pass)
    if interrupted:
        say('流程被异常中断（见上方 Traceback）——这不是通过，多半是定位/环境问题。')
    if missing:
        say('以下阶段未产生结果（可能在它们之前就中断了）：%s' % missing)
    if failed:
        say('以下阶段 FAIL：%s' % failed)
    if all_pass:
        say('全部通过：管线在 Kaggle 上可用。阶段 10 的输出给出了本 GPU 的安全 batch_size，')
        say('据此用 -u 覆盖 train_dataloader.total_batch_size 即可开始真实训练。')
    else:
        say('请把本报告完整发回，特别是中断/FAIL 阶段上方的 Traceback。')
    say('\n报告文件: %s' % REPORT)

    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    _rep.close()
    return 0 if all_pass else 1


if __name__ == '__main__':
    _rc = main()
    # Notebook(交互式)里 sys.exit 会显示成 "SystemExit"，看着像报错其实不是；
    # 只有真正的 Script 模式才用退出码。两种模式都不影响 SMOKE_REPORT.txt 的内容。
    if 'get_ipython' in globals():
        say('\n[Notebook 模式] 返回码 = %d（0=ALL_PASS，1=有问题）；不调用 sys.exit 以免显示 SystemExit。' % _rc)
    else:
        sys.exit(_rc)
