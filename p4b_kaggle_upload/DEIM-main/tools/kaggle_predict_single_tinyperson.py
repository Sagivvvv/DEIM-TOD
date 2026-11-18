# -*- coding: utf-8 -*-
"""
Kaggle 单图推理（自适应切块）：TinyPerson DEIM / D-FINE-S
=========================================================
IMG_PATH 可为**单张图**或**文件夹**（文件夹递归预测全部图片，产出镜像子目录）。单图自动适配两种输入：
  · 输入已是 640×576 切块  → 单块直接前向（等价 1 个 corner）；
  · 输入是更大的整图      → 按**评测器同源**的动态均匀重叠切块
    (engine/data/dataset/tinyperson_tiling.generate_tile_corners，
     tile_w/tile_h/overlap 从 yml evaluator 段读取)，逐块推理 → 加 corner 偏移映射回整图
    → 跨块 NMS(merge_nms_th) 合并（只 merge 一次），与 TinyPersonMergeEvaluator 完全同口径。

切块/merge 参数全部取自 yml 的 evaluator 段，不硬编码 → 与训练评测不会漂移；
权重优先取 ema.module（与评测一致）。单卡即可，无 DDP。

运行前（Kaggle Notebook）
------------------------
  1. Accelerator 选 GPU（单卡）；
  2. Input 挂载：代码 dataset（含 DEIM-main）+ 训练产出权重（best_stg2/best_stg1/last.pth）；
     **不需要**挂整个 Tinyperson 数据集（单图自给自足）；
  3. 把 IMG_PATH 改成要预测的图或文件夹（640×576 / 更大整图 / 图片目录均可），Run。

产出：/kaggle/working/single_pred/
  <原文件名>.png        画好预测框的图（整图坐标）
  <原文件名>.json       预测框（xyxy / xywh / score / label，整图坐标）
"""

import os
import re
import sys
import json
import subprocess
from collections import defaultdict

import torch
import torchvision
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ==================== 用户配置（通常只改这里）====================
REPO_SRC = '/kaggle/input/datasets/xyjxyjxyj55/deim-tod/DEIM-main'
CFG_REL = 'configs/deim_dfine/dfine_hgnetv2_s_tinyperson_online.yml'

# 要预测的图：单张(640×576 切块/更大整图) 或 文件夹(递归预测全部图片)
IMG_PATH = '/kaggle/input/datasets/xyjxyjxyj55/apicture/youtube_V0001_I0000080_bl_640x576.png'

CKPT = ''                 # 留空=自动探测(best_stg2 > best_stg1 > last.pth)；也可填 .pth 完整路径或其所在目录

DRAW_SCORE_TH = 0.30      # 画框的分数阈值（目视建议 0.2~0.4）
SAVE_JSON = True          # 是否导出预测 json
JSON_SCORE_TH = 0.01      # json 的分数阈值（与评测口径一致）
LINE_WIDTH = 2            # 框线宽（TinyPerson 目标仅 ~8px，调粗会糊成色块）
C_PRED = (0, 110, 255)    # 蓝：预测框
C_GT = (0, 200, 0)        # 绿：正常 GT
C_UNCERTAIN = (255, 150, 0)  # 橙：uncertain GT
C_IGNORE = (165, 165, 165)   # 灰：ignore GT
TILE_BS = 16              # 切块推理的 batch（同尺寸切块打包前向；单块输入时=1）

DRAW_GT = False           # 是否画真实框(GT)；默认关。自己上传的图通常无 GT json → 保持 False
GT_JSON = ''              # 可选 COCO GT json（坐标需与输入图同坐标系）；留空/不存在=自动不画 GT
DRAW_SCORE_TEXT = True    # 框够高时才写分数（微小框自动跳过）
DRAW_LEGEND = True        # 左上角画图例 + 本图框数

WORK = '/kaggle/working'
OUT_DIR = os.path.join(WORK, 'single_pred')

_CKPT_RE = re.compile(r'^(best_stg2\.pth|best_stg1\.pth|last\.pth)$')
_SKIP_ROOTS = (REPO_SRC,)


def say(msg):
    print(msg, flush=True)


def _can_import(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def install_deps():
    """engine 导入链会 import faster_coco_eval(coco_dataset 顶部)，Kaggle 镜像常缺，需现装。"""
    want = [('pycocotools', 'pycocotools'),
            ('faster_coco_eval', 'faster-coco-eval>=1.6.5')]
    for mod, pip_name in want:
        if _can_import(mod):
            say('  [ok] %s' % mod)
            continue
        say('  安装 %s ...' % pip_name)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-input', pip_name], check=False)
    missing = [m for m, _ in want if not _can_import(m)]
    if missing:
        raise RuntimeError('必需依赖装不上：%s（请把上面 pip 输出发回）' % missing)


# ---------- 定位权重 ----------
def _rank(name):
    return {'best_stg2.pth': 0, 'best_stg1.pth': 1, 'last.pth': 2}.get(name, 9)


def _under_skip(path):
    return any(path == r or path.startswith(r + os.sep) for r in _SKIP_ROOTS)


def _scan(root):
    best, best_r = '', 99
    for cur, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not _under_skip(os.path.join(cur, d))]
        if _under_skip(cur):
            continue
        for fn in files:
            if _CKPT_RE.match(fn):
                k = _rank(fn)
                if k < best_r:
                    best, best_r = os.path.join(cur, fn), k
    return best


def resolve_ckpt():
    if CKPT:
        if os.path.isfile(CKPT):
            return CKPT
        if os.path.isdir(CKPT):
            p = _scan(CKPT)
            if p:
                return p
            raise FileNotFoundError('目录里没有 best_stg2/best_stg1/last.pth：%s' % CKPT)
        raise FileNotFoundError('CKPT 不存在：%s' % CKPT)
    p = _scan('/kaggle/input')
    if not p:
        raise FileNotFoundError('未探测到 best_stg2/best_stg1/last.pth。请把训练产出挂为 Kaggle Dataset，或把 CKPT 填成 .pth 完整路径。')
    return p


# ---------- 建模 + 载权重 ----------
def build_model(ckpt_path):
    from engine.core import YAMLConfig
    cfg = YAMLConfig(CFG_REL)
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False   # 权重全来自 ckpt，别再下骨干

    ck = torch.load(ckpt_path, map_location='cpu')
    use_ema = 'ema' in ck
    state = ck['ema']['module'] if use_ema else ck['model']
    say('权重：%s' % ckpt_path)
    say('  last_epoch=%s，载入 %s' % (ck.get('last_epoch', '?'), 'ema.module' if use_ema else 'model'))
    cfg.model.load_state_dict(state)

    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    model = cfg.model.eval().to(device)
    postprocessor = cfg.postprocessor.eval().to(device)
    say('  device=%s' % device)
    return cfg, model, postprocessor, device


def tile_params(cfg):
    """切块/merge 参数：从 yml evaluator 段读，与训练评测同源。"""
    ev = cfg.yaml_cfg.get('evaluator', {}) or {}
    return (int(ev.get('tile_w', 640)), int(ev.get('tile_h', 576)), int(ev.get('overlap', 50)),
            float(ev.get('merge_nms_th', 0.5)), float(ev.get('score_th', 0.01)), ev.get('max_det', None))


# ---------- 单块前处理：与 val transforms(Resize[576,640] + ConvertPILImage /255) 同口径 ----------
def preprocess(crop, eval_hw):
    """返回 (tensor[3,h,w], 该块真实像素尺寸(w,h))。尺寸不符 eval 时 Resize（边缘小块）。"""
    cw, ch = crop.size
    tw, th = eval_hw[1], eval_hw[0]
    if (cw, ch) != (tw, th):
        crop = crop.resize((tw, th), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(crop)).permute(2, 0, 1).float() / 255.0
    return t, (cw, ch)


@torch.no_grad()
def infer_tiles(img, corners, model, postprocessor, device, eval_hw, score_th, tile_bs, verbose=True):
    """逐块推理 → 加 corner 偏移 → 整图全局坐标（复刻 TinyPersonMergeEvaluator.update）。"""
    tensors, metas = [], []
    for corner in corners:
        crop = img.crop(corner)
        t, (cw, ch) = preprocess(crop, eval_hw)
        tensors.append(t)
        metas.append((corner, (cw, ch)))

    byshape = defaultdict(list)
    for i, t in enumerate(tensors):
        byshape[tuple(t.shape)].append(i)

    gb, gs, gl = [], [], []
    n_done = 0
    for shape, idxs in byshape.items():
        for s in range(0, len(idxs), tile_bs):
            sel = idxs[s:s + tile_bs]
            batch = torch.stack([tensors[i] for i in sel]).to(device)
            # postprocessor 口径 [w,h]：传该块真实尺寸 → 输出即切块局部像素坐标（无需再缩放）
            orig = torch.tensor([[metas[i][1][0], metas[i][1][1]] for i in sel],
                                device=device, dtype=torch.float32)
            results = postprocessor(model(batch), orig)
            for j, i in enumerate(sel):
                r = results[j]
                b, sc, lb = r['boxes'], r['scores'], r['labels']
                n_done += 1
                if b is None or len(b) == 0:
                    continue
                keep = sc > score_th                 # 评测器同一低分过滤
                b, sc, lb = b[keep], sc[keep], lb[keep]
                if len(b) == 0:
                    continue
                b = b.detach().cpu().float().clone()
                corner = metas[i][0]
                b[:, 0::2] += float(corner[0])       # 切块局部 → 整图全局
                b[:, 1::2] += float(corner[1])
                gb.append(b)
                gs.append(sc.detach().cpu().float())
                gl.append(lb.detach().cpu().long())
            if verbose:
                say('  推理 %d/%d 块' % (n_done, len(corners)))
    if gb:
        return torch.cat(gb), torch.cat(gs), torch.cat(gl)
    return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)


def merge_nms(b, s, l, nms_th, max_det):
    """整图跨块 NMS（只 merge 一次）→ 截断 max_det（复刻 _build_predictions）。"""
    if len(b) > 0:
        keep = torchvision.ops.nms(b.float(), s.float(), nms_th)
        b, s, l = b[keep], s[keep], l[keep]
        if max_det is not None and max_det > 0 and len(s) > max_det:
            idx = torch.argsort(s, descending=True)[:max_det]
            b, s, l = b[idx], s[idx], l[idx]
    return b, s, l


def load_gt_map(gt_json):
    """读 COCO GT → {basename(file_name): [anns]}；路径空/不存在返回 {}（自动不画 GT）。"""
    if not gt_json or not os.path.isfile(gt_json):
        return {}
    with open(gt_json, encoding='utf-8') as f:
        d = json.load(f)
    id2name = {int(im['id']): os.path.basename(im['file_name']) for im in d.get('images', [])}
    m = {}
    for a in d.get('annotations', []):
        name = id2name.get(int(a['image_id']))
        if name:
            m.setdefault(name, []).append(a)
    return m


def draw_one(img, boxes, scores, labels, draw_th, anns=(), draw_gt=False):
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    n_gt = n_unc = n_ign = 0
    if draw_gt:
        for a in anns:
            x, y, w, h = a['bbox']
            if a.get('ignore', False):
                col = C_IGNORE
                n_ign += 1
            elif a.get('uncertain', False):
                col = C_UNCERTAIN
                n_unc += 1
            else:
                col = C_GT
                n_gt += 1
            d.rectangle([x, y, x + w, y + h], outline=col, width=LINE_WIDTH)
    n = 0
    for k in range(len(boxes)):
        sc = float(scores[k])
        if sc < draw_th:
            continue
        x1, y1, x2, y2 = [float(v) for v in boxes[k]]
        d.rectangle([x1, y1, x2, y2], outline=C_PRED, width=LINE_WIDTH)
        n += 1
        if DRAW_SCORE_TEXT and (y2 - y1) >= 14:
            d.text((x1 + 1, max(0.0, y1 - 11)), '%.2f' % sc, fill=C_PRED, font=font)
    if DRAW_LEGEND:
        rows = [('pred >= %.2f : %d' % (draw_th, n), C_PRED)]
        if draw_gt:
            rows += [('GT : %d' % n_gt, C_GT),
                     ('GT uncertain : %d' % n_unc, C_UNCERTAIN),
                     ('GT ignore : %d' % n_ign, C_IGNORE)]
        d.rectangle([6, 6, 250, 6 + 16 * len(rows) + 4], fill=(0, 0, 0))
        for j, (txt, col) in enumerate(rows):
            y = 10 + 16 * j
            d.rectangle([12, y, 28, y + 11], outline=col, width=2)
            d.text((34, y), txt, fill=col, font=font)
    return n, (n_gt + n_unc + n_ign)


def list_images(root):
    """递归收集 root 下所有图片 → [(abs_path, rel_subdir), ...]（rel_subdir 用于镜像产出目录）。"""
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
    items = []
    for cur, dirs, files in os.walk(root):
        dirs.sort()
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() in exts:
                rel = os.path.relpath(cur, root)
                items.append((os.path.join(cur, fn), '' if rel == '.' else rel))
    items.sort(key=lambda e: os.path.join(e[1], os.path.basename(e[0])))
    return items


def predict_one(img_path, out_dir, model, postprocessor, device, eval_hw, tp, gen_corners, gt_map=None, verbose=True):
    """单张图：自适应切块 → merge → 画框 + json（整图坐标）。返回 (画框数, json 条数)。"""
    tile_w, tile_h, overlap, nms_th, score_th, max_det = tp
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    corners = gen_corners(w, h, tile_w, tile_h, overlap)
    single = (len(corners) == 1 and corners[0] == (0, 0, w, h))
    say('[%s] size=%dx%d → %s（%d 块）'
        % (os.path.basename(img_path), w, h, '单块直接前向' if single else '切块推理+跨块merge', len(corners)))

    b, s, l = infer_tiles(img, corners, model, postprocessor, device, eval_hw, score_th, TILE_BS, verbose=verbose)
    b, s, l = merge_nms(b, s, l, nms_th, max_det)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(img_path))[0]
    anns = (gt_map or {}).get(os.path.basename(img_path), [])
    draw_gt = DRAW_GT and bool(gt_map)
    n_draw, n_gt = draw_one(img, b, s, l, DRAW_SCORE_TH, anns=anns, draw_gt=draw_gt)
    png_out = os.path.join(out_dir, base + '.png')
    img.save(png_out)

    det = []
    if SAVE_JSON:
        for k in range(len(b)):
            sc = float(s[k])
            if sc < JSON_SCORE_TH:
                continue
            x1, y1, x2, y2 = [float(v) for v in b[k]]
            det.append({'image_id': 1, 'category_id': int(l[k]),
                        'bbox_xyxy': [x1, y1, x2, y2],
                        'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': sc})
        det.sort(key=lambda e: -e['score'])
        jp = os.path.join(out_dir, base + '.json')
        with open(jp, 'w') as f:
            json.dump(det, f, indent=1)
    say('  画框 %d(>=%.2f) / GT %d / json %d 条 → %s' % (n_draw, DRAW_SCORE_TH, n_gt, len(det), png_out))
    if verbose:
        for e in det[:20]:
            bb = e['bbox']
            say('  score=%.3f bbox_xywh=[%.1f, %.1f, %.1f, %.1f]' % (e['score'], bb[0], bb[1], bb[2], bb[3]))
    return n_draw, len(det)


def main():
    say('=' * 72)
    say('TinyPerson 推理（IMG_PATH=单图 或 文件夹；自适应 640×576 单块 / 大图切块+merge）')
    say('=' * 72)
    if not (os.path.isfile(IMG_PATH) or os.path.isdir(IMG_PATH)):
        raise FileNotFoundError('IMG_PATH 既不是文件也不是目录：%s' % IMG_PATH)
    if not (os.path.isdir(REPO_SRC) and os.path.isfile(os.path.join(REPO_SRC, 'train.py'))):
        raise FileNotFoundError('REPO_SRC 无效（目录需含 train.py）：%s' % REPO_SRC)

    sys.path.insert(0, REPO_SRC)
    os.chdir(REPO_SRC)          # 配置里是相对 REPO 的路径
    os.makedirs(OUT_DIR, exist_ok=True)
    install_deps()              # 必须在 import engine 之前

    ckpt_path = resolve_ckpt()
    cfg, model, postprocessor, device = build_model(ckpt_path)

    eval_hw = tuple(int(v) for v in cfg.yaml_cfg.get('eval_spatial_size', [576, 640]))
    tp = tile_params(cfg)
    say('eval_spatial_size(h,w)=%s | tile=%dx%d overlap=%d nms_th=%.2f score_th=%.3f max_det=%s'
        % (eval_hw, tp[0], tp[1], tp[2], tp[3], tp[4], tp[5]))

    # 切块几何与评测器/离线切块脚本同源（同函数同参）
    from engine.data.dataset.tinyperson_tiling import generate_tile_corners

    gt_map = load_gt_map(GT_JSON) if DRAW_GT else {}
    if DRAW_GT and not gt_map:
        say('DRAW_GT=True 但 GT_JSON 为空/不存在(%s) → 自动跳过 GT 框' % GT_JSON)
    elif DRAW_GT:
        say('GT 已加载：%s（%d 张图有标注）' % (GT_JSON, len(gt_map)))

    if os.path.isdir(IMG_PATH):
        items = list_images(IMG_PATH)
        if not items:
            raise FileNotFoundError('文件夹下没有图片：%s' % IMG_PATH)
        say('文件夹模式：%s → %d 张图片（递归）' % (IMG_PATH, len(items)))
        tot_draw = tot_det = 0
        for k, (p, rel) in enumerate(items):
            name = os.path.join(rel, os.path.basename(p)) if rel else os.path.basename(p)
            say('--- [%d/%d] %s' % (k + 1, len(items), name))
            nd, nj = predict_one(p, os.path.join(OUT_DIR, rel), model, postprocessor,
                                 device, eval_hw, tp, generate_tile_corners, gt_map=gt_map, verbose=False)
            tot_draw += nd
            tot_det += nj
        say('文件夹完成：%d 张，共画框 %d(>=%.2f) / json %d 条' % (len(items), tot_draw, DRAW_SCORE_TH, tot_det))
    else:
        predict_one(IMG_PATH, OUT_DIR, model, postprocessor, device, eval_hw, tp,
                    generate_tile_corners, gt_map=gt_map, verbose=True)

    say('=' * 72)
    say('完成。产出目录：%s（Kaggle 右侧 Output 面板可下载）' % OUT_DIR)


if __name__ == '__main__':
    main()
