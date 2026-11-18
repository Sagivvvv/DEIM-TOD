#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TinyPerson 测试集**离线预切块**脚本(在 AutoDL / 其它 GPU 机命令行运行)。

为什么需要
----------
验证若用在线切块(旧的 TinyPersonTileDataset),每个切块都要把整张原图重新解码再裁剪
(≈21.6 次/图,786 图 → ~17009 块),导致验证比训练还慢。本脚本把测试整图**一次性**切成
物理小图落盘,验证直接读小图(每块只解码一次),大幅提速;merge 回整图 + 官方 AP 逻辑不变。

与评测器同源(关键)
------------------
本脚本与评测器 TinyPersonMergeEvaluator **调用同一个 build_tile_index**
(engine/data/dataset/tinyperson_tiling.py),且都吃**整图 GT**(默认 tiny_set_test_all.json)
+ 相同 tile_w/tile_h/overlap。因此:
  - 本脚本写出的每个切块 image `id` == build_tile_index 的 tile_id;
  - 评测器用整图 GT 经同一函数复现 {tile_id: (corner, 整图 id)};
  两侧映射天然一致 → 切块预测 + corner 可正确 merge 回整图,再算官方 AP。
  ⚠ tile_w/tile_h/overlap 必须与评测器配置(configs/dataset/tinyperson_online_detection.yml)完全一致。

产物
----
  --out-img /<subdir>/<base>__r{RRR}_c{CCC}.<png|jpg>    沿用训练集切块命名(行优先;保留 labeled_images/pure_bg_images 子目录)
  --out-json                                        images 的 id=tile_id;annotations 为空(评测用整图 GT)

保存格式(--format,默认 png)
------------------------------
  - png(默认): **无损**,与在线内存 crop 逐像素一致(无第二次有损压缩)→ 对 tiny 目标最保真、
    AP 对拍最干净;代价是体积大(~17009 块约 7~12GB)。
  - jpg: 有损(第二次压缩),体积小(q95 约 1~2GB),对检测 AP 影响通常可忽略;介意质量可加 --quality 100。

用法(在 DEIM-main/ 目录下)
--------------------------
  python tools/dataset/tile_tinyperson_test.py
  # 或显式指定真实路径:
  python tools/dataset/tile_tinyperson_test.py --gt datasets/Tinyperson/annotations/tiny_set_test_all.json --img-root datasets/Tinyperson/test --out-img datasets/Tinyperson/test_tiles --out-json datasets/Tinyperson/annotations/tiny_set_test_tiles.json

跑完把 test_tiles/ + tiny_set_test_tiles.json 一起打包进 Kaggle 数据集(PNG 约 7~12GB;JPG q95 约 1~2GB)。

依赖:pillow、pycocotools(原生)。无需完整训练环境(build_tile_index 按文件直接加载)。
"""

import argparse
import importlib.util
import json
import os
import sys

from PIL import Image
from pycocotools.coco import COCO

# 脚本位于 <repo>/tools/dataset/;<repo> 根目录(用于定位 tinyperson_tiling.py 与默认相对路径)
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

Image.MAX_IMAGE_PIXELS = None   # 测试整图可达 3840×2160+,解除 PIL 像素上限告警


def _load_build_tile_index():
    """加载与评测器**同源**的 build_tile_index。

    直接按文件路径加载 tinyperson_tiling.py(该模块无任何第三方 import),使本脚本仅依赖
    pillow + pycocotools、无需完整训练环境;文件缺失时回退到常规包导入。两种方式加载的是
    同一份源码 → 与评测器的切块枚举完全一致。"""
    path = os.path.join(_REPO, 'engine', 'data', 'dataset', 'tinyperson_tiling.py')
    if os.path.isfile(path):
        spec = importlib.util.spec_from_file_location('_tinyperson_tiling', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.build_tile_index
    from engine.data.dataset.tinyperson_tiling import build_tile_index
    return build_tile_index


def _split_name(file_name):
    """'labeled_images/xxx.jpg' -> ('labeled_images', 'xxx');兼容无子目录/其它扩展名。
    统一以 --format 指定格式输出(默认 .png),故此处丢弃原扩展名。"""
    file_name = file_name.replace('\\', '/')
    rel_dir = os.path.dirname(file_name)              # 'labeled_images' / 'pure_bg_images' / ''
    base = os.path.splitext(os.path.basename(file_name))[0]
    return rel_dir, base


def _save_tile(img, path, fmt, quality):
    """保存单个切块。png=无损(与在线内存 crop 逐像素一致);jpg=有损(quality 越高越接近无损)。"""
    if fmt == 'png':
        img.save(path)                    # 无损;compress_level 用 PIL 默认(6),只影响体积/速度不影响像素
    else:
        img.save(path, quality=quality)   # 有损;第二次压缩,quality 控制代际损失


def parse_args():
    p = argparse.ArgumentParser(
        description='TinyPerson 测试集离线预切块(与评测器 build_tile_index 同源同参)')
    p.add_argument('--gt', default='datasets/Tinyperson/annotations/tiny_set_test_all.json',
                   help='整图 GT json(= 评测器 merge_gt_file,官方评估基准)')
    p.add_argument('--img-root', default='datasets/Tinyperson/test',
                   help='整图根目录(GT 里 file_name 相对此目录,如 labeled_images/xxx.jpg)')
    p.add_argument('--out-img', default='datasets/Tinyperson/test_tiles',
                   help='切块小图输出目录')
    p.add_argument('--out-json', default='datasets/Tinyperson/annotations/tiny_set_test_tiles.json',
                   help='切块标注输出 json(image id=tile_id;annotations 为空)')
    p.add_argument('--tile-w', type=int, default=640, help='切块宽(须与评测器一致)')
    p.add_argument('--tile-h', type=int, default=576, help='切块高(须与评测器一致)')
    p.add_argument('--overlap', type=int, default=50, help='最小保底重叠(动态均匀重叠下限;须与评测器一致)')
    p.add_argument('--format', choices=['png', 'jpg'], default='png',
                   help='切块保存格式:png=无损(默认,与在线 crop 逐像素一致);jpg=有损但体积小')
    p.add_argument('--quality', type=int, default=95, help='JPEG 质量(仅 --format jpg 生效;越高越接近无损)')
    return p.parse_args()


def main():
    args = parse_args()
    gt, img_root = args.gt, args.img_root
    out_img, out_json = args.out_img, args.out_json
    build_tile_index = _load_build_tile_index()

    print('=' * 72)
    print('TinyPerson 测试集离线预切块')
    print('  GT        :', os.path.abspath(gt))
    print('  img_root  :', os.path.abspath(img_root))
    print('  out_img   :', os.path.abspath(out_img))
    print('  out_json  :', os.path.abspath(out_json))
    ext = 'png' if args.format == 'png' else 'jpg'
    qinfo = '' if ext == 'png' else '  quality=%d' % args.quality
    print('  tile/ovlp : %d × %d / 最小保底重叠≥%d(动态均匀)  format=%s%s' % (
        args.tile_w, args.tile_h, args.overlap, ext, qinfo))
    print('=' * 72)

    if not os.path.isfile(gt):
        raise FileNotFoundError(
            '找不到整图 GT: %s\n请在 DEIM-main/ 目录下运行,或用 --gt 指定绝对路径。' % gt)
    if not os.path.isdir(img_root):
        raise FileNotFoundError('找不到整图根目录: %s(用 --img-root 指定)' % img_root)

    coco = COCO(gt)
    tiles = build_tile_index(coco, args.tile_w, args.tile_h, args.overlap)
    n_imgs = len(coco.getImgIds())
    print('[tile] %d 整图 -> %d 切块(build_tile_index,行优先)' % (n_imgs, len(tiles)))
    if not tiles:
        raise RuntimeError('build_tile_index 返回空,请检查 GT。')

    os.makedirs(out_img, exist_ok=True)

    images_out = []
    N = len(tiles)
    i = 0
    done_imgs = 0
    while i < N:
        # 同一整图的切块在 tiles 中连续(build_tile_index 按 sorted(orig_id) 生成)
        orig_id = tiles[i][1]
        j = i
        while j < N and tiles[j][1] == orig_id:
            j += 1
        group = tiles[i:j]
        i = j
        done_imgs += 1

        info = coco.imgs[orig_id]
        src_name = info['file_name']                  # e.g. 'labeled_images/xxx.jpg'
        rel_dir, base = _split_name(src_name)
        src_path = os.path.join(img_root, *src_name.replace('\\', '/').split('/'))
        if not os.path.isfile(src_path):
            raise FileNotFoundError('找不到原图: %s' % src_path)

        # 逐图只解码一次(消除在线切块对整图的重复解码);convert('RGB') 统一模式(PNG/JPG 均适用)
        src_im = Image.open(src_path)
        im = src_im.convert('RGB')
        src_im.close()

        # n_cols:首行(y1 相同)的切块数;每行列数相同 → 行优先线性 idx 可还原 (r,c)
        first_y1 = group[0][3][1]
        n_cols = 0
        for (_t, _o, _f, corner) in group:
            if corner[1] == first_y1:
                n_cols += 1
            else:
                break
        n_cols = max(n_cols, 1)

        # 输出子目录(保留 labeled_images / pure_bg_images 结构)
        sub_out = os.path.join(out_img, *([rel_dir] if rel_dir else []))
        os.makedirs(sub_out, exist_ok=True)

        try:
            for k, (tile_id, _oid, _fn, (x1, y1, x2, y2)) in enumerate(group):
                r, c = k // n_cols, k % n_cols
                fname = '%s__r%03d_c%03d.%s' % (base, r, c, ext)
                rel = ('%s/%s' % (rel_dir, fname)) if rel_dir else fname
                crop = im.crop((x1, y1, x2, y2))
                dst = os.path.join(sub_out, fname)
                _save_tile(crop, dst, args.format, args.quality)
                images_out.append({
                    'id': int(tile_id),               # == build_tile_index 的 tile_id(评测器据此 merge)
                    'file_name': rel,
                    'width': int(x2 - x1),
                    'height': int(y2 - y1),
                    'corner': [int(x1), int(y1), int(x2), int(y2)],
                    'ori_image_id': int(orig_id),
                    'ori_file_name': src_name,
                })
        finally:
            im.close()

        if done_imgs % 50 == 0 or done_imgs == n_imgs:
            print('  进度: %d/%d 整图, 累计 %d 切块' % (done_imgs, n_imgs, len(images_out)))

    # 写 json:annotations 为空(评测用整图 GT,不需块级标注);id 连续 = tile_id
    out = {
        'info': {
            'description': 'TinyPerson test tiles (offline pre-tiling)',
            'source_gt': os.path.basename(gt),
            'tile_w': args.tile_w, 'tile_h': args.tile_h, 'overlap': args.overlap,
            'format': args.format, 'quality': (None if args.format == 'png' else args.quality),
            'num_source_images': n_imgs, 'num_tiles': len(images_out),
        },
        'images': images_out,
        'annotations': [],
        'categories': [{'id': 1, 'name': 'person', 'supercategory': 'none'}],
        'type': 'instance',
    }
    out_json_abs = os.path.abspath(out_json)
    os.makedirs(os.path.dirname(out_json_abs), exist_ok=True)
    with open(out_json_abs, 'w') as f:
        json.dump(out, f)
    print('[tile] 写出 json: %s(%d images, annotations=[])' % (out_json_abs, len(images_out)))

    # 抽样校验首/末块:落盘尺寸必须 == corner 推出的 w×h
    for tag, rec in (('首块', images_out[0]), ('末块', images_out[-1])):
        p = os.path.join(out_img, *rec['file_name'].split('/'))
        with Image.open(p) as im2:
            ok = (im2.size == (rec['width'], rec['height']))
            print('[tile] %s id=%d %s 落盘尺寸=%s corner=%s 期望=(%d,%d) %s' % (
                tag, rec['id'], rec['file_name'], im2.size, tuple(rec['corner']),
                rec['width'], rec['height'], 'OK' if ok else 'MISMATCH!'))
            if not ok:
                raise AssertionError('切块落盘尺寸与 corner 不一致: %s' % p)

    print('[tile] 完成:%d 整图 -> %d 切块。请把 %s/ 与 %s 一起打包上传 Kaggle 数据集。' % (
        n_imgs, len(images_out), out_img, os.path.basename(out_json)))


if __name__ == '__main__':
    main()
