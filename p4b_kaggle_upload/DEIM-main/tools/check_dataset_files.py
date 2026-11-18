"""
校验 TinyPerson 注解 JSON 与磁盘图片的一致性，并修复 ZIP 解压造成的文件名乱码。

背景
----
Windows 自带的解压（及部分工具）在处理未标记 UTF-8 的 ZIP 条目时，会把文件名按 **CP437**
解码后再写入 NTFS。于是注解里写的 `©`(UTF-8: C2 A9) 在磁盘上变成了 `┬⌐`(CP437 对 C2/A9 的映射)，
导致 torchvision CocoDetection._load_image() 抛 FileNotFoundError。

判定与修复
----------
若注解名 Y 满足 `Y.encode('utf-8').decode('cp437') == 磁盘名 S`，即可确认是同一文件的乱码副本，
把磁盘文件重命名回 Y 即可（**不改注解**，保持 GT 与官方评测口径一致）。

用法
----
    # 只检查，不改动（默认）
    python tools/check_dataset_files.py

    # 检查并执行重命名修复
    python tools/check_dataset_files.py --fix

    # 只看某一部分
    python tools/check_dataset_files.py --part test
"""

import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (标注文件, 图片根目录, 说明)
TARGETS = [
    ('datasets/Tinyperson/annotations/tiny_set_test_all.json',
     'datasets/Tinyperson/test', 'test'),
    ('datasets/Tinyperson/annotations/tiny_set_train_tiles.json',
     'datasets/Tinyperson/train', 'train'),
]

# 常见扩展名，用于建立磁盘索引
EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def index_disk(root):
    """返回 {相对路径(用 / 分隔): 绝对路径}，覆盖 root 下所有图片。"""
    idx = {}
    if not os.path.isdir(root):
        return idx
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(EXTS):
                ap = os.path.join(dirpath, fn)
                rel = os.path.relpath(ap, root).replace(os.sep, '/')
                idx[rel] = ap
    return idx


def cp437_mojibake(name):
    """把注解名按 'UTF-8 字节被 CP437 误解' 还原成磁盘上可能的乱码名。失败返回 None。"""
    try:
        return name.encode('utf-8').decode('cp437')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None


def check(ann_file, img_root, part, do_fix):
    print("\n" + "=" * 78)
    print("[%s] %s" % (part, ann_file))
    print("       img_folder = %s" % img_root)
    print("=" * 78)

    if not os.path.isfile(ann_file):
        print("  !! 注解文件不存在，跳过")
        return dict(part=part, ok=False, reason='ann_missing')

    with open(ann_file, 'r', encoding='utf-8') as f:
        ann = json.load(f)
    images = ann.get('images', [])
    print("  注解 images 条数 : %d" % len(images))
    print("  注解 annotations : %d" % len(ann.get('annotations', [])))

    disk = index_disk(img_root)
    print("  磁盘图片文件数   : %d" % len(disk))

    missing, mojibake, found = [], [], 0
    for im in images:
        fn = im.get('file_name', '')
        if fn in disk:
            found += 1
            continue
        # 也容忍 ./ 或反斜杠写法
        alt = fn.replace('\\', '/').lstrip('./')
        if alt in disk:
            found += 1
            continue
        bad = cp437_mojibake(fn)
        if bad and bad.replace('\\', '/') in disk:
            mojibake.append((fn, bad.replace('\\', '/'), disk[bad.replace('\\', '/')], im.get('id')))
        else:
            missing.append((fn, im.get('id')))

    print("\n  直接命中         : %d / %d" % (found, len(images)))
    print("  CP437 乱码可修复 : %d" % len(mojibake))
    print("  真正缺失         : %d" % len(missing))

    if mojibake:
        print("\n  --- 乱码清单（注解名  <-  磁盘名）---")
        for want, bad, _ap, iid in mojibake[:20]:
            print("    id=%-8s %s" % (iid, want))
            print("    %s磁盘实际: %s" % (' ' * 12, bad))
        if len(mojibake) > 20:
            print("    ... 另有 %d 条" % (len(mojibake) - 20))

    if missing:
        print("\n  --- 真正缺失（磁盘上找不到任何近似名）---")
        for fn, iid in missing[:20]:
            print("    id=%-8s %s" % (iid, fn))
        if len(missing) > 20:
            print("    ... 另有 %d 条" % (len(missing) - 20))

    fixed, failed = 0, []
    if do_fix and mojibake:
        print("\n  --- 执行重命名修复 ---")
        for want, bad, ap, iid in mojibake:
            target = os.path.join(img_root, want.replace('/', os.sep))
            tdir = os.path.dirname(target)
            try:
                if os.path.exists(target):
                    # 目标已存在，说明磁盘上同时有正确名和乱码名 -> 删乱码副本更安全，但这里不动
                    failed.append((want, 'target_exists'))
                    continue
                if tdir and not os.path.isdir(tdir):
                    os.makedirs(tdir, exist_ok=True)
                os.rename(ap, target)
                fixed += 1
            except OSError as e:
                failed.append((want, '%s: %s' % (type(e).__name__, e)))
        print("    重命名成功 : %d" % fixed)
        if failed:
            print("    失败       : %d" % len(failed))
            for w, why in failed[:10]:
                print("      %s  (%s)" % (w, why))

    ok = (len(missing) == 0) and (len(mojibake) == 0 or (do_fix and fixed == len(mojibake)))
    return dict(part=part, ok=ok, total=len(images), found=found,
                mojibake=len(mojibake), missing=len(missing), fixed=fixed)


def main():
    ap = argparse.ArgumentParser(description='Check/repair TinyPerson annotation-to-disk filename consistency.')
    ap.add_argument('--fix', action='store_true', help='执行重命名修复（默认只检查）')
    ap.add_argument('--part', type=str, default='both', choices=['train', 'test', 'both'])
    args = ap.parse_args()

    os.chdir(ROOT)
    print("工作目录:", ROOT)
    print("模式    :", "检查 + 修复" if args.fix else "仅检查（加 --fix 才会改名）")

    rows = []
    for ann, root, part in TARGETS:
        if args.part != 'both' and args.part != part:
            continue
        rows.append(check(ann, root, part, args.fix))

    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    for r in rows:
        print("  %-6s %s  %s" % (r['part'], 'OK' if r['ok'] else 'PROBLEM',
                                  {k: v for k, v in r.items() if k not in ('part', 'ok')}))
    all_ok = all(r['ok'] for r in rows)
    print("\nALL_OK =", all_ok)
    if not all_ok and not args.fix:
        print("提示：存在可修复的 CP437 乱码名，加 --fix 执行重命名。")
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
