"""Collect completed official-786 runs; keep published upstream and third-party rows identifiable."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from benchmark_tile_postprocess import bootstrap
from official_tile_candidates import CACHE, GT, GT_SHA256, IMAGE_COUNT, OUTPUT, REPORTS, SCHEMA, SPLIT, run_dir, sha256
from TileStitchingNet.p5 import sampling_group


METHODS = {
    'nms': ('Greedy-NMS', '训练验证使用的整图 torchvision NMS'),
    'tatm-off': ('TA-TM off', 'TA-TM 作者，对照'),
    'tatm': ('TA-TM', 'TA-TM 作者'),
    'car-off': ('CAR off', 'CAR 作者，对照'),
    'car': ('CAR', 'CAR 作者，官方验证切块视图'),
    'diou-nms': ('DIoU-NMS', 'Darknet 作者实现'),
    'cluster-nms': ('Cluster-NMS', 'Zheng 等作者实现'),
    'cluster-diou': ('Cluster-DIoU-NMS', 'Zheng 等上游实现'),
    'soft-nms': ('Gaussian Soft-NMS', 'Soft-NMS 作者'),
    'wbf': ('WBF', 'WBF 作者'),
    'sahi-nmm': ('SAHI-NMM', 'SAHI 项目'),
    'nmw-third-party': ('NMW', '第三方 ensemble-boxes'),
    'cma-third-party': ('CMA-NMS', '第三方 CAR 归档，仅切块'),
}


def ap_rows(path):
    with gzip.open(path,'rt',encoding='utf8') as f:
        obj = json.load(f)
    return obj['ids'], [(np.asarray(s),np.asarray(h,dtype=bool),n) for s,h,n in obj['rows']]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path)
    p.add_argument('--cache',type=Path,default=CACHE)
    p.add_argument('--gt',type=Path,default=GT)
    p.add_argument('--output',type=Path)
    args = p.parse_args()
    if args.runs is None:
        args.runs = run_dir(OUTPUT, sha256(args.cache))
    docs = {k:json.loads((args.runs/(k+'.json')).read_text(encoding='utf8')) for k in METHODS
            if (args.runs/(k+'.json')).is_file()}
    if 'nms' not in docs:
        p.error(f'Run the whole-image NMS baseline first: {args.runs}')
    reference = docs['nms']
    inputs = reference['inputs']
    if inputs.get('schema') != SCHEMA or inputs.get('gt_sha256') != GT_SHA256 or sha256(args.gt) != GT_SHA256:
        raise ValueError('Expected official pre-NMS candidates and the matching official GT')
    args.output = args.output or REPORTS / inputs['cache_sha256'][:16]
    rows = []
    groups = {str(im['id']):sampling_group(im) for im in json.loads(args.gt.read_text(encoding='utf8'))['images']}
    for k, r in docs.items():
        if (r['inputs'] != inputs or r['images'] != IMAGE_COUNT or r['candidates'] != reference['candidates']
                or r['split'] != SPLIT):
            raise ValueError('Unmatched input for '+k)
        m = r['metrics']
        row = dict(method=METHODS[k][0],source=METHODS[k][1],device=r['config'].get('device','CPU'),
                   **{a:m[a] for a in ('Tiny_AP25','Tiny_AP50','Tiny_AP75','AP50','AP','Split_Recall75')},
                   delta_nms=r['delta_nms'],ci95_low=r.get('ci95',[0,0])[0],ci95_high=r.get('ci95',[0,0])[1],
                   median_ms=r['latency_ms']['median'],p95_ms=r['latency_ms']['p95'],
                   setup_ms=r['latency_ms']['split_prepare_ms'],original_split_ms=r['latency_ms'].get('original_split_run_ms'),
                   output_boxes=r['output_boxes'],
                   tiny50_hits_200=r['tiny50_hits']['200'],tiny50_hits_all=r['tiny50_hits']['None'])
        rows.append(row)
    args.output.mkdir(exist_ok=True,parents=True)
    with (args.output/'official_786.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paired = {}
    for k in ('tatm','car'):
        off = k+'-off'
        if k not in docs or off not in docs:
            continue
        ids, active = ap_rows(args.runs/(k+'.ap.json.gz'))
        off_ids, inactive = ap_rows(args.runs/(off+'.ap.json.gz'))
        if ids != off_ids or set(ids) != set(groups):
            raise ValueError('Mismatched paired image sets')
        paired[k] = dict(delta=docs[k]['metrics']['Tiny_AP50']-docs[off]['metrics']['Tiny_AP50'],
                         ci95=bootstrap(inactive, active, ids, groups))
    (args.output/'paired_controls.json').write_text(json.dumps(paired,indent=2),encoding='utf8')
    note = [
        '# TinyPerson 官方 786 张整图后处理对比',
        '',
        f"各方法共享 {reference['candidates']:,} 个 person 候选。按验证配置保留分数 > {inputs['score_threshold']:g} "
        "的切块输出，映射到整图后才执行后处理。普通 NMS 在整图执行一次，IoU 为 0.50。"
        "官方评测每张整图最多计入 200 个检测，差值单位为 AP 百分点。",
        '',
        f"权重 `{inputs['checkpoint_sha256']}`，epoch {inputs['epoch']}，使用 {inputs['weights']}。"
        f"候选缓存 SHA256 为 `{inputs['cache_sha256']}`。",
        '',
        '| 方法 | 实现来源 | Tiny AP50 | Δ NMS [95% CI] | Tiny AP75 | 全尺寸 AP50 | 全尺寸 AP50:95 | 中位 / P95 ms |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        delta = '参照' if r['method']=='Greedy-NMS' else (
            f"{r['delta_nms']:+.3f} [{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]"
        )
        note.append(f"| {r['method']} | {r['source']}（{r['device']}） | {r['Tiny_AP50']:.3f} | {delta} | "
                    f"{r['Tiny_AP75']:.3f} | {r['AP50']:.3f} | {r['AP']:.3f} | {r['median_ms']:.2f} / {r['p95_ms']:.2f} |")
    note.extend(['','## 同流程开关对比',''])
    for k,v in paired.items():
        note.append(f"- {METHODS[k][0]} 开启重评分后，相对其自身 off 行的 Tiny AP50 变化为 "
                    f"{v['delta']:+.3f} 点，95% 区间 [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]。")
    counts = reference['boundary_counts']
    note.extend([
        '','## 复核与适用范围','',
        f"- 普通 NMS 保留 {reference['output_boxes']:,} 个框。导出阶段记录了 {inputs['nonperson_candidates']:,} 个"
        f"非 person 候选。按 person 评测时，整图 NMS 的输出与验证评测器有 "
        f"{inputs['nms_parity_differences']} 张图存在差异。",
        f"- 区间以 {len(set(groups.values()))} 个视频序列或独立图像组作 1,000 次配对抽样，固定检测器与配置。",
        f"- 共有 {counts['outside_tile']:,} 个框超出来源切块，{counts['outside_image']:,} 个框超出原图边界。"
        "WBF 与 NMW 保留作者代码对整图边界的裁剪。TA-TM 采用共同的整图框，并使用像素边界距离。",
        "- 三轮逐图测量排除了检测器、磁盘和 GT 评测。Cluster 两行使用 CUDA，计时包含数据传输。"
        "CAR 精度由作者原始整池入口生成，表中时间为逐图接口；CSV 另附作者入口整池总时间及归一化准备时间。",
        "- Group-Evidence 缺少已核实的完整作者实现。CAR 归档的空间分支缺少语义门控，保留未测。"
        "2026 ASAHI 本文实现、DAHI 原作者实现和 CAD 原作者 NMW 的源码身份也仍待核实。",
        '',
        '[完整指标 CSV](official_786.csv) · [配对开关统计](paired_controls.json) · '
    ])
    (args.output/'RESULTS.md').write_text('\n'.join(note)+'\n',encoding='utf8')
    print(f'{len(rows)} completed rows, {len(set(groups.values()))} independent sampling groups')


if __name__=='__main__':
    main()
