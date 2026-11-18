"""
探测当前 GPU 上 DEIM / D-FINE 的显存预算与吞吐，给出安全的 batch_size 与多尺度档位组合。

为什么需要它
------------
显存不足时，Windows(WDDM) 与部分 Linux 驱动**不会立刻报 OOM**，而是把超出物理显存的部分
溢出到共享系统内存。结果是"能跑通"但每个 iter 慢几倍到几十倍。因此本脚本同时测量：
    peak allocated / peak reserved / 单 step 耗时 / 吞吐(img/s)
并把 peak 超过物理显存的组合标记为 SPILL(溢出=不可用)，而不是简单的 OK/OOM。

用法
----
    # 1) 全量扫描（推荐在目标平台上跑，把完整输出发回）
    python tools/probe_vram_budget.py -c configs/deim_dfine/dfine_hgnetv2_s_tinyperson_online.yml

    # 2) 开 AMP（混合精度，显存约省 30~40%）
    python tools/probe_vram_budget.py -c <cfg> --amp

    # 3) 只扫指定 batch_size（省时）
    python tools/probe_vram_budget.py -c <cfg> --bs 4,2,1

    # 4) 只扫训练 / 只扫验证
    python tools/probe_vram_budget.py -c <cfg> --only train
    python tools/probe_vram_budget.py -c <cfg> --only val

    # 5) 自定义显存预算与尺度（尺度用 h,w 分号分隔）
    python tools/probe_vram_budget.py -c <cfg> --budget 15.0 --scales "576,640;768,832"

注意：本脚本只用合成张量，**不读数据集**，因此与数据路径无关，可在任何机器上直接跑。
Windows 上若报 GBK 解码错误，请先设 $env:PYTHONUTF8='1'。
"""

import os
import sys
import gc
import time
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def human_gb(x):
    return "%.2f" % (x / 1024 ** 3)


def make_targets(bs, sz, device, n_box=3):
    """构造 criterion 需要的最小 target（归一化 cxcywh，与 ConvertBoxes 输出一致）。"""
    tgts = []
    for _ in range(bs):
        labels = torch.zeros(n_box, dtype=torch.long, device=device)
        boxes = torch.tensor(
            [[0.30, 0.40, 0.05, 0.08],
             [0.55, 0.60, 0.04, 0.07],
             [0.70, 0.25, 0.06, 0.09]][:n_box],
            dtype=torch.float32, device=device)
        tgts.append({
            'labels': labels,
            'boxes': boxes,
            'orig_size': torch.tensor([sz[0], sz[1]], dtype=torch.long, device=device),
        })
    return tgts


def reset_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


class Probe:
    """封装一次 前向(+反向) 的显存与耗时测量。"""

    def __init__(self, model, criterion, optimizer, device, amp=False, n_warm=1, n_rep=2):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.amp = amp
        self.n_warm = n_warm
        self.n_rep = n_rep

    # -- 训练：前向 + loss + 反向 + optimizer.step -------------------------- #
    def train_step(self, bs, sz):
        x = torch.randn(bs, 3, sz[0], sz[1], device=self.device)
        tgts = make_targets(bs, sz, self.device)
        self.model.train()
        self.criterion.train()

        if self.amp:
            with torch.autocast(device_type=str(self.device), cache_enabled=True):
                out = self.model(x, targets=tgts)
            with torch.autocast(device_type=str(self.device), enabled=False):
                ld = self.criterion(out, tgts, epoch=0, step=0,
                                    global_step=0, epoch_step=1)
        else:
            out = self.model(x, targets=tgts)
            ld = self.criterion(out, tgts, epoch=0, step=0, global_step=0, epoch_step=1)

        loss = sum(ld.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        lv = float(loss.detach())
        del x, tgts, out, ld, loss
        return lv

    # -- 验证：eval 前向 ---------------------------------------------------- #
    def val_step(self, bs, sz):
        x = torch.randn(bs, 3, sz[0], sz[1], device=self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(x)
        del x, out
        return 0.0

    def measure(self, mode, bs, sz):
        """返回 dict 或抛异常。含 warmup（首次含 cuDNN 选算法开销，不计入耗时）。"""
        fn = self.train_step if mode == 'train' else self.val_step

        reset_mem()
        for _ in range(self.n_warm):
            fn(bs, sz)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        reset_mem()

        ts = []
        lv = 0.0
        for _ in range(self.n_rep):
            t0 = time.time()
            lv = fn(bs, sz)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ts.append(time.time() - t0)

        t = sorted(ts)[len(ts) // 2]          # 中位数，抗抖动
        res = {
            'bs': bs,
            'scale': list(sz),
            'peak_alloc_gb': torch.cuda.max_memory_allocated() / 1024 ** 3
            if torch.cuda.is_available() else float('nan'),
            'peak_reserved_gb': torch.cuda.max_memory_reserved() / 1024 ** 3
            if torch.cuda.is_available() else float('nan'),
            'sec_per_step': t,
            'img_per_sec': bs / t if t > 0 else float('inf'),
            'loss': lv,
        }
        reset_mem()
        return res


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description='Probe DEIM/D-FINE VRAM budget & throughput on the current GPU.')
    ap.add_argument('-c', '--config', type=str, required=True, help='YAML 配置文件路径')
    ap.add_argument('--bs', type=str, default='16,12,8,6,4,2,1',
                    help='要扫描的 batch_size 列表，逗号分隔（默认 16,12,8,6,4,2,1）')
    ap.add_argument('--scales', type=str, default=None,
                    help='要扫描的尺度 "h,w;h,w"；默认取配置里 collate_fn.scales')
    ap.add_argument('--budget', type=float, default=None,
                    help='显存预算(GB)；默认取物理显存的 90%%')
    ap.add_argument('--amp', action='store_true', help='用混合精度(autocast)测量')
    ap.add_argument('--only', type=str, default='both', choices=['train', 'val', 'both'],
                    help='只扫训练 / 只扫验证 / 两者（默认 both）')
    ap.add_argument('--no-pretrained', action='store_true', default=True,
                    help='不下载 backbone 预训练权重（默认开启，探测无需权重）')
    ap.add_argument('--val-scale', type=str, default=None,
                    help='验证扫描用的尺度 "h,w"；默认取配置 eval_spatial_size')
    ap.add_argument('--json', type=str, default=None, help='把结果另存为 JSON 的路径')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，将在 CPU 上测量（显存列无意义，仅看耗时）")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    print("=" * 78)
    print("环境")
    print("=" * 78)
    print("  torch        :", torch.__version__)
    print("  cuda(build)  :", torch.version.cuda)
    print("  device       :", device)
    phys_gb = None
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(0)
        phys_gb = p.total_memory / 1024 ** 3
        print("  GPU          :", p.name)
        print("  VRAM(physical): %.2f GB" % phys_gb)
        print("  capability   :", p.major, ".", p.minor)
    budget = args.budget if args.budget else (phys_gb * 0.9 if phys_gb else float('inf'))
    print("  VRAM budget  : %.2f GB%s" % (
        budget, "  (= 物理显存 x 0.9)" if args.budget is None else "  (手动指定)"))
    print("  AMP          :", args.amp)

    # ---------------- 构建模型 ---------------- #
    from engine.core import YAMLConfig
    cfg = YAMLConfig(args.config,
                     HGNetv2={'pretrained': False},
                     train_dataloader={'num_workers': 0},
                     val_dataloader={'num_workers': 0})
    model = cfg.model.to(device)
    criterion = cfg.criterion.to(device) if cfg.criterion is not None else None
    optimizer = cfg.optimizer
    n_param = sum(p.numel() for p in model.parameters())
    print("  model        : %.3f M params" % (n_param / 1e6))

    # ---------------- 待扫描的尺度 ---------------- #
    if args.scales:
        scales = [[int(v) for v in s.split(',')] for s in args.scales.split(';') if s.strip()]
    else:
        scales = [list(s) for s in cfg.train_dataloader.collate_fn.scales]
    # 去重，保持顺序
    seen, uniq = set(), []
    for s in scales:
        k = tuple(s)
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    scales = uniq

    if args.val_scale:
        val_scale = [int(v) for v in args.val_scale.split(',')]
    else:
        val_scale = list(cfg.yaml_cfg.get('eval_spatial_size', scales[0]))

    bs_list = [int(b) for b in args.bs.split(',') if b.strip()]
    cfg_bs = getattr(cfg.train_dataloader.collate_fn, 'total_batch_size', None) or \
        cfg.yaml_cfg['train_dataloader'].get('total_batch_size')
    print("\n  config train total_batch_size :", cfg_bs)
    print("  config eval_spatial_size      :", val_scale)
    print("  scales to probe               :", scales)
    print("  batch sizes to probe          :", bs_list)

    probe = Probe(model, criterion, optimizer, device, amp=args.amp)
    results = {'env': {'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
                       'vram_physical_gb': phys_gb, 'budget_gb': budget,
                       'torch': torch.__version__, 'amp': args.amp},
               'train': [], 'val': []}

    # ---------------- A. 训练扫描 ---------------- #
    if args.only in ('train', 'both'):
        print("\n" + "=" * 78)
        print("A. TRAIN 前向+反向+step  (peak_alloc GB / peak_resv GB / sec_per_step / img/s)")
        print("=" * 78)
        print("%-12s %5s %10s %10s %11s %10s %8s  %s"
              % ("scale[h,w]", "bs", "alloc", "resv", "sec/step", "img/s", "loss", "判定"))
        for sz in scales:
            for bs in bs_list:
                try:
                    r = probe.measure('train', bs, sz)
                except torch.cuda.OutOfMemoryError:
                    print("%-12s %5d %10s %10s %11s %10s %8s  %s"
                          % (str(sz), bs, "-", "-", "-", "-", "-", "OOM"))
                    results['train'].append({'scale': sz, 'bs': bs, 'verdict': 'OOM'})
                    reset_mem()
                    continue
                except Exception as e:
                    print("%-12s %5d %10s %10s %11s %10s %8s  %s"
                          % (str(sz), bs, "-", "-", "-", "-", "-",
                             "ERR:%s" % type(e).__name__))
                    results['train'].append({'scale': sz, 'bs': bs, 'verdict': 'ERR',
                                             'err': "%s: %s" % (type(e).__name__, e)})
                    reset_mem()
                    continue

                if phys_gb and r['peak_reserved_gb'] > phys_gb:
                    verdict = 'SPILL'      # 溢出到共享内存：能跑但极慢，视为不可用
                elif r['peak_reserved_gb'] > budget:
                    verdict = 'OVER-BUDGET'
                else:
                    verdict = 'OK'
                r['verdict'] = verdict
                results['train'].append(r)
                print("%-12s %5d %10s %10s %11.2f %10.2f %8.3f  %s"
                      % (str(sz), bs, human_gb(r['peak_alloc_gb'] * 1024 ** 3),
                         human_gb(r['peak_reserved_gb'] * 1024 ** 3),
                         r['sec_per_step'], r['img_per_sec'], r['loss'], verdict))
                # 大 bs 若已 SPILL，更小的 bs 仍可能 OK，继续扫；但同尺度不再往上试
                # （bs_list 建议从大到小给，此处保持原顺序不剪枝，保证输出完整）

    # ---------------- B. 验证扫描 ---------------- #
    if args.only in ('val', 'both'):
        print("\n" + "=" * 78)
        print("B. VAL(eval) 前向 @ %s  — 决定 val_dataloader.total_batch_size" % str(val_scale))
        print("=" * 78)
        print("%-12s %5s %10s %10s %11s %10s  %s"
              % ("scale[h,w]", "bs", "alloc", "resv", "sec/step", "img/s", "判定"))
        for bs in bs_list:
            try:
                r = probe.measure('val', bs, val_scale)
            except torch.cuda.OutOfMemoryError:
                print("%-12s %5d %10s %10s %11s %10s  %s"
                      % (str(val_scale), bs, "-", "-", "-", "-", "OOM"))
                results['val'].append({'scale': val_scale, 'bs': bs, 'verdict': 'OOM'})
                reset_mem()
                continue
            except Exception as e:
                print("%-12s %5d %10s %10s %11s %10s  %s"
                      % (str(val_scale), bs, "-", "-", "-", "-", "ERR:%s" % type(e).__name__))
                results['val'].append({'scale': val_scale, 'bs': bs, 'verdict': 'ERR',
                                       'err': "%s: %s" % (type(e).__name__, e)})
                reset_mem()
                continue

            if phys_gb and r['peak_reserved_gb'] > phys_gb:
                verdict = 'SPILL'
            elif r['peak_reserved_gb'] > budget:
                verdict = 'OVER-BUDGET'
            else:
                verdict = 'OK'
            r['verdict'] = verdict
            results['val'].append(r)
            print("%-12s %5d %10s %10s %11.2f %10.2f  %s"
                  % (str(val_scale), bs, human_gb(r['peak_alloc_gb'] * 1024 ** 3),
                     human_gb(r['peak_reserved_gb'] * 1024 ** 3),
                     r['sec_per_step'], r['img_per_sec'], verdict))

    # ---------------- C. 结论 ---------------- #
    print("\n" + "=" * 78)
    print("C. 结论（判定=OK 表示 peak_reserved 在预算内，未溢出到共享内存）")
    print("=" * 78)

    def best(rows):
        ok = [r for r in rows if r.get('verdict') == 'OK']
        if not ok:
            return None
        return max(ok, key=lambda r: r['bs'])

    tr_ok = [r for r in results['train'] if r.get('verdict') == 'OK']
    if tr_ok:
        max_bs_all = max(r['bs'] for r in tr_ok)
        full = [r['scale'] for r in tr_ok if r['bs'] == max_bs_all]
        print("  [train] 全档位都 OK 的最大 bs 需满足所有尺度；")
        print("          bs=%d 时预算内档位 %d/%d : %s"
              % (max_bs_all, len(full), len(scales), full))
        # 找"所有尺度都 OK"的最大 bs
        safe_bs = None
        for bs in sorted(set(r['bs'] for r in tr_ok), reverse=True):
            cov = {tuple(r['scale']) for r in tr_ok if r['bs'] == bs}
            if cov == {tuple(s) for s in scales}:
                safe_bs = bs
                break
        if safe_bs:
            print("  [train] => 推荐 total_batch_size = %d（覆盖全部 %d 个尺度档，无溢出）"
                  % (safe_bs, len(scales)))
            slow = max((r['sec_per_step'] for r in tr_ok if r['bs'] == safe_bs), default=0)
            print("           最慢档 sec/step = %.2f" % slow)
        else:
            print("  [train] => 没有任何 bs 能覆盖全部尺度档；建议裁剪 scales 或开 --amp")
            for bs in sorted(set(r['bs'] for r in tr_ok), reverse=True):
                cov = sorted({tuple(r['scale']) for r in tr_ok if r['bs'] == bs})
                print("           bs=%-3d 预算内档位: %s" % (bs, cov))
    else:
        print("  [train] 无任何 OK 组合：请开 --amp、减小 --bs，或换更大显存的 GPU")

    vb = best(results['val'])
    if vb:
        print("  [val]   => 推荐 val total_batch_size = %d (%.2f img/s)"
              % (vb['bs'], vb['img_per_sec']))
    else:
        print("  [val]   无任何 OK 组合")

    # 预估 epoch 时长（需数据集规模，这里只给每千张图耗时）
    if tr_ok:
        r = max(tr_ok, key=lambda x: x['img_per_sec'])
        print("\n  [参考] 最快组合 %s bs=%d : %.2f img/s -> 每 1000 张约 %.1f 分钟"
              % (r['scale'], r['bs'], r['img_per_sec'], 1000 / r['img_per_sec'] / 60))

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print("\n  结果已写入:", args.json)

    print("\n" + "=" * 78)
    print("把上面 A/B/C 三段的完整输出发回即可据此定配置。")
    print("=" * 78)


if __name__ == '__main__':
    main()
