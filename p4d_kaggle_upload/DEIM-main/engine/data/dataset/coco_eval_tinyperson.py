'''
TinyPerson 切块评测器:切块推理结果映射回整图并 merge,
再用**整图 GT** 按官方 TinyPerson 规则(evaluate_tiny.py)评测 AP/(可选)MR。

测试集用**离线预切块**的物理小图(由 tools/dataset/tile_tinyperson_test.py 生成),本评测器:
1. 验证直接读小图(每块只解码一次,消除在线切块对整图的重复解码)。tile_id ↔ (corner, 整图)
   映射由 build_tile_index(**整图 GT**) 复现,与切块脚本同源同参,保证一致(亦兼容旧的在线切块)。
2. AP 计算**进程内复用官方** huicv COCOExpandEval(evaluate_tiny.py 用的同一个类),
   严格保留 ignore/uncertain/IOD 与 'tiny' 口径(iouThrs=[0.25,0.5,0.75], maxDets=[200]),
   不重写、不简化官方规则。
3. merge 只执行一次:DEIM 侧用 torchvision NMS(merge_nms_th=0.5)把切块预测合并到整图,
   直接喂给官方 AP/MR,不再调用 evaluate_tiny.merge_det_result(规避其 mini_maskrcnn_benchmark 依赖)。

关键约束(DEIM 验证链):
- det_engine.evaluate() 时序:cleanup → update → synchronize_between_processes
  → accumulate → summarize → 读 coco_eval['bbox'].stats.tolist()。
- det_solver 用 stats[0] 选 best、torch.save(coco_eval['bbox'].eval)。
  因此 accumulate() 里必须调用官方 coco_eval.accumulate()(填 .eval,可 pickle),
  summarize() 里调用 coco_eval.summarize()(填 .stats,numpy)。

官方规则由整图 GT + COCOExpandEval 承载(已核实 tiny_set_test_all.json):
- 单类 person(id=1,sea/earth 已合并);in_dense_image=0(dense 图已排除);
  ignore/uncertain 为独立标记,由 use_ignore_attr/ignore_uncertain/use_iod_for_ignore 处理:
  匹配这些区域的检测既不计 TP 也不计 FP。
'''

import contextlib
import importlib
import json
import os
import sys
import tempfile

import torch
import torchvision

from ...core import register
from ...misc import dist_utils
from .tinyperson_tiling import build_tile_index

__all__ = ['TinyPersonMergeEvaluator']


def _tools_dir():
    """DEIM-main/tools 绝对路径(官方 huicv 评测代码所在目录)。"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tools'))


def _load_official_evaluators():
    """
    进程内复用官方 COCOExpandEval,返回 (原生 pycocotools.COCO, COCOExpandEval)。

    为什么必须绑定**原生** pycocotools:
    coco_dataset.py 顶部调用 faster_coco_eval.init_as_pycocotools(),会把 sys.modules 里的
    pycocotools.coco/.cocoeval 替换成 faster_coco_eval 的**编译版**影子。而官方
    expand_cocofmt_eval.py 顶部 `from pycocotools.cocoeval import Params, COCOeval` 并**重写**
    evaluate/_prepare/evaluateImg、且 accumulate 依赖原生纯 Python 实现消费 self.evalImgs。
    若绑定到 faster 影子,重写的 Python 逻辑与 IOD 处理会失效/不匹配。故这里:
      1) 移除 sys.modules 中的 pycocotools* 影子与可能已缓存的 expand_cocofmt_eval;
      2) 重新 import 原生 pycocotools(缺失则给出清晰安装提示);
      3) 把 tools/ 加入 sys.path 后 import COCOExpandEval(此刻绑定原生 COCOeval)。
    成功后保留原生绑定(其它组件用的是 `from faster_coco_eval import ...`,不受影响)。
    """
    # 1) 备份并移除 faster_coco_eval 注入的 pycocotools 影子
    saved_pycoco = {}
    for name in list(sys.modules.keys()):
        if name == 'pycocotools' or name.startswith('pycocotools.'):
            saved_pycoco[name] = sys.modules.pop(name)
    # expand_cocofmt_eval 在 import 时即绑定 COCOeval,清缓存以强制重绑原生实现
    saved_expand = {}
    for name in list(sys.modules.keys()):
        if name.startswith('huicv.evaluation.expand_cocofmt_eval'):
            saved_expand[name] = sys.modules.pop(name)

    # 2) 导入原生 pycocotools
    try:
        coco_mod = importlib.import_module('pycocotools.coco')
        cocoeval_mod = importlib.import_module('pycocotools.cocoeval')
    except Exception as e:
        sys.modules.update(saved_pycoco)      # 失败还原影子,避免破坏其它组件
        sys.modules.update(saved_expand)
        raise ImportError(
            "TinyPersonMergeEvaluator 需要**原生 pycocotools**(纯 Python COCOeval)以进程内复用官方 "
            "COCOExpandEval(evaluate_tiny.py 使用的同一个类)。faster_coco_eval 的编译版 COCOeval 无法"
            "配合被重写的 evaluate/_prepare/evaluateImg 与 IOD 逻辑。请先执行: pip install pycocotools\n"
            f"原始错误: {e!r}") from e

    if 'faster_coco_eval' in str(getattr(cocoeval_mod, '__file__', '')):
        sys.modules.update(saved_pycoco)
        sys.modules.update(saved_expand)
        raise RuntimeError(
            "导入到的 pycocotools.cocoeval 仍是 faster_coco_eval 影子,无法复用官方 COCOExpandEval。"
            "请确认已安装原生版本: pip install pycocotools")

    # 3) 把 tools/ 加入 sys.path,导入官方 COCOExpandEval(此刻绑定原生 COCOeval)
    tools = _tools_dir()
    if tools not in sys.path:
        sys.path.insert(0, tools)
    expand_mod = importlib.import_module('huicv.evaluation.expand_cocofmt_eval')

    return coco_mod.COCO, expand_mod.COCOExpandEval


@register()
class TinyPersonMergeEvaluator(object):
    '''
    TinyPerson 切块评测器(官方 AP/MR);支持离线预切块(默认)与在线切块两种验证数据集。

    参数:
        coco_gt: 由 yaml_config 自动注入 = get_coco_api_from_dataset(val_dataloader.dataset)。
                 离线切块下它是**切块 COCO**;block_info 不用它,而用整图 GT(self.gt_coco,来自
                 merge_gt_file)经 build_tile_index 复现映射(与切块脚本同源同参;在线切块时
                 coco_gt 本就是整图 GT,两者结果一致)。
        iou_types: ['bbox']。
        merge_gt_file: **整图 GT** json(= evaluate_tiny.py 的 --merge-gt,官方评估基准)。
        tile_w/tile_h/overlap: 切块参数,必须与离线切块脚本一致(默认 640×576, overlap 50)。
        merge_nms_th: 跨块合并 NMS 阈值(0.5,来自 evaluate_tiny.py L221)。
        score_th: 低分过滤阈值(控内存;官方不显式设,这里仅丢弃极低分框)。
        max_det: 每整图最多保留框数。
        eval_mr: 是否额外计算 MR(默认 False;训练每轮与 --test-only 均只算 AP,保留手动开关)。
        mr_ious/mr_sizes: MR 的 IoU 与 size 设置(默认对齐 evaluate_tiny.py L204-207)。
    '''

    def __init__(self, coco_gt, iou_types, merge_gt_file,
                 tile_w=640, tile_h=576, overlap=50,
                 merge_nms_th=0.5, score_th=0.01, max_det=None,
                 eval_max_dets=200,
                 eval_mr=False, mr_ious=(0.25, 0.5, 0.75),
                 mr_sizes=('tiny1', 'tiny2', 'tiny3', 'tiny', 'small', 'All')):
        assert isinstance(iou_types, (list, tuple))
        self.iou_types = iou_types
        self.merge_gt_file = merge_gt_file
        self.tile_w, self.tile_h, self.overlap = int(tile_w), int(tile_h), int(overlap)
        self.merge_nms_th = merge_nms_th    # 官方 merge_nms_th=0.5(evaluate_tiny.py L221)
        self.score_th = score_th
        self.max_det = max_det
        self.eval_max_dets = eval_max_dets
        self.eval_mr = eval_mr
        self.mr_ious = tuple(mr_ious)
        self.mr_sizes = tuple(mr_sizes)

        # 原生 COCO + 官方 COCOExpandEval(进程内复用)
        self._COCO, self._COCOExpandEval = _load_official_evaluators()

        # 官方评估基准:整图 GT(与 evaluate_tiny.py --merge-gt 同一文件)
        self.gt_coco = self._COCO(merge_gt_file)
        # imgIds 含**全部整图**:无预测的整图其 GT 计为 FN,否则 AP 虚高
        self.img_ids = sorted(int(i) for i in self.gt_coco.getImgIds())

        # tile_id -> (corner(x1,y1,x2,y2), orig_id):用**整图 GT**(self.gt_coco,来自 merge_gt_file)
        # 经 build_tile_index 复现,与离线切块脚本(同函数/同参)同源,映射完全一致。
        # 不用注入的 coco_gt:离线时它是切块 COCO;在线时它与整图 GT 同文件,结果相同。
        self.block_info = {}
        for tile_id, orig_id, _fn, corner in build_tile_index(
                self.gt_coco, self.tile_w, self.tile_h, self.overlap):
            self.block_info[int(tile_id)] = (tuple(int(v) for v in corner), int(orig_id))

        self.coco_eval = {}
        self._last_results = []
        self._reset_buffer()

    def _reset_buffer(self):
        # orig_id -> [boxes(xyxy,原图坐标), scores, labels]
        self.merged = {}
        self._seen_tiles = set()

    def cleanup(self):
        self._reset_buffer()
        self.coco_eval = {}
        self._last_results = []

    def update(self, predictions):
        """predictions: {tile_id: {'boxes'(xyxy,切块局部), 'scores', 'labels'}}"""
        for tile_id, pred in predictions.items():
            tile_id = int(tile_id)
            if tile_id not in self.block_info:
                continue
            if tile_id in self._seen_tiles:      # 卡内去重(DistributedSampler padding 重复)
                continue
            self._seen_tiles.add(tile_id)

            corner, orig_id = self.block_info[tile_id]

            boxes = pred['boxes']
            if boxes is None or len(boxes) == 0:
                continue
            scores = pred['scores']
            labels = pred['labels']

            # 低分过滤:大幅减少累积内存(整图切块数 × num_top_queries)
            keep = scores > self.score_th
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if len(boxes) == 0:
                continue

            # 坐标映射:切块局部 → 原图全局(加 corner 左上角 x1,y1)
            boxes = boxes.detach().to('cpu', torch.float32).clone()
            boxes[:, 0::2] += float(corner[0])
            boxes[:, 1::2] += float(corner[1])
            scores = scores.detach().to('cpu', torch.float32)
            labels = labels.detach().to('cpu', torch.int64)

            if orig_id not in self.merged:
                self.merged[orig_id] = [boxes, scores, labels]
            else:
                m = self.merged[orig_id]
                m[0] = torch.cat([m[0], boxes])
                m[1] = torch.cat([m[1], scores])
                m[2] = torch.cat([m[2], labels])

    def synchronize_between_processes(self):
        """DDP:all_gather 汇聚各卡按整图累积的切块预测(list of tuple + numpy,兼容 all_gather)。

        跨卡 padding 产生的完全重复框会在 accumulate 的 NMS(th=0.5)中被自然去除。"""
        if not dist_utils.is_dist_available_and_initialized():
            return
        local_list = [(oid, v[0].numpy(), v[1].numpy(), v[2].numpy()) for oid, v in self.merged.items()]
        gathered = dist_utils.all_gather(local_list)
        merged_all = {}
        for proc_list in gathered:
            for oid, b, s, l in proc_list:
                b, s, l = torch.from_numpy(b), torch.from_numpy(s), torch.from_numpy(l)
                if oid not in merged_all:
                    merged_all[oid] = [b, s, l]
                else:
                    m = merged_all[oid]
                    m[0] = torch.cat([m[0], b])
                    m[1] = torch.cat([m[1], s])
                    m[2] = torch.cat([m[2], l])
        self.merged = merged_all

    def _build_predictions(self):
        """按整图跨块 NMS(merge_nms_th) → 截断 max_det → {orig_id: pred}(merge 只做一次)。"""
        predictions = {}
        for orig_id, (boxes, scores, labels) in self.merged.items():
            if len(boxes) > 0:
                keep = torchvision.ops.nms(boxes.float(), scores.float(), self.merge_nms_th)
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                if self.max_det is not None and self.max_det > 0 and len(scores) > self.max_det:
                    idx = torch.argsort(scores, descending=True)[:self.max_det]
                    boxes, scores, labels = boxes[idx], scores[idx], labels[idx]
            predictions[orig_id] = {'boxes': boxes, 'scores': scores, 'labels': labels}
        return predictions

    def _prepare_for_coco_detection(self, predictions):
        coco_results = []
        for orig_id, pred in predictions.items():
            boxes = pred['boxes']
            if boxes is None or len(boxes) == 0:
                continue
            boxes = convert_to_xywh(boxes).tolist()
            scores = pred['scores'].tolist()
            labels = pred['labels'].tolist()
            coco_results.extend([
                {'image_id': orig_id, 'category_id': labels[k], 'bbox': box, 'score': scores[k]}
                for k, box in enumerate(boxes)
            ])
        return coco_results

    def accumulate(self):
        predictions = self._build_predictions()
        results = self._prepare_for_coco_detection(predictions)
        self._last_results = results          # 供可选 MR 复用(不重复合并)

        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
            coco_dt = self.gt_coco.loadRes(results) if results else self._COCO()

        # ===== 官方 AP:进程内复用 COCOExpandEval(evaluate_tiny.py 同一个类) =====
        # 参数逐一对齐 evaluate_tiny.py:
        #   evaluate_ap() 的 cocofmt_kwargs        -> L126-149(尤其 L133-143)
        #   metric='ap'/'all' 的实际调用            -> L233
        #     (ignore_uncertain=True, use_iod_for_ignore=True, eval_standard='tiny')
        #   'tiny' 口径(iouThrs/maxDets/areaRngLbl) -> expand_cocofmt_eval.py L12-31
        if self.eval_max_dets is None:
            max_dets = [200]
        elif isinstance(self.eval_max_dets, int):
            max_dets = [self.eval_max_dets]
        else:
            max_dets = list(self.eval_max_dets)

        coco_eval = self._COCOExpandEval(
            self.gt_coco, coco_dt, 'bbox',
            ignore_uncertain=True,                              # evaluate_tiny.py L233
            use_ignore_attr=True,                               # evaluate_tiny.py L135
            use_iod_for_ignore=True,                            # evaluate_tiny.py L136 / L233
            iod_th_of_iou_f="lambda iou: (2*iou)/(1+iou)",      # evaluate_tiny.py L137
            cocofmt_param=dict(
                evaluate_standard='tiny',                       # evaluate_tiny.py L139 / L233
                iouThrs=[0.25, 0.5, 0.75],                      # evaluate_tiny.py L140
                maxDets=max_dets,                               # 官方默认 [200], 支持自定义覆盖
            ),
        )
        # imgIds 必须含**全部整图**(含无预测者,计 FN),忠实官方口径
        coco_eval.params.imgIds = list(self.img_ids)

        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
            coco_eval.evaluate()
        # 原生 accumulate():填充 .eval(dict,可 pickle;供 det_solver torch.save)
        coco_eval.accumulate()
        self.coco_eval['bbox'] = coco_eval

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            # 官方 tiny AP/AR 表(summarize 内部设置 .stats;stats[0]=AP@0.25/all,供选 best)
            # 输出格式来源:expand_cocofmt_eval.py L298-343(_summarizeDets_tiny)
            coco_eval.summarize()
        if self.eval_mr:
            self._run_mr(self._last_results)

    def _run_mr(self, results):
        """
        可选 MR(默认关闭)。懒加载官方 cityperson_eval(依赖 matplotlib/scipy)。
        参数对齐 evaluate_tiny.py:
          evaluate_mr() -> cityperson_eval(...)   L167-171
          metric='mr'/'all' 的调用                 L234-238
            (ignore_uncertain=True, use_iod_for_ignore=True, use_citypersons_standard=False)
          mr_ious / mr_sizes 默认值                L204-207
        """
        if not results:
            print('[TinyPerson MR] 无预测结果,跳过 MR。')
            return
        try:
            from huicv.deps.Cityscapes.cityperson_eval import cityperson_eval
        except Exception as e:
            print(f'[TinyPerson MR] 跳过:加载 cityperson_eval 失败(通常缺 matplotlib/scipy)。'
                  f'请 `pip install matplotlib scipy`。原始错误: {e!r}')
            return

        tmp = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8')
        json.dump(results, tmp)
        tmp.close()
        print(f'[TinyPerson MR] det={tmp.name}  gt={self.merge_gt_file}')
        cityperson_eval(
            tmp.name, self.merge_gt_file, CUT_WH=(1, 1),
            ignore_uncertain=True, use_iod_for_ignore=True,
            use_citypersons_standard=False,
            iou_ths=list(self.mr_ious), setup_labels=list(self.mr_sizes))


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
