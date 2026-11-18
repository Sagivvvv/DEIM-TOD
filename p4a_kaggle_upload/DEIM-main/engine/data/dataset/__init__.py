"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# from ._dataset import DetDataset
from .coco_dataset import CocoDetection
from .coco_dataset import (
    mscoco_category2name,
    mscoco_category2label,
    mscoco_label2category,
)
from .coco_eval import CocoEvaluator
# coco_eval_aitod / coco_eval_visdrone 依赖外部包(faster_coco_eval_aitod / faster_coco_eval)，
# 仅 AITOD / VisDrone 评测用到；缺依赖时跳过其注册，避免连带整个 `import engine` 失败。
# TinyPerson 用的是下面的 TinyPersonMergeEvaluator，不受影响。
try:
    from .coco_eval_aitod import AitodCocoEvaluator
except ImportError as _e:  # 缺 faster_coco_eval_aitod
    AitodCocoEvaluator = None
    import warnings as _warnings
    _warnings.warn('AitodCocoEvaluator 不可用(缺 faster_coco_eval_aitod): %s' % _e)
try:
    from .coco_eval_visdrone import VisdroneCocoEvaluator
except ImportError as _e:
    VisdroneCocoEvaluator = None
    import warnings as _warnings
    _warnings.warn('VisdroneCocoEvaluator 不可用(缺依赖): %s' % _e)
from .coco_eval_tinyperson import TinyPersonMergeEvaluator
from .coco_utils import get_coco_api_from_dataset
from .voc_detection import VOCDetection
from .voc_eval import VOCEvaluator
