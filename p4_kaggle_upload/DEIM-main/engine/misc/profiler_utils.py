"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
from typing import Tuple

def stats(
    cfg,
    input_shape: Tuple=(1, 3, 640, 640), ) -> Tuple[int, dict]:

    # KAGGLE_PROFILER_FIX: 用 eval_spatial_size(可能非方形，如 [576,640])估 FLOPs，而非正方形 base_size。
    # deploy() 后模型处于 eval 态，HybridEncoder 用按 eval_spatial_size 预生成的静态 pos_embed
    # (stride32 下 18x20=360 token)；喂正方形 640x640(20x20=400 token)会 token 数不匹配而 RuntimeError。
    # FLOPs 仅用于信息打印，故整段再包 try/except，任何异常都不得拖垮真实训练。
    ess = None
    if isinstance(getattr(cfg, 'yaml_cfg', None), dict):
        ess = cfg.yaml_cfg.get('eval_spatial_size')
    if ess is None:
        ess = getattr(cfg, 'eval_spatial_size', None)
    if ess is not None and len(ess) == 2:
        input_shape = (1, 3, int(ess[0]), int(ess[1]))
    else:
        try:
            base_size = cfg.train_dataloader.collate_fn.base_size
            input_shape = (1, 3, base_size, base_size)
        except Exception:
            input_shape = (1, 3, 640, 640)

    try:
        from calflops import calculate_flops  # 延迟导入：calflops 缺失也不拖垮 import engine.misc
        model_for_info = copy.deepcopy(cfg.model).deploy()
        flops, macs, _ = calculate_flops(model=model_for_info,
                                            input_shape=input_shape,
                                            output_as_string=True,
                                            output_precision=4,
                                            print_detailed=False)
        params = sum(p.numel() for p in model_for_info.parameters())
        del model_for_info
        return params, {"Model FLOPs:%s   MACs:%s   Params:%s" %(flops, macs, params)}
    except Exception as e:
        try:
            params = sum(p.numel() for p in cfg.model.parameters())
        except Exception:
            params = 0
        return params, {"Model FLOPs: N/A (%s)   Params:%s" %(e, params)}
