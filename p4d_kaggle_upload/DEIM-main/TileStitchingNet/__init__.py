# -*- coding: utf-8 -*-
"""
TileStitchingNet Package
Boundary-Aware Local Relation Attention Post-Processing Network for High-Resolution Tiled Object Detection
"""

import os
import sys
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

try:
    from .model import TileStitchingNet
    from .losses import TileStitchingLoss
    from .datasetBuilder import (
        TileSample,
        CroppedGTLabelAssigner,
        collect_slice_predictions_from_detector,
        generate_tile_grid,
        compute_cutline_boundary_distances,
        build_candidate_feature_vector,
        SyntheticTiledDatasetGenerator,
    )
    from .trainEval import (
        TileStitchingPredictor,
        train_tile_stitching_net,
        evaluate_postprocessor_on_dataset,
        run_greedy_nms,
        run_soft_nms,
        run_wbf,
        run_sahi_nmm_ios,
    )
    from .pipeline_deim_tilestitching import OfflineDatasetIO
except ImportError:
    from model import TileStitchingNet
    from losses import TileStitchingLoss
    from datasetBuilder import (
        TileSample,
        CroppedGTLabelAssigner,
        collect_slice_predictions_from_detector,
        generate_tile_grid,
        compute_cutline_boundary_distances,
        build_candidate_feature_vector,
        SyntheticTiledDatasetGenerator,
    )
    from trainEval import (
        TileStitchingPredictor,
        train_tile_stitching_net,
        evaluate_postprocessor_on_dataset,
        run_greedy_nms,
        run_soft_nms,
        run_wbf,
        run_sahi_nmm_ios,
    )
    from pipeline_deim_tilestitching import OfflineDatasetIO

__all__ = [
    "TileStitchingNet",
    "TileStitchingLoss",
    "TileSample",
    "CroppedGTLabelAssigner",
    "collect_slice_predictions_from_detector",
    "generate_tile_grid",
    "compute_cutline_boundary_distances",
    "build_candidate_feature_vector",
    "SyntheticTiledDatasetGenerator",
    "TileStitchingPredictor",
    "train_tile_stitching_net",
    "evaluate_postprocessor_on_dataset",
    "run_greedy_nms",
    "run_soft_nms",
    "run_wbf",
    "run_sahi_nmm_ios",
    "OfflineDatasetIO",
]

