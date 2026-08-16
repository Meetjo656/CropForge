"""
Analysis package for CropForge Disease Progression Forecasting.
"""

from cropforge.diffusion.analysis.forecasting_failure_analysis import (
    ForecastingFailureAnalyzer,
    compute_spatial_centroid_distance,
    compute_mask_centroid,
)
from cropforge.diffusion.analysis.ablation_study import LossAblationRunner
from cropforge.diffusion.analysis.isolated_spatial_evaluator import IsolatedSpatialEvaluator
from cropforge.diffusion.analysis.spatial_grid_generator import create_isolated_mask_grid
from cropforge.diffusion.analysis.temporal_horizon_forecaster import RecursiveSpatialForecaster
from cropforge.diffusion.analysis.gt_mask_synthesizer import GTMaskConditionedSynthesizer
from cropforge.diffusion.analysis.spatial_conditioning_engine import SpatialConditioningSynthesizer

__all__ = [
    "ForecastingFailureAnalyzer",
    "compute_spatial_centroid_distance",
    "compute_mask_centroid",
    "LossAblationRunner",
    "IsolatedSpatialEvaluator",
    "create_isolated_mask_grid",
    "RecursiveSpatialForecaster",
    "GTMaskConditionedSynthesizer",
    "SpatialConditioningSynthesizer",
]
