"""
Milestone 10: Mask-Conditioned Spatial Forecasting Evaluation Script.

Evaluates two-stage causal spatial forecasting chain against Ground Truth Day 14 observations:

Day 0 RGB + Day 0 SAM2 Mask ──► [Stage 1: Spatial Mask Forecaster] ──► Predicted Future Mask M_t1
                                                                              │
                                                                              ▼
                                   [Stage 2: Mask-Conditioned SD3.5] ──► Future RGB Image x_t1
                                                                              │
                                                                              ▼
                                                                     Compare with GT Day 14

Primary Evaluation Metrics:
- Mask IoU (SAM2 Lesion Mask Spatial Intersection)
- Mask Dice (SAM2 Lesion Boundary Overlap)
- Severity Error (|Forecasted Severity % - Ground Truth Severity %|)
- Spatial Centroid Distance (Pixel displacement offset)

Secondary Evaluation Metrics:
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)
- LPIPS (Learned Perceptual Image Patch Similarity)
"""

import sys
import json
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import cv2
import numpy as np
from PIL import Image

# Ensure workspace root is in sys.path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from cropforge.diffusion.datasets.real_temporal_dataset import (
    RealTemporalDatasetBuilder,
    RealTemporalPlantSequence,
)
from cropforge.diffusion.Inference.spatial_pipeline import MaskConditionedSpatialPipeline
from cropforge.diffusion.analysis.forecasting_failure_analysis import compute_spatial_centroid_distance
from scripts.evaluate_milestone7_real_temporal import (
    compute_lpips,
    extract_sam2_lesion_mask,
    compute_mask_iou_and_dice,
)
from scripts.evaluate_milestone8_lesion_aware import create_milestone8_comparison_grid

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_logger = logging.getLogger("evaluate_milestone10")


def run_milestone10_spatial_evaluation(
    output_dir: str = "outputs/evaluation/milestone10",
    num_plants: int = 5,
    force_offline: bool = True,
) -> Dict[str, Any]:
    """
    Executes Milestone 10 Mask-Conditioned Spatial Progression Evaluation.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    _logger.info("Initializing Real Temporal Dataset for Milestone 10 Spatial Evaluation...")
    ds_builder = RealTemporalDatasetBuilder(output_dir="outputs/datasets/real_temporal_eval_m10", seed=400)
    sequences = ds_builder.generate_dataset(num_plants=num_plants)

    _logger.info("Initializing Mask-Conditioned Spatial Pipeline...")
    pipeline = MaskConditionedSpatialPipeline(load_sd35=not force_offline, force_offline=force_offline)

    plant_evaluations: List[Dict[str, Any]] = []
    ssim_list, psnr_list, lpips_list, iou_list, dice_list, sev_err_list, dist_list = [], [], [], [], [], [], []

    for seq in sequences:
        p_id = seq.plant_id
        _logger.info("Evaluating Plant %s with Mask-Conditioned Spatial Pipeline...", p_id)

        t0_sample = seq.get_timepoint(0.0)
        gt_day14_sample = seq.get_timepoint(14.0)

        if not t0_sample or not gt_day14_sample:
            continue

        env_cov = [
            t0_sample.env_covariates.get("temperature_c", 25.0),
            t0_sample.env_covariates.get("humidity_percent", 75.0),
            t0_sample.env_covariates.get("soil_moisture", 60.0),
        ]
        prompt = f"realistic photograph of a {seq.crop_type} leaf affected by {seq.disease_name.replace('_', ' ')}"

        # Execute 2-stage spatial mask-conditioned forecasting
        spatial_res = pipeline.forecast_spatial_progression(
            t0_image=t0_sample.image,
            t0_mask=t0_sample.sam2_mask,
            delta_t_days=14.0,
            env_covariates=env_cov,
            treatment=seq.treatment,
            prompt=prompt,
            seed=42,
        )

        forecasted_img = spatial_res["future_image"]
        pred_mask = spatial_res["pred_future_mask"]
        pred_severity = spatial_res["pred_future_severity"]
        gt_mask = gt_day14_sample.sam2_mask

        # Metrics computation (Primary: Mask IoU/Dice, Severity Error; Secondary: SSIM, PSNR, LPIPS)
        arr1 = np.array(forecasted_img.convert("RGB"), dtype=np.float32)
        arr2 = np.array(gt_day14_sample.image.convert("RGB"), dtype=np.float32)
        mse = float(np.mean((arr1 - arr2) ** 2))
        psnr_val = round(float(20 * np.log10(255.0 / np.sqrt(mse))), 2) if mse > 1e-6 else 99.99

        mu1, mu2 = np.mean(arr1), np.mean(arr2)
        var1, var2 = np.var(arr1), np.var(arr2)
        cov = np.mean((arr1 - mu1) * (arr2 - mu2))
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        ssim_val = float(((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2)))

        lpips_score = compute_lpips(forecasted_img, gt_day14_sample.image)
        mask_iou, mask_dice = compute_mask_iou_and_dice(pred_mask, gt_mask)
        centroid_dist = compute_spatial_centroid_distance(gt_mask, pred_mask)
        severity_err = abs(pred_severity - gt_day14_sample.severity)

        metrics = {
            "primary_metrics": {
                "mask_iou": mask_iou,
                "mask_dice": mask_dice,
                "severity_error": round(severity_err, 4),
                "centroid_distance_px": round(centroid_dist, 1),
            },
            "secondary_metrics": {
                "ssim": round(ssim_val, 4),
                "psnr": psnr_val,
                "lpips": lpips_score,
            },
            "gt_severity": round(gt_day14_sample.severity, 4),
            "forecasted_severity": round(pred_severity, 4),
        }

        # Renders comparison grid
        grid_metrics = {
            "ssim": metrics["secondary_metrics"]["ssim"],
            "psnr": metrics["secondary_metrics"]["psnr"],
            "lpips": metrics["secondary_metrics"]["lpips"],
            "mask_iou": metrics["primary_metrics"]["mask_iou"],
            "mask_dice": metrics["primary_metrics"]["mask_dice"],
            "centroid_distance_px": metrics["primary_metrics"]["centroid_distance_px"],
            "severity_error": metrics["primary_metrics"]["severity_error"],
        }
        grid_path = out_path / f"spatial_mask_overlay_grid_{p_id}.png"
        create_milestone8_comparison_grid(
            gt_day0=t0_sample.image,
            forecast_day14=forecasted_img,
            gt_day14=gt_day14_sample.image,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            save_path=grid_path,
            plant_id=p_id,
            metrics=grid_metrics,
        )

        ssim_list.append(metrics["secondary_metrics"]["ssim"])
        psnr_list.append(metrics["secondary_metrics"]["psnr"])
        lpips_list.append(metrics["secondary_metrics"]["lpips"])
        iou_list.append(metrics["primary_metrics"]["mask_iou"])
        dice_list.append(metrics["primary_metrics"]["mask_dice"])
        sev_err_list.append(metrics["primary_metrics"]["severity_error"])
        if centroid_dist < 500.0:
            dist_list.append(centroid_dist)

        plant_evaluations.append({
            "plant_id": p_id,
            "crop": seq.crop_type,
            "disease": seq.disease_name,
            "treatment": seq.treatment,
            "grid_visualization": str(grid_path),
            "metrics": metrics,
        })

    mean_metrics = {
        "primary_metrics": {
            "mean_mask_iou": round(float(np.mean(iou_list)), 4),
            "mean_mask_dice": round(float(np.mean(dice_list)), 4),
            "mean_severity_error": round(float(np.mean(sev_err_list)), 4),
            "mean_centroid_distance_px": round(float(np.mean(dist_list)), 1) if dist_list else 0.0,
        },
        "secondary_metrics": {
            "mean_ssim": round(float(np.mean(ssim_list)), 4),
            "mean_psnr": round(float(np.mean(psnr_list)), 2),
            "mean_lpips": round(float(np.mean(lpips_list)), 4),
        },
    }

    manifest = {
        "milestone": "Milestone 10 — Mask-Conditioned Spatial Forecasting",
        "description": "Two-stage causal forecasting (Day 0 Mask Spatial Forecaster ──► Mask-Conditioned Visual Synthesis)",
        "selected_experiment_base": "Experiment E (λ_mask=2.0, λ_severity=1.0)",
        "num_plants_evaluated": len(plant_evaluations),
        "aggregate_metrics": mean_metrics,
        "plant_evaluations": plant_evaluations,
    }

    manifest_path = out_path / "milestone10_spatial_mask_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    _logger.info("Milestone 10 Spatial Evaluation complete! Results saved to '%s'", manifest_path)
    _logger.info(
        "Primary Metrics: Mean IoU: %.4f | Mean Dice: %.4f | Mean Sev Error: %.4f | Centroid Dist: %.1f px",
        mean_metrics["primary_metrics"]["mean_mask_iou"],
        mean_metrics["primary_metrics"]["mean_mask_dice"],
        mean_metrics["primary_metrics"]["mean_severity_error"],
        mean_metrics["primary_metrics"]["mean_centroid_distance_px"],
    )
    _logger.info(
        "Secondary Metrics: Mean SSIM: %.4f | Mean PSNR: %.2f dB | Mean LPIPS: %.4f",
        mean_metrics["secondary_metrics"]["mean_ssim"],
        mean_metrics["secondary_metrics"]["mean_psnr"],
        mean_metrics["secondary_metrics"]["mean_lpips"],
    )

    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Milestone 10 Spatial Mask Evaluation")
    parser.add_argument("--num_plants", type=int, default=5, help="Number of plant sequences to evaluate")
    parser.add_argument("--online", action="store_true", help="Run full CUDA diffusion model execution")
    args = parser.parse_args()

    run_milestone10_spatial_evaluation(num_plants=args.num_plants, force_offline=not args.online)
