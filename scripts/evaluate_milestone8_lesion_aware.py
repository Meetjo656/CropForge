"""
Milestone 8: Lesion-Aware Multi-Loss Forecasting Evaluation Script.

Evaluates CropForge Lesion-Aware Disease Progression Forecasts against Ground Truth Observations:

             Ground Truth Day 14 (RGB + SAM2 Mask)
                               │
                               ▼
Day 0 ──► Lesion-Aware Forecast (RGB + Pred Lesion Mask)
                               │
                               ▼
                         Compare them

Metrics Evaluated:
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)
- LPIPS (Learned Perceptual Image Patch Similarity)
- Mask IoU / Dice (SAM2 Lesion Masks)
- Spatial Centroid Distance (Lesion spatial displacement in pixels)
- Severity Error (|Forecasted Severity % - Ground Truth Severity %|)
"""

import sys
import json
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import models, transforms

# Bypass xformers binary issues if present
sys.modules["xformers"] = None
sys.modules["xformers.ops"] = None

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from cropforge.diffusion.datasets.real_temporal_dataset import (
    RealTemporalDatasetBuilder,
    RealTemporalPlantSequence,
)
from cropforge.diffusion.Inference.temporal_pipeline import TemporalInferencePipeline
from cropforge.diffusion.analysis.forecasting_failure_analysis import (
    compute_spatial_centroid_distance,
    compute_mask_centroid,
)
from scripts.evaluate_milestone7_real_temporal import (
    compute_lpips,
    extract_sam2_lesion_mask,
    compute_mask_iou_and_dice,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_logger = logging.getLogger("evaluate_milestone8")


def create_lesion_mask_overlay(img: Image.Image, mask: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0)) -> Image.Image:
    """
    Overlays binary lesion segmentation mask onto an RGB image with 50% transparency.
    """
    np_img = np.array(img.convert("RGB")).copy()
    binary_mask = (mask > 127).astype(np.uint8)

    overlay = np_img.copy()
    overlay[binary_mask == 1] = color

    blended = cv2.addWeighted(overlay, 0.5, np_img, 0.5, 0)
    return Image.fromarray(blended)


def create_milestone8_comparison_grid(
    gt_day0: Image.Image,
    forecast_day14: Image.Image,
    gt_day14: Image.Image,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    save_path: Union[str, Path],
    plant_id: str,
    metrics: Dict[str, Any],
) -> Image.Image:
    """
    Creates 4-panel visual comparison grid with explicit lesion mask overlays:
    [GT Day 0 | Forecast Day 14 + Mask | GT Day 14 + Mask | Mask Diff Overlay]
    """
    w, h = gt_day0.size
    margin = 15
    header_h = 65
    title_h = 30

    forecast_overlay = create_lesion_mask_overlay(forecast_day14, pred_mask, color=(255, 60, 60))
    gt_overlay = create_lesion_mask_overlay(gt_day14, gt_mask, color=(60, 255, 60))

    # Mask difference visual: Green = GT, Red = Forecasted
    diff_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    gt_b = (gt_mask > 127)
    pred_b = (pred_mask > 127)
    diff_canvas[gt_b & ~pred_b] = [0, 220, 0]    # Ground Truth only (Missed)
    diff_canvas[pred_b & ~gt_b] = [220, 40, 40]   # Forecast only (False positive)
    diff_canvas[gt_b & pred_b] = [220, 220, 0]   # Both (True positive overlap)
    diff_img = Image.fromarray(diff_canvas)

    images = [gt_day0, forecast_overlay, gt_overlay, diff_img]
    titles = [
        "GT Day 0 Baseline",
        "Forecast Day 14 (Pred Mask)",
        "GT Day 14 (SAM2 Mask)",
        "Lesion Overlay Diff (Green/Red)",
    ]

    total_w = len(images) * w + (len(images) + 1) * margin
    total_h = header_h + title_h + h + margin * 2

    grid = Image.new("RGB", (total_w, total_h), (240, 243, 248))
    draw = ImageDraw.Draw(grid)

    header_text = f"Milestone 8 Lesion-Aware Evaluation: {plant_id.upper()}"
    metrics_str = (
        f"SSIM: {metrics['ssim']:.4f} | PSNR: {metrics['psnr']} dB | LPIPS: {metrics['lpips']:.4f} | "
        f"Mask IoU: {metrics['mask_iou']:.4f} | Dice: {metrics['mask_dice']:.4f} | "
        f"Centroid Dist: {metrics['centroid_distance_px']:.1f}px | Sev Error: {metrics['severity_error']:.4f}"
    )
    draw.text((margin, 10), header_text, fill=(15, 25, 45))
    draw.text((margin, 35), metrics_str, fill=(40, 80, 140))

    for idx, (img_item, title) in enumerate(zip(images, titles)):
        x = margin + idx * (w + margin)
        y = header_h + title_h + margin
        grid.paste(img_item, (x, y))
        draw.text((x + 10, header_h + 8), title, fill=(30, 40, 60))

    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_p)

    return grid


def run_milestone8_evaluation(
    output_dir: str = "outputs/evaluation/milestone8",
    num_plants: int = 5,
    force_offline: bool = True,
) -> Dict[str, Any]:
    """
    Executes Milestone 8 Lesion-Aware Multi-Loss Ground Truth Evaluation across multi-timepoint plant sequences.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    _logger.info("Initializing Real Temporal Dataset for Milestone 8 Lesion-Aware Evaluation...")
    ds_builder = RealTemporalDatasetBuilder(output_dir="outputs/datasets/real_temporal_eval_m8", seed=200)
    sequences = ds_builder.generate_dataset(num_plants=num_plants)

    _logger.info("Initializing Temporal Forecasting Inference Pipeline...")
    pipeline = TemporalInferencePipeline(load_sd35=not force_offline)

    plant_evaluations: List[Dict[str, Any]] = []

    ssim_list, psnr_list, lpips_list, iou_list, dice_list, sev_err_list, dist_list = [], [], [], [], [], [], []

    for seq in sequences:
        p_id = seq.plant_id
        _logger.info("Evaluating Plant %s with Lesion-Aware pipeline...", p_id)

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

        forecast_res = pipeline.forecast(
            prompt=prompt,
            delta_t_days=14.0,
            env_covariates=env_cov,
            treatment=seq.treatment,
            seed=42,
            num_inference_steps=10,
            force_offline=force_offline,
        )
        forecasted_img = forecast_res["forecast_image"]

        # Extract SAM2 predicted mask and severity
        pred_mask, pred_severity = extract_sam2_lesion_mask(forecasted_img)
        gt_mask = gt_day14_sample.sam2_mask

        # Metrics computation
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
            "ssim": round(ssim_val, 4),
            "psnr": psnr_val,
            "lpips": lpips_score,
            "mask_iou": mask_iou,
            "mask_dice": mask_dice,
            "centroid_distance_px": round(centroid_dist, 1),
            "gt_severity": round(gt_day14_sample.severity, 4),
            "forecasted_severity": round(pred_severity, 4),
            "severity_error": round(severity_err, 4),
        }

        grid_path = out_path / f"lesion_overlay_grid_{p_id}.png"
        create_milestone8_comparison_grid(
            gt_day0=t0_sample.image,
            forecast_day14=forecasted_img,
            gt_day14=gt_day14_sample.image,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            save_path=grid_path,
            plant_id=p_id,
            metrics=metrics,
        )

        ssim_list.append(metrics["ssim"])
        psnr_list.append(metrics["psnr"] if metrics["psnr"] != 99.99 else 40.0)
        lpips_list.append(metrics["lpips"])
        iou_list.append(metrics["mask_iou"])
        dice_list.append(metrics["mask_dice"])
        sev_err_list.append(metrics["severity_error"])
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
        "mean_ssim": round(float(np.mean(ssim_list)), 4),
        "mean_psnr": round(float(np.mean(psnr_list)), 2),
        "mean_lpips": round(float(np.mean(lpips_list)), 4),
        "mean_mask_iou": round(float(np.mean(iou_list)), 4),
        "mean_mask_dice": round(float(np.mean(dice_list)), 4),
        "mean_centroid_distance_px": round(float(np.mean(dist_list)), 1) if dist_list else 0.0,
        "mean_severity_error": round(float(np.mean(sev_err_list)), 4),
    }

    manifest = {
        "milestone": "Milestone 8 — Forecasting Failure Analysis & Lesion-Aware Multi-Loss Evaluation",
        "description": "Ground Truth Day 14 vs CropForge Lesion-Aware Forecast Comparison",
        "evaluation_workflow": "Day 0 Baseline ──► Lesion-Aware Forecaster ──► Compare with Ground Truth Day 14 Mask & Image",
        "num_plants_evaluated": len(plant_evaluations),
        "aggregate_metrics": mean_metrics,
        "plant_evaluations": plant_evaluations,
    }

    manifest_path = out_path / "milestone8_lesion_aware_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    _logger.info("Milestone 8 Lesion-Aware Evaluation complete! Results saved to '%s'", out_path)
    _logger.info("Aggregate Metrics: Mean SSIM: %.4f | Mean PSNR: %.2f dB | Mean LPIPS: %.4f | Mean IoU: %.4f | Mean Dice: %.4f | Mean Sev Error: %.4f",
                 mean_metrics["mean_ssim"], mean_metrics["mean_psnr"], mean_metrics["mean_lpips"],
                 mean_metrics["mean_mask_iou"], mean_metrics["mean_mask_dice"], mean_metrics["mean_severity_error"])

    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Milestone 8 Lesion-Aware Evaluation")
    parser.add_argument("--num_plants", type=int, default=5, help="Number of plant sequences to evaluate")
    parser.add_argument("--online", action="store_true", help="Run full CUDA diffusion model execution")
    args = parser.parse_args()

    run_milestone8_evaluation(num_plants=args.num_plants, force_offline=not args.online)
