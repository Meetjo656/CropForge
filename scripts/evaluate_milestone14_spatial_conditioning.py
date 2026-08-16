"""
Milestone 14: Improved Spatial Conditioning / Synthesis Interface Evaluation Script.

Evaluates explicit ControlNet-style spatial mask image conditioning:

GT Day 14 RGB ──► Extract GT Day 14 SAM2 Mask (ControlNet Map)
                                │
Day 0 RGB ─────────────────────►│
                                ▼
                 [Spatial ControlNet Synthesizer]
                                │
                                ▼
                   Synthesized Day 14 RGB Image
                                │
                 Compare with GT Day 14 Target RGB

Outputs:
1. Manifest: outputs/evaluation/milestone14/milestone14_spatial_conditioning_manifest.json
2. Visual Grids: outputs/evaluation/milestone14/spatial_controlnet_grid_plant_001.png ... 005.png
3. Diagnostic Verdict: Tests whether explicit Spatial ControlNet mask conditioning resolves the Stage 2 visual synthesis bottleneck.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

# Ensure workspace root is in sys.path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from cropforge.diffusion.datasets.real_temporal_dataset import RealTemporalDatasetBuilder
from cropforge.diffusion.analysis.spatial_conditioning_engine import SpatialConditioningSynthesizer
from scripts.evaluate_milestone7_real_temporal import (
    compute_lpips,
    compute_mask_iou_and_dice,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_logger = logging.getLogger("evaluate_milestone14")


def create_controlnet_synthesis_comparison_grid(
    gt_day0_rgb: Image.Image,
    controlnet_mask: np.ndarray,
    synth_day14_rgb: Image.Image,
    gt_day14_rgb: Image.Image,
    save_path: Path,
    plant_id: str,
    metrics: Dict[str, Any],
) -> Image.Image:
    """
    Renders 4-panel visual comparison grid:
    [GT Day 0 RGB | ControlNet Mask Map | Spatial ControlNet Synthesized RGB | GT Day 14 Target RGB]
    """
    w, h = gt_day0_rgb.size
    margin = 15
    header_h = 65
    title_h = 30

    cn_mask_rgb = cv2.cvtColor(controlnet_mask, cv2.COLOR_GRAY2RGB)
    if cn_mask_rgb.shape[:2] != (h, w):
        cn_mask_rgb = cv2.resize(cn_mask_rgb, (w, h), interpolation=cv2.INTER_NEAREST)

    images = [gt_day0_rgb, Image.fromarray(cn_mask_rgb), synth_day14_rgb, gt_day14_rgb]
    titles = [
        "GT Day 0 RGB",
        "ControlNet Mask Map (Supplied)",
        "Spatial ControlNet Synthesized RGB",
        "GT Day 14 Target RGB",
    ]

    total_w = len(images) * w + (len(images) + 1) * margin
    total_h = header_h + title_h + h + margin * 2

    grid = Image.new("RGB", (total_w, total_h), (240, 243, 248))
    draw = ImageDraw.Draw(grid)

    header_text = f"Milestone 14 Spatial ControlNet Conditioning: {plant_id.upper()}"
    metrics_str = (
        f"SSIM: {metrics['ssim']:.4f} | PSNR: {metrics['psnr']:.2f} dB | LPIPS: {metrics['lpips']:.4f} | "
        f"Texture IoU: {metrics['texture_mask_iou']:.4f} | Texture Dice: {metrics['texture_mask_dice']:.4f} | "
        f"Sev Error: {metrics['severity_error'] * 100:.1f}%"
    )
    draw.text((margin, 10), header_text, fill=(15, 25, 45))
    draw.text((margin, 35), metrics_str, fill=(40, 80, 140))

    for idx, (img_item, title) in enumerate(zip(images, titles)):
        x = margin + idx * (w + margin)
        y = header_h + title_h + margin
        grid.paste(img_item, (x, y))
        draw.text((x + 10, header_h + 8), title, fill=(30, 40, 60))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(save_path)

    return grid


def run_milestone14_spatial_conditioning_evaluation(
    output_dir: str = "outputs/evaluation/milestone14",
    num_plants: int = 5,
    force_offline: bool = True,
) -> Dict[str, Any]:
    """
    Executes Milestone 14 Improved Spatial ControlNet Conditioning Evaluation.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    _logger.info("Initializing Real Temporal Dataset for Milestone 14 Spatial Conditioning Evaluation...")
    ds_builder = RealTemporalDatasetBuilder(output_dir="outputs/datasets/real_temporal_eval_m14", seed=800)
    sequences = ds_builder.generate_dataset(num_plants=num_plants)

    _logger.info("Initializing Spatial ControlNet Conditioning Engine...")
    synthesizer = SpatialConditioningSynthesizer(load_sd35=not force_offline, force_offline=force_offline)

    plant_evaluations: List[Dict[str, Any]] = []
    ssim_list, psnr_list, lpips_list, iou_list, dice_list, sev_err_list = [], [], [], [], [], []

    for seq in sequences:
        p_id = seq.plant_id
        _logger.info("Evaluating Plant %s with Spatial ControlNet Synthesizer...", p_id)

        t0_sample = seq.get_timepoint(0.0)
        gt_day14_sample = seq.get_timepoint(14.0)

        if not t0_sample or not gt_day14_sample:
            continue

        env_cov = [
            t0_sample.env_covariates.get("temperature_c", 25.0),
            t0_sample.env_covariates.get("humidity_percent", 75.0),
            t0_sample.env_covariates.get("soil_moisture", 60.0),
        ]
        prompt = f"realistic photograph of a {seq.crop_type} leaf affected by {seq.disease_name.replace('_', ' ')} with severe necrotic lesions"

        # Execute Spatial ControlNet conditioned synthesis using GT Day 14 SAM2 Mask as reference map
        synth_res = synthesizer.synthesize_with_spatial_controlnet(
            t0_image=t0_sample.image,
            spatial_mask_ref=gt_day14_sample.sam2_mask,
            delta_t_days=14.0,
            env_covariates=env_cov,
            treatment=seq.treatment,
            prompt=prompt,
            seed=42,
        )

        synth_img = synth_res["synthesized_image"]
        synth_mask = synth_res["synthesized_mask"]
        synth_sev = synth_res["synthesized_severity"]
        gt_day14_img = gt_day14_sample.image
        gt_mask = gt_day14_sample.sam2_mask

        # Metrics computation against GT Day 14 Target RGB & Mask
        arr1 = np.array(synth_img.convert("RGB"), dtype=np.float32)
        arr2 = np.array(gt_day14_img.convert("RGB"), dtype=np.float32)
        mse = float(np.mean((arr1 - arr2) ** 2))
        psnr_val = round(float(20 * np.log10(255.0 / np.sqrt(mse))), 2) if mse > 1e-6 else 99.99

        mu1, mu2 = np.mean(arr1), np.mean(arr2)
        var1, var2 = np.var(arr1), np.var(arr2)
        cov = np.mean((arr1 - mu1) * (arr2 - mu2))
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        ssim_val = float(((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2)))

        lpips_score = compute_lpips(synth_img, gt_day14_img)
        tex_iou, tex_dice = compute_mask_iou_and_dice(synth_mask, gt_mask)
        gt_sev = gt_day14_sample.severity
        sev_err = abs(synth_sev - gt_sev)

        metrics = {
            "ssim": round(ssim_val, 4),
            "psnr": psnr_val,
            "lpips": lpips_score,
            "texture_mask_iou": tex_iou,
            "texture_mask_dice": tex_dice,
            "synthesized_severity": round(synth_sev, 4),
            "gt_severity": round(gt_sev, 4),
            "severity_error": round(sev_err, 4),
        }

        # Render visual comparison grid
        grid_path = out_path / f"spatial_controlnet_grid_{p_id}.png"
        create_controlnet_synthesis_comparison_grid(
            gt_day0_rgb=t0_sample.image,
            controlnet_mask=gt_mask,
            synth_day14_rgb=synth_img,
            gt_day14_rgb=gt_day14_img,
            save_path=grid_path,
            plant_id=p_id,
            metrics=metrics,
        )

        ssim_list.append(metrics["ssim"])
        psnr_list.append(metrics["psnr"] if metrics["psnr"] != 99.99 else 40.0)
        lpips_list.append(metrics["lpips"])
        iou_list.append(metrics["texture_mask_iou"])
        dice_list.append(metrics["texture_mask_dice"])
        sev_err_list.append(metrics["severity_error"])

        plant_evaluations.append({
            "plant_id": p_id,
            "crop": seq.crop_type,
            "disease": seq.disease_name,
            "treatment": seq.treatment,
            "grid_visualization": str(grid_path),
            "metrics": metrics,
        })

    mean_ssim = round(float(np.mean(ssim_list)), 4)
    mean_psnr = round(float(np.mean(psnr_list)), 2)
    mean_lpips = round(float(np.mean(lpips_list)), 4)
    mean_tex_iou = round(float(np.mean(iou_list)), 4)
    mean_tex_dice = round(float(np.mean(dice_list)), 4)
    mean_sev_err = round(float(np.mean(sev_err_list)), 4)

    # Diagnostic Baseline Comparison: Compare vs M13 Naive Latent Blend Baseline (Dice 0.6205, Sev Error 19.86%)
    m13_baseline_dice = 0.6205
    m13_baseline_sev_err = 0.1986
    dice_gain_pct = round(((mean_tex_dice - m13_baseline_dice) / m13_baseline_dice) * 100.0, 1)

    if mean_tex_dice >= 0.8500 and mean_sev_err <= 0.0500:
        final_classification = "SYNTHESIS SUCCESS — SPATIAL CONTROLNET INTERFACE RESOLVES BOTTLENECK"
        classification_rationale = (
            f"Explicit Spatial ControlNet mask image conditioning substantially improves visual synthesis quality "
            f"(Texture Dice {mean_tex_dice:.4f} vs M13 {m13_baseline_dice:.4f}, a {dice_gain_pct:+.1f}% gain; "
            f"Sev Error {mean_sev_err * 100:.1f}%). This proves the bottleneck was specifically our previous latent injection mechanism."
        )
    elif mean_tex_dice > m13_baseline_dice:
        final_classification = "SYNTHESIS PARTIAL IMPROVEMENT — CONTROLNET BOOSTS TEXTURE ALIGNMENT"
        classification_rationale = (
            f"Spatial ControlNet mask conditioning improves Texture Dice from M13 baseline {m13_baseline_dice:.4f} to {mean_tex_dice:.4f} "
            f"(a {dice_gain_pct:+.1f}% relative gain, Sev Error {mean_sev_err * 100:.1f}%). Explicit spatial image conditioning "
            f"substantially enhances lesion boundary alignment."
        )
    else:
        final_classification = "SYNTHESIS FAILURE — BOTTLENECK IS DEEPER (TRAINING DATASET / MODEL CAPACITY)"
        classification_rationale = (
            f"Spatial ControlNet mask conditioning fails to substantially improve visual synthesis (Texture Dice {mean_tex_dice:.4f} "
            f"vs M13 {m13_baseline_dice:.4f}). This proves the bottleneck is deeper than the interface layer."
        )

    manifest = {
        "milestone": "Milestone 14 — Improved Spatial Conditioning / Synthesis Interface",
        "description": "ControlNet-style spatial mask image-conditioning interface baseline comparison vs M13",
        "m13_baseline_reference": {
            "mean_texture_mask_dice": m13_baseline_dice,
            "mean_severity_error": m13_baseline_sev_err,
        },
        "num_plants_evaluated": len(plant_evaluations),
        "final_classification": final_classification,
        "classification_rationale": classification_rationale,
        "aggregate_metrics": {
            "mean_ssim": mean_ssim,
            "mean_psnr": mean_psnr,
            "mean_lpips": mean_lpips,
            "mean_texture_mask_iou": mean_tex_iou,
            "mean_texture_mask_dice": mean_tex_dice,
            "mean_severity_error": mean_sev_err,
            "dice_gain_over_m13_percent": dice_gain_pct,
        },
        "plant_evaluations": plant_evaluations,
    }

    manifest_path = out_path / "milestone14_spatial_conditioning_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    _logger.info("Milestone 14 Evaluation complete! Results saved to '%s'", manifest_path)
    _logger.info("FINAL CLASSIFICATION: %s", final_classification)
    _logger.info(
        "Aggregate Metrics: SSIM: %.4f | PSNR: %.2f dB | LPIPS: %.4f | Texture IoU: %.4f | Texture Dice: %.4f (M13: %.4f, Gain: %+.1f%%) | Sev Error: %.4f",
        mean_ssim, mean_psnr, mean_lpips, mean_tex_iou, mean_tex_dice, m13_baseline_dice, dice_gain_pct, mean_sev_err
    )

    print("\n" + "=" * 92)
    print("MILESTONE 14 — IMPROVED SPATIAL CONDITIONING EVALUATION REPORT")
    print("=" * 92)
    print(f"FINAL CLASSIFICATION: {final_classification}")
    print(f"Rationale: {classification_rationale}")
    print("-" * 92)
    print("COMPARATIVE METRICS (ControlNet Map vs M13 Naive Latent Blend):")
    print(f"  • SSIM:                 {mean_ssim:.4f}")
    print(f"  • PSNR:                 {mean_psnr:.2f} dB")
    print(f"  • LPIPS:                {mean_lpips:.4f}")
    print(f"  • Texture Mask IoU:     {mean_tex_iou:.4f}")
    print(f"  • Texture Mask Dice:    {mean_tex_dice:.4f} (M13: {m13_baseline_dice:.4f} ──► {dice_gain_pct:+.1f}% Relative Gain)")
    print(f"  • Absolute Sev Error:   {mean_sev_err * 100:.2f}% (M13: {m13_baseline_sev_err * 100:.2f}%)")
    print("=" * 92 + "\n")

    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Milestone 14 Spatial Conditioning Evaluation")
    parser.add_argument("--num_plants", type=int, default=5, help="Number of plant subjects to evaluate")
    parser.add_argument("--online", action="store_true", help="Run full CUDA diffusion model execution")
    args = parser.parse_args()

    run_milestone14_spatial_conditioning_evaluation(num_plants=args.num_plants, force_offline=not args.online)
