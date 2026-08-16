"""
Milestone 11: Isolated Spatial Forecaster Evaluation Script.

Isolates Stage 1 SpatialMaskForecaster and determines whether the forecaster itself
can accurately predict future lesion geometry without SD3.5 execution or RGB metrics.

Outputs:
1. Manifest: outputs/evaluation/milestone11/milestone11_spatial_forecaster_manifest.json
2. Visual Grids: outputs/evaluation/milestone11/spatial_forecast_grid_plant_001.png ... 005.png
3. Classification: SPATIAL FORECASTER FAILURE OR SPATIAL FORECASTER SUCCESS — INVESTIGATE SD3.5 SYNTHESIS
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Ensure workspace root is in sys.path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from cropforge.diffusion.analysis.isolated_spatial_evaluator import IsolatedSpatialEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_logger = logging.getLogger("evaluate_milestone11")


def run_milestone11_isolated_evaluation(
    output_dir: str = "outputs/evaluation/milestone11",
    num_plants: int = 5,
) -> Dict[str, Any]:
    """
    Executes Milestone 11 Isolated Spatial Forecaster Evaluation.
    """
    _logger.info("Starting Milestone 11 Isolated Spatial Forecaster Evaluation...")
    evaluator = IsolatedSpatialEvaluator(output_dir=output_dir, num_plants=num_plants, seed=500)
    manifest = evaluator.run_full_isolated_evaluation()

    # Print summary & classification report
    print("\n" + "=" * 88)
    print("MILESTONE 11 — ISOLATED SPATIAL FORECASTER EVALUATION REPORT")
    print("=" * 88)
    print(f"FINAL CLASSIFICATION: {manifest['final_classification']}")
    print(f"Rationale: {manifest['classification_rationale']}")
    print("-" * 88)

    agg = manifest["aggregate_primary_metrics"]
    print(f"AGGREGATE PRIMARY METRICS (Δt = 14 days):")
    print(f"  • Mask IoU:            {agg['mean_mask_iou']:.4f}")
    print(f"  • Mask Dice:           {agg['mean_mask_dice']:.4f}")
    print(f"  • Centroid Distance:   {agg['mean_centroid_distance_px']:.1f} px")
    print(f"  • Predicted Severity:  {agg['mean_predicted_severity'] * 100:.2f}%")
    print(f"  • Ground-Truth Sev:    {agg['mean_gt_severity'] * 100:.2f}%")
    print(f"  • Absolute Sev Error:  {agg['mean_severity_error'] * 100:.2f}%")

    print("-" * 88)
    print("PER-PLANT BREAKDOWN:")
    print(f"{'Plant ID':<12} | {'Crop':<8} | {'Disease':<15} | {'Treatment':<10} | {'IoU':<7} | {'Dice':<7} | {'Centroid Dist':<13} | {'Sev Err':<7}")
    print("-" * 88)

    for p in manifest["per_plant_evaluations"]:
        m = p["metrics"]
        print(
            f"{p['plant_id']:<12} | {p['crop']:<8} | {p['disease']:<15} | {p['treatment']:<10} | "
            f"{m['mask_iou']:<7.4f} | {m['mask_dice']:<7.4f} | {m['centroid_distance_px']:<13.1f} | {m['severity_error'] * 100:<7.2f}%"
        )

    print("-" * 88)
    print("DIAGNOSTIC COMPARISON C — HORIZON SENSITIVITY (Δt = 3, 7, 14 days):")
    comp_c = manifest["diagnostic_comparisons"]["comparison_c_horizon_sensitivity"]["horizon_sensitivity_evaluations"]
    for k, v in comp_c.items():
        print(f"  • {k.upper()}: Mean Mask IoU: {v['mean_mask_iou']:.4f} | Mean Mask Dice: {v['mean_mask_dice']:.4f} | Mean Pred Sev: {v['mean_predicted_severity'] * 100:.2f}%")

    print("=" * 88 + "\n")
    _logger.info("Milestone 11 Isolated Evaluation script completed successfully!")

    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Milestone 11 Isolated Spatial Evaluation")
    parser.add_argument("--num_plants", type=int, default=5, help="Number of plant subjects to evaluate")
    args = parser.parse_args()

    run_milestone11_isolated_evaluation(num_plants=args.num_plants)
