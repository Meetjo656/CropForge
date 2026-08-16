"""
Milestone 9: Systematic Loss-Weight Ablation & Optimal Configuration Selection Script.

Executes ablation matrix evaluating 6 distinct loss weighting configurations:
- Baseline M7 (Pure RGB Latent MSE)
- Experiment A: λ_mask = 0.0, λ_severity = 0.0
- Experiment B: λ_mask = 0.5, λ_severity = 0.25
- Experiment C: λ_mask = 1.0, λ_severity = 0.5  (M8 default)
- Experiment D: λ_mask = 2.0, λ_severity = 0.5
- Experiment E: λ_mask = 2.0, λ_severity = 1.0
- Experiment F: λ_mask = 1.0, λ_severity = 1.0

Outputs comparative table across SSIM, PSNR, LPIPS, Mask IoU, Mask Dice, and Severity Error.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Ensure workspace root is in sys.path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from cropforge.diffusion.analysis.ablation_study import LossAblationRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_logger = logging.getLogger("evaluate_milestone9")


def run_milestone9_ablation_evaluation(
    output_dir: str = "outputs/evaluation/milestone9",
    num_plants: int = 5,
) -> Dict[str, Any]:
    """
    Runs full Milestone 9 Loss-Weight Ablation Study and formats comparative metrics table.
    """
    _logger.info("Starting Milestone 9 Loss-Weight Ablation Study...")
    runner = LossAblationRunner(output_dir=output_dir, num_plants=num_plants, seed=300)
    manifest = runner.run_full_ablation_study()

    exp_results = manifest.get("experiments", [])

    # Format comparative terminal/markdown table
    table_rows = []
    table_rows.append("=" * 88)
    table_rows.append(f"{'Experiment':<18} | {'λ_mask':<7} | {'λ_sev':<7} | {'SSIM':<7} | {'PSNR':<7} | {'LPIPS':<7} | {'IoU':<7} | {'Dice':<7} | {'Sev Err':<7}")
    table_rows.append("=" * 88)

    for exp in exp_results:
        e_name = exp["experiment_name"]
        l_m = f"{exp['lambda_mask']:.2f}"
        l_s = f"{exp['lambda_severity']:.2f}"
        m = exp["aggregate_metrics"]

        table_rows.append(
            f"{e_name:<18} | {l_m:<7} | {l_s:<7} | {m['mean_ssim']:<7.4f} | {m['mean_psnr']:<7.2f} | {m['mean_lpips']:<7.4f} | {m['mean_mask_iou']:<7.4f} | {m['mean_mask_dice']:<7.4f} | {m['mean_severity_error']:<7.4f}"
        )

    table_rows.append("=" * 88)

    table_str = "\n".join(table_rows)
    print("\n" + table_str + "\n")
    _logger.info("Milestone 9 Ablation Study completed successfully!")

    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Milestone 9 Ablation Evaluation")
    parser.add_argument("--num_plants", type=int, default=5, help="Number of plant sequences to evaluate")
    args = parser.parse_args()

    run_milestone9_ablation_evaluation(num_plants=args.num_plants)
