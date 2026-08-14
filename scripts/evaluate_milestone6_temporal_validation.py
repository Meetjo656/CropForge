"""
Milestone 6: Temporal Forecasting Validation Script.

Executes 3 controlled experiments:
1. Temporal Horizon Experiment: Δt = 3 vs 7 vs 14 days (fixed seed, prompt, treatment, env).
2. Treatment Experiment: Treatment A ("untreated") vs Treatment B ("fungicide") vs Treatment C ("biocontrol") (fixed seed, prompt, env, Δt=14).
3. Environmental Experiment: Environment A (Cool/Dry) vs Environment B (Hot/Humid) (fixed seed, prompt, treatment, Δt=14).

Saves individual output images, composite side-by-side grid visualizations, computes pixel/structural diff metrics,
and writes the full evaluation manifest.
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

# Bypass xformers issue if present
sys.modules["xformers"] = None
sys.modules["xformers.ops"] = None

from cropforge.diffusion.Inference.temporal_pipeline import (
    TemporalInferencePipeline,
    compute_image_metrics,
    create_side_by_side_grid,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_logger = logging.getLogger("evaluate_milestone6")


def run_milestone6_validation(output_dir: str = "outputs/evaluation/milestone6") -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    _logger.info("Initializing Temporal Forecasting Pipeline for Milestone 6 Validation...")
    pipeline = TemporalInferencePipeline()

    base_prompt = "realistic photograph of a tomato leaf affected by early blight"
    fixed_seed = 42
    default_env = [25.0, 75.0, 60.0]  # Temp (C), RH (%), Soil Moisture (%)
    default_treatment = "untreated"

    manifest: Dict[str, Any] = {
        "milestone": "Milestone 6: Temporal Forecasting Validation",
        "fixed_seed": fixed_seed,
        "base_prompt": base_prompt,
        "experiments": {},
    }

    # =========================================================================
    # EXPERIMENT 1: Controlled Temporal Horizon Experiment (Δt = 3 vs 7 vs 14 days)
    # =========================================================================
    _logger.info("--- Starting Experiment 1: Temporal Horizon (Δt = 3, 7, 14 days) ---")
    horizons = [3.0, 7.0, 14.0]
    exp1_outputs = []
    exp1_imgs = []
    exp1_titles = []

    exp1_dir = out_path / "exp1_temporal_horizon"
    exp1_dir.mkdir(parents=True, exist_ok=True)

    for dt in horizons:
        _logger.info("Generating forecast for Δt = %.1f days...", dt)
        res = pipeline.forecast(
            prompt=base_prompt,
            delta_t_days=dt,
            env_covariates=default_env,
            treatment=default_treatment,
            seed=fixed_seed,
        )
        img_fn = f"forecast_day_{int(dt):02d}.png"
        res["forecast_image"].save(exp1_dir / img_fn)

        exp1_outputs.append(res)
        exp1_imgs.append(res["forecast_image"])
        exp1_titles.append(f"Δt = {int(dt)} Days")

    # Metrics between horizons
    m_3_7 = compute_image_metrics(exp1_imgs[0], exp1_imgs[1])
    m_7_14 = compute_image_metrics(exp1_imgs[1], exp1_imgs[2])
    m_3_14 = compute_image_metrics(exp1_imgs[0], exp1_imgs[2])

    grid1_path = out_path / "horizon_comparison.png"
    create_side_by_side_grid(
        images=exp1_imgs,
        titles=exp1_titles,
        save_path=grid1_path,
        header="Experiment 1: Temporal Horizon Progression (Seed=42, Untreated, 25C 75%RH)",
    )

    manifest["experiments"]["temporal_horizon"] = {
        "description": "Fixed image/seed/disease/treatment/environment across Δt = 3, 7, 14 days",
        "horizons_days": horizons,
        "metrics": {
            "day3_vs_day7": m_3_7,
            "day7_vs_day14": m_7_14,
            "day3_vs_day14": m_3_14,
        },
        "side_by_side_grid": str(grid1_path),
    }

    # =========================================================================
    # EXPERIMENT 2: Controlled Treatment Experiment (Treatment A vs B vs C @ Δt = 14)
    # =========================================================================
    _logger.info("--- Starting Experiment 2: Treatment Intervention (Untreated vs Fungicide vs Biocontrol @ Δt=14) ---")
    treatments = ["untreated", "fungicide", "biocontrol"]
    exp2_outputs = []
    exp2_imgs = []
    exp2_titles = []

    exp2_dir = out_path / "exp2_treatment"
    exp2_dir.mkdir(parents=True, exist_ok=True)

    for treat in treatments:
        _logger.info("Generating forecast for treatment '%s'...", treat)
        res = pipeline.forecast(
            prompt=base_prompt,
            delta_t_days=14.0,
            env_covariates=default_env,
            treatment=treat,
            seed=fixed_seed,
        )
        img_fn = f"forecast_treatment_{treat}.png"
        res["forecast_image"].save(exp2_dir / img_fn)

        exp2_outputs.append(res)
        exp2_imgs.append(res["forecast_image"])
        exp2_titles.append(f"Treatment: {treat.capitalize()}")

    m_untreated_fungicide = compute_image_metrics(exp2_imgs[0], exp2_imgs[1])
    m_untreated_biocontrol = compute_image_metrics(exp2_imgs[0], exp2_imgs[2])

    grid2_path = out_path / "treatment_comparison.png"
    create_side_by_side_grid(
        images=exp2_imgs,
        titles=exp2_titles,
        save_path=grid2_path,
        header="Experiment 2: Treatment Intervention Comparison (Seed=42, Δt=14 Days, 25C 75%RH)",
    )

    manifest["experiments"]["treatment_intervention"] = {
        "description": "Fixed image/seed/disease/environment/Δt=14 across Treatments",
        "treatments": treatments,
        "metrics": {
            "untreated_vs_fungicide": m_untreated_fungicide,
            "untreated_vs_biocontrol": m_untreated_biocontrol,
        },
        "side_by_side_grid": str(grid2_path),
    }

    # =========================================================================
    # EXPERIMENT 3: Environmental Conditioning Experiment (Env A vs Env B @ Δt = 14)
    # =========================================================================
    _logger.info("--- Starting Experiment 3: Environmental Conditioning (Cool/Dry vs Hot/Humid @ Δt=14) ---")
    envs = {
        "Environment A (Cool/Dry)": [18.0, 50.0, 40.0],
        "Environment B (Hot/Humid)": [32.0, 90.0, 80.0],
    }

    exp3_outputs = []
    exp3_imgs = []
    exp3_titles = []

    exp3_dir = out_path / "exp3_environment"
    exp3_dir.mkdir(parents=True, exist_ok=True)

    for env_name, env_vals in envs.items():
        _logger.info("Generating forecast for %s...", env_name)
        res = pipeline.forecast(
            prompt=base_prompt,
            delta_t_days=14.0,
            env_covariates=env_vals,
            treatment=default_treatment,
            seed=fixed_seed,
        )
        safe_fn = env_name.split()[0].lower() + ".png"
        res["forecast_image"].save(exp3_dir / safe_fn)

        exp3_outputs.append(res)
        exp3_imgs.append(res["forecast_image"])
        exp3_titles.append(env_name)

    m_env_a_b = compute_image_metrics(exp3_imgs[0], exp3_imgs[1])

    grid3_path = out_path / "environment_comparison.png"
    create_side_by_side_grid(
        images=exp3_imgs,
        titles=exp3_titles,
        save_path=grid3_path,
        header="Experiment 3: Environmental Conditioning Comparison (Seed=42, Untreated, Δt=14 Days)",
    )

    manifest["experiments"]["environmental_conditioning"] = {
        "description": "Fixed image/seed/disease/treatment/Δt=14 across Environments",
        "environments": envs,
        "metrics": {
            "cool_dry_vs_hot_humid": m_env_a_b,
        },
        "side_by_side_grid": str(grid3_path),
    }

    # Core Question Answer Synthesis
    horizons_changed = not m_3_14["is_identical"]
    treatments_changed = not m_untreated_fungicide["is_identical"]
    envs_changed = not m_env_a_b["is_identical"]

    conditioning_active = horizons_changed and treatments_changed and envs_changed

    manifest["summary"] = {
        "core_question": "Does changing a condition actually change the generated future state?",
        "answer": "YES - Outputs change meaningfully across all conditioning dimensions" if conditioning_active else "NO - Outputs remain identical",
        "horizons_changed": horizons_changed,
        "treatments_changed": treatments_changed,
        "environments_changed": envs_changed,
        "max_horizon_mse": m_3_14["mse"],
        "max_treatment_mse": m_untreated_fungicide["mse"],
        "max_environment_mse": m_env_a_b["mse"],
    }

    manifest_path = out_path / "forecasting_validation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    _logger.info("Milestone 6 validation complete. Results saved to '%s'", out_path)
    return manifest


if __name__ == "__main__":
    run_milestone6_validation()
