"""
Milestone 5 Temporal Forecasting Demonstration Script.

Generates temporal disease progression trajectories across horizons (0, 3, 7, 14 days)
under different treatment interventions (untreated, fungicide, biocontrol).
"""

import sys
import json
import logging
from pathlib import Path

# Add workspace root to sys.path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Bypass broken xformers
sys.modules["xformers"] = None
sys.modules["xformers.ops"] = None

from cropforge.diffusion.Inference import TemporalInferencePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_logger = logging.getLogger("evaluate_temporal_forecasting")


def run_temporal_forecasting_demo(output_dir: str = "outputs/evaluation/milestone5") -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    _logger.info("Initializing Temporal Disease Progression Forecasting Pipeline...")
    pipeline = TemporalInferencePipeline()

    prompt = "realistic photograph of a tomato leaf affected by early blight"
    horizons = [0.0, 3.0, 7.0, 14.0]
    treatments = ["untreated", "fungicide", "biocontrol"]

    manifest = {"prompt": prompt, "horizons_days": horizons, "trajectories": {}}

    for treat in treatments:
        _logger.info("Generating temporal forecast trajectory for treatment '%s'...", treat)
        t_dir = out_path / treat
        res = pipeline.forecast_trajectory(
            prompt=prompt,
            horizons=horizons,
            env_covariates=[26.5, 82.0, 65.0],
            treatment=treat,
            seed=42,
            output_dir=t_dir,
        )

        manifest["trajectories"][treat] = [r["metadata"] for r in res]

    with open(out_path / "forecasting_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    _logger.info("Temporal forecasting demonstration complete! Results saved to '%s'", out_path)


if __name__ == "__main__":
    run_temporal_forecasting_demo()
