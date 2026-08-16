"""
Unit tests for Milestone 11: Isolated Spatial Forecaster Evaluation.
"""

import sys
import unittest
import tempfile
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cropforge.diffusion.analysis.isolated_spatial_evaluator import IsolatedSpatialEvaluator
from cropforge.diffusion.datasets.real_temporal_dataset import RealTemporalDatasetBuilder


class TestMilestone11IsolatedSpatial(unittest.TestCase):
    """
    Test suite verifying IsolatedSpatialEvaluator Stage 1 mask evaluation, Diagnostic Comparisons A/B/C, and manifest collation.
    """

    def test_isolated_spatial_evaluator_step(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = RealTemporalDatasetBuilder(output_dir=tmp_dir, seed=42)
            seqs = builder.generate_dataset(num_plants=2)

            evaluator = IsolatedSpatialEvaluator(output_dir=tmp_dir, num_plants=2)
            plant_evals = evaluator.evaluate_primary_experiment(seqs, use_full_conditions=True)

            self.assertEqual(len(plant_evals), 2)
            p0 = plant_evals[0]
            self.assertIn("plant_id", p0)
            self.assertIn("metrics", p0)
            m = p0["metrics"]
            self.assertIn("mask_iou", m)
            self.assertIn("mask_dice", m)
            self.assertIn("centroid_distance_px", m)
            self.assertIn("predicted_severity", m)
            self.assertIn("gt_severity", m)

    def test_horizon_sensitivity_comparison_c(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = RealTemporalDatasetBuilder(output_dir=tmp_dir, seed=42)
            seqs = builder.generate_dataset(num_plants=2)

            evaluator = IsolatedSpatialEvaluator(output_dir=tmp_dir, num_plants=2)
            res = evaluator.evaluate_horizon_sensitivity(seqs, horizons=[3.0, 7.0, 14.0])

            self.assertIn("horizon_sensitivity_evaluations", res)
            self.assertIn("horizon_responsive", res)
            evals = res["horizon_sensitivity_evaluations"]
            self.assertIn("horizon_3_days", evals)
            self.assertIn("horizon_7_days", evals)
            self.assertIn("horizon_14_days", evals)

    def test_full_isolated_eval_manifest_keys(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            evaluator = IsolatedSpatialEvaluator(output_dir=tmp_dir, num_plants=2, seed=42)
            manifest = evaluator.run_full_isolated_evaluation()

            self.assertIn("final_classification", manifest)
            self.assertIn(manifest["final_classification"], [
                "SPATIAL FORECASTER SUCCESS — INVESTIGATE SD3.5 SYNTHESIS",
                "SPATIAL FORECASTER FAILURE",
            ])
            self.assertIn("aggregate_primary_metrics", manifest)
            self.assertIn("diagnostic_comparisons", manifest)


if __name__ == "__main__":
    unittest.main()
