"""
Unit tests for Milestone 9: Loss-Weight Ablation Study & Model Selection.
"""

import sys
import unittest
import tempfile
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cropforge.diffusion.analysis.ablation_study import LossAblationRunner
from cropforge.diffusion.datasets.real_temporal_dataset import RealTemporalDatasetBuilder


class TestMilestone9Ablation(unittest.TestCase):
    """
    Test suite verifying LossAblationRunner experiment matrix setup, single experiment execution, and manifest collation.
    """

    def test_ablation_experiments_matrix_keys(self):
        runner = LossAblationRunner()
        self.assertIn("Baseline_M7", runner.EXPERIMENTS)
        self.assertIn("Experiment_A", runner.EXPERIMENTS)
        self.assertIn("Experiment_B", runner.EXPERIMENTS)
        self.assertIn("Experiment_C", runner.EXPERIMENTS)
        self.assertIn("Experiment_D", runner.EXPERIMENTS)
        self.assertIn("Experiment_E", runner.EXPERIMENTS)
        self.assertIn("Experiment_F", runner.EXPERIMENTS)
        self.assertEqual(len(runner.EXPERIMENTS), 7)

    def test_ablation_single_experiment_step(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = RealTemporalDatasetBuilder(output_dir=tmp_dir, seed=42)
            seqs = builder.generate_dataset(num_plants=2)

            runner = LossAblationRunner(output_dir=tmp_dir, num_plants=2)
            res = runner.run_single_experiment(
                exp_name="Test_Exp_B",
                config={"lambda_mask": 0.5, "lambda_severity": 0.25, "use_lesion_head": True},
                sequences=seqs,
            )

            self.assertEqual(res["experiment_name"], "Test_Exp_B")
            self.assertIn("aggregate_metrics", res)
            m = res["aggregate_metrics"]
            self.assertIn("mean_ssim", m)
            self.assertIn("mean_mask_iou", m)
            self.assertIn("mean_mask_dice", m)
            self.assertIn("mean_severity_error", m)


if __name__ == "__main__":
    unittest.main()
