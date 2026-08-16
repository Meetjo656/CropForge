"""
Unit tests for Milestone 15: Real Temporal Pair Fine-Tuning for Leaf-Preserving SD3.5.
"""

import sys
import unittest
import tempfile
from pathlib import Path
import numpy as np

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cropforge.diffusion.datasets.temporal_pair_dataset import TemporalPairDataset
from cropforge.diffusion.training.train_temporal_inpainting import run_training_or_dryrun


class TestMilestone15TemporalInpainting(unittest.TestCase):
    """
    Test suite verifying TemporalPairDataset subject-disjoint splitting, dry-run mode, and manifest output structure.
    """

    def test_subject_disjoint_splitting(self):
        ds = TemporalPairDataset(output_dir="outputs/datasets/real_temporal_eval_test_m15", num_plants=5, seed=42)
        leakage_report = ds.leakage_report

        self.assertEqual(leakage_report["subject_leakage_count"], 0)
        self.assertGreater(len(ds.train_pairs), 0)
        self.assertGreater(len(ds.val_pairs), 0)
        self.assertGreater(len(ds.test_pairs), 0)

    def test_dry_run_training_execution(self):
        cfg_path = "cropforge/diffusion/configs/temporal_inpainting_training.yaml"
        dry_res = run_training_or_dryrun(config_path=cfg_path, dry_run=True)

        self.assertTrue(dry_res["dry_run"])
        self.assertTrue(dry_res["valid_gradient_flow"])
        self.assertIn("losses", dry_res)
        self.assertIn("loss_total", dry_res["losses"])
        self.assertIn("loss_diffusion", dry_res["losses"])
        self.assertIn("loss_identity", dry_res["losses"])
        self.assertIn("loss_mask_region", dry_res["losses"])
        self.assertIn("loss_reconstruction", dry_res["losses"])

    def test_milestone15_evaluation_manifest_structure(self):
        from scripts.evaluate_milestone15_temporal_inpainting import run_milestone15_evaluation

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = run_milestone15_evaluation(output_dir=tmp_dir, num_plants=5, force_offline=True)

            self.assertIn("final_classification", manifest)
            self.assertIn(manifest["final_classification"], [
                "TEMPORAL INPAINTING IMPROVEMENT",
                "TEMPORAL INPAINTING DID NOT IMPROVE",
            ])
            self.assertIn("overall_aggregate_metrics", manifest)
            self.assertIn("metrics_by_horizon", manifest)


if __name__ == "__main__":
    unittest.main()
