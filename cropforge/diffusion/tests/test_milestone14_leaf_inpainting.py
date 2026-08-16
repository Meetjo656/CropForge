"""
Unit tests for Milestone 14: Leaf-Preserving Conditional Synthesis.
"""

import sys
import unittest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cropforge.diffusion.Inference.leaf_inpainting_pipeline import LeafPreservingInpaintingPipeline
from cropforge.diffusion.datasets.real_temporal_dataset import RealTemporalDatasetBuilder


class TestMilestone14LeafInpainting(unittest.TestCase):
    """
    Test suite verifying LeafPreservingInpaintingPipeline execution across Experiments A, B, and C.
    """

    def test_leaf_inpainting_experiments(self):
        pipeline = LeafPreservingInpaintingPipeline(load_sd35=False, force_offline=True)
        t0_img = Image.new("RGB", (64, 64), color=(45, 155, 45))
        gt_mask = np.zeros((64, 64), dtype=np.uint8)
        gt_mask[20:40, 20:40] = 255

        # Exp A
        res_a = pipeline.synthesize_exp_a_identity(t0_img)
        self.assertIn("synthesized_image", res_a)
        self.assertEqual(res_a["experiment"], "Exp A (Identity Preservation)")

        # Exp B
        res_b = pipeline.synthesize_exp_b_gt_mask(t0_img, gt_mask)
        self.assertIn("synthesized_image", res_b)
        self.assertEqual(res_b["experiment"], "Exp B (Ground-Truth Future Mask Inpainting)")

        # Exp C
        res_c = pipeline.synthesize_exp_c_predicted_mask(t0_img, gt_mask)
        self.assertIn("synthesized_image", res_c)
        self.assertEqual(res_c["experiment"], "Exp C (Predicted Future Mask End-to-End Forecast)")

    def test_leaf_inpainting_manifest_structure(self):
        from scripts.evaluate_milestone14_leaf_inpainting import run_milestone14_leaf_inpainting_evaluation

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = run_milestone14_leaf_inpainting_evaluation(output_dir=tmp_dir, num_plants=2, force_offline=True)

            self.assertIn("final_classification", manifest)
            self.assertIn("experiments_summary", manifest)
            self.assertIn("per_experiment_evaluations", manifest)
            self.assertIn("exp_a", manifest["per_experiment_evaluations"])
            self.assertIn("exp_b", manifest["per_experiment_evaluations"])
            self.assertIn("exp_c", manifest["per_experiment_evaluations"])


if __name__ == "__main__":
    unittest.main()
