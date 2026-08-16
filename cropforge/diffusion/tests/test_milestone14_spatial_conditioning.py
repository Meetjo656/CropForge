"""
Unit tests for Milestone 14: Improved Spatial Conditioning / Synthesis Interface.
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

from cropforge.diffusion.analysis.spatial_conditioning_engine import SpatialConditioningSynthesizer
from cropforge.diffusion.datasets.real_temporal_dataset import RealTemporalDatasetBuilder


class TestMilestone14SpatialConditioning(unittest.TestCase):
    """
    Test suite verifying SpatialConditioningSynthesizer ControlNet execution, metric collation, and manifest output structure.
    """

    def test_spatial_controlnet_synthesizer_step(self):
        synthesizer = SpatialConditioningSynthesizer(load_sd35=False, force_offline=True)
        t0_img = Image.new("RGB", (64, 64), color=(40, 160, 40))
        gt_mask = np.zeros((64, 64), dtype=np.uint8)
        gt_mask[20:40, 20:40] = 255

        res = synthesizer.synthesize_with_spatial_controlnet(
            t0_image=t0_img,
            spatial_mask_ref=gt_mask,
            delta_t_days=14.0,
            treatment="fungicide",
            seed=42,
        )

        self.assertIn("synthesized_image", res)
        self.assertIn("synthesized_mask", res)
        self.assertIn("synthesized_severity", res)
        self.assertEqual(res["synthesized_image"].size, (512, 512))

    def test_spatial_conditioning_manifest_structure(self):
        from scripts.evaluate_milestone14_spatial_conditioning import run_milestone14_spatial_conditioning_evaluation

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = run_milestone14_spatial_conditioning_evaluation(output_dir=tmp_dir, num_plants=2, force_offline=True)

            self.assertIn("final_classification", manifest)
            self.assertIn("aggregate_metrics", manifest)
            self.assertIn("dice_gain_over_m13_percent", manifest["aggregate_metrics"])
            self.assertIn("plant_evaluations", manifest)


if __name__ == "__main__":
    unittest.main()
