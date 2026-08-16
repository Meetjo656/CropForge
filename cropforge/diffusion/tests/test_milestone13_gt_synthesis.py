"""
Unit tests for Milestone 13: Mask-Conditioned SD3.5 Synthesis Evaluation.
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

from cropforge.diffusion.analysis.gt_mask_synthesizer import GTMaskConditionedSynthesizer
from cropforge.diffusion.datasets.real_temporal_dataset import RealTemporalDatasetBuilder


class TestMilestone13GTSynthesis(unittest.TestCase):
    """
    Test suite verifying GTMaskConditionedSynthesizer execution, metrics computation, and manifest output structure.
    """

    def test_gt_mask_synthesizer_step(self):
        synthesizer = GTMaskConditionedSynthesizer(load_sd35=False, force_offline=True)
        t0_img = Image.new("RGB", (64, 64), color=(40, 160, 40))
        gt_mask = np.zeros((64, 64), dtype=np.uint8)
        gt_mask[20:40, 20:40] = 255

        res = synthesizer.synthesize_with_gt_mask(
            t0_image=t0_img,
            gt_day14_mask=gt_mask,
            delta_t_days=14.0,
            treatment="fungicide",
            seed=42,
        )

        self.assertIn("synthesized_image", res)
        self.assertIn("synthesized_mask", res)
        self.assertIn("synthesized_severity", res)
        self.assertEqual(res["synthesized_image"].size, (512, 512))

    def test_gt_synthesis_manifest_structure(self):
        from scripts.evaluate_milestone13_gt_synthesis import run_milestone13_gt_synthesis_evaluation

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = run_milestone13_gt_synthesis_evaluation(output_dir=tmp_dir, num_plants=2, force_offline=True)

            self.assertIn("final_classification", manifest)
            self.assertIn(manifest["final_classification"], [
                "SYNTHESIS SUCCESS — BOTTLENECK IS MASK FORECAST ACCURACY / MASK INTERFACE",
                "SYNTHESIS FAILURE — SD3.5 CANNOT RENDER GIVEN LESION GEOMETRY",
            ])
            self.assertIn("aggregate_metrics", manifest)
            self.assertIn("plant_evaluations", manifest)


if __name__ == "__main__":
    unittest.main()
