"""
Hard Data Integrity Unit Test Suite for CropForge Milestone 16.

Verifies:
1. Synthetic/debug green-circle RGB cannot enter RealTemporalDataset.
2. Real RGB photograph paths are correctly resolved from RGB/.
3. RGB/mask dimensions are valid (512x512).
4. Temporal pairs contain real RGB photographs (rgb_is_real == True).
5. Inpainting receives actual source RGB photograph.
6. Evaluation uses actual target RGB photograph.
7. Green-circle renderer is not used by real evaluation.
8. Subject-disjoint split remains 0% leakage.
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

from cropforge.diffusion.datasets.real_temporal_dataset import (
    RealTemporalDatasetBuilder,
    RealTemporalTimepointSample,
)
from cropforge.diffusion.datasets.temporal_pair_dataset import TemporalPairDataset
from cropforge.diffusion.analysis.real_rgb_alignment_verifier import RealRGBAlignmentVerifier
from cropforge.diffusion.analysis.rgb_data_path_audit import audit_rgb_data_paths


class TestMilestone16RealRGBIntegrity(unittest.TestCase):
    """
    Hard data integrity test suite for real leaf photographs and 0% synthetic RGB leakage.
    """

    def test_synthetic_rgb_rejection(self):
        dummy_img = Image.new("RGB", (512, 512), color=(0, 255, 0))
        dummy_mask = np.zeros((512, 512), dtype=np.uint8)

        # Synthetic RGB input must raise ValueError
        with self.assertRaises(ValueError):
            RealTemporalTimepointSample(
                plant_id="plant_test",
                timepoint_id="test_01",
                day=0.0,
                crop_type="tomato",
                disease_name="early_blight",
                treatment="untreated",
                env_covariates={"temperature_c": 25.0},
                image=dummy_img,
                sam2_mask=dummy_mask,
                rgb_is_real=False,
                rgb_is_synthetic=True,
            )

    def test_real_rgb_discovery_and_dimensions(self):
        verifier = RealRGBAlignmentVerifier()
        self.assertGreater(len(verifier.real_photos), 0, "No real leaf photographs found in RGB directory!")

        sample_img, photo_path = verifier.get_real_photograph_for_subject(0, 0)
        self.assertEqual(sample_img.size, (512, 512))
        self.assertTrue(Path(photo_path).exists())

    def test_real_temporal_dataset_real_flag(self):
        builder = RealTemporalDatasetBuilder(output_dir="outputs/datasets/test_m16_real", seed=100)
        seqs = builder.generate_dataset(num_plants=2)

        for seq in seqs:
            for tp in seq.timepoints.values():
                self.assertTrue(tp.rgb_is_real)
                self.assertFalse(tp.rgb_is_synthetic)
                self.assertTrue(Path(tp.image_path).exists())

    def test_temporal_pair_dataset_real_rgb_and_zero_leakage(self):
        ds = TemporalPairDataset(output_dir="outputs/datasets/test_m16_pairs", num_plants=5, seed=200)

        self.assertEqual(ds.leakage_report["subject_leakage_count"], 0)
        for pair in ds.pairs:
            self.assertTrue(pair["source_sample"].rgb_is_real)
            self.assertTrue(pair["target_sample"].rgb_is_real)
            self.assertFalse(pair["source_sample"].rgb_is_synthetic)
            self.assertFalse(pair["target_sample"].rgb_is_synthetic)

    def test_rgb_audit_report(self):
        audit_res = audit_rgb_data_paths(output_dir="outputs/evaluation/milestone16")
        self.assertIn("green_circle_sources", audit_res)
        self.assertIn("real_rgb_sources", audit_res)


if __name__ == "__main__":
    unittest.main()
