"""
Unit tests for Milestone 7: Real Temporal Dataset + Forecasting Training & Ground Truth Evaluation.
"""

import sys
import unittest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cropforge.diffusion.datasets.real_temporal_dataset import (
    RealTemporalDatasetBuilder,
    RealTemporalPlantSequence,
    RealTemporalTimepointSample,
)
from cropforge.diffusion.training.train_temporal_forecaster import (
    RealTemporalForecastingDataset,
    TemporalForecasterTrainer,
)
from scripts.evaluate_milestone7_real_temporal import (
    compute_lpips,
    compute_mask_iou_and_dice,
    compute_comprehensive_metrics,
)


class TestMilestone7RealTemporal(unittest.TestCase):
    """
    Test suite verifying Real Temporal dataset building, forecasting training, and ground-truth metrics.
    """

    def test_real_temporal_dataset_builder(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = RealTemporalDatasetBuilder(output_dir=tmp_dir, seed=42)
            sequences = builder.generate_dataset(num_plants=2)

            self.assertEqual(len(sequences), 2)
            seq = sequences[0]
            self.assertEqual(seq.plant_id, "plant_001")
            self.assertIn(0.0, seq.timepoints)
            self.assertIn(14.0, seq.timepoints)

            t0 = seq.get_timepoint(0.0)
            t14 = seq.get_timepoint(14.0)

            self.assertGreater(t14.severity, t0.severity)
            self.assertTrue(Path(t0.image_path).exists())
            self.assertTrue(Path(t0.mask_path).exists())

    def test_real_temporal_forecasting_dataset(self):
        dataset = RealTemporalForecastingDataset()
        self.assertGreater(len(dataset), 0)

        item = dataset[0]
        self.assertIn("t0_image", item)
        self.assertIn("target_image", item)
        self.assertIn("target_mask", item)
        self.assertEqual(item["t0_image"].shape, (3, 512, 512))
        self.assertEqual(item["target_image"].shape, (3, 512, 512))

    def test_trainer_dry_run_step(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = TemporalForecasterTrainer(output_dir=tmp_dir, device="cpu")
            dataset = RealTemporalForecastingDataset()
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
            batch = next(iter(dataloader))

            metrics = trainer.train_epoch([batch])
            self.assertIn("train_loss", metrics)
            self.assertIsInstance(metrics["train_loss"], float)

    def test_lpips_and_mask_metrics(self):
        img1 = Image.new("RGB", (256, 256), (200, 200, 200))
        img2 = Image.new("RGB", (256, 256), (200, 200, 200))

        # Test LPIPS on identical images
        lpips_identical = compute_lpips(img1, img2)
        self.assertAlmostEqual(lpips_identical, 0.0, delta=0.05)

        # Test Mask IoU & Dice
        m1 = np.zeros((100, 100), dtype=np.uint8)
        m2 = np.zeros((100, 100), dtype=np.uint8)
        m1[20:50, 20:50] = 255
        m2[20:50, 20:50] = 255

        iou, dice = compute_mask_iou_and_dice(m1, m2)
        self.assertEqual(iou, 1.0)
        self.assertEqual(dice, 1.0)

        # Test comprehensive metrics
        gt_mask = np.zeros((256, 256), dtype=np.uint8)
        gt_mask[50:100, 50:100] = 255
        metrics = compute_comprehensive_metrics(img1, img2, gt_mask, gt_day14_severity=0.25)

        self.assertIn("ssim", metrics)
        self.assertIn("psnr", metrics)
        self.assertIn("lpips", metrics)
        self.assertIn("mask_iou", metrics)
        self.assertIn("mask_dice", metrics)
        self.assertIn("severity_error", metrics)


if __name__ == "__main__":
    unittest.main()
