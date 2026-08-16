"""
Unit tests for Milestone 8: Forecasting Failure Analysis & Lesion-Aware Multi-Loss Training.
"""

import sys
import unittest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cropforge.diffusion.datasets.real_temporal_dataset import RealTemporalDatasetBuilder
from cropforge.diffusion.analysis.forecasting_failure_analysis import (
    ForecastingFailureAnalyzer,
    compute_mask_centroid,
    compute_spatial_centroid_distance,
)
from cropforge.diffusion.training.lesion_aware_trainer import (
    LesionAwareForecasterTrainer,
    DiceLoss,
)
from cropforge.diffusion.training.train_temporal_forecaster import RealTemporalForecastingDataset


class TestMilestone8LesionAware(unittest.TestCase):
    """
    Test suite verifying failure analysis, multi-task loss computation (BCE + Dice + Severity), and centroid distance.
    """

    def test_centroid_and_spatial_distance(self):
        m1 = np.zeros((100, 100), dtype=np.uint8)
        m2 = np.zeros((100, 100), dtype=np.uint8)

        m1[20:40, 20:40] = 255  # Centroid at (29.5, 29.5)
        m2[20:40, 40:60] = 255  # Centroid at (49.5, 29.5)

        c1 = compute_mask_centroid(m1)
        c2 = compute_mask_centroid(m2)

        self.assertIsNotNone(c1)
        self.assertIsNotNone(c2)

        dist = compute_spatial_centroid_distance(m1, m2)
        self.assertAlmostEqual(dist, 20.0, delta=1.0)

    def test_dice_loss_module(self):
        dice_fn = DiceLoss()
        logits = torch.zeros((1, 1, 64, 64))
        targets = torch.zeros((1, 1, 64, 64))
        targets[:, :, 10:30, 10:30] = 1.0

        loss = dice_fn(logits, targets)
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0.0)

    def test_failure_analyzer_diagnostics(self):
        plant_data = {
            "plant_id": "plant_test_001",
            "crop": "tomato",
            "disease": "early_blight",
            "treatment": "untreated",
            "metrics": {
                "ssim": 0.85,
                "mask_iou": 0.04,
                "mask_dice": 0.08,
                "severity_error": 0.28,
                "gt_severity": 0.35,
                "forecasted_severity": 0.07,
            },
        }

        analyzer = ForecastingFailureAnalyzer.__new__(ForecastingFailureAnalyzer)
        diag = analyzer.analyze_plant_evaluation(plant_data)

        self.assertEqual(diag["plant_id"], "plant_test_001")
        self.assertIn("Spatial Localization Failure", diag["primary_failure_cause"])
        self.assertGreaterEqual(len(diag["identified_failure_modes"]), 2)

    def test_lesion_aware_trainer_multi_loss_step(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = LesionAwareForecasterTrainer(output_dir=tmp_dir, device="cpu")
            builder = RealTemporalDatasetBuilder(output_dir=tmp_dir, seed=42)
            seqs = builder.generate_dataset(num_plants=2)
            dataset = RealTemporalForecastingDataset(sequences=seqs)
            loader = DataLoader(dataset, batch_size=2)
            batch = next(iter(loader))

            metrics = trainer.train_epoch([batch])
            self.assertIn("train_loss", metrics)
            self.assertIn("image_loss", metrics)
            self.assertIn("mask_loss", metrics)
            self.assertIn("severity_loss", metrics)
            self.assertIsInstance(metrics["train_loss"], float)


if __name__ == "__main__":
    unittest.main()
