"""
Unit tests for Milestone 10: Mask-Conditioned Spatial Forecasting.
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

from cropforge.diffusion.models.spatial_mask_forecaster import SpatialMaskForecaster
from cropforge.diffusion.Inference.spatial_pipeline import MaskConditionedSpatialPipeline
from cropforge.diffusion.training.train_spatial_forecaster import SpatialMaskForecastingTrainer
from cropforge.diffusion.datasets.real_temporal_dataset import RealTemporalDatasetBuilder
from cropforge.diffusion.training.train_temporal_forecaster import RealTemporalForecastingDataset


class TestMilestone10SpatialMask(unittest.TestCase):
    """
    Test suite verifying SpatialMaskForecaster forward pass, MaskConditionedSpatialPipeline execution, and trainer loss computation.
    """

    def test_spatial_mask_forecaster_forward(self):
        model = SpatialMaskForecaster(cond_dim=128, in_channels=4)
        t0_img = torch.randn(2, 3, 64, 64)
        t0_mask = torch.randn(2, 1, 64, 64)
        cond_vec = torch.randn(2, 128)

        mask_logits, sev_pred = model(t0_mask, t0_img, cond_vec)

        self.assertEqual(mask_logits.shape, (2, 1, 64, 64))
        self.assertEqual(sev_pred.shape, (2, 1))

    def test_spatial_mask_numpy_forecast(self):
        model = SpatialMaskForecaster()
        t0_mask = np.zeros((100, 100), dtype=np.uint8)
        t0_mask[40:60, 40:60] = 255  # Square lesion

        fut_mask, sev = model.forecast_mask_numpy(
            t0_mask_np=t0_mask,
            delta_t_days=14.0,
            temp_c=25.0,
            rh_percent=75.0,
            treatment="untreated",
        )

        self.assertEqual(fut_mask.shape, (100, 100))
        self.assertGreater(sev, 0.0)
        self.assertGreater(np.count_nonzero(fut_mask), np.count_nonzero(t0_mask))

    def test_mask_conditioned_spatial_pipeline(self):
        pipeline = MaskConditionedSpatialPipeline(load_sd35=False, force_offline=True)
        t0_img = Image.new("RGB", (64, 64), color=(50, 150, 50))
        t0_mask = np.zeros((64, 64), dtype=np.uint8)
        t0_mask[20:40, 20:40] = 255

        res = pipeline.forecast_spatial_progression(
            t0_image=t0_img,
            t0_mask=t0_mask,
            delta_t_days=14.0,
            treatment="fungicide",
            seed=42,
        )

        self.assertIn("future_image", res)
        self.assertIn("pred_future_mask", res)
        self.assertIn("pred_future_severity", res)
        self.assertEqual(res["future_image"].size, (512, 512))

    def test_spatial_trainer_exp_e_weights(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = SpatialMaskForecastingTrainer(output_dir=tmp_dir, device="cpu")
            builder = RealTemporalDatasetBuilder(output_dir=tmp_dir, seed=42)
            seqs = builder.generate_dataset(num_plants=2)
            dataset = RealTemporalForecastingDataset(sequences=seqs)
            loader = DataLoader(dataset, batch_size=2)
            batch = next(iter(loader))

            metrics = trainer.train_epoch([batch])
            self.assertEqual(trainer.lambda_mask, 2.0)
            self.assertEqual(trainer.lambda_severity, 1.0)
            self.assertIn("train_loss", metrics)


if __name__ == "__main__":
    unittest.main()
