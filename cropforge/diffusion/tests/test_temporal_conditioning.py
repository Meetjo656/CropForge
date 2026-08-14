"""
Unit tests for Milestone 5: Temporal Conditioning & Disease Forecasting Architecture.
"""

import sys
import unittest
from pathlib import Path
from PIL import Image
import torch

# Add workspace root to sys.path
_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Bypass broken xformers
sys.modules["xformers"] = None
sys.modules["xformers.ops"] = None

from cropforge.diffusion.conditions import TemporalConditionEncoder
from cropforge.diffusion.datasets.temporal_builder import TemporalDatasetBuilder, TemporalDatasetSample
from cropforge.diffusion.Inference import TemporalInferencePipeline


class TestTemporalConditioning(unittest.TestCase):
    """Test suite for Milestone 5 temporal conditioning modules."""

    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

        self.encoder = TemporalConditionEncoder(
            pooled_projection_dim=256,
            joint_attention_dim=512,
            device=self.device,
            dtype=self.dtype,
        )

    def test_encoder_output_shapes(self):
        """Verify TemporalConditionEncoder produces correct tensor dimensions."""
        pooled, seq = self.encoder.encode_conditions(
            delta_t=7.0,
            env_covariates=[25.0, 75.0, 60.0],
            treatment="fungicide",
            batch_size=2,
            device=self.device,
            dtype=self.dtype,
        )

        self.assertEqual(pooled.shape, (2, 256))
        self.assertEqual(seq.shape, (2, 16, 512))
        self.assertFalse(torch.isnan(pooled).any())
        self.assertFalse(torch.isnan(seq).any())

    def test_treatment_intervention_mapping(self):
        """Verify string vs integer treatment mapping."""
        p1, _ = self.encoder.encode_conditions(delta_t=3.0, treatment="untreated")
        p2, _ = self.encoder.encode_conditions(delta_t=3.0, treatment="fungicide")
        p3, _ = self.encoder.encode_conditions(delta_t=3.0, treatment="biocontrol")

        self.assertFalse(torch.equal(p1, p2))
        self.assertFalse(torch.equal(p2, p3))

    def test_temporal_dataset_builder(self):
        """Verify TemporalDatasetBuilder produces sequence pairs and severity trajectories."""
        builder = TemporalDatasetBuilder(seed=42)
        sample = builder.generate_pair_sample(0)

        self.assertIsInstance(sample, TemporalDatasetSample)
        self.assertIsInstance(sample.t0_image, Image.Image)
        self.assertIsInstance(sample.t1_image, Image.Image)
        self.assertGreaterEqual(sample.t1_severity, 0.0)
        self.assertLessEqual(sample.t1_severity, 1.0)

        dataset = builder.generate_dataset(num_samples=10)
        self.assertEqual(len(dataset), 10)

    def test_progression_calculation(self):
        """Verify disease progression math reflects treatment suppression."""
        builder = TemporalDatasetBuilder(seed=42)
        untreated_sev = builder.calculate_progression(0.2, 7.0, 25.0, 80.0, "untreated")
        fungicide_sev = builder.calculate_progression(0.2, 7.0, 25.0, 80.0, "fungicide")

        self.assertGreater(untreated_sev, fungicide_sev)

    def test_temporal_inference_pipeline_dummy(self):
        """Verify TemporalInferencePipeline forecast execution."""
        class MockPipe:
            device = "cpu"

        mock_pipe = MockPipe()
        pipeline = TemporalInferencePipeline(
            pipeline=mock_pipe,
            condition_encoder=self.encoder,
            device=self.device,
            dtype=self.dtype,
        )

        res = pipeline.forecast(
            prompt="realistic photograph of a tomato leaf affected by early blight",
            delta_t_days=7.0,
            treatment="fungicide",
        )

        self.assertIn("forecast_image", res)
        self.assertIsInstance(res["forecast_image"], Image.Image)
        self.assertEqual(res["metadata"]["delta_t_days"], 7.0)


if __name__ == "__main__":
    unittest.main()
