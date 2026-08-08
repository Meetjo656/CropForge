"""
Unit tests for Task 5 (Logging), Task 6 (YAML Config), and Task 7 (generate_dataset batch generation).
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from PIL import Image

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cropforge.diffusion.Inference.sd35_generator import SD35Generator
from cropforge.diffusion.Inference.sd35_pipeline import SD35InferencePipeline, generate_dataset
from cropforge.diffusion.configs import load_config


class TestLoggingAndConfig(unittest.TestCase):

    def setUp(self):
        self.mock_image = Image.new("RGB", (64, 64), color="yellow")
        self.mock_output = MagicMock()
        self.mock_output.images = [self.mock_image]

        self.mock_pipe = MagicMock()
        self.mock_pipe.return_value = self.mock_output
        self.mock_pipe.device = "cpu"
        self.mock_pipe.scheduler = MagicMock()
        self.mock_pipe.scheduler.__class__.__name__ = "EulerDiscreteScheduler"

        self.generator = SD35Generator(pipe=self.mock_pipe)

    def test_yaml_config_inference_defaults(self):
        """Task 6: Test that inference configuration parameters are loaded from dataset_config.yaml."""
        config = load_config()
        self.assertIn("inference", config)
        inf_cfg = config["inference"]
        self.assertEqual(inf_cfg["steps"], 30)
        self.assertEqual(inf_cfg["guidance_scale"], 7.5)
        self.assertEqual(inf_cfg["height"], 1024)
        self.assertEqual(inf_cfg["width"], 1024)
        self.assertEqual(inf_cfg["seed"], 42)

    def test_generation_log_keys(self):
        """Task 5: Test that every generation records required logging fields."""
        self.generator.generate_from_prompt("test botanical prompt")
        log_data = self.generator.last_generation_log

        required_keys = {
            "timestamp",
            "seed",
            "guidance_scale",
            "steps",
            "scheduler",
            "model",
            "negative_prompt",
            "generation_time",
        }
        self.assertTrue(required_keys.issubset(set(log_data.keys())))
        self.assertIsInstance(log_data["generation_time"], float)
        self.assertGreaterEqual(log_data["generation_time"], 0.0)
        self.assertEqual(log_data["scheduler"], "EulerDiscreteScheduler")

    def test_pipeline_saves_generation_log_file(self):
        """Task 5: Test that run_sample writes generation_log.json to disk."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = SD35InferencePipeline(
                generator=self.generator,
                output_base_dir=tmp_dir,
            )
            sample = {
                "sample_id": "001",
                "crop": "Tomato",
                "disease": "Late Blight",
                "severity": "Moderate",
                "treatment": "Copper Fungicide",
                "days_after_treatment": 14,
                "temperature": 28.0,
                "humidity": 75.0,
                "input_image": "day0.png",
                "target_image": "day14.png",
                "segmentation_mask": "mask.png",
            }
            res = pipeline.run_sample(sample)

            log_file = Path(res["generation_log_json"])
            self.assertTrue(log_file.exists())
            with open(log_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertIn("timestamp", data)
            self.assertIn("generation_time", data)
            self.assertIn("steps", data)

    def test_generate_dataset_batch_pipeline(self):
        """Task 7: Test end-to-end generate_dataset function for batch sample creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = SD35InferencePipeline(
                generator=self.generator,
                output_base_dir=tmp_dir,
            )
            results = pipeline.generate_dataset(num_samples=5, seed=100)

            self.assertEqual(len(results), 5)
            for idx, res in enumerate(results, start=1):
                sample_dir = Path(res["sample_dir"])
                self.assertTrue(sample_dir.exists())
                self.assertTrue((sample_dir / "generated.png").exists())
                self.assertTrue((sample_dir / "prompt.txt").exists())
                self.assertTrue((sample_dir / "metadata.json").exists())
                self.assertTrue((sample_dir / "generation_log.json").exists())


if __name__ == "__main__":
    unittest.main()
