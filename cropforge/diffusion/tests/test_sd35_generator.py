"""
Unit tests for SD35Generator in CropForge Diffusion Inference.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cropforge.diffusion.Inference.sd35_generator import SD35Generator
from cropforge.diffusion.schemas import DatasetSample


class TestSD35Generator(unittest.TestCase):

    def setUp(self):
        self.mock_image = Image.new("RGB", (64, 64), color="green")
        self.mock_output = MagicMock()
        self.mock_output.images = [self.mock_image]

        self.mock_pipe = MagicMock()
        self.mock_pipe.return_value = self.mock_output
        self.mock_pipe.device = "cpu"

        self.generator = SD35Generator(pipe=self.mock_pipe)

        self.sample = DatasetSample(
            sample_id="test_001",
            crop="Tomato",
            disease="Late Blight",
            severity="Moderate",
            treatment="Copper Fungicide",
            days_after_treatment=14,
            temperature=28.0,
            humidity=75.0,
            input_image="day0.png",
            target_image="day14.png",
            segmentation_mask="mask.png",
        )

    def test_generate_from_prompt(self):
        img = self.generator.generate_from_prompt(
            prompt="A macro shot of a healthy tomato leaf",
            negative_prompt="blurry, distorted",
            height=512,
            width=512,
        )
        self.assertIsInstance(img, Image.Image)
        self.mock_pipe.assert_called_once()
        call_kwargs = self.mock_pipe.call_args.kwargs
        self.assertEqual(call_kwargs["prompt"], "A macro shot of a healthy tomato leaf")
        self.assertEqual(call_kwargs["negative_prompt"], "blurry, distorted")
        self.assertEqual(call_kwargs["height"], 512)
        self.assertEqual(call_kwargs["width"], 512)

    def test_generate_from_sample(self):
        img = self.generator.generate(self.sample)
        self.assertIsInstance(img, Image.Image)
        call_kwargs = self.mock_pipe.call_args.kwargs
        self.assertIn("Tomato", call_kwargs["prompt"])
        self.assertIn("Late Blight", call_kwargs["prompt"])

    def test_generate_from_dict(self):
        sample_dict = self.sample.to_dict()
        img = self.generator.generate(sample_dict)
        self.assertIsInstance(img, Image.Image)
        call_kwargs = self.mock_pipe.call_args.kwargs
        self.assertIn("Tomato", call_kwargs["prompt"])

    def test_generate_batch(self):
        items = [
            "Simple text prompt 1",
            self.sample,
            self.sample.to_dict(),
        ]
        images = self.generator.generate_batch(items)
        self.assertEqual(len(images), 3)
        for img in images:
            self.assertIsInstance(img, Image.Image)

    def test_seed_reproducibility(self):
        self.generator.generate_from_prompt("test prompt", seed=12345)
        call_kwargs = self.mock_pipe.call_args.kwargs
        self.assertIn("generator", call_kwargs)

    def test_invalid_sample_raises(self):
        with self.assertRaises(TypeError):
            self.generator.generate(12345)


if __name__ == "__main__":
    unittest.main()
