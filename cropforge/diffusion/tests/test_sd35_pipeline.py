"""
Unit tests for SD35InferencePipeline and Comparison Generator.
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
from cropforge.diffusion.Inference.sd35_pipeline import SD35InferencePipeline
from cropforge.diffusion.datasets.metadata import MetadataManager
from cropforge.diffusion.schemas import DatasetSample


class TestSD35InferencePipeline(unittest.TestCase):

    def setUp(self):
        self.mock_image = Image.new("RGB", (64, 64), color="blue")
        self.mock_generator = MagicMock(spec=SD35Generator)
        self.mock_generator.generate_from_prompt.return_value = self.mock_image

        self.sample = DatasetSample(
            sample_id="001",
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

    def test_run_sample_directory_structure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = SD35InferencePipeline(
                generator=self.mock_generator,
                output_base_dir=tmp_dir,
            )

            result = pipeline.run_sample(self.sample)

            sample_dir = Path(result["sample_dir"])
            self.assertTrue(sample_dir.exists())
            self.assertTrue(sample_dir.is_dir())
            self.assertEqual(sample_dir.name, "sample_001")

            # Check required files
            gen_png = Path(result["generated_png"])
            prompt_txt = Path(result["prompt_txt"])
            meta_json = Path(result["metadata_json"])

            self.assertTrue(gen_png.exists())
            self.assertTrue(prompt_txt.exists())
            self.assertTrue(meta_json.exists())

            self.assertEqual(gen_png.name, "generated.png")
            self.assertEqual(prompt_txt.name, "prompt.txt")
            self.assertEqual(meta_json.name, "metadata.json")

    def test_prompt_txt_content(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = SD35InferencePipeline(
                generator=self.mock_generator,
                output_base_dir=tmp_dir,
            )

            result = pipeline.run_sample(self.sample)
            prompt_content = Path(result["prompt_txt"]).read_text(encoding="utf-8")

            self.assertIn("PROMPT:", prompt_content)
            self.assertIn("NEGATIVE_PROMPT:", prompt_content)
            self.assertIn("Tomato", prompt_content)
            self.assertIn("Late Blight", prompt_content)

    def test_metadata_json_content(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = SD35InferencePipeline(
                generator=self.mock_generator,
                output_base_dir=tmp_dir,
            )

            result = pipeline.run_sample(self.sample)
            loaded_sample = MetadataManager.load_metadata(result["metadata_json"])

            self.assertEqual(loaded_sample.sample_id, "001")
            self.assertEqual(loaded_sample.crop, "Tomato")
            self.assertEqual(loaded_sample.disease, "Late Blight")

    def test_ground_truth_copying(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            target_img_file = tmp_path / "day14.png"
            target_img_file.write_bytes(b"dummy image content")

            sample_with_real_path = self.sample.model_copy()
            sample_with_real_path.target_image = str(target_img_file)

            pipeline = SD35InferencePipeline(
                generator=self.mock_generator,
                output_base_dir=tmp_path / "outputs",
            )

            result = pipeline.run_sample(sample_with_real_path, copy_ground_truth=True)
            self.assertIsNotNone(result["ground_truth_png"])
            gt_path = Path(result["ground_truth_png"])
            self.assertTrue(gt_path.exists())
            self.assertEqual(gt_path.name, "ground_truth.png")

    def test_run_batch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = SD35InferencePipeline(
                generator=self.mock_generator,
                output_base_dir=tmp_dir,
            )

            sample2 = self.sample.model_copy()
            sample2.sample_id = "002"

            results = pipeline.run_batch([self.sample, sample2])
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["sample_id"], "001")
            self.assertEqual(results[1]["sample_id"], "002")


if __name__ == "__main__":
    unittest.main()
