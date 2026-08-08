"""
Test runner using standard library unittest for CropForge Diffusion Dataset Generation.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cropforge.diffusion.schemas import DatasetSample
from cropforge.diffusion.configs import load_config
from cropforge.diffusion.prompting import PromptBuilder, PromptTemplateEngine
from cropforge.diffusion.conditions import ConditionEncoder
from cropforge.diffusion.datasets import DatasetValidator, MetadataManager, DatasetBuilder, ValidationError
from cropforge.diffusion.tests.test_model_loader import TestModelLoader
from cropforge.diffusion.tests.test_sd35_generator import TestSD35Generator
from cropforge.diffusion.tests.test_sd35_pipeline import TestSD35InferencePipeline


class TestPrompting(unittest.TestCase):
    def setUp(self):
        self.sample = DatasetSample(
            sample_id="000001",
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

    def test_template_rendering(self):
        subject = PromptTemplateEngine.render_subject("Tomato", "Late Blight", "Moderate")
        self.assertIn("Tomato", subject)
        self.assertIn("Late Blight", subject)
        self.assertIn("moderate", subject)

        healthy_subject = PromptTemplateEngine.render_subject("Tomato", "Healthy", "Healthy")
        self.assertIn("Healthy Tomato", healthy_subject)

        treatment = PromptTemplateEngine.render_treatment("Copper Fungicide", 14)
        self.assertIn("Copper Fungicide", treatment)
        self.assertIn("14 days", treatment)

        environment = PromptTemplateEngine.render_environment(28.0, 75.0)
        self.assertIn("28", environment)
        self.assertIn("75", environment)

    def test_prompt_builder(self):
        builder = PromptBuilder()
        positive, negative = builder.build_prompt_pair(self.sample)

        self.assertIn("Tomato", positive)
        self.assertIn("Late Blight", positive)
        self.assertIn("Copper Fungicide", positive)
        self.assertIn("28", positive)
        self.assertIn("75", positive)

        self.assertIn("illustration", negative)
        self.assertIn("painting", negative)
        self.assertIn("anime", negative)
        self.assertIn("cartoon", negative)
        self.assertIn("CGI", negative)
        self.assertIn("blurry", negative)
        self.assertIn("watermark", negative)


class TestValidator(unittest.TestCase):
    def setUp(self):
        self.validator = DatasetValidator()
        self.sample = DatasetSample(
            sample_id="000001",
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

    def test_valid_sample(self):
        result = self.validator.validate(self.sample)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)

    def test_invalid_temperature(self):
        self.sample.temperature = 85.0
        result = self.validator.validate(self.sample)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("Temperature" in e for e in result.errors))

    def test_invalid_humidity(self):
        self.sample.humidity = -10.0
        result = self.validator.validate(self.sample)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("Humidity" in e for e in result.errors))

    def test_invalid_days(self):
        self.sample.days_after_treatment = -5
        result = self.validator.validate(self.sample)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("Days after treatment" in e for e in result.errors))

    def test_invalid_severity(self):
        self.sample.severity = "Catastrophic"
        result = self.validator.validate(self.sample)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("Severity" in e for e in result.errors))


class TestConditionEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = ConditionEncoder()
        self.sample = DatasetSample(
            sample_id="000001",
            crop="Tomato",
            disease="Late Blight",
            severity="Moderate",
            treatment="Copper Fungicide",
            days_after_treatment=14,
            temperature=30.0,
            humidity=50.0,
            input_image="day0.png",
            target_image="day14.png",
            segmentation_mask="mask.png",
        )

    def test_categorical_encoding(self):
        vec = self.encoder.encode_categorical("crop", "Tomato")
        self.assertEqual(float(vec.sum()), 1.0)

    def test_continuous_scaling(self):
        scaled_temp = self.encoder.scale_continuous("temperature", 30.0)
        self.assertAlmostEqual(scaled_temp, 0.5, places=2)

    def test_sample_encoding(self):
        vec = self.encoder.encode_sample(self.sample)
        self.assertEqual(len(vec), self.encoder.get_vector_dimension())


class TestDatasetBuilder(unittest.TestCase):
    def test_builder_build(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = DatasetBuilder(base_output_dir=tmp_dir)
            sample_dict = {
                "sample_id": "test_001",
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
            result = builder.build_sample(sample_dict, save_files=True)
            self.assertEqual(result.sample_id, "test_001")
            self.assertTrue(result.validation_result.is_valid)
            self.assertTrue(result.metadata_path.exists())
            self.assertTrue(result.prompt_path.exists())


if __name__ == "__main__":
    unittest.main()
