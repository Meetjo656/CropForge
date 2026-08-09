"""
Sanity tests for Milestone 4 SD 3.5 LoRA Fine-Tuning architecture.
"""

import sys
import tempfile
import unittest
from pathlib import Path
import torch
import torch.nn as nn

# Bypass xformers binary mismatch if present
try:
    import xformers.ops  # noqa: F401
except Exception:
    sys.modules["xformers"] = None
    sys.modules["xformers.ops"] = None

from diffusers import SD3Transformer2DModel
from cropforge.diffusion.training.config import load_lora_training_config, LoRATrainingConfig
from cropforge.diffusion.training.dataset import CropForgeDiffusionDataset, TrainingCondition
from cropforge.diffusion.training.lora import setup_sd35_lora, get_parameter_summary, freeze_base_model
from cropforge.diffusion.training.checkpoint import CheckpointManager
from cropforge.diffusion.training.validation import ValidationEvaluator
from cropforge.diffusion.training.trainer import LoRATrainer


class TestLoRATrainingArchitecture(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_config_loading_and_overrides(self):
        """Test loading YAML configuration and applying CLI overrides."""
        config = load_lora_training_config(
            overrides={"output_dir": str(self.output_dir / "custom_train"), "max_train_steps": 500}
        )
        self.assertIsInstance(config, LoRATrainingConfig)
        self.assertEqual(config.training.output_dir, str(self.output_dir / "custom_train"))
        self.assertEqual(config.training.max_train_steps, 500)
        self.assertEqual(config.model.pretrained_model_name_or_path, "stabilityai/stable-diffusion-3.5-medium")

    def test_dataset_loading_and_extensible_condition(self):
        """Test PyTorch dataset loading and TrainingCondition extensible container."""
        ds = CropForgeDiffusionDataset(num_synthetic_samples=5, resolution=512)
        self.assertEqual(len(ds), 5)

        item = ds[0]
        self.assertIn("pixel_values", item)
        self.assertIn("prompt", item)
        self.assertIn("condition", item)

        cond = item["condition"]
        self.assertIsInstance(cond, TrainingCondition)
        self.assertTrue(len(cond.prompt) > 0)
        self.assertIsNotNone(cond.condition_vector)
        self.assertIn("disease", cond.metadata)

    def test_lora_attachment_and_frozen_base(self):
        """
        Verify that LoRA adapters attach correctly, base model parameters are frozen,
        and trainable parameters are strictly positive.
        """
        transformer = SD3Transformer2DModel(
            sample_size=32,
            patch_size=2,
            in_channels=16,
            num_layers=1,
            attention_head_dim=32,
            num_attention_heads=4,
            caption_projection_dim=128,
            joint_attention_dim=128,
            pooled_projection_dim=32,
        )

        peft_model, summary = setup_sd35_lora(transformer)

        frozen_count = summary["frozen_params"]
        trainable_count = summary["trainable_params"]
        total_count = summary["total_params"]

        self.assertGreater(frozen_count, 0, "Base parameters must be frozen!")
        self.assertGreater(trainable_count, 0, "LoRA parameters must be trainable!")
        self.assertEqual(frozen_count + trainable_count, total_count)

        # Check explicit parameter requiring grad state
        trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
        frozen_params = [p for p in peft_model.parameters() if not p.requires_grad]

        self.assertTrue(len(trainable_params) > 0)
        self.assertTrue(len(frozen_params) > 0)

    def test_checkpointing_workflow(self):
        """Test saving, loading, and resuming checkpoint state."""
        ckpt_mgr = CheckpointManager(output_dir=self.output_dir)

        transformer = SD3Transformer2DModel(
            sample_size=32,
            patch_size=2,
            in_channels=16,
            num_layers=1,
            attention_head_dim=32,
            num_attention_heads=4,
            caption_projection_dim=128,
            joint_attention_dim=128,
            pooled_projection_dim=32,
        )
        peft_model, _ = setup_sd35_lora(transformer)

        optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-4)

        # Save checkpoint
        saved_dir = ckpt_mgr.save_checkpoint(
            step=100,
            model=peft_model,
            optimizer=optimizer,
            seed=42,
        )
        self.assertTrue(saved_dir.exists())
        self.assertTrue((saved_dir / "pytorch_lora_weights.safetensors").exists())

        # Resume checkpoint
        new_model, _ = setup_sd35_lora(transformer)
        resumed_step, state = ckpt_mgr.load_checkpoint(
            checkpoint_dir=saved_dir,
            model=new_model,
        )
        self.assertEqual(resumed_step, 100)

    def test_validation_evaluator_triggers(self):
        """Test validation evaluator trigger logic and metadata export."""
        val_eval = ValidationEvaluator(output_dir=self.output_dir)
        self.assertFalse(val_eval.should_validate(0))
        self.assertTrue(val_eval.should_validate(500))

        step_dir = val_eval.run_validation(step=500, pipeline=None)
        self.assertTrue(step_dir.exists())
        self.assertTrue((step_dir / "validation_summary.json").exists())

    def test_trainer_dry_run_execution(self):
        """Test LoRATrainer dry-run mode non-modifying execution."""
        config = load_lora_training_config(
            overrides={"output_dir": str(self.output_dir / "dry_run_test")}
        )
        trainer = LoRATrainer(config=config)
        report = trainer.dry_run()

        self.assertIn("base_params", report)
        self.assertIn("trainable_lora_params", report)
        self.assertGreater(report["base_params"], 0)
        self.assertGreater(report["trainable_lora_params"], 0)
        self.assertGreater(report["trainable_percent"], 0.0)


if __name__ == "__main__":
    unittest.main()
