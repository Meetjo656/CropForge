"""
Command Line Interface for CropForge Stable Diffusion 3.5 LoRA Fine-Tuning.

Usage:
    python -m cropforge.diffusion.training.train --config cropforge/diffusion/configs/lora_training.yaml --dry-run
    python -m cropforge.diffusion.training.train --config cropforge/diffusion/configs/lora_training.yaml --output_dir outputs/test
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure xformers safety if mismatched binaries exist
try:
    import xformers.ops  # noqa: F401
except Exception:
    sys.modules["xformers"] = None
    sys.modules["xformers.ops"] = None

from cropforge.diffusion.training.config import load_lora_training_config
from cropforge.diffusion.training.trainer import LoRATrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CropForge SD 3.5 LoRA Training CLI"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="cropforge/diffusion/configs/lora_training.yaml",
        help="Path to YAML training configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run model/dataset initialization and parameter inspection without training or saving.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="Resume training from latest or specified checkpoint directory.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Override maximum training steps.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override training output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed.",
    )
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()

    overrides = {
        "output_dir": args.output_dir,
        "max_train_steps": args.max_train_steps,
        "seed": args.seed,
    }

    _logger.info("Loading training configuration from '%s'...", args.config)
    config = load_lora_training_config(config_path=args.config, overrides=overrides)

    trainer = LoRATrainer(config=config)

    if args.dry_run:
        _logger.info("Starting dry-run inspection...")
        report = trainer.dry_run()
        _logger.info("Dry-run inspection completed successfully.")
        return

    _logger.info("Starting LoRA training...")
    res_ckpt = None if args.resume is None or args.resume == "latest" else args.resume
    trainer.train(resume_from_checkpoint=res_ckpt)


if __name__ == "__main__":
    main()
