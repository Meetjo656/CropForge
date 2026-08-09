"""
Checkpoint manager for SD 3.5 LoRA training.
Handles periodic step checkpointing, state serialization, resume functionality,
safetensors export, and directory pruning.
"""

import json
import logging
from pathlib import Path
import shutil
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.optim import Optimizer

from cropforge.diffusion.training.lora import save_lora_weights, load_lora_weights
from cropforge.diffusion.training.config import CheckpointConfig, LoRATrainingConfig

_logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages saving, loading, resuming, and pruning of training checkpoints.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        checkpoint_config: Optional[CheckpointConfig] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.final_dir = self.output_dir / "final"
        self.config = checkpoint_config if checkpoint_config is not None else CheckpointConfig()

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        config: Optional[LoRATrainingConfig] = None,
        seed: int = 42,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save a step checkpoint directory containing LoRA weights, optimizer state,
        scheduler state, and metadata JSON.
        """
        ckpt_dir = self.checkpoints_dir / f"checkpoint-{step:06d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save LoRA weights in safetensors
        weights_path = ckpt_dir / "pytorch_lora_weights.safetensors"
        save_lora_weights(model, weights_path)

        # 2. Save Optimizer state
        if optimizer is not None:
            torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

        # 3. Save Scheduler state
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")

        # 4. Save metadata JSON
        state_info = {
            "step": step,
            "seed": seed,
            "config": config.to_dict() if config is not None else {},
            "metrics": metrics or {},
        }
        with open(ckpt_dir / "checkpoint_state.json", "w", encoding="utf-8") as f:
            json.dump(state_info, f, indent=4)

        _logger.info("Saved step checkpoint to '%s'", ckpt_dir)

        # 5. Prune old checkpoints if max_checkpoints exceeded
        self._prune_checkpoints()

        return ckpt_dir

    def save_final_weights(
        self,
        model: nn.Module,
        output_filename: str = "pytorch_lora_weights.safetensors",
    ) -> Path:
        """Save final LoRA weights into outputs/diffusion/final/."""
        final_path = self.final_dir / output_filename
        save_lora_weights(model, final_path)
        _logger.info("Saved final LoRA weights to '%s'", final_path)
        return final_path

    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest checkpoint directory under checkpoints/."""
        if not self.checkpoints_dir.exists():
            return None
        dirs = [
            d for d in self.checkpoints_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ]
        if not dirs:
            return None
        dirs.sort(key=lambda x: int(x.name.split("-")[-1]))
        return dirs[-1]

    def load_checkpoint(
        self,
        checkpoint_dir: Optional[Union[str, Path]],
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Load training state from checkpoint directory.
        
        Returns:
            Tuple of (resumed_step, state_info_dict).
        """
        if checkpoint_dir is None:
            latest = self.find_latest_checkpoint()
            if latest is None:
                raise FileNotFoundError("No checkpoint found to resume from.")
            ckpt_p = latest
        else:
            ckpt_p = Path(checkpoint_dir)

        if not ckpt_p.exists():
            raise FileNotFoundError(f"Checkpoint directory '{ckpt_p}' does not exist.")

        # 1. Load LoRA weights
        weights_p = ckpt_p / "pytorch_lora_weights.safetensors"
        if weights_p.exists():
            load_lora_weights(model, weights_p)

        # 2. Load Optimizer
        opt_p = ckpt_p / "optimizer.pt"
        if optimizer is not None and opt_p.exists():
            optimizer.load_state_dict(torch.load(opt_p, map_location="cpu"))

        # 3. Load Scheduler
        sched_p = ckpt_p / "scheduler.pt"
        if scheduler is not None and sched_p.exists() and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(torch.load(sched_p, map_location="cpu"))

        # 4. Load State JSON
        step = 0
        state_dict: Dict[str, Any] = {}
        json_p = ckpt_p / "checkpoint_state.json"
        if json_p.exists():
            with open(json_p, "r", encoding="utf-8") as f:
                state_dict = json.load(f)
                step = state_dict.get("step", 0)

        _logger.info("Resumed training from checkpoint '%s' at step %d", ckpt_p, step)
        return step, state_dict

    def _prune_checkpoints(self) -> None:
        """Keep only the most recent `max_checkpoints` checkpoint directories."""
        max_ckpts = self.config.max_checkpoints
        if max_ckpts <= 0:
            return

        dirs = [
            d for d in self.checkpoints_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ]
        if len(dirs) <= max_ckpts:
            return

        dirs.sort(key=lambda x: int(x.name.split("-")[-1]))
        to_delete = dirs[:-max_ckpts]
        for d in to_delete:
            _logger.info("Pruning old checkpoint directory '%s'", d)
            shutil.rmtree(d, ignore_errors=True)
