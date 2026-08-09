"""
Configuration loader and schema dataclasses for SD 3.5 LoRA fine-tuning.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


@dataclass
class ModelConfig:
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-3.5-medium"


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/diffusion/lora_train"
    resolution: int = 1024
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 1000
    mixed_precision: str = "fp16"  # "fp16", "bf16", or "no"
    gradient_checkpointing: bool = True
    seed: int = 42

    def __post_init__(self):
        self.resolution = int(self.resolution)
        self.batch_size = int(self.batch_size)
        self.gradient_accumulation_steps = int(self.gradient_accumulation_steps)
        self.max_train_steps = int(self.max_train_steps)
        self.seed = int(self.seed)


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    epsilon: float = 1e-8

    def __post_init__(self):
        self.learning_rate = float(self.learning_rate)
        self.weight_decay = float(self.weight_decay)
        self.epsilon = float(self.epsilon)
        self.betas = [float(b) for b in self.betas]


@dataclass
class SchedulerConfig:
    name: str = "constant"
    warmup_steps: int = 100


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: List[str] = field(
        default_factory=lambda: [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        ]
    )


@dataclass
class CheckpointConfig:
    save_every_n_steps: int = 500
    max_checkpoints: int = 3


@dataclass
class ValidationConfig:
    enabled: bool = True
    every_n_steps: int = 500
    num_images: int = 1
    seed: int = 42
    prompts: List[str] = field(
        default_factory=lambda: [
            "realistic photograph of a tomato leaf affected by early blight",
            "realistic photograph of a tomato leaf affected by late blight",
            "realistic photograph of a healthy tomato leaf",
            "realistic photograph of a potato leaf affected by early blight",
        ]
    )


@dataclass
class LoRATrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to standard nested dictionary."""
        return {
            "model": {
                "pretrained_model_name_or_path": self.model.pretrained_model_name_or_path,
            },
            "training": {
                "output_dir": self.training.output_dir,
                "resolution": self.training.resolution,
                "batch_size": self.training.batch_size,
                "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
                "max_train_steps": self.training.max_train_steps,
                "mixed_precision": self.training.mixed_precision,
                "gradient_checkpointing": self.training.gradient_checkpointing,
                "seed": self.training.seed,
            },
            "optimizer": {
                "name": self.optimizer.name,
                "learning_rate": self.optimizer.learning_rate,
                "weight_decay": self.optimizer.weight_decay,
                "betas": list(self.optimizer.betas),
                "epsilon": self.optimizer.epsilon,
            },
            "scheduler": {
                "name": self.scheduler.name,
                "warmup_steps": self.scheduler.warmup_steps,
            },
            "lora": {
                "rank": self.lora.rank,
                "alpha": self.lora.alpha,
                "dropout": self.lora.dropout,
                "target_modules": list(self.lora.target_modules),
            },
            "checkpointing": {
                "save_every_n_steps": self.checkpointing.save_every_n_steps,
                "max_checkpoints": self.checkpointing.max_checkpoints,
            },
            "validation": {
                "enabled": self.validation.enabled,
                "every_n_steps": self.validation.every_n_steps,
                "num_images": self.validation.num_images,
                "seed": self.validation.seed,
                "prompts": list(self.validation.prompts),
            },
        }


def load_lora_training_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> LoRATrainingConfig:
    """
    Load LoRA training configuration from YAML file and apply optional CLI overrides.
    """
    if config_path is None:
        default_p = Path(__file__).resolve().parents[1] / "configs" / "lora_training.yaml"
        if default_p.exists():
            config_path = default_p
        else:
            config_path = Path("cropforge/diffusion/configs/lora_training.yaml")

    cfg_file = Path(config_path)
    raw_dict: Dict[str, Any] = {}
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                raw_dict = loaded

    # Build dataclasses from loaded YAML dict
    model_cfg = ModelConfig(**raw_dict.get("model", {}))
    training_cfg = TrainingConfig(**raw_dict.get("training", {}))
    optimizer_cfg = OptimizerConfig(**raw_dict.get("optimizer", {}))
    scheduler_cfg = SchedulerConfig(**raw_dict.get("scheduler", {}))
    lora_cfg = LoRAConfig(**raw_dict.get("lora", {}))
    checkpointing_cfg = CheckpointConfig(**raw_dict.get("checkpointing", {}))
    validation_cfg = ValidationConfig(**raw_dict.get("validation", {}))

    config_obj = LoRATrainingConfig(
        model=model_cfg,
        training=training_cfg,
        optimizer=optimizer_cfg,
        scheduler=scheduler_cfg,
        lora=lora_cfg,
        checkpointing=checkpointing_cfg,
        validation=validation_cfg,
    )

    # Apply overrides if present
    if overrides:
        if "output_dir" in overrides and overrides["output_dir"] is not None:
            config_obj.training.output_dir = str(overrides["output_dir"])
        if "max_train_steps" in overrides and overrides["max_train_steps"] is not None:
            config_obj.training.max_train_steps = int(overrides["max_train_steps"])
        if "seed" in overrides and overrides["seed"] is not None:
            config_obj.training.seed = int(overrides["seed"])
        if "resolution" in overrides and overrides["resolution"] is not None:
            config_obj.training.resolution = int(overrides["resolution"])
        if "batch_size" in overrides and overrides["batch_size"] is not None:
            config_obj.training.batch_size = int(overrides["batch_size"])

    return config_obj
