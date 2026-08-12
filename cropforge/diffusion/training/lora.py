"""
PEFT LoRA adapter setup, base model freezing, parameter count inspection,
and weight persistence helpers for SD3.5 Transformer.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model, PeftModel
from peft.utils import get_peft_model_state_dict

from cropforge.diffusion.training.config import LoRAConfig

_logger = logging.getLogger(__name__)


def freeze_base_model(model: nn.Module) -> None:
    """Freeze all base model parameters so no gradients flow to base weights."""
    model.requires_grad_(False)
    for param in model.parameters():
        param.requires_grad = False


def setup_sd35_lora(
    transformer: nn.Module,
    lora_config: Optional[Union[LoRAConfig, Dict[str, Any]]] = None,
    full_pipeline: Optional[Any] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Attach PEFT LoRA adapters to SD3.5 transformer model and freeze base weights.

    Args:
        transformer: SD3Transformer2DModel instance.
        lora_config: LoRAConfig object or dictionary.
        full_pipeline: Optional complete StableDiffusion3Pipeline instance for parameter accounting.

    Returns:
        Tuple of (lora_wrapped_transformer, summary_dict).
    """
    if lora_config is None:
        cfg = LoRAConfig()
    elif isinstance(lora_config, dict):
        cfg = LoRAConfig(**lora_config)
    else:
        cfg = lora_config

    # 1. Freeze base model
    freeze_base_model(transformer)

    # 2. Build PEFT LoraConfig
    peft_lora_config = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=list(cfg.target_modules),
        init_lora_weights="gaussian",
    )

    # 3. Inject LoRA adapters
    peft_model = get_peft_model(transformer, peft_lora_config)

    # 4. Inspect parameters and assert frozen/trainable balance
    summary = get_parameter_summary(peft_model, full_pipeline=full_pipeline)
    
    assert summary["trainable_params"] > 0, "No trainable LoRA parameters were found after adapter injection!"
    assert summary["frozen_params"] > 0, "Base parameters are not frozen!"

    _logger.info(
        "Attached LoRA adapters (rank=%d, alpha=%d). Trainable params: %s / %s (%.2f%%)",
        cfg.rank,
        cfg.alpha,
        f"{summary['trainable_params']:,}",
        f"{summary['total_params']:,}",
        summary["trainable_percent"],
    )

    return peft_model, summary


def get_parameter_summary(model: nn.Module, full_pipeline: Optional[Any] = None) -> Dict[str, Any]:
    """
    Compute full SD3.5 model parameter counts component by component,
    trainable (LoRA), frozen, and percentage parameter counts.

    Args:
        model: LoRA target model (e.g., SD3Transformer2DModel with PEFT).
        full_pipeline: Optional complete StableDiffusion3Pipeline instance.

    Returns:
        Dictionary containing parameter accounting statistics.
    """
    return inspect_component_parameters(model, full_pipeline=full_pipeline)


def inspect_component_parameters(model: nn.Module, full_pipeline: Optional[Any] = None) -> Dict[str, Any]:
    """
    Inspect parameter counts component by component for SD3.5 architecture.

    Args:
        model: LoRA target model (e.g., SD3Transformer2DModel).
        full_pipeline: Optional complete StableDiffusion3Pipeline instance.

    Returns:
        Dictionary containing component-by-component parameter counts.
    """
    def _count_mod(mod: Optional[Any]) -> int:
        if mod is not None and isinstance(mod, nn.Module):
            return sum(p.numel() for p in mod.parameters())
        return 0

    transformer_params = 0
    vae_params = 0
    te1_params = 0
    te2_params = 0
    te3_params = 0
    other_params = 0

    if full_pipeline is not None:
        trans_mod = getattr(full_pipeline, "transformer", model)
        vae_mod = getattr(full_pipeline, "vae", None)
        te1_mod = getattr(full_pipeline, "text_encoder", None)
        te2_mod = getattr(full_pipeline, "text_encoder_2", None)
        te3_mod = getattr(full_pipeline, "text_encoder_3", None)

        transformer_params = _count_mod(trans_mod)
        vae_params = _count_mod(vae_mod)
        te1_params = _count_mod(te1_mod)
        te2_params = _count_mod(te2_mod)
        te3_params = _count_mod(te3_mod)

        known_mods = {trans_mod, vae_mod, te1_mod, te2_mod, te3_mod}
        
        if hasattr(full_pipeline, "components"):
            for k, v in full_pipeline.components.items():
                if isinstance(v, nn.Module) and v not in known_mods:
                    other_params += _count_mod(v)
    else:
        transformer_params = sum(p.numel() for p in model.parameters())

    total_full_params = transformer_params + vae_params + te1_params + te2_params + te3_params + other_params

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = trainable_params
    frozen_params = total_full_params - trainable_params
    trainable_percent = (trainable_params / total_full_params * 100.0) if total_full_params > 0 else 0.0

    return {
        "transformer_params": transformer_params,
        "vae_params": vae_params,
        "text_encoder_1_params": te1_params,
        "text_encoder_2_params": te2_params,
        "text_encoder_3_params": te3_params,
        "other_params": other_params,
        "total_full_params": total_full_params,
        "full_sd35_params": total_full_params,
        "total_params": total_full_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "lora_params": lora_params,
        "trainable_percent": round(trainable_percent, 4),
    }




def save_lora_weights(
    model: nn.Module,
    output_path: Union[str, Path],
    adapter_name: str = "default",
) -> Path:
    """
    Save LoRA weights in safetensors format.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model, PeftModel):
        state_dict = get_peft_model_state_dict(model, adapter_name=adapter_name)
    else:
        state_dict = {
            k: v for k, v in model.state_dict().items() if "lora_" in k and v.requires_grad
        }

    try:
        from safetensors.torch import save_file
        save_file(state_dict, str(out_p))
    except ImportError:
        torch.save(state_dict, str(out_p))

    _logger.info("Saved LoRA weights to '%s'", out_p)
    return out_p


def load_lora_weights(
    model: nn.Module,
    weight_path: Union[str, Path],
) -> nn.Module:
    """
    Load saved LoRA safetensors weights into a model.
    """
    w_path = Path(weight_path)
    if not w_path.exists():
        raise FileNotFoundError(f"LoRA weight file not found at '{w_path}'")

    try:
        from safetensors.torch import load_file
        state_dict = load_file(str(w_path))
    except ImportError:
        state_dict = torch.load(str(w_path), map_location="cpu")

    if isinstance(model, PeftModel):
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)

    _logger.info("Loaded LoRA weights from '%s'", w_path)
    return model
