"""
Model Loader for CropForge Diffusion Pipelines.

Provides a unified interface for loading Stable Diffusion 3.5 models,
applying optional LoRA weights, moving models to appropriate compute devices (GPU/CPU),
and caching loaded pipeline instances.

Public API:
    load_model(...) -> StableDiffusion3Pipeline
"""

from pathlib import Path as _Path
from typing import Any as _Any, Dict as _Dict, Optional as _Optional, Tuple as _Tuple, Union as _Union
import logging as _logging
import torch as _torch

from cropforge.diffusion.configs import load_config as _load_config

_logger = _logging.getLogger(__name__)

# Private global cache mapping cache keys -> pipeline instances
_MODEL_CACHE: _Dict[_Tuple[str, _Optional[str], str, str], _Any] = {}

__all__ = ["load_model"]


def load_model(
    model_id: _Optional[str] = None,
    lora_path: _Optional[_Union[str, _Path]] = None,
    device: _Optional[_Union[str, _torch.device]] = None,
    torch_dtype: _Optional[_torch.dtype] = None,
    force_reload: bool = False,
    **kwargs: _Any,
) -> _Any:
    """
    Load a Stable Diffusion 3.5 pipeline with optional LoRA weights, device assignment, and caching.

    Only public method of this module. Everything else stays internal.

    Args:
        model_id: HuggingFace model repo ID or local path for SD 3.5 (defaults to config).
        lora_path: Optional path to LoRA weights (file or directory) for future/custom adapter loading.
        device: Target compute device ('cuda', 'cpu', 'mps', or auto-detected if None).
        torch_dtype: Torch precision dtype (e.g., torch.bfloat16, torch.float16, torch.float32).
        force_reload: If True, bypasses the internal cache and reloads the pipeline.
        **kwargs: Additional keyword arguments passed to `from_pretrained`.

    Returns:
        The loaded diffusion pipeline ready for inference.
    """
    if model_id is None:
        try:
            cfg = _load_config()
            model_id = cfg.get("inference", {}).get("model", "stabilityai/stable-diffusion-3.5-medium")
        except Exception:
            model_id = "stabilityai/stable-diffusion-3.5-medium"

    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(torch_dtype, resolved_device)
    lora_key = str(_Path(lora_path).resolve()) if lora_path is not None else None
    cache_key = (model_id, lora_key, str(resolved_device), str(resolved_dtype))

    if not force_reload and cache_key in _MODEL_CACHE:
        _logger.info("Returning cached model instance for model_id='%s'", model_id)
        return _MODEL_CACHE[cache_key]

    _logger.info(
        "Loading SD3.5 model '%s' on device '%s' with dtype '%s'...",
        model_id,
        resolved_device,
        resolved_dtype,
    )
    pipe = _load_sd35_base_model(model_id=model_id, torch_dtype=resolved_dtype, **kwargs)

    if lora_path is not None:
        pipe = _apply_lora(pipe=pipe, lora_path=lora_path)

    pipe = _move_to_device(pipe=pipe, device=resolved_device)

    _MODEL_CACHE[cache_key] = pipe
    return pipe


def _resolve_device(device: _Optional[_Union[str, _torch.device]]) -> _torch.device:
    """Internal helper to determine the optimal available PyTorch device if not specified."""
    if device is not None:
        return _torch.device(device)
    if _torch.cuda.is_available():
        return _torch.device("cuda")
    if hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
        return _torch.device("mps")
    return _torch.device("cpu")


def _resolve_dtype(torch_dtype: _Optional[_torch.dtype], device: _torch.device) -> _torch.dtype:
    """Internal helper to determine default torch dtype based on device capabilities if not specified."""
    if torch_dtype is not None:
        return torch_dtype
    if device.type == "cuda":
        if _torch.cuda.is_bf16_supported():
            return _torch.bfloat16
        return _torch.float16
    return _torch.float32


import os as _os

# Load .env file if present in workspace root
_env_path = _Path(__file__).resolve().parents[3] / ".env"
if _env_path.exists():
    try:
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                if "=" in _line and not _line.startswith("#"):
                    _k, _v = _line.strip().split("=", 1)
                    _os.environ[_k.strip()] = _v.strip().strip('"').strip("'")
    except Exception:
        pass


def _load_sd35_base_model(model_id: str, torch_dtype: _torch.dtype, **kwargs: _Any) -> _Any:
    """Internal helper to load the SD 3.5 base pipeline from Hugging Face or local path."""
    try:
        from diffusers import StableDiffusion3Pipeline
    except ImportError as e:
        _logger.warning("diffusers package not available: %s", e)
        return None

    token = _os.environ.get("HF_TOKEN")
    if token and "token" not in kwargs:
        kwargs["token"] = token

    try:
        # First try loading cached local files
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            local_files_only=True,
            **kwargs,
        )
        return pipe
    except Exception:
        # If local cached weights are not present, try loading with network if HF token is present
        if token:
            try:
                pipe = StableDiffusion3Pipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    local_files_only=False,
                    **kwargs,
                )
                return pipe
            except Exception as err:
                _logger.warning("Could not load SD3.5 model '%s' from HF Hub: %s. Operating in simulation mode.", model_id, err)
                return None
        else:
            _logger.info("No local weights or HF_TOKEN found for '%s'. Operating in simulation pipeline mode.", model_id)
            return None



def _apply_lora(
    pipe: _Any,
    lora_path: _Union[str, _Path],
    adapter_name: _Optional[str] = None,
    weight_name: _Optional[str] = None,
) -> _Any:
    """Internal helper to load and attach LoRA weights to the pipeline (future expansion)."""
    lora_path_str = str(lora_path)
    _logger.info("Loading LoRA weights from '%s'...", lora_path_str)
    if hasattr(pipe, "load_lora_weights"):
        pipe.load_lora_weights(lora_path_str, weight_name=weight_name, adapter_name=adapter_name)
    else:
        _logger.warning("Pipeline object does not support `load_lora_weights`. Skipping LoRA loading.")
    return pipe


def _move_to_device(pipe: _Any, device: _torch.device) -> _Any:
    """Internal helper to move pipeline components to target compute device safely."""
    _logger.info("Moving pipeline to device %s", device)
    if device.type == "cuda":
        try:
            total_vram_gb = _torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            _logger.info("Target CUDA device has %.2f GB VRAM.", total_vram_gb)
            if total_vram_gb < 20.0 and hasattr(pipe, "enable_model_cpu_offload"):
                _logger.info("Enabling model CPU offloading for low-VRAM optimization.")
                pipe.enable_model_cpu_offload(device=device)
                return pipe
        except Exception as e:
            _logger.warning("Could not check VRAM properties: %s", e)

    if hasattr(pipe, "to"):
        try:
            res = pipe.to(device)
            return res if res is not None else pipe
        except Exception as err:
            _logger.warning("Direct GPU placement encountered warning: %s. Enabling model CPU offload fallback.", err)
            if hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload(device=device)
                return pipe
            elif hasattr(pipe, "enable_sequential_cpu_offload"):
                pipe.enable_sequential_cpu_offload(device=device)
                return pipe
    return pipe


