"""
SD3.5 Generator for CropForge Diffusion.

Provides the SD35Generator class to generate PIL Images from prompts or dataset samples.
Decoupled from dataset builders and storage logic.
All default parameters are loaded dynamically from YAML configuration.
"""

from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import torch

from cropforge.diffusion.configs import load_config
from cropforge.diffusion.models.model_loader import load_model
from cropforge.diffusion.prompting import PromptBuilder
from cropforge.diffusion.schemas.sample_schema import DatasetSample

_logger = logging.getLogger(__name__)

__all__ = ["SD35Generator"]


class SD35Generator:
    """
    High-level generator interface for Stable Diffusion 3.5.

    Accepts prompts or DatasetSample metadata, runs diffusion pipeline inference,
    records detailed generation telemetry logs, and returns generated PIL Image objects.
    """

    def __init__(
        self,
        pipe: Optional[Any] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        config: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = None,
        lora_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Initialize SD35Generator.

        Args:
            pipe: Pre-loaded diffusion pipeline instance. If None, loaded via `load_model(...)`.
            prompt_builder: Custom PromptBuilder instance. If None, initialized with default config.
            config: Custom configuration dictionary. If None, loaded via `load_config()`.
            model_id: Model repository ID or local path (defaults to config if None).
            lora_path: Optional path to LoRA weights (defaults to config if None).
            device: Target device (used if pipe is None).
            torch_dtype: Target torch precision (used if pipe is None).
        """
        self.config = config if config is not None else load_config()
        self.inference_cfg = self.config.get("inference", {})

        effective_model_id = (
            model_id if model_id is not None else self.inference_cfg.get("model", "stabilityai/stable-diffusion-3.5-medium")
        )
        effective_lora = lora_path if lora_path is not None else self.inference_cfg.get("lora_path")
        self.model_id = effective_model_id

        if pipe is not None:
            self.pipe = pipe
        else:
            _logger.info("Initializing SD35Generator by loading model '%s'...", effective_model_id)
            self.pipe = load_model(
                model_id=effective_model_id,
                lora_path=effective_lora,
                device=device,
                torch_dtype=torch_dtype,
            )

        self.prompt_builder = prompt_builder if prompt_builder is not None else PromptBuilder(config=self.config)
        self.last_generation_log: Dict[str, Any] = {}

    def generate_from_prompt(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """
        Generate a single PIL Image directly from prompt and negative_prompt strings.
        Records detailed generation telemetry logs (timestamp, seed, guidance, steps, scheduler, model, time).

        Args:
            prompt: Positive text prompt describing the desired image.
            negative_prompt: Optional negative text prompt. Defaults to PromptBuilder negative keywords if None.
            height: Output image height in pixels (defaults to config).
            width: Output image width in pixels (defaults to config).
            num_inference_steps: Number of diffusion sampling steps (defaults to config).
            guidance_scale: Classifier-free guidance scale (defaults to config).
            seed: Optional random seed for reproducible generation (defaults to config).
            **kwargs: Additional parameters passed to the diffusion pipeline call.

        Returns:
            Generated PIL Image object.
        """
        if negative_prompt is None:
            negative_prompt = self.prompt_builder.get_negative_prompt()

        eff_height = height if height is not None else self.inference_cfg.get("height", 1024)
        eff_width = width if width is not None else self.inference_cfg.get("width", 1024)
        eff_steps = num_inference_steps if num_inference_steps is not None else self.inference_cfg.get("steps", 30)
        eff_guidance = guidance_scale if guidance_scale is not None else self.inference_cfg.get("guidance_scale", 7.5)
        eff_seed = seed if seed is not None else self.inference_cfg.get("seed", 42)

        pipe_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": eff_height,
            "width": eff_width,
            "num_inference_steps": eff_steps,
            "guidance_scale": eff_guidance,
            **kwargs,
        }

        if eff_seed is not None:
            pipe_device = getattr(self.pipe, "device", "cpu")
            generator = torch.Generator(device=pipe_device).manual_seed(eff_seed)
            pipe_kwargs["generator"] = generator

        # Measure generation time & log telemetry (Task 5)
        start_time = time.perf_counter()
        _logger.info("Generating image from prompt: '%s'", prompt[:60])

        output = self.pipe(**pipe_kwargs)

        elapsed = time.perf_counter() - start_time

        scheduler_obj = getattr(self.pipe, "scheduler", None)
        scheduler_name = (
            scheduler_obj.__class__.__name__ if scheduler_obj is not None else "UnknownScheduler"
        )
        timestamp_str = datetime.now(timezone.utc).isoformat()

        self.last_generation_log = {
            "timestamp": timestamp_str,
            "seed": eff_seed,
            "guidance_scale": eff_guidance,
            "steps": eff_steps,
            "scheduler": scheduler_name,
            "model": self.model_id,
            "negative_prompt": negative_prompt,
            "generation_time": round(elapsed, 4),
        }

        _logger.info(
            "Generation finished in %.4fs | Seed: %s | Steps: %d | Guidance: %.2f | Scheduler: %s",
            elapsed,
            eff_seed,
            eff_steps,
            eff_guidance,
            scheduler_name,
        )

        if hasattr(output, "images") and len(output.images) > 0:
            return output.images[0]
        elif isinstance(output, list) and len(output) > 0:
            return output[0]
        elif isinstance(output, Image.Image):
            return output
        else:
            raise RuntimeError(f"Unexpected output format from pipeline: {type(output)}")

    def generate(
        self,
        sample: Union[DatasetSample, Dict[str, Any]],
        **kwargs: Any,
    ) -> Image.Image:
        """
        Generate a PIL Image for a given DatasetSample or sample metadata dictionary.

        Args:
            sample: DatasetSample instance or dictionary conforming to sample schema.
            **kwargs: Generation parameters (e.g. height, width, seed) passed to generate_from_prompt.

        Returns:
            Generated PIL Image object.
        """
        if isinstance(sample, dict):
            sample_obj = DatasetSample(**sample)
        elif isinstance(sample, DatasetSample):
            sample_obj = sample
        else:
            raise TypeError(f"Expected DatasetSample or dict, got {type(sample)}")

        positive_prompt, negative_prompt = self.prompt_builder.build_prompt_pair(sample_obj)
        return self.generate_from_prompt(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            **kwargs,
        )

    def generate_batch(
        self,
        samples: List[Union[DatasetSample, Dict[str, Any], str]],
        **kwargs: Any,
    ) -> List[Image.Image]:
        """
        Generate a batch of PIL Images for a list of DatasetSamples, dictionaries, or prompt strings.

        Args:
            samples: List of DatasetSample objects, sample dicts, or prompt strings.
            **kwargs: Common generation options.

        Returns:
            List of generated PIL Image objects.
        """
        images: List[Image.Image] = []
        for idx, item in enumerate(samples):
            _logger.info("Processing batch item %d/%d", idx + 1, len(samples))
            if isinstance(item, str):
                img = self.generate_from_prompt(prompt=item, **kwargs)
            else:
                img = self.generate(sample=item, **kwargs)
            images.append(img)
        return images
