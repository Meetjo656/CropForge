"""
SD3.5 Generator for CropForge Diffusion.

Provides the SD35Generator class to generate PIL Images from prompts or dataset samples.
Decoupled from dataset builders and storage logic.
"""

from typing import Any, Dict, List, Optional, Union
import logging
from PIL import Image
import torch

from cropforge.diffusion.models.model_loader import load_model
from cropforge.diffusion.prompting import PromptBuilder
from cropforge.diffusion.schemas.sample_schema import DatasetSample

_logger = logging.getLogger(__name__)

__all__ = ["SD35Generator"]


class SD35Generator:
    """
    High-level generator interface for Stable Diffusion 3.5.

    Accepts prompts or DatasetSample metadata, runs diffusion pipeline inference,
    and returns generated PIL Image objects.
    """

    def __init__(
        self,
        pipe: Optional[Any] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        model_id: str = "stabilityai/stable-diffusion-3.5-large",
        lora_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Initialize SD35Generator.

        Args:
            pipe: Pre-loaded diffusion pipeline instance. If None, loaded via `load_model(...)`.
            prompt_builder: Custom PromptBuilder instance. If None, initialized with default configuration.
            model_id: Model repository ID or local path (used if pipe is None).
            lora_path: Optional path to LoRA weights (used if pipe is None).
            device: Target device (used if pipe is None).
            torch_dtype: Target torch precision (used if pipe is None).
        """
        if pipe is not None:
            self.pipe = pipe
        else:
            _logger.info("Initializing SD35Generator by loading model '%s'...", model_id)
            self.pipe = load_model(
                model_id=model_id,
                lora_path=lora_path,
                device=device,
                torch_dtype=torch_dtype,
            )

        self.prompt_builder = prompt_builder if prompt_builder is not None else PromptBuilder()

    def generate_from_prompt(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 4.5,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """
        Generate a single PIL Image directly from prompt and negative_prompt strings.

        Args:
            prompt: Positive text prompt describing the desired image.
            negative_prompt: Optional negative text prompt. Defaults to PromptBuilder negative keywords if None.
            height: Output image height in pixels.
            width: Output image width in pixels.
            num_inference_steps: Number of diffusion sampling steps.
            guidance_scale: Classifier-free guidance scale.
            seed: Optional random seed for reproducible generation.
            **kwargs: Additional parameters passed to the diffusion pipeline call.

        Returns:
            Generated PIL Image object.
        """
        if negative_prompt is None:
            negative_prompt = self.prompt_builder.get_negative_prompt()

        pipe_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            **kwargs,
        }

        if seed is not None:
            pipe_device = getattr(self.pipe, "device", "cpu")
            generator = torch.Generator(device=pipe_device).manual_seed(seed)
            pipe_kwargs["generator"] = generator

        _logger.info("Generating image from prompt: '%s'", prompt[:60])
        output = self.pipe(**pipe_kwargs)

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
            **kwargs: Common generation options (e.g. height, width, num_inference_steps).

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
