"""
Validation image evaluator for SD 3.5 LoRA training.
Generates images at scheduled step intervals using fixed prompts and seeds,
storing output PNGs alongside JSON metadata to track model progression.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import torch

from cropforge.diffusion.training.config import ValidationConfig

_logger = logging.getLogger(__name__)


class ValidationEvaluator:
    """
    Evaluates LoRA model checkpoints during training by generating images from fixed prompts.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        config: Optional[ValidationConfig] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.validation_dir = self.output_dir / "validation"
        self.config = config if config is not None else ValidationConfig()
        self.validation_dir.mkdir(parents=True, exist_ok=True)

    def should_validate(self, step: int) -> bool:
        """Return True if validation should run at this step."""
        if not self.config.enabled:
            return False
        if step <= 0:
            return False
        return (step % self.config.every_n_steps) == 0

    def run_validation(
        self,
        step: int,
        pipeline: Any,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Path:
        """
        Execute validation loop across all configured validation prompts and save results.

        Args:
            step: Current training step.
            pipeline: Pre-loaded SD3.5 pipeline (with current LoRA adapters active).
            device: Compute device.

        Returns:
            Path to the step validation output directory.
        """
        step_dir = self.validation_dir / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        _logger.info("Executing validation run for step %d (%d prompts)...", step, len(self.config.prompts))

        base_seed = self.config.seed
        results_meta: List[Dict[str, Any]] = []

        for idx, prompt in enumerate(self.config.prompts):
            sample_seed = base_seed + idx
            image_filename = f"val_prompt_{idx + 1:02d}.png"
            image_path = step_dir / image_filename

            img = self._generate_image(
                pipeline=pipeline,
                prompt=prompt,
                seed=sample_seed,
                device=device,
            )

            img.save(image_path, format="PNG")

            meta = {
                "step": step,
                "prompt_index": idx + 1,
                "prompt": prompt,
                "seed": sample_seed,
                "image_filename": image_filename,
                "image_path": str(image_path.resolve()),
            }
            results_meta.append(meta)
            _logger.info("Saved validation image %d/%d: '%s'", idx + 1, len(self.config.prompts), image_path)

        with open(step_dir / "validation_summary.json", "w", encoding="utf-8") as f:
            json.dump({"step": step, "results": results_meta}, f, indent=4)

        return step_dir

    def _generate_image(
        self,
        pipeline: Any,
        prompt: str,
        seed: int,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Image.Image:
        """Internal helper to run generation on pipeline or fallback dummy image generation for CPU dry-run."""
        if pipeline is None:
            # Fallback for mock pipeline dry runs
            arr = torch.randint(0, 255, (512, 512, 3), dtype=torch.uint8).numpy()
            return Image.fromarray(arr)

        dev = device if device is not None else getattr(pipeline, "device", "cpu")
        generator = torch.Generator(device=dev).manual_seed(seed)

        # Call pipeline
        try:
            with torch.inference_mode():
                out = pipeline(
                    prompt=prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    generator=generator,
                )
                if hasattr(out, "images") and len(out.images) > 0:
                    return out.images[0]
                elif isinstance(out, list) and len(out) > 0:
                    return out[0]
                elif isinstance(out, Image.Image):
                    return out
        except Exception as err:
            _logger.warning("Pipeline generation failed in validation: %s. Using placeholder.", err)

        arr = torch.randint(0, 255, (512, 512, 3), dtype=torch.uint8).numpy()
        return Image.fromarray(arr)
