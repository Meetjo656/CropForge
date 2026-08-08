"""
SD3.5 Inference Pipeline and Comparison Generator for CropForge Diffusion.

Orchestrates the end-to-end workflow:
DatasetSample -> PromptBuilder -> Prompt -> SD35Generator -> Generated Image -> Save PNG -> Save Metadata
Generates standard comparison directory layout under outputs/comparison/sample_{sample_id}/.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image

from cropforge.diffusion.Inference.sd35_generator import SD35Generator
from cropforge.diffusion.datasets.metadata import MetadataManager
from cropforge.diffusion.prompting import PromptBuilder
from cropforge.diffusion.schemas.sample_schema import DatasetSample

_logger = logging.getLogger(__name__)

__all__ = ["SD35InferencePipeline"]


class SD35InferencePipeline:
    """
    End-to-end inference pipeline and comparison directory generator for SD 3.5.

    Transforms DatasetSample metadata into prompts, executes generation via SD35Generator,
    saves the output image, positive/negative prompts, and sample metadata into
    structured comparison folders (`outputs/comparison/sample_{id}/`).
    """

    def __init__(
        self,
        generator: Optional[SD35Generator] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        output_base_dir: Union[str, Path] = "outputs/comparison",
    ) -> None:
        """
        Initialize SD35InferencePipeline.

        Args:
            generator: SD35Generator instance. If None, initialized with default parameters.
            prompt_builder: PromptBuilder instance. If None, initialized with default config.
            output_base_dir: Base output directory for comparison outputs (default: 'outputs/comparison').
        """
        self.prompt_builder = prompt_builder if prompt_builder is not None else PromptBuilder()
        self.generator = (
            generator if generator is not None else SD35Generator(prompt_builder=self.prompt_builder)
        )
        self.output_base_dir = Path(output_base_dir)

    def run_sample(
        self,
        sample: Union[DatasetSample, Dict[str, Any]],
        output_dir: Optional[Union[str, Path]] = None,
        copy_ground_truth: bool = True,
        **generator_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run the inference pipeline for a single DatasetSample and write comparison output.

        DatasetSample -> PromptBuilder -> Prompt -> SD35Generator -> Save PNG & Metadata

        Args:
            sample: DatasetSample object or dictionary.
            output_dir: Custom output folder for this sample. If None, uses `outputs/comparison/sample_{id}/`.
            copy_ground_truth: If True and target_image exists on disk, copies it as `ground_truth.png`.
            **generator_kwargs: Parameters passed to image generation (e.g. seed, height, width).

        Returns:
            Dictionary containing paths to all generated comparison artifacts:
            {'sample_dir', 'generated_png', 'prompt_txt', 'metadata_json', 'ground_truth_png'}
        """
        if isinstance(sample, dict):
            sample_obj = DatasetSample(**sample)
        elif isinstance(sample, DatasetSample):
            sample_obj = sample
        else:
            raise TypeError(f"Expected DatasetSample or dict, got {type(sample)}")

        # 1. Build Prompts
        positive_prompt, negative_prompt = self.prompt_builder.build_prompt_pair(sample_obj)

        # 2. Run SD35 Generator
        _logger.info("Executing generation for sample ID '%s'...", sample_obj.sample_id)
        generated_image: Image.Image = self.generator.generate_from_prompt(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            **generator_kwargs,
        )

        # 3. Determine and prepare output directory structure
        if output_dir is not None:
            sample_dir = Path(output_dir)
        else:
            sample_dir = self.output_base_dir / f"sample_{sample_obj.sample_id}"

        sample_dir.mkdir(parents=True, exist_ok=True)

        # 4. Save PNG (generated.png)
        generated_png_path = sample_dir / "generated.png"
        generated_image.save(generated_png_path, format="PNG")
        _logger.info("Saved generated image to '%s'", generated_png_path)

        # 5. Save Prompt text (prompt.txt)
        prompt_txt_path = sample_dir / "prompt.txt"
        with open(prompt_txt_path, "w", encoding="utf-8") as f:
            f.write(f"PROMPT:\n{positive_prompt}\n\nNEGATIVE_PROMPT:\n{negative_prompt}\n")
        _logger.info("Saved prompt details to '%s'", prompt_txt_path)

        # 6. Save Metadata JSON (metadata.json)
        metadata_json_path = sample_dir / "metadata.json"
        MetadataManager.save_metadata(sample_obj, metadata_json_path)
        _logger.info("Saved metadata JSON to '%s'", metadata_json_path)

        # 7. (Optional) Copy Ground Truth image if available
        ground_truth_path: Optional[Path] = None
        if copy_ground_truth:
            gt_source = Path(sample_obj.target_image)
            if not gt_source.exists():
                gt_source = Path(sample_obj.input_image)

            if gt_source.exists() and gt_source.is_file():
                dest_gt = sample_dir / "ground_truth.png"
                try:
                    shutil.copy2(gt_source, dest_gt)
                    ground_truth_path = dest_gt
                    _logger.info("Copied ground truth image to '%s'", dest_gt)
                except Exception as err:
                    _logger.warning("Failed to copy ground truth image: %s", err)

        return {
            "sample_id": sample_obj.sample_id,
            "sample_dir": str(sample_dir.resolve()),
            "generated_png": str(generated_png_path.resolve()),
            "prompt_txt": str(prompt_txt_path.resolve()),
            "metadata_json": str(metadata_json_path.resolve()),
            "ground_truth_png": str(ground_truth_path.resolve()) if ground_truth_path else None,
        }

    def run_batch(
        self,
        samples: List[Union[DatasetSample, Dict[str, Any]]],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Run the inference pipeline and comparison generator across a list of samples.

        Args:
            samples: List of DatasetSample objects or sample dictionaries.
            **kwargs: Generation options passed to `run_sample`.

        Returns:
            List of result dictionaries containing output artifact paths for each sample.
        """
        results: List[Dict[str, Any]] = []
        for idx, sample in enumerate(samples):
            _logger.info("Processing pipeline sample %d/%d", idx + 1, len(samples))
            res = self.run_sample(sample=sample, **kwargs)
            results.append(res)
        return results
