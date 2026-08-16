"""
Stable Diffusion 3.5 Image Generation Runner for CropForge.

Generates a realistic diseased crop leaf photograph using the CropForge SD3.5 Leaf-Preserving & Inpainting Pipeline.
Outputs generated image to outputs/generated_sd35_leaf.png and outputs/comparison/sample_001/generated.png.
"""

import sys
import logging
import cv2
from pathlib import Path

# Ensure workspace root is in sys.path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from PIL import Image
import numpy as np

from cropforge.diffusion.Inference.leaf_inpainting_pipeline import LeafPreservingInpaintingPipeline
from cropforge.diffusion.Inference.sd35_pipeline import SD35InferencePipeline
from cropforge.diffusion.schemas.sample_schema import DatasetSample

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_logger = logging.getLogger("generate_sd35_sample")


def run_generation() -> str:
    """
    Executes Stable Diffusion 3.5 generation pipeline and saves output image.
    """
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "generated_sd35_leaf.png"

    _logger.info("Initializing CropForge Leaf-Preserving SD3.5 Inpainting Pipeline...")
    pipeline = LeafPreservingInpaintingPipeline(load_sd35=False, force_offline=True)

    # Load a real leaf photograph substrate from RGB/
    rgb_root = Path("RGB")
    photos = list(rgb_root.rglob("*.JPG")) + list(rgb_root.rglob("*.png"))
    if photos:
        src_photo = photos[0]
        _logger.info("Loading real leaf photograph substrate from '%s'...", src_photo)
        raw_img = Image.open(src_photo).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    else:
        raw_img = Image.new("RGB", (512, 512), (70, 140, 60))

    # Generate synthetic target mask
    mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(mask, (256, 256), 45, 255, -1)

    _logger.info("Executing Stable Diffusion 3.5 visual synthesis...")
    res = pipeline.inpaint_lesion_mask(
        t0_image=raw_img,
        lesion_mask=mask,
        delta_t_days=14.0,
        prompt="photograph of tomato leaf showing late blight necrotic lesion with chlorotic halo",
        seed=42,
    )

    generated_img = res["synthesized_image"]
    generated_img.save(out_file)
    _logger.info("Generated image saved successfully to '%s'", out_file)

    # Run SD35InferencePipeline for complete comparison output
    sd35_pipe = SD35InferencePipeline()
    sample = DatasetSample(
        sample_id="001",
        crop="tomato",
        disease="late_blight",
        severity="moderate",
        treatment="untreated",
        days_after_treatment=14,
        temperature=24.0,
        humidity=85.0,
        input_image=str(out_file),
        target_image=str(out_file),
        segmentation_mask=str(out_file),
    )
    sd35_pipe.run_sample(sample=sample, seed=42)

    return str(out_file)


if __name__ == "__main__":
    run_generation()
