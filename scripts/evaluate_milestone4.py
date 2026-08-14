"""
Milestone 4 Evaluation Script for CropForge SD3.5 LoRA Pipeline.

Performs visual and qualitative evaluation comparing:
  Base SD3.5 vs. LoRA progression (Step 250, Step 500, Step 750, Step 1000, Final LoRA)
across fixed prompts and seeds.

Saves individual output PNGs, grid comparisons, and an evaluation report JSON.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image, ImageDraw, ImageFont

# Add workspace root to sys.path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Bypass broken xformers if needed
sys.modules["xformers"] = None
sys.modules["xformers.ops"] = None

import torch
from cropforge.diffusion.models import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_logger = logging.getLogger("evaluate_milestone4")

EVAL_PROMPTS = [
    {"id": "prompt_01", "name": "Tomato Early Blight", "prompt": "realistic photograph of a tomato leaf affected by early blight", "seed": 42},
    {"id": "prompt_02", "name": "Tomato Late Blight", "prompt": "realistic photograph of a tomato leaf affected by late blight", "seed": 43},
    {"id": "prompt_03", "name": "Healthy Tomato Leaf", "prompt": "realistic photograph of a healthy tomato leaf", "seed": 44},
    {"id": "prompt_04", "name": "Potato Early Blight", "prompt": "realistic photograph of a potato leaf affected by early blight", "seed": 45},
]

CHECKPOINTS = [
    {"label": "Base SD3.5", "type": "base", "path": None},
    {"label": "Step 250", "type": "checkpoint", "path": "outputs/diffusion/lora_train/checkpoints/checkpoint-000250/pytorch_lora_weights.safetensors"},
    {"label": "Step 500", "type": "checkpoint", "path": "outputs/diffusion/lora_train/checkpoints/checkpoint-000500/pytorch_lora_weights.safetensors"},
    {"label": "Step 750", "type": "checkpoint", "path": "outputs/diffusion/lora_train/checkpoints/checkpoint-000750/pytorch_lora_weights.safetensors"},
    {"label": "Step 1000", "type": "checkpoint", "path": "outputs/diffusion/lora_train/checkpoints/checkpoint-001000/pytorch_lora_weights.safetensors"},
    {"label": "Final LoRA", "type": "final", "path": "outputs/diffusion/lora_train/final/pytorch_lora_weights.safetensors"},
]


def create_comparison_grid(images: List[Image.Image], labels: List[str], title: str) -> Image.Image:
    """Create a side-by-side comparison grid with labels for milestone evaluation."""
    if not images:
        return Image.new("RGB", (512, 512), (255, 255, 255))

    img_w, img_h = images[0].size
    header_h = 60
    label_h = 40
    num_imgs = len(images)

    grid_w = img_w * num_imgs
    grid_h = img_h + header_h + label_h

    grid = Image.new("RGB", (grid_w, grid_h), (240, 240, 240))
    draw = ImageDraw.Draw(grid)

    # Title header
    draw.rectangle([(0, 0), (grid_w, header_h)], fill=(30, 40, 60))
    draw.text((20, 15), title, fill=(255, 255, 255))

    # Add images and labels
    for idx, (img, label) in enumerate(zip(images, labels)):
        x_offset = idx * img_w
        y_offset = header_h

        grid.paste(img, (x_offset, y_offset))

        # Label footer
        draw.rectangle([(x_offset, y_offset + img_h), (x_offset + img_w, y_offset + img_h + label_h)], fill=(50, 60, 80))
        draw.text((x_offset + 15, y_offset + img_h + 10), label, fill=(255, 255, 255))

    return grid


def evaluate_milestone4(output_dir: str = "outputs/evaluation/milestone4") -> Dict[str, Any]:
    """Execute Milestone 4 evaluation comparing Base SD3.5 vs LoRA checkpoints."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    _logger.info("Starting Milestone 4 Evaluation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # 1. Load Base SD3.5 Pipeline
    _logger.info("Loading Base SD3.5 pipeline...")
    pipe = load_model(device=device, torch_dtype=dtype)

    summary_results: Dict[str, Any] = {
        "evaluation_name": "Milestone 4 — Base vs. LoRA Progression",
        "checkpoints": [c["label"] for c in CHECKPOINTS],
        "prompts": EVAL_PROMPTS,
        "eval_outputs": {},
    }

    for item in EVAL_PROMPTS:
        p_id = item["id"]
        p_name = item["name"]
        prompt = item["prompt"]
        seed = item["seed"]

        p_dir = out_path / f"{p_id}_{p_name.lower().replace(' ', '_')}"
        p_dir.mkdir(parents=True, exist_ok=True)

        _logger.info("Evaluating prompt '%s' (seed=%d)...", p_name, seed)
        item_images: List[Image.Image] = []
        item_labels: List[str] = []

        for ckpt in CHECKPOINTS:
            label = ckpt["label"]
            ckpt_path = ckpt["path"]

            generator = torch.Generator(device=device).manual_seed(seed)

            if ckpt["type"] == "base":
                # Unload any previously loaded LoRA
                if hasattr(pipe, "unload_lora_weights"):
                    try:
                        pipe.unload_lora_weights()
                    except Exception:
                        pass
                _logger.info("Generating Base SD3.5 image for '%s'...", label)
            else:
                # Load specific LoRA weight
                if ckpt_path and Path(ckpt_path).exists():
                    _logger.info("Loading LoRA adapter '%s' from '%s'...", label, ckpt_path)
                    if hasattr(pipe, "unload_lora_weights"):
                        try:
                            pipe.unload_lora_weights()
                        except Exception:
                            pass
                    pipe.load_lora_weights(str(Path(ckpt_path).parent), weight_name=Path(ckpt_path).name)

            # Generate image
            with torch.inference_mode():
                res = pipe(
                    prompt=prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    generator=generator,
                )
                img = res.images[0]

            filename = f"{label.lower().replace(' ', '_')}.png"
            img_path = p_dir / filename
            img.save(img_path)

            item_images.append(img)
            item_labels.append(label)

        # Create side-by-side grid
        grid_img = create_comparison_grid(item_images, item_labels, title=f"Milestone 4 Progression: {p_name}")
        grid_path = p_dir / "comparison_grid.png"
        grid_img.save(grid_path)

        summary_results["eval_outputs"][p_id] = {
            "prompt_name": p_name,
            "prompt": prompt,
            "seed": seed,
            "directory": str(p_dir.resolve()),
            "grid_path": str(grid_path.resolve()),
        }

    # Save summary report
    report_file = out_path / "evaluation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=4)

    _logger.info("Milestone 4 Evaluation completed successfully! Report saved to '%s'", report_file)
    return summary_results


if __name__ == "__main__":
    evaluate_milestone4()
