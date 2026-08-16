"""
Visual Mask Comparison Grid Generator for CropForge Milestone 11.

Generates 4-panel binary mask comparison grids:
[GT Day 0 Mask | Predicted Day-t Mask | GT Day-t Mask | Difference Overlay]
Difference Overlay Encoding:
- Green: Ground Truth only (Missed lesion area)
- Red: Predicted only (False positive lesion area)
- Yellow: Overlap (True positive lesion area)
"""

from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw


def create_isolated_mask_grid(
    gt_day0_mask: np.ndarray,
    pred_day14_mask: np.ndarray,
    gt_day14_mask: np.ndarray,
    save_path: Union[str, Path],
    plant_id: str,
    metrics: dict,
) -> Image.Image:
    """
    Renders 4-panel binary mask comparison grid for Milestone 11.
    """
    h, w = gt_day0_mask.shape[:2]
    margin = 15
    header_h = 65
    title_h = 30

    # Panel 1: GT Day 0 Mask (Grayscale)
    p1 = cv2.cvtColor(gt_day0_mask, cv2.COLOR_GRAY2RGB)

    # Panel 2: Predicted Day 14 Mask (Grayscale)
    p2 = cv2.cvtColor(pred_day14_mask, cv2.COLOR_GRAY2RGB)

    # Panel 3: GT Day 14 Mask (Grayscale)
    p3 = cv2.cvtColor(gt_day14_mask, cv2.COLOR_GRAY2RGB)

    # Panel 4: Difference Overlay
    diff = np.zeros((h, w, 3), dtype=np.uint8)
    gt_b = (gt_day14_mask > 127)
    pred_b = (pred_day14_mask > 127)
    diff[gt_b & ~pred_b] = [0, 220, 0]    # Green = Ground Truth only
    diff[pred_b & ~gt_b] = [220, 40, 40]   # Red = Forecasted only
    diff[gt_b & pred_b] = [220, 220, 0]   # Yellow = Both (Overlap)

    images = [Image.fromarray(p1), Image.fromarray(p2), Image.fromarray(p3), Image.fromarray(diff)]
    titles = [
        "GT Day 0 SAM2 Mask",
        "Predicted Day 14 Mask",
        "GT Day 14 SAM2 Mask",
        "Mask Diff Overlay (Green/Red)",
    ]

    total_w = len(images) * w + (len(images) + 1) * margin
    total_h = header_h + title_h + h + margin * 2

    grid = Image.new("RGB", (total_w, total_h), (240, 243, 248))
    draw = ImageDraw.Draw(grid)

    header_text = f"Milestone 11 Isolated Spatial Mask Evaluation: {plant_id.upper()}"
    metrics_str = (
        f"Mask IoU: {metrics['mask_iou']:.4f} | Mask Dice: {metrics['mask_dice']:.4f} | "
        f"Centroid Dist: {metrics['centroid_distance_px']:.1f}px | "
        f"Pred Sev: {metrics['predicted_severity'] * 100:.1f}% | GT Sev: {metrics['gt_severity'] * 100:.1f}% | "
        f"Sev Err: {metrics['severity_error'] * 100:.1f}%"
    )
    draw.text((margin, 10), header_text, fill=(15, 25, 45))
    draw.text((margin, 35), metrics_str, fill=(40, 80, 140))

    for idx, (img_item, title) in enumerate(zip(images, titles)):
        x = margin + idx * (w + margin)
        y = header_h + title_h + margin
        grid.paste(img_item, (x, y))
        draw.text((x + 10, header_h + 8), title, fill=(30, 40, 60))

    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_p)

    return grid
