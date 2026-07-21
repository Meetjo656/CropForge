"""
visualize_yolo_labels.py
=========================
Draw YOLO bounding-box annotations on their matching images and save
annotated previews to an output folder.

Supports any YOLO dataset laid out as:

    root/
    ├── images/   ← .jpg / .png files
    └── labels/   ← .txt files with lines  "class_id cx cy nw nh"

Usage
-----
  # Visualise a random sample of 50 images:
  python visualize_yolo_labels.py

  # Visualise all images in a specific dataset:
  python visualize_yolo_labels.py \\
      --img-dir  D:/Crop-Forge/.../images \\
      --lbl-dir  D:/Crop-Forge/.../labels \\
      --out-dir  D:/Crop-Forge/.../previews \\
      --n 100

  # Visualise the plantvillage_tight_labels dataset:
  python visualize_yolo_labels.py --dataset pv_tight
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Default dataset roots (edit or override with CLI flags)
# ---------------------------------------------------------------------------
DATASETS = {
    "pv_tight": {
        "img_dir": r"D:\Crop-Forge\cropforge\datasets\processed\plantvillage_tight_labels\images",
        "lbl_dir": r"D:\Crop-Forge\cropforge\datasets\processed\plantvillage_tight_labels\labels",
        "out_dir": r"D:\Crop-Forge\cropforge\datasets\processed\plantvillage_tight_labels\previews",
    },
    "merged_balanced_train": {
        "img_dir": r"D:\Crop-Forge\cropforge\datasets\processed\merged_balanced\train\images",
        "lbl_dir": r"D:\Crop-Forge\cropforge\datasets\processed\merged_balanced\train\labels",
        "out_dir": r"D:\Crop-Forge\cropforge\datasets\processed\merged_balanced\train\previews",
    },
}

# 30-class canonical list (matches dataset.yaml)
UNIFIED_CLASSES = sorted([
    'Apple_Scab', 'Apple_healthy', 'Apple_rust', 'Blueberry_healthy',
    'Cherry_healthy', 'Corn_Gray_leaf_spot', 'Corn_leaf_blight', 'Corn_rust',
    'Grape_black_rot', 'Grape_healthy', 'Peach_healthy',
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew',
    'Strawberry_healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
    'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy',
])
ID_TO_CLASS = {i: n for i, n in enumerate(UNIFIED_CLASSES)}

# Colour palette (BGR) — one per class, cycling if > 30 classes
_PALETTE = [
    (0, 255, 0),   (0, 165, 255), (0, 0, 255),   (255, 0, 0),
    (255, 255, 0), (0, 255, 255), (128, 0, 128),  (255, 165, 0),
    (0, 128, 255), (255, 0, 128), (0, 200, 200),  (200, 0, 200),
    (200, 200, 0), (100, 255, 100),(255, 100, 100),(100, 100, 255),
    (180, 60, 60), (60, 180, 60), (60, 60, 180),  (180, 180, 60),
    (60, 180, 180),(180, 60, 180),(230, 115, 0),  (0, 230, 115),
    (115, 0, 230), (230, 0, 115), (115, 230, 0),  (0, 115, 230),
    (150, 80, 200),(200, 150, 80),
]


def class_colour(class_id: int):
    return _PALETTE[class_id % len(_PALETTE)]


def draw_boxes(img_bgr: np.ndarray, label_path: Path) -> np.ndarray:
    """
    Overlay YOLO boxes and class labels on a copy of *img_bgr*.
    Returns the annotated copy.
    """
    H, W = img_bgr.shape[:2]
    out = img_bgr.copy()

    if not label_path.exists():
        cv2.putText(out, "NO LABEL FILE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return out

    with open(label_path) as f:
        lines = f.read().strip().splitlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(parts[0])
        cx, cy, nw, nh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        x1 = int((cx - nw / 2) * W)
        y1 = int((cy - nh / 2) * H)
        x2 = int((cx + nw / 2) * W)
        y2 = int((cy + nh / 2) * H)

        # Clamp to image bounds
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, W - 1), min(y2, H - 1)

        colour = class_colour(class_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        class_name = ID_TO_CLASS.get(class_id, str(class_id))
        short_name = class_name[:25]
        text = f"{short_name} ({cx:.3f},{cy:.3f},{nw:.3f},{nh:.3f})"
        text_y = max(y1 - 6, 14)
        cv2.putText(out, text, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)

    return out


def visualize(img_dir: Path, lbl_dir: Path, out_dir: Path, n: int, seed: int):
    img_dir = Path(img_dir)
    lbl_dir = Path(lbl_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = (".jpg", ".JPG", ".jpeg", ".png", ".PNG")
    all_imgs = [p for p in img_dir.iterdir() if p.suffix in exts]

    if not all_imgs:
        print(f"[ERROR] No images found in {img_dir}")
        sys.exit(1)

    random.seed(seed)
    sample = random.sample(all_imgs, min(n, len(all_imgs)))
    print(f"[INFO] Visualising {len(sample)} / {len(all_imgs)} images → {out_dir}")

    n_ok = 0
    n_missing_label = 0
    n_error = 0

    for img_path in sample:
        img = cv2.imread(str(img_path))
        if img is None:
            n_error += 1
            continue

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            n_missing_label += 1

        annotated = draw_boxes(img, lbl_path)
        cv2.imwrite(str(out_dir / img_path.name), annotated)
        n_ok += 1

    print(f"[DONE] Written  : {n_ok}")
    print(f"[WARN] Missing labels : {n_missing_label}")
    print(f"[ERR]  Read errors    : {n_error}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Overlay YOLO annotations on images and save previews."
    )
    p.add_argument(
        "--dataset", choices=list(DATASETS.keys()), default=None,
        help="Named shortcut for a known dataset (overrides --img-dir etc.)."
    )
    p.add_argument("--img-dir", default=None,
                   help="Directory containing images (.jpg/.png).")
    p.add_argument("--lbl-dir", default=None,
                   help="Directory containing YOLO .txt label files.")
    p.add_argument("--out-dir", default=None,
                   help="Output directory for annotated preview images.")
    p.add_argument("--n", type=int, default=50,
                   help="Number of images to visualise (default: 50).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for image sampling (default: 42).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.dataset:
        cfg = DATASETS[args.dataset]
        img_dir = Path(args.img_dir or cfg["img_dir"])
        lbl_dir = Path(args.lbl_dir or cfg["lbl_dir"])
        out_dir = Path(args.out_dir or cfg["out_dir"])
    elif args.img_dir and args.lbl_dir:
        img_dir = Path(args.img_dir)
        lbl_dir = Path(args.lbl_dir)
        out_dir = Path(args.out_dir) if args.out_dir else img_dir.parent / "previews"
    else:
        # Default: pv_tight
        cfg = DATASETS["pv_tight"]
        img_dir = Path(cfg["img_dir"])
        lbl_dir = Path(cfg["lbl_dir"])
        out_dir = Path(cfg["out_dir"])

    visualize(img_dir, lbl_dir, out_dir, n=args.n, seed=args.seed)
