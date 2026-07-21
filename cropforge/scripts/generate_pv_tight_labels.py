"""
generate_pv_tight_labels.py
============================
Part 1 of the Tight-Label Pipeline.

Replaces the dummy whole-image YOLO annotations (0.5 0.5 1.0 1.0) with
tight bounding rectangles derived from HSV colour segmentation.

Pipeline (per image)
--------------------
  1. Resize to 640 x 640
  2. Convert BGR -> HSV
  3. Apply broad colour mask  (captures green, yellow, and brown disease tissue)
  4. Morphological open + close  (remove noise, fill gaps)
  5. Find the largest external contour
  6. cv2.boundingRect -> (x, y, w, h) in pixels
  7. Normalise to YOLO format  (cx, cy, nw, nh) in [0, 1]
  8. Write  <ClassName>__<stem>.txt  next to each image

Naming scheme (collision-proof)
--------------------------------
Each output file is prefixed with the class name:
    Tomato_healthy__<original_stem>.jpg
    Tomato_healthy__<original_stem>.txt
This ensures images from different classes NEVER overwrite each other
when stored in a single flat directory.

Usage
-----
  # Sanity-check mode — process 50 random images and save annotated previews:
  python generate_pv_tight_labels.py --mode verify

  # Full-dataset mode — process all PlantVillage images:
  python generate_pv_tight_labels.py --mode full

  # Custom source / destination:
  python generate_pv_tight_labels.py --mode full \\
      --src  D:/Crop-Forge/cropforge/datasets/raw/PlantVillage \\
      --dest D:/Crop-Forge/cropforge/datasets/processed/plantvillage_tight_labels

Output
------
  dest/
  ├── images/          ← 640×640 JPEGs (copied / written)
  ├── labels/          ← YOLO .txt files  (class_id cx cy nw nh)
  └── verify_previews/ ← annotated previews (verify mode only)
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Canonical 30-class list  (must match the existing dataset.yaml exactly)
# ---------------------------------------------------------------------------
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
CLASS_TO_ID = {name: idx for idx, name in enumerate(UNIFIED_CLASSES)}

# ---------------------------------------------------------------------------
# HSV thresholds  (H: 0-179, S: 0-255, V: 0-255 in OpenCV)
# Broad range to capture:
#   • healthy green leaves
#   • yellow / pale leaves
#   • brown diseased tissue
# ---------------------------------------------------------------------------
HSV_LOWER = np.array([20,  30, 30],  dtype=np.uint8)
HSV_UPPER = np.array([100, 255, 255], dtype=np.uint8)

TARGET_SIZE   = (640, 640)   # (width, height) for cv2.resize
MIN_MASK_FRAC = 0.02         # if mask < 2% of pixels → fallback to full image


# ---------------------------------------------------------------------------
# Core segmentation helpers
# ---------------------------------------------------------------------------

def hsv_leaf_mask(bgr: np.ndarray) -> np.ndarray:
    """Return a binary uint8 mask of leaf-coloured pixels."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    # Morphological cleanup: remove salt-pepper noise, close small holes
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def largest_contour_bbox(mask: np.ndarray):
    """
    Find the largest external contour in the mask and return
    (x, y, w, h) in pixel coordinates.  Returns None if no contour found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest)   # (x, y, w, h)


def to_yolo(x: int, y: int, w: int, h: int, W: int, H: int):
    """
    Convert pixel bbox to normalised YOLO format (cx, cy, nw, nh).
    All values are clamped to [0, 1].
    """
    cx = (x + w / 2.0) / W
    cy = (y + h / 2.0) / H
    nw = w / W
    nh = h / H
    cx, cy, nw, nh = (
        float(np.clip(cx, 0.0, 1.0)),
        float(np.clip(cy, 0.0, 1.0)),
        float(np.clip(nw, 0.0, 1.0)),
        float(np.clip(nh, 0.0, 1.0)),
    )
    return cx, cy, nw, nh


def segment_image(img_bgr: np.ndarray):
    """
    Run the full pipeline on a pre-loaded BGR image.

    Returns
    -------
    (cx, cy, nw, nh)  YOLO-normalised box
    used_fallback     bool — True if no contour was found and the full image box was used
    """
    W, H = TARGET_SIZE  # after resize these are always 640, 640
    mask = hsv_leaf_mask(img_bgr)

    bbox = None
    used_fallback = False

    total_pixels = W * H
    mask_pixels  = int(np.count_nonzero(mask))

    if mask_pixels / total_pixels >= MIN_MASK_FRAC:
        bbox = largest_contour_bbox(mask)

    if bbox is None:
        # Fallback: full-image box (same as the old dummy label)
        bbox = (0, 0, W, H)
        used_fallback = True

    cx, cy, nw, nh = to_yolo(*bbox, W, H)
    return (cx, cy, nw, nh), used_fallback


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def safe_stem(class_name: str, img_path: Path) -> str:
    """
    Build a collision-proof output stem:
        <ClassName>__<original_stem>
    e.g.  Tomato_healthy__0a1b2c3d___RS_HL_0001
    """
    return f"{class_name}__{img_path.stem}"


def process_image(img_path: Path, class_id: int, class_name: str,
                  img_out_dir: Path, lbl_out_dir: Path):
    """
    Read -> resize -> segment -> write image + label.

    Output filenames are prefixed with the class name to prevent collisions
    when images from multiple classes share the same original stem.

    Returns (used_fallback: bool) or None if the image could not be read.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    img_resized = cv2.resize(img, TARGET_SIZE)
    (cx, cy, nw, nh), used_fallback = segment_image(img_resized)

    stem = safe_stem(class_name, img_path)

    # Save image
    img_out_path = img_out_dir / (stem + ".jpg")
    cv2.imwrite(str(img_out_path), img_resized)

    # Save YOLO label
    lbl_out_path = lbl_out_dir / (stem + ".txt")
    with open(lbl_out_path, "w") as f:
        f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    return used_fallback


# ---------------------------------------------------------------------------
# Verify mode  (50 random images with visual annotations)
# ---------------------------------------------------------------------------

def run_verify(src_dir: Path, dest_dir: Path, n_samples: int = 50):
    """
    Randomly sample n_samples images from PlantVillage, run the pipeline,
    and save annotated preview images to dest/verify_previews/.
    Prints a summary with fallback statistics.
    """
    preview_dir = dest_dir / "verify_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    # Collect all (img_path, class_id, class_name) pairs
    all_items = []
    for class_dir in sorted(src_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if class_name not in CLASS_TO_ID:
            print(f"[WARN] Unknown class '{class_name}' - skipping.")
            continue
        class_id = CLASS_TO_ID[class_name]
        for ext in ("*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg"):
            for img_path in class_dir.glob(ext):
                all_items.append((img_path, class_id, class_name))

    if not all_items:
        print(f"[ERROR] No images found under {src_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(all_items):,} images across "
          f"{len(set(c for _, _, c in all_items))} classes.")

    random.seed(42)
    sample = random.sample(all_items, min(n_samples, len(all_items)))

    n_fallback = 0
    n_ok       = 0

    for img_path, class_id, class_name in sample:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[SKIP] Cannot read {img_path.name}")
            continue

        img_resized = cv2.resize(img, TARGET_SIZE)
        (cx, cy, nw, nh), used_fallback = segment_image(img_resized)

        if used_fallback:
            n_fallback += 1
        else:
            n_ok += 1

        # ---------- draw annotated preview ----------
        W, H = TARGET_SIZE
        x_pix = int((cx - nw / 2) * W)
        y_pix = int((cy - nh / 2) * H)
        w_pix = int(nw * W)
        h_pix = int(nh * H)

        preview = img_resized.copy()
        colour = (0, 0, 255) if used_fallback else (0, 255, 0)
        cv2.rectangle(preview, (x_pix, y_pix),
                      (x_pix + w_pix, y_pix + h_pix), colour, 2)

        label_txt = f"{class_name} | {'FALLBACK' if used_fallback else 'OK'}"
        cv2.putText(preview, label_txt, (x_pix, max(y_pix - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

        out_name = f"{class_name[:20]}_{img_path.stem[:20]}.jpg"
        cv2.imwrite(str(preview_dir / out_name), preview)

    print(f"\n{'='*55}")
    print("  VERIFY MODE — Results")
    print(f"{'='*55}")
    print(f"  Sampled              : {len(sample)}")
    print(f"  Tight boxes (OK)     : {n_ok}   ({n_ok/len(sample)*100:.1f}%)")
    print(f"  Fallback (full-img)  : {n_fallback}   ({n_fallback/len(sample)*100:.1f}%)")
    print(f"  Previews saved to    : {preview_dir}")
    print(f"{'='*55}\n")
    print("Inspect the previews.  Green box = tight leaf bbox.  "
          "Red box = fallback (full image).")
    print("If the green boxes look correct, run  --mode full  to process "
          "the entire dataset.")


# ---------------------------------------------------------------------------
# Full mode
# ---------------------------------------------------------------------------

def run_full(src_dir: Path, dest_dir: Path):
    """
    Process every PlantVillage image:
    Writes 640×640 JPEG + YOLO .txt label to dest/{images, labels}/.
    """
    img_out_dir = dest_dir / "images"
    lbl_out_dir = dest_dir / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    # Build task list
    all_items = []
    class_counts = {}
    for class_dir in sorted(src_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if class_name not in CLASS_TO_ID:
            print(f"[WARN] Unknown class '{class_name}' — skipping.")
            continue
        class_id = CLASS_TO_ID[class_name]
        imgs = []
        for ext in ("*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg"):
            imgs.extend(class_dir.glob(ext))
        class_counts[class_name] = len(imgs)
        for img_path in imgs:
            all_items.append((img_path, class_id))

    total = len(all_items)
    print(f"[INFO] Processing {total:,} images from {len(class_counts)} classes...")
    print(f"[INFO] Output -> {dest_dir}")

    n_ok       = 0
    n_fallback = 0
    n_error    = 0
    fallback_by_class: dict[str, int] = {}

    for i, (img_path, class_id, class_name) in enumerate(all_items):
        result = process_image(img_path, class_id, class_name, img_out_dir, lbl_out_dir)

        if result is None:
            n_error += 1
        elif result:   # used_fallback == True
            n_fallback += 1
            cname = img_path.parent.name
            fallback_by_class[cname] = fallback_by_class.get(cname, 0) + 1
        else:
            n_ok += 1

        if (i + 1) % 1000 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            print(f"  [{i+1:>6}/{total}]  {pct:5.1f}%  "
                  f"ok={n_ok}  fallback={n_fallback}  error={n_error}")

    print(f"\n{'='*60}")
    print("  FULL MODE — Summary")
    print(f"{'='*60}")
    print(f"  Total images         : {total:,}")
    print(f"  Tight boxes (OK)     : {n_ok:,}   ({n_ok/total*100:.2f}%)")
    print(f"  Fallback (full-img)  : {n_fallback:,}   ({n_fallback/total*100:.2f}%)")
    print(f"  Read errors          : {n_error:,}")
    print(f"{'='*60}")

    if fallback_by_class:
        print("\n  Fallback counts by class (worst first):")
        for cname, cnt in sorted(fallback_by_class.items(),
                                  key=lambda x: -x[1])[:10]:
            class_total = class_counts.get(cname, 1)
            print(f"    {cname:<45}  {cnt:>4}  ({cnt/class_total*100:.1f}%)")

    print(f"\n[DONE] Labels written to {lbl_out_dir}")
    print(f"[DONE] Images written to  {img_out_dir}")
    print("\nNext step: run  split_pv_dataset.py  to create train/val/test splits.")


# ---------------------------------------------------------------------------
# Argument parsing + entry-point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate tight YOLO labels for PlantVillage via HSV segmentation."
    )
    p.add_argument(
        "--mode", choices=["verify", "full"], required=True,
        help="'verify' = 50-image sanity check with visual previews; "
             "'full' = process all images."
    )
    p.add_argument(
        "--src",
        default=r"D:\Crop-Forge\cropforge\datasets\raw\PlantVillage",
        help="Root directory of the raw PlantVillage dataset "
             "(subdirectories = class names)."
    )
    p.add_argument(
        "--dest",
        default=r"D:\Crop-Forge\cropforge\datasets\processed\plantvillage_tight_labels",
        help="Output root directory."
    )
    p.add_argument(
        "--n-verify", type=int, default=50,
        help="Number of random images to sample in verify mode (default: 50)."
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    src_dir  = Path(args.src)
    dest_dir = Path(args.dest)

    if not src_dir.exists():
        print(f"[ERROR] Source directory not found: {src_dir}")
        sys.exit(1)

    if args.mode == "verify":
        run_verify(src_dir, dest_dir, n_samples=args.n_verify)
    else:
        run_full(src_dir, dest_dir)
