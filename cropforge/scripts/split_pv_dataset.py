"""
split_pv_dataset.py
====================
Takes the flat plantvillage_tight_labels/{images,labels}/ output from
generate_pv_tight_labels.py and builds a proper YOLO dataset with:

    plantvillage_tight_labels/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── dataset.yaml

Split ratio: 70 / 20 / 10  (train / val / test)
Stratified by class (inferred from the class_id in each .txt file).
"""

import random
import shutil
import yaml
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(r"D:\Crop-Forge\cropforge\datasets\processed\plantvillage_tight_labels")
SPLIT_RATIO = (0.70, 0.20, 0.10)
SEED = 42
CLASS_TO_ID = None

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

# ---------------------------------------------------------------------------

def class_from_stem(stem: str) -> str | None:
    """
    Extract the class name from a collision-proof stem.
    Format: ClassName__original_stem
    Falls back to reading the label file for legacy flat stems.
    """
    if "__" in stem:
        return stem.split("__", 1)[0]
    return None


def read_class_id(lbl_path: Path) -> int | None:
    """Return the integer class_id from the first line of a YOLO label file."""
    try:
        with open(lbl_path) as f:
            line = f.readline().strip()
        return int(line.split()[0])
    except Exception:
        return None


def create_split_dirs():
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            (ROOT / split / sub).mkdir(parents=True, exist_ok=True)


def copy_item(img_path: Path, lbl_path: Path, split: str):
    shutil.copy2(img_path, ROOT / split / "images" / img_path.name)
    shutil.copy2(lbl_path, ROOT / split / "labels" / lbl_path.name)


def write_yaml():
    yaml_path = ROOT / "dataset.yaml"
    data = {
        "path": str(ROOT.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    len(UNIFIED_CLASSES),
        "names": ID_TO_CLASS,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"[INFO] dataset.yaml written to {yaml_path}")


def main():
    random.seed(SEED)

    img_dir = ROOT / "images"
    lbl_dir = ROOT / "labels"

    if not img_dir.exists():
        raise FileNotFoundError(f"Images not found at {img_dir}")

    # ----- Group stems by class_id -----
    by_class: dict[int, list[str]] = defaultdict(list)
    skipped = 0

    for img_path in sorted(img_dir.glob("*.jpg")):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            skipped += 1
            continue

        # Fast path: extract class from collision-proof stem prefix
        cname = class_from_stem(img_path.stem)
        if cname and cname in CLASS_TO_ID:
            cid = CLASS_TO_ID[cname]
        else:
            # Legacy flat stems: fall back to reading the label file
            cid = read_class_id(lbl_path)

        if cid is None:
            skipped += 1
            continue
        by_class[cid].append(img_path.stem)

    total = sum(len(v) for v in by_class.values())
    print(f"[INFO] {total:,} paired image+label files found across "
          f"{len(by_class)} classes.  Skipped: {skipped}")

    # ----- Stratified split -----
    split_counts = {"train": 0, "val": 0, "test": 0}
    create_split_dirs()

    for cid, stems in sorted(by_class.items()):
        random.shuffle(stems)
        n = len(stems)
        n_train = int(n * SPLIT_RATIO[0])
        n_val   = int(n * SPLIT_RATIO[1])

        partitions = {
            "train": stems[:n_train],
            "val":   stems[n_train : n_train + n_val],
            "test":  stems[n_train + n_val :],
        }

        for split, part_stems in partitions.items():
            for stem in part_stems:
                copy_item(
                    img_dir / (stem + ".jpg"),
                    lbl_dir / (stem + ".txt"),
                    split,
                )
                split_counts[split] += 1

    # ----- Report -----
    print(f"\n{'='*55}")
    print("  Split Summary")
    print(f"{'='*55}")
    for split in ("train", "val", "test"):
        n = split_counts[split]
        print(f"  {split:<6}: {n:>6,}  ({n/total*100:.1f}%)")
    print(f"  Total : {total:>6,}")
    print(f"{'='*55}")

    # Per-class breakdown
    print("\n  Per-class breakdown (class_id | name | train | val | test):")
    for cid, stems in sorted(by_class.items()):
        n  = len(stems)
        nt = int(n * SPLIT_RATIO[0])
        nv = int(n * SPLIT_RATIO[1])
        nts = n - nt - nv
        cname = ID_TO_CLASS.get(cid, str(cid))
        print(f"    {cid:>2}  {cname:<50}  {nt:>4} | {nv:>4} | {nts:>4}")

    write_yaml()
    print("\n[DONE] Split complete. Dataset ready for YOLO training.")
    print(f"       YAML: {ROOT / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
