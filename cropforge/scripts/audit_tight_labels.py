"""
audit_tight_labels.py
======================
Corrected class-count audit.

The FIRST audit double-counted raw files because on Windows,
Path.glob('*.jpg') and Path.glob('*.JPG') match the SAME case-insensitive
files.  This version deduplicates by stem before counting.

It also correctly reads the class_id from each label file's first line
rather than relying on filename prefixes (which only exist after the
collision-proof fix is applied).
"""
from pathlib import Path
from collections import defaultdict

PV_SRC  = Path(r"D:\Crop-Forge\cropforge\datasets\raw\PlantVillage")
LBL_DIR = Path(r"D:\Crop-Forge\cropforge\datasets\processed\plantvillage_tight_labels\labels")

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
CLASS_TO_ID = {n: i for i, n in enumerate(UNIFIED_CLASSES)}
ID_TO_CLASS = {i: n for i, n in enumerate(UNIFIED_CLASSES)}

# ── Step 1: unique raw counts (deduplicated by stem on Windows) ──────────────
raw_counts = {}
for d in sorted(PV_SRC.iterdir()):
    if d.is_dir():
        unique_stems = {f.stem for f in d.iterdir() if f.is_file()}
        raw_counts[d.name] = len(unique_stems)

# ── Step 2: tight label counts – always read class_id from file content ───────
tight_by_class = defaultdict(int)
read_errors = 0

for lbl in LBL_DIR.glob("*.txt"):
    try:
        with open(lbl) as f:
            first = f.readline().strip()
        cid   = int(first.split()[0])
        cname = ID_TO_CLASS.get(cid)
        if cname:
            tight_by_class[cname] += 1
    except Exception:
        read_errors += 1

if read_errors:
    print(f"[WARN] {read_errors} label files could not be read.\n")

# ── Step 3: Print report ─────────────────────────────────────────────────────
W = 54
print(f"\n{'Class':<{W}} {'Raw(unique)':>11}  {'Tight':>7}  {'Lost':>6}  {'Loss%':>6}  Status")
print("-" * (W + 48))

total_raw   = 0
total_tight = 0

for cname in sorted(raw_counts):
    n_raw   = raw_counts[cname]
    n_tight = tight_by_class.get(cname, 0)
    lost    = n_raw - n_tight
    pct     = lost / n_raw * 100 if n_raw else 0
    status  = "*** HIGH LOSS" if pct > 10 else ("OK" if pct <= 2 else "minor loss")
    print(f"{cname:<{W}} {n_raw:>11,}  {n_tight:>7,}  {lost:>6,}  {pct:>5.1f}%  {status}")
    total_raw   += n_raw
    total_tight += n_tight

print("-" * (W + 48))
total_lost = total_raw - total_tight
total_pct  = total_lost / total_raw * 100 if total_raw else 0
print(f"{'TOTAL':<{W}} {total_raw:>11,}  {total_tight:>7,}  {total_lost:>6,}  {total_pct:>5.1f}%")

print()
print("Raw(unique) : stems deduplicated per class")
print("              (fixes Windows glob double-count of .jpg / .JPG)")
print("Tight       : files actually written to the flat labels/ directory")
print("              (class_id read from first line of each .txt)")
