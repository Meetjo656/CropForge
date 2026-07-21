"""
visualize_detections.py
-----------------------
Runs the CropForge YOLO detector (best.pt) on 20 randomly sampled images
from the RGB folder tree and produces:
  1. Individual annotated images  -> outputs/detections/individual/
  2. A single composite grid      -> outputs/detections/detection_grid.jpg
"""

import os, sys, random, textwrap
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT        = Path(r"d:\Crop-Forge")
MODEL_PATH  = ROOT / "runs/detect/cropforge_detector_v3_a/weights/best.pt"
RGB_DIR     = ROOT / "RGB"
OUT_DIR     = ROOT / "outputs/detections"
N_IMAGES    = 20
CONF_THRESH = 0.25
SEED        = 42

PALETTE = [
    (255, 87,  34),(33, 150, 243),(76, 175, 80),(156, 39, 176),
    (255, 193,  7),(0, 188, 212),(244, 67, 54),(103, 58, 183),
    (0, 150, 136),(255, 152,  0),
]

def get_color(cls_id):
    return PALETTE[int(cls_id) % len(PALETTE)]

def draw_box(img, x1, y1, x2, y2, label, conf, color):
    thickness  = max(2, int(min(img.shape[:2]) * 0.003))
    font_scale = max(0.45, min(img.shape[:2]) * 0.001)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    text = f"{label}  {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    pad = 4
    ly1 = max(y1 - th - 2*pad, 0)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, ly1), (x1 + tw + 2*pad, y1), color, -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.putText(img, text, (x1+pad, max(y1-pad, th)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 1, cv2.LINE_AA)
    return img

def letterbox_resize(img, target=(480, 480)):
    h, w   = img.shape[:2]
    th, tw = target
    scale  = min(tw/w, th/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (nw, nh))
    canvas  = np.full((th, tw, 3), 30, dtype=np.uint8)
    dy, dx  = (th-nh)//2, (tw-nw)//2
    canvas[dy:dy+nh, dx:dx+nw] = resized
    return canvas

def add_header(img, title, species):
    h, w   = img.shape[:2]
    bar_h  = max(36, int(h * 0.07))
    bar    = np.full((bar_h, w, 3), 30, dtype=np.uint8)
    cv2.line(bar, (0, bar_h-2), (w, bar_h-2), (255, 160, 0), 2)
    fs     = max(0.4, w * 0.0013)
    cv2.putText(bar, title, (8, bar_h-10),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), 1, cv2.LINE_AA)
    sp     = textwrap.shorten(species, width=30, placeholder="...")
    (sw,_),_ = cv2.getTextSize(sp, cv2.FONT_HERSHEY_SIMPLEX, fs*0.8, 1)
    cv2.putText(bar, sp, (w-sw-8, bar_h-10),
                cv2.FONT_HERSHEY_SIMPLEX, fs*0.8, (255,160,0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])

def build_grid(images, cols=5):
    rows = []
    for i in range(0, len(images), cols):
        row_imgs = images[i:i+cols]
        while len(row_imgs) < cols:
            row_imgs.append(np.zeros_like(images[0]))
        rows.append(np.hstack(row_imgs))
    return np.vstack(rows)

def add_grid_title(grid, n_dets):
    h, w   = grid.shape[:2]
    bar_h  = 64
    bar    = np.full((bar_h, w, 3), 18, dtype=np.uint8)
    for x in range(w):
        bar[bar_h-4:, x] = (int(255*(1-x/w)), int(160*x/w), 0)
    title  = f"CropForge Detector v3a  --  20 Images  --  {n_dets} Detections Total"
    fs     = max(0.6, w * 0.0007)
    (tw,th),_ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    cv2.putText(bar, title, ((w-tw)//2, (bar_h+th)//2-4),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), 1, cv2.LINE_AA)
    return np.vstack([bar, grid])

def main():
    random.seed(SEED)
    all_images = []
    for sd in sorted(RGB_DIR.iterdir()):
        if not sd.is_dir(): continue
        for ext in ("*.JPG","*.jpg","*.png","*.PNG"):
            for p in sd.glob(ext):
                all_images.append((p, sd.name))

    print(f"[INFO] Found {len(all_images)} total images")
    selected = random.sample(all_images, min(N_IMAGES, len(all_images)))

    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    names = model.names

    ind_dir = OUT_DIR / "individual"
    ind_dir.mkdir(parents=True, exist_ok=True)

    tiles, total = [], 0
    for idx, (img_path, species) in enumerate(selected, 1):
        print(f"[{idx:02d}/{N_IMAGES}] {img_path.name}  ({species})")
        img = cv2.imread(str(img_path))
        if img is None:
            print("  WARN: could not read, skipping")
            continue
        results = model(img, conf=CONF_THRESH, verbose=False)[0]
        boxes   = results.boxes
        ann     = img.copy()
        n       = 0
        if boxes is not None and len(boxes):
            for b in boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                cid  = int(b.cls[0]); cf = float(b.conf[0])
                draw_box(ann, x1, y1, x2, y2, names.get(cid,str(cid)), cf, get_color(cid))
                n += 1
        total += n
        ann = add_header(ann, f"#{idx} | {n} det{'s' if n!=1 else ''}", species)
        cv2.imwrite(str(ind_dir / f"{idx:02d}_{img_path.stem}_det.jpg"), ann,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        tiles.append(letterbox_resize(ann))

    while len(tiles) < N_IMAGES:
        tiles.append(np.zeros_like(tiles[0]))

    grid = build_grid(tiles, cols=5)
    grid = add_grid_title(grid, total)
    gp   = OUT_DIR / "detection_grid.jpg"
    cv2.imwrite(str(gp), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\n[OK] Grid  -> {gp}")
    print(f"[OK] Tiles -> {ind_dir}")
    print(f"[OK] Total detections: {total}")

if __name__ == "__main__":
    main()
