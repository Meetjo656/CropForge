import os
from pathlib import Path
import random

def main():
    merged_dir = Path(r"D:\Crop-Forge\cropforge\datasets\processed\merged_balanced")
    train_images_dir = merged_dir / "train" / "images"
    
    pv_paths = []
    pd_paths = []
    
    for img_path in train_images_dir.glob("*.jpg"):
        if "___" in img_path.name:
            pv_paths.append(str(img_path))
        else:
            pd_paths.append(str(img_path))
            
    print(f"[INFO] Found {len(pv_paths)} PlantVillage images.")
    print(f"[INFO] Found {len(pd_paths)} PlantDoc images.")
    
    # Calculate duplication factor to balance the classes
    # We want len(pd_paths) * factor ≈ len(pv_paths)
    factor = len(pv_paths) // max(1, len(pd_paths))
    print(f"[INFO] Oversampling PlantDoc in the text list by a factor of {factor}x.")
    
    balanced_paths = pv_paths.copy()
    for _ in range(factor):
        balanced_paths.extend(pd_paths)
        
    random.shuffle(balanced_paths)
    
    txt_out = merged_dir / "train_balanced.txt"
    with open(txt_out, "w") as f:
        for p in balanced_paths:
            f.write(f"{p}\n")
            
    print(f"[INFO] Wrote {len(balanced_paths)} image paths to {txt_out}")
    print("[INFO] This achieves a 50/50 balance in each training epoch without duplicating .jpg files on disk!")

if __name__ == "__main__":
    main()
