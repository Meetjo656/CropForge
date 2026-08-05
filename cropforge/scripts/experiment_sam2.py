import os
import random
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO, SAM

mapping = {
    "Bell_pepper leaf spot": "Pepper__bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper__bell___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Tomato leaf": "Tomato_healthy",
    "Tomato leaf mosaic virus": "Tomato__Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato leaf bacterial spot": "Tomato_Bacterial_spot",
    "Tomato Early blight leaf": "Tomato_Early_blight",
    "Tomato leaf late blight": "Tomato_Late_blight",
    "Tomato mold leaf": "Tomato_Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato_Septoria_leaf_spot",
    "Tomato two spotted spider mites leaf": "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Apple Scab Leaf": "Apple_Scab",
    "Apple leaf": "Apple_healthy",
    "Apple rust leaf": "Apple_rust",
    "Blueberry leaf": "Blueberry_healthy",
    "Cherry leaf": "Cherry_healthy",
    "Corn Gray leaf spot": "Corn_Gray_leaf_spot",
    "Corn leaf blight": "Corn_leaf_blight",
    "Corn rust leaf": "Corn_rust",
    "Peach leaf": "Peach_healthy",
    "Raspberry leaf": "Raspberry_healthy",
    "Soyabean leaf": "Soybean_healthy",
    "Squash Powdery mildew leaf": "Squash_Powdery_mildew",
    "Strawberry leaf": "Strawberry_healthy",
    "grape leaf": "Grape_healthy",
    "grape leaf black rot": "Grape_black_rot"
}

def main():
    # 1. Load models
    sam_model = SAM("sam2.1_t.pt")
    yolo_model = YOLO(r"D:\Crop-Forge\runs\detect\fine_tuned_merged-3\weights\best.pt")

    plantdoc_test_dir = Path(r"D:\Crop-Forge\cropforge\datasets\raw\PlantDoc\test")

    # 2. Collect all images that have a mapped class
    all_images = []
    for class_dir in plantdoc_test_dir.iterdir():
        if not class_dir.is_dir():
            continue
        gt_class_name = class_dir.name
        mapped_gt = mapping.get(gt_class_name, None)
        if mapped_gt is not None:
            for img_path in class_dir.glob("*.jpg"):
                all_images.append((img_path, mapped_gt))

    # 3. Randomly select 100 images
    random.seed(42)
    selected = random.sample(all_images, min(100, len(all_images)))

    correct_original = 0
    correct_masked = 0
    total = len(selected)

    print(f"[INFO] Evaluating {total} images...")

    for i, (img_path, mapped_gt) in enumerate(selected):
        img_str = str(img_path)
        img = cv2.imread(img_str)
        if img is None:
            total -= 1
            continue

        # --- A. Original Inference ---
        res_orig = yolo_model.predict(source=img, verbose=False)[0]
        orig_pred = None
        if len(res_orig.boxes) > 0:
            best_box = res_orig.boxes[res_orig.boxes.conf.argmax()]
            orig_pred = yolo_model.names[int(best_box.cls.item())]
            if orig_pred == mapped_gt:
                correct_original += 1

        # --- B. SAM2.1 Masking ---
        res_sam = sam_model(img_str, verbose=False)[0]
        
        masked_img = np.zeros_like(img)
        if res_sam.masks is not None and len(res_sam.masks.data) > 0:
            # Find largest mask by area
            areas = [mask.sum().item() for mask in res_sam.masks.data]
            largest_idx = np.argmax(areas)
            
            mask_data = res_sam.masks.data[largest_idx].cpu().numpy().astype(np.uint8)
            mask_resized = cv2.resize(mask_data, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask_resized > 0.5
            masked_img[mask_bool] = img[mask_bool]
        else:
            # Fallback if SAM fails
            masked_img = img

        # --- C. Masked Inference ---
        res_masked = yolo_model.predict(source=masked_img, verbose=False)[0]
        if len(res_masked.boxes) > 0:
            best_box = res_masked.boxes[res_masked.boxes.conf.argmax()]
            masked_pred = yolo_model.names[int(best_box.cls.item())]
            if masked_pred == mapped_gt:
                correct_masked += 1
                
        if (i+1) % 10 == 0:
            print(f"[INFO] Processed {i+1}/{total}...")

    print("\n" + "="*50)
    print("--- SAM2.1 Domain Shift Experiment Results ---")
    print(f"Total Images Evaluated: {total}")
    if total > 0:
        print(f"Original Accuracy: {correct_original / total * 100:.2f}%")
        print(f"Masked Accuracy:   {correct_masked / total * 100:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
