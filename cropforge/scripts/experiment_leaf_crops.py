import os
import random
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
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

def get_best_prediction(res, yolo_model):
    if len(res.boxes) == 0:
        return "Background/None", 0.0
    best_idx = res.boxes.conf.argmax()
    best_box = res.boxes[best_idx]
    pred_class_name = yolo_model.names[int(best_box.cls.item())]
    pred_conf = float(best_box.conf.item())
    return pred_class_name, pred_conf

def main():
    print("[INFO] Loading models...")
    sam_model = SAM("sam2.1_t.pt")
    yolo_model = YOLO(r"D:\Crop-Forge\runs\detect\fine_tuned_merged-3\weights\best.pt")

    plantdoc_test_dir = Path(r"D:\Crop-Forge\cropforge\datasets\raw\PlantDoc\test")
    out_dir = Path(r"D:\Crop-Forge\outputs\leaf_crop_failures")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_images = []
    for class_dir in plantdoc_test_dir.iterdir():
        if not class_dir.is_dir():
            continue
        gt_class_name = class_dir.name
        mapped_gt = mapping.get(gt_class_name, None)
        if mapped_gt is not None:
            for img_path in class_dir.glob("*.jpg"):
                all_images.append((img_path, mapped_gt))

    # We will use all images instead of 100 so we get a rigorous metric, 
    # but limiting to 100 if we want it fast. Let's do 100 to match user's previous spec and to iterate quickly.
    random.seed(42)
    selected = random.sample(all_images, min(100, len(all_images)))

    total = len(selected)
    baseline_correct = 0
    crop_correct = 0
    failures_saved = 0

    class_stats = defaultdict(lambda: {"total": 0, "baseline_correct": 0, "crop_correct": 0})

    print(f"[INFO] Evaluating {total} images through SAM2.1 Leaf Instance Pipeline...")

    for i, (img_path, mapped_gt) in enumerate(selected):
        img_str = str(img_path)
        img = cv2.imread(img_str)
        if img is None:
            total -= 1
            continue

        class_stats[mapped_gt]["total"] += 1

        # --- A. Baseline Inference ---
        res_orig = yolo_model.predict(source=img, verbose=False)[0]
        baseline_pred, baseline_conf = get_best_prediction(res_orig, yolo_model)
        
        if baseline_pred == mapped_gt:
            baseline_correct += 1
            class_stats[mapped_gt]["baseline_correct"] += 1

        # --- B. SAM2.1 Crop Extraction & Inference ---
        res_sam = sam_model(img_str, verbose=False)[0]
        
        best_crop_pred = "Background/None"
        best_crop_conf = 0.0
        
        valid_crops_found = 0

        if res_sam.masks is not None and len(res_sam.masks.data) > 0:
            for mask_tensor in res_sam.masks.data:
                mask_data = mask_tensor.cpu().numpy().astype(np.uint8)
                mask_resized = cv2.resize(mask_data, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_bool = mask_resized > 0.5
                
                area = mask_bool.sum()
                if area < 1000: # Discard small noise masks
                    continue
                    
                valid_crops_found += 1
                
                # Get bounding box for crop
                y_indices, x_indices = np.where(mask_bool)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    y_min, y_max = y_indices.min(), y_indices.max()
                    x_min, x_max = x_indices.min(), x_indices.max()
                    
                    # Optional padding
                    pad = 10
                    y_min = max(0, y_min - pad)
                    y_max = min(img.shape[0], y_max + pad)
                    x_min = max(0, x_min - pad)
                    x_max = min(img.shape[1], x_max + pad)
                    
                    # Extract crop
                    crop = img[y_min:y_max, x_min:x_max].copy()
                    
                    # Blackout background in the crop using the mask
                    crop_mask = mask_bool[y_min:y_max, x_min:x_max]
                    crop[~crop_mask] = 0
                    
                    # Inference on crop
                    res_crop = yolo_model.predict(source=crop, verbose=False)[0]
                    crop_pred, crop_conf = get_best_prediction(res_crop, yolo_model)
                    
                    if crop_conf > best_crop_conf:
                        best_crop_conf = crop_conf
                        best_crop_pred = crop_pred

        # If SAM fails to find valid crops, fallback to baseline
        if valid_crops_found == 0:
            best_crop_pred = baseline_pred

        if best_crop_pred == mapped_gt:
            crop_correct += 1
            class_stats[mapped_gt]["crop_correct"] += 1
        else:
            # Save failure example
            if failures_saved < 100:
                failure_img = img.copy()
                cv2.putText(failure_img, f"GT: {mapped_gt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(failure_img, f"Base: {baseline_pred}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(failure_img, f"Crop: {best_crop_pred}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imwrite(str(out_dir / f"{failures_saved}_crop_fail_{img_path.name}"), failure_img)
                failures_saved += 1
                
        if (i+1) % 10 == 0:
            print(f"[INFO] Processed {i+1}/{total}...")

    print("\n" + "="*60)
    print("--- SAM2.1 Leaf Instance Pipeline Results ---")
    print(f"Total Images Evaluated: {total}")
    if total > 0:
        print(f"Baseline Accuracy: {baseline_correct / total * 100:.2f}%")
        print(f"Leaf-Crop Accuracy:   {crop_correct / total * 100:.2f}%")
    
    print("\nClass-wise Accuracy (Baseline -> Crop):")
    for cls_name, stats in sorted(class_stats.items()):
        if stats["total"] > 0:
            base_acc = stats["baseline_correct"] / stats["total"] * 100
            crop_acc = stats["crop_correct"] / stats["total"] * 100
            print(f"{cls_name:<45} | {base_acc:>6.2f}% -> {crop_acc:>6.2f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
