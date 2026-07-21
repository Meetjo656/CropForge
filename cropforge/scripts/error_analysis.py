import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from ultralytics import YOLO

# Mappings from PlantDoc classes to Model classes
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
    model = YOLO(r"D:\Crop-Forge\runs\detect\fine_tuned_merged-3\weights\best.pt")
    plantdoc_test_dir = Path(r"D:\Crop-Forge\cropforge\datasets\raw\PlantDoc\test")
    
    out_dir = Path(r"D:\Crop-Forge\outputs\error_analysis")
    correct_dir = out_dir / "correct"
    incorrect_dir = out_dir / "incorrect"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    correct_dir.mkdir(parents=True, exist_ok=True)
    incorrect_dir.mkdir(parents=True, exist_ok=True)

    y_true = []
    y_pred = []

    class_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    incorrect_count = 0
    correct_count = 0
    max_images_to_save = 100

    print("[INFO] Starting error analysis on PlantDoc...")

    for class_dir in plantdoc_test_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        gt_class_name = class_dir.name
        mapped_gt = mapping.get(gt_class_name, None)
        
        if mapped_gt is None:
            continue
            
        for img_path in class_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            results = model.predict(source=img, verbose=False)
            res = results[0]
            
            pred_class_name = "Background/None"
            
            if len(res.boxes) > 0:
                best_box = res.boxes[res.boxes.conf.argmax()]
                pred_class_name = model.names[int(best_box.cls.item())]
                xyxy = best_box.xyxy[0].cpu().numpy()
            else:
                xyxy = None
                
            y_true.append(mapped_gt)
            y_pred.append(pred_class_name)
            
            is_correct = (pred_class_name == mapped_gt)
            class_stats[mapped_gt]["total"] += 1
            if is_correct:
                class_stats[mapped_gt]["correct"] += 1

            # Save Visualizations
            if is_correct and correct_count < max_images_to_save:
                if xyxy is not None:
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"Pred/GT: {pred_class_name}", (x1, max(y1-10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                save_path = correct_dir / f"{correct_count}_{img_path.name}"
                cv2.imwrite(str(save_path), img)
                correct_count += 1
                
            elif not is_correct and incorrect_count < max_images_to_save:
                if xyxy is not None:
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f"Pred: {pred_class_name}", (x1, max(y1-10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                cv2.putText(img, f"GT: {mapped_gt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                save_path = incorrect_dir / f"{incorrect_count}_{img_path.name}"
                cv2.imwrite(str(save_path), img)
                incorrect_count += 1

    print("\n" + "="*60)
    print("Class-wise Accuracy:")
    print(f"{'Class':<50} | {'Accuracy':<10} | {'Correct/Total'}")
    print("-" * 80)
    for cls_name, stats in sorted(class_stats.items()):
        total = stats["total"]
        if total > 0:
            acc = stats["correct"] / total * 100
            print(f"{cls_name:<50} | {acc:>6.2f}%    | {stats['correct']}/{total}")
    print("="*60 + "\n")

    # Generate Confusion Matrix
    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(str(out_dir / "confusion_matrix.png"), dpi=300)
    print(f"[INFO] Confusion matrix saved to {out_dir / 'confusion_matrix.png'}")

    print("\n" + "="*60)
    print("Classes with <20% accuracy - Confusion Summary:")
    print("="*60)
    
    class_predictions = defaultdict(list)
    for t, p in zip(y_true, y_pred):
        class_predictions[t].append(p)
        
    for cls_name, stats in sorted(class_stats.items()):
        total = stats["total"]
        if total > 0:
            acc = stats["correct"] / total * 100
            if acc < 20.0:
                print(f"\n{cls_name}:")
                preds = class_predictions[cls_name]
                pred_counts = {}
                for p in preds:
                    pred_counts[p] = pred_counts.get(p, 0) + 1
                
                for p, count in sorted(pred_counts.items(), key=lambda item: item[1], reverse=True):
                    print(f"  {count} -> {p}")
    
if __name__ == "__main__":
    main()
