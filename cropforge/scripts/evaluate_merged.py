import os
from pathlib import Path
from ultralytics import YOLO

# Mapping from PlantDoc folders to the unified class names
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

if __name__ == "__main__":
    # Load the fine-tuned model
    model = YOLO(r"D:\Crop-Forge\runs\detect\fine_tuned_merged-3\weights\best.pt")
    plantdoc_test_dir = Path(r"D:\Crop-Forge\cropforge\datasets\raw\PlantDoc\test")

    correct = 0
    total = 0
    total_mapped = 0

    for class_dir in plantdoc_test_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        gt_class_name = class_dir.name
        mapped_gt = mapping.get(gt_class_name, None)
        
        for img_path in class_dir.glob("*.jpg"):
            total += 1
            if mapped_gt is not None:
                total_mapped += 1
            
            results = model.predict(source=str(img_path), verbose=False)
            result = results[0]
            
            if len(result.boxes) > 0:
                best_box = result.boxes[result.boxes.conf.argmax()]
                pred_class_id = int(best_box.cls.item())
                pred_class_name = model.names[pred_class_id]
                
                if mapped_gt is not None and pred_class_name == mapped_gt:
                    correct += 1

    print(f"\n--- PlantDoc Test Set Evaluation (Fine-Tuned Model) ---")
    print(f"Total Images Evaluated: {total}")
    print(f"Total Images with Mapped Classes: {total_mapped}")
    print(f"Correct Predictions (Top-1 Box): {correct}")
    if total > 0:
        print(f"Overall Accuracy: {correct / total * 100:.2f}%")
    if total_mapped > 0:
        print(f"Accuracy on Mapped Classes: {correct / total_mapped * 100:.2f}%")
