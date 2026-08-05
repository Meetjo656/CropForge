import argparse
import random
import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import csv

def evaluate_random(model_path, dataset_path, num_folders, num_images):
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        print(f"Dataset path {dataset_dir} does not exist.")
        return
        
    # Find all directories that contain at least one image
    all_folders = []
    for root, _, files in os.walk(dataset_dir):
        if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
            all_folders.append(Path(root))
            
    if not all_folders:
        print(f"No class folders with images found in {dataset_dir}")
        return
        
    # Pick random folders
    sampled_folders = random.sample(all_folders, min(num_folders, len(all_folders)))
    print(f"Selected {len(sampled_folders)} random folders:")
    for f in sampled_folders:
        print(f"  - {f.name}")
        
    # Collect all images from the sampled folders
    all_images = []
    for f in sampled_folders:
        for ext in ("*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg"):
            all_images.extend(list(f.glob(ext)))
            
    if not all_images:
        print("No images found in the selected folders.")
        return
        
    # Sample images
    sampled_images = random.sample(all_images, min(num_images, len(all_images)))
    print(f"\nSelected {len(sampled_images)} random images for evaluation.")
    
    # Create output directories
    out_dir = Path("d:/Crop-Forge/runs/detect/random_evaluation")
    out_img_dir = out_dir / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    
    out_csv = out_dir / "predictions.csv"
    
    csv_data = [["image_name", "true_class", "pred_class", "confidence", "bbox_xyxy"]]
    
    print("\nRunning inference...")
    
    # Run predictions
    for img_path in sampled_images:
        true_class = img_path.parent.name
        
        # Run inference and save image
        results = model.predict(source=str(img_path), save=False, verbose=False)
        
        for result in results:
            # Save visual prediction
            out_path = out_img_dir / f"{true_class}__{img_path.name}"
            result.save(filename=str(out_path))
            
            # Extract raw predictions for CSV
            boxes = result.boxes
            if len(boxes) == 0:
                 csv_data.append([img_path.name, true_class, "None", "", ""])
                 
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                pred_class_name = result.names[cls_id]
                
                csv_data.append([
                    img_path.name,
                    true_class,
                    pred_class_name,
                    f"{conf:.4f}",
                    str(xyxy)
                ])
                
    # Save CSV
    with open(out_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
        
    print(f"\nEvaluation complete!")
    print(f"Images saved to: {out_img_dir}")
    print(f"Predictions saved to: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on random images")
    parser.add_argument("--model", type=str, default=r"d:\Crop-Forge\runs\detect\cropforge_yolov8s_tight\weights\best.pt", help="Path to model weights")
    parser.add_argument("--data", type=str, default=r"d:\Crop-Forge\cropforge\datasets\raw\PlantDoc", help="Path to dataset")
    parser.add_argument("--folders", type=int, default=10, help="Number of random folders to sample")
    parser.add_argument("--images", type=int, default=50, help="Total number of random images to evaluate")
    
    args = parser.parse_args()
    evaluate_random(args.model, args.data, args.folders, args.images)
