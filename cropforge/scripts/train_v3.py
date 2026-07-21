import os
from ultralytics import YOLO

def train_model(base_model_path, save_name):
    model = YOLO(base_model_path)
    
    # Run training
    model.train(
        data=r"D:\Crop-Forge\cropforge\datasets\processed\merged_balanced\dataset.yaml",
        epochs=100,
        patience=15,
        imgsz=640,
        name=save_name,
        project=r"D:\Crop-Forge\runs\detect",
        device=0,
        workers=8, # Increased workers to speed up dataloading
        cache=True, # Cache images to RAM for ultra-fast epochs
        exist_ok=True
    )

def main():
    # V3-A: Train from scratch
    print("\n" + "="*50)
    print("Training V3-A (From Scratch: yolov8n.pt)")
    print("="*50)
    train_model("yolov8n.pt", "cropforge_detector_v3_a")
    
    # V3-B: Train from V2
    print("\n" + "="*50)
    print("Training V3-B (From V2: best.pt)")
    print("="*50)
    train_model(r"D:\Crop-Forge\runs\detect\fine_tuned_merged-3\weights\best.pt", "cropforge_detector_v3_b")

if __name__ == "__main__":
    main()
