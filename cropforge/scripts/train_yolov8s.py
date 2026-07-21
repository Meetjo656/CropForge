import os
from ultralytics import YOLO

def main():
    # Path to the last checkpoint
    checkpoint_path = r"D:\Crop-Forge\runs\detect\cropforge_yolov8s_tight\weights\last.pt"
    
    # Check if we should resume
    if os.path.exists(checkpoint_path):
        print("\n" + "="*50)
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        print("="*50)
        model = YOLO(checkpoint_path)
        resume = True
    else:
        print("\n" + "="*50)
        print("Starting fresh training YOLOv8 Small on Tight Labels Dataset")
        print("="*50)
        # Initialize YOLOv8 Small model (pretrained on COCO)
        model = YOLO('yolov8s.pt') 
        resume = False

    # Run training
    model.train(
        data=r"D:\Crop-Forge\cropforge\datasets\processed\plantvillage_tight_labels\dataset.yaml",
        epochs=50,
        batch=16,           # Increased batch size for massive speedup
        workers=4,          # Increased workers for faster data loading
        imgsz=640,          # Full resolution
        name="cropforge_yolov8s_tight",
        project=r"D:\Crop-Forge\runs\detect",
        device=0,
        cache=False,        
        exist_ok=True,
        resume=resume,
        save_period=5       # Save a checkpoint every 5 epochs
    )

if __name__ == "__main__":
    main()
