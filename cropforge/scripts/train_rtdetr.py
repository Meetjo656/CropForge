import os
from ultralytics import RTDETR

def main():
    # Initialize RT-DETR model (falling back to Large as it's the only one with official COCO weights)
    model = RTDETR('rtdetr-l.pt')

    print("\n" + "="*50)
    print("Training RT-DETR on Tight Labels Dataset")
    print("="*50)

    # Run training
    model.train(
        data=r"D:\Crop-Forge\cropforge\datasets\processed\plantvillage_tight_labels\dataset.yaml",
        epochs=50,          # Reduced to 50 epochs
        batch=2,            # Reduced batch to 2 for 6GB VRAM
        workers=2,          # Reduced workers to limit CPU/RAM load
        imgsz=512,
        name="cropforge_rtdetr_tight",
        project=r"D:\Crop-Forge\runs\detect",
        device=0,
        cache=False,        # Disable caching to limit system RAM usage
        exist_ok=True
    )

if __name__ == "__main__":
    main()
