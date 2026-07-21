from ultralytics import YOLO

if __name__ == '__main__':
    # Load the best.pt model
    model = YOLO(r"D:\Crop-Forge\outputs\checkpoints\best.pt")

    # Dataset path
    dataset_path = r"D:\Crop-Forge\cropforge\datasets\processed\yolo_merged\dataset.yaml"

    # Fine-tune the model for 20 epochs
    results = model.train(
        data=dataset_path,
        epochs=20,
        imgsz=640,
        batch=16,
        project=r"D:\Crop-Forge\runs\detect",
        name="fine_tuned_merged"
    )

    print("[INFO] Fine-tuning complete. Model saved in D:\\Crop-Forge\\runs\\detect\\fine_tuned_merged")
