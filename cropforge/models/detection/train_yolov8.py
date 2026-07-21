import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import wandb

def main():
    # Base paths
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parents[2] # points to d:\Crop-Forge
    outputs_dir = base_dir / "outputs"

    # Define specific output directories
    checkpoints_dir = outputs_dir / "checkpoints"
    logs_dir = outputs_dir / "logs"
    metrics_dir = outputs_dir / "metrics"
    predictions_dir = outputs_dir / "predictions"

    # Create directories if they don't exist
    for d in [checkpoints_dir, logs_dir, metrics_dir, predictions_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Initialize weights and biases (wandb)
    # The run will automatically track train_loss, val_loss, precision, recall, and mAP50
    # as Ultralytics integrates natively with wandb.
    wandb.init(
        project="cropforge-detection",
        name="yolov8-training",
        config={
            "architecture": "YOLOv8",
            "dataset": "CropForge",
            "epochs": 50,
            "imgsz": 640
        }
    )

    # Dataset path
    data_yaml = base_dir / "cropforge" / "datasets" / "processed" / "yolo" / "dataset.yaml"

    # Initialize model
    last_ckpt = outputs_dir / "train_run" / "weights" / "last.pt"
    if last_ckpt.exists():
        print(f"Resuming training from {last_ckpt}")
        model = YOLO(str(last_ckpt))
        results = model.train(resume=True)
    else:
        model = YOLO("yolov8n.pt") # Load a pretrained YOLOv8 model

        # Train the model
        # Results are initially saved to outputs/train_run
        print(f"Starting training on dataset: {data_yaml}")
        results = model.train(
            data=str(data_yaml),
            epochs=50,
            imgsz=640,
            project=str(outputs_dir),
            name="train_run",
            exist_ok=True,
        )

    train_run_dir = outputs_dir / "train_run"

    if train_run_dir.exists():
        # 1. Store checkpoints
        weights_dir = train_run_dir / "weights"
        if weights_dir.exists():
            for pt_file in weights_dir.glob("*.pt"):
                shutil.copy2(pt_file, checkpoints_dir / pt_file.name)
            print(f"Checkpoints stored in {checkpoints_dir}")

        # 2. Store logs
        # Copy log files (.txt, .yaml, tfevents)
        for log_file in train_run_dir.glob("*"):
            if log_file.is_file() and (log_file.suffix in ['.txt', '.yaml', '.csv'] or 'events.out.tfevents' in log_file.name):
                shutil.copy2(log_file, logs_dir / log_file.name)
        print(f"Logs stored in {logs_dir}")

        # 3. Store metrics
        # Copy metrics related plots and csv
        metrics_files = [
            "results.csv", "results.png", "confusion_matrix.png", 
            "confusion_matrix_normalized.png", "F1_curve.png", 
            "PR_curve.png", "P_curve.png", "R_curve.png"
        ]
        for m_file in metrics_files:
            src_file = train_run_dir / m_file
            if src_file.exists():
                shutil.copy2(src_file, metrics_dir / src_file.name)
        print(f"Metrics stored in {metrics_dir}")

        # 4. Store predictions
        # Copy validation/prediction result images
        for img_file in train_run_dir.glob("*.jpg"):
            if "val" in img_file.name or "pred" in img_file.name:
                shutil.copy2(img_file, predictions_dir / img_file.name)
        for img_file in train_run_dir.glob("*.png"):
            if ("val" in img_file.name or "pred" in img_file.name) and img_file.name not in metrics_files:
                shutil.copy2(img_file, predictions_dir / img_file.name)
        print(f"Predictions stored in {predictions_dir}")

    print(f"Training complete. All organized outputs are stored in {outputs_dir}")
    wandb.finish()

if __name__ == "__main__":
    main()
