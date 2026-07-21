import os
from pathlib import Path

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
}

plantdoc_test_dir = Path(r"D:\Crop-Forge\cropforge\datasets\raw\PlantDoc\test")

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

print(f"Total: {total}, Mapped: {total_mapped}")
