import os
from pathlib import Path

def count_images(base_path):
    counts = {}
    base = Path(base_path)
    if not base.exists(): return counts
    for class_dir in base.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
            counts[class_dir.name] = count
    return counts

pv_path = r"D:\Crop-Forge\cropforge\datasets\raw\PlantVillage"
pd_train = r"D:\Crop-Forge\cropforge\datasets\raw\PlantDoc\train"
pd_test = r"D:\Crop-Forge\cropforge\datasets\raw\PlantDoc\test"

pv_counts = count_images(pv_path)
pd_train_counts = count_images(pd_train)
pd_test_counts = count_images(pd_test)

pd_counts = {}
for k, v in pd_train_counts.items():
    pd_counts[k] = pd_counts.get(k, 0) + v
for k, v in pd_test_counts.items():
    pd_counts[k] = pd_counts.get(k, 0) + v

print("PlantVillage Total:", sum(pv_counts.values()))
print("PlantDoc Total:", sum(pd_counts.values()))
