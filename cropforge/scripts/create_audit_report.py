import os
from pathlib import Path
from collections import defaultdict

def main():
    print("Generating dataset report...")
    pv_path = Path(r"D:\Crop-Forge\cropforge\datasets\raw\PlantVillage")
    pd_train = Path(r"D:\Crop-Forge\cropforge\datasets\raw\PlantDoc\train")
    pd_test = Path(r"D:\Crop-Forge\cropforge\datasets\raw\PlantDoc\test")

    pv_counts = {}
    if pv_path.exists():
        for c in pv_path.iterdir():
            if c.is_dir():
                pv_counts[c.name] = len(list(c.glob("*.jpg"))) + len(list(c.glob("*.png")))

    pd_counts = {}
    for pd_p in [pd_train, pd_test]:
        if pd_p.exists():
            for c in pd_p.iterdir():
                if c.is_dir():
                    pd_counts[c.name] = pd_counts.get(c.name, 0) + len(list(c.glob("*.jpg"))) + len(list(c.glob("*.png")))

    UNIFIED_CLASSES = sorted(list(set(pv_counts.keys()).union(set(pd_counts.keys()))))
    
    with open(r"C:\Users\meetj\.gemini\antigravity-ide\brain\6d91cc86-3b70-4c91-b4d2-766985448b9d\merged_dataset_report.md", "w") as f:
        f.write("# Merged Dataset Audit Report\n\n")
        f.write("## Overview\n")
        f.write(f"- **PlantVillage Images:** {sum(pv_counts.values())}\n")
        f.write(f"- **PlantDoc Images:** {sum(pd_counts.values())}\n")
        f.write(f"- **Total Images:** {sum(pv_counts.values()) + sum(pd_counts.values())}\n\n")
        
        f.write("## Per-Class Distributions\n\n")
        f.write("| Class | PlantVillage | PlantDoc | Total |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        
        for c in UNIFIED_CLASSES:
            pv = pv_counts.get(c, 0)
            pd = pd_counts.get(c, 0)
            f.write(f"| {c} | {pv} | {pd} | {pv + pd} |\n")
            
    print("Report generated.")

if __name__ == "__main__":
    main()
